"""jev_client.py

SourcingX's client for TypeSafe's Jev model -- a cheap AI judgment model
Alexey is evaluating as an alternative to today's OpenAI/Anthropic screening
call (docs/GOAL-screening-model-luna-jev.md, "Build item B: Jev client").

WHERE THIS CALL SHAPE COMES FROM. There is no Jev client anywhere in
SourcingX or on agent-kalamata's checked-out master. The only real
reference implementation is on agent-kalamata's UNMERGED branch
origin/feat/jev-phase2-shadow-runner. This module was built by reading
that real code with `git show` (scripts/jev_probes.py, scripts/jev_shadow_run.py,
pipeline/jev_questions.py) -- not by guessing at Jev's API. Two things the
goal doc got slightly wrong, corrected here after reading the real branch:
the shadow-runner script is named scripts/jev_shadow_run.py, not
jev_shadow_runner.py; and the doc's `scripts/jev_compare.py` path is
correct as written.

WHAT JEV RETURNS. Jev answers TYPED questions through one SDK call --
`client.system_one(state=<candidate text>, questions=<dict>)` -- it does
not write freeform explanations. Three question shapes, verified from the
real branch code:
  - Noul(instructions=str)
        -> answer.noul: float in [0, 1] (no separate confidence field --
           kalamata derives one as abs(noul - 0.5) * 2, reused below)
  - Score(instructions=str, criteria=<ORDERED LIST, lowest level first>)
        -> answer.score: float, answer.confidence: float, answer.legend: dict
        (passing Score.criteria a dict instead of a list raises a pydantic
        ValidationError -- verified fact, not guessed)
  - Choice(instructions=str, criteria=<DICT {option: description}>)
        -> answer.choice: str, answer.confidence: float
None of these carry a reasoning/explanation text field. This module never
invents one: every `reason` string below is built ONLY from text we
ourselves wrote into a question's own instructions/criteria (the same
"Fit score: <level text>" / "Hard filter: <dealbreaker text>" pattern
kalamata's own propose_verdict uses) -- never from Jev-generated prose,
because Jev doesn't generate any.

FAIL OPEN ON SHAKY EVIDENCE. Same rule kalamata's propose_verdict follows,
copied faithfully (constants MIN_CONFIDENCE=0.60 and QUALIFY_SCORE_FLOOR=6
are the real values read from pipeline/jev_questions.py, not invented): a
missing/malformed answer on any question, or a confidence below
MIN_CONFIDENCE, never becomes a silent reject -- the verdict comes back as
screening_result=None, send_to_luna=True instead ("send_to_luna" is kept as
the field name so this dict is a drop-in match for kalamata's own shape;
SourcingX has no Luna fallback path today, so callers here should read it
as "screening_incomplete").

WHAT'S DELIBERATELY NOT PORTED. Kalamata's real module keys its question
set off a per-position registry (`_POSITION_QUESTION_SETS`, hand-calibrated
per client position) and supports position-declared "cap questions" (extra
Noul checks that cap the score). SourcingX has no position registry -- it
screens against a freeform job_description (+ optional structured
must_haves/nice_to_haves/exclusions, see screen_profile() in dashboard.py)
supplied at call time. This client builds its three questions' instructions
FROM that text at call time instead of looking a position up by id, and
does not implement cap questions -- there is nothing SourcingX-side to
calibrate them against yet.

NO PAID CALLS FROM THIS MODULE'S OWN CODE. build_client() is the only
place that can construct a real typesafe_sdk.TypeSafeClient, and nothing
here calls it automatically -- screen_with_jev() only builds one when the
caller passes client=None, and callers must only do that after Alexey has
given per-call dollar approval (CLAUDE.md / the goal doc's "no paid API
call without a direct yes, every time" rule). Tests always pass in a fake
client, so build_client() and the real `typesafe_sdk` import are never
exercised by `pytest`.

typesafe_sdk MAY NOT BE INSTALLED. SourcingX's requirements.txt does not
list it today (only agent-kalamata's own environment does). The import
below is guarded: TYPESAFE_SDK_AVAILABLE is False and build_client() /
build_questions() raise a clear JevUnavailableError instead of crashing on
import, if/when someone tries to use this module before the dependency is
added.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

try:
    from typesafe_sdk import (
        Choice,
        Noul,
        RetryPolicy,
        Score,
        TypeSafeAPIConnectionError,
        TypeSafeAPITimeoutError,
        TypeSafeClient,
        TypeSafeInternalServerError,
        TypeSafeRateLimitError,
    )
    TYPESAFE_SDK_AVAILABLE = True
except ImportError:
    Choice = Noul = Score = TypeSafeClient = RetryPolicy = None  # type: ignore[assignment]
    TypeSafeAPIConnectionError = TypeSafeAPITimeoutError = None  # type: ignore[assignment]
    TypeSafeInternalServerError = TypeSafeRateLimitError = None  # type: ignore[assignment]
    TYPESAFE_SDK_AVAILABLE = False


class JevUnavailableError(RuntimeError):
    """Raised when real Jev access is requested (build_client() or
    build_questions()) but the `typesafe_sdk` package isn't installed."""


# ============================================================================
# CONSTANTS -- MIN_CONFIDENCE and QUALIFY_SCORE_FLOOR are the REAL values
# read from agent-kalamata's pipeline/jev_questions.py (verified, not
# invented). FIT_SCORE_LEVELS/ANCHORS and the question ids are SourcingX's
# own choices -- there is no SourcingX position registry to copy those
# from, so a reasonable, documented 4-level ladder is used instead.
# ============================================================================

MIN_CONFIDENCE = 0.60
QUALIFY_SCORE_FLOOR = 6  # on the 1-10 screening_score scale below

QUESTION_ID_HARD_FILTER = "hard_filter_pass"
QUESTION_ID_FIT_SCORE = "fit_score"
QUESTION_ID_REJECT_REASON = "reject_reason"

# Ordered low -> high, per the verified Score.criteria contract (a list, not
# a dict). Interpolated onto FIT_SCORE_ANCHORS the same way kalamata's own
# _interpolate_screening_score does.
FIT_SCORE_LEVELS = ["not_qualified", "borderline", "qualified", "strong_fit"]
FIT_SCORE_ANCHORS = (1, 4, 7, 10)

UNCLEAR_OR_OTHER = "unclear_or_other"
GENERIC_REJECT_REASON_CRITERIA = {
    "does_not_meet_requirements": (
        "The candidate does not meet the job description's core requirements."
    ),
    UNCLEAR_OR_OTHER: (
        "The profile does not give enough evidence to name a specific reason."
    ),
}

JEV_MAX_ATTEMPTS = 3  # 1 try + 2 retries on a transient error, same as kalamata
JEV_RETRY_BASE_DELAY_SECONDS = 0.5


def _transient_jev_errors() -> tuple:
    """The exception classes worth retrying. Filters out anything that
    isn't actually a BaseException subclass (e.g. None, when typesafe_sdk
    isn't installed and TYPESAFE_SDK_AVAILABLE is False -- or a test that
    flips TYPESAFE_SDK_AVAILABLE to True without also patching every error
    class) so `except transient_errors` never itself raises a TypeError."""
    candidates = (
        TypeSafeAPIConnectionError,
        TypeSafeAPITimeoutError,
        TypeSafeInternalServerError,
        TypeSafeRateLimitError,
    )
    return tuple(c for c in candidates if isinstance(c, type) and issubclass(c, BaseException))


# ============================================================================
# Candidate text -- built from the same profile shape screen_profile() in
# dashboard.py already accepts (raw_crustdata / raw_data / flat fields), not
# by importing dashboard.py itself (that module has Streamlit UI code at
# import time and isn't meant to be imported headless). A light structural
# PII strip is applied first -- kalamata's own module strips name/email/
# phone/URL/photo fields before anything reaches TypeSafe; this is a
# smaller version of that same idea (structural field removal only, not
# kalamata's full text-level regex scrub, which is out of scope here).
# ============================================================================

_PERSONAL_TOP_LEVEL_KEYS = (
    "name", "first_name", "last_name",
    "email", "email_address", "personal_email", "work_email", "business_email",
    "phone", "phone_number", "personal_phone", "work_phone", "mobile_phone",
    "linkedin_url", "linkedin_profile_url", "flagship_profile_url",
    "linkedin_flagship_url", "profile_pic_url", "profile_picture_url",
    "photo_url",
)


def strip_personal_fields(raw: dict) -> dict:
    """Shallow copy of `raw` with the common personal-identity fields
    removed. Never mutates the caller's dict. Non-dict input returns {}."""
    if not isinstance(raw, dict):
        return {}
    stripped = dict(raw)
    for key in _PERSONAL_TOP_LEVEL_KEYS:
        stripped.pop(key, None)
    return stripped


def _get_raw_profile(profile: dict) -> dict:
    """Same extraction screen_profile() uses: prefer raw_crustdata, then
    raw_data, then the profile dict itself."""
    if not isinstance(profile, dict):
        return {}
    raw = profile.get("raw_crustdata") or profile.get("raw_data") or profile
    if isinstance(raw, str):
        import json
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            raw = profile
    return raw if isinstance(raw, dict) else {}


def _as_list(value: Any) -> list:
    """Codex review on PR #133: the flattened profile shape can carry
    `skills`/`all_schools` as a single comma-free string instead of a list
    (e.g. "Python") -- joining a bare string with `", ".join(...)` iterates
    it character by character ("P, y, t, h, o, n"), corrupting the evidence
    Jev sees. Coerce a scalar string into a one-item list; leave a real list
    (or anything else iterable-and-intentional) alone."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return value
    return list(value) if value else []


def _format_employer(entry: dict) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    title = entry.get("employee_title") or entry.get("title")
    company = entry.get("employer_name") or entry.get("company_name") or entry.get("company")
    start = entry.get("start_date") or ""
    end = entry.get("end_date") or "present"
    if not title and not company:
        return None
    line = f"{title or 'Unknown title'} at {company or 'Unknown company'} ({start} - {end})"
    description = entry.get("employee_description") or entry.get("description")
    if description:
        line += f"\n    {description}"
    return line


def build_candidate_text(profile: dict) -> str:
    """Plain-text candidate summary for Jev's `state=` argument. Built from
    the same fields screen_profile() reads off a profile dict, with
    personal-identity fields stripped first (see strip_personal_fields).
    Career text -- titles, companies, dates, skills, summary -- stays, same
    as kalamata's own "career text ... stay" rule.
    """
    raw = strip_personal_fields(_get_raw_profile(profile))

    # Codex review on PR #133: a nonempty-but-thin raw_data/raw_crustdata
    # object (e.g. only a headline, no employers/skills) wins over `profile`
    # in _get_raw_profile() even when `profile` itself carries usable flat
    # current_title/current_company/skills fields alongside it -- exactly
    # the condition screen_profile() in dashboard.py already detects
    # (its own `if not raw.get('current_employers') and not
    # raw.get('past_employers') and not raw.get('skills')` check, ~line
    # 4772) and falls back on. Match that here so the same thin-profile
    # shape doesn't silently lose evidence just for Jev.
    if (
        isinstance(profile, dict)
        and not raw.get("current_employers")
        and not raw.get("past_employers")
        and not raw.get("skills")
    ):
        if profile.get("current_title") and "current_title" not in raw:
            raw = {**raw, "current_title": profile["current_title"]}
        if profile.get("current_company") and "current_company" not in raw:
            raw = {**raw, "current_company": profile["current_company"]}
        if profile.get("skills") and not raw.get("skills"):
            raw = {**raw, "skills": profile["skills"]}

    lines = []
    headline = raw.get("headline") or raw.get("current_title")
    if headline:
        lines.append(f"Headline: {headline}")

    summary = raw.get("summary")
    if summary:
        lines.append(f"Summary: {summary}")

    current = raw.get("current_employers") or []
    if isinstance(current, list) and current:
        lines.append("Current role(s):")
        for entry in current:
            formatted = _format_employer(entry)
            if formatted:
                lines.append(f"  - {formatted}")
    elif raw.get("current_title") or raw.get("current_company"):
        lines.append(
            f"Current role: {raw.get('current_title') or 'Unknown title'} at "
            f"{raw.get('current_company') or 'Unknown company'}"
        )

    past = raw.get("past_employers") or []
    if isinstance(past, list) and past:
        lines.append("Past role(s):")
        for entry in past:
            formatted = _format_employer(entry)
            if formatted:
                lines.append(f"  - {formatted}")

    skills = raw.get("skills") or []
    if skills:
        lines.append("Skills: " + ", ".join(str(s) for s in _as_list(skills)))

    schools = raw.get("all_schools") or raw.get("schools") or []
    if schools:
        lines.append("Education: " + ", ".join(str(s) for s in _as_list(schools)))

    location = raw.get("location") or raw.get("region")
    if location:
        lines.append(f"Location: {location}")

    return "\n".join(lines) if lines else "(no profile data available)"


# ============================================================================
# Building the three questions Jev sees in one call
# ============================================================================

def build_reject_reason_criteria(screening_brief: Optional[dict] = None) -> dict:
    """{option: description} for the reject_reason Choice question, always
    ending with the mandatory UNCLEAR_OR_OTHER option (per the verified
    fail-open rule: a Choice answer of 'unclear_or_other' means Jev didn't
    have enough evidence, and must defer, not reject). Built from the
    caller's own must_haves/exclusions when a screening_brief is supplied
    (matching screen_profile()'s structured path in dashboard.py);
    otherwise falls back to one generic reason."""
    if not screening_brief:
        return dict(GENERIC_REJECT_REASON_CRITERIA)

    criteria: dict[str, str] = {}
    for i, must_have in enumerate(screening_brief.get("must_haves") or []):
        text = str(must_have).strip()
        if text:
            criteria[f"must_have_{i}_not_met"] = f"Does not meet this requirement: {text}"
    for i, exclusion in enumerate(screening_brief.get("exclusions") or []):
        text = str(exclusion).strip()
        if text:
            criteria[f"exclusion_{i}_matched"] = f"Matches this exclusion: {text}"

    if not criteria:
        criteria = dict(GENERIC_REJECT_REASON_CRITERIA)
        return criteria

    criteria[UNCLEAR_OR_OTHER] = (
        "The profile does not give enough evidence to name a specific reason."
    )
    return criteria


def build_questions(job_description: str, screening_brief: Optional[dict] = None) -> dict:
    """{question_id: Noul|Score|Choice} for one client.system_one() call.
    Raises JevUnavailableError if typesafe_sdk isn't installed."""
    if not TYPESAFE_SDK_AVAILABLE:
        raise JevUnavailableError(
            "typesafe_sdk is not installed -- cannot build real Jev question "
            "objects. Add it to requirements.txt and install it first."
        )

    role_context = (screening_brief or {}).get("role_context") or ""
    must_haves = (screening_brief or {}).get("must_haves") or []
    exclusions = (screening_brief or {}).get("exclusions") or []
    jd_block = job_description or role_context or ""

    hard_filter_instructions = (
        "Job description / requirements:\n"
        f"{jd_block}\n\n"
        + (f"Must-have requirements:\n" + "\n".join(f"- {m}" for m in must_haves) + "\n\n"
           if must_haves else "")
        + (f"Exclusions (candidate FAILS if any of these apply):\n"
           + "\n".join(f"- {e}" for e in exclusions) + "\n\n"
           if exclusions else "")
        + "Based only on the candidate profile text above, does this candidate "
        "clearly pass the hard, must-have requirements of this job description "
        "AND clearly avoid every exclusion listed above? Answer as a probability "
        "that they pass (near 1.0 = clearly passes and matches no exclusion, "
        "near 0.0 = clearly fails a must-have or matches an exclusion, "
        "near 0.5 = genuinely unclear from the text)."
    )

    fit_score_instructions = (
        "Job description / requirements:\n"
        f"{jd_block}\n\n"
        "Based only on the candidate profile text above, and assuming this "
        "candidate clears the job's hard requirements, how strong a fit is "
        "this candidate for the role overall?"
    )

    reject_reason_criteria = build_reject_reason_criteria(screening_brief)
    reject_reason_instructions = (
        "Job description / requirements:\n"
        f"{jd_block}\n\n"
        "This candidate did not clearly pass the job's hard requirements. "
        "Based only on the candidate profile text above, which of the "
        "following best explains why? Choose 'unclear_or_other' if the "
        "profile does not give enough evidence to name a specific reason."
    )

    return {
        QUESTION_ID_HARD_FILTER: Noul(instructions=hard_filter_instructions),
        QUESTION_ID_FIT_SCORE: Score(instructions=fit_score_instructions, criteria=FIT_SCORE_LEVELS),
        QUESTION_ID_REJECT_REASON: Choice(
            instructions=reject_reason_instructions, criteria=reject_reason_criteria
        ),
    }


# ============================================================================
# Parsing Jev's response -- duck-typed field reads (works on a real
# typesafe_sdk *Answer pydantic model OR a plain dict/fake test object),
# same _get_field pattern kalamata's own module uses, so these functions
# never need typesafe_sdk installed to be exercised by a test.
# ============================================================================

def _get_field(answer: Any, name: str) -> Any:
    if answer is None:
        return None
    try:
        if isinstance(answer, dict):
            return answer.get(name)
        return getattr(answer, name, None)
    except Exception:
        return None


def _extract_noul(answer: Any) -> Optional[float]:
    value = _get_field(answer, "noul")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    value = float(value)
    if not (0.0 <= value <= 1.0):
        return None
    return value


def _extract_score(answer: Any) -> Optional[tuple]:
    score = _get_field(answer, "score")
    confidence = _get_field(answer, "confidence")
    legend = _get_field(answer, "legend")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return None
    if not (0.0 <= float(confidence) <= 1.0):
        return None
    if legend is None or not hasattr(legend, "get"):
        return None
    return float(score), float(confidence), legend


def _extract_choice(answer: Any) -> Optional[tuple]:
    choice = _get_field(answer, "choice")
    confidence = _get_field(answer, "confidence")
    if not isinstance(choice, str) or not choice:
        return None
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return None
    if not (0.0 <= float(confidence) <= 1.0):
        return None
    return choice, float(confidence)


def _interpolate_screening_score(level_score: float, anchors: tuple = FIT_SCORE_ANCHORS) -> int:
    """Linearly interpolate a fractional Score.score into a 1-10
    screening_score, using anchors[i] as the score for ladder level i.
    Clamped to the first/last anchor rather than extrapolating. Identical
    math to kalamata's own _interpolate_screening_score, verified from the
    real branch."""
    n = len(anchors) - 1
    if level_score <= 0:
        return anchors[0]
    if level_score >= n:
        return anchors[n]
    lo_idx = int(level_score)
    frac = level_score - lo_idx
    lo, hi = anchors[lo_idx], anchors[lo_idx + 1]
    value = lo + (hi - lo) * frac
    return max(1, min(10, round(value)))


def _defer(reason: str) -> dict:
    return {
        "screening_result": None,
        "screening_score": None,
        "reason": reason,
        "send_to_luna": True,
        "confidence": None,
    }


def propose_verdict(answers: dict, reject_reason_criteria: Optional[dict] = None) -> dict:
    """Turn `response.answers` from one client.system_one() call into a
    verdict dict:
      - screening_result: 'qualified' | 'not_qualified' | None
      - screening_score: int (1-10) or None
      - reason: str, always present -- built only from our own question
        text (the fit ladder's level text, or the chosen reject option's
        description), never invented free-text reasoning
      - send_to_luna: bool -- True exactly when screening_result is None
        (SourcingX has no Luna fallback; read this as "screening_incomplete")
      - confidence: float or None -- the lowest confidence figure the
        decision actually relied on

    FAIL OPEN -- defers (screening_result=None) whenever any answer is
    missing/malformed, any relevant confidence is below MIN_CONFIDENCE, or
    (on a failed hard filter) the reject-reason choice isn't a recognized
    option or is 'unclear_or_other'. This mirrors agent-kalamata's real
    propose_verdict() (pipeline/jev_questions.py on the unmerged
    feat/jev-phase2-shadow-runner branch), minus its position-specific cap
    questions (SourcingX has none to apply).
    """
    reject_reason_criteria = reject_reason_criteria or GENERIC_REJECT_REASON_CRITERIA

    if not isinstance(answers, dict):
        return _defer("answers was not a dict of question_id -> answer -- failing open")

    noul = _extract_noul(answers.get(QUESTION_ID_HARD_FILTER))
    if noul is None:
        return _defer(f"{QUESTION_ID_HARD_FILTER} answer missing or malformed -- failing open")

    fit = _extract_score(answers.get(QUESTION_ID_FIT_SCORE))
    if fit is None:
        return _defer(f"{QUESTION_ID_FIT_SCORE} answer missing or malformed -- failing open")
    fit_score_value, fit_confidence, fit_legend = fit

    if not (0 <= fit_score_value <= len(FIT_SCORE_LEVELS) - 1):
        return _defer(
            f"{QUESTION_ID_FIT_SCORE} score {fit_score_value!r} is outside the valid "
            f"level range (0..{len(FIT_SCORE_LEVELS) - 1}) -- failing open"
        )

    reject = _extract_choice(answers.get(QUESTION_ID_REJECT_REASON))
    if reject is None:
        return _defer(f"{QUESTION_ID_REJECT_REASON} answer missing or malformed -- failing open")
    reject_choice, reject_confidence = reject

    hf_confidence = abs(noul - 0.5) * 2
    if hf_confidence < MIN_CONFIDENCE:
        return _defer(f"hard filter answer too uncertain (noul={noul:.2f}) -- failing open")

    if fit_confidence < MIN_CONFIDENCE:
        return _defer(f"fit score confidence too low ({fit_confidence:.2f}) -- failing open")

    if noul < 0.5:
        if reject_confidence < MIN_CONFIDENCE:
            return _defer(f"reject reason confidence too low ({reject_confidence:.2f}) -- failing open")
        if reject_choice not in reject_reason_criteria:
            return _defer(
                f"{QUESTION_ID_REJECT_REASON} answer {reject_choice!r} is not a recognized "
                "option -- failing open"
            )
        if reject_choice == UNCLEAR_OR_OTHER:
            return _defer(
                "hard filter failed but reject_reason was 'unclear_or_other' -- not enough "
                "evidence to name a specific dealbreaker, failing open"
            )
        reason_text = reject_reason_criteria[reject_choice]
        return {
            "screening_result": "not_qualified",
            "screening_score": FIT_SCORE_ANCHORS[0],
            "reason": f"Hard filter: {reason_text}",
            "send_to_luna": False,
            "confidence": round(min(hf_confidence, reject_confidence), 4),
        }

    screening_score = _interpolate_screening_score(fit_score_value)
    level_idx = max(0, min(len(FIT_SCORE_LEVELS) - 1, round(fit_score_value)))
    level_text = fit_legend.get(level_idx)
    if level_text is None:
        level_text = fit_legend.get(str(level_idx), FIT_SCORE_LEVELS[level_idx])
    result = "qualified" if screening_score >= QUALIFY_SCORE_FLOOR else "not_qualified"
    return {
        "screening_result": result,
        "screening_score": screening_score,
        "reason": f"Fit score: {level_text}",
        "send_to_luna": False,
        "confidence": round(fit_confidence, 4),
    }


# ============================================================================
# The real call, with retries -- same pattern as kalamata's own call_jev()
# ============================================================================

def call_jev(client, text: str, questions: dict) -> tuple:
    """One client.system_one(state=text, questions=questions) call, retried
    up to JEV_MAX_ATTEMPTS-1 times on a transient SDK error, short backoff.
    Never raises: returns (response, None, latency_ms) on success, or
    (None, error_str, latency_ms) on failure, so a bad candidate can never
    crash a caller's batch loop."""
    transient_errors = _transient_jev_errors()
    last_err = None
    latency_ms = 0.0
    for attempt in range(1, JEV_MAX_ATTEMPTS + 1):
        t0 = time.perf_counter()
        try:
            response = client.system_one(state=text, questions=questions)
            latency_ms = (time.perf_counter() - t0) * 1000
            return response, None, latency_ms
        except transient_errors as e:  # noqa: BLE001 -- deliberately broad within the tuple
            latency_ms = (time.perf_counter() - t0) * 1000
            last_err = f"{type(e).__name__}: {e}"
            if attempt < JEV_MAX_ATTEMPTS:
                time.sleep(JEV_RETRY_BASE_DELAY_SECONDS * attempt)
                continue
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:  # noqa: BLE001 -- a failure here is DATA, never a crash
            latency_ms = (time.perf_counter() - t0) * 1000
            return None, f"{type(e).__name__}: {e}", latency_ms
    return None, last_err, latency_ms


def build_client() -> "TypeSafeClient":
    """The ONE place this module may construct a real typesafe_sdk client.
    Reads TYPESAFE_API_KEY from the environment (the real SDK's own
    convention, verified from kalamata's build_real_client()) -- never read
    or printed here. If TYPESAFE_API_KEY isn't set but a `typesafe_api_key`
    value exists in SourcingX's config.json, that value is exported into
    the environment first (matching this repo's own config.json pattern --
    see config.example.json). Raises JevUnavailableError if typesafe_sdk
    isn't installed. This function makes NO network call itself; the first
    real spend happens on the next system_one() call a caller makes with
    the client it returns.
    """
    if not TYPESAFE_SDK_AVAILABLE:
        raise JevUnavailableError(
            "typesafe_sdk is not installed -- cannot build a real Jev client. "
            "Add it to requirements.txt and install it first."
        )
    if not os.environ.get("TYPESAFE_API_KEY"):
        try:
            import json as _json
            with open("config.json", "r", encoding="utf-8") as f:
                config = _json.load(f)
            key = config.get("typesafe_api_key")
            if key:
                os.environ["TYPESAFE_API_KEY"] = key
        except (OSError, ValueError):
            pass  # no config.json / unreadable -- TypeSafeClient() will raise its own error
    # Codex review on PR #133: the SDK's own default RetryPolicy(max_retries=2)
    # means every system_one() call it makes can already take up to 3 tries on
    # its own -- stacked with call_jev()'s own JEV_MAX_ATTEMPTS=3 outer loop,
    # a single screening call could balloon to 9 HTTP requests instead of the
    # documented 3. Disable the SDK's internal retries so call_jev()'s loop is
    # the one and only retry layer.
    return TypeSafeClient(retry=RetryPolicy(max_retries=0))


# ============================================================================
# Top-level entrypoint
# ============================================================================

def screen_with_jev(profile: dict, job_description: str, client=None,
                     screening_brief: Optional[dict] = None) -> dict:
    """Screen one candidate profile against a job description via Jev.

    Args:
        profile: Profile dict, same shape screen_profile() in dashboard.py
            accepts (raw_crustdata / raw_data / flat current_title+
            current_company fields).
        job_description: Freeform job requirements text.
        client: A typesafe_sdk.TypeSafeClient-like object exposing
            .system_one(state=, questions=). Tests MUST pass a fake client
            here. If None, build_client() is called -- this is the one path
            that can spend real money, so callers must only pass None after
            Alexey has given per-call dollar approval for that specific
            call (see CLAUDE.md / the goal doc's spend rule). This function
            never decides on its own to build a real client "just in case".
        screening_brief: Optional structured brief dict with keys
            role_context/must_haves/nice_to_haves/exclusions, same shape
            screen_profile() accepts. When supplied, the reject_reason
            options are built from must_haves/exclusions; otherwise a
            generic reason set is used.

    Returns a verdict dict (see propose_verdict()'s docstring for the
    shape), plus jev_model / input_tokens / output_tokens / latency_ms.
    On a Jev call failure (including exhausted retries), returns the same
    fail-open deferral shape propose_verdict() uses, with the error message
    in `reason`.
    """
    if client is None:
        client = build_client()

    text = build_candidate_text(profile)
    questions = build_questions(job_description, screening_brief)
    reject_reason_criteria = build_reject_reason_criteria(screening_brief)

    response, error, latency_ms = call_jev(client, text, questions)
    if error is not None:
        verdict = _defer(f"Jev call failed: {error}")
        verdict.update(jev_model=None, input_tokens=None, output_tokens=None,
                        latency_ms=round(latency_ms, 1))
        return verdict

    verdict = propose_verdict(response.answers, reject_reason_criteria)
    usage = _get_field(response, "usage")
    verdict.update(
        jev_model=_get_field(response, "model"),
        input_tokens=_get_field(usage, "input_tokens"),
        output_tokens=_get_field(usage, "output_tokens"),
        latency_ms=round(latency_ms, 1),
    )
    return verdict
