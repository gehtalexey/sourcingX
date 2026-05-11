"""
Deterministic post-screening validator for tenure hard constraints.

Issue 7 (Chen): the AI screener returned profiles with <1 year at current
company as "relevant" even when the recruiter wrote
"no one under 1 year at current company" in the screening criteria.

Strengthening the prompt is not enough — the model can still ignore or
trade off the rule. This module enforces tenure constraints AFTER the
model returns, deterministically.

Flow:
  1. parse_tenure_constraint_months(text) — pulls the strictest minimum
     tenure threshold (in months) from the recruiter's request / must-haves.
  2. current_company_tenure_months(raw) — computes the candidate's tenure
     at the current company in months from raw Crustdata data
     (current_employers[0].start_date → today).
  3. enforce_tenure_constraint(result, jd_text, raw, ...) — if the JD
     contains a tenure threshold AND the candidate has fewer months than
     the threshold, override the result to "Not a Fit" (score capped low).

The validator is intentionally CONSERVATIVE:
  - if no tenure phrase parses → no-op (pass through).
  - if no current-employer start_date is available → no-op (we can't
    prove a violation, so we don't override).

Patterns are English + Hebrew. All phrasings Chen and the recruiters
have used in practice are listed in the regexes below.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Parse the threshold from text
# ---------------------------------------------------------------------------

# English patterns. Order does NOT matter — we run them all and take the max.
# Each pattern captures (number, unit). Units: year(s)/yr(s) or month(s)/mo(s).
_EN_PATTERNS = [
    # "minimum 1 year at current company"
    # "minimum 6 months at current employer"
    re.compile(
        r"minimum\s+(\d+(?:\.\d+)?)\s*"
        r"(years?|yrs?|months?|mos?)\s+"
        r"at\s+(?:the\s+)?(?:current\s+)?(?:employer|company)",
        re.IGNORECASE,
    ),
    # "no one under 1 year at current company"
    # "nobody under 6 months at company"
    # "no under 12 months at company"
    re.compile(
        r"no(?:body|\s+one)?\s+under\s+(\d+(?:\.\d+)?)\s*"
        r"(years?|yrs?|months?)\s+"
        r"at\s+(?:the\s+)?(?:current\s+)?(?:employer|company)",
        re.IGNORECASE,
    ),
    # "must have been at current employer for 1 year"
    # "must have been at company for 18 months"
    re.compile(
        r"must\s+have\s+been\s+at\s+(?:the\s+)?(?:current\s+)?(?:employer|company)"
        r"\s+for\s+(\d+(?:\.\d+)?)\s*"
        r"(years?|months?)",
        re.IGNORECASE,
    ),
    # "at least 1 year at current company"
    # "at least 2 years at company"
    # "at least 6 months tenure"
    re.compile(
        r"at\s+least\s+(\d+(?:\.\d+)?)\s*"
        r"(years?|months?)\s+"
        r"(?:at\s+(?:the\s+)?(?:current\s+)?(?:employer|company)|tenure)",
        re.IGNORECASE,
    ),
]

# Hebrew patterns.
# "לפחות שנה אחת בחברה" / "לפחות 12 חודשים בחברה"
# Note: Hebrew "שנה" = "year", "שנים" = "years", "חודש" = "month", "חודשים" = "months".
# "בחברה" = "at the company", "בחברה הנוכחית" = "at the current company".
_HE_PATTERNS = [
    re.compile(
        r"לפחות\s+(\d+(?:\.\d+)?)\s*"
        r"(שנה|שנים|חודש|חודשים)\s+"
        r"בחברה(?:\s+הנוכחית)?",
    ),
    # "מינימום שנה בחברה"
    re.compile(
        r"מינימום\s+(\d+(?:\.\d+)?)\s*"
        r"(שנה|שנים|חודש|חודשים)\s+"
        r"בחברה(?:\s+הנוכחית)?",
    ),
]

_ENGLISH_YEAR_UNITS = {"year", "years", "yr", "yrs"}
_ENGLISH_MONTH_UNITS = {"month", "months", "mo", "mos"}
_HEBREW_YEAR_UNITS = {"שנה", "שנים"}
_HEBREW_MONTH_UNITS = {"חודש", "חודשים"}


# English number words → digits. Codex flagged on PR #16 that Chen's actual
# phrasing was natural language ("no one under one year at company"), not
# numeric. We pre-normalize these words to digits before running the regexes
# so every existing pattern picks them up unchanged. Range covers what a
# recruiter would plausibly type in a tenure rule.
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12",
}

_NUMBER_WORD_RE = re.compile(
    r"\b(" + "|".join(_NUMBER_WORDS) + r")\b",
    re.IGNORECASE,
)


# "no one" / "no-one" is an idiomatic phrase ("nobody"), NOT the number one.
# We protect it before number-word normalization so "no one under one year"
# doesn't collapse to "no 1 under 1 year" and break the "no one under" regex.
_NO_ONE_RE = re.compile(r"\bno[-\s]+one\b", re.IGNORECASE)
_NO_ONE_PLACEHOLDER = "\x00NOONE\x00"


def _normalize_number_words(text: str) -> str:
    """Replace English number words (one..twelve) with digits.

    Case-insensitive, word-boundary based — won't touch "onerous", "tens",
    etc. The idiomatic phrase "no one" (= nobody) is preserved verbatim so
    the "no one under N" regex still matches. Hebrew and digits pass
    through unchanged.
    """
    if not text:
        return text
    # Stash "no one" -> placeholder -> normalize -> restore.
    text = _NO_ONE_RE.sub(_NO_ONE_PLACEHOLDER, text)
    text = _NUMBER_WORD_RE.sub(lambda m: _NUMBER_WORDS[m.group(1).lower()], text)
    text = text.replace(_NO_ONE_PLACEHOLDER, "no one")
    return text


def _unit_to_months(qty: float, unit: str) -> int:
    """Convert (qty, unit) into a whole number of months.

    Year → 12 months. Month → 1 month. Fractional input rounded up so that
    "1.5 years" → 18 months (no ambiguity to the candidate's detriment).
    """
    unit_l = unit.lower()
    if unit_l in _ENGLISH_YEAR_UNITS or unit_l in _HEBREW_YEAR_UNITS:
        return int(round(qty * 12))
    if unit_l in _ENGLISH_MONTH_UNITS or unit_l in _HEBREW_MONTH_UNITS:
        return int(round(qty))
    # Unknown unit — be safe, no constraint.
    return 0


def parse_tenure_constraint_months(text: Optional[str]) -> Optional[int]:
    """Return the strictest minimum-tenure threshold (in months) found
    anywhere in ``text``, or None if no tenure constraint is present.

    "Strictest" = largest threshold. If a recruiter says both
    "at least 6 months at company" AND "minimum 1 year at current company",
    the 12-month rule wins (most restrictive).
    """
    if not text:
        return None

    # Normalize English number words ("one", "two", ...) to digits BEFORE
    # running the regexes — Chen's actual phrasing was natural language.
    text = _normalize_number_words(text)

    found_months = []
    for pattern in _EN_PATTERNS + _HE_PATTERNS:
        for match in pattern.finditer(text):
            try:
                qty = float(match.group(1))
            except (TypeError, ValueError):
                continue
            unit = match.group(2)
            months = _unit_to_months(qty, unit)
            if months > 0:
                found_months.append(months)

    if not found_months:
        return None
    return max(found_months)


# ---------------------------------------------------------------------------
# 2. Compute candidate's current-company tenure
# ---------------------------------------------------------------------------

def _parse_iso_date(value):
    """Best-effort ISO date parse. Returns aware datetime or None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _months_between(start: datetime, end: datetime) -> int:
    """Whole months between two aware datetimes (end - start)."""
    if start is None or end is None:
        return 0
    months = (end.year - start.year) * 12 + (end.month - start.month)
    return max(0, months)


def current_company_tenure_months(raw: Optional[dict]) -> Optional[int]:
    """Months at the current company, computed at the COMPANY level.

    Looks at raw['current_employers']. If multiple current positions are at
    the SAME company (internal promotion), uses the earliest start_date —
    consistent with the rest of the codebase ("stability is company-level,
    not position-level", see commit 1ee490a).

    Returns None when no current employer with a parseable start_date is
    present. (Returning None means "can't validate" — caller must NOT
    override the model in that case.)
    """
    if not isinstance(raw, dict):
        return None
    current_employers = raw.get("current_employers") or []
    if not isinstance(current_employers, list) or not current_employers:
        return None

    # Find earliest start_date among current employers that share the
    # same employer_name as the first/primary entry. Internal promotions
    # at the same company should not reset tenure.
    primary = current_employers[0] if isinstance(current_employers[0], dict) else {}
    primary_company = (primary.get("employer_name") or "").strip().lower()

    earliest = None
    for emp in current_employers:
        if not isinstance(emp, dict):
            continue
        emp_company = (emp.get("employer_name") or "").strip().lower()
        # If we have a primary company name, only count entries at that company.
        # If we don't, accept any current employer.
        if primary_company and emp_company and emp_company != primary_company:
            continue
        start = _parse_iso_date(emp.get("start_date"))
        if start is None:
            continue
        if earliest is None or start < earliest:
            earliest = start

    if earliest is None:
        return None

    today = datetime.now(timezone.utc)
    return _months_between(earliest, today)


# ---------------------------------------------------------------------------
# 3. Enforce — override the model result if it violates the constraint
# ---------------------------------------------------------------------------

# Score we cap to when overriding. The unified policy reserves 1-2 for
# "hard filter triggered" — we use 2 so the override is distinguishable from
# a truly empty / insufficient-data result (score 1).
TENURE_OVERRIDE_SCORE = 2


def enforce_tenure_constraint(
    result: dict,
    jd_text: Optional[str],
    raw: Optional[dict],
    *,
    threshold_months: Optional[int] = None,
    actual_months: Optional[int] = None,
) -> dict:
    """Deterministic post-screening override.

    If ``jd_text`` contains a tenure threshold AND ``raw`` provides a
    current-company tenure shorter than that threshold, mutate ``result``
    in place to "Not a Fit"/"NO GO" with score capped at TENURE_OVERRIDE_SCORE.
    Otherwise leave ``result`` untouched.

    ``threshold_months`` and ``actual_months`` can be passed pre-computed
    to avoid re-parsing / re-computing in tight loops; otherwise this
    function derives them itself.

    Result schema handled:
      - {"score", "fit", "summary"}           — legacy / dashboard.py
      - {"score", "fit", "summary",
         "decision", "reasoning"}             — unified-policy path
      - {"score", "category", "reasoning"}    — structured_prompt_builder
    """
    if not isinstance(result, dict):
        return result

    if threshold_months is None:
        threshold_months = parse_tenure_constraint_months(jd_text)
    if threshold_months is None:
        # No tenure constraint in the request → pass-through.
        return result

    if actual_months is None:
        actual_months = current_company_tenure_months(raw)
    if actual_months is None:
        # No current-company start_date → can't prove a violation. Don't
        # override (avoid false negatives when data is missing).
        return result

    if actual_months >= threshold_months:
        # Candidate satisfies the constraint. Leave the result alone.
        return result

    # --- Override ---
    override_msg = (
        f"Hard constraint violated: recruiter requested minimum "
        f"{threshold_months} months at current company; candidate has "
        f"{actual_months} months."
    )

    original_reasoning = (
        result.get("reasoning")
        or result.get("summary")
        or ""
    )
    new_reasoning = override_msg
    if original_reasoning:
        new_reasoning = f"{override_msg} (Original model reasoning: {original_reasoning})"

    # Cap the score (don't raise it if model already gave 1).
    try:
        existing_score = int(result.get("score") or 0)
    except (TypeError, ValueError):
        existing_score = 0
    new_score = min(existing_score, TENURE_OVERRIDE_SCORE) if existing_score > 0 else TENURE_OVERRIDE_SCORE

    result["score"] = new_score

    # Set fit/category to the project's "rejected" label, whichever the
    # caller is using.
    if "fit" in result:
        result["fit"] = "Not a Fit"
    if "category" in result:
        result["category"] = "No Fit"

    # Unified-policy path uses decision=NO GO.
    if "decision" in result:
        result["decision"] = "NO GO"

    # Update both reasoning fields (different paths read different keys).
    if "reasoning" in result:
        result["reasoning"] = new_reasoning
    if "summary" in result:
        result["summary"] = new_reasoning

    # Tag for downstream inspection / debugging.
    result["tenure_override"] = {
        "threshold_months": threshold_months,
        "actual_months": actual_months,
    }

    return result
