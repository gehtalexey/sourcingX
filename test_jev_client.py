"""Tests for jev_client.py.

No real network calls, no real TYPESAFE_API_KEY, and no dependency on the
`typesafe_sdk` package being installed -- every Jev "answer" object here is
a small fake class with plain attributes (noul / score / confidence /
legend / choice), matching the duck-typed field reads jev_client._get_field
performs on a real typesafe_sdk pydantic model. Question-building
(build_questions) is exercised separately with typesafe_sdk's Noul/Score/
Choice monkeypatched to simple fakes, so that code path is also covered
without the real package.
"""
import pytest

import jev_client


# ============================================================================
# Fakes -- match the REAL typesafe_sdk shapes verified from
# agent-kalamata's origin/feat/jev-phase2-shadow-runner branch.
# ============================================================================

class FakeNoulAnswer:
    def __init__(self, noul):
        self.noul = noul


class FakeScoreAnswer:
    def __init__(self, score, confidence, legend):
        self.score = score
        self.confidence = confidence
        self.legend = legend


class FakeChoiceAnswer:
    def __init__(self, choice, confidence):
        self.choice = choice
        self.confidence = confidence


class FakeUsage:
    def __init__(self, input_tokens=1234, output_tokens=12):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeSystemOneResponse:
    def __init__(self, answers, model="jev-1.13.0", usage=None):
        self.answers = answers
        self.model = model
        self.usage = usage or FakeUsage()


class FakeJevClient:
    """Records every call it receives; returns a preset response, or raises
    a preset exception."""

    def __init__(self, response=None, exception=None, fail_times=0):
        self.calls = []
        self._response = response
        self._exception = exception
        self._fail_times = fail_times

    def system_one(self, *, state, questions, **kwargs):
        self.calls.append({"state": state, "questions": questions})
        if self._fail_times > 0 and len(self.calls) <= self._fail_times:
            raise self._exception
        if self._exception is not None and self._fail_times == 0:
            raise self._exception
        return self._response


FIT_LEGEND = {i: level for i, level in enumerate(jev_client.FIT_SCORE_LEVELS)}
REJECT_CRITERIA = dict(jev_client.GENERIC_REJECT_REASON_CRITERIA)


def qualifying_answers(fit_score=2.0, fit_confidence=0.9, hf_noul=0.95):
    return {
        jev_client.QUESTION_ID_HARD_FILTER: FakeNoulAnswer(hf_noul),
        jev_client.QUESTION_ID_FIT_SCORE: FakeScoreAnswer(fit_score, fit_confidence, FIT_LEGEND),
        jev_client.QUESTION_ID_REJECT_REASON: FakeChoiceAnswer("unclear_or_other", 0.5),
    }


def rejecting_answers(reason="does_not_meet_requirements", reject_confidence=0.9, hf_noul=0.05):
    return {
        jev_client.QUESTION_ID_HARD_FILTER: FakeNoulAnswer(hf_noul),
        jev_client.QUESTION_ID_FIT_SCORE: FakeScoreAnswer(0.0, 0.9, FIT_LEGEND),
        jev_client.QUESTION_ID_REJECT_REASON: FakeChoiceAnswer(reason, reject_confidence),
    }


# ============================================================================
# build_candidate_text / strip_personal_fields
# ============================================================================

def test_build_candidate_text_strips_pii_keeps_career_data():
    profile = {
        "raw_data": {
            "name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "+1-415-555-0180",
            "linkedin_url": "https://linkedin.com/in/janedoe",
            "headline": "Senior Backend Engineer",
            "summary": "Owns the core payments service end to end.",
            "current_employers": [{
                "employee_title": "Senior Backend Engineer",
                "employer_name": "Acme Corp",
                "start_date": "2023-01", "end_date": None,
                "description": "Built the payments API in Python.",
            }],
            "skills": ["Python", "PostgreSQL"],
        }
    }
    text = jev_client.build_candidate_text(profile)
    assert "Jane Doe" not in text
    assert "jane@example.com" not in text
    assert "+1-415-555-0180" not in text
    assert "linkedin.com/in/janedoe" not in text
    assert "Senior Backend Engineer" in text
    assert "Acme Corp" in text
    assert "Python" in text


def test_build_candidate_text_handles_flat_fields_fallback():
    profile = {"current_title": "Data Engineer", "current_company": "Foo Inc"}
    text = jev_client.build_candidate_text(profile)
    assert "Data Engineer" in text
    assert "Foo Inc" in text


def test_build_candidate_text_never_crashes_on_empty_profile():
    assert jev_client.build_candidate_text({}) == "(no profile data available)"
    assert jev_client.build_candidate_text(None) == "(no profile data available)"


def test_strip_personal_fields_never_mutates_input():
    raw = {"name": "Jane", "skills": ["Python"]}
    stripped = jev_client.strip_personal_fields(raw)
    assert "name" not in stripped
    assert raw["name"] == "Jane"  # original untouched


# ============================================================================
# build_reject_reason_criteria
# ============================================================================

def test_reject_reason_criteria_generic_without_brief():
    criteria = jev_client.build_reject_reason_criteria(None)
    assert jev_client.UNCLEAR_OR_OTHER in criteria


def test_reject_reason_criteria_from_screening_brief():
    brief = {
        "must_haves": ["5+ years Python"],
        "exclusions": ["No agency background"],
    }
    criteria = jev_client.build_reject_reason_criteria(brief)
    assert any("5+ years Python" in v for v in criteria.values())
    assert any("No agency background" in v for v in criteria.values())
    assert jev_client.UNCLEAR_OR_OTHER in criteria


# ============================================================================
# propose_verdict -- happy paths
# ============================================================================

def test_propose_verdict_qualified():
    verdict = jev_client.propose_verdict(
        qualifying_answers(fit_score=2.0, fit_confidence=0.9), REJECT_CRITERIA
    )
    assert verdict["screening_result"] == "qualified"
    assert verdict["send_to_luna"] is False
    assert verdict["screening_score"] >= jev_client.QUALIFY_SCORE_FLOOR
    assert "Fit score:" in verdict["reason"]
    # No invented reasoning text -- only the level's own label appears.
    assert verdict["reason"] == "Fit score: qualified"


def test_propose_verdict_not_qualified_low_fit_score():
    verdict = jev_client.propose_verdict(
        qualifying_answers(fit_score=1.0, fit_confidence=0.9), REJECT_CRITERIA
    )
    assert verdict["screening_result"] == "not_qualified"
    assert verdict["send_to_luna"] is False


def test_propose_verdict_hard_filter_fails_returns_reject_reason():
    verdict = jev_client.propose_verdict(rejecting_answers(), REJECT_CRITERIA)
    assert verdict["screening_result"] == "not_qualified"
    assert verdict["send_to_luna"] is False
    assert verdict["screening_score"] == jev_client.FIT_SCORE_ANCHORS[0]
    assert "Hard filter:" in verdict["reason"]
    assert REJECT_CRITERIA["does_not_meet_requirements"] in verdict["reason"]


# ============================================================================
# propose_verdict -- fail open, every branch
# ============================================================================

def test_propose_verdict_defers_on_non_dict_answers():
    verdict = jev_client.propose_verdict(None, REJECT_CRITERIA)
    assert verdict["screening_result"] is None
    assert verdict["send_to_luna"] is True


def test_propose_verdict_defers_on_missing_hard_filter():
    answers = qualifying_answers()
    del answers[jev_client.QUESTION_ID_HARD_FILTER]
    verdict = jev_client.propose_verdict(answers, REJECT_CRITERIA)
    assert verdict["screening_result"] is None
    assert verdict["send_to_luna"] is True


def test_propose_verdict_defers_on_missing_fit_score():
    answers = qualifying_answers()
    del answers[jev_client.QUESTION_ID_FIT_SCORE]
    verdict = jev_client.propose_verdict(answers, REJECT_CRITERIA)
    assert verdict["screening_result"] is None


def test_propose_verdict_defers_on_missing_reject_reason():
    answers = qualifying_answers()
    del answers[jev_client.QUESTION_ID_REJECT_REASON]
    verdict = jev_client.propose_verdict(answers, REJECT_CRITERIA)
    assert verdict["screening_result"] is None


def test_propose_verdict_defers_on_uncertain_hard_filter():
    # noul near 0.5 -> hf_confidence = abs(noul - 0.5) * 2 < MIN_CONFIDENCE
    verdict = jev_client.propose_verdict(qualifying_answers(hf_noul=0.52), REJECT_CRITERIA)
    assert verdict["screening_result"] is None
    assert verdict["send_to_luna"] is True


def test_propose_verdict_defers_on_low_fit_confidence():
    verdict = jev_client.propose_verdict(
        qualifying_answers(fit_confidence=0.3), REJECT_CRITERIA
    )
    assert verdict["screening_result"] is None


def test_propose_verdict_defers_on_fit_score_out_of_range():
    answers = qualifying_answers()
    answers[jev_client.QUESTION_ID_FIT_SCORE] = FakeScoreAnswer(99.0, 0.9, FIT_LEGEND)
    verdict = jev_client.propose_verdict(answers, REJECT_CRITERIA)
    assert verdict["screening_result"] is None


def test_propose_verdict_defers_on_low_reject_confidence():
    verdict = jev_client.propose_verdict(
        rejecting_answers(reject_confidence=0.1), REJECT_CRITERIA
    )
    assert verdict["screening_result"] is None


def test_propose_verdict_defers_on_unrecognized_reject_choice():
    verdict = jev_client.propose_verdict(
        rejecting_answers(reason="some_made_up_reason"), REJECT_CRITERIA
    )
    assert verdict["screening_result"] is None


def test_propose_verdict_defers_on_unclear_or_other_reject_choice():
    verdict = jev_client.propose_verdict(
        rejecting_answers(reason=jev_client.UNCLEAR_OR_OTHER), REJECT_CRITERIA
    )
    assert verdict["screening_result"] is None
    assert verdict["send_to_luna"] is True


# ============================================================================
# _interpolate_screening_score
# ============================================================================

@pytest.mark.parametrize("level_score,expected", [
    (-1.0, jev_client.FIT_SCORE_ANCHORS[0]),
    (0.0, jev_client.FIT_SCORE_ANCHORS[0]),
    (3.0, jev_client.FIT_SCORE_ANCHORS[-1]),
    (99.0, jev_client.FIT_SCORE_ANCHORS[-1]),
])
def test_interpolate_screening_score_clamps_at_edges(level_score, expected):
    assert jev_client._interpolate_screening_score(level_score) == expected


def test_interpolate_screening_score_interpolates_midpoint():
    # Between level 1 (anchor 4) and level 2 (anchor 7): 1.5 -> 5.5 -> round to 6
    assert jev_client._interpolate_screening_score(1.5) == 6


# ============================================================================
# call_jev -- success, non-transient failure, transient retry
# ============================================================================

def test_call_jev_success_returns_response_no_error():
    response = FakeSystemOneResponse(qualifying_answers())
    client = FakeJevClient(response=response)
    result, error, latency_ms = jev_client.call_jev(client, "profile text", {})
    assert result is response
    assert error is None
    assert latency_ms >= 0
    assert len(client.calls) == 1


def test_call_jev_non_transient_failure_never_retries():
    client = FakeJevClient(exception=ValueError("bad request"))
    result, error, latency_ms = jev_client.call_jev(client, "profile text", {})
    assert result is None
    assert "bad request" in error
    assert len(client.calls) == 1  # no retry on a non-transient error


def test_call_jev_retries_on_transient_error_then_succeeds(monkeypatch):
    class FakeTransientError(Exception):
        pass

    monkeypatch.setattr(jev_client, "TYPESAFE_SDK_AVAILABLE", True)
    monkeypatch.setattr(jev_client, "TypeSafeAPIConnectionError", FakeTransientError)
    monkeypatch.setattr(jev_client, "TypeSafeAPITimeoutError", FakeTransientError)
    monkeypatch.setattr(jev_client, "TypeSafeInternalServerError", FakeTransientError)
    monkeypatch.setattr(jev_client, "TypeSafeRateLimitError", FakeTransientError)
    monkeypatch.setattr(jev_client, "JEV_RETRY_BASE_DELAY_SECONDS", 0)  # keep test fast

    response = FakeSystemOneResponse(qualifying_answers())
    client = FakeJevClient(response=response, exception=FakeTransientError("timeout"), fail_times=2)
    result, error, latency_ms = jev_client.call_jev(client, "profile text", {})
    assert result is response
    assert error is None
    assert len(client.calls) == 3  # 2 failures + 1 success, within JEV_MAX_ATTEMPTS


def test_call_jev_gives_up_after_max_attempts(monkeypatch):
    class FakeTransientError(Exception):
        pass

    monkeypatch.setattr(jev_client, "TYPESAFE_SDK_AVAILABLE", True)
    monkeypatch.setattr(jev_client, "TypeSafeAPIConnectionError", FakeTransientError)
    monkeypatch.setattr(jev_client, "TypeSafeAPITimeoutError", FakeTransientError)
    monkeypatch.setattr(jev_client, "TypeSafeInternalServerError", FakeTransientError)
    monkeypatch.setattr(jev_client, "TypeSafeRateLimitError", FakeTransientError)
    monkeypatch.setattr(jev_client, "JEV_RETRY_BASE_DELAY_SECONDS", 0)

    client = FakeJevClient(exception=FakeTransientError("still down"), fail_times=jev_client.JEV_MAX_ATTEMPTS)
    result, error, latency_ms = jev_client.call_jev(client, "profile text", {})
    assert result is None
    assert "still down" in error
    assert len(client.calls) == jev_client.JEV_MAX_ATTEMPTS


# ============================================================================
# build_questions / build_client -- typesafe_sdk availability
# ============================================================================

def test_build_questions_raises_when_sdk_unavailable(monkeypatch):
    monkeypatch.setattr(jev_client, "TYPESAFE_SDK_AVAILABLE", False)
    with pytest.raises(jev_client.JevUnavailableError):
        jev_client.build_questions("Some job description")


def test_build_client_raises_when_sdk_unavailable(monkeypatch):
    monkeypatch.setattr(jev_client, "TYPESAFE_SDK_AVAILABLE", False)
    with pytest.raises(jev_client.JevUnavailableError):
        jev_client.build_client()


def test_build_questions_shape_with_fake_sdk_classes(monkeypatch):
    """Exercises build_questions()'s real logic (which question gets which
    instructions/criteria) with typesafe_sdk's Noul/Score/Choice
    monkeypatched to simple fakes -- no real package needed."""

    class FakeNoul:
        def __init__(self, instructions):
            self.instructions = instructions

    class FakeScore:
        def __init__(self, instructions, criteria):
            self.instructions = instructions
            self.criteria = criteria

    class FakeChoice:
        def __init__(self, instructions, criteria):
            self.instructions = instructions
            self.criteria = criteria

    monkeypatch.setattr(jev_client, "TYPESAFE_SDK_AVAILABLE", True)
    monkeypatch.setattr(jev_client, "Noul", FakeNoul)
    monkeypatch.setattr(jev_client, "Score", FakeScore)
    monkeypatch.setattr(jev_client, "Choice", FakeChoice)

    questions = jev_client.build_questions(
        "Senior backend engineer, Python, 5+ years",
        screening_brief={"must_haves": ["5+ years Python"], "exclusions": []},
    )

    assert set(questions) == {
        jev_client.QUESTION_ID_HARD_FILTER,
        jev_client.QUESTION_ID_FIT_SCORE,
        jev_client.QUESTION_ID_REJECT_REASON,
    }
    hard_filter = questions[jev_client.QUESTION_ID_HARD_FILTER]
    assert isinstance(hard_filter, FakeNoul)
    assert "Senior backend engineer" in hard_filter.instructions

    fit_score = questions[jev_client.QUESTION_ID_FIT_SCORE]
    assert isinstance(fit_score, FakeScore)
    assert fit_score.criteria == jev_client.FIT_SCORE_LEVELS  # ordered list, not a dict

    reject_reason = questions[jev_client.QUESTION_ID_REJECT_REASON]
    assert isinstance(reject_reason, FakeChoice)
    assert isinstance(reject_reason.criteria, dict)  # dict, not a list
    assert jev_client.UNCLEAR_OR_OTHER in reject_reason.criteria


# ============================================================================
# screen_with_jev -- end to end with a fake client (no network, no key)
# ============================================================================

def _patch_fake_sdk_question_types(monkeypatch):
    """screen_with_jev() calls build_questions() internally, which needs
    real (or faked) Noul/Score/Choice classes to construct question
    objects -- typesafe_sdk isn't installed in this test environment, so
    tests that go through screen_with_jev() patch in simple fakes, same as
    test_build_questions_shape_with_fake_sdk_classes above."""

    class FakeNoul:
        def __init__(self, instructions):
            self.instructions = instructions

    class FakeScore:
        def __init__(self, instructions, criteria):
            self.instructions = instructions
            self.criteria = criteria

    class FakeChoice:
        def __init__(self, instructions, criteria):
            self.instructions = instructions
            self.criteria = criteria

    monkeypatch.setattr(jev_client, "TYPESAFE_SDK_AVAILABLE", True)
    monkeypatch.setattr(jev_client, "Noul", FakeNoul)
    monkeypatch.setattr(jev_client, "Score", FakeScore)
    monkeypatch.setattr(jev_client, "Choice", FakeChoice)


def test_screen_with_jev_end_to_end_qualified(monkeypatch):
    _patch_fake_sdk_question_types(monkeypatch)
    response = FakeSystemOneResponse(
        qualifying_answers(fit_score=2.0, fit_confidence=0.9), model="jev-1.13.0",
        usage=FakeUsage(input_tokens=500, output_tokens=10),
    )
    client = FakeJevClient(response=response)
    profile = {"raw_data": {"name": "Jane Doe", "current_title": "Backend Engineer",
                             "current_company": "Acme"}}

    verdict = jev_client.screen_with_jev(profile, "Senior backend engineer", client=client)

    assert verdict["screening_result"] == "qualified"
    assert verdict["jev_model"] == "jev-1.13.0"
    assert verdict["input_tokens"] == 500
    assert verdict["output_tokens"] == 10
    assert verdict["latency_ms"] >= 0
    # The call actually made to the fake client never carries the candidate's name.
    assert "Jane Doe" not in client.calls[0]["state"]


def test_screen_with_jev_call_failure_defers_instead_of_rejecting(monkeypatch):
    _patch_fake_sdk_question_types(monkeypatch)
    client = FakeJevClient(exception=RuntimeError("network down"))
    profile = {"current_title": "Backend Engineer", "current_company": "Acme"}

    verdict = jev_client.screen_with_jev(profile, "Senior backend engineer", client=client)

    assert verdict["screening_result"] is None
    assert verdict["send_to_luna"] is True
    assert "Jev call failed" in verdict["reason"]
    assert "network down" in verdict["reason"]


def test_screen_with_jev_without_client_never_calls_real_sdk_when_unavailable():
    """client=None means "build a real one" -- with typesafe_sdk not
    installed in this test environment, that must raise a clear error
    immediately, never attempt a network call or need TYPESAFE_API_KEY."""
    if jev_client.TYPESAFE_SDK_AVAILABLE:
        pytest.skip("typesafe_sdk is installed in this environment; "
                     "unavailability path not exercised here")
    with pytest.raises(jev_client.JevUnavailableError):
        jev_client.screen_with_jev({"current_title": "x"}, "some JD", client=None)
