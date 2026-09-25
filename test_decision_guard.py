"""
Tests for the deterministic decision-guard safety catch
(``dashboard._verdicts_force_no_go``).

Background: in the structured screening path, the model returns a
per-criterion verdict on every must-have (``met``) and exclusion
(``matched``) AND a separate top-level ``decision`` (GO/NO GO). Nothing
previously checked the two against each other, so a model that marked a
must-have "met": false could still return "decision": "GO" and the app
would take the GO at face value.

``_verdicts_force_no_go`` is the deterministic guard that enforces the
rule the prompt already states: GO only when every must-have is met and
no exclusion matched. It is intentionally conservative — with no verdicts
at all it never overrides a decision the model made without emitting any.

The guard is tri-state (via the ``_tri`` helper): a verdict field is only
ever treated as an explicit True or False signal; anything else (missing
key, None, empty string, an unrecognized value) is "unknown" and NEVER
counted as a failure/match. This matters because a malformed or incomplete
verdict (e.g. a must-have with no "met" key at all) must never wrongly
force a legitimate GO into a NO GO.

These tests call the pure helper directly. They never call an LLM and
never hit the network.
"""

from __future__ import annotations

import pytest

import db

from dashboard import (
    _align_verdicts_by_id,
    _with_criterion_text,
    _tri,
    _verdicts_force_no_go,
    _must_have_verdict_state,
    _verdicts_needs_verification,
    _decision_to_fit_label,
    _stability_verdict_failed,
    _resolve_three_state_decision,
    _result_bucket,
    _hard_filter_failure_named,
)


class TestVerdictsForceNoGo:
    """_verdicts_force_no_go(must_have_verdicts, exclusion_verdicts) ->
    (force_no_go: bool, reason: str)."""

    def test_all_must_haves_met_no_exclusion_matched(self):
        """Clean GO case: every must-have met, no exclusion matched ->
        no override."""
        must_haves = [
            {"text": "5+ years Python", "met": True},
            {"text": "AWS experience", "met": True},
        ]
        exclusions = [
            {"text": "Currently at a competitor", "matched": False},
        ]
        force, reason = _verdicts_force_no_go(must_haves, exclusions)
        assert force is False
        assert reason == ""

    def test_one_must_have_not_met_forces_no_go(self):
        """A single must-have marked met=False must force NO GO, and the
        reason must mention which one failed."""
        must_haves = [
            {"text": "5+ years Python", "met": True},
            {"text": "Team lead experience", "met": False},
        ]
        exclusions = []
        force, reason = _verdicts_force_no_go(must_haves, exclusions)
        assert force is True
        assert "Team lead experience" in reason

    def test_exclusion_matched_forces_no_go(self):
        """A matched exclusion must force NO GO, and the reason must
        mention which one matched."""
        must_haves = [
            {"text": "5+ years Python", "met": True},
        ]
        exclusions = [
            {"text": "Currently at a competitor", "matched": True},
        ]
        force, reason = _verdicts_force_no_go(must_haves, exclusions)
        assert force is True
        assert "Currently at a competitor" in reason

    def test_empty_or_none_verdicts_never_override(self):
        """Conservative default: if there are NO verdicts at all (empty
        lists, or None), never force an override — the model may not have
        emitted any verdicts for this call, and we must not second-guess
        a decision it made without them."""
        assert _verdicts_force_no_go([], []) == (False, "")
        assert _verdicts_force_no_go(None, None) == (False, "")
        assert _verdicts_force_no_go(None, []) == (False, "")
        assert _verdicts_force_no_go([], None) == (False, "")

    @pytest.mark.parametrize("met_value, expect_force", [
        (True, False),
        (False, True),
        (1, False),
        (1.0, False),
        (0, True),
        (0.0, True),
        ("true", False),
        ("True", False),
        ("yes", False),
        ("Y", False),
        ("1", False),
        ("false", True),
        ("False", True),
        ("no", True),
        ("n", True),
        ("0", True),
    ])
    def test_met_truthiness_is_robust_for_explicit_values(self, met_value, expect_force):
        """The model sometimes returns 'met' as a literal string ("true"/
        "false"/"yes"/"no"/"1"/"0") or a numeric 1/0 instead of a JSON
        boolean. The guard must treat all of these explicit forms the same
        as the native boolean."""
        must_haves = [{"text": "Some requirement", "met": met_value}]
        force, _reason = _verdicts_force_no_go(must_haves, [])
        assert force is expect_force

    @pytest.mark.parametrize("matched_value, expect_force", [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("True", True),
        ("yes", True),
        ("1", True),
        ("false", False),
        ("False", False),
        ("no", False),
        ("0", False),
    ])
    def test_matched_truthiness_is_robust_for_explicit_values(self, matched_value, expect_force):
        """Same explicit-value robustness for exclusion 'matched' values."""
        exclusions = [{"text": "Some exclusion", "matched": matched_value}]
        force, _reason = _verdicts_force_no_go([], exclusions)
        assert force is expect_force

    def test_must_have_missing_met_key_entirely_is_not_a_failure(self):
        """A must-have verdict that never included a 'met' key at all (e.g.
        the model only returned {"text": ..., "evidence": ...}) is malformed,
        not an explicit contradiction. It must NOT force NO GO."""
        must_haves = [{"text": "5+ years Python", "evidence": "unclear"}]
        force, reason = _verdicts_force_no_go(must_haves, [])
        assert force is False
        assert reason == ""

    def test_must_have_met_none_is_not_a_failure(self):
        """met=None (explicit null in the JSON) is ambiguous, not a
        contradiction -- must not force NO GO."""
        must_haves = [{"text": "5+ years Python", "met": None}]
        force, reason = _verdicts_force_no_go(must_haves, [])
        assert force is False
        assert reason == ""

    def test_must_have_met_unrecognized_string_is_not_a_failure(self):
        """An unrecognized string value (not one of the known truthy/falsy
        tokens) is ambiguous, not an explicit contradiction."""
        must_haves = [{"text": "5+ years Python", "met": "partial"}]
        force, reason = _verdicts_force_no_go(must_haves, [])
        assert force is False
        assert reason == ""

    def test_exclusion_missing_matched_key_entirely_is_not_a_match(self):
        """An exclusion verdict with no 'matched' key at all must not be
        treated as matched."""
        exclusions = [{"text": "Currently at a competitor", "evidence": "n/a"}]
        force, reason = _verdicts_force_no_go([], exclusions)
        assert force is False
        assert reason == ""

    def test_exclusion_matched_none_is_not_a_match(self):
        """matched=None is ambiguous, not an explicit match."""
        exclusions = [{"text": "Currently at a competitor", "matched": None}]
        force, reason = _verdicts_force_no_go([], exclusions)
        assert force is False
        assert reason == ""

    def test_mixed_explicit_and_ambiguous_must_haves_only_flags_explicit(self):
        """When one must-have is explicitly failed and another has a
        missing/ambiguous 'met', only the explicit failure should appear in
        the reason and force the override -- the ambiguous one is ignored."""
        must_haves = [
            {"text": "5+ years Python", "met": False},
            {"text": "AWS experience"},  # no 'met' key at all
        ]
        force, reason = _verdicts_force_no_go(must_haves, [])
        assert force is True
        assert "5+ years Python" in reason
        assert "AWS experience" not in reason

    def test_both_failed_must_have_and_matched_exclusion_combine_in_reason(self):
        """When both a failed must-have and a matched exclusion are
        present, the reason should mention both."""
        must_haves = [{"text": "Must know Kubernetes", "met": False}]
        exclusions = [{"text": "Based outside Israel", "matched": True}]
        force, reason = _verdicts_force_no_go(must_haves, exclusions)
        assert force is True
        assert "Must know Kubernetes" in reason
        assert "Based outside Israel" in reason


class TestTri:
    """Direct tests of the _tri(v) tri-state parser used by
    _verdicts_force_no_go. Only explicit truthy/falsy signals return
    True/False; everything else (missing, None, empty, unrecognized,
    wrong type) returns None ("unknown")."""

    @pytest.mark.parametrize("value, expected", [
        (True, True),
        (False, False),
        (1, True),
        (1.0, True),
        (0, False),
        (0.0, False),
        ("true", True),
        ("TRUE", True),
        (" yes ", True),
        ("y", True),
        ("1", True),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("n", False),
        ("0", False),
        (None, None),
        ("", None),
        ("partial", None),
        ("maybe", None),
        (2, None),
        ([], None),
        ({}, None),
    ])
    def test_tri(self, value, expected):
        assert _tri(value) is expected


class TestMustHaveVerdictState:
    """_must_have_verdict_state(v) -> 'met' | 'not_met' | 'needs_verification'
    | None. Three-way parser for the new must-have 'met' field, with a
    legacy boolean accepted for backward compatibility."""

    @pytest.mark.parametrize("value, expected", [
        ("met", "met"),
        ("not_met", "not_met"),
        ("not met", "not_met"),
        ("needs_verification", "needs_verification"),
        ("needs verification", "needs_verification"),
        (True, "met"),
        (False, "not_met"),
        ("true", "met"),
        ("false", "not_met"),
        (None, None),
        ("", None),
        ("partial", None),
    ])
    def test_states(self, value, expected):
        assert _must_have_verdict_state(value) == expected


class TestNeedsVerificationGuard:
    """_verdicts_needs_verification(must_have_verdicts) ->
    (needs_verification: bool, items: list[str])."""

    def test_one_needs_verification_item_flagged(self):
        must_haves = [
            {"text": "5+ years Python", "met": "met"},
            {"text": "EU-based", "met": "needs_verification"},
        ]
        needs_verif, items = _verdicts_needs_verification(must_haves)
        assert needs_verif is True
        assert items == ["EU-based"]

    def test_no_needs_verification_items(self):
        must_haves = [{"text": "5+ years Python", "met": "met"}]
        assert _verdicts_needs_verification(must_haves) == (False, [])

    def test_not_met_item_is_not_counted_as_needs_verification(self):
        must_haves = [{"text": "5+ years Python", "met": "not_met"}]
        assert _verdicts_needs_verification(must_haves) == (False, [])

    def test_empty_or_none_never_flags(self):
        assert _verdicts_needs_verification([]) == (False, [])
        assert _verdicts_needs_verification(None) == (False, [])


class TestThreeStateDecisionGuardIntegration:
    """The combination of _verdicts_force_no_go and
    _verdicts_needs_verification is what screen_profile uses to pick the
    final decision. These tests pin that combined behaviour directly,
    without calling an LLM."""

    def test_one_not_met_plus_others_met_forces_no_go(self):
        must_haves = [
            {"text": "5+ years Python", "met": "met"},
            {"text": "Located in Europe", "met": "not_met"},
        ]
        force_no_go, guard_reason = _verdicts_force_no_go(must_haves, [])
        assert force_no_go is True
        assert "Located in Europe" in guard_reason

    def test_no_not_met_one_needs_verification_is_not_a_force_no_go(self):
        # A needs_verification must-have is unproven, not contradicted --
        # it must never force NO GO on its own. There is no manual-review
        # bucket any more: whether this ends up GO or NO GO is decided by
        # _resolve_three_state_decision from the score (see
        # TestResolveThreeStateDecision), not by this guard.
        must_haves = [
            {"text": "5+ years Python", "met": "met"},
            {"text": "Located in Europe", "met": "needs_verification"},
        ]
        force_no_go, _ = _verdicts_force_no_go(must_haves, [])
        needs_verif, items = _verdicts_needs_verification(must_haves)
        assert force_no_go is False
        assert needs_verif is True
        assert items == ["Located in Europe"]

    def test_all_met_high_score_is_a_go(self):
        must_haves = [
            {"text": "5+ years Python", "met": "met"},
            {"text": "Located in Europe", "met": "met"},
        ]
        force_no_go, _ = _verdicts_force_no_go(must_haves, [])
        needs_verif, _ = _verdicts_needs_verification(must_haves)
        assert force_no_go is False
        assert needs_verif is False
        assert _decision_to_fit_label("GO", 9) == "Good Fit"

    def test_legacy_boolean_met_false_still_forces_no_go(self):
        # Backward compatibility: a model that returns the old boolean
        # schema (met: true/false) must still be treated as met/not_met.
        must_haves = [{"text": "5+ years Python", "met": False}]
        force_no_go, guard_reason = _verdicts_force_no_go(must_haves, [])
        assert force_no_go is True
        assert "5+ years Python" in guard_reason


class TestDecisionToFitLabelKeepsThreshold:
    """_decision_to_fit_label(decision, score) -> str. The structured path
    only ever calls this with a decision _resolve_three_state_decision
    already computed (GO implies score >= GO_CONFIDENCE_THRESHOLD there),
    so the threshold check here is redundant for it -- but the LEGACY
    freeform path (screen_profile called without a screening_brief) never
    runs the resolver and passes the model's raw {decision, score}
    straight through. Without keeping the threshold here too, a legacy
    {"decision": "GO", "score": 6} would wrongly become "Good Fit" instead
    of "Not a Fit" (Codex review, PR #150). There is still no third "Maybe"
    bucket -- a GO below the threshold is "Not a Fit", same as a NO GO."""

    def test_no_go_is_always_not_a_fit_regardless_of_score(self):
        assert _decision_to_fit_label("NO GO", 9) == "Not a Fit"
        assert _decision_to_fit_label("NO GO", 1) == "Not a Fit"

    def test_go_at_or_above_threshold_is_good_fit(self):
        assert _decision_to_fit_label("GO", 7) == "Good Fit"
        assert _decision_to_fit_label("GO", 10) == "Good Fit"

    def test_go_below_threshold_is_not_a_fit_not_maybe(self):
        # This is the legacy-path regression: a low-score GO must not
        # become "Good Fit", and must NOT become "Maybe" either -- there
        # is no manual-review bucket any more.
        assert _decision_to_fit_label("GO", 6) == "Not a Fit"
        assert _decision_to_fit_label("GO", 1) == "Not a Fit"


class TestPolicyNoLongerTreatsMissingEvidenceAsFail:
    """screening_policy.py must no longer instruct the model to treat
    missing/unproven evidence as a fail, and must instead spell out that a
    CONTRADICTION is required for not_met."""

    def test_policy_no_longer_says_missing_evidence_is_a_fail(self):
        from screening_policy import SCREENING_POLICY
        assert "if the profile lacks evidence it is satisfied, treat it as a fail" not in SCREENING_POLICY

    def test_policy_states_contradiction_required_for_not_met(self):
        from screening_policy import SCREENING_POLICY
        assert "not_met" in SCREENING_POLICY
        assert "CONTRADICTS" in SCREENING_POLICY

    def test_policy_states_needs_verification_for_absent_evidence(self):
        from screening_policy import SCREENING_POLICY
        assert "needs_verification" in SCREENING_POLICY
        assert "absence of evidence is never" in SCREENING_POLICY.lower() or \
               "absence of evidence is NEVER" in SCREENING_POLICY

    def test_policy_no_longer_lists_insufficient_data_as_a_score_reason(self):
        from screening_policy import SCREENING_POLICY
        assert "insufficient data (NO GO)" not in SCREENING_POLICY
        assert "insufficient data is not a score reason" in SCREENING_POLICY.lower()

    def test_policy_states_defining_requirement_score_cap(self):
        from screening_policy import SCREENING_POLICY
        assert "the requirement named in the role title" in SCREENING_POLICY
        assert "A strong engineer with no signal on the defining requirement is a 6, not a 7." in SCREENING_POLICY

    def test_no_needs_verification_decision_text_left_in_schema(self):
        from screening_policy import SCREENING_POLICY
        assert '"decision": "GO" or "NO GO" or "NEEDS VERIFICATION"' not in SCREENING_POLICY
        assert '"decision": "GO" or "NO GO"' in SCREENING_POLICY
        assert "GO/NO GO/NEEDS VERIFICATION" not in SCREENING_POLICY


class TestStabilityVerdictFailed:
    """_stability_verdict_failed(durations_text) -> bool. Parses the exact
    STABILITY VERDICT marker compute_role_durations() emits."""

    def test_fail_marker_detected(self):
        text = "some lines\n>>> STABILITY VERDICT: FAIL — 3 short-stint companies >= 3 → MAX SCORE 4 <<<\nmore"
        assert _stability_verdict_failed(text) is True

    def test_pass_marker_not_a_failure(self):
        text = "some lines\n>>> STABILITY VERDICT: PASS <<<\nmore"
        assert _stability_verdict_failed(text) is False

    def test_empty_or_none_text_not_a_failure(self):
        assert _stability_verdict_failed("") is False
        assert _stability_verdict_failed(None) is False


class TestResolveThreeStateDecision:
    """_resolve_three_state_decision(...) -> (decision, note). There is no
    manual-review bucket -- every candidate ends GO or NO GO, decided
    purely from the verdicts and score, regardless of what the model's own
    top-level `decision` field said:
      1. Any not_met must-have or matched exclusion -> NO GO.
      2. Otherwise a real hard filter / STABILITY VERDICT FAIL -> NO GO.
      3. Otherwise GO only when score >= GO_CONFIDENCE_THRESHOLD (7); any
         needs_verification must-haves ride along as a "Verify in call"
         note, never lowering the score themselves.
      4. Otherwise (score below threshold) -> NO GO, with a "Not shown..."
         note when needs_verification must-haves were involved.
    A must-have the model never returned a verdict for (fewer verdicts
    than the brief listed) or gave an unparseable state for is treated as
    needs_verification too -- a GO must never rest on a must-have the
    model never actually judged."""

    def test_not_met_forces_no_go_regardless_of_score(self):
        must_haves = [{"text": "5+ years Python", "met": "not_met"}]
        decision, note = _resolve_three_state_decision(
            "GO", must_haves, [], score=9,
        )
        assert decision == "NO GO"
        assert "5+ years Python" in note

    def test_needs_verification_high_score_is_go_with_verify_in_call_note(self):
        must_haves = [
            {"text": "5+ years Python", "met": "met"},
            {"text": "EU-based", "met": "needs_verification"},
        ]
        decision, note = _resolve_three_state_decision(
            "GO", must_haves, [], score=8,
        )
        assert decision == "GO"
        assert note.startswith("Verify in call:")
        assert "EU-based" in note

    def test_needs_verification_low_score_is_no_go_with_not_shown_note(self):
        must_haves = [
            {"text": "5+ years Python", "met": "met"},
            {"text": "EU-based", "met": "needs_verification"},
        ]
        decision, note = _resolve_three_state_decision(
            "GO", must_haves, [], score=6,
        )
        assert decision == "NO GO"
        assert note.startswith("Not shown and the visible career doesn't clearly imply:")
        assert "EU-based" in note

    def test_all_met_score_six_is_no_go(self):
        must_haves = [{"text": "5+ years Python", "met": "met"}]
        decision, note = _resolve_three_state_decision("GO", must_haves, [], score=6)
        assert decision == "NO GO"

    def test_all_met_score_seven_is_go(self):
        must_haves = [{"text": "5+ years Python", "met": "met"}]
        decision, note = _resolve_three_state_decision("GO", must_haves, [], score=7)
        assert decision == "GO"
        assert note == ""

    def test_hard_filter_with_score_nine_still_no_go(self):
        must_haves = [{"text": "5+ years Python", "met": "met"}]
        decision, note = _resolve_three_state_decision(
            "GO", must_haves, [], score=9,
            hard_filter_failed="Job hopper: 4 roles under 1 year",
        )
        assert decision == "NO GO"
        assert "Job hopper" in note

    # --- Answers are tied to criteria by ID (PR #150 review rounds 2-5 and
    # both expert consults, 2026-09-25). The prompt numbers each criterion
    # (M1, M2 / E1, E2); a criterion is answered only by EXACTLY ONE verdict
    # carrying its id. Any missing / repeated / unknown id, or an invalid
    # value, means the screening is incomplete: NO GO, never GO.

    MH = ["5+ years Python", "Kubernetes in production"]
    EX = ["Currently at a competitor", "No pure managers"]

    @staticmethod
    def _mh(cid, met="met", **extra):
        return {"id": cid, "met": met, **extra}

    @staticmethod
    def _ex(cid, matched=False, **extra):
        return {"id": cid, "matched": matched, **extra}

    def _resolve(self, must_haves, exclusions, score=9, mh=None, ex=None, **kw):
        return _resolve_three_state_decision(
            "GO", must_haves, exclusions, score=score,
            expected_must_haves=self.MH if mh is None else mh,
            expected_exclusions=self.EX if ex is None else ex, **kw,
        )

    def test_every_criterion_answered_once_can_go_and_order_does_not_matter(self):
        decision, note = self._resolve(
            [self._mh("M2"), self._mh("M1")], [self._ex("E2"), self._ex("E1")],
        )
        assert (decision, note) == ("GO", "")

    def test_ids_are_case_and_space_tolerant(self):
        decision, note = self._resolve(
            [self._mh(" m1 "), self._mh("M2")], [self._ex("e1"), self._ex("E2")],
        )
        assert (decision, note) == ("GO", "")

    def test_the_wording_the_model_sends_is_ignored(self):
        # Matching is by id only: whatever text comes back never decides
        # which criterion an answer belongs to.
        decision, note = self._resolve(
            [self._mh("M1", text="something else entirely"), self._mh("M2", text="5+ years Python")],
            [self._ex("E1", text="No pure managers"), self._ex("E2", text="")],
        )
        assert (decision, note) == ("GO", "")

    def test_similar_looking_requirements_stay_separate(self):
        # "C++" vs "C#" (and "Java" vs "Python", ".NET" vs "NET") can never
        # be confused, because nothing is compared by wording.
        decision, note = self._resolve(
            [self._mh("M1"), self._mh("M2", "not_met")], [],
            mh=["C++ experience", "C# experience"], ex=[],
        )
        assert decision == "NO GO"
        assert "C# experience" in note and "C++ experience" not in note

    def test_missing_must_have_answer_is_no_go_even_at_a_high_score(self):
        # No answer is not the same as "needs_verification": it is an
        # incomplete screening.
        decision, note = self._resolve([self._mh("M1")], [self._ex("E1"), self._ex("E2")], score=9)
        assert decision == "NO GO"
        assert "Must-have not judged: Kubernetes in production" in note

    def test_explicit_needs_verification_can_still_go_at_a_high_score(self):
        decision, note = self._resolve(
            [self._mh("M1"), self._mh("M2", "needs_verification")],
            [self._ex("E1"), self._ex("E2")], score=8,
        )
        assert decision == "GO"
        assert note == "Verify in call: Kubernetes in production"

    def test_explicit_needs_verification_at_a_low_score_is_no_go_naming_it(self):
        decision, note = self._resolve(
            [self._mh("M1"), self._mh("M2", "needs_verification")],
            [self._ex("E1"), self._ex("E2")], score=5,
        )
        assert decision == "NO GO"
        assert note.startswith("Not shown and the visible career doesn't clearly imply:")
        assert "Kubernetes in production" in note

    def test_same_must_have_answered_twice_is_incomplete(self):
        decision, note = self._resolve(
            [self._mh("M1"), self._mh("M1"), self._mh("M2")], [self._ex("E1"), self._ex("E2")],
        )
        assert decision == "NO GO"
        assert "Screening incomplete" in note and "M1 answered 2 times" in note

    def test_same_exclusion_answered_twice_leaves_the_other_unjudged(self):
        # Two exclusions asked; the same cleared one returned twice. A count
        # of verdicts would call that complete (Codex, PR #150 round 2).
        decision, note = self._resolve(
            [self._mh("M1"), self._mh("M2")], [self._ex("E1"), self._ex("E1")],
        )
        assert decision == "NO GO"
        assert "E1 answered 2 times" in note
        assert "Exclusion not judged: Currently at a competitor; No pure managers" in note

    def test_unknown_or_wrong_group_or_blank_id_is_incomplete(self):
        for bad in ("M7", "E1", "", None, "banana", 3):
            decision, note = self._resolve(
                [self._mh("M1"), self._mh("M2"), {"id": bad, "met": "met"}],
                [self._ex("E1"), self._ex("E2")],
            )
            assert decision == "NO GO", bad
            assert "Screening incomplete" in note, bad

    def test_an_answer_that_is_not_an_object_is_incomplete(self):
        decision, note = self._resolve(
            [self._mh("M1"), self._mh("M2"), "yes"], [self._ex("E1"), self._ex("E2")],
        )
        assert decision == "NO GO"
        assert "Screening incomplete" in note

    def test_answers_that_are_not_a_list_are_incomplete(self):
        decision, note = self._resolve({"M1": "met"}, [self._ex("E1"), self._ex("E2")])
        assert decision == "NO GO"
        assert "Screening incomplete" in note

    def test_invalid_must_have_value_is_not_judged(self):
        decision, note = self._resolve(
            [self._mh("M1", "banana"), self._mh("M2")], [self._ex("E1"), self._ex("E2")],
        )
        assert decision == "NO GO"
        assert "Must-have not judged: 5+ years Python" in note

    def test_invalid_exclusion_value_is_not_judged(self):
        decision, note = self._resolve(
            [self._mh("M1"), self._mh("M2")], [self._ex("E1", "partial"), self._ex("E2")],
        )
        assert decision == "NO GO"
        assert "Exclusion not judged: Currently at a competitor" in note

    def test_no_answers_at_all_is_no_go(self):
        decision, note = self._resolve([], [])
        assert decision == "NO GO"
        assert "Must-have not judged" in note and "Exclusion not judged" in note

    def test_no_criteria_asked_and_none_returned_can_go(self):
        decision, note = self._resolve([], [], score=8, mh=[], ex=[])
        assert (decision, note) == ("GO", "")

    def test_no_criteria_asked_but_answers_returned_is_incomplete(self):
        decision, note = self._resolve([self._mh("M1")], [], score=8, mh=[], ex=[])
        assert decision == "NO GO"
        assert "Screening incomplete" in note

    def test_explicit_not_met_without_any_text_still_blocks_go(self):
        decision, note = self._resolve(
            [self._mh("M1", "not_met"), self._mh("M2")], [self._ex("E1"), self._ex("E2")],
        )
        assert decision == "NO GO"
        assert "5+ years Python" in note  # the wording comes from the brief

    def test_matched_exclusion_blocks_go(self):
        decision, note = self._resolve(
            [self._mh("M1"), self._mh("M2")], [self._ex("E1"), self._ex("E2", True)],
        )
        assert decision == "NO GO"
        assert "No pure managers" in note

    def test_align_verdicts_by_id_reports_each_problem(self):
        aligned, problems = _align_verdicts_by_id(
            "M", ["a", "b"], [{"id": "M1"}, {"id": "M1"}, {"id": "E1"}, "x"],
        )
        assert [(cid, text, v) for cid, text, v in aligned] == [("M1", "a", None), ("M2", "b", None)]
        assert "M1 answered 2 times" in problems
        assert "unknown id (E1)" in problems
        assert "an answer was not an object" in problems

    def test_with_criterion_text_fills_the_brief_wording(self):
        out = _with_criterion_text("M", ["a", "b"], [{"id": "M2", "met": "met"}, {"id": "zzz"}, "junk"])
        assert out[0]["text"] == "b"
        assert "text" not in out[1]
        assert len(out) == 2

    def test_model_no_go_with_hard_filter_named_stays_no_go(self):
        must_haves = [{"text": "EU-based", "met": "needs_verification"}]
        decision, note = _resolve_three_state_decision(
            "NO GO", must_haves, [],
            hard_filter_failed="Job hopper: 4 roles under 1 year",
        )
        assert decision == "NO GO"
        assert note == ""  # model's own NO GO + reasoning is preserved as-is

    def test_model_no_go_with_stability_fail_stays_no_go(self):
        must_haves = [{"text": "EU-based", "met": "needs_verification"}]
        decision, note = _resolve_three_state_decision(
            "NO GO", must_haves, [], stability_failed=True,
        )
        assert decision == "NO GO"
        assert note == ""

    def test_model_go_with_stability_fail_forced_to_no_go(self):
        # Codex review (PR #144, round 1): a real hard filter must override
        # a model GO too, not just a wrongly-lenient NEEDS VERIFICATION.
        must_haves = [{"text": "5+ years Python", "met": "met"}]
        decision, note = _resolve_three_state_decision(
            "GO", must_haves, [], stability_failed=True,
        )
        assert decision == "NO GO"
        assert "STABILITY VERDICT" in note

    def test_model_go_with_hard_filter_named_forced_to_no_go(self):
        must_haves = [{"text": "5+ years Python", "met": "met"}]
        decision, note = _resolve_three_state_decision(
            "GO", must_haves, [],
            hard_filter_failed="Telecom/outsourcing background, no exception requested",
        )
        assert decision == "NO GO"
        assert "Telecom/outsourcing" in note


class _FakeUpsertBatchClient:
    """Minimal fake SupabaseClient — captures upsert_batch calls only."""

    def __init__(self):
        self.upsert_batch_calls = []

    def upsert_batch(self, table, rows, on_conflict=None):
        self.upsert_batch_calls.append({"table": table, "rows": rows, "on_conflict": on_conflict})
        return rows


class TestScreeningNotesClearedOnRescreen:
    """Codex review (PR #144, round 2, issue 2): a profile first screened as
    NEEDS VERIFICATION writes a screening_notes row. If it's later
    re-screened as GO/NO GO (no needs_verification items -> notes=None),
    the stale note must be CLEARED, not left in place. upsert_batch's
    merge-duplicates resolution only touches columns present in the JSON
    payload, so the fix must send screening_notes explicitly (as JSON
    null) rather than stripping the key like the other None-valued
    columns."""

    def test_notes_key_present_and_null_when_notes_is_none(self):
        client = _FakeUpsertBatchClient()
        db.update_profile_screening_batch(
            client,
            [{"linkedin_url": "https://www.linkedin.com/in/a", "score": 8,
              "fit_level": "Good Fit", "summary": "x", "reasoning": "y",
              "notes": None}],
        )
        row = client.upsert_batch_calls[0]["rows"][0]
        assert "screening_notes" in row
        assert row["screening_notes"] is None

    def test_notes_key_carries_the_needs_verification_text(self):
        client = _FakeUpsertBatchClient()
        db.update_profile_screening_batch(
            client,
            [{"linkedin_url": "https://www.linkedin.com/in/a", "score": 6,
              "fit_level": "Maybe", "summary": "x", "reasoning": "y",
              "notes": "Needs verification: EU-based"}],
        )
        row = client.upsert_batch_calls[0]["rows"][0]
        assert row["screening_notes"] == "Needs verification: EU-based"

    def test_other_none_columns_still_stripped(self):
        # jd_title/ai_model are None here and must NOT appear in the
        # payload -- only screening_notes gets the explicit-null treatment.
        client = _FakeUpsertBatchClient()
        db.update_profile_screening_batch(
            client,
            [{"linkedin_url": "https://www.linkedin.com/in/a", "score": 8,
              "fit_level": "Good Fit", "summary": "x", "reasoning": "y"}],
        )
        row = client.upsert_batch_calls[0]["rows"][0]
        assert "jd_title" not in row
        assert "ai_model" not in row
        assert "screening_notes" in row and row["screening_notes"] is None


class TestResultBucket:
    """_result_bucket(r) -> str. There is no more NEEDS VERIFICATION
    decision to break out separately -- new results only ever carry fit
    "Good Fit" or "Not a Fit". This just needs to keep passing through
    whatever fit_level is on the row, including a "Maybe" left over from
    an older session result from a previous build."""

    def test_ordinary_maybe_stays_maybe(self):
        r = {"fit": "Maybe", "decision": "GO"}
        assert _result_bucket(r) == "Maybe"

    def test_good_fit_passes_through(self):
        r = {"fit": "Good Fit", "decision": "GO"}
        assert _result_bucket(r) == "Good Fit"

    def test_not_a_fit_passes_through(self):
        r = {"fit": "Not a Fit", "decision": "NO GO"}
        assert _result_bucket(r) == "Not a Fit"

    def test_missing_decision_key_falls_back_to_fit(self):
        # Legacy results without a `decision` field at all.
        r = {"fit": "Maybe"}
        assert _result_bucket(r) == "Maybe"


class TestHardFilterFailureNamed:
    """_hard_filter_failure_named(hard_filter_failed) -> bool. Codex review
    (PR #144, round 2, issue 1): models sometimes fill the field with a
    "nothing failed" placeholder instead of leaving it empty as instructed
    -- those must NOT be treated as a real hard-filter failure, or a clean
    profile gets wrongly forced to NO GO."""

    @pytest.mark.parametrize("placeholder", [
        "", "  ", "none", "None", "NONE", "none.", "None of the hard filters apply",
        "na", "n/a", "N/A", "no", "No", "no.",
        "no hard filter", "No hard filter", "no hard filters",
        "no hard filter failed", "no hard filters failed", "No hard filters failed.",
        "null", "NULL", "false", "False", "not applicable", "Not Applicable", "-",
    ])
    def test_placeholders_are_not_a_failure(self, placeholder):
        assert _hard_filter_failure_named(placeholder) is False

    def test_none_value_is_not_a_failure(self):
        assert _hard_filter_failure_named(None) is False

    @pytest.mark.parametrize("real_reason", [
        "Job hopper: 4 roles under 1 year",
        "Career arc predominantly non-tech (sales/retail background)",
        "Telecom/outsourcing background, no exception requested",
        "8+ years at one company with static scope",
    ])
    def test_real_hard_filter_reasons_are_a_failure(self, real_reason):
        assert _hard_filter_failure_named(real_reason) is True

    def test_not_met_still_forces_no_go_even_with_hard_filter_set(self):
        must_haves = [
            {"text": "5+ years Python", "met": "not_met"},
            {"text": "EU-based", "met": "needs_verification"},
        ]
        decision, note = _resolve_three_state_decision(
            "GO", must_haves, [], hard_filter_failed="irrelevant",
        )
        assert decision == "NO GO"
        assert "5+ years Python" in note

    def test_all_met_no_hard_filter_stays_go(self):
        must_haves = [{"text": "5+ years Python", "met": "met"}]
        decision, note = _resolve_three_state_decision("GO", must_haves, [], score=8)
        assert decision == "GO"
        assert note == ""


class TestResultBucketHistoricNeedsVerification:
    """Codex review, PR #150 round 3: a carried-over row still marked
    NEEDS VERIFICATION (fit "Maybe") has not been re-screened under the
    no-manual-bucket policy, so it must not join the Maybe filter/outreach
    flow."""

    def test_historic_needs_verification_shows_as_not_a_fit(self):
        from dashboard import _result_bucket
        assert _result_bucket({"decision": "NEEDS VERIFICATION", "fit": "Maybe"}) == "Not a Fit"

    def test_plain_historic_maybe_still_displays_as_maybe(self):
        from dashboard import _result_bucket
        assert _result_bucket({"decision": "GO", "fit": "Maybe"}) == "Maybe"

    def test_new_results_unchanged(self):
        from dashboard import _result_bucket
        assert _result_bucket({"decision": "GO", "fit": "Good Fit"}) == "Good Fit"
        assert _result_bucket({"decision": "NO GO", "fit": "Not a Fit"}) == "Not a Fit"


class TestScreenProfileTiesAnswersToIds:
    """End to end through screen_profile with a faked model reply: the prompt
    must number the criteria (M1.. / E1..), the decision must come from the
    ids, and stored verdicts must carry the brief's own wording."""

    BRIEF = {
        "role_context": "Senior backend engineer",
        "must_haves": ["5+ years Python", "Kubernetes in production"],
        "exclusions": ["Currently at a competitor"],
        "nice_to_haves": [],
    }
    PROFILE = {
        "linkedin_url": "https://www.linkedin.com/in/someone",
        "name": "Test Person",
        "raw_crustdata": {
            "name": "Test Person",
            "current_employers": [{"employer_name": "Acme", "employee_title": "Engineer",
                                   "start_date": "2020-01-01T00:00:00"}],
        },
    }

    def _run(self, monkeypatch, reply):
        import dashboard
        seen = {}

        def fake_call(client, ai_provider, ai_model, system_prompt, user_prompt, **kw):
            seen["prompt"] = user_prompt
            return reply

        monkeypatch.setattr(dashboard, "_screening_api_call", fake_call)
        result = dashboard.screen_profile(
            dict(self.PROFILE), "", object(), screening_brief=self.BRIEF,
        )
        return result, seen["prompt"]

    def test_prompt_numbers_every_criterion_and_go_keeps_brief_wording(self, monkeypatch):
        reply = {
            "must_haves": [{"id": "M2", "met": "met"}, {"id": "M1", "met": "met"}],
            "exclusions": [{"id": "E1", "matched": False}],
            "decision": "GO", "score": 9, "reasoning": "ok",
        }
        result, prompt = self._run(monkeypatch, reply)
        assert "M1. 5+ years Python" in prompt
        assert "M2. Kubernetes in production" in prompt
        assert "E1. Currently at a competitor" in prompt
        assert result["decision"] == "GO"
        assert result["fit"] == "Good Fit"
        texts = {v["id"]: v["text"] for v in result["must_have_verdicts"]}
        assert texts == {"M1": "5+ years Python", "M2": "Kubernetes in production"}
        assert result["exclusion_verdicts"][0]["text"] == "Currently at a competitor"

    def test_reply_without_ids_is_no_go_and_says_why(self, monkeypatch):
        # The model ignored the id instruction and echoed text instead.
        reply = {
            "must_haves": [{"text": "5+ years Python", "met": "met"},
                           {"text": "Kubernetes in production", "met": "met"}],
            "exclusions": [{"text": "Currently at a competitor", "matched": False}],
            "decision": "GO", "score": 9, "reasoning": "looks great",
        }
        result, _ = self._run(monkeypatch, reply)
        assert result["decision"] == "NO GO"
        assert result["fit"] == "Not a Fit"
        assert "Screening incomplete" in result["summary"]
        assert "looks great" in result["summary"]

    def test_missing_answer_is_explained_even_when_model_said_no_go(self, monkeypatch):
        reply = {
            "must_haves": [{"id": "M1", "met": "met"}],
            "exclusions": [{"id": "E1", "matched": False}],
            "decision": "NO GO", "score": 3, "reasoning": "weak",
        }
        result, _ = self._run(monkeypatch, reply)
        assert result["decision"] == "NO GO"
        assert "Must-have not judged: Kubernetes in production" in result["summary"]

    def test_tenure_override_to_no_go_clears_the_verify_note(self, monkeypatch):
        # Codex review, PR #150 round 5: a GO with a "Verify in call" note
        # that a later tenure check turns into NO GO must not keep the note.
        import tenure_constraint_validator as tcv

        def force_no_go(result, *_a, **_k):
            result = dict(result)
            result["decision"] = "NO GO"
            result["fit"] = "Not a Fit"
            return result

        monkeypatch.setattr(tcv, "enforce_tenure_constraint", force_no_go)
        reply = {
            "must_haves": [{"id": "M1", "met": "met"}, {"id": "M2", "met": "needs_verification"}],
            "exclusions": [{"id": "E1", "matched": False}],
            "decision": "GO", "score": 8, "reasoning": "ok",
        }
        result, _ = self._run(monkeypatch, reply)
        assert result["decision"] == "NO GO"
        assert result["verify_note"] is None

    def test_go_keeps_its_verify_note_when_nothing_overrides_it(self, monkeypatch):
        reply = {
            "must_haves": [{"id": "M1", "met": "met"}, {"id": "M2", "met": "needs_verification"}],
            "exclusions": [{"id": "E1", "matched": False}],
            "decision": "GO", "score": 8, "reasoning": "ok",
        }
        result, _ = self._run(monkeypatch, reply)
        assert result["decision"] == "GO"
        assert result["verify_note"] == "Verify in call: Kubernetes in production"
