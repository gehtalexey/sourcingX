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

from dashboard import (
    _tri,
    _verdicts_force_no_go,
    _must_have_verdict_state,
    _verdicts_needs_verification,
    _decision_to_fit_label,
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

    def test_no_not_met_one_needs_verification_yields_needs_verification_decision(self):
        must_haves = [
            {"text": "5+ years Python", "met": "met"},
            {"text": "Located in Europe", "met": "needs_verification"},
        ]
        force_no_go, _ = _verdicts_force_no_go(must_haves, [])
        needs_verif, items = _verdicts_needs_verification(must_haves)
        assert force_no_go is False
        assert needs_verif is True
        assert items == ["Located in Europe"]
        # The fit label a NEEDS VERIFICATION decision maps to is "Maybe" —
        # screening_fit_level keeps its existing three values; the caller
        # tells needs-verification apart via the separate `decision` field
        # and the `notes`/screening_notes text ("Needs verification: ...").
        assert _decision_to_fit_label("NEEDS VERIFICATION", 8) == "Maybe"
        notes = "Needs verification: " + "; ".join(items)
        assert notes == "Needs verification: Located in Europe"

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
