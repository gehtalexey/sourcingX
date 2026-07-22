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

These tests call the pure helper directly. They never call an LLM and
never hit the network.
"""

from __future__ import annotations

import pytest

from dashboard import _verdicts_force_no_go


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
        ("true", False),
        ("True", False),
        ("false", True),
        ("False", True),
    ])
    def test_met_string_truthiness_is_robust(self, met_value, expect_force):
        """The model sometimes returns 'met' as the literal string "true"/
        "false" instead of a JSON boolean. The guard must treat both the
        same as the native boolean."""
        must_haves = [{"text": "Some requirement", "met": met_value}]
        force, _reason = _verdicts_force_no_go(must_haves, [])
        assert force is expect_force

    @pytest.mark.parametrize("matched_value, expect_force", [
        (True, True),
        (False, False),
        ("true", True),
        ("True", True),
        ("false", False),
        ("False", False),
    ])
    def test_matched_string_truthiness_is_robust(self, matched_value, expect_force):
        """Same string-vs-bool robustness for exclusion 'matched' values."""
        exclusions = [{"text": "Some exclusion", "matched": matched_value}]
        force, _reason = _verdicts_force_no_go([], exclusions)
        assert force is expect_force

    def test_both_failed_must_have_and_matched_exclusion_combine_in_reason(self):
        """When both a failed must-have and a matched exclusion are
        present, the reason should mention both."""
        must_haves = [{"text": "Must know Kubernetes", "met": False}]
        exclusions = [{"text": "Based outside Israel", "matched": True}]
        force, reason = _verdicts_force_no_go(must_haves, exclusions)
        assert force is True
        assert "Must know Kubernetes" in reason
        assert "Based outside Israel" in reason
