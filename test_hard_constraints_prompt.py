"""
Regression tests for Issue 7: AI Screen ignored user-stated hard constraints
(e.g. "no one under 1 year at current company") and still returned violating
profiles as relevant.

These tests verify that the screening prompts produced by:
  - screening_policy.get_system_prompt()           (unified path — main default)
  - screening_prompt_builder.STRUCTURED_SYSTEM_PROMPT (structured path)

both contain explicit instructions to treat user-stated conditions as HARD
CONSTRAINTS (binary filters), not soft preferences, and to return NO GO /
No Fit when a candidate violates one.

These are prompt-content tests — they never call an LLM and never hit the
network. They are intended to lock in the wording so future edits don't
silently weaken hard-constraint enforcement again.
"""

import pytest

from screening_policy import get_system_prompt, build_user_prompt
from screening_prompt_builder import STRUCTURED_SYSTEM_PROMPT
from screening_prompt_builder import build_user_prompt as build_structured_user_prompt


# ---------------------------------------------------------------------------
# Unified screening_policy.py
# ---------------------------------------------------------------------------

class TestUnifiedPolicyHardConstraints:
    """The unified screening policy MUST mention user-stated hard constraints
    and instruct the model to treat them as binary filters."""

    def test_policy_has_hard_constraints_section(self):
        system_prompt = get_system_prompt()
        assert "User-Stated Hard Constraints" in system_prompt, (
            "Unified policy is missing the 'User-Stated Hard Constraints' "
            "section that was added to fix Issue 7."
        )

    @pytest.mark.parametrize("phrase", [
        "HARD CONSTRAINT",
        "binary filter",
        "NO GO",
        "Do NOT soft-score",
        "Name the violated constraint",
    ])
    def test_policy_uses_strict_enforcement_language(self, phrase):
        system_prompt = get_system_prompt()
        assert phrase in system_prompt, (
            f"Unified policy should contain '{phrase}' to enforce user-stated "
            f"hard constraints (Issue 7 regression)."
        )

    @pytest.mark.parametrize("example", [
        "minimum 1 year at current company",
        "at least 2 years at company",
        "no career switchers",
        "must currently work at",
    ])
    def test_policy_calls_out_concrete_constraint_examples(self, example):
        """The policy must list concrete tenure / current-employer / category
        examples so the model recognizes Chen-style phrasings."""
        system_prompt = get_system_prompt()
        assert example.lower() in system_prompt.lower(), (
            f"Unified policy should include the example '{example}' so the "
            f"model treats it as a hard constraint."
        )

    def test_user_prompt_still_includes_recruiter_request(self):
        """Sanity check: build_user_prompt still surfaces the recruiter's
        request (which is where the hard constraint actually lives)."""
        user_request = "Backend engineer. Must have minimum 1 year at current company."
        user_prompt = build_user_prompt(
            user_request=user_request,
            durations_text="",
            trimmed_raw={"name": "Test"},
        )
        assert "minimum 1 year at current company" in user_prompt
        assert "Recruiter Request" in user_prompt


# ---------------------------------------------------------------------------
# screening_prompt_builder.py (structured path)
# ---------------------------------------------------------------------------

class TestStructuredPromptHardConstraints:
    """The structured prompt builder MUST also enforce user-stated hard
    constraints — Chen's flow may route through either path."""

    def test_structured_prompt_has_hard_constraints_section(self):
        assert "User-Stated Hard Constraints" in STRUCTURED_SYSTEM_PROMPT, (
            "Structured system prompt is missing the 'User-Stated Hard "
            "Constraints' section (Issue 7)."
        )

    @pytest.mark.parametrize("phrase", [
        "HARD CONSTRAINT",
        "binary filter",
        "No Fit",
        "Do NOT soft-score",
    ])
    def test_structured_prompt_uses_strict_enforcement_language(self, phrase):
        assert phrase in STRUCTURED_SYSTEM_PROMPT, (
            f"Structured system prompt should contain '{phrase}' "
            f"(Issue 7 regression)."
        )

    def test_structured_user_prompt_still_includes_must_haves(self):
        """Sanity check: must-have requirements still appear in the user
        prompt verbatim (that's where the user-stated constraint lives)."""
        user_prompt = build_structured_user_prompt(
            must_haves=["Minimum 1 year at current company"],
            nice_to_haves=[],
            reject_ifs=[],
            profile_data={"name": "Test"},
        )
        assert "Minimum 1 year at current company" in user_prompt
