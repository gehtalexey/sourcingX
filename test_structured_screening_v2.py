"""Regression tests for the structured (per-criterion) screening path.

Pins the guarantees of the v2 architecture:
  1. The structured system prompt keeps the full senior-recruiter rubric and
     uses the per-criterion output schema (a verdict on each must-have and
     each exclusion) — not the old {decision, score, reasoning} blob.
  2. Nice-to-haves are NOT in the screening prompt at all — they go to a
     separate bonus pass — so a nice-to-have can never cause a NO GO.
  3. screen_profile / screen_profiles_batch accept a ``screening_brief``.

These are prompt-content / signature tests — they never call an LLM and never
hit the network.
"""

import inspect

import pytest

from screening_policy import (
    get_structured_system_prompt,
    build_structured_user_prompt,
    build_nice_to_have_prompt,
    NICE_TO_HAVE_SYSTEM_PROMPT,
)


class TestStructuredSystemPrompt:
    """The structured screening prompt keeps the rubric, swaps the output."""

    def test_keeps_core_rubric(self):
        p = get_structured_system_prompt()
        # The senior-recruiter rubric is unchanged — spot-check load-bearing parts.
        for phrase in ["senior technical recruiter", "User-Stated Hard Constraints",
                       "INDUSTRY EXPERIENCE", "STABILITY VERDICT"]:
            assert phrase in p, f"structured prompt lost rubric phrase '{phrase}'"

    def test_preserves_military_experience_rule(self):
        # The military / experience-cap rule must survive the output swap.
        p = get_structured_system_prompt()
        assert "never TOTAL CAREER SPAN" in p

    def test_uses_per_criterion_output_schema(self):
        p = get_structured_system_prompt()
        # Per-criterion output: a verdict on each must-have and exclusion.
        assert '"must_haves"' in p and '"met"' in p
        assert '"exclusions"' in p and '"matched"' in p
        assert "Give an explicit verdict on every must-have and every exclusion" in p

    def test_decision_definition_is_present(self):
        p = get_structured_system_prompt()
        assert "GO only when every must-have is met and no exclusion matched" in p

    def test_no_nice_to_haves_in_screening_prompt(self):
        # Fix 2: nice-to-haves are handled by a SEPARATE pass. They must not
        # appear in the decision-making prompt at all.
        p = get_structured_system_prompt()
        assert "NICE-TO-HAVE" not in p.upper()


class TestStructuredUserPrompt:
    """The structured user prompt carries must-haves + exclusions only."""

    def _build(self):
        return build_structured_user_prompt(
            role_context="Senior Backend Engineer at a fintech startup",
            must_haves=["5+ years backend experience", "Python or Node.js"],
            exclusions=["No pure managers"],
            durations_text="ROLE DURATIONS: ...",
            trimmed_raw={"name": "Test Candidate"},
        )

    def test_includes_role_must_haves_exclusions(self):
        up = self._build()
        assert "Senior Backend Engineer at a fintech startup" in up
        assert "5+ years backend experience" in up
        assert "Python or Node.js" in up
        assert "No pure managers" in up
        assert "Must-Haves" in up and "Exclusions" in up

    def test_excludes_nice_to_haves_by_construction(self):
        # build_structured_user_prompt has no nice_to_haves parameter — a
        # nice-to-have literally cannot reach the decision call.
        sig = inspect.signature(build_structured_user_prompt)
        assert "nice_to_haves" not in sig.parameters

    def test_profile_json_included(self):
        assert "Test Candidate" in self._build()


class TestNiceToHavePass:
    """The separate nice-to-have bonus pass."""

    def test_system_prompt_states_it_never_decides(self):
        assert "NEVER affects any hire decision" in NICE_TO_HAVE_SYSTEM_PROMPT

    def test_user_prompt_lists_nice_to_haves(self):
        up = build_nice_to_have_prompt(
            ["Fintech background", "Open-source contributions"],
            {"name": "Test Candidate"},
        )
        assert "Fintech background" in up
        assert "Open-source contributions" in up
        assert "Nice-to-Haves" in up
        assert "Test Candidate" in up


class TestScreenProfileAcceptsBrief:
    """screen_profile / screen_profiles_batch take a structured brief."""

    def test_screen_profile_has_screening_brief_param(self):
        import dashboard
        assert "screening_brief" in inspect.signature(dashboard.screen_profile).parameters

    def test_screen_profiles_batch_has_screening_brief_param(self):
        import dashboard
        assert "screening_brief" in inspect.signature(dashboard.screen_profiles_batch).parameters
