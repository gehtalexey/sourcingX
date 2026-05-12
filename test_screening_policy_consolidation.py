"""
Regression tests for the screening-path consolidation.

These tests pin three guarantees:
  1. ``screen_profile`` no longer accepts ``role_prompt`` — it was removed
     when the dashboard collapsed its three prompt branches into one path.
  2. ``screen_profile`` called WITHOUT ``user_request`` still routes through
     the unified ``screening_policy.SCREENING_POLICY`` — the system prompt
     sent to OpenAI must contain the policy signature, not the deleted
     legacy SCREENING_PROMPT constant.
  3. The policy text still carries the lowered "current tenure < 1 year"
     wording (the hard filter that was relaxed from 2 years to 1 year in
     this PR).

These are prompt-content tests — they never call an LLM and never hit the
network. They lock in the wording so future edits don't silently regress.
"""

import inspect
import pytest

from screening_policy import SCREENING_POLICY, get_system_prompt


class TestScreenProfileSignature:
    """The legacy ``role_prompt`` parameter MUST be gone."""

    def test_role_prompt_not_in_signature(self):
        import dashboard
        sig = inspect.signature(dashboard.screen_profile)
        assert "role_prompt" not in sig.parameters, (
            "screen_profile must no longer accept role_prompt — the "
            "consolidation deleted the legacy role-specific prompt branch."
        )

    def test_role_prompt_kwarg_raises_typeerror(self,
                                                strong_backend_profile,
                                                backend_job_description,
                                                mock_openai_client):
        import dashboard
        with pytest.raises(TypeError):
            dashboard.screen_profile(
                strong_backend_profile,
                backend_job_description,
                mock_openai_client,
                role_prompt="You are a specialized DevOps recruiter.",
            )


class TestUnifiedPolicyAlwaysUsed:
    """Calling screen_profile without user_request must still use the
    unified screening_policy rubric — there is no fallback path."""

    POLICY_SIGNATURE = "User-Stated Hard Constraints"

    def test_no_user_request_still_uses_policy(self,
                                               strong_backend_profile,
                                               backend_job_description,
                                               mock_openai_client):
        import dashboard
        # No user_request passed — legacy callers used to fall through to
        # SCREENING_PROMPT. They now must land in the unified policy path.
        dashboard.screen_profile(
            strong_backend_profile,
            backend_job_description,
            mock_openai_client,
        )

        system_prompt = mock_openai_client.get_system_prompt()
        assert self.POLICY_SIGNATURE in system_prompt, (
            "When user_request is omitted, screen_profile must still send "
            "the unified SCREENING_POLICY as the system prompt."
        )

    def test_explicit_none_user_request_still_uses_policy(self,
                                                          strong_backend_profile,
                                                          backend_job_description,
                                                          mock_openai_client):
        import dashboard
        dashboard.screen_profile(
            strong_backend_profile,
            backend_job_description,
            mock_openai_client,
            user_request=None,
        )

        system_prompt = mock_openai_client.get_system_prompt()
        assert self.POLICY_SIGNATURE in system_prompt


class TestTenureRuleLowered:
    """The hard-filter tenure threshold was lowered from 2 years to 1 year."""

    def test_policy_mentions_one_year_current_tenure(self):
        system_prompt = get_system_prompt()
        assert "Current tenure at current COMPANY under 1 year" in system_prompt, (
            "The Hard Filters section must say 'under 1 year' for current "
            "tenure — lowered from 2 years in the consolidation PR."
        )

    def test_policy_no_longer_says_under_2_years_for_current_tenure(self):
        # The "3+ short-stint COMPANIES (each <2 years total)" line is a
        # different rule and stays at 2 years. The CURRENT-tenure rule must
        # not still say "under 2 years".
        assert "Current tenure at current COMPANY under 2 years" not in SCREENING_POLICY

    def test_short_stint_rule_still_at_two_years(self):
        # Sanity check: the short-stint rule is per-company and is unchanged.
        assert "each <2 years total" in SCREENING_POLICY

    def test_override_example_uses_six_month_default(self):
        # The "user-stated constraints override the generic default" example
        # was rewritten to use 'min 6 months' overriding the new 1-year
        # generic default (since the old example used the 2-year default
        # that no longer exists).
        system_prompt = get_system_prompt()
        assert "min 6 months" in system_prompt
        assert "generic 1-year default" in system_prompt
