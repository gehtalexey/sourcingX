"""
Tests for the deterministic tenure-constraint validator
(``tenure_constraint_validator.py``).

Background — Issue 7, Codex review on PR #16:
The first fix only strengthened the AI screening prompt. That is not
deterministic enforcement — the model can still ignore or trade off the
rule. These tests lock in a Python-side override that runs AFTER the
model returns: it parses the recruiter's tenure constraint, looks at
the candidate's current-company tenure, and forces "Not a Fit" / NO GO
when the candidate is below the threshold.

Three test groups:

1. Parser unit tests — every phrase pattern (English + Hebrew) yields
   the right month count.
2. Override behavior — strong model result + violating tenure → result
   is overridden to score <= 2, fit="Not a Fit", decision="NO GO".
3. Pass-through — when there's no tenure constraint in the JD (or the
   candidate satisfies it, or we have no tenure data), the validator
   MUST NOT touch the result.

These tests never call an LLM and never hit the network.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from tenure_constraint_validator import (
    parse_tenure_constraint_months,
    current_company_tenure_months,
    enforce_tenure_constraint,
    TENURE_OVERRIDE_SCORE,
)


# ---------------------------------------------------------------------------
# 1. Parser
# ---------------------------------------------------------------------------

class TestParseTenureConstraint:
    """parse_tenure_constraint_months pulls the strictest minimum tenure
    (in months) out of free-form recruiter text. English + Hebrew."""

    @pytest.mark.parametrize("text, expected_months", [
        # Codex's three pattern families.
        ("Looking for backend devs. Minimum 1 year at current company.", 12),
        ("Backend role. No one under 1 year at current company.", 12),
        ("Must have been at current employer for 18 months.", 18),
        # 'at least N years/months at company / tenure'
        ("Senior SWE. At least 2 years at current company please.", 24),
        ("Need stable candidates — at least 6 months at company.", 6),
        ("At least 18 months tenure required.", 18),
        # Plural / abbreviated units.
        ("minimum 2 years at company", 24),
        ("no one under 6 months at current company", 6),
        ("Minimum 3 yrs at the current employer.", 36),
        # 'nobody under'
        ("Nobody under 12 months at company.", 12),
        # 'must have been at company for X year(s)'
        ("Must have been at the current company for 2 years.", 24),
    ])
    def test_english_phrasings_parse(self, text, expected_months):
        assert parse_tenure_constraint_months(text) == expected_months

    @pytest.mark.parametrize("text, expected_months", [
        # "at least 1 year at the company"
        ("לפחות 1 שנה בחברה", 12),
        # "at least 12 months at the company"
        ("לפחות 12 חודשים בחברה", 12),
        # "minimum 2 years at the current company"
        ("מינימום 2 שנים בחברה הנוכחית", 24),
        # "at least 6 months at the company"
        ("לפחות 6 חודשים בחברה", 6),
    ])
    def test_hebrew_phrasings_parse(self, text, expected_months):
        # Numeric phrasings parse; word-forms like "שנתיים" (= "two years"
        # as a single word, no digit) are intentionally NOT supported —
        # recruiters writing constraints type numbers.
        assert parse_tenure_constraint_months(text) == expected_months

    def test_no_constraint_returns_none(self):
        """Plain text with no tenure phrase → None (no-op signal)."""
        assert parse_tenure_constraint_months(
            "Backend engineer with Python, AWS, Kubernetes experience."
        ) is None
        assert parse_tenure_constraint_months("") is None
        assert parse_tenure_constraint_months(None) is None

    def test_strictest_threshold_wins(self):
        """When multiple tenure phrases match, the LARGEST threshold (most
        restrictive) is returned."""
        text = (
            "At least 6 months at company. "
            "Actually, minimum 1 year at current company. "
            "Must have been at current employer for 18 months."
        )
        # 6m vs 12m vs 18m → 18m wins.
        assert parse_tenure_constraint_months(text) == 18

    @pytest.mark.parametrize("text, expected_months", [
        # Codex's three exact phrasings from the PR #16 re-review —
        # Chen's actual wording was natural language, not numeric.
        ("no one under one year at current company", 12),
        ("minimum one year at current company", 12),
        ("must have been at current employer for two years", 24),
        # More small-number coverage.
        ("at least three years at current company", 36),
        ("minimum six months at company", 6),
        ("no one under twelve months at current company", 12),
        # Case-insensitive (recruiters capitalize unpredictably).
        ("Minimum One Year at Current Company", 12),
    ])
    def test_english_number_words_parse(self, text, expected_months):
        """Parser must accept small English number words (one..twelve)
        and convert them to digits before regex matching."""
        assert parse_tenure_constraint_months(text) == expected_months

    def test_number_word_substrings_do_not_match(self):
        """Word-boundary guarantee: 'onerous', 'sixteen', etc. must NOT be
        misread as the number 'one' / 'six'. Even if they did, the lack of
        a tenure phrase around them means parse should still return None."""
        assert parse_tenure_constraint_months(
            "The role is onerous and requires self-direction."
        ) is None
        assert parse_tenure_constraint_months(
            "The team is sixteen people strong."
        ) is None


# ---------------------------------------------------------------------------
# 2. current_company_tenure_months
# ---------------------------------------------------------------------------

class TestCurrentCompanyTenureMonths:
    """Computes months at the candidate's current company from raw
    Crustdata data. Uses the earliest start_date among current employers
    at the same company (so internal promotions don't reset tenure)."""

    def test_basic_single_current_employer(self):
        # Start 6 months ago.
        today = datetime.now(timezone.utc)
        start_year = today.year if today.month > 6 else today.year - 1
        start_month = today.month - 6 if today.month > 6 else today.month + 6
        start_date = f"{start_year}-{start_month:02d}-01"
        raw = {
            "current_employers": [
                {
                    "employer_name": "Acme Corp",
                    "start_date": start_date,
                    "end_date": None,
                }
            ]
        }
        months = current_company_tenure_months(raw)
        # ~6 months (allow off-by-one for month-edge timing).
        assert months is not None
        assert 5 <= months <= 7

    def test_internal_promotion_does_not_reset_tenure(self):
        """Two current positions at the SAME company → tenure is from the
        EARLIEST start_date."""
        raw = {
            "current_employers": [
                {
                    "employer_name": "Acme Corp",
                    "employee_title": "Senior SWE",
                    "start_date": "2025-01-01",  # most-recent role
                    "end_date": None,
                },
                {
                    "employer_name": "Acme Corp",
                    "employee_title": "SWE",
                    "start_date": "2023-01-01",  # original join date
                    "end_date": None,
                },
            ]
        }
        months = current_company_tenure_months(raw)
        # From 2023-01-01 to today (May 2026) ~= 40 months. Just assert it
        # picked the earlier date (>= 24 months).
        assert months is not None
        assert months >= 24

    def test_no_current_employer_returns_none(self):
        assert current_company_tenure_months({"current_employers": []}) is None
        assert current_company_tenure_months({}) is None
        assert current_company_tenure_months(None) is None

    def test_missing_start_date_returns_none(self):
        raw = {
            "current_employers": [
                {"employer_name": "Acme Corp", "start_date": None}
            ]
        }
        assert current_company_tenure_months(raw) is None


# ---------------------------------------------------------------------------
# 3. enforce_tenure_constraint — the override
# ---------------------------------------------------------------------------

class TestEnforceTenureConstraint:
    """The deterministic override: strong model result + violating tenure
    → result is forced to 'Not a Fit' (score <= TENURE_OVERRIDE_SCORE)."""

    def _result_strong_fit_legacy(self):
        return {
            "score": 8,
            "fit": "Good Fit",
            "summary": "Strong Python backend candidate.",
        }

    def _result_strong_fit_unified(self):
        return {
            "score": 8,
            "fit": "Good Fit",
            "summary": "Strong Python backend candidate.",
            "decision": "GO",
            "reasoning": "Strong Python backend candidate.",
        }

    def _result_strong_fit_structured(self):
        return {
            "name": "Test Candidate",
            "score": 8,
            "category": "Good Fit",
            "reasoning": "Strong Python backend candidate.",
        }

    def _candidate_with_n_months(self, months: int):
        """Build raw data where current_employers[0].start_date is N months
        before today."""
        today = datetime.now(timezone.utc)
        total_months = today.year * 12 + (today.month - 1) - months
        start_year = total_months // 12
        start_month = total_months % 12 + 1
        return {
            "current_employers": [
                {
                    "employer_name": "Acme Corp",
                    "start_date": f"{start_year}-{start_month:02d}-15",
                    "end_date": None,
                }
            ]
        }

    def test_strong_candidate_under_tenure_is_overridden_to_not_a_fit(self):
        """The headline case Codex called out."""
        result = self._result_strong_fit_legacy()
        jd_text = "Backend role. Minimum 1 year at current company. Python, AWS."
        raw = self._candidate_with_n_months(6)  # 6 months — violates 12mo rule

        out = enforce_tenure_constraint(result, jd_text, raw)

        assert out["fit"] == "Not a Fit"
        assert out["score"] <= TENURE_OVERRIDE_SCORE
        assert "Hard constraint violated" in out["summary"]
        assert "12 months" in out["summary"]  # threshold cited
        # Override metadata exposed for debugging.
        assert out["tenure_override"]["threshold_months"] == 12
        assert 5 <= out["tenure_override"]["actual_months"] <= 7

    def test_override_works_on_unified_policy_result(self):
        """Same override, GO → NO GO on the unified-policy shape."""
        result = self._result_strong_fit_unified()
        jd_text = "No one under 1 year at current company."
        raw = self._candidate_with_n_months(4)

        out = enforce_tenure_constraint(result, jd_text, raw)

        assert out["decision"] == "NO GO"
        assert out["fit"] == "Not a Fit"
        assert out["score"] <= TENURE_OVERRIDE_SCORE
        assert "Hard constraint violated" in out["reasoning"]

    def test_override_works_on_structured_result(self):
        """Same override on the structured_prompt_builder shape."""
        result = self._result_strong_fit_structured()
        must_have_text = "Minimum 1 year at current company"
        raw = self._candidate_with_n_months(3)

        out = enforce_tenure_constraint(result, must_have_text, raw)

        assert out["category"] == "No Fit"
        assert out["score"] <= TENURE_OVERRIDE_SCORE
        assert "Hard constraint violated" in out["reasoning"]

    def test_pass_through_when_no_tenure_constraint_in_jd(self):
        """No tenure phrase in the JD → validator MUST NOT touch the result."""
        result = self._result_strong_fit_legacy()
        original = dict(result)
        jd_text = "Backend engineer with Python, AWS, Kubernetes. Senior level."
        raw = self._candidate_with_n_months(2)  # very short — but no rule

        out = enforce_tenure_constraint(result, jd_text, raw)

        assert out == original
        assert "tenure_override" not in out

    def test_pass_through_when_candidate_satisfies_threshold(self):
        """JD has a tenure rule and the candidate clears it → no change."""
        result = self._result_strong_fit_legacy()
        original = dict(result)
        jd_text = "Minimum 1 year at current company. Python."
        raw = self._candidate_with_n_months(36)  # 3 years — passes

        out = enforce_tenure_constraint(result, jd_text, raw)

        assert out == original
        assert "tenure_override" not in out

    def test_pass_through_when_no_current_employer_data(self):
        """Constraint in JD but candidate has no start_date → CANNOT prove
        a violation → do not override (avoid false negatives)."""
        result = self._result_strong_fit_legacy()
        original = dict(result)
        jd_text = "Minimum 1 year at current company."
        raw = {"current_employers": []}

        out = enforce_tenure_constraint(result, jd_text, raw)

        assert out == original
        assert "tenure_override" not in out
