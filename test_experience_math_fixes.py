"""Regression tests for three experience/duration math correctness fixes.

Background: `dashboard.compute_role_durations()` pre-calculates per-role and
per-company tenure that the AI screener is told to treat as ground truth
(never recalculate). Three bugs in that math could silently mislead the
model:

1. A blank/"Present"/"Current" end_date was NOT treated as the current role
   (only `end_date is None` was), so a past role with a blank end_date got
   `end = today` (inflated duration) while still being labelled non-current.
   Fixed by `_role_end()`, used consistently across the three loops in
   `compute_role_durations` that interpret end_date.

2. Company-level tenure was `max_end - min_start` across all of a company's
   roles, which BRIDGES rehire gaps: two short stints at the same company
   years apart got counted as one long tenure spanning the gap. Fixed by
   merging (start, end) intervals per company and summing the merged
   durations — gaps excluded, but adjacent/overlapping roles (internal
   promotions) still merge into one continuous stint. The merge helper
   (`tenure_constraint_validator.merge_company_intervals_months`) is shared
   between `compute_role_durations` (informational summary for the AI) and
   the new deterministic max-tenure validator below, so both agree.

3. The tenure-constraint validator only enforced a MINIMUM tenure at the
   current company (Issue 7, Chen). It had no mirror for recruiters who
   want candidates who have NOT stayed unusually long at one company
   ("no more than N years at one company"). Added
   `parse_max_tenure_constraint_months`, `longest_company_tenure_months`,
   and `enforce_max_tenure_constraint` as the maximum-tenure counterpart to
   the existing `parse_tenure_constraint_months` /
   `current_company_tenure_months` / `enforce_tenure_constraint`.

These tests never call an LLM and never hit the network.
"""

import pytest

import dashboard
from tenure_constraint_validator import (
    parse_max_tenure_constraint_months,
    longest_company_tenure_months,
    enforce_max_tenure_constraint,
    TENURE_OVERRIDE_SCORE,
)


# ---------------------------------------------------------------------------
# (a) Rehire gap — must not bridge into one long tenure
# ---------------------------------------------------------------------------

def test_rehire_gap_sums_stints_not_gap_spanning_span():
    """Two non-adjacent stints at the same company (Jan-Jun 2018, then
    Jan-Jun 2024) must be counted as ~10 months total worked at that
    company, NOT ~6 years (the gap-spanning span from first start to last
    end)."""
    raw = {
        'past_employers': [
            {'employer_name': 'Acme', 'employee_title': 'Engineer',
             'start_date': '2018-01-01', 'end_date': '2018-06-01'},
            {'employer_name': 'Acme', 'employee_title': 'Senior Engineer',
             'start_date': '2024-01-01', 'end_date': '2024-06-01'},
        ],
        'current_employers': [],
    }
    text = dashboard.compute_role_durations(raw)

    # Each stint is individually reported as 5 months.
    assert 'Engineer at Acme: Jan 2018 - Jun 2018 = 5m' in text
    assert 'Senior Engineer at Acme: Jan 2024 - Jun 2024 = 5m' in text

    # Company-level summary must show the summed worked time (10 months,
    # a short-stint company), never a multi-year bridged tenure.
    assert 'Acme (10m)' in text
    assert '6y' not in text

    # tenure_constraint_validator agrees.
    assert longest_company_tenure_months(raw) == 10


def test_rehire_gap_two_companies_longest_is_not_bridged():
    """longest_company_tenure_months across companies must reflect the
    longest MERGED tenure, not a gap-bridged span."""
    raw = {
        'past_employers': [
            {'employer_name': 'Acme', 'employee_title': 'Engineer',
             'start_date': '2018-01-01', 'end_date': '2018-06-01'},
            {'employer_name': 'Acme', 'employee_title': 'Senior Engineer',
             'start_date': '2024-01-01', 'end_date': '2024-06-01'},
            {'employer_name': 'Globex', 'employee_title': 'Engineer',
             'start_date': '2019-01-01', 'end_date': '2020-01-01'},
        ],
        'current_employers': [],
    }
    # Acme = 10 months (merged, gap excluded). Globex = 12 months.
    # Longest should be Globex's 12, not a bridged Acme span of ~78 months.
    assert longest_company_tenure_months(raw) == 12


# ---------------------------------------------------------------------------
# (b) Internal promotion — adjacent roles at one company stay continuous
# ---------------------------------------------------------------------------

def test_internal_promotion_stays_one_continuous_tenure():
    """Two back-to-back roles at the same company (promotion, no gap) must
    still merge into one continuous tenure, not two short stints."""
    raw = {
        'past_employers': [
            {'employer_name': 'Beta', 'employee_title': 'Engineer',
             'start_date': '2018-01-01', 'end_date': '2020-01-01'},
            {'employer_name': 'Beta', 'employee_title': 'Senior Engineer',
             'start_date': '2020-01-01', 'end_date': '2022-01-01'},
        ],
        'current_employers': [],
    }
    text = dashboard.compute_role_durations(raw)

    # 4 years continuous, not flagged as a short-stint company.
    assert 'Beta (4y 0m)' not in text  # not listed under short-stint (>=12mo)
    assert 'Short-stint companies (<12 months' in text
    assert ': 0' in text.split('Short-stint companies')[1].splitlines()[0]

    assert longest_company_tenure_months(raw) == 48


def test_internal_promotion_with_small_gap_still_merges():
    """A few months' gap between two roles at the same company (e.g. data
    entry slop, or a short break covered by the 3-month merge tolerance)
    still merges into one continuous tenure."""
    raw = {
        'past_employers': [
            {'employer_name': 'Beta', 'employee_title': 'Engineer',
             'start_date': '2018-01-01', 'end_date': '2020-01-01'},
            {'employer_name': 'Beta', 'employee_title': 'Senior Engineer',
             'start_date': '2020-03-01', 'end_date': '2022-01-01'},
        ],
        'current_employers': [],
    }
    # Gap is 2 months (<=3) so intervals merge: total = Jan18->Jan22 minus
    # nothing skipped = 48 months (the merge extends cur_end to the later
    # role's end, gap itself isn't subtracted, matching "contiguous" intent).
    assert longest_company_tenure_months(raw) == 48


# ---------------------------------------------------------------------------
# (c) Blank end_date treated as current
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("end_date_value", ['', 'Present', 'Current', None])
def test_blank_or_present_end_date_treated_as_current(end_date_value):
    """A blank string, "Present"/"Current" (any case), or a real None must
    all be treated as CURRENT — running to today — not silently inflated
    to today while still being labelled as a past role."""
    raw = {
        'past_employers': [
            {'employer_name': 'OldCo', 'employee_title': 'Engineer',
             'start_date': '2015-01-01', 'end_date': end_date_value},
        ],
        'current_employers': [],
    }
    text = dashboard.compute_role_durations(raw)
    assert 'CURRENT ROLE' in text
    assert 'Current company: OldCo' in text


def test_unparseable_non_current_end_date_is_unknown_not_inflated():
    """A non-empty but garbage end_date on a role that is clearly NOT
    current (has a real end_date value, just not a parseable one) must not
    be inflated to run until today. It should show 0 months / unknown, and
    must not be mislabeled CURRENT ROLE."""
    raw = {
        'past_employers': [
            {'employer_name': 'OldCo', 'employee_title': 'Engineer',
             'start_date': '2015-01-01', 'end_date': 'not-a-real-date'},
        ],
        'current_employers': [],
    }
    text = dashboard.compute_role_durations(raw)
    assert 'CURRENT ROLE' not in text
    assert 'Engineer at OldCo: Jan 2015 - ? = 0m' in text
    # Must not have been inflated to a decade-plus tenure.
    assert 'Current company: none found' in text


# ---------------------------------------------------------------------------
# (d) parse_max_tenure_constraint_months on realistic phrasings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected_months", [
    ("Looking for backend devs. No more than 5 years at one company.", 60),
    ("Senior role. At most 3 years at a company please.", 36),
    ("Need someone who hasn't stayed too long — max 4 years at current company.", 48),
    ("Prefer candidates with less than 2 years at one company.", 24),
    ("No longer than 5 years at company.", 60),
])
def test_parse_max_tenure_constraint_realistic_phrasings(text, expected_months):
    assert parse_max_tenure_constraint_months(text) == expected_months


def test_parse_max_tenure_constraint_returns_none_when_absent():
    assert parse_max_tenure_constraint_months("Looking for a great backend engineer.") is None
    assert parse_max_tenure_constraint_months(None) is None


def test_parse_max_tenure_constraint_takes_strictest_smallest():
    """If two max-tenure phrases appear, the SMALLEST (strictest) cap wins."""
    text = "No more than 6 years at one company. Also, at most 3 years at a company."
    assert parse_max_tenure_constraint_months(text) == 36


def test_parse_min_tenure_still_works_after_max_patterns_added():
    """Regression: adding max-tenure patterns and reusing the shared
    number-word normalizer must not break the existing minimum-tenure
    parser."""
    from tenure_constraint_validator import parse_tenure_constraint_months
    assert parse_tenure_constraint_months("no one under 1 year at current company") == 12
    assert parse_tenure_constraint_months("minimum 2 years at company") == 24


# ---------------------------------------------------------------------------
# (e) enforce_max_tenure_constraint overrides / passes through correctly
# ---------------------------------------------------------------------------

def _nine_year_one_company_raw():
    return {
        'past_employers': [],
        'current_employers': [
            {'employer_name': 'BigCorp', 'employee_title': 'Staff Engineer',
             'start_date': '2017-07-22', 'end_date': None},
        ],
    }


def _three_year_one_company_raw():
    return {
        'past_employers': [],
        'current_employers': [
            {'employer_name': 'MidCorp', 'employee_title': 'Engineer',
             'start_date': '2023-07-22', 'end_date': None},
        ],
    }


def test_enforce_max_tenure_overrides_nine_year_candidate_to_no_go():
    jd_text = "We want mobility — no more than 5 years at one company."
    result = {
        "score": 8,
        "fit": "Great Fit",
        "summary": "Strong background.",
        "decision": "GO",
        "reasoning": "Strong background.",
    }
    raw = _nine_year_one_company_raw()

    out = enforce_max_tenure_constraint(dict(result), jd_text, raw)

    assert out["decision"] == "NO GO"
    assert out["fit"] == "Not a Fit"
    assert out["score"] <= TENURE_OVERRIDE_SCORE
    assert "max_tenure_override" in out
    assert out["max_tenure_override"]["threshold_months"] == 60
    assert out["max_tenure_override"]["actual_months"] >= 108  # ~9 years
    assert "no more than 60 months" in out["reasoning"]
    assert "BigCorp" in out["reasoning"]


def test_enforce_max_tenure_passes_three_year_candidate():
    jd_text = "We want mobility — no more than 5 years at one company."
    result = {
        "score": 8,
        "fit": "Great Fit",
        "summary": "Strong background.",
        "decision": "GO",
        "reasoning": "Strong background.",
    }
    raw = _three_year_one_company_raw()

    out = enforce_max_tenure_constraint(dict(result), jd_text, raw)

    # 3 years < 5-year cap — result must be untouched.
    assert out["decision"] == "GO"
    assert out["fit"] == "Great Fit"
    assert out["score"] == 8
    assert "max_tenure_override" not in out


def test_enforce_max_tenure_no_constraint_is_noop():
    result = {"score": 8, "fit": "Great Fit", "decision": "GO", "reasoning": "ok"}
    raw = _nine_year_one_company_raw()
    out = enforce_max_tenure_constraint(dict(result), "Great backend role, no constraints.", raw)
    assert out == result


def test_enforce_max_tenure_no_dates_is_noop():
    result = {"score": 8, "fit": "Great Fit", "decision": "GO", "reasoning": "ok"}
    raw = {"past_employers": [], "current_employers": []}
    out = enforce_max_tenure_constraint(dict(result), "no more than 5 years at one company", raw)
    assert out == result
