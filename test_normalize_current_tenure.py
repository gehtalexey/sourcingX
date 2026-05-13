"""Regression tests for normalize_crustdata_profile's tenure fields.

`normalize_crustdata_profile` emits `current_start_date` (raw string) and
`current_years_at_company` (float) so the Filter+ and DB-search tabs can filter
by tenure-at-current-company without re-loading raw_crustdata. These tests
pin the contract:

- The fields are extracted from the MOST RECENT current employer (same one
  pick_current_employer returns), not from raw[0].
- Unparseable start_date strings leave current_years_at_company as None instead
  of producing nonsensical values like 2025 years.
- Missing current_employers leaves both fields as None — never crashes.
- The computed years value is roughly today minus start_date, in years.
"""

from datetime import datetime, timedelta

from normalizers import normalize_crustdata_profile


def _stub_raw(current_employers):
    """Minimal raw dict that passes normalize_crustdata_profile's gate."""
    return {
        'linkedin_profile_url': 'https://www.linkedin.com/in/test-person',
        'first_name': 'Test',
        'last_name': 'Person',
        'current_employers': current_employers,
    }


def test_tenure_fields_pulled_from_most_recent_current_employer():
    """When current_employers has multiple parallel roles, the tenure fields
    must match the MOST RECENT one (the one pick_current_employer surfaces),
    not raw index [0]."""
    older = {
        'employer_name': 'OldCo',
        'employee_title': 'Old Title',
        'start_date': '2018-03-01T00:00:00+00:00',
    }
    newer = {
        'employer_name': 'NewCo',
        'employee_title': 'New Title',
        'start_date': '2023-10-01T00:00:00+00:00',
    }
    result = normalize_crustdata_profile(_stub_raw([older, newer]))
    assert result['current_start_date'] == '2023-10-01T00:00:00+00:00'
    # Years value must reflect ~2.5y+ tenure as of today.
    assert result['current_years_at_company'] is not None
    expected_years = (datetime.now() - datetime(2023, 10, 1)).days / 365.25
    assert abs(result['current_years_at_company'] - expected_years) < 0.05


def test_tenure_years_is_none_when_start_date_unparseable():
    """A "Present" / "May 2025" string surfaces as current_start_date verbatim
    but does NOT produce a 2025-year tenure number — the field stays None so
    downstream filters can treat it as 'unknown' rather than reject."""
    emp = {
        'employer_name': 'AmbiguousCo',
        'employee_title': 'Engineer',
        'start_date': 'Present',
    }
    result = normalize_crustdata_profile(_stub_raw([emp]))
    assert result['current_start_date'] == 'Present'
    assert result['current_years_at_company'] is None


def test_tenure_fields_none_when_no_current_employer():
    """Profiles with no current_employers (career break, retired, etc.) leave
    both fields as None — must not crash."""
    raw = {
        'linkedin_profile_url': 'https://www.linkedin.com/in/test-person',
        'first_name': 'Test',
        'last_name': 'Person',
        'current_employers': [],
    }
    result = normalize_crustdata_profile(raw)
    assert result is not None
    assert result['current_start_date'] is None
    assert result['current_years_at_company'] is None


def test_tenure_fields_none_when_start_date_missing():
    """Current employer present but no start_date at all — both fields None."""
    emp = {
        'employer_name': 'NoDateCo',
        'employee_title': 'Engineer',
    }
    result = normalize_crustdata_profile(_stub_raw([emp]))
    assert result['current_start_date'] is None
    assert result['current_years_at_company'] is None


def test_tenure_handles_year_only_format():
    """Crustdata sometimes returns year-only ("2024") start_date. The parser
    treats it as Jan 1 of that year and computes years accordingly."""
    emp = {
        'employer_name': 'YearOnlyCo',
        'employee_title': 'Engineer',
        'start_date': '2024',
    }
    result = normalize_crustdata_profile(_stub_raw([emp]))
    assert result['current_start_date'] == '2024'
    assert result['current_years_at_company'] is not None
    expected = (datetime.now() - datetime(2024, 1, 1)).days / 365.25
    assert abs(result['current_years_at_company'] - expected) < 0.05
