"""Regression tests for normalizers.pick_current_employer.

Crustdata returns current_employers as a list that can contain multiple
active roles (advisor + full-time, two current jobs after a switch, IDF
reserve + civilian role). The list is not guaranteed to be ordered by
recency, so picking [0] blindly surfaces the WRONG current employer.

These tests pin two real-world scenarios that surfaced the bug:

- Gil Gitlin: Unit 8200 (older) + Cytactic (since 2025-05). SourcingX stored
  Unit 8200 as current_company because Crustdata returned 8200 at [0].
- Barak Ben Shimon: Vesttoo (since 2022-12) + monday.com (since 2023-10).
  SourcingX stored Vesttoo for the same reason.
"""

from normalizers import pick_current_employer


def test_returns_none_for_non_list_inputs():
    assert pick_current_employer(None) is None
    assert pick_current_employer({}) is None
    assert pick_current_employer("not a list") is None
    assert pick_current_employer(0) is None


def test_returns_none_for_empty_list():
    assert pick_current_employer([]) is None


def test_returns_none_when_no_dict_entries():
    assert pick_current_employer([None, "", 0]) is None


def test_returns_single_entry_unchanged():
    entry = {'employer_name': 'Acme', 'start_date': '2020-01-01'}
    assert pick_current_employer([entry]) is entry


def test_picks_most_recent_by_start_date_descending():
    older = {'employer_name': 'OldCo', 'start_date': '2018-03-01T00:00:00+00:00'}
    newer = {'employer_name': 'NewCo', 'start_date': '2023-10-01T00:00:00+00:00'}
    assert pick_current_employer([older, newer]) is newer
    # Order in the input list must not matter
    assert pick_current_employer([newer, older]) is newer


def test_missing_start_date_sorts_last():
    no_date = {'employer_name': 'NoDateCo'}
    with_date = {'employer_name': 'DatedCo', 'start_date': '2020-01-01'}
    assert pick_current_employer([no_date, with_date]) is with_date
    assert pick_current_employer([with_date, no_date]) is with_date


def test_gil_gitlin_scenario():
    """Gil's Crustdata profile lists Unit 8200 (older) + Cytactic (since
    2025-05). The fix should surface Cytactic, not 8200, as the current job."""
    unit_8200 = {
        'employer_name': 'Unit 8200 - Israeli Intelligence Corps',
        'employee_title': 'Full-stack Developer',
        'start_date': '2019-01-01T00:00:00+00:00',
    }
    cytactic = {
        'employer_name': 'Cytactic',
        'employee_title': 'Senior Software Engineer',
        'start_date': '2025-05-01T00:00:00+00:00',
    }
    assert pick_current_employer([unit_8200, cytactic])['employer_name'] == 'Cytactic'
    # Order independence: Crustdata sometimes returns reserve first.
    assert pick_current_employer([cytactic, unit_8200])['employer_name'] == 'Cytactic'


def test_barak_ben_shimon_scenario():
    """Barak's Crustdata profile lists Vesttoo (since 2022-12) + monday.com
    (since 2023-10). The fix should surface monday.com."""
    vesttoo = {
        'employer_name': 'Vesttoo',
        'employee_title': 'Full Stack Developer',
        'start_date': '2022-12-01T00:00:00+00:00',
    }
    monday = {
        'employer_name': 'monday.com',
        'employee_title': 'Senior Software Engineer',
        'start_date': '2023-10-01T00:00:00+00:00',
    }
    assert pick_current_employer([vesttoo, monday])['employer_name'] == 'monday.com'
    assert pick_current_employer([monday, vesttoo])['employer_name'] == 'monday.com'


def test_filters_non_dict_entries():
    valid = {'employer_name': 'Acme', 'start_date': '2020-01-01'}
    assert pick_current_employer([None, valid, "garbage", 42]) is valid


def test_unparseable_string_start_date_sorts_last():
    """Non-ISO start_date values ("Present", "May 2025") must sort LAST so
    they never win against a real ISO date. Lexicographic string sort would
    fail here because "P"/"M" > "2" in ASCII."""
    present = {'employer_name': 'PresentCo', 'start_date': 'Present'}
    real_iso = {'employer_name': 'RealCo', 'start_date': '2020-01-01'}
    assert pick_current_employer([present, real_iso])['employer_name'] == 'RealCo'
    assert pick_current_employer([real_iso, present])['employer_name'] == 'RealCo'

    localized = {'employer_name': 'LocCo', 'start_date': 'May 2025'}
    iso_2024 = {'employer_name': 'IsoCo', 'start_date': '2024-06-01'}
    assert pick_current_employer([localized, iso_2024])['employer_name'] == 'IsoCo'
    assert pick_current_employer([iso_2024, localized])['employer_name'] == 'IsoCo'


def test_parses_year_month_short_format():
    """Crustdata short forms like "2025-07" must parse and compare correctly
    against full ISO timestamps."""
    short = {'employer_name': 'ShortCo', 'start_date': '2025-07'}
    older_full = {'employer_name': 'OlderCo', 'start_date': '2020-01-01T00:00:00+00:00'}
    assert pick_current_employer([short, older_full])['employer_name'] == 'ShortCo'
    assert pick_current_employer([older_full, short])['employer_name'] == 'ShortCo'


def test_parses_year_only_format():
    """Year-only ("2024") must parse and compare correctly."""
    year_only = {'employer_name': 'YearCo', 'start_date': '2024'}
    older = {'employer_name': 'OlderCo', 'start_date': '2023-12-31'}
    assert pick_current_employer([year_only, older])['employer_name'] == 'YearCo'


def test_parses_iso_with_z_suffix():
    """ISO with 'Z' (UTC) and ISO with explicit '+00:00' must both parse."""
    with_z = {'employer_name': 'ZCo', 'start_date': '2025-05-01T00:00:00Z'}
    with_offset = {'employer_name': 'OffsetCo', 'start_date': '2020-05-01T00:00:00+00:00'}
    assert pick_current_employer([with_z, with_offset])['employer_name'] == 'ZCo'


def test_naive_and_aware_datetimes_compare_without_typeerror():
    """Mixed tz-aware ISO and naive year-month strings must not raise
    TypeError when sorted. The parser strips tzinfo to keep all datetimes
    naive."""
    aware = {'employer_name': 'AwareCo', 'start_date': '2025-05-01T00:00:00+00:00'}
    naive = {'employer_name': 'NaiveCo', 'start_date': '2024-01'}
    result = pick_current_employer([aware, naive])
    assert result['employer_name'] == 'AwareCo'


def test_tie_break_is_deterministic_on_equal_dates():
    """When two entries have the same start_date, stable sort preserves input
    order. The first one in the input wins."""
    first = {'employer_name': 'FirstIn', 'start_date': '2024-01-01'}
    second = {'employer_name': 'SecondIn', 'start_date': '2024-01-01'}
    assert pick_current_employer([first, second])['employer_name'] == 'FirstIn'
    assert pick_current_employer([second, first])['employer_name'] == 'SecondIn'


def test_two_unparseable_dates_tie_break_is_stable():
    """When both entries are unparseable, stable sort preserves input order."""
    a = {'employer_name': 'A', 'start_date': 'Present'}
    b = {'employer_name': 'B', 'start_date': 'who knows'}
    assert pick_current_employer([a, b])['employer_name'] == 'A'
    assert pick_current_employer([b, a])['employer_name'] == 'B'


def test_normalize_crustdata_profile_uses_most_recent():
    """End-to-end pin: normalize_crustdata_profile should surface the most
    recent current employer in current_company / current_title."""
    from normalizers import normalize_crustdata_profile

    raw = {
        'linkedin_profile_url': 'https://www.linkedin.com/in/gil-gitlin-87b720200',
        'first_name': 'Gil',
        'last_name': 'Gitlin',
        'current_employers': [
            {
                'employer_name': 'Unit 8200 - Israeli Intelligence Corps',
                'employee_title': 'Full-stack Developer',
                'start_date': '2019-01-01T00:00:00+00:00',
            },
            {
                'employer_name': 'Cytactic',
                'employee_title': 'Senior Software Engineer',
                'start_date': '2025-05-01T00:00:00+00:00',
            },
        ],
    }
    result = normalize_crustdata_profile(raw)
    assert result is not None
    assert result['current_company'] == 'Cytactic'
    assert result['current_title'] == 'Senior Software Engineer'
