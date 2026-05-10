"""Regression tests for the years-of-experience metric (Issue 3).

Chen reported that SourcingX's "Exp" column on the results table counts IDF
military service as engineering experience. Concrete example: Alon Ben Haim's
profile showed 9y total, but ~3 of those years were army positions, not
engineering. See also:
- https://www.linkedin.com/in/alon-ben-haim-607469144/
- https://www.linkedin.com/in/roytmax/
- https://www.linkedin.com/in/amitay-bremer-2a1a03169/

These tests pin the contract: military / IDF positions must not be summed into
the displayed YoE total, but civilian positions (and the overall fallback)
should still work.

Run with `pytest test_experience_excludes_military.py -v`.
"""

import pytest

from normalizers import (
    _is_military_position,
    compute_years_of_experience,
    parse_duration,
)


# ----------------------------------------------------------------------------
# Predicate: _is_military_position
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("title,company", [
    ("IDF Soldier", "Israel Defense Forces"),
    ("Software Developer", "IDF"),
    ("Combat Soldier", ""),
    ("Officer", "Israeli Defense Forces"),
    ("Developer", "Unit 8200"),
    ("Engineer", "Mamram"),
    ("Cadet", "Talpiot"),
    ("Officer", "Israeli Army"),
    ("Soldier", "צה\"ל"),
    ("חייל", "צבא"),
    ("Military Intelligence Officer", "Some Unit"),
])
def test_is_military_position_true(title, company):
    assert _is_military_position(title, company) is True


@pytest.mark.parametrize("title,company", [
    ("Backend Engineer", "Wiz"),
    ("Software Engineer", "Google"),
    ("Security Researcher", "Check Point"),  # 'security' alone is NOT military
    ("Defense Industry Analyst", "Rafael"),  # 'defense' alone is NOT military
    ("", ""),
    (None, None),
    # Word-boundary regressions (Codex review on PR #17):
    # short Latin tokens like "idf" / "army" must not match unrelated words.
    ("Midfield Engineer", "Soccer Club"),       # 'idf' substring of 'midfield'
    ("Senior Engineer", "Midfielder Analytics"),  # 'idf' substring on company
    ("Armyard Sales", "Logistics Co"),           # 'army' substring of 'armyard'
])
def test_is_military_position_false(title, company):
    assert _is_military_position(title, company) is False


# Explicit regression tests called out by Codex's review on PR #17.
def test_midfield_engineer_is_not_military():
    """`_is_military_position` must use word-boundary matching for 'idf' so
    'Midfield Engineer' does NOT register as a military position."""
    assert _is_military_position("Midfield Engineer", "Soccer Club") is False


def test_army_word_boundary_still_matches():
    """Boundary matching must still flag the real 'army' token."""
    assert _is_military_position("Army Engineer", "Defense Co") is True


def test_idf_phrase_and_token_combo():
    """Phrase match on company + short-token match on title both fire."""
    assert _is_military_position(
        "IDF Software Engineer", "Israel Defense Forces"
    ) is True


def test_word_bounded_short_token_in_company():
    """Short tokens (e.g. '8200', 'mamram') should still match inside a
    multi-word company string when surrounded by whitespace."""
    assert _is_military_position("Engineer", "Mamram Unit 8200") is True


# ----------------------------------------------------------------------------
# parse_duration: short Crustdata format ("2y 9m")
# ----------------------------------------------------------------------------

def test_parse_duration_short_format():
    assert parse_duration("2y 9m") == pytest.approx(2.75, rel=1e-2)
    assert parse_duration("3y") == 3.0
    assert parse_duration("11m") == pytest.approx(11 / 12, rel=1e-2)


# ----------------------------------------------------------------------------
# compute_years_of_experience: military positions excluded
# ----------------------------------------------------------------------------

def test_military_position_excluded_from_yoe_sum():
    """The headline regression: a profile that mixes IDF + civilian positions
    should report only the civilian total — not the combined value."""
    profile = {
        'current_employers': [
            {'employee_title': 'Senior Backend Engineer', 'employer_name': 'Wiz',
             'duration': '3y'},
        ],
        'past_employers': [
            {'employee_title': 'Backend Engineer', 'employer_name': 'Monday.com',
             'duration': '3y'},
            # 3 years of military service — must be skipped
            {'employee_title': 'Software Developer', 'employer_name': 'IDF',
             'duration': '3y'},
        ],
        # Raw aggregate from Crustdata says 9 — but that's the buggy number
        'years_of_experience_raw': 9,
    }

    yoe = compute_years_of_experience(profile)
    assert yoe == pytest.approx(6.0, abs=0.1), (
        f"Expected 6y (civilian only) but got {yoe} — military positions "
        f"are leaking into the YoE sum."
    )


def test_idf_in_title_also_excluded():
    """Even when the company name is innocuous, an IDF/Army keyword in the
    title is enough to skip the position."""
    profile = {
        'current_employers': [
            {'employee_title': 'Backend Engineer', 'employer_name': 'Wiz',
             'duration': '2y'},
        ],
        'past_employers': [
            # Some profiles list "IDF — Software Developer" in the title with a
            # blank or generic company. Must still be excluded.
            {'employee_title': 'IDF Software Developer', 'employer_name': '',
             'duration': '3y'},
            {'employee_title': 'Army Officer', 'employer_name': 'Government',
             'duration': '2y'},
        ],
    }
    yoe = compute_years_of_experience(profile)
    assert yoe == pytest.approx(2.0, abs=0.1)


def test_civilian_only_profile_unchanged():
    """Profiles with no military positions should sum normally."""
    profile = {
        'current_employers': [
            {'employee_title': 'Senior Backend Engineer', 'employer_name': 'Wiz',
             'duration': '2y'},
        ],
        'past_employers': [
            {'employee_title': 'Software Engineer', 'employer_name': 'Google',
             'duration': '3y'},
            {'employee_title': 'Junior Developer', 'employer_name': 'Startup',
             'duration': '1y 6m'},
        ],
    }
    yoe = compute_years_of_experience(profile)
    assert yoe == pytest.approx(6.5, abs=0.1)


def test_security_role_is_not_treated_as_military():
    """Civilian security/defense work must NOT be filtered."""
    profile = {
        'current_employers': [
            {'employee_title': 'Security Researcher', 'employer_name': 'Check Point',
             'duration': '4y'},
        ],
        'past_employers': [
            {'employee_title': 'Defense Industry Engineer', 'employer_name': 'Rafael',
             'duration': '3y'},
        ],
    }
    yoe = compute_years_of_experience(profile)
    assert yoe == pytest.approx(7.0, abs=0.1)


def test_fallback_to_raw_when_no_positions():
    """If we don't have per-position records, fall back to the raw aggregate."""
    profile = {
        'years_of_experience_raw': 5,
    }
    assert compute_years_of_experience(profile) == 5.0


def test_compute_from_date_range_when_duration_missing():
    """When `duration` is absent we should still derive years from
    start_date / end_date."""
    profile = {
        'past_employers': [
            {'employee_title': 'Engineer', 'employer_name': 'Acme',
             'start_date': '2018-01', 'end_date': '2021-01'},  # 3y
            {'employee_title': 'IDF Soldier', 'employer_name': 'IDF',
             'start_date': '2014-01', 'end_date': '2017-01'},  # 3y — excluded
        ],
    }
    yoe = compute_years_of_experience(profile)
    assert yoe == pytest.approx(3.0, abs=0.1)


def test_exclude_military_can_be_disabled():
    """Opting out restores the old behavior (sum of all positions)."""
    profile = {
        'current_employers': [
            {'employee_title': 'Backend Engineer', 'employer_name': 'Wiz',
             'duration': '3y'},
        ],
        'past_employers': [
            {'employee_title': 'Software Developer', 'employer_name': 'IDF',
             'duration': '3y'},
        ],
    }
    civilian_only = compute_years_of_experience(profile, exclude_military=True)
    everything = compute_years_of_experience(profile, exclude_military=False)
    assert civilian_only == pytest.approx(3.0, abs=0.1)
    assert everything == pytest.approx(6.0, abs=0.1)


def test_none_and_empty_inputs_return_none():
    assert compute_years_of_experience(None) is None
    assert compute_years_of_experience({}) is None
    assert compute_years_of_experience({'current_employers': [], 'past_employers': []}) is None
