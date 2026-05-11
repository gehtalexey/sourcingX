"""Regression tests for helpers.extract_display_fields.

PR #21 ("pick most recent current_employer") removed the
`current_employers = cd.get('current_employers') or []` assignment in
helpers.extract_display_fields but left two downstream references intact
(returned dict key `current_employers` and num_positions math). At runtime
this raised `NameError: name 'current_employers' is not defined` whenever
extract_display_fields was reached — observed when clicking "Send to
Filter+" after a DB search (the path runs through profiles_to_dataframe
which calls profile_to_display_row which calls extract_display_fields).

These tests pin:
- the basic shape of the returned dict
- that `current_employers` is surfaced in the output
- that `num_positions` adds past + current correctly
- the bug repro (multi-current-employer profile, screenshot-equivalent)
"""

from helpers import extract_display_fields


def test_returns_empty_dict_shape_for_empty_input():
    out = extract_display_fields({})
    assert out['current_employers'] == []
    assert out['past_employers'] == []
    assert out['num_positions'] == 0
    assert out['current_company'] == ''
    assert out['current_title'] == ''


def test_returns_empty_dict_shape_for_none_input():
    out = extract_display_fields(None)
    assert out['current_employers'] == []
    assert out['num_positions'] == 0


def test_surfaces_current_employers_list_in_output():
    """Pins the regression: PR #21 lost this reference and the function
    raised NameError. The returned dict must include the full
    current_employers list, not just the picked most-recent entry."""
    emps = [
        {'employer_name': 'NewCo', 'start_date': '2024-01-01'},
        {'employer_name': 'OldCo', 'start_date': '2020-01-01'},
    ]
    out = extract_display_fields({'current_employers': emps})
    assert out['current_employers'] == emps
    assert out['current_company'] == 'NewCo'  # picked by start_date desc


def test_num_positions_adds_past_and_current():
    out = extract_display_fields({
        'current_employers': [{'employer_name': 'A'}, {'employer_name': 'B'}],
        'past_employers': [{'employer_name': 'C'}, {'employer_name': 'D'}, {'employer_name': 'E'}],
    })
    assert out['num_positions'] == 5


def test_send_to_filter_plus_repro_does_not_raise():
    """End-to-end pin for the user-reported bug.

    Calling extract_display_fields on a realistic Crustdata-shaped profile
    (the kind returned by get_profiles_by_urls) must not raise NameError.
    """
    profile = {
        'name': 'Test User',
        'first_name': 'Test',
        'last_name': 'User',
        'headline': 'Software Engineer at SomeCo',
        'location': 'Tel Aviv',
        'summary': 'sample summary',
        'current_employers': [
            {
                'employer_name': 'SomeCo',
                'employee_title': 'Software Engineer',
                'start_date': '2023-01-01T00:00:00+00:00',
            },
        ],
        'past_employers': [
            {'employer_name': 'OldCo', 'employee_title': 'Junior Engineer',
             'start_date': '2020-01-01', 'end_date': '2022-12-01'},
        ],
        'all_employers': ['SomeCo', 'OldCo'],
        'all_titles': ['Software Engineer', 'Junior Engineer'],
        'all_schools': ['Tel Aviv University'],
        'skills': ['Python', 'TypeScript'],
        'num_of_connections': 500,
    }
    # The bug surfaced as a NameError here. The assertion is "no exception".
    out = extract_display_fields(profile)
    assert out['current_company'] == 'SomeCo'
    assert out['current_title'] == 'Software Engineer'
    assert out['num_positions'] == 2
