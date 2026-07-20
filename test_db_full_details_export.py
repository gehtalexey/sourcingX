"""Tests for the Database tab's opt-in "Full Details CSV" export helper.

Background: the Database tab's on-screen listing is built WITHOUT each
profile's raw Crustdata JSON (``raw_data``) to stay fast for large lists.
That means the normal "Download CSV" is missing the ``summary`` and the full
per-role job descriptions (``past_positions``), which only get populated when
``raw_data`` is present.

The fix adds a separate, opt-in export that first fetches ``raw_data`` for the
filtered rows and then composes the export DataFrame via
``build_full_details_export_df`` (in ``dashboard.py``), which runs the profiles
through the same ``profiles_to_dataframe`` -> ``prepare_df_for_export`` pipeline
every other CSV export uses.

These tests pin the pure composition helper (no Streamlit, no network, no DB):

  1. With ``raw_data`` present, ``summary`` and a full ``past_positions`` JSON
     blob (with per-role descriptions) DO show up in the export.
  2. Without ``raw_data`` (how the listing loads today), those same columns come
     out empty — this is exactly the gap the opt-in export closes.
  3. Column ordering matches ``prepare_df_for_export`` (first_name/last_name lead).
  4. An empty input list yields an empty DataFrame rather than raising.
  5. The ``crustdata_raw_json`` column carries the complete raw_data blob
     verbatim (including fields nothing else in this export surfaces, e.g.
     the current role's own description) — empty string when there's no
     raw_data to dump.
"""

from __future__ import annotations

import json

import pandas as pd


def _raw_profile_with_raw_data() -> dict:
    """A DB profile row that includes the raw Crustdata blob."""
    return {
        'linkedin_url': 'https://www.linkedin.com/in/jane-dev-123/',
        'name': 'Jane Dev',
        'current_title': 'Senior Backend Engineer',
        'current_company': 'Acme',
        'raw_data': {
            'name': 'Jane Dev',
            'first_name': 'Jane',
            'last_name': 'Dev',
            'summary': 'Backend engineer who loves distributed systems.',
            'current_employers': [
                {
                    'employee_title': 'Senior Backend Engineer',
                    'employer_name': 'Acme',
                    'start_date': '2022-01-01',
                    'description': 'Owns the payments service end to end.',
                }
            ],
            'past_employers': [
                {
                    'employee_title': 'Backend Engineer',
                    'employer_name': 'Globex',
                    'start_date': '2019-06-01',
                    'end_date': '2021-12-01',
                    'description': 'Built the billing pipeline in Python and Go.',
                }
            ],
            'all_employers': ['Acme', 'Globex'],
            'all_titles': ['Senior Backend Engineer', 'Backend Engineer'],
            'all_schools': ['Technion'],
            'skills': ['Python', 'Go', 'PostgreSQL'],
        },
    }


def _raw_profile_without_raw_data() -> dict:
    """A DB profile row as the fast listing loads it — no raw_data."""
    return {
        'linkedin_url': 'https://www.linkedin.com/in/john-doe-456/',
        'name': 'John Doe',
        'current_title': 'Frontend Engineer',
        'current_company': 'Initech',
        'location': 'Tel Aviv',
        'all_employers': ['Initech'],
        'all_titles': ['Frontend Engineer'],
        'all_schools': ['Tel Aviv University'],
        'skills': ['React', 'TypeScript'],
    }


def _build(raw_profiles):
    # Imported lazily so the (heavier) dashboard import only happens when a test
    # actually runs, and any collection-time import error is attributed here.
    from dashboard import build_full_details_export_df
    return build_full_details_export_df(raw_profiles)


def test_full_details_export_includes_summary_and_job_descriptions():
    df = _build([_raw_profile_with_raw_data()])

    assert len(df) == 1
    row = df.iloc[0]

    # Summary comes from raw_data and must be present in the full export.
    assert row['summary'] == 'Backend engineer who loves distributed systems.'

    # past_positions is json.dumps(past_employers) — the full job-description
    # text that the lightweight listing export is missing.
    assert row['past_positions'], "past_positions should be a non-empty JSON blob"
    parsed = json.loads(row['past_positions'])
    assert isinstance(parsed, list) and parsed
    assert parsed[0]['employer_name'] == 'Globex'
    assert 'billing pipeline' in parsed[0]['description']


def test_export_without_raw_data_has_empty_summary_and_positions():
    # This documents the gap the opt-in export closes: with no raw_data the
    # same columns exist but are empty.
    df = _build([_raw_profile_without_raw_data()])

    assert len(df) == 1
    row = df.iloc[0]
    assert row['summary'] == ''
    assert row['past_positions'] == ''


def test_column_order_matches_prepare_df_for_export():
    df = _build([_raw_profile_with_raw_data()])
    cols = list(df.columns)

    # prepare_df_for_export leads with personal-info columns.
    assert cols[0] == 'first_name'
    assert cols[1] == 'last_name'
    assert cols.index('summary') > cols.index('first_name')
    # No internal/underscore-prefixed columns leak into the export.
    assert not any(c.startswith('_') for c in cols)


def test_empty_input_returns_empty_dataframe():
    df = _build([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_crustdata_raw_json_column_has_the_complete_raw_response():
    # The flattened columns (summary, past_positions, etc.) only surface a
    # curated subset. crustdata_raw_json must carry the ENTIRE raw_data blob
    # verbatim, including fields nothing else in this export exposes (here:
    # the current role's own description, which past_positions never covers
    # since it only serializes past_employers).
    df = _build([_raw_profile_with_raw_data()])
    row = df.iloc[0]

    assert 'crustdata_raw_json' in df.columns
    parsed = json.loads(row['crustdata_raw_json'])
    assert parsed['summary'] == 'Backend engineer who loves distributed systems.'
    assert parsed['current_employers'][0]['description'] == 'Owns the payments service end to end.'
    assert parsed['past_employers'][0]['employer_name'] == 'Globex'


def test_crustdata_raw_json_is_empty_object_when_no_raw_data():
    # json.dumps({}) -> '{}', a valid (empty) JSON object — distinguishable
    # from a missing/unparseable value, not a bare empty string.
    df = _build([_raw_profile_without_raw_data()])
    assert df.iloc[0]['crustdata_raw_json'] == '{}'
