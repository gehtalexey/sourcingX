"""Tests for the Search tab's opt-in "Full Details" CSV export helper.

Background: the regular Search tab CSV (normalize_search_results_to_df ->
prepare_df_for_export) only flattens a curated subset of what Crustdata
returns per person — it drops things like certifications, languages, the
current role's own description, and (for description/semantic search) most
of the nested v2 response shape.

build_search_full_details_export_df() (in dashboard.py) attaches a
crustdata_raw_json column with the complete raw Crustdata response per
person, verbatim, with NO extra API calls:

  - A regular filter-search profile already IS the full raw Crustdata
    response (compact=false) — nothing to unwrap.
  - A description (semantic) search result is a lossy shim
    (semantic_profile_to_legacy_shape); its true original nested response is
    kept separately under _raw_semantic_result.

These tests cover both search types, plus the empty-input edge case. No
Streamlit, no network, no DB.
"""

from __future__ import annotations

import json

import pandas as pd

import crustdata_search as cs


def _raw_filter_search_profile() -> dict:
    """A raw profile exactly as it sits in crustdata_search_results for a
    regular filter search — the item itself IS the full Crustdata response."""
    return {
        'name': 'Jane Doe',
        'flagship_profile_url': 'https://www.linkedin.com/in/janedoe',
        'headline': 'Senior Backend Engineer',
        'skills': ['Python', 'Go'],
        'current_employers': [
            {'name': 'Acme', 'title': 'Engineer', 'start_date': '2022-01-01'}
        ],
        # Fields the flattened columns never surface — only the raw JSON does.
        'certifications': ['AWS Certified Solutions Architect'],
        'languages': ['English', 'Hebrew'],
    }


def _semantic_shim_profile() -> dict:
    """A shimmed description-search result exactly as it sits in
    crustdata_search_results — semantic_profile_to_legacy_shape() output."""
    return cs.semantic_profile_to_legacy_shape({
        'crustdata_person_id': 42,
        'fit': 'strong',
        'basic_profile': {
            'name': 'Sam Semantic',
            'headline': 'Founding Engineer',
            'normalized_title': {'matched_title': 'Software Engineer', 'confident': True},
        },
        'social_handles': {
            'professional_network_identifier': {
                'profile_url': 'https://www.linkedin.com/in/samsemantic'
            }
        },
    })


def _build(raw_profiles):
    from dashboard import build_search_full_details_export_df
    return build_search_full_details_export_df(raw_profiles)


def test_filter_search_raw_json_has_fields_the_flattened_columns_drop():
    df = _build([_raw_filter_search_profile()])

    assert len(df) == 1
    row = df.iloc[0]
    assert 'crustdata_raw_json' in df.columns

    parsed = json.loads(row['crustdata_raw_json'])
    assert parsed['certifications'] == ['AWS Certified Solutions Architect']
    assert parsed['languages'] == ['English', 'Hebrew']


def test_semantic_search_raw_json_uses_the_original_nested_response():
    df = _build([_semantic_shim_profile()])

    assert len(df) == 1
    row = df.iloc[0]

    parsed = json.loads(row['crustdata_raw_json'])
    # Must be the ORIGINAL nested v2 response (_raw_semantic_result), not the
    # lossy shim — the shim itself has no "basic_profile" key.
    assert 'basic_profile' in parsed
    assert parsed['basic_profile']['normalized_title']['matched_title'] == 'Software Engineer'
    assert parsed['fit'] == 'strong'


def test_mixed_batch_both_search_types_together():
    df = _build([_raw_filter_search_profile(), _semantic_shim_profile()])
    assert len(df) == 2

    by_name = {row['name']: row for _, row in df.iterrows()}
    assert 'certifications' in json.loads(by_name['Jane Doe']['crustdata_raw_json'])
    assert 'basic_profile' in json.loads(by_name['Sam Semantic']['crustdata_raw_json'])


def test_empty_input_returns_empty_dataframe():
    df = _build([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_profile_missing_linkedin_url_is_dropped_not_raised():
    # normalize_search_result() returns None (and gets skipped) for a profile
    # with no resolvable LinkedIn URL — the raw-json attach step must not
    # choke on that, it should just end up with fewer rows.
    no_url_profile = {'name': 'No URL Person', 'skills': ['Python']}
    df = _build([no_url_profile, _raw_filter_search_profile()])
    assert len(df) == 1
    assert df.iloc[0]['name'] == 'Jane Doe'
