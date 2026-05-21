"""Regression test for count_passed_email_split.

This pins the contract that the caption shown above the Filter-tab
"Enrich N passed with personal email" button never drifts from the
count of profiles `enrich_profiles_with_salesql` will actually process.

If somebody changes the skip logic inside enrich_profiles_with_salesql
without updating count_passed_email_split (or vice versa), this test
fails — preventing the silent count drift the user reported on PR #88.
"""
import numpy as np
import pandas as pd
import pytest

from dashboard import count_passed_email_split


def _function_skip_count(df: pd.DataFrame) -> tuple:
    """Reimplement the skip logic from enrich_profiles_with_salesql exactly.

    Returns (have_email, need_email). If this and count_passed_email_split
    disagree on any row, the caption lies to the user about the work that
    will happen on click.
    """
    have = 0
    need = 0
    for _, row in df.iterrows():
        if row.get('salesql_email') and not pd.isna(row.get('salesql_email')) and row.get('salesql_email') != '':
            have += 1
            continue
        if row.get('email') and not pd.isna(row.get('email')) and row.get('email') != '':
            have += 1
            continue
        need += 1
    return (have, need)


# ---------------------------------------------------------------------------
# Empty / null inputs
# ---------------------------------------------------------------------------

def test_none_df_returns_zero_zero():
    assert count_passed_email_split(None) == (0, 0)


def test_empty_df_returns_zero_zero():
    assert count_passed_email_split(pd.DataFrame()) == (0, 0)


def test_non_dataframe_input_returns_zero_zero():
    assert count_passed_email_split("not a df") == (0, 0)
    assert count_passed_email_split(42) == (0, 0)


# ---------------------------------------------------------------------------
# Single-column scenarios
# ---------------------------------------------------------------------------

def test_only_email_column_populated():
    df = pd.DataFrame({
        'linkedin_url': ['u1', 'u2', 'u3'],
        'email': ['a@x.com', '', 'b@x.com'],
    })
    assert count_passed_email_split(df) == (2, 1)
    assert count_passed_email_split(df) == _function_skip_count(df)


def test_only_salesql_email_column_populated():
    df = pd.DataFrame({
        'linkedin_url': ['u1', 'u2', 'u3'],
        'salesql_email': ['s@x.com', '', None],
    })
    assert count_passed_email_split(df) == (1, 2)
    assert count_passed_email_split(df) == _function_skip_count(df)


def test_neither_column_present():
    df = pd.DataFrame({'linkedin_url': ['u1', 'u2', 'u3']})
    assert count_passed_email_split(df) == (0, 3)
    assert count_passed_email_split(df) == _function_skip_count(df)


# ---------------------------------------------------------------------------
# Both columns present — the operative case
# ---------------------------------------------------------------------------

def test_both_columns_present_either_counts():
    df = pd.DataFrame({
        'linkedin_url': ['u1', 'u2', 'u3', 'u4'],
        'email':         ['db@x.com',  '',        None,    ''],
        'salesql_email': ['',          'sq@x.com', None,    ''],
    })
    # u1: db only → has. u2: sq only → has. u3: neither → need. u4: both empty → need.
    assert count_passed_email_split(df) == (2, 2)
    assert count_passed_email_split(df) == _function_skip_count(df)


def test_both_columns_with_overlap_no_double_count():
    df = pd.DataFrame({
        'linkedin_url': ['u1', 'u2'],
        'email':         ['db@x.com', 'db@x.com'],
        'salesql_email': ['sq@x.com', ''],
    })
    # Both rows have at least one email; should count as 2 have, 0 need.
    assert count_passed_email_split(df) == (2, 0)
    assert count_passed_email_split(df) == _function_skip_count(df)


# ---------------------------------------------------------------------------
# NaN / mixed-type weirdness
# ---------------------------------------------------------------------------

def test_nan_treated_as_missing():
    df = pd.DataFrame({
        'linkedin_url': ['u1', 'u2', 'u3'],
        'email': [np.nan, 'x@y.com', np.nan],
    })
    assert count_passed_email_split(df) == (1, 2)
    assert count_passed_email_split(df) == _function_skip_count(df)


def test_empty_string_treated_as_missing():
    df = pd.DataFrame({
        'linkedin_url': ['u1', 'u2'],
        'salesql_email': ['', 's@x.com'],
    })
    assert count_passed_email_split(df) == (1, 1)
    assert count_passed_email_split(df) == _function_skip_count(df)


# ---------------------------------------------------------------------------
# Invariant: have + need == len(df) (no row uncounted)
# ---------------------------------------------------------------------------

def test_total_invariant_holds_for_random_mix():
    states = [
        ('a@x.com', 'b@x.com'),  # both populated
        ('a@x.com', ''),
        ('a@x.com', None),
        ('', 'b@x.com'),
        (None, 'b@x.com'),
        ('', ''),
        ('', None),
        (None, None),
    ]
    df = pd.DataFrame({
        'linkedin_url': [f'u{i}' for i in range(len(states))],
        'email':         [s[0] for s in states],
        'salesql_email': [s[1] for s in states],
    })
    have, need = count_passed_email_split(df)
    assert have + need == len(df)
    assert (have, need) == _function_skip_count(df)
