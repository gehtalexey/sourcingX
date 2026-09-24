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
    """Reimplement the skip logic from enrich_profiles_with_salesql exactly:
    a row is skipped only when `salesql_email` or `email` holds a real
    address (a string with an '@').

    Returns (have_email, need_email). If this and count_passed_email_split
    disagree on any row, the caption lies to the user about the work that
    will happen on click.
    """
    def _real(v):
        return isinstance(v, str) and '@' in v.strip()

    have = 0
    need = 0
    for _, row in df.iterrows():
        if _real(row.get('salesql_email')) or _real(row.get('email')):
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


# ---------------------------------------------------------------------------
# The false "All profiles already have emails!" message
# ---------------------------------------------------------------------------

PLACEHOLDERS = ['', None, np.nan, '   ', 'nan', 'None', 'not found', 'N/A']


def test_placeholders_after_a_missed_lookup_count_as_no_email():
    """A lookup that found nothing leaves salesql_email empty or holding a
    placeholder. None of those may count as 'has email' -- that is what made
    the UI say 'All profiles already have emails!' after 0 were found."""
    from dashboard import has_email_mask, count_salesql_lookups

    df = pd.DataFrame({
        'linkedin_url': [f'https://www.linkedin.com/in/p{i}' for i in range(len(PLACEHOLDERS))],
        'salesql_email': PLACEHOLDERS,
        'email': PLACEHOLDERS,
    })
    assert int(has_email_mask(df).sum()) == 0
    assert count_passed_email_split(df) == (0, len(PLACEHOLDERS))
    # Every one of them is still offered for lookup.
    assert count_salesql_lookups(df) == len(PLACEHOLDERS)


def test_lookup_count_skips_rows_without_url_and_respects_limit():
    from dashboard import count_salesql_lookups

    df = pd.DataFrame({
        'linkedin_url': ['https://www.linkedin.com/in/a', None, 'https://www.linkedin.com/in/c', ''],
        'email': ['', '', 'c@x.com', ''],
    })
    assert count_salesql_lookups(df) == 1
    assert count_salesql_lookups(df, limit=0) == 1
    df2 = pd.DataFrame({'linkedin_url': [f'https://www.linkedin.com/in/{i}' for i in range(5)]})
    assert count_salesql_lookups(df2, limit=3) == 3


def test_run_message_says_found_n_of_m():
    from dashboard import salesql_run_message

    assert "found 0 of 7" in salesql_run_message(0, 7)
    assert "found 3 of 7" in salesql_run_message(3, 7)
    assert "already have" not in salesql_run_message(0, 7)


def test_enrich_run_reports_found_and_looked_up(monkeypatch):
    """A run where SalesQL misses everything reports found 0 of N, and the
    rows stay counted as needing an email (no false 'all have emails')."""
    import dashboard
    from dashboard import enrich_profiles_with_salesql, has_email_mask

    monkeypatch.setattr(dashboard, "_global_rate_limit_wait", lambda: None)
    monkeypatch.setattr(dashboard, "get_usage_tracker", lambda: None)
    monkeypatch.setattr(dashboard, "enrich_with_salesql",
                        lambda url, key, personal_only=True, tracker=None: {'emails': [], 'error': 'Profile not found'})

    df = pd.DataFrame({'linkedin_url': [f'https://www.linkedin.com/in/m{i}' for i in range(4)]})
    out = enrich_profiles_with_salesql(df, "test-key", max_workers=2)
    assert out.attrs['salesql_stats'] == {'looked_up': 4, 'found': 0}
    assert int(has_email_mask(out).sum()) == 0


# ---------------------------------------------------------------------------
# SalesQL billing: a miss costs nothing (salesql-api skill billing rule)
# ---------------------------------------------------------------------------

def _logged(**kwargs):
    from unittest.mock import MagicMock
    from usage_tracker import UsageTracker

    db = MagicMock()
    db.insert.return_value = [{}]
    UsageTracker(db).log_salesql(**kwargs)
    return db.insert.call_args.args[1]


def test_salesql_hit_logs_one_credit():
    row = _logged(lookups=1, emails_found=1)
    assert row['credits_used'] == 1
    assert row['status'] == 'success'


def test_salesql_miss_logs_zero_credits_but_keeps_the_row():
    row = _logged(lookups=1, emails_found=0)
    assert row['credits_used'] == 0
    assert row['status'] == 'not_found'
    assert row['request_count'] == 1


def _salesql_rows():
    hit = {'provider': 'salesql', 'status': 'success', 'request_count': 1, 'credits_used': 1,
           'created_at': '2026-09-24T10:00:00'}
    miss = {'provider': 'salesql', 'status': 'not_found', 'request_count': 1, 'credits_used': 0,
            'created_at': '2026-09-24T10:00:01'}
    err = {'provider': 'salesql', 'status': 'error', 'request_count': 0, 'credits_used': 0,
           'created_at': '2026-09-24T10:00:02'}
    return [hit, hit, miss, miss, miss, err]


def test_usage_summary_counts_misses_as_lookups_not_credits():
    """Usage tab: misses log 0 credits but still count against the
    5,000/day lookup limit, so lookups and credits are separate figures."""
    from unittest.mock import MagicMock
    from db import get_usage_summary

    client = MagicMock()
    client.select.return_value = _salesql_rows()
    s = get_usage_summary(client)['salesql']
    assert s['lookups'] == 5
    assert s['credits'] == 2
    assert s['errors'] == 1


def test_usage_by_date_charts_lookups():
    from unittest.mock import MagicMock
    from db import get_usage_by_date

    client = MagicMock()
    client.select.return_value = _salesql_rows()
    assert get_usage_by_date(client)[0]['salesql'] == 5


def test_salesql_error_is_not_billed():
    row = _logged(lookups=1, status='error', error_message='API error 500', billed=False)
    assert row['credits_used'] == 0
    assert row['status'] == 'error'
