"""DB-tab title filter: OR across current_title and headline.

The "Current Title" column filter on the Database tab used to match only
the structured `current_title` column. Many profiles have their effective
title only in their `headline` (e.g. "Senior Python Engineer at Wiz") and
were silently dropped from the funnel even when the full-text phase had
already returned them via the search_text tsvector (which indexes
headline — see migration 018).

Migration 019 returns headline from `search_profiles_text`. The DB-tab
column filter now ORs the title query across `current_title` and
`headline`. These tests pin that contract.

The actual filter helpers (`_parse_boolean_query_client`,
`_eval_boolean_ast`, `_boolean_filter`) are nested inside the Streamlit
tab in dashboard.py so they can't be imported directly. This module
replicates the substring-AST-OR semantics in a tiny pure helper and
tests THAT plus the policy that title_mask = (mask on current_title) OR
(mask on headline) when headline is present.
"""

import pandas as pd
import pytest


def _term_in(text, term):
    """Case-insensitive substring match, mirroring _eval_boolean_ast TERM."""
    return term.lower() in (text or "").lower()


def _apply_title_filter(df, query):
    """Return the boolean mask the DB tab now applies for `current_title`.

    Encodes the policy: match against current_title; if headline column is
    present, also OR against headline. Backward-compatible when headline
    is missing (older RPC / pre-migration deploys).
    """
    title_mask = df["current_title"].fillna("").apply(lambda v: _term_in(v, query))
    if "headline" in df.columns:
        head_mask = df["headline"].fillna("").apply(lambda v: _term_in(v, query))
        title_mask = title_mask | head_mask
    return title_mask


# Sample dataframe shape mirroring what the RPC now returns post-migration 019.
SAMPLE = pd.DataFrame(
    [
        # 0: title matches directly
        {"current_title": "Software Engineer", "headline": "i build things"},
        # 1: title is the wrong thing, headline is the real one
        {"current_title": "Career Coach", "headline": "Senior Python Engineer at Wiz"},
        # 2: title is empty, headline has the role
        {"current_title": "", "headline": "Senior Software Engineer"},
        # 3: title is empty AND headline doesn't mention engineer
        {"current_title": "", "headline": "Founder, ex-Google"},
        # 4: title is wrong AND headline is wrong
        {"current_title": "Product Manager", "headline": "PM at Stripe"},
        # 5: headline NaN (still must not crash)
        {"current_title": "Backend Engineer", "headline": None},
    ]
)


def test_title_only_in_current_title_matches():
    mask = _apply_title_filter(SAMPLE, "engineer")
    # Rows 0, 1, 2, 5 mention engineer in title OR headline. Row 3 and 4 don't.
    assert list(mask) == [True, True, True, False, False, True]


def test_title_only_in_headline_now_matches():
    """The whole point of the migration: row 1 has the role only in headline."""
    mask = _apply_title_filter(SAMPLE, "python")
    # Only row 1 has 'python' anywhere; previously would have matched 0 rows.
    assert mask.iloc[1] is True or mask.iloc[1] == True  # noqa: E712
    assert mask.sum() == 1


def test_empty_current_title_does_not_short_circuit_headline():
    mask = _apply_title_filter(SAMPLE, "senior software")
    # Row 2 has empty current_title but headline says "Senior Software Engineer".
    assert mask.iloc[2] == True  # noqa: E712


def test_missing_headline_column_falls_back_to_current_title_only():
    """Backward compat: if RPC pre-migration is still deployed, headline column
    is absent — filter must still work using just current_title."""
    df_no_headline = SAMPLE.drop(columns=["headline"])
    mask = _apply_title_filter(df_no_headline, "engineer")
    # current_title only: rows 0, 5 match; others don't.
    assert list(mask) == [True, False, False, False, False, True]


def test_headline_null_does_not_crash():
    """Row 5 has headline=None. The fillna('') in the filter must handle it."""
    mask = _apply_title_filter(SAMPLE, "engineer")
    # Row 5 has Backend Engineer in current_title — matches via current_title side.
    assert mask.iloc[5] == True  # noqa: E712


@pytest.mark.parametrize(
    "query,expected_count",
    [
        ("engineer", 4),    # rows 0, 1, 2, 5
        ("python", 1),       # row 1 (only in headline)
        ("stripe", 1),       # row 4 (only in headline)
        ("nonsense", 0),     # nothing
    ],
)
def test_recall_lift_examples(query, expected_count):
    mask = _apply_title_filter(SAMPLE, query)
    assert mask.sum() == expected_count


# ---------------------------------------------------------------------------
# Coverage: db.py KEEP_FIELDS must include 'headline' so the RPC value
# survives the dataframe trim that search_profiles_fulltext does.
# ---------------------------------------------------------------------------


def test_keep_fields_contains_headline():
    """If somebody trims headline back out of KEEP_FIELDS, the filter silently
    falls back to current_title only and the bug returns."""
    import inspect

    import db

    src = inspect.getsource(db.search_profiles_fulltext)
    assert "'headline'" in src or '"headline"' in src
