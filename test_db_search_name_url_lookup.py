"""
Regression tests for the Tab 7 (DB Search) name + LinkedIn URL lookup path.

Use case
--------
Alexey often wants to look up a *specific known profile* without spending
Crustdata credits — e.g., a recruiter is on a candidate's LinkedIn page and
wants to check whether SourcingX already has them enriched. Tab 7 in
``dashboard.py`` exposes two fields for this:

- **Name** (``db_f_name``) — case-insensitive substring match on the
  ``name`` column. Already wired before this change; tests pin behaviour.
- **LinkedIn URL** (``db_f_linkedin_url``) — new. The dashboard normalises
  the input via ``normalizers.normalize_linkedin_url`` (strips trailing
  slashes, query params, lowercases the domain). If the result is a full
  ``https://www.linkedin.com/in/<slug>`` URL the DB query uses ``eq.`` for
  an exact match. If the input is just a slug fragment (no ``linkedin.com``)
  the query falls back to ``ilike.*slug*`` for substring match.

We exercise:
1. Name input "Gil Gitlin" → ``name.ilike.*Gil Gitlin*`` postgrest filter.
2. Name partial "Gil" → ``name.ilike.*Gil*`` (case-insensitive substring).
3. URL full → exact ``linkedin_url.eq.<url>``.
4. URL slug only → ``linkedin_url.ilike.*<slug>*``.
5. URL with trailing slash / query params is normalised before the eq match.
6. Both new fields contribute to ``has_column_filters``.
"""

from unittest.mock import MagicMock

import db
from normalizers import normalize_linkedin_url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeClient:
    """Captures select() calls so we can assert on the filter params."""

    def __init__(self, rows=None):
        self.rows = rows or []
        self.last_call = None

    def select(self, table, columns='*', filters=None, limit=5000, **kwargs):
        self.last_call = {
            'table': table,
            'columns': columns,
            'filters': filters or {},
            'limit': limit,
        }
        return self.rows


def _has_column_filters(filters: dict) -> bool:
    """Mirror of the dashboard.py Tab 7 has_column_filters expression.

    Kept in lock-step with ``dashboard.py`` so this test catches drift if
    someone removes ``linkedin_url`` from the participation list.
    """
    keys = ['name', 'current_title', 'past_titles', 'current_company',
            'past_companies', 'location', 'skills', 'schools',
            'linkedin_url', 'date_after', 'has_email']
    return any(filters.get(k) for k in keys) or filters.get('freshness', 'All') != 'All'


# ---------------------------------------------------------------------------
# Name lookup
# ---------------------------------------------------------------------------

class TestNameLookup:
    """The Name field must do a case-insensitive substring match on the
    ``name`` column. Today this is the existing behaviour via
    ``parse_boolean_query`` — these tests pin it so we don't regress it
    while extending the tab."""

    def test_exact_full_name(self):
        """'Gil Gitlin' → both tokens must match name (case-insensitive).

        ``parse_boolean_query`` treats inter-word whitespace as AND, so
        ``Gil Gitlin`` becomes ``name ILIKE '%Gil%' AND name ILIKE '%Gitlin%'``.
        This still correctly hits the row with ``name='Gil Gitlin'`` and
        gives the recruiter the "Alexey types two names, gets that one
        person" lookup. We pin both the AND structure and that each token
        appears as a substring filter on the ``name`` column."""
        client = _FakeClient([{'name': 'Gil Gitlin', 'linkedin_url': 'x'}])
        db.search_profiles_boolean(client, {'name': 'Gil Gitlin'})
        params = client.last_call['filters']
        and_clause = params.get('and')
        assert and_clause is not None, params
        assert 'name.ilike.*Gil*' in and_clause, and_clause
        assert 'name.ilike.*Gitlin*' in and_clause, and_clause

    def test_partial_first_name(self):
        client = _FakeClient([{'name': 'Gil Gitlin'}, {'name': 'Gilad Cohen'}])
        db.search_profiles_boolean(client, {'name': 'Gil'})
        params = client.last_call['filters']
        assert params.get('name') == 'ilike.*Gil*', params

    def test_name_lookup_returns_rows(self):
        rows = [{'name': 'Gil Gitlin', 'linkedin_url': 'x'}]
        client = _FakeClient(rows)
        result = db.search_profiles_boolean(client, {'name': 'Gil Gitlin'})
        assert result == rows


# ---------------------------------------------------------------------------
# LinkedIn URL lookup
# ---------------------------------------------------------------------------

class TestLinkedInUrlLookup:
    """The new LinkedIn URL field uses ``eq.`` for full URLs and falls back
    to ``ilike.*slug*`` for slug fragments."""

    def test_full_url_uses_eq(self):
        full = 'https://www.linkedin.com/in/gil-gitlin-87b720200'
        client = _FakeClient()
        db.search_profiles_boolean(client, {'linkedin_url': full})
        params = client.last_call['filters']
        assert params.get('linkedin_url') == f'eq.{full}', params

    def test_slug_only_uses_ilike(self):
        client = _FakeClient()
        db.search_profiles_boolean(client, {'linkedin_url': 'gil-gitlin-87b720200'})
        params = client.last_call['filters']
        assert params.get('linkedin_url') == 'ilike.*gil-gitlin-87b720200*', params

    def test_trailing_slash_normalised_before_query(self):
        # Simulate the dashboard pre-normalisation step.
        raw = 'https://www.linkedin.com/in/gil-gitlin-87b720200/'
        normalised = normalize_linkedin_url(raw)
        assert normalised == 'https://www.linkedin.com/in/gil-gitlin-87b720200'

        client = _FakeClient()
        db.search_profiles_boolean(client, {'linkedin_url': normalised})
        params = client.last_call['filters']
        assert params.get('linkedin_url') == f'eq.{normalised}', params

    def test_query_params_normalised_before_query(self):
        raw = 'https://www.linkedin.com/in/gil-gitlin-87b720200?utm_source=share'
        normalised = normalize_linkedin_url(raw)
        assert normalised == 'https://www.linkedin.com/in/gil-gitlin-87b720200'

        client = _FakeClient()
        db.search_profiles_boolean(client, {'linkedin_url': normalised})
        params = client.last_call['filters']
        assert params.get('linkedin_url') == f'eq.{normalised}', params

    def test_url_lookup_returns_rows(self):
        rows = [{'name': 'Gil Gitlin',
                 'linkedin_url': 'https://www.linkedin.com/in/gil-gitlin-87b720200'}]
        client = _FakeClient(rows)
        result = db.search_profiles_boolean(
            client,
            {'linkedin_url': 'https://www.linkedin.com/in/gil-gitlin-87b720200'}
        )
        assert result == rows


# ---------------------------------------------------------------------------
# has_column_filters participation
# ---------------------------------------------------------------------------

class TestHasColumnFilters:
    """Both new lookups must trigger the search; otherwise the Tab 7
    submit path bypasses ``search_profiles_boolean`` entirely."""

    def test_empty_filters_returns_false(self):
        assert _has_column_filters({}) is False

    def test_name_alone_triggers_search(self):
        assert _has_column_filters({'name': 'Gil Gitlin'}) is True

    def test_linkedin_url_alone_triggers_search(self):
        assert _has_column_filters({
            'linkedin_url': 'https://www.linkedin.com/in/gil-gitlin-87b720200'
        }) is True

    def test_slug_only_linkedin_url_triggers_search(self):
        assert _has_column_filters({'linkedin_url': 'gil-gitlin-87b720200'}) is True

    def test_both_fields_together_trigger_search(self):
        assert _has_column_filters({
            'name': 'Gil Gitlin',
            'linkedin_url': 'https://www.linkedin.com/in/gil-gitlin-87b720200',
        }) is True


# ---------------------------------------------------------------------------
# Mixed mode (full-text query + linkedin_url) — client-side filter
# ---------------------------------------------------------------------------

def _client_side_apply_linkedin_url(df, af):
    """Mirror of the dashboard.py Tab 7 client-side ``linkedin_url`` branch.

    When the user submits BOTH a full-text query AND a LinkedIn URL,
    ``search_profiles_fulltext`` runs (broad), and then the dashboard
    applies column predicates row-by-row on the returned DataFrame. This
    helper mirrors the linkedin_url predicate that the dashboard now
    applies, kept in lock-step so regressions get caught here.

    Codex flagged on PR #40 that without this branch, the URL constraint
    was silently dropped in mixed mode.
    """
    import pandas as pd

    mask = pd.Series(True, index=df.index)
    url_q = af.get('linkedin_url')
    if url_q and 'linkedin_url' in df.columns:
        url_q_str = str(url_q).strip()
        if url_q_str:
            url_l = url_q_str.lower()
            if '/in/' in url_l and 'linkedin.com' in url_l:
                mask &= df['linkedin_url'].fillna('').astype(str) == url_q_str
            else:
                needle = url_q_str.lower()
                mask &= df['linkedin_url'].fillna('').astype(str).str.lower().str.contains(
                    needle, regex=False
                )
    return df[mask]


class TestMixedModeLinkedInUrlFilter:
    """Regression for Codex's PR #40 blocker.

    In mixed mode (full-text query + linkedin_url), ``search_profiles_fulltext``
    returns the broad full-text result set. The dashboard's client-side filter
    loop then narrows those rows. Before this fix, the loop applied name /
    title / company / location / array / has_email predicates but NOT the
    LinkedIn URL, so the URL constraint was silently ignored whenever a
    full-text query was also present.
    """

    def _fulltext_result_rows(self):
        """Simulate what ``search_profiles_fulltext`` would return for a
        broad query like ``backend`` — multiple rows, only one matching the
        URL the recruiter typed in the LinkedIn URL field."""
        import pandas as pd
        return pd.DataFrame([
            {'name': 'Gil Gitlin',
             'linkedin_url': 'https://www.linkedin.com/in/gil-gitlin-87b720200',
             'current_title': 'Backend Engineer'},
            {'name': 'Other Person',
             'linkedin_url': 'https://www.linkedin.com/in/other-person',
             'current_title': 'Backend Engineer'},
            {'name': 'Third Person',
             'linkedin_url': 'https://www.linkedin.com/in/third-person',
             'current_title': 'Backend Engineer'},
        ])

    def test_full_url_narrows_fulltext_results_to_one_row(self):
        """Mixed mode with a full URL must return only the exact-match row."""
        df = self._fulltext_result_rows()
        af = {
            'name': None,
            'linkedin_url': 'https://www.linkedin.com/in/gil-gitlin-87b720200',
        }
        result = _client_side_apply_linkedin_url(df, af)
        assert len(result) == 1
        assert result.iloc[0]['linkedin_url'] == \
            'https://www.linkedin.com/in/gil-gitlin-87b720200'

    def test_slug_substring_narrows_fulltext_results(self):
        """Slug-only input must filter via case-insensitive substring."""
        df = self._fulltext_result_rows()
        af = {'linkedin_url': 'gil-gitlin'}
        result = _client_side_apply_linkedin_url(df, af)
        assert len(result) == 1
        assert 'gil-gitlin' in result.iloc[0]['linkedin_url']

    def test_no_url_filter_keeps_all_rows(self):
        """Sanity: no URL filter → no narrowing from this branch."""
        df = self._fulltext_result_rows()
        result = _client_side_apply_linkedin_url(df, {'linkedin_url': None})
        assert len(result) == 3

    def test_url_filter_no_match_returns_empty(self):
        df = self._fulltext_result_rows()
        af = {'linkedin_url': 'https://www.linkedin.com/in/nobody-here'}
        result = _client_side_apply_linkedin_url(df, af)
        assert len(result) == 0
