"""Regression tests for the Tab 7 DB Search "Current Company" filter.

User report (2026-05-11): typing ``elad, egged, harel, gav, log-on`` into the
Current Company box on Tab 7 returned a Celadon Books profile
(https://www.linkedin.com/in/jaime-noven-147ab113/) because ``elad`` was a
substring of ``Celadon``. The old PostgREST query was
``current_company.ilike.*elad*`` — naive substring match, no word boundaries.

The fix moves the Current Company filter onto PostgREST's ``imatch`` operator
(case-insensitive POSIX regex) with ``\\y`` word-boundary anchors around each
term. ``parse_boolean_query_word_boundary`` in ``db.py`` encapsulates that.

These tests pin:
  1. The bug — ``elad`` must NOT match ``Celadon Books``.
  2. The happy path — ``elad`` should still match ``Elad Systems``,
     ``Elad Software Systems``, and ``Elad`` (exact).
  3. Multi-term — ``elad, egged, harel`` should match profiles at any of
     those companies but NOT ``Celadon``.
  4. The DB query itself uses word-boundary regex (so we don't fetch
     thousands of false-positive rows over the network just to filter them
     out client-side).

We test at two levels:

  a) ``parse_boolean_query_word_boundary`` directly — pure-function check
     that the emitted PostgREST filter string looks right and, when
     re-interpreted in Python against a candidate company name, gives the
     correct verdict.

  b) ``search_profiles_boolean`` end-to-end with a mocked Supabase client.
     We assert the request params carry ``imatch``/``\\y`` (not ``ilike``)
     for the ``current_company`` column, and simulate the server returning
     only rows whose ``current_company`` actually matches the regex.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

import db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# PostgreSQL's POSIX ``\y`` word-boundary is equivalent to Python's ``\b`` for
# the ASCII inputs we get from LinkedIn company names. We re-implement the
# server-side match in Python so the tests stay hermetic (no live DB).
def _server_side_match(current_company: str, term: str) -> bool:
    """Emulate a single PostgREST ``current_company.imatch.\\yterm\\y`` clause."""
    pattern = r'\b' + re.escape(term) + r'\b'
    return re.search(pattern, current_company, flags=re.IGNORECASE) is not None


def _simulate_query(filter_string: str, rows: list[dict]) -> list[dict]:
    """Given a filter string from ``parse_boolean_query_word_boundary`` and a
    list of candidate row dicts (each with a ``current_company`` field), return
    the rows that the server would actually have sent back.

    The filter string from ``parse_boolean_query_word_boundary`` is shaped
    like:
        - ``current_company.imatch.\\yelad\\y``                          (single term)
        - ``or(current_company.imatch.\\yelad\\y,current_company.imatch.\\yegged\\y)``  (multi-term)

    We only need to handle the shapes the helper actually produces, plus NOT.
    """
    # Extract each leaf imatch clause and evaluate it. The filter string
    # contains a literal backslash + ``y`` for each word boundary, so we
    # match a single escaped backslash in the source pattern.
    leaf_re = re.compile(r'current_company\.(not\.)?imatch\.\\y(.+?)\\y')
    # We accept both single-leaf and or(...) wrapped strings; split top-level.

    def matches_leaf(row: dict, leaf: str) -> bool:
        m = leaf_re.fullmatch(leaf)
        if not m:
            return False
        negated, term = m.groups()
        # Unescape regex specials we added so we can compare against the
        # original literal term. For the test inputs (alnum + hyphen) this
        # is a no-op, but it keeps the simulator honest.
        term = re.sub(r'\\(.)', r'\1', term)
        hit = _server_side_match(row['current_company'], term)
        return (not hit) if negated else hit

    if filter_string.startswith('or('):
        inner = filter_string[3:-1]
        leaves = inner.split(',')
        return [r for r in rows if any(matches_leaf(r, leaf) for leaf in leaves)]
    if filter_string.startswith('and('):
        inner = filter_string[4:-1]
        leaves = inner.split(',')
        return [r for r in rows if all(matches_leaf(r, leaf) for leaf in leaves)]
    # Single leaf
    return [r for r in rows if matches_leaf(r, filter_string)]


# ---------------------------------------------------------------------------
# parse_boolean_query_word_boundary — emits word-boundary regex
# ---------------------------------------------------------------------------

class TestEmitsWordBoundaryRegex:
    """Pin the PostgREST filter shape so future refactors can't silently
    fall back to ``ilike`` substring matching."""

    def test_single_term_uses_imatch_and_word_boundary(self):
        out = db.parse_boolean_query_word_boundary('elad', 'current_company')
        # \\y in the filter string is the literal text we send to PostgREST,
        # which sends the backslash-y POSIX word-boundary anchor to Postgres.
        assert out == r'current_company.imatch.\yelad\y'
        # Critically — NOT an ilike substring match.
        assert '.ilike.' not in out

    def test_multiple_terms_or_grouped(self):
        out = db.parse_boolean_query_word_boundary('elad, egged, harel', 'current_company')
        assert out.startswith('or(')
        assert out.endswith(')')
        assert r'current_company.imatch.\yelad\y' in out
        assert r'current_company.imatch.\yegged\y' in out
        assert r'current_company.imatch.\yharel\y' in out
        assert '.ilike.' not in out

    def test_term_with_hyphen_preserved(self):
        # "log-on" is one of the user's real filter terms.
        out = db.parse_boolean_query_word_boundary('log-on', 'current_company')
        assert r'\ylog-on\y' in out or r'\ylog\-on\y' in out

    def test_empty_input_emits_empty(self):
        assert db.parse_boolean_query_word_boundary('', 'current_company') == ''


# ---------------------------------------------------------------------------
# Behavioural — the regex must reject Celadon and accept Elad variants
# ---------------------------------------------------------------------------

class TestCompanyMatchingBehaviour:
    """Server-side match simulation. These are the tests Alexey will care
    about: the bug repro and the expected positives."""

    @pytest.fixture
    def candidate_rows(self):
        # A small synthetic universe spanning the bug repro plus the
        # legitimate matches we don't want to lose.
        return [
            {'linkedin_url': 'jaime-noven',     'current_company': 'Celadon Books'},
            {'linkedin_url': 'elad-systems',    'current_company': 'Elad Systems'},
            {'linkedin_url': 'elad-software',   'current_company': 'Elad Software Systems'},
            {'linkedin_url': 'elad-exact',      'current_company': 'Elad'},
            {'linkedin_url': 'bank-elad',       'current_company': 'Bank Elad'},
            {'linkedin_url': 'egged-driver',    'current_company': 'Egged'},
            {'linkedin_url': 'harel-insurance', 'current_company': 'Harel Insurance'},
            {'linkedin_url': 'log-on-eng',      'current_company': 'Log-On Software'},
            {'linkedin_url': 'unrelated',       'current_company': 'Microsoft'},
        ]

    def test_elad_does_not_match_celadon(self, candidate_rows):
        """The exact user-reported false positive."""
        filter_str = db.parse_boolean_query_word_boundary('elad', 'current_company')
        matched = _simulate_query(filter_str, candidate_rows)
        urls = {r['linkedin_url'] for r in matched}
        assert 'jaime-noven' not in urls, (
            'Celadon Books leaked through — the substring fix did not take.'
        )

    def test_elad_matches_elad_variants(self, candidate_rows):
        filter_str = db.parse_boolean_query_word_boundary('elad', 'current_company')
        matched = _simulate_query(filter_str, candidate_rows)
        urls = {r['linkedin_url'] for r in matched}
        assert 'elad-systems' in urls
        assert 'elad-software' in urls
        assert 'elad-exact' in urls
        assert 'bank-elad' in urls

    def test_multiterm_matches_all_intended_but_not_celadon(self, candidate_rows):
        """User's real input: elad, egged, harel — must catch the three
        intended companies but still reject Celadon."""
        filter_str = db.parse_boolean_query_word_boundary(
            'elad, egged, harel', 'current_company'
        )
        matched = _simulate_query(filter_str, candidate_rows)
        urls = {r['linkedin_url'] for r in matched}
        assert 'elad-systems' in urls
        assert 'egged-driver' in urls
        assert 'harel-insurance' in urls
        assert 'jaime-noven' not in urls

    def test_log_on_matches_log_on_software(self, candidate_rows):
        """Hyphenated terms must still work (one of the user's real terms)."""
        filter_str = db.parse_boolean_query_word_boundary('log-on', 'current_company')
        matched = _simulate_query(filter_str, candidate_rows)
        urls = {r['linkedin_url'] for r in matched}
        assert 'log-on-eng' in urls


# ---------------------------------------------------------------------------
# End-to-end — search_profiles_boolean wires the word-boundary path
# ---------------------------------------------------------------------------

class TestSearchProfilesBooleanWiring:
    """Make sure ``search_profiles_boolean`` actually routes
    ``current_company`` through the word-boundary helper, not the legacy
    substring path."""

    def test_current_company_filter_goes_through_imatch(self):
        client = MagicMock()
        client.select.return_value = []
        db.search_profiles_boolean(client, {'current_company': 'elad'})

        assert client.select.called
        _, args, kwargs = client.select.mock_calls[0]
        # search_profiles_boolean is positional: (table, columns, params, limit=)
        # but tolerate either signature.
        params = args[2] if len(args) >= 3 else kwargs.get('filters') or kwargs.get('params')
        assert params is not None
        # Single-term path: param key is the column, value carries imatch + \y
        assert 'current_company' in params, params
        val = params['current_company']
        assert val.startswith('imatch.'), (
            f'Expected imatch operator, got {val!r}. Substring matching has '
            'returned — Celadon Books will leak again.'
        )
        assert r'\yelad\y' in val
        assert 'ilike' not in val

    def test_multiple_companies_wrapped_in_or(self):
        client = MagicMock()
        client.select.return_value = []
        db.search_profiles_boolean(client, {'current_company': 'elad, egged, harel'})

        _, args, kwargs = client.select.mock_calls[0]
        params = args[2] if len(args) >= 3 else kwargs.get('filters') or kwargs.get('params')
        # Multi-term path lifts the or(...) into the 'or' param
        assert 'or' in params, params
        or_clause = params['or']
        assert r'current_company.imatch.\yelad\y' in or_clause
        assert r'current_company.imatch.\yegged\y' in or_clause
        assert r'current_company.imatch.\yharel\y' in or_clause

    def test_other_string_columns_still_use_ilike(self):
        """Don't accidentally tighten title/location matching — only current
        company has the word-boundary requirement today."""
        client = MagicMock()
        client.select.return_value = []
        db.search_profiles_boolean(client, {'current_title': 'developer'})

        _, args, kwargs = client.select.mock_calls[0]
        params = args[2] if len(args) >= 3 else kwargs.get('filters') or kwargs.get('params')
        assert 'current_title' in params
        assert params['current_title'].startswith('ilike.'), params

    def test_full_repro_end_to_end(self):
        """The complete user repro: filter input ``elad, egged, harel, gav,
        log-on``, fake Supabase returns ONLY rows the regex actually matches,
        and we assert Celadon does not come back."""
        rows = [
            {'linkedin_url': 'jaime-noven',     'current_company': 'Celadon Books'},
            {'linkedin_url': 'elad-systems',    'current_company': 'Elad Systems'},
            {'linkedin_url': 'egged-driver',    'current_company': 'Egged'},
            {'linkedin_url': 'harel-insurance', 'current_company': 'Harel Insurance'},
            {'linkedin_url': 'log-on-eng',      'current_company': 'Log-On Software'},
            {'linkedin_url': 'gav-yam',         'current_company': 'Gav-Yam'},
            {'linkedin_url': 'microsoft',       'current_company': 'Microsoft'},
        ]

        # Build the filter the same way search_profiles_boolean would, then
        # use it to filter the candidate rows in-memory — that's our stand-in
        # for the Postgres regex engine.
        filter_str = db.parse_boolean_query_word_boundary(
            'elad, egged, harel, gav, log-on', 'current_company'
        )

        client = MagicMock()
        client.select.side_effect = lambda *a, **kw: _simulate_query(filter_str, rows)

        results = db.search_profiles_boolean(
            client, {'current_company': 'elad, egged, harel, gav, log-on'}
        )

        urls = {r['linkedin_url'] for r in results}
        # The bug: Celadon must NOT be in the result set.
        assert 'jaime-noven' not in urls, (
            'Regression: Celadon Books came back from a current_company '
            'filter of "elad" — word-boundary matching is broken.'
        )
        # Sanity: the legitimate matches still come back.
        assert {'elad-systems', 'egged-driver', 'harel-insurance',
                'log-on-eng', 'gav-yam'}.issubset(urls)
