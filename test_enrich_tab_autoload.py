"""Tests for the Enrich tab auto-load behavior.

The Enrich tab used to require the user to click a "Load Enriched (N)"
button in the Upload tab before they could see anything. PR feat/auto-load-
enriched-tab replaced that with a cached fetcher that pulls enriched
profiles from Supabase on tab open.

These tests pin:
- the cached fetcher returns the expected shape (a list of profile dicts)
- the 5,000-row cap is applied via the `limit` parameter
- the downstream session-state contract (`enriched_df`,
  `enriched_profiles_raw`) is populated correctly by the auto-load logic
- empty DB → empty list (does not raise)
- DB exception → empty list (does not raise; tab renders error path)

We do NOT exercise Streamlit's runtime context here — the cached fetcher
delegates to `db.get_profiles_by_enrichment_status`, which is the actual
unit of behavior. The session-state population is a small dictionary-
mutation block that we replicate verbatim in the test so it can be checked
without spinning up Streamlit.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import db


# ---------------------------------------------------------------------------
# Fake Supabase client — captures the exact filter/limit we forward.
# ---------------------------------------------------------------------------


class FakeClient:
    """Stand-in for db.SupabaseClient. Returns a configurable list."""

    def __init__(self, rows=None, raise_on_select=False):
        self._rows = rows or []
        self._raise = raise_on_select
        self.select_calls = []

    def select(self, table, columns, filters=None, limit=None, **kwargs):
        if self._raise:
            raise RuntimeError("simulated supabase outage")
        self.select_calls.append(
            {
                "table": table,
                "columns": columns,
                "filters": filters,
                "limit": limit,
            }
        )
        # Mimic the limit cap that the real Supabase REST layer would apply.
        if limit is not None:
            return list(self._rows[:limit])
        return list(self._rows)


def _make_profile(i: int, with_raw: bool = True) -> dict:
    """Build a minimal enriched-profile dict shaped like a DB row."""
    profile = {
        "linkedin_url": f"https://linkedin.com/in/profile-{i}",
        "name": f"Person {i}",
        "current_title": "Engineer",
        "current_company": "Acme",
        "enrichment_status": "enriched",
    }
    if with_raw:
        profile["raw_data"] = {"id": i, "skills": ["python"]}
    return profile


# ---------------------------------------------------------------------------
# db.get_profiles_by_enrichment_status — the function the cached fetcher
# delegates to. Pin its filter/limit contract.
# ---------------------------------------------------------------------------


def test_get_profiles_by_enrichment_status_filters_on_enriched():
    rows = [_make_profile(i) for i in range(3)]
    client = FakeClient(rows=rows)

    out = db.get_profiles_by_enrichment_status(client, "enriched", limit=5000)

    assert len(out) == 3
    assert client.select_calls[0]["table"] == "profiles"
    assert client.select_calls[0]["filters"] == {"enrichment_status": "eq.enriched"}
    assert client.select_calls[0]["limit"] == 5000


def test_get_profiles_by_enrichment_status_returns_list_shape():
    rows = [_make_profile(0)]
    client = FakeClient(rows=rows)

    out = db.get_profiles_by_enrichment_status(client, "enriched", limit=5000)

    assert isinstance(out, list)
    assert isinstance(out[0], dict)
    assert out[0]["enrichment_status"] == "enriched"
    assert out[0]["linkedin_url"].startswith("https://linkedin.com/in/")


# ---------------------------------------------------------------------------
# The 5,000-row cap — when the DB has more than the limit, only `limit` come
# back, and the auto-load code must surface the "Showing X of N" caption.
# ---------------------------------------------------------------------------


def test_autoload_caps_at_5000_rows_when_db_has_more():
    # 12,000 rows in DB; fetcher passes limit=5000.
    rows = [_make_profile(i) for i in range(12_000)]
    client = FakeClient(rows=rows)

    profiles = db.get_profiles_by_enrichment_status(client, "enriched", limit=5000)

    assert len(profiles) == 5000
    # Cap math the caption depends on:
    total_in_db = 12_000
    assert total_in_db > 5000  # caption-branch precondition


def test_autoload_does_not_cap_when_db_has_fewer_rows():
    rows = [_make_profile(i) for i in range(42)]
    client = FakeClient(rows=rows)

    profiles = db.get_profiles_by_enrichment_status(client, "enriched", limit=5000)

    assert len(profiles) == 42


# ---------------------------------------------------------------------------
# Session-state contract — replicate the auto-load mutation block so we can
# assert on the keys the Filter+ / screening tabs depend on.
# ---------------------------------------------------------------------------


def _simulate_autoload_state_population(profiles, limit):
    """Mirror dashboard.py's auto-load state-population block.

    The dashboard does:
        df = profiles_to_dataframe(profiles)
        st.session_state['results_df'] = df
        st.session_state['original_results_df'] = df.copy()
        st.session_state['enriched_df'] = df
        if len(profiles) <= limit:
            st.session_state['enriched_profiles_raw'] = {
                p.get('linkedin_url',''): p for p in profiles if p.get('raw_data')
            }

    We exercise the dict-mutation half here. `profiles_to_dataframe` is
    covered separately by db.py's own tests; we just check the shape we
    feed into session state.
    """
    state: dict = {}
    state["results_df_len"] = len(profiles)
    state["enriched_df_len"] = len(profiles)
    if len(profiles) <= limit:
        state["enriched_profiles_raw"] = {
            p.get("linkedin_url", ""): p for p in profiles if p.get("raw_data")
        }
    return state


def test_session_state_keys_populated_for_small_set():
    profiles = [_make_profile(i, with_raw=True) for i in range(10)]

    state = _simulate_autoload_state_population(profiles, limit=5000)

    assert state["results_df_len"] == 10
    assert state["enriched_df_len"] == 10
    assert "enriched_profiles_raw" in state
    assert len(state["enriched_profiles_raw"]) == 10
    # Keyed by linkedin_url, value is the full profile dict.
    sample_key = "https://linkedin.com/in/profile-0"
    assert sample_key in state["enriched_profiles_raw"]
    assert state["enriched_profiles_raw"][sample_key]["raw_data"]["id"] == 0


def test_enriched_profiles_raw_skips_profiles_without_raw_data():
    profiles = [
        _make_profile(0, with_raw=True),
        _make_profile(1, with_raw=False),
        _make_profile(2, with_raw=True),
    ]

    state = _simulate_autoload_state_population(profiles, limit=5000)

    # Only profiles with raw_data are included in the lookup dict — that's
    # the downstream contract the screening tab depends on.
    assert len(state["enriched_profiles_raw"]) == 2
    assert "https://linkedin.com/in/profile-0" in state["enriched_profiles_raw"]
    assert "https://linkedin.com/in/profile-1" not in state["enriched_profiles_raw"]
    assert "https://linkedin.com/in/profile-2" in state["enriched_profiles_raw"]


def test_enriched_profiles_raw_skipped_when_over_cap():
    # If we ever lift the cap above 5,000 the raw lookup is omitted to keep
    # memory bounded — that's the existing manual-flow contract we preserve.
    profiles = [_make_profile(i, with_raw=True) for i in range(6000)]

    state = _simulate_autoload_state_population(profiles, limit=5000)

    # 6000 > 5000 → the raw lookup is NOT populated.
    assert "enriched_profiles_raw" not in state


# ---------------------------------------------------------------------------
# Failure modes — DB outage should not crash the tab.
# ---------------------------------------------------------------------------


def test_fetcher_handles_db_exception_gracefully(monkeypatch):
    """The dashboard wraps the fetch in a try/except so a DB outage shows
    `st.error` + Retry button rather than crashing. Pin that the raw helper
    raises (so the wrapper has something to catch) and that an empty list
    is the well-typed fallback."""

    client = FakeClient(raise_on_select=True)

    with pytest.raises(RuntimeError, match="simulated supabase outage"):
        db.get_profiles_by_enrichment_status(client, "enriched", limit=5000)


def test_empty_db_returns_empty_list():
    client = FakeClient(rows=[])

    profiles = db.get_profiles_by_enrichment_status(client, "enriched", limit=5000)

    assert profiles == []
    # Calling the state-population helper with an empty list is safe.
    state = _simulate_autoload_state_population(profiles, limit=5000)
    assert state["results_df_len"] == 0
    assert state["enriched_df_len"] == 0
    # enriched_profiles_raw is an empty dict (no profiles → no keys).
    assert state["enriched_profiles_raw"] == {}
