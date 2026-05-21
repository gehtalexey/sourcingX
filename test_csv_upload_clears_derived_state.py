"""Regression tests for the CSV/JSON upload flow on the Load tab.

PR #45 added an append-vs-replace prompt when profiles are already loaded.
The commit must clear all derived session state (passed_candidates_df,
enriched_df, filter funnel stats, etc.) — otherwise downstream tabs keep
serving stale data after the user uploads a new file. The Enrich tab in
particular prefers passed_candidates_df when present, so leaving it
silently ignores the freshly uploaded set.
"""

import pandas as pd
import pytest

from dashboard import (
    _RESULTS_DERIVED_KEYS,
    clear_results_derived_state,
)


# ---------------------------------------------------------------------------
# Helper-level: contract of clear_results_derived_state
# ---------------------------------------------------------------------------


def test_clears_every_derived_key():
    """Every key in _RESULTS_DERIVED_KEYS must be removed."""
    state = {key: f"stale-{key}" for key in _RESULTS_DERIVED_KEYS}
    state["results_df"] = pd.DataFrame([{"linkedin_url": "https://x"}])
    state["original_results_df"] = state["results_df"].copy()
    state["unrelated_setting"] = "keep me"

    clear_results_derived_state(state)

    for key in _RESULTS_DERIVED_KEYS:
        assert key not in state, f"{key} should have been cleared"
    # The canonical results data and unrelated settings must NOT be cleared.
    assert "results_df" in state
    assert "original_results_df" in state
    assert state["unrelated_setting"] == "keep me"


def test_clear_is_idempotent_on_empty_state():
    """Calling on a session with no derived state must not raise."""
    state = {}
    clear_results_derived_state(state)
    assert state == {}


def test_clear_handles_partial_state():
    """Only some derived keys present — partial removal must still work."""
    state = {
        "passed_candidates_df": pd.DataFrame([{"a": 1}]),
        "enriched_df": pd.DataFrame([{"b": 2}]),
        "results_df": pd.DataFrame([{"c": 3}]),
    }
    clear_results_derived_state(state)
    assert "passed_candidates_df" not in state
    assert "enriched_df" not in state
    assert "results_df" in state  # canonical data preserved


# ---------------------------------------------------------------------------
# Coverage: the named derived keys actually exist in the codebase
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "passed_candidates_df",   # Filter+ output, preferred by Enrich tab
        "enriched_df",            # Enrich tab output
        "enriched_profiles_raw",  # raw profile cache keyed by URL
        "filter_stats",           # Filter tab funnel
        # f2_filter_stats / f2_filtered_out were intentionally removed when
        # the Advanced Filtering ("Filter+") section was merged into the
        # main Filter tab and deleted (commit a5bbea9). Don't re-add here.
    ],
)
def test_known_stale_keys_are_in_clear_list(key):
    """Lock the canonical derived keys we know matter into the clear list.

    If somebody removes one of these without updating
    _RESULTS_DERIVED_KEYS, the Enrich/Filter+/Screen tabs will start
    silently serving stale data again.
    """
    assert key in _RESULTS_DERIVED_KEYS
