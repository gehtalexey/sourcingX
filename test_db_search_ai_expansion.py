"""
Regression tests for the AI field-expansion helpers used by Tab 0 (Crustdata
People DB Search) and Tab 7 (Database / Supabase search).

The helpers were extracted to module level in dashboard.py so both tabs can
share the same parametric logic (they differ only by session-state key
prefix). These tests cover:

1. ``_ai_effective_value`` correctly merges manual input + AI multiselect
   selections with both Tab 0 (``crust_search_*``) and Tab 7 (``db_f_*``)
   key prefixes. The function must be agnostic to the prefix.
2. ``_ai_expand_callback`` round-trips an AI-mocked expansion into
   ``db_expanded_<field>`` and pre-populates ``sel_db_expanded_<field>``.
3. ``_ai_add_custom_value_callback`` appends user-typed custom values to the
   expanded list and the selection.
4. ``_ai_clear_expanded_callback`` removes only the expanded list.

We mock ``expand_variations`` so no real OpenAI call happens, and we replace
``st.session_state`` with a plain dict to keep the tests deterministic and
independent of Streamlit's session runtime.
"""

from unittest.mock import patch, MagicMock

import pytest

import dashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeSessionState(dict):
    """Plain dict that mimics ``st.session_state`` for the bits the helpers use."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = value


@pytest.fixture
def fake_session():
    """Patch dashboard.st.session_state with a plain dict for the duration of a test."""
    fake = _FakeSessionState()
    with patch.object(dashboard.st, 'session_state', fake):
        yield fake


# ---------------------------------------------------------------------------
# _ai_effective_value
# ---------------------------------------------------------------------------

class TestEffectiveValue:
    """Pins the merge behaviour expected by both Tab 0 and Tab 7."""

    def test_returns_empty_when_neither_input_nor_selection(self, fake_session):
        assert dashboard._ai_effective_value("db_f_current_title", "db_expanded_current_title") == ""

    def test_returns_manual_input_only(self, fake_session):
        fake_session["db_f_current_title"] = "software developer"
        assert dashboard._ai_effective_value(
            "db_f_current_title", "db_expanded_current_title"
        ) == "software developer"

    def test_returns_selection_only(self, fake_session):
        fake_session["sel_db_expanded_current_title"] = ["software engineer", "swe"]
        out = dashboard._ai_effective_value("db_f_current_title", "db_expanded_current_title")
        assert out == "software engineer, swe"

    def test_merges_manual_and_selection_with_dedupe(self, fake_session):
        """Real recruiter case: user types 'software developer', AI suggests
        'software engineer' and 'software developer', user keeps both. The
        merged string must dedupe (case-insensitive) and preserve order."""
        fake_session["db_f_current_title"] = "software developer, Backend"
        fake_session["sel_db_expanded_current_title"] = [
            "software engineer", "software developer", "BACKEND",
        ]
        out = dashboard._ai_effective_value("db_f_current_title", "db_expanded_current_title")
        # Manual first (in input order), then unique selections
        assert out == "software developer, Backend, software engineer"

    def test_works_with_tab0_key_prefix(self, fake_session):
        """The helper must remain parameter-agnostic — same logic, different
        key namespace, no Tab 0 regression."""
        fake_session["crust_search_title"] = "team leader"
        fake_session["sel_expanded_titles"] = ["engineering manager", "team lead"]
        out = dashboard._ai_effective_value("crust_search_title", "expanded_titles")
        assert out == "team leader, engineering manager, team lead"

    def test_works_with_tab7_key_prefix(self, fake_session):
        """Tab 7 uses ``db_f_*`` for inputs and ``db_expanded_*`` for AI lists."""
        fake_session["db_f_current_company"] = "Wix"
        fake_session["sel_db_expanded_current_company"] = ["Wix.com", "Wix Studio"]
        out = dashboard._ai_effective_value("db_f_current_company", "db_expanded_current_company")
        assert out == "Wix, Wix.com, Wix Studio"


# ---------------------------------------------------------------------------
# _ai_expand_callback
# ---------------------------------------------------------------------------

class TestExpandCallback:
    """The expand callback must call the LLM mock and land results in the
    expected session-state keys for Tab 7."""

    def test_no_op_without_api_key(self, fake_session):
        fake_session["db_f_current_title"] = "software developer"
        with patch.object(dashboard, 'expand_variations') as mock_expand:
            dashboard._ai_expand_callback(
                "db_f_current_title", "db_expanded_current_title", "title", openai_api_key=""
            )
        mock_expand.assert_not_called()
        assert "db_expanded_current_title" not in fake_session

    def test_no_op_without_input(self, fake_session):
        with patch.object(dashboard, 'expand_variations') as mock_expand:
            dashboard._ai_expand_callback(
                "db_f_current_title", "db_expanded_current_title", "title",
                openai_api_key="test-key",
            )
        mock_expand.assert_not_called()
        assert "db_expanded_current_title" not in fake_session

    def test_expansion_lands_in_db_expanded_key(self, fake_session):
        """The whole point of the wiring: user typed 'software developer',
        AI returns variants, multiselect is pre-populated with everything
        selected so the user can deselect if needed."""
        fake_session["db_f_current_title"] = "software developer"
        with patch.object(dashboard, 'expand_variations', return_value=[
            "software developer", "software engineer", "swe", "full stack engineer",
        ]):
            with patch.object(dashboard, 'HAS_CRUSTDATA_SEARCH', True):
                dashboard._ai_expand_callback(
                    "db_f_current_title", "db_expanded_current_title", "title",
                    openai_api_key="test-key",
                )

        # Suggestions stored under the Tab-7 namespaced key
        assert fake_session["db_expanded_current_title"] == [
            "software developer", "software engineer", "swe", "full stack engineer",
        ]
        # Multiselect default selection mirrors the full list
        assert fake_session["sel_db_expanded_current_title"] == [
            "software developer", "software engineer", "swe", "full stack engineer",
        ]

    def test_multiple_manual_terms_each_get_expanded(self, fake_session):
        """Comma-separated manual input means "expand each term"."""
        fake_session["db_f_current_company"] = "Wix, Monday"
        call_log = []
        def fake_expand(term, field_type, openai_api_key, **kwargs):
            call_log.append(term)
            return [f"{term} variant"]
        with patch.object(dashboard, 'expand_variations', side_effect=fake_expand):
            with patch.object(dashboard, 'HAS_CRUSTDATA_SEARCH', True):
                dashboard._ai_expand_callback(
                    "db_f_current_company", "db_expanded_current_company", "company",
                    openai_api_key="test-key",
                )
        assert call_log == ["Wix", "Monday"]
        # Merged list contains manual terms + AI variants, deduped
        assert "Wix" in fake_session["db_expanded_current_company"]
        assert "Wix variant" in fake_session["db_expanded_current_company"]
        assert "Monday" in fake_session["db_expanded_current_company"]
        assert "Monday variant" in fake_session["db_expanded_current_company"]


# ---------------------------------------------------------------------------
# _ai_add_custom_value_callback and _ai_clear_expanded_callback
# ---------------------------------------------------------------------------

class TestCustomAndClearCallbacks:

    def test_add_custom_appends_to_expanded_and_selection(self, fake_session):
        fake_session["db_expanded_location"] = ["Tel Aviv"]
        fake_session["sel_db_expanded_location"] = ["Tel Aviv"]
        fake_session["custom_db_expanded_location"] = "Ramat Gan, Herzliya"

        dashboard._ai_add_custom_value_callback(
            "custom_db_expanded_location", "db_expanded_location"
        )

        assert fake_session["db_expanded_location"] == ["Tel Aviv", "Ramat Gan", "Herzliya"]
        assert fake_session["sel_db_expanded_location"] == ["Tel Aviv", "Ramat Gan", "Herzliya"]
        # Custom-input text field is cleared after add
        assert fake_session["custom_db_expanded_location"] == ""

    def test_add_custom_skips_case_insensitive_duplicates(self, fake_session):
        fake_session["db_expanded_location"] = ["Tel Aviv"]
        fake_session["sel_db_expanded_location"] = ["Tel Aviv"]
        fake_session["custom_db_expanded_location"] = "tel aviv, Herzliya"

        dashboard._ai_add_custom_value_callback(
            "custom_db_expanded_location", "db_expanded_location"
        )
        assert fake_session["db_expanded_location"] == ["Tel Aviv", "Herzliya"]

    def test_clear_removes_expanded_list(self, fake_session):
        fake_session["db_expanded_current_title"] = ["a", "b"]
        fake_session["sel_db_expanded_current_title"] = ["a"]
        dashboard._ai_clear_expanded_callback("db_expanded_current_title")
        assert "db_expanded_current_title" not in fake_session
        # The selection key is intentionally untouched so a subsequent
        # expansion respects a previous user choice if Streamlit replays it.
        assert "sel_db_expanded_current_title" in fake_session
