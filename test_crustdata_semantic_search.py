"""Tests for the natural-language ("search by description") people search.

Crustdata's beta v2025-11-01 `/person/search` semantic mode ranks people by
how well their whole profile matches a plain-language query, instead of
exact filter conditions. It's a different endpoint, auth header, and
response shape than the legacy `/screener/persondb/search` the rest of this
module talks to — see the comment above CRUSTDATA_SEMANTIC_SEARCH_ENDPOINT
in crustdata_search.py.

These tests cover two things without any real network calls:
1. search_people_semantic() sends the request Crustdata's docs specify
   (docs.crustdata.com/person-docs/search/introduction, verified 2026-07-20).
2. semantic_profile_to_legacy_shape() adapts a nested v2 profile into the
   flat shape the rest of the pipeline (normalize_search_result, the Search
   tab results table) already knows how to read, so no parallel code path
   is needed downstream.
"""

from unittest.mock import MagicMock, patch

import pytest

from crustdata_search import (
    search_people_semantic,
    semantic_profile_to_legacy_shape,
    normalize_search_results_to_df,
    CRUSTDATA_SEMANTIC_SEARCH_ENDPOINT,
    CRUSTDATA_API_VERSION,
)


def _mock_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.headers = {}
    return resp


# ---------------------------------------------------------------------------
# search_people_semantic — request shape
# ---------------------------------------------------------------------------


class TestSemanticSearchRequest:
    def test_sends_bearer_auth_and_version_header(self):
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                json_data={"profiles": [], "total_count": 0}
            )
            search_people_semantic("founding engineers", api_key="test-key")

            _, kwargs = mock_post.call_args
            headers = kwargs["headers"]
            assert headers["Authorization"] == "Bearer test-key"
            assert headers["x-api-version"] == CRUSTDATA_API_VERSION
            # Legacy endpoint uses "Token <key>" — must not leak in here.
            assert "Token" not in headers["Authorization"]

    def test_posts_to_the_v2_person_search_endpoint(self):
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                json_data={"profiles": [], "total_count": 0}
            )
            search_people_semantic("founding engineers", api_key="test-key")

            args, _ = mock_post.call_args
            assert args[0] == CRUSTDATA_SEMANTIC_SEARCH_ENDPOINT
            assert CRUSTDATA_SEMANTIC_SEARCH_ENDPOINT.endswith("/person/search")

    def test_query_and_mode_go_under_nested_search_object(self):
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                json_data={"profiles": [], "total_count": 0}
            )
            search_people_semantic(
                "senior backend engineers in Tel Aviv",
                search_mode="lexical",
                api_key="test-key",
            )

            _, kwargs = mock_post.call_args
            body = kwargs["json"]
            assert body["search"] == {
                "query": "senior backend engineers in Tel Aviv",
                "mode": "lexical",
            }
            # Default recall_mode ("managed") must NOT add a top-level "mode" key —
            # only "exact" does. Sending both would be redundant with search.mode.
            assert "mode" not in body

    def test_exact_recall_mode_sets_top_level_mode(self):
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                json_data={"profiles": [], "total_count": 0}
            )
            search_people_semantic(
                "founding engineers", recall_mode="exact", api_key="test-key"
            )

            _, kwargs = mock_post.call_args
            assert kwargs["json"]["mode"] == "exact"

    def test_limit_is_clamped_to_1_100(self):
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                json_data={"profiles": [], "total_count": 0}
            )
            search_people_semantic("x", limit=500, api_key="test-key")
            assert mock_post.call_args.kwargs["json"]["limit"] == 100

            search_people_semantic("x", limit=0, api_key="test-key")
            assert mock_post.call_args.kwargs["json"]["limit"] == 1

    def test_cursor_included_when_provided(self):
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                json_data={"profiles": [], "total_count": 0}
            )
            search_people_semantic("x", cursor="abc123", api_key="test-key")
            assert mock_post.call_args.kwargs["json"]["cursor"] == "abc123"

    def test_empty_query_raises_without_calling_api(self):
        with patch("crustdata_search.requests.post") as mock_post:
            with pytest.raises(ValueError):
                search_people_semantic("   ", api_key="test-key")
            mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# search_people_semantic — response parsing
# ---------------------------------------------------------------------------


class TestSemanticSearchResponse:
    def test_parses_profiles_cursor_and_credits(self):
        profiles = [{"crustdata_person_id": 1}, {"crustdata_person_id": 2}]
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                json_data={
                    "profiles": profiles,
                    "next_cursor": "next-page",
                    "total_count": 1250,
                }
            )
            result = search_people_semantic("x", api_key="test-key")

        assert result["profiles"] == profiles
        assert result["cursor"] == "next-page"
        assert result["total_count"] == 1250
        # 0.03 credits per result returned, not per 100.
        assert result["credits_used"] == pytest.approx(0.06)

    def test_401_raises_authentication_error(self):
        from error_handling import AuthenticationError

        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(status_code=401)
            with pytest.raises(AuthenticationError):
                search_people_semantic("x", api_key="bad-key")


# ---------------------------------------------------------------------------
# semantic_profile_to_legacy_shape
# ---------------------------------------------------------------------------


_SAMPLE_V2_PROFILE = {
    "crustdata_person_id": 123,
    "fit": "strong",
    "basic_profile": {
        "name": "Jane Doe",
        "headline": "Founding Engineer",
        "summary": "Builds developer tools.",
        "location": {"raw": "Tel Aviv, Israel", "city": "Tel Aviv", "country": "Israel"},
    },
    "experience": {
        "employment_details": {
            "current": [
                {
                    "name": "Acme",
                    "title": "Founding Engineer",
                    "seniority_level": "Senior",
                    "company_headcount_latest": "11-50",
                    "start_date": "2023-01-01",
                }
            ],
            "past": [
                {
                    "name": "OldCo",
                    "title": "Engineer",
                    "start_date": "2019-01-01",
                    "end_date": "2022-12-31",
                }
            ],
        }
    },
    "education": {"schools": [{"school": "Technion"}]},
    "skills": {"professional_network_skills": ["Python", "Kubernetes"]},
    "professional_network": {"connections": 500},
    "social_handles": {
        "professional_network_identifier": {
            "profile_url": "https://www.linkedin.com/in/janedoe"
        }
    },
}


class TestSemanticProfileShim:
    def test_maps_basic_fields(self):
        shim = semantic_profile_to_legacy_shape(_SAMPLE_V2_PROFILE)
        assert shim["name"] == "Jane Doe"
        assert shim["headline"] == "Founding Engineer"
        assert shim["region"] == "Tel Aviv, Israel"
        assert shim["location_country"] == "Israel"
        assert shim["summary"] == "Builds developer tools."
        assert shim["flagship_profile_url"] == "https://www.linkedin.com/in/janedoe"
        assert shim["num_of_connections"] == 500
        assert shim["skills"] == ["Python", "Kubernetes"]
        assert shim["all_schools"] == ["Technion"]
        assert shim["crustdata_person_id"] == 123
        assert shim["_fit"] == "strong"
        assert shim["_semantic_incomplete"] is True

    def test_aliases_company_headcount_latest_to_range(self):
        """The legacy display/normalize code reads company_headcount_range;
        the new API returns company_headcount_latest. Must be bridged so
        company size still shows up without touching that shared code."""
        shim = semantic_profile_to_legacy_shape(_SAMPLE_V2_PROFILE)
        current = shim["current_employers"][0]
        assert current["company_headcount_range"] == "11-50"

    def test_past_employers_preserved(self):
        shim = semantic_profile_to_legacy_shape(_SAMPLE_V2_PROFILE)
        assert shim["past_employers"][0]["name"] == "OldCo"
        assert shim["past_employers"][0]["title"] == "Engineer"

    def test_empty_profile_returns_empty_dict(self):
        assert semantic_profile_to_legacy_shape({}) == {}
        assert semantic_profile_to_legacy_shape(None) == {}

    def test_missing_sections_do_not_raise(self):
        # A minimal profile missing most nested sections should still shim
        # cleanly instead of raising KeyError/AttributeError.
        shim = semantic_profile_to_legacy_shape({"crustdata_person_id": 9, "fit": "weak"})
        assert shim["name"] == ""
        assert shim["current_employers"] == []
        assert shim["skills"] == []
        assert shim["_fit"] == "weak"

    def test_shimmed_profile_normalizes_into_pipeline_dataframe(self):
        """End-to-end: the shim's output must flow through the existing
        normalize_search_results_to_df() unchanged, so description-search
        results reuse the same Filter/Screen pipeline as filter search."""
        shim = semantic_profile_to_legacy_shape(_SAMPLE_V2_PROFILE)
        df = normalize_search_results_to_df([shim])

        assert len(df) == 1
        row = df.iloc[0]
        assert row["linkedin_url"] == "https://www.linkedin.com/in/janedoe"
        assert row["name"] == "Jane Doe"
        assert row["current_title"] == "Founding Engineer"
        assert row["current_company"] == "Acme"
        assert row["seniority"] == "Senior"
        assert row["company_size"] == "11-50"
        assert row["skills"] == "Python, Kubernetes"

    def test_shimmed_profile_is_flagged_as_needing_enrichment(self):
        """Description search doesn't return skills/summary/experience, and
        the AI screening prompt treats missing skills as a hard FAIL rather
        than "unknown". Rows from it must be flagged so they land in the
        normal enrichment queue instead of going straight to AI Screen."""
        shim = semantic_profile_to_legacy_shape(_SAMPLE_V2_PROFILE)
        df = normalize_search_results_to_df([shim])
        assert bool(df.iloc[0]["_needs_enrichment"]) is True

    def test_regular_filter_search_profile_still_marked_complete(self):
        """A normal (non-semantic) search result has no _semantic_incomplete
        marker and must keep _needs_enrichment=False, as before — this
        change must not affect the existing filter search path."""
        filter_profile = {
            "name": "John Smith",
            "flagship_profile_url": "https://www.linkedin.com/in/johnsmith",
        }
        df = normalize_search_results_to_df([filter_profile])
        assert bool(df.iloc[0]["_needs_enrichment"]) is False
