"""Tests for the filter-based v2025-11-01 people search (search_people_db_v2)
and the legacy-column -> v2-field filter remap it depends on.

search_people_db_v2() is the only filter search the dashboard uses
(Crustdata stops serving every legacy /screener/* endpoint on 2026-09-30).
These tests also cover check_credits() on the new headers. No real network
calls: requests.post/get are patched throughout.
"""

from unittest.mock import MagicMock, patch

import pytest

from crustdata_search import (
    search_people_db_v2,
    build_filters,
    _remap_filters,
    _LEGACY_TO_V2_FIELD,
    CRUSTDATA_SEARCH_V2_ENDPOINT,
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
# _remap_filters — legacy `column` -> v2 `field`
# ---------------------------------------------------------------------------


class TestRemapFilters:
    def test_leaf_condition_column_renamed_to_field(self):
        remapped = _remap_filters(
            {"column": "current_employers.title", "type": "[.]", "value": "Engineer"}
        )
        assert remapped == {
            "field": "experience.employment_details.current.title",
            "type": "[.]",
            "value": "Engineer",
        }
        assert "column" not in remapped

    def test_nested_op_groups_recurse(self):
        tree = {
            "op": "and",
            "conditions": [
                {"column": "region", "type": "[.]", "value": "Israel"},
                {
                    "op": "or",
                    "conditions": [
                        {"column": "current_employers.title", "type": "[.]", "value": "Engineer"},
                        {"column": "headline", "type": "[.]", "value": "Engineer"},
                    ],
                },
            ],
        }
        remapped = _remap_filters(tree)
        assert remapped["conditions"][0]["field"] == "professional_network.location.raw"
        nested = remapped["conditions"][1]["conditions"]
        assert nested[0]["field"] == "experience.employment_details.current.title"
        assert nested[1]["field"] == "basic_profile.headline"

    def test_unmapped_column_passes_through_unchanged(self):
        remapped = _remap_filters({"column": "some_new_field", "type": "=", "value": 1})
        assert remapped["field"] == "some_new_field"

    def test_gte_lte_rewritten_to_arrow_operators(self):
        assert _remap_filters({"column": "x", "type": ">=", "value": 5})["type"] == "=>"
        assert _remap_filters({"column": "x", "type": "<=", "value": 5})["type"] == "=<"

    def test_every_build_filters_column_has_a_v2_mapping(self):
        """Enumerate every `column` build_filters() can actually emit (via a
        kitchen-sink call using every parameter) and assert each one has an
        entry in _LEGACY_TO_V2_FIELD — a filter added later without a
        mapping fails this test loudly instead of silently returning zero
        results in production. (Codex review, 2026-07-20 — the original
        mapping table only covered the commonly-used fields.)"""
        result = build_filters(
            title="Engineer",
            company="Google",
            location="Israel",
            seniority=["Senior"],
            headcount=["11-50"],
            experience_min=2,
            experience_max=10,
            skill_groups=["python, go"],
            keywords="kubernetes",
            past_companies="OldCo",
            past_titles="Junior Engineer",
            school="Technion",
            recently_changed_jobs=True,
            has_verified_email=True,
            function_categories=["Engineering"],
            industries=["Software Development"],
            country="ISR",
            continent="Asia",
            geo_city="Tel Aviv",
            geo_radius_km=50,
            min_connections=100,
            not_relevant_companies=["Bad Co"],
        )

        def _collect_columns(node, out):
            if not isinstance(node, dict):
                return
            if "conditions" in node:
                for c in node["conditions"]:
                    _collect_columns(c, out)
            elif "column" in node:
                out.add(node["column"])

        columns = set()
        _collect_columns(result.get("filters"), columns)

        assert columns, "kitchen-sink build_filters() call produced no conditions to check"
        unmapped = columns - set(_LEGACY_TO_V2_FIELD.keys())
        assert not unmapped, f"columns with no v2 field mapping: {unmapped}"


# ---------------------------------------------------------------------------
# search_people_db_v2 — request shape
# ---------------------------------------------------------------------------


class TestSearchV2Request:
    def test_sends_bearer_auth_and_version_header(self):
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(json_data={"profiles": [], "total_count": 0})
            search_people_db_v2({}, api_key="test-key")

            _, kwargs = mock_post.call_args
            headers = kwargs["headers"]
            assert headers["Authorization"] == "Bearer test-key"
            assert headers["x-api-version"] == CRUSTDATA_API_VERSION
            assert "Token" not in headers["Authorization"]

    def test_posts_to_v2_person_search_endpoint(self):
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(json_data={"profiles": [], "total_count": 0})
            search_people_db_v2({}, api_key="test-key")

            args, _ = mock_post.call_args
            assert args[0] == CRUSTDATA_SEARCH_V2_ENDPOINT

    def test_no_compact_param_sent(self):
        """compact is a legacy-only concept — v2 has no equivalent, must not
        be sent (Crustdata rejects unknown body keys on some endpoints)."""
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(json_data={"profiles": [], "total_count": 0})
            search_people_db_v2({}, api_key="test-key")
            assert "compact" not in mock_post.call_args.kwargs["json"]

    def test_filters_remapped_before_sending(self):
        legacy_filters = build_filters(title="Engineer")
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(json_data={"profiles": [], "total_count": 0})
            search_people_db_v2(legacy_filters, api_key="test-key")

            body = mock_post.call_args.kwargs["json"]
            # title filter is an OR group of two `[.]` conditions.
            sent_fields = {c["field"] for c in body["filters"]["conditions"]}
            assert "experience.employment_details.current.title" in sent_fields
            assert "basic_profile.headline" in sent_fields

    def test_sorts_use_field_not_column(self):
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(json_data={"profiles": [], "total_count": 0})
            search_people_db_v2({}, sorts=[{"column": "years_of_experience_raw", "order": "desc"}], api_key="test-key")
            body = mock_post.call_args.kwargs["json"]
            assert body["sorts"] == [{"field": "years_of_experience_raw", "order": "desc"}]

    def test_sort_field_value_is_remapped_not_just_the_key(self):
        """Regression test for a Codex-caught bug (2026-07-20): renaming
        `column` -> `field` isn't enough — the VALUE must go through the same
        _LEGACY_TO_V2_FIELD map filters use. The dashboard's default sort is
        `num_of_connections`, which has no v2 field of that name."""
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(json_data={"profiles": [], "total_count": 0})
            search_people_db_v2({}, sorts=[{"column": "num_of_connections", "order": "desc"}], api_key="test-key")
            body = mock_post.call_args.kwargs["json"]
            assert body["sorts"] == [{"field": "professional_network.connections", "order": "desc"}]

    def test_exclude_profiles_normalized_under_post_processing(self):
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(json_data={"profiles": [], "total_count": 0})
            search_people_db_v2({}, exclude_profiles=["https://www.linkedin.com/in/foo/"], api_key="test-key")
            body = mock_post.call_args.kwargs["json"]
            assert body["post_processing"]["exclude_profiles"] == ["https://www.linkedin.com/in/foo"]

    def test_fields_does_not_request_locked_fields(self):
        """Regression test (2026-09-24): our Crustdata account is not
        permitted to RETURN years_of_experience_raw or recently_changed_jobs
        — requesting either in `fields` fails the whole call with
        `permission_error: Access denied to fields: ...`. They still work as
        FILTERS (see test_filters_remapped_before_sending-style tests); this
        only guards the returned-fields list."""
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(json_data={"profiles": [], "total_count": 0})
            search_people_db_v2({}, api_key="test-key")
            body = mock_post.call_args.kwargs["json"]
            assert "years_of_experience_raw" not in body["fields"]
            assert "recently_changed_jobs" not in body["fields"]


# ---------------------------------------------------------------------------
# search_people_db_v2 — response parsing
# ---------------------------------------------------------------------------


_SAMPLE_V2_SEARCH_PROFILE = {
    "crustdata_person_id": 1,
    "basic_profile": {"name": "Jane Doe", "headline": "Founding Engineer"},
    "experience": {
        "employment_details": {
            "current": [{"name": "Acme", "title": "Founding Engineer", "start_date": "2023-01-01"}],
            "past": [],
        }
    },
    "social_handles": {
        "professional_network_identifier": {"profile_url": "https://www.linkedin.com/in/janedoe"}
    },
}


class TestSearchV2Response:
    def test_profiles_translated_via_shared_shim(self):
        """Reuses semantic_profile_to_legacy_shape() — same translator
        search_people_semantic() results already go through — so the results
        table code doesn't need a second code path."""
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                json_data={"profiles": [_SAMPLE_V2_SEARCH_PROFILE], "total_count": 1}
            )
            result = search_people_db_v2({}, api_key="test-key")

        profile = result["profiles"][0]
        assert profile["name"] == "Jane Doe"
        assert profile["current_employers"][0]["name"] == "Acme"
        assert profile["flagship_profile_url"] == "https://www.linkedin.com/in/janedoe"
        # The whole reason this endpoint needs enrichment before AI Screen.
        assert profile["_semantic_incomplete"] is True

    def test_credits_used_is_003_per_result(self):
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                json_data={"profiles": [_SAMPLE_V2_SEARCH_PROFILE] * 100, "total_count": 100}
            )
            result = search_people_db_v2({}, api_key="test-key")
        assert result["credits_used"] == pytest.approx(3.0)

    def test_401_raises_authentication_error(self):
        from error_handling import AuthenticationError

        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(status_code=401)
            with pytest.raises(AuthenticationError):
                search_people_db_v2({}, api_key="bad-key")


# ---------------------------------------------------------------------------
# Dashboard wiring — the Search tab must only ever call the new endpoint
# ---------------------------------------------------------------------------


class TestDashboardUsesV2SearchOnly:
    def test_active_search_alias_is_v2(self):
        import dashboard

        assert dashboard._active_search_people_db is search_people_db_v2
        assert not hasattr(dashboard, "CRUSTDATA_USE_SEARCH_V2")

    def test_dashboard_search_call_hits_new_endpoint_with_bearer(self):
        import dashboard

        filters = build_filters(title="Engineer", country="Israel")
        with patch(
            "crustdata_search.requests.post",
            return_value=_mock_response(200, {"profiles": [], "total_count": 0}),
        ) as mock_post:
            dashboard._active_search_people_db(
                filters, limit=100,
                sorts=[{"column": "num_of_connections", "order": "desc"}],
                api_key="test-key",
                exclude_profiles=["https://www.linkedin.com/in/past-candidate"],
            )

        url = mock_post.call_args.args[0]
        headers = mock_post.call_args.kwargs["headers"]
        body = mock_post.call_args.kwargs["json"]
        assert url == CRUSTDATA_SEARCH_V2_ENDPOINT
        assert "/screener/" not in url
        assert headers["Authorization"] == "Bearer test-key"
        assert "Token" not in headers["Authorization"]
        assert headers["x-api-version"] == CRUSTDATA_API_VERSION
        assert body["post_processing"]["exclude_profiles"] == [
            "https://www.linkedin.com/in/past-candidate"
        ]

    def test_v2_search_row_is_topped_up_before_screening(self, monkeypatch):
        """A profile sourced from the v2 filter search has no skills/summary,
        so enrich_thin_profiles_for_batch() must pick it up and send it to
        the new batch enrich (not skip it as already rich or on cooldown)."""
        import dashboard
        from crustdata_search import normalize_linkedin_url

        raw_v2 = {
            "basic_profile": {"name": "Dana Levi", "headline": "Engineer"},
            "experience": {"employment_details": {"current": [{"name": "Acme", "title": "Engineer"}]}},
            "social_handles": {"professional_network_identifier": {
                "profile_url": "https://www.linkedin.com/in/dana-levi"}},
            "crustdata_person_id": 123,
        }
        with patch(
            "crustdata_search.requests.post",
            return_value=_mock_response(200, {"profiles": [raw_v2], "total_count": 1}),
        ):
            result = dashboard._active_search_people_db({}, limit=1, api_key="test-key")

        row = result["profiles"][0]
        url = normalize_linkedin_url(row["flagship_profile_url"])
        batch_profile = {"linkedin_url": url, "name": row["name"], "raw_crustdata": row}

        captured = {}

        def fake_batch_enrich(urls, api_key=None):
            captured["urls"] = urls
            return {"by_url": {}, "requested": len(urls), "fulfilled": 0,
                    "unmatched": urls, "credits_used": 0, "batch_ids": []}

        monkeypatch.setattr(dashboard, "batch_enrich_profiles", fake_batch_enrich)
        stats = dashboard.enrich_thin_profiles_for_batch([batch_profile], api_key="test-key")

        assert captured["urls"] == [url]
        assert stats["thin_found"] == 1
        assert stats["on_cooldown"] == 0


# ---------------------------------------------------------------------------
# check_credits — new headers, both response shapes
# ---------------------------------------------------------------------------


class TestCheckCreditsNewHeaders:
    def test_sends_bearer_and_version_header_to_new_path(self):
        from crustdata_search import check_credits, CRUSTDATA_CREDITS_ENDPOINT

        with patch(
            "crustdata_search.requests.get",
            return_value=_mock_response(200, {"account": {"credits": 5}}),
        ) as mock_get:
            check_credits(api_key="test-key")

        url = mock_get.call_args.args[0]
        headers = mock_get.call_args.kwargs["headers"]
        assert url == CRUSTDATA_CREDITS_ENDPOINT
        assert url == "https://api.crustdata.com/account/credits"
        assert "/screener/" not in url
        assert headers["Authorization"] == "Bearer test-key"
        assert "Token" not in headers["Authorization"]
        assert headers["x-api-version"] == CRUSTDATA_API_VERSION

    def test_parses_nested_account_shape(self):
        from crustdata_search import check_credits

        payload = {"account": {
            "credits": 138940.37, "recurring_credits": 150000,
            "recurring_credits_frequency": "monthly", "wallets": [],
        }}
        with patch("crustdata_search.requests.get", return_value=_mock_response(200, payload)):
            info = check_credits(api_key="test-key")

        assert info["remaining"] == 138940.37
        assert info["total"] == 150000

    def test_parses_flat_shape(self):
        from crustdata_search import check_credits

        with patch(
            "crustdata_search.requests.get",
            return_value=_mock_response(200, {"credits": 9406}),
        ):
            info = check_credits(api_key="test-key")

        assert info["remaining"] == 9406
        assert info["used"] == 0
        assert info["total"] == 0

    def test_zero_balance_is_not_mistaken_for_missing(self):
        from crustdata_search import check_credits

        with patch(
            "crustdata_search.requests.get",
            return_value=_mock_response(200, {"account": {"credits": 0, "remaining": 99}}),
        ):
            info = check_credits(api_key="test-key")

        assert info["remaining"] == 0
