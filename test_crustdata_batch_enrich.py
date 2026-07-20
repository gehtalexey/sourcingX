"""Tests for the new v2025-11-01 batch enrichment pipeline:
submit_batch_enrich -> get_batch_status -> batch_enrich_profiles
(submit/poll/download orchestrator) -> enrich_profile_to_legacy_shape
(nested v2 -> flat legacy translator).

This is the fix for the new Crustdata search endpoints not returning
skills/summary: fill them back in, right before AI Screen, via
POST /batch/person/enrich (up to 10,000 URLs/job, 1 credit/profile base —
see crustdata_search.py's module docstring section for the full context).

No real network calls. The translator field mapping is verified against a
REAL /person/enrich response captured live 2026-07-20 (same nested shape
the batch endpoint returns) — see _REAL_ENRICH_RESPONSE below.
"""

from unittest.mock import MagicMock, patch

import pytest

from crustdata_search import (
    submit_batch_enrich,
    get_batch_status,
    batch_enrich_profiles,
    enrich_profile_to_legacy_shape,
    CRUSTDATA_BATCH_ENRICH_ENDPOINT,
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
# submit_batch_enrich
# ---------------------------------------------------------------------------


class TestSubmitBatchEnrich:
    def test_sends_bearer_auth_and_version_header(self):
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(json_data={"batch_id": "b1"})
            submit_batch_enrich(["https://www.linkedin.com/in/foo"], api_key="test-key")
            headers = mock_post.call_args.kwargs["headers"]
            assert headers["Authorization"] == "Bearer test-key"
            assert headers["x-api-version"] == CRUSTDATA_API_VERSION

    def test_posts_to_batch_enrich_endpoint_with_urls_and_fields(self):
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(json_data={"batch_id": "b1"})
            urls = ["https://www.linkedin.com/in/foo", "https://www.linkedin.com/in/bar"]
            submit_batch_enrich(urls, api_key="test-key")

            args, kwargs = mock_post.call_args
            assert args[0] == CRUSTDATA_BATCH_ENRICH_ENDPOINT
            assert kwargs["json"]["professional_network_profile_urls"] == urls
            assert "basic_profile" in kwargs["json"]["fields"]
            assert "skills" in kwargs["json"]["fields"]

    def test_chunk_size_clamped_10_to_1000(self):
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(json_data={"batch_id": "b1"})
            submit_batch_enrich(["https://www.linkedin.com/in/foo"], chunk_size=5, api_key="test-key")
            assert mock_post.call_args.kwargs["json"]["chunk_size"] == 10

            submit_batch_enrich(["https://www.linkedin.com/in/foo"], chunk_size=5000, api_key="test-key")
            assert mock_post.call_args.kwargs["json"]["chunk_size"] == 1000

    def test_over_10000_urls_raises_without_calling_api(self):
        with patch("crustdata_search.requests.post") as mock_post:
            with pytest.raises(ValueError):
                submit_batch_enrich(["u"] * 10001, api_key="test-key")
            mock_post.assert_not_called()

    def test_empty_list_raises_without_calling_api(self):
        with patch("crustdata_search.requests.post") as mock_post:
            with pytest.raises(ValueError):
                submit_batch_enrich([], api_key="test-key")
            mock_post.assert_not_called()

    def test_returns_batch_id_from_either_key_name(self):
        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(json_data={"id": "b2"})
            batch_id = submit_batch_enrich(["https://www.linkedin.com/in/foo"], api_key="test-key")
            assert batch_id == "b2"

    def test_401_raises_authentication_error(self):
        from error_handling import AuthenticationError

        with patch("crustdata_search.requests.post") as mock_post:
            mock_post.return_value = _mock_response(status_code=401)
            with pytest.raises(AuthenticationError):
                submit_batch_enrich(["https://www.linkedin.com/in/foo"], api_key="bad-key")


class TestGetBatchStatus:
    def test_gets_batch_status_url(self):
        with patch("crustdata_search.requests.get") as mock_get:
            mock_get.return_value = _mock_response(json_data={"status": "processing"})
            get_batch_status("b1", api_key="test-key")
            args, kwargs = mock_get.call_args
            assert args[0].endswith("/batch/b1")
            assert kwargs["headers"]["x-api-version"] == CRUSTDATA_API_VERSION


# ---------------------------------------------------------------------------
# batch_enrich_profiles — submit/poll/download orchestrator
# ---------------------------------------------------------------------------


def _enrich_record(url, name="Jane Doe"):
    return {
        "original_identifier": url,
        "internal_id": 1,
        "data": {
            "basic_profile": {"name": name, "current_title": "Engineer", "headline": "", "summary": "Builds things."},
            "experience": {"employment_details": {"current": [], "past": []}},
            "education": {"schools": []},
            "skills": {"professional_network_skills": ["Python"]},
            "social_handles": {"professional_network_identifier": {"profile_url": url}},
        },
    }


class TestBatchEnrichProfilesOrchestrator:
    def test_happy_path_all_matched(self, monkeypatch):
        monkeypatch.setattr("crustdata_search.time.sleep", lambda s: None)
        urls = [
            "https://www.linkedin.com/in/a",
            "https://www.linkedin.com/in/b",
            "https://www.linkedin.com/in/c",
        ]
        with patch("crustdata_search.submit_batch_enrich", return_value="batch1") as mock_submit, \
             patch("crustdata_search.get_batch_status") as mock_status, \
             patch("crustdata_search._download_batch_results") as mock_download:
            mock_status.return_value = {"status": "completed"}
            mock_download.return_value = [_enrich_record(u) for u in urls]

            result = batch_enrich_profiles(urls, api_key="test-key")

        assert result["requested"] == 3
        assert result["fulfilled"] == 3
        assert result["unmatched"] == []
        assert result["credits_used"] == 3
        assert set(result["by_url"].keys()) >= set(urls)
        for u in urls:
            assert result["by_url"][u]["skills"] == ["Python"]
        mock_submit.assert_called_once()

    def test_partial_no_match_lands_in_unmatched(self, monkeypatch):
        monkeypatch.setattr("crustdata_search.time.sleep", lambda s: None)
        urls = ["https://www.linkedin.com/in/a", "https://www.linkedin.com/in/b", "https://www.linkedin.com/in/c"]
        with patch("crustdata_search.submit_batch_enrich", return_value="batch1"), \
             patch("crustdata_search.get_batch_status") as mock_status, \
             patch("crustdata_search._download_batch_results") as mock_download:
            mock_status.return_value = {"status": "completed", "entities_requested": 3, "entities_fulfilled": 2}
            mock_download.return_value = [_enrich_record(urls[0]), _enrich_record(urls[1])]

            result = batch_enrich_profiles(urls, api_key="test-key")

        assert result["fulfilled"] == 2
        assert result["unmatched"] == [urls[2]]
        assert result["credits_used"] == 2
        # Never raises over a partial no-match — screening proceeds with
        # whatever data is available (Alexey's decision, 2026-07-20).

    def test_poll_timeout_returns_done_so_far_no_exception(self, monkeypatch):
        monkeypatch.setattr("crustdata_search.time.sleep", lambda s: None)
        urls = ["https://www.linkedin.com/in/a"]
        with patch("crustdata_search.submit_batch_enrich", return_value="batch1"), \
             patch("crustdata_search.get_batch_status") as mock_status:
            mock_status.return_value = {"status": "processing"}  # never terminal

            result = batch_enrich_profiles(urls, api_key="test-key", poll_interval_s=1, max_wait_s=3)

        assert result["fulfilled"] == 0
        assert result["unmatched"] == urls  # not an exception — screened as-is

    def test_over_10000_urls_splits_into_multiple_jobs(self, monkeypatch):
        monkeypatch.setattr("crustdata_search.time.sleep", lambda s: None)
        urls = [f"https://www.linkedin.com/in/u{i}" for i in range(10001)]
        with patch("crustdata_search.submit_batch_enrich", return_value="batchX") as mock_submit, \
             patch("crustdata_search.get_batch_status", return_value={"status": "completed"}), \
             patch("crustdata_search._download_batch_results", return_value=[]):
            batch_enrich_profiles(urls, api_key="test-key")

        assert mock_submit.call_count == 2
        first_call_urls = mock_submit.call_args_list[0].args[0]
        assert len(first_call_urls) == 10000

    def test_submit_failure_leaves_urls_unmatched_not_raised(self, monkeypatch):
        monkeypatch.setattr("crustdata_search.time.sleep", lambda s: None)
        urls = ["https://www.linkedin.com/in/a"]
        with patch("crustdata_search.submit_batch_enrich", side_effect=Exception("boom")):
            result = batch_enrich_profiles(urls, api_key="test-key")

        assert result["fulfilled"] == 0
        assert result["unmatched"] == urls

    def test_empty_input_returns_zeroed_result_without_any_call(self):
        with patch("crustdata_search.submit_batch_enrich") as mock_submit:
            result = batch_enrich_profiles([], api_key="test-key")
        mock_submit.assert_not_called()
        assert result == {"by_url": {}, "requested": 0, "fulfilled": 0, "unmatched": [], "credits_used": 0, "batch_ids": []}


# ---------------------------------------------------------------------------
# enrich_profile_to_legacy_shape — verified against a REAL live response
# (captured 2026-07-20 via crustdata_people_enrich_v2, linkedin.com/in/dvdhsu)
# ---------------------------------------------------------------------------

_REAL_ENRICH_RESPONSE = {
    "basic_profile": {
        "current_title": "Founder, CEO",
        "first_name": None,
        "headline": "Founder, CEO @ Retool",
        "languages": [],
        "last_name": None,
        "location": {
            "city": "", "continent": "North America", "country": "United States",
            "raw": "San Francisco Bay Area", "state": "California",
        },
        "name": "David Hsu",
        "professional_network_name": "David Hsu",
        "profile_picture_permalink": "https://example.com/pic.jpg",
        "summary": "",
    },
    "education": {
        "schools": [{
            "activities_and_societies": "", "degree": "Bachelor of Arts (B.A.)",
            "end_year": None, "field_of_study": "Philosophy and Computer Science",
            "institute_logo_url": "https://example.com/logo.jpg",
            "professional_network_id": "4477", "school": "University of Oxford",
            "start_year": None,
        }]
    },
    "experience": {
        "employment_details": {
            "current": [{
                "company_professional_network_profile_url": "https://linkedin.com/company/11869260",
                "company_profile_picture_permalink": "https://example.com/logo2.jpg",
                "company_website_domain": "retool.com", "crustdata_company_id": 633593,
                "description": "", "employment_type": "", "end_date": None,
                "function_category": "", "is_default": True, "location": {"raw": ""},
                "name": "Retool", "position_id": 1041554620, "professional_network_id": "11869260",
                "seniority_level": "CXO", "start_date": "2017-01-01T00:00:00+00:00",
                "title": "Founder, CEO", "years_at_company": "6 to 10 years",
                "years_at_company_raw": 9.0,
            }],
            "past": [],
        }
    },
    "professional_network": {
        "connections": 701, "current_title": "Founder, CEO", "followers": 14621,
        "headline": "Founder, CEO @ Retool",
        "location": {"city": "", "continent": "North America", "country": "United States",
                      "raw": "San Francisco Bay Area", "state": "California"},
        "metadata": {"last_scraped_source": None}, "name": "David Hsu",
        "profile_picture_permalink": "https://example.com/pic.jpg", "pronoun": "", "summary": "",
    },
    "skills": {"professional_network_skills": []},
    "social_handles": {
        "dev_platform_identifier": {"profile_url": None},
        "professional_network_identifier": {"profile_url": "https://www.linkedin.com/in/dvdhsu"},
        "twitter_identifier": {"slug": ""},
    },
    "crustdata_person_id": 14540,
}


class TestEnrichProfileToLegacyShape:
    def test_maps_real_response_to_flat_shape(self):
        flat = enrich_profile_to_legacy_shape(_REAL_ENRICH_RESPONSE)

        assert flat["name"] == "David Hsu"
        assert flat["title"] == "Founder, CEO"
        assert flat["headline"] == "Founder, CEO @ Retool"
        assert flat["linkedin_flagship_url"] == "https://www.linkedin.com/in/dvdhsu"
        assert flat["num_of_connections"] == 701
        assert flat["crustdata_person_id"] == 14540

    def test_current_employer_uses_flat_legacy_keys(self):
        flat = enrich_profile_to_legacy_shape(_REAL_ENRICH_RESPONSE)
        emp = flat["current_employers"][0]
        assert emp["employee_title"] == "Founder, CEO"
        assert emp["employer_name"] == "Retool"
        assert emp["start_date"] == "2017-01-01T00:00:00+00:00"
        assert emp["end_date"] is None  # current role — compute_role_durations() relies on this
        assert "title" not in emp  # must be the FLAT shape, not the raw v2 keys
        assert "name" not in emp

    def test_education_uses_flat_legacy_keys(self):
        flat = enrich_profile_to_legacy_shape(_REAL_ENRICH_RESPONSE)
        school = flat["education_background"][0]
        assert school["institute_name"] == "University of Oxford"
        assert school["degree_name"] == "Bachelor of Arts (B.A.)"
        assert school["field_of_study"] == "Philosophy and Computer Science"

    def test_derived_all_lists(self):
        flat = enrich_profile_to_legacy_shape(_REAL_ENRICH_RESPONSE)
        assert flat["all_employers"] == ["Retool"]
        assert flat["all_titles"] == ["Founder, CEO"]
        assert flat["all_schools"] == ["University of Oxford"]
        assert flat["all_degrees"] == ["Bachelor of Arts (B.A.)"]

    def test_empty_input_returns_empty_dict(self):
        assert enrich_profile_to_legacy_shape({}) == {}
        assert enrich_profile_to_legacy_shape(None) == {}

    def test_missing_sections_do_not_raise(self):
        flat = enrich_profile_to_legacy_shape({"crustdata_person_id": 1})
        assert flat["name"] == ""
        assert flat["current_employers"] == []
        assert flat["skills"] == []

    def test_skills_as_bare_strings(self):
        data = {"skills": {"professional_network_skills": ["Python", "Go"]}}
        flat = enrich_profile_to_legacy_shape(data)
        assert flat["skills"] == ["Python", "Go"]

    def test_skills_as_objects_coerced_to_strings(self):
        """Element type wasn't confirmed live (sample profiles had empty
        skills lists) — must handle both shapes defensively."""
        data = {"skills": {"professional_network_skills": [{"name": "Python"}, {"name": "Go"}]}}
        flat = enrich_profile_to_legacy_shape(data)
        assert flat["skills"] == ["Python", "Go"]

    def test_output_feeds_trim_raw_profile_and_compute_role_durations(self):
        """End-to-end guarantee: the translator's output keys actually match
        what the screening pipeline reads. Imports dashboard directly rather
        than duplicating its field list here."""
        import dashboard

        flat = enrich_profile_to_legacy_shape(_REAL_ENRICH_RESPONSE)
        trimmed = dashboard.trim_raw_profile(flat)
        assert trimmed["title"] == "Founder, CEO"
        assert trimmed["current_employers"][0]["employee_title"] == "Founder, CEO"
        assert trimmed["current_employers"][0]["employer_name"] == "Retool"

        durations_text = dashboard.compute_role_durations(flat)
        assert "Founder, CEO at Retool" in durations_text
        assert "CURRENT ROLE" in durations_text
