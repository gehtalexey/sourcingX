"""Parity: SourcingX enrich_profile_to_legacy_shape == Supanova to_legacy_shape.

Supanova's converter source is given by the env var SUPANOVA_ENRICH_REPLACE
(path to a copy of supanova db_core/enrich_replace.py). Parity tests skip if unset.
The "old output keeps its keys" test always runs (uses a frozen copy of the old logic).
"""
import os
import pytest

from crustdata_search import enrich_profile_to_legacy_shape

PERSON = {
    "crustdata_person_id": 42,
    "basic_profile": {
        "name": "Dana Cohen", "first_name": "Dana", "last_name": "Cohen",
        "headline": "Staff Engineer", "summary": "Builds things",
        "current_title": "Staff Engineer", "normalized_title": "software engineer",
        "languages": ["English", "Hebrew"],
        "location": {"raw": "Tel Aviv, Israel", "city": "Tel Aviv", "country": "Israel"},
        "profile_picture_permalink": "https://img/x.jpg",
    },
    "professional_network": {
        "connections": 500, "followers": 900, "pronoun": "she/her",
        "location": {"raw": "Tel Aviv District, Israel"},
    },
    "social_handles": {"professional_network_identifier": {"profile_url": "https://www.linkedin.com/in/dana"}},
    "experience": {
        "years_of_experience": 9, "years_of_experience_raw": 9.4,
        "employment_details": {
            "current": [{
                "title": " Staff Engineer ", "name": "Acme", "start_date": "2020-01-01T00:00:00+00:00",
                "end_date": None, "description": "Platform",
                "company_professional_network_profile_url": "https://www.linkedin.com/company/acme",
                "company_headcount": 250, "company_description": "Acme makes widgets. More text.",
                "company_id": 7,
            }],
            "past": [{
                "title": "Dev", "name": "Beta", "start_date": "2016-01-01T00:00:00+00:00",
                "end_date": "2019-12-01T00:00:00+00:00", "description": "",
                "company_professional_network_profile_url": "https://www.linkedin.com/company/beta",
                "company_headcount": 40,
            }],
        },
    },
    "education": {"schools": [{"school": "Technion", "degree": "BSc", "field_of_study": "CS", "start_date": "2012"}]},
    "skills": {"professional_network_skills": ["Python", {"name": "Go"}]},
    "contact": {"emails": ["d@x.com"]},
    "emails": ["d@x.com"],
}


def _load_supanova():
    path = os.environ.get("SUPANOVA_ENRICH_REPLACE")
    if not path or not os.path.exists(path):
        pytest.skip("SUPANOVA_ENRICH_REPLACE not set")
    src = open(path, encoding="utf-8").read()
    # only the converter + _empty helper; skip the db import and merge code
    head = src.split("# 2. Merge")[0].replace("from db import _prepare_profile_row  # noqa: E402", "")
    helper = "def _empty(v):\n    return v is None or v == '' or v == [] or v == {}\n"
    ns = {"__file__": path}
    exec(compile(head + helper, path, "exec"), ns)
    return ns["to_legacy_shape"]


def _old_job(e):
    return {
        "employee_title": (e.get("title") or "").strip(), "employer_name": e.get("name") or "",
        "start_date": e.get("start_date"), "end_date": e.get("end_date"),
        "employee_description": e.get("description") or None,
        "employer_linkedin_description": None,
    }


def test_job_entries_identical_to_supanova():
    sup = _load_supanova()(PERSON)
    sx = enrich_profile_to_legacy_shape(PERSON)
    assert sx["current_employers"] == sup["current_employers"]
    assert sx["past_employers"] == sup["past_employers"]
    assert sx["education_background"] == sup["education_background"]
    assert sx["current_employers"][0]["company_linkedin_profile_url"].endswith("/company/acme")
    assert sx["current_employers"][0]["company_headcount"] == 250


def test_whole_output_identical_to_supanova():
    assert enrich_profile_to_legacy_shape(PERSON) == _load_supanova()(PERSON)


def test_old_keys_keep_same_values():
    new = enrich_profile_to_legacy_shape(PERSON)
    for k in ("name", "title", "headline", "summary", "location", "region", "num_of_connections",
              "skills", "languages", "linkedin_flagship_url", "linkedin_url", "all_employers",
              "all_titles", "all_schools", "all_degrees", "crustdata_person_id"):
        assert k in new
    assert new["skills"] == ["Python", "Go"]
    assert new["title"] == "Staff Engineer" and new["region"] == "Tel Aviv District, Israel"
    for key, src in (("current_employers", "current"), ("past_employers", "past")):
        for got, raw in zip(new[key], PERSON["experience"]["employment_details"][src]):
            for k, v in _old_job(raw).items():
                if k == "employer_linkedin_description":
                    continue  # was always None; now filled from company_description
                assert got[k] == v
    edu = new["education_background"][0]
    assert (edu["institute_name"], edu["degree_name"], edu["field_of_study"]) == ("Technion", "BSc", "CS")
    assert new["years_of_experience_raw"] == 9.4 and new["location_details"]["city"] == "Tel Aviv"
    assert new["contact"] == {"emails": ["d@x.com"]} and new["emails"] == ["d@x.com"]
