"""
Tests for geo_terms — the location term-expansion behind the "Find Similar
Profiles" country/city filter.

The promise to the user: type one plain word and the backend matches every
related term. These tests pin that promise (Israel + a city expand to many
terms, including ones that don't literally contain the country name), and the
graceful fallback for places we haven't curated.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import requests

from geo_terms import (
    expand_city,
    expand_country,
    expand_location_terms,
)


# ---------------------------------------------------------------------------
# Country expansion
# ---------------------------------------------------------------------------
def test_country_israel_expands_to_cities_not_just_the_word():
    terms = expand_country("Israel")
    # The country name itself...
    assert "israel" in terms
    # ...plus cities that do NOT contain the word "israel" — the whole point.
    assert "tel aviv" in terms
    assert "haifa" in terms
    assert "herzliya" in terms
    assert "jerusalem" in terms


def test_country_israel_includes_city_spelling_variants():
    terms = expand_country("Israel")
    assert "tel aviv-yafo" in terms
    assert "be'er sheva" in terms or "beer sheva" in terms


def test_israel_terms_do_not_match_foreign_cities():
    # Substring matching must not pull foreign places into an Israel filter.
    terms = expand_country("Israel")
    foreign = ["lodz, poland", "arad, romania", "acre, brazil", "massacre bay"]
    for loc in foreign:
        hits = [t for t in terms if t in loc]
        assert hits == [], f"{loc!r} wrongly matched Israel terms: {hits}"


def test_israel_still_matches_real_israeli_cities():
    terms = expand_country("Israel")
    for loc in ["tel aviv, israel", "haifa", "hadera, israel", "akko"]:
        assert any(t in loc for t in terms), f"{loc!r} should match Israel"


def test_unknown_country_falls_back_to_its_name():
    # Not curated, but should still match the literal country text.
    assert expand_country("Venezuela") == ["venezuela"]


def test_empty_country_returns_nothing():
    assert expand_country("") == []
    assert expand_country(None) == []


# ---------------------------------------------------------------------------
# City expansion
# ---------------------------------------------------------------------------
def test_city_tel_aviv_expands_to_metro_and_spellings():
    terms = expand_city("Tel Aviv")
    assert "tel aviv" in terms
    assert "tel aviv-yafo" in terms
    assert "jaffa" in terms


def test_city_lookup_is_case_and_dash_insensitive():
    assert "jaffa" in expand_city("tel-aviv")
    assert "jaffa" in expand_city("  TEL AVIV  ")


def test_uncurated_city_falls_back_to_literal_text():
    # Reykjavik isn't in the map — still a valid substring match, lowercased.
    assert expand_city("Reykjavik") == ["reykjavik"]


def test_like_wildcards_are_stripped_from_typed_city():
    # A typed "%" / "_" must not become a SQL LIKE wildcard.
    assert "%" not in "".join(expand_location_terms(city="100%remote_x"))
    assert "_" not in "".join(expand_location_terms(city="100%remote_x"))


def test_empty_city_returns_nothing():
    assert expand_city("") == []
    assert expand_city(None) == []


# ---------------------------------------------------------------------------
# Combined expansion
# ---------------------------------------------------------------------------
def test_combined_dedupes_and_lowercases():
    # Israel already includes tel aviv terms; adding the city must not dupe.
    terms = expand_location_terms(country="Israel", city="Tel Aviv")
    assert terms == [t.lower() for t in terms]
    assert len(terms) == len(set(terms))
    assert "tel aviv" in terms


def test_combined_with_neither_is_empty():
    assert expand_location_terms() == []
    assert expand_location_terms(country=None, city=None) == []


def test_combined_city_only():
    terms = expand_location_terms(city="London")
    assert "london" in terms
    assert "greater london" in terms


# ---------------------------------------------------------------------------
# Wiring: search_similar must forward expanded terms to the RPC.
# ---------------------------------------------------------------------------
def test_search_similar_forwards_location_terms(monkeypatch):
    import similar_profiles as sp

    captured = {}

    def fake_get_or_build(db_client, openai_client, url, crustdata_key=None):
        return ([0.1] * 1536, {"linkedin_url": "https://www.linkedin.com/in/me/"}, "cached")

    def fake_rpc(db_client, query_embedding, match_count, min_similarity, location_terms):
        captured["location_terms"] = location_terms
        return [
            {"linkedin_url": "https://www.linkedin.com/in/a/", "location": "Tel Aviv, Israel", "similarity": 0.9},
        ]

    monkeypatch.setattr(sp, "get_or_build_query_embedding", fake_get_or_build)
    monkeypatch.setattr(sp, "find_similar_profiles_rpc", fake_rpc)

    sp.search_similar(
        db_client=None,
        openai_client=None,
        linkedin_url="https://www.linkedin.com/in/me/",
        match_count=10,
        country="Israel",
        city="Tel Aviv",
    )

    assert "tel aviv" in captured["location_terms"]
    assert "haifa" in captured["location_terms"]


def test_search_similar_no_location_passes_empty_terms(monkeypatch):
    import similar_profiles as sp

    captured = {}

    monkeypatch.setattr(
        sp, "get_or_build_query_embedding",
        lambda *a, **k: ([0.1] * 1536, {"linkedin_url": "u"}, "cached"),
    )

    def fake_rpc(db_client, query_embedding, match_count, min_similarity, location_terms):
        captured["location_terms"] = location_terms
        return []

    monkeypatch.setattr(sp, "find_similar_profiles_rpc", fake_rpc)

    sp.search_similar(
        db_client=None, openai_client=None,
        linkedin_url="https://www.linkedin.com/in/me/",
    )
    assert captured["location_terms"] == []


# ---------------------------------------------------------------------------
# Wiring: the RPC client must include location_terms in the request body.
# ---------------------------------------------------------------------------
def test_rpc_includes_location_terms_in_payload(monkeypatch):
    import db

    captured = {}

    class FakeResponse:
        status_code = 200
        text = "[]"

        def json(self):
            return []

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    client = SimpleNamespace(url="https://fake.supabase.co", headers={})
    db.find_similar_profiles_rpc(
        client, [0.1] * 1536, match_count=5,
        location_terms=["israel", "tel aviv"],
    )
    assert captured["json"]["location_terms"] == ["israel", "tel aviv"]


def test_rpc_defaults_location_terms_to_empty_list(monkeypatch):
    import db

    captured = {}

    class FakeResponse:
        status_code = 200
        text = "[]"

        def json(self):
            return []

    monkeypatch.setattr(
        requests, "post",
        lambda url, headers=None, json=None, timeout=None: (
            captured.__setitem__("json", json) or FakeResponse()
        ),
    )

    client = SimpleNamespace(url="https://fake.supabase.co", headers={})
    db.find_similar_profiles_rpc(client, [0.1] * 1536, match_count=5)
    assert captured["json"]["location_terms"] == []
