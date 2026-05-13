"""Regression tests for the dropped profiles.status column.

The profiles.status column was removed in the shared-DB migration. Writing
to it (or filtering by it) causes 400s from Supabase, which previously
silently broke batch upserts (failed/not_found enrichments would not save
and kept appearing as "to enrich").

These tests pin the contract: profile-write helpers must never emit
'status' in their upsert/update payload, and read helpers must not filter
on it. Run with `pytest test_profiles_status_dropped.py -v`.
"""

import pytest

import db


class FakeClient:
    """Captures every Supabase call so the test can assert on the payloads."""

    def __init__(self, select_return=None):
        self.url = "https://example.supabase.co"
        self.key = "test-key"
        self.upsert_calls = []
        self.upsert_batch_calls = []
        self.update_calls = []
        self.select_calls = []
        self.count_calls = []
        self._select_return = select_return or []

    def upsert(self, table, data, on_conflict=None):
        self.upsert_calls.append({"table": table, "data": data, "on_conflict": on_conflict})
        return [data]

    def upsert_batch(self, table, rows, on_conflict=None):
        self.upsert_batch_calls.append({"table": table, "rows": rows, "on_conflict": on_conflict})
        return rows

    def update(self, table, data, filters):
        self.update_calls.append({"table": table, "data": data, "filters": filters})
        return [data]

    def select(self, table, columns="*", filters=None, limit=5000, **kwargs):
        self.select_calls.append({"table": table, "columns": columns, "filters": filters, "limit": limit})
        if isinstance(self._select_return, dict):
            return self._select_return.get(table, [])
        return self._select_return

    def count(self, table, filters=None):
        self.count_calls.append({"table": table, "filters": filters})
        return 0


def _all_payloads(client):
    """Flatten every write payload that touched the profiles table."""
    out = []
    for call in client.upsert_calls:
        if call["table"] == "profiles":
            out.append(call["data"])
    for call in client.upsert_batch_calls:
        if call["table"] == "profiles":
            out.extend(call["rows"])
    for call in client.update_calls:
        if call["table"] == "profiles":
            out.append(call["data"])
    return out


def _all_writes(client):
    """Flatten every write payload, regardless of target table."""
    out = []
    for call in client.upsert_calls:
        out.append({"table": call["table"], "payload": call["data"]})
    for call in client.upsert_batch_calls:
        for row in call["rows"]:
            out.append({"table": call["table"], "payload": row})
    for call in client.update_calls:
        out.append({"table": call["table"], "payload": call["data"]})
    return out


def _all_profile_select_columns(client):
    """Return the columns requested from profiles."""
    return [
        call["columns"]
        for call in client.select_calls
        if call["table"] == "profiles"
    ]


def _all_profile_filters(client):
    """Flatten every filter dict targeting the profiles table."""
    out = []
    for call in client.select_calls:
        if call["table"] == "profiles" and call["filters"]:
            out.append(call["filters"])
    for call in client.count_calls:
        if call["table"] == "profiles" and call["filters"]:
            out.append(call["filters"])
    for call in client.update_calls:
        if call["table"] == "profiles":
            out.append(call["filters"])
    return out


# ----------------------------------------------------------------------------
# Writes must not include 'status'
# ----------------------------------------------------------------------------

def test_save_failed_enrichment_omits_status():
    client = FakeClient()
    db.save_failed_enrichment(
        client,
        "https://www.linkedin.com/in/test-user",
        error_message="not found",
        original_url="https://www.linkedin.com/in/test-user",
    )
    for payload in _all_payloads(client):
        assert "status" not in payload, f"profiles write leaked 'status': {payload}"
    # Sanity: it did write the enrichment_status sentinel
    assert any(p.get("enrichment_status") == "not_found" for p in _all_payloads(client))


def test_update_profile_screening_omits_status():
    client = FakeClient()
    db.update_profile_screening(
        client,
        "https://www.linkedin.com/in/test-user",
        score=7,
        fit_level="Good Fit",
        summary="ok",
        reasoning="because",
    )
    for payload in _all_payloads(client):
        assert "status" not in payload, f"profiles write leaked 'status': {payload}"
    assert not _all_payloads(client), "screening writes must not target profiles"
    assert client.upsert_calls[0]["table"] == "screening_results"


def test_update_profile_screening_batch_omits_status():
    client = FakeClient()
    db.update_profile_screening_batch(
        client,
        [
            {"linkedin_url": "https://www.linkedin.com/in/a", "score": 5, "fit_level": "Maybe",
             "summary": "x", "reasoning": "y"},
            {"linkedin_url": "https://www.linkedin.com/in/b", "score": 8, "fit_level": "Good Fit",
             "summary": "x", "reasoning": "y"},
        ],
    )
    for payload in _all_payloads(client):
        assert "status" not in payload, f"profiles write leaked 'status': {payload}"
    assert not _all_payloads(client), "batch screening writes must not target profiles"
    assert client.upsert_batch_calls[0]["table"] == "screening_results"


def test_screening_writes_use_only_screening_results_columns():
    client = FakeClient()
    db.update_profile_screening(
        client,
        "https://www.linkedin.com/in/test-user",
        score=7,
        fit_level="Good Fit",
        summary="ok",
        reasoning="because",
    )
    forbidden_profile_columns = {
        "status",
        "screening_score",
        "screening_fit_level",
        "screening_summary",
        "screening_reasoning",
        "screening_notes",
        "screened_at",
    }
    for write in _all_writes(client):
        if write["table"] == "profiles":
            leaked = forbidden_profile_columns & set(write["payload"])
            assert not leaked, f"profiles write leaked missing columns: {leaked}"


def test_prepare_profile_row_omits_status():
    crustdata = {
        "name": "Test User",
        "headline": "Engineer at Acme",
        "location": "Tel Aviv",
    }
    row = db._prepare_profile_row(
        "https://www.linkedin.com/in/test-user",
        crustdata,
        original_url="https://www.linkedin.com/in/test-user",
    )
    assert "status" not in row, f"_prepare_profile_row leaked 'status': {row}"


# ----------------------------------------------------------------------------
# Reads must not filter by 'status'
# ----------------------------------------------------------------------------

def test_get_profiles_needing_screening_filters_enrichment_status():
    client = FakeClient()
    db.get_profiles_needing_screening(client, limit=10)
    for filters in _all_profile_filters(client):
        assert "status" not in filters, f"profile filter leaked 'status': {filters}"
    # Sanity: it does filter on the replacement column
    assert any("enrichment_status" in f for f in _all_profile_filters(client))


def test_get_profiles_needing_screening_does_not_filter_profiles_by_screening_score():
    client = FakeClient()
    db.get_profiles_needing_screening(client, limit=10)
    for filters in _all_profile_filters(client):
        assert "screening_score" not in filters, (
            f"profile filter leaked missing screening_score column: {filters}"
        )


def test_get_profiles_by_fit_level_reads_latest_screening_not_profiles():
    client = FakeClient(select_return={
        "latest_screening": [
            {
                "linkedin_url": "https://www.linkedin.com/in/a",
                "screening_score": 8,
                "screening_fit_level": "Good Fit",
            }
        ],
        "profiles": [
            {
                "linkedin_url": "https://www.linkedin.com/in/a",
                "name": "A",
            }
        ],
    })
    rows = db.get_profiles_by_fit_level(client, "Good Fit", limit=10)
    profile_calls = [c for c in client.select_calls if c["table"] == "profiles"]
    latest_calls = [c for c in client.select_calls if c["table"] == "latest_screening"]

    assert latest_calls
    assert latest_calls[0]["filters"]["screening_fit_level"] == "eq.Good Fit"
    for call in profile_calls:
        filters = call["filters"] or {}
        assert "screening_fit_level" not in filters
    assert rows[0]["screening_fit_level"] == "Good Fit"


def test_get_all_profiles_selects_only_live_profile_columns_and_merges_screening():
    client = FakeClient(select_return={
        "profiles": [
            {
                "linkedin_url": "https://www.linkedin.com/in/a",
                "name": "A",
                "enriched_at": "2026-01-01T00:00:00",
            }
        ],
        "latest_screening": [
            {
                "linkedin_url": "https://www.linkedin.com/in/a",
                "screening_score": 9,
                "screening_fit_level": "Good Fit",
                "screened_at": "2026-05-13T00:00:00",
            }
        ],
    })
    rows = db.get_all_profiles(client, limit=10, lightweight=False)
    forbidden_profile_columns = {
        "status",
        "screening_score",
        "screening_fit_level",
        "screening_summary",
        "screening_reasoning",
        "screening_notes",
        "screened_at",
    }

    for columns in _all_profile_select_columns(client):
        selected = set(columns.split(",")) if columns != "*" else set()
        leaked = forbidden_profile_columns & selected
        assert not leaked, f"profiles SELECT leaked missing columns: {leaked}"
    assert rows[0]["screening_score"] == 9
    assert rows[0]["screening_fit_level"] == "Good Fit"


def test_get_all_profiles_lightweight_selects_only_live_profile_columns():
    client = FakeClient(select_return={
        "profiles": [{"linkedin_url": "https://www.linkedin.com/in/a"}],
        "latest_screening": [],
    })
    db.get_all_profiles(client, limit=10, lightweight=True)
    for columns in _all_profile_select_columns(client):
        assert "status" not in columns
        assert "screening_score" not in columns
        assert "screened_at" not in columns


def test_get_profiles_by_enrichment_status_filters_enrichment_status():
    client = FakeClient()
    db.get_profiles_by_enrichment_status(client, "enriched", limit=10)
    for filters in _all_profile_filters(client):
        assert "status" not in filters, f"profile filter leaked 'status': {filters}"
    assert any("enrichment_status" in f for f in _all_profile_filters(client))


def test_legacy_get_profiles_by_status_is_gone():
    """The old generic-status helper was removed; only the renamed one remains."""
    assert not hasattr(db, "get_profiles_by_status")
    assert hasattr(db, "get_profiles_by_enrichment_status")
