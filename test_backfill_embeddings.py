"""
Tests for the backfill worker.

These tests use fake Supabase and OpenAI clients so they run with no
network access. They verify the two pagination fixes from PR #77 review:
- ``--dry-run`` advances through pages instead of looping forever.
- ``--re-embed-changed`` includes ``enriched_at`` in its SELECT and the
  cursor actually moves.
"""

from __future__ import annotations

import sys
import types

import backfill_embeddings as bf


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeSupabaseClient:
    """Stand-in for db.SupabaseClient that records every read/write call."""

    url = "https://fake.supabase.co"
    headers = {"apikey": "fake", "Authorization": "Bearer fake"}

    def __init__(self, pages: list[list[dict]]):
        # `pages` is a list of pages keyed by call order. Each call returns
        # the next page; once exhausted, returns [].
        self.pages = list(pages)
        self.select_calls: list[dict] = []
        self.upsert_calls: list[list[dict]] = []

    def select(self, table, columns, filters, limit, order_by=None, **kwargs):
        self.select_calls.append({
            "table": table,
            "columns": columns,
            "filters": dict(filters),
            "limit": limit,
            "order_by": order_by,
            "path": "select",
        })
        return self.pages.pop(0) if self.pages else []

    def _request(self, method, endpoint, params=None, json_data=None):
        # Used by the offset-based fetch path in dry-run mode.
        self.select_calls.append({
            "table": endpoint,
            "columns": params.get("select"),
            "filters": {k: v for k, v in params.items()
                        if k not in ("select", "order", "limit", "offset")},
            "limit": params.get("limit"),
            "order_by": params.get("order"),
            "offset": params.get("offset"),
            "path": "request",
        })
        return self.pages.pop(0) if self.pages else []

    def upsert_batch(self, table, rows, on_conflict=None):
        self.upsert_calls.append(list(rows))
        return rows


def _make_row(i: int, with_embedded=False, hash_value="abc") -> dict:
    return {
        "linkedin_url": f"https://www.linkedin.com/in/p{i}/",
        "name": f"Person {i}",
        "current_title": "Engineer",
        "current_company": "Acme",
        "location": "Tel Aviv",
        "all_titles": ["Engineer"],
        "all_employers": ["Acme"],
        "all_schools": ["Technion"],
        "skills": ["Python"],
        "raw_data": {"headline": f"profile {i}", "summary": "ten years"},
        "enriched_at": f"2026-05-{(i % 28) + 1:02d}T00:00:00+00:00",
        "embedding_input_hash": hash_value if with_embedded else None,
        "embedded_at": "2026-05-01T00:00:00+00:00" if with_embedded else None,
    }


# ---------------------------------------------------------------------------
# Patch get_supabase_client so run_backfill picks up the fake.
# ---------------------------------------------------------------------------
def _install_client(monkeypatch, client):
    import db
    monkeypatch.setattr(db, "get_supabase_client", lambda: client)
    monkeypatch.setattr(bf, "get_supabase_client", lambda: client)


# ---------------------------------------------------------------------------
# Tests for the dry-run pagination fix.
# ---------------------------------------------------------------------------
def test_dry_run_advances_offset_and_terminates(monkeypatch, capsys):
    """Without the fix this loops forever. With the fix, three pages
    (2 + 2 + empty) terminate cleanly and the cost estimate is cumulative,
    not double-counted."""
    page_a = [_make_row(1), _make_row(2)]
    page_b = [_make_row(3), _make_row(4)]
    client = FakeSupabaseClient(pages=[page_a, page_b, []])
    _install_client(monkeypatch, client)

    bf.run_backfill(limit=None, dry_run=True, re_embed_changed=False, page_size=2)

    # Three fetches: page_a, page_b, empty.
    assert len(client.select_calls) == 3

    # Second call must use offset > 0 (the fix). Without the fix, every call
    # would have no offset and the loop would never terminate.
    second = client.select_calls[1]
    assert second.get("offset") == 2, (
        f"dry-run did not advance offset on second call: {second}"
    )
    # And the third call should advance further.
    third = client.select_calls[2]
    assert third.get("offset") == 4

    # No upserts should have happened in dry-run.
    assert client.upsert_calls == []

    out = capsys.readouterr().out
    assert "dry-run" in out.lower()


def test_live_run_does_not_use_offset(monkeypatch):
    """Live (writing) mode shouldn't pass an offset — writes naturally
    drop rows from the ``embedding IS NULL`` set."""
    client = FakeSupabaseClient(pages=[[_make_row(1)], []])
    _install_client(monkeypatch, client)

    # Patch OpenAI so we don't try to hit the network.
    monkeypatch.setattr(
        bf, "embed_texts",
        lambda openai_client, texts, model: [[0.0] * 1536 for _ in texts],
    )
    monkeypatch.setattr(
        bf, "OpenAI",
        lambda **kwargs: types.SimpleNamespace(),
    )
    monkeypatch.setattr(bf, "load_openai_key", lambda: "test-key")

    bf.run_backfill(limit=None, dry_run=False, re_embed_changed=False, page_size=10)

    first = client.select_calls[0]
    assert first.get("offset") in (None, 0), (
        f"live run unexpectedly used offset: {first}"
    )


# ---------------------------------------------------------------------------
# Tests for the --re-embed-changed cursor fix.
# ---------------------------------------------------------------------------
def test_re_embed_changed_selects_enriched_at(monkeypatch):
    """The drift-check SELECT must include enriched_at so the cursor can
    advance. Without enriched_at, row.get('enriched_at') is None and the
    same page is fetched forever."""
    page_a = [_make_row(1, with_embedded=True, hash_value="OLD"),
              _make_row(2, with_embedded=True, hash_value="OLD")]
    client = FakeSupabaseClient(pages=[page_a, []])
    _install_client(monkeypatch, client)
    monkeypatch.setattr(
        bf, "embed_texts",
        lambda openai_client, texts, model: [[0.1] * 1536 for _ in texts],
    )
    monkeypatch.setattr(bf, "OpenAI", lambda **k: types.SimpleNamespace())
    monkeypatch.setattr(bf, "load_openai_key", lambda: "test-key")

    bf.run_backfill(limit=None, dry_run=False, re_embed_changed=True, page_size=10)

    # All select calls in the drift-check path should pull enriched_at.
    for call in client.select_calls:
        cols = call.get("columns") or ""
        assert "enriched_at" in cols, (
            f"drift-check SELECT missing enriched_at: {call}"
        )

    # The second SELECT must include the cursor filter (enriched_at < ...)
    # — proof the cursor actually moved.
    second = client.select_calls[1]
    cursor_filter = second["filters"].get("enriched_at", "")
    assert cursor_filter.startswith("lt."), (
        f"second drift-check call did not apply keyset cursor: {second}"
    )


def test_fetch_columns_includes_enriched_at():
    """Sanity check the constant directly — covers both code paths."""
    assert "enriched_at" in bf.FETCH_COLUMNS
