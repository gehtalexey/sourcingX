"""Tests for scripts/reenrich_sparse_profiles.py.

All external services are mocked. No real network calls, no real DB writes.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Module import shim
# ---------------------------------------------------------------------------
# The script lives under scripts/, which is NOT on sys.path by default. Load
# it as a module so the tests can import it without packaging gymnastics.

REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "reenrich_sparse_profiles.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "reenrich_sparse_profiles", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    # Make sure the script can import db.py / api_helpers.py / normalizers.py
    # from the repo root.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


# ---------------------------------------------------------------------------
# Fake Supabase client
# ---------------------------------------------------------------------------


class FakeSupabaseClient:
    """Minimal stand-in for db.SupabaseClient.

    Captures select/update/upsert calls so tests can assert behaviour without
    touching Supabase. Supports a small subset of PostgREST operators that
    the production code actually uses:
      - linkedin_url=eq.<url>          (pre-write "before" read)
      - linkedin_url=not.in.(a,b,c)    (Phase B dedupe)
      - screened_at=gte.<iso>          (Phase A predicate)
      - enriched_at=lt.<iso>           (staleness filter — applied client-
                                        side here only to keep the test
                                        fixture simple; the real DB enforces
                                        it server-side)
    """

    def __init__(self, rows=None):
        self.rows = rows or []
        self.select_calls = []
        self.upsert_calls = []

    def select(self, table, columns="*", filters=None, limit=5000,
               order_by=None, cursor_column=None, cursor_value=None):
        self.select_calls.append({
            "table": table,
            "columns": columns,
            "filters": dict(filters or {}),
            "limit": limit,
            "order_by": order_by,
        })
        filters = filters or {}

        # Honour the linkedin_url eq filter for the "before" pre-write read.
        for key, value in filters.items():
            if key == "linkedin_url" and str(value).startswith("eq."):
                url = str(value)[3:]
                return [r for r in self.rows if r.get("linkedin_url") == url][:1]

        # Apply the small filter subset to the fixture rows.
        out = list(self.rows)

        not_in_urls = None
        require_screened_gte = None
        for key, value in filters.items():
            sval = str(value)
            if key == "linkedin_url" and sval.startswith("not.in."):
                inside = sval[len("not.in."):].strip("()")
                not_in_urls = {u for u in inside.split(",") if u}
            elif key == "screened_at" and sval.startswith("gte."):
                require_screened_gte = sval[len("gte."):]

        if not_in_urls is not None:
            out = [r for r in out if r.get("linkedin_url") not in not_in_urls]
        if require_screened_gte is not None:
            out = [r for r in out if r.get("screened_at")
                   and r.get("screened_at") >= require_screened_gte]

        return out[:limit]

    def upsert(self, table, data, on_conflict=None):
        self.upsert_calls.append({"table": table, "data": data, "on_conflict": on_conflict})
        return [data]


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestArgParsing:

    def test_limit_is_required(self, script):
        parser = script.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self, script):
        parser = script.build_arg_parser()
        args = parser.parse_args(["--limit", "5"])
        assert args.limit == 5
        assert args.max_age_days == 30
        assert args.min_current_employers == 1
        assert args.prefer_screened_days == 90
        assert args.execute is False
        assert args.verbose is False

    def test_all_flags(self, script):
        parser = script.build_arg_parser()
        args = parser.parse_args([
            "--max-age-days", "14",
            "--limit", "100",
            "--min-current-employers", "0",
            "--prefer-screened-days", "45",
            "--execute",
            "--verbose",
            "--batch-size", "3",
        ])
        assert args.max_age_days == 14
        assert args.limit == 100
        assert args.min_current_employers == 0
        assert args.prefer_screened_days == 45
        assert args.execute is True
        assert args.verbose is True
        assert args.batch_size == 3

    def test_help_does_not_crash(self, script):
        parser = script.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])


# ---------------------------------------------------------------------------
# Filtering / candidate selection logic
# ---------------------------------------------------------------------------


class TestCurrentEmployersCount:

    def test_missing_raw_data(self, script):
        assert script._current_employers_count(None) == 0
        assert script._current_employers_count({}) == 0
        assert script._current_employers_count("string") == 0

    def test_missing_field(self, script):
        assert script._current_employers_count({"name": "x"}) == 0

    def test_non_list_field(self, script):
        assert script._current_employers_count({"current_employers": {"a": 1}}) == 0

    def test_only_dict_entries_counted(self, script):
        raw = {"current_employers": [{"a": 1}, None, "s", 42, {"b": 2}]}
        assert script._current_employers_count(raw) == 2

    def test_normal_case(self, script):
        raw = {"current_employers": [
            {"employer_name": "A", "start_date": "2020-01"},
            {"employer_name": "B", "start_date": "2024-05"},
        ]}
        assert script._current_employers_count(raw) == 2


class TestFetchSparseCandidates:

    def _row(self, url, name, age_days, employer_count):
        # Build a raw_data with employer_count dict entries.
        employers = [{"employer_name": f"Co{i}"} for i in range(employer_count)]
        return {
            "linkedin_url": url,
            "name": name,
            "enriched_at": (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat(),
            "raw_data": {"current_employers": employers},
        }

    def test_filters_out_dense_profiles(self, script):
        client = FakeSupabaseClient(rows=[
            self._row("https://www.linkedin.com/in/sparse-1", "Sparse One", 60, 1),
            self._row("https://www.linkedin.com/in/dense-1", "Dense One", 60, 3),
            self._row("https://www.linkedin.com/in/sparse-2", "Sparse Two", 60, 0),
        ])
        # prefer_screened_days=0 disables Phase A so this test isolates the
        # legacy "oldest sparse" behaviour.
        candidates, breakdown = script.fetch_sparse_candidates(
            client, max_age_days=30, limit=10, min_current_employers=1,
            prefer_screened_days=0,
        )
        urls = [c["linkedin_url"] for c in candidates]
        assert "https://www.linkedin.com/in/sparse-1" in urls
        assert "https://www.linkedin.com/in/sparse-2" in urls
        assert "https://www.linkedin.com/in/dense-1" not in urls
        assert breakdown["phase_a"] == 0
        assert breakdown["phase_b"] == len(candidates)

    def test_min_current_employers_zero_excludes_one(self, script):
        client = FakeSupabaseClient(rows=[
            self._row("https://www.linkedin.com/in/sparse-1", "Sparse One", 60, 1),
            self._row("https://www.linkedin.com/in/sparse-2", "Sparse Two", 60, 0),
        ])
        candidates, _ = script.fetch_sparse_candidates(
            client, max_age_days=30, limit=10, min_current_employers=0,
            prefer_screened_days=0,
        )
        urls = [c["linkedin_url"] for c in candidates]
        assert urls == ["https://www.linkedin.com/in/sparse-2"]

    def test_limit_is_respected(self, script):
        rows = [
            self._row(f"https://www.linkedin.com/in/p{i}", f"P {i}", 60, 1)
            for i in range(20)
        ]
        client = FakeSupabaseClient(rows=rows)
        candidates, _ = script.fetch_sparse_candidates(
            client, max_age_days=30, limit=3, min_current_employers=1,
            prefer_screened_days=0,
        )
        assert len(candidates) == 3

    def test_empty_when_zero_limit(self, script):
        client = FakeSupabaseClient(rows=[
            self._row("https://www.linkedin.com/in/sparse-1", "Sparse One", 60, 1),
        ])
        candidates, breakdown = script.fetch_sparse_candidates(
            client, max_age_days=30, limit=0, min_current_employers=1,
            prefer_screened_days=0,
        )
        assert candidates == []
        assert breakdown == {"phase_a": 0, "phase_b": 0, "total": 0}

    def test_cutoff_filter_passed_to_select(self, script):
        now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
        client = FakeSupabaseClient(rows=[])
        script.fetch_sparse_candidates(
            client, max_age_days=10, limit=5, min_current_employers=1,
            prefer_screened_days=0, now=now,
        )
        assert client.select_calls, "expected at least one select call"
        flt = client.select_calls[0]["filters"]
        assert "enriched_at" in flt
        # 10 days before 2026-05-11T12:00:00Z = 2026-05-01T12:00:00Z
        assert flt["enriched_at"] == "lt.2026-05-01T12:00:00Z"


# ---------------------------------------------------------------------------
# Cutoff math
# ---------------------------------------------------------------------------


class TestCutoffIso:

    def test_basic(self, script):
        now = datetime(2026, 5, 11, 0, 0, 0, tzinfo=timezone.utc)
        assert script._cutoff_iso(30, now=now) == "2026-04-11T00:00:00Z"

    def test_zero_days(self, script):
        now = datetime(2026, 5, 11, 12, 34, 56, tzinfo=timezone.utc)
        assert script._cutoff_iso(0, now=now) == "2026-05-11T12:34:56Z"


# ---------------------------------------------------------------------------
# Response matching
# ---------------------------------------------------------------------------


class TestMatchResponseToInput:

    def test_matches_query_field(self, script):
        response = [
            {"query_linkedin_profile_urn_or_slug": ["gil-gitlin-87b720200"],
             "current_employers": [{"employer_name": "Cytactic"}]},
            {"query_linkedin_profile_urn_or_slug": ["someone-else"],
             "current_employers": [{"employer_name": "Other"}]},
        ]
        url = "https://www.linkedin.com/in/gil-gitlin-87b720200"
        matched = script._match_response_to_input(response, url)
        assert matched is not None
        assert matched["current_employers"][0]["employer_name"] == "Cytactic"

    def test_falls_back_to_flagship_slug(self, script):
        response = [
            {"linkedin_flagship_url": "https://linkedin.com/in/barak-ben-shimon",
             "current_employers": [{"employer_name": "monday.com"}]},
        ]
        url = "https://www.linkedin.com/in/barak-ben-shimon"
        matched = script._match_response_to_input(response, url)
        assert matched is not None
        assert matched["current_employers"][0]["employer_name"] == "monday.com"

    def test_returns_none_when_no_match(self, script):
        response = [
            {"query_linkedin_profile_urn_or_slug": ["other-person"]},
            {"query_linkedin_profile_urn_or_slug": ["yet-another"]},
        ]
        assert script._match_response_to_input(
            response, "https://www.linkedin.com/in/nobody"
        ) is None

    def test_single_response_takes_when_unique(self, script):
        response = [{"current_employers": [{"employer_name": "Sole"}]}]
        matched = script._match_response_to_input(
            response, "https://www.linkedin.com/in/someone"
        )
        assert matched is not None


# ---------------------------------------------------------------------------
# Dry-run flow (main entry point)
# ---------------------------------------------------------------------------


class TestDryRunFlow:

    def _make_client(self, script):
        # Build rows old enough to trip the filter.
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        rows = [
            {
                "linkedin_url": "https://www.linkedin.com/in/gil-gitlin",
                "name": "Gil Gitlin",
                "enriched_at": old,
                "raw_data": {"current_employers": [{"employer_name": "Unit 8200"}]},
            },
            {
                "linkedin_url": "https://www.linkedin.com/in/barak-ben-shimon",
                "name": "Barak Ben Shimon",
                "enriched_at": old,
                "raw_data": {"current_employers": [{"employer_name": "Vesttoo"}]},
            },
            {
                "linkedin_url": "https://www.linkedin.com/in/dense-person",
                "name": "Dense Person",
                "enriched_at": old,
                "raw_data": {"current_employers": [
                    {"employer_name": "A"}, {"employer_name": "B"}
                ]},
            },
        ]
        return FakeSupabaseClient(rows=rows)

    def test_dry_run_prints_candidates_and_does_not_write(self, script, capsys):
        client = self._make_client(script)
        with patch.object(script, "get_supabase_client", return_value=client), \
             patch.object(script, "crustdata_enrich_batch") as mock_enrich, \
             patch.object(script, "_load_crustdata_api_key") as mock_key:
            rc = script.main(["--limit", "10", "--max-age-days", "30"])

        out = capsys.readouterr().out
        assert rc == 0
        # Dry run must not call Crustdata or read the API key.
        mock_enrich.assert_not_called()
        mock_key.assert_not_called()
        # Must not write anything.
        assert client.upsert_calls == []
        # Output mentions the two sparse profiles and excludes the dense one.
        assert "Gil Gitlin" in out
        assert "Barak Ben Shimon" in out
        assert "Dense Person" not in out
        assert "DRY-RUN" in out
        assert "credits" in out.lower()

    def test_dry_run_zero_limit_returns_error(self, script, capsys):
        client = self._make_client(script)
        with patch.object(script, "get_supabase_client", return_value=client):
            rc = script.main(["--limit", "0"])
        assert rc == 2
        captured = capsys.readouterr()
        assert "limit" in captured.err.lower()

    def test_missing_supabase_client_returns_error(self, script, capsys):
        with patch.object(script, "get_supabase_client", return_value=None):
            rc = script.main(["--limit", "5"])
        assert rc == 2
        assert "supabase" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# Execute flow (mocked Crustdata + DB)
# ---------------------------------------------------------------------------


class TestExecuteFlow:

    def test_execute_calls_crustdata_and_writes(self, script, capsys):
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        rows = [{
            "linkedin_url": "https://www.linkedin.com/in/gil-gitlin",
            "name": "Gil Gitlin",
            "enriched_at": old,
            "raw_data": {"current_employers": [{"employer_name": "Unit 8200"}]},
            "current_company": "Unit 8200",
            "current_title": "Full-stack Developer",
        }]
        client = FakeSupabaseClient(rows=rows)

        fake_enrich_response = [{
            "query_linkedin_profile_urn_or_slug": ["gil-gitlin"],
            "linkedin_flagship_url": "https://www.linkedin.com/in/gil-gitlin",
            "first_name": "Gil",
            "last_name": "Gitlin",
            "current_employers": [
                {"employer_name": "Unit 8200", "start_date": "2019-01-01"},
                {"employer_name": "Cytactic", "employee_title": "Senior Software Engineer",
                 "start_date": "2025-05-01"},
            ],
        }]

        with patch.object(script, "get_supabase_client", return_value=client), \
             patch.object(script, "_load_crustdata_api_key", return_value="test-key"), \
             patch.object(script, "crustdata_enrich_batch", return_value=fake_enrich_response) as mock_enrich, \
             patch.object(script, "save_enriched_profile") as mock_save:
            rc = script.main(["--limit", "5", "--max-age-days", "30", "--execute"])

        assert rc == 0
        mock_enrich.assert_called_once()
        # save_enriched_profile is called with the matched payload + url.
        mock_save.assert_called_once()
        args, kwargs = mock_save.call_args
        assert args[1] == "https://www.linkedin.com/in/gil-gitlin"
        assert args[2] is fake_enrich_response[0]
        out = capsys.readouterr().out
        assert "SUMMARY" in out
        assert "Updated" in out

    def test_execute_with_no_candidates_short_circuits(self, script, capsys):
        client = FakeSupabaseClient(rows=[])
        with patch.object(script, "get_supabase_client", return_value=client), \
             patch.object(script, "crustdata_enrich_batch") as mock_enrich, \
             patch.object(script, "_load_crustdata_api_key") as mock_key:
            rc = script.main(["--limit", "5", "--execute"])
        assert rc == 0
        mock_enrich.assert_not_called()
        mock_key.assert_not_called()
        assert "nothing to do" in capsys.readouterr().out.lower()

    def test_execute_records_failure_when_crustdata_raises(self, script):
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        rows = [{
            "linkedin_url": "https://www.linkedin.com/in/some-person",
            "name": "Some Person",
            "enriched_at": old,
            "raw_data": {"current_employers": [{"employer_name": "OldCo"}]},
        }]
        client = FakeSupabaseClient(rows=rows)

        def boom(*args, **kwargs):
            raise RuntimeError("crustdata down")

        candidates, _ = script.fetch_sparse_candidates(
            client, max_age_days=30, limit=5, min_current_employers=1,
            prefer_screened_days=0,
        )
        with patch.object(script, "crustdata_enrich_batch", side_effect=boom):
            stats = script.process_candidates(
                client, candidates, api_key="test-key", batch_size=5
            )
        assert stats["failed"] == 1
        assert stats["updated"] == 0
        assert "crustdata" in stats["failures"][0]["error"].lower()


# ---------------------------------------------------------------------------
# Smoke: process_candidates updates change list correctly
# ---------------------------------------------------------------------------


class TestProcessCandidatesChangeDetection:

    def test_updated_when_current_company_changes(self, script):
        rows = [{
            "linkedin_url": "https://www.linkedin.com/in/x",
            "name": "X",
            "current_company": "Old Co",
            "current_title": "Old Title",
            "raw_data": {},
        }]
        client = FakeSupabaseClient(rows=rows)
        candidates = [{
            "linkedin_url": "https://www.linkedin.com/in/x",
            "name": "X",
            "enriched_at": "2026-01-01T00:00:00Z",
            "current_employers_count": 1,
        }]

        fake_response = [{
            "query_linkedin_profile_urn_or_slug": ["x"],
            "current_employers": [
                {"employer_name": "New Co", "employee_title": "New Title",
                 "start_date": "2025-05-01"},
            ],
        }]

        with patch.object(script, "crustdata_enrich_batch", return_value=fake_response), \
             patch.object(script, "save_enriched_profile") as mock_save:
            stats = script.process_candidates(
                client, candidates, api_key="test-key", batch_size=5
            )
        assert stats["processed"] == 1
        assert stats["updated"] == 1
        assert stats["unchanged"] == 0
        assert stats["failed"] == 0
        assert stats["changes"][0]["after"].endswith("New Co")
        mock_save.assert_called_once()

    def test_unchanged_when_current_company_same(self, script):
        rows = [{
            "linkedin_url": "https://www.linkedin.com/in/y",
            "name": "Y",
            "current_company": "Same Co",
            "current_title": "Eng",
            "raw_data": {},
        }]
        client = FakeSupabaseClient(rows=rows)
        candidates = [{
            "linkedin_url": "https://www.linkedin.com/in/y",
            "name": "Y",
            "enriched_at": "2026-01-01T00:00:00Z",
            "current_employers_count": 1,
        }]

        fake_response = [{
            "query_linkedin_profile_urn_or_slug": ["y"],
            "current_employers": [
                {"employer_name": "Same Co", "employee_title": "Eng",
                 "start_date": "2024-01-01"},
            ],
        }]

        with patch.object(script, "crustdata_enrich_batch", return_value=fake_response), \
             patch.object(script, "save_enriched_profile") as mock_save:
            stats = script.process_candidates(
                client, candidates, api_key="test-key", batch_size=5
            )
        assert stats["processed"] == 1
        assert stats["updated"] == 0
        assert stats["unchanged"] == 1
        assert stats["failed"] == 0
        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Phase A (recently-screened) / Phase B (oldest-sparse fill) selection
# ---------------------------------------------------------------------------


class _StubbedSelectClient:
    """Supabase stub that returns a different fixture per phase.

    Phase A is the first select(...) call (predicate includes screened_at);
    Phase B is the second. Lets us assert each query's filters independently
    of what rows the other phase returned.
    """

    def __init__(self, phase_a_rows, phase_b_rows):
        self.phase_a_rows = phase_a_rows
        self.phase_b_rows = phase_b_rows
        self.select_calls = []

    def select(self, table, columns="*", filters=None, limit=5000,
               order_by=None, cursor_column=None, cursor_value=None):
        self.select_calls.append({
            "table": table,
            "columns": columns,
            "filters": dict(filters or {}),
            "limit": limit,
            "order_by": order_by,
        })
        # Phase A is identified by the screened_at predicate.
        if filters and "screened_at" in filters:
            return list(self.phase_a_rows)[:limit]
        return list(self.phase_b_rows)[:limit]


def _sparse_row(url, name, screened_at=None, enriched_iso="2024-01-01T00:00:00Z"):
    return {
        "linkedin_url": url,
        "name": name,
        "enriched_at": enriched_iso,
        "screened_at": screened_at,
        "raw_data": {"current_employers": [{"employer_name": "OldCo"}]},
    }


class TestPhaseSelection:

    def test_phase_a_predicates_use_screened_at_not_contacted_at(self, script):
        """Phase A must filter on screened_at AND enriched_at, never contacted_at."""
        now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
        client = _StubbedSelectClient(phase_a_rows=[], phase_b_rows=[])
        script.fetch_sparse_candidates(
            client,
            max_age_days=60,
            limit=10,
            min_current_employers=1,
            prefer_screened_days=90,
            now=now,
        )
        assert client.select_calls, "expected at least one select call"
        phase_a = client.select_calls[0]
        flt = phase_a["filters"]
        assert "screened_at" in flt
        assert flt["screened_at"].startswith("gte.")
        assert "enriched_at" in flt
        assert flt["enriched_at"].startswith("lt.")
        # 90 days before 2026-05-11T12:00:00Z = 2026-02-10T12:00:00Z
        assert flt["screened_at"] == "gte.2026-02-10T12:00:00Z"
        # contacted_at must NEVER appear in the filter or the serialized query.
        for call in client.select_calls:
            assert "contacted_at" not in call["filters"]
            assert "contacted_at" not in repr(call["filters"])

    def test_phase_a_results_come_first(self, script):
        now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
        recent = "2026-04-15T00:00:00Z"  # within 90d of now
        phase_a_rows = [
            _sparse_row("https://www.linkedin.com/in/a1", "A1", screened_at=recent),
            _sparse_row("https://www.linkedin.com/in/a2", "A2", screened_at=recent),
        ]
        phase_b_rows = [
            _sparse_row("https://www.linkedin.com/in/b1", "B1"),
            _sparse_row("https://www.linkedin.com/in/b2", "B2"),
        ]
        client = _StubbedSelectClient(phase_a_rows=phase_a_rows, phase_b_rows=phase_b_rows)

        candidates, breakdown = script.fetch_sparse_candidates(
            client, max_age_days=60, limit=4, min_current_employers=1,
            prefer_screened_days=90, now=now,
        )
        urls = [c["linkedin_url"] for c in candidates]
        # Phase A first, then Phase B.
        assert urls[:2] == [
            "https://www.linkedin.com/in/a1",
            "https://www.linkedin.com/in/a2",
        ]
        assert set(urls[2:]) == {
            "https://www.linkedin.com/in/b1",
            "https://www.linkedin.com/in/b2",
        }
        assert breakdown == {"phase_a": 2, "phase_b": 2, "total": 4}

    def test_phase_b_skipped_when_phase_a_fills_limit(self, script):
        now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
        recent = "2026-04-15T00:00:00Z"
        phase_a_rows = [
            _sparse_row(f"https://www.linkedin.com/in/a{i}", f"A{i}", screened_at=recent)
            for i in range(3)
        ]
        # If Phase B is queried, surface a poison row so the test fails loudly.
        phase_b_rows = [_sparse_row("https://www.linkedin.com/in/SHOULD_NOT_APPEAR", "X")]
        client = _StubbedSelectClient(phase_a_rows=phase_a_rows, phase_b_rows=phase_b_rows)

        candidates, breakdown = script.fetch_sparse_candidates(
            client, max_age_days=60, limit=3, min_current_employers=1,
            prefer_screened_days=90, now=now,
        )
        assert len(candidates) == 3
        assert all("SHOULD_NOT_APPEAR" not in c["linkedin_url"] for c in candidates)
        # Only one select call was issued — the Phase A one.
        assert len(client.select_calls) == 1
        assert "screened_at" in client.select_calls[0]["filters"]
        assert breakdown == {"phase_a": 3, "phase_b": 0, "total": 3}

    def test_phase_b_fills_remainder_and_excludes_phase_a_urls(self, script):
        now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
        recent = "2026-04-15T00:00:00Z"
        phase_a_rows = [
            _sparse_row("https://www.linkedin.com/in/a1", "A1", screened_at=recent),
        ]
        # Phase B fixture deliberately includes the same URL as Phase A so we
        # can verify the dedupe path. The real DB would also exclude it via
        # the not.in.(...) predicate; the test checks both layers.
        phase_b_rows = [
            _sparse_row("https://www.linkedin.com/in/a1", "A1 dup"),
            _sparse_row("https://www.linkedin.com/in/b1", "B1"),
            _sparse_row("https://www.linkedin.com/in/b2", "B2"),
        ]
        client = _StubbedSelectClient(phase_a_rows=phase_a_rows, phase_b_rows=phase_b_rows)

        candidates, breakdown = script.fetch_sparse_candidates(
            client, max_age_days=60, limit=3, min_current_employers=1,
            prefer_screened_days=90, now=now,
        )
        urls = [c["linkedin_url"] for c in candidates]
        assert urls[0] == "https://www.linkedin.com/in/a1"
        # The Phase A URL must not appear a second time.
        assert urls.count("https://www.linkedin.com/in/a1") == 1
        assert set(urls[1:]) == {
            "https://www.linkedin.com/in/b1",
            "https://www.linkedin.com/in/b2",
        }
        assert breakdown == {"phase_a": 1, "phase_b": 2, "total": 3}

        # The Phase B query must carry a not.in.(...) predicate on linkedin_url
        # that lists the Phase A URL.
        phase_b_call = client.select_calls[1]
        assert "linkedin_url" in phase_b_call["filters"]
        not_in = phase_b_call["filters"]["linkedin_url"]
        assert not_in.startswith("not.in.(")
        assert "https://www.linkedin.com/in/a1" in not_in
        # Phase B query must NOT carry a screened_at predicate.
        assert "screened_at" not in phase_b_call["filters"]

    def test_prefer_screened_days_zero_disables_phase_a(self, script):
        now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
        client = _StubbedSelectClient(phase_a_rows=[], phase_b_rows=[
            _sparse_row("https://www.linkedin.com/in/b1", "B1"),
        ])
        candidates, breakdown = script.fetch_sparse_candidates(
            client, max_age_days=60, limit=5, min_current_employers=1,
            prefer_screened_days=0, now=now,
        )
        assert len(client.select_calls) == 1
        assert "screened_at" not in client.select_calls[0]["filters"]
        assert breakdown["phase_a"] == 0
        assert len(candidates) == 1

    def test_custom_prefer_screened_days_used_in_cutoff(self, script):
        """Non-default --prefer-screened-days flows into the screened_at cutoff."""
        now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
        client = _StubbedSelectClient(phase_a_rows=[], phase_b_rows=[])
        script.fetch_sparse_candidates(
            client, max_age_days=60, limit=5, min_current_employers=1,
            prefer_screened_days=30, now=now,
        )
        # 30 days before 2026-05-11T12:00:00Z = 2026-04-11T12:00:00Z
        assert client.select_calls[0]["filters"]["screened_at"] == \
            "gte.2026-04-11T12:00:00Z"

    def test_phase_breakdown_printed_in_dry_run(self, script, capsys):
        # Use very recent screened_at relative to real "now" so Phase A keeps
        # the row even though main() calls datetime.now() internally.
        recent = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        phase_a_rows = [
            _sparse_row("https://www.linkedin.com/in/a1", "A1", screened_at=recent),
        ]
        phase_b_rows = [
            _sparse_row("https://www.linkedin.com/in/b1", "B1"),
        ]
        client = _StubbedSelectClient(phase_a_rows=phase_a_rows, phase_b_rows=phase_b_rows)

        with patch.object(script, "get_supabase_client", return_value=client):
            rc = script.main([
                "--limit", "5", "--max-age-days", "60",
                "--prefer-screened-days", "90",
            ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Phase A" in out
        assert "Phase B" in out
        assert "Total:" in out
