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
    touching Supabase.
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
        # Honour the linkedin_url eq filter for the "before" pre-write read.
        if filters and any(k == "linkedin_url" and str(v).startswith("eq.")
                           for k, v in filters.items()):
            url = next(v[3:] for k, v in filters.items() if k == "linkedin_url")
            return [r for r in self.rows if r.get("linkedin_url") == url][:1]
        return list(self.rows)[:limit]

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
        assert args.execute is False
        assert args.verbose is False

    def test_all_flags(self, script):
        parser = script.build_arg_parser()
        args = parser.parse_args([
            "--max-age-days", "14",
            "--limit", "100",
            "--min-current-employers", "0",
            "--execute",
            "--verbose",
            "--batch-size", "3",
        ])
        assert args.max_age_days == 14
        assert args.limit == 100
        assert args.min_current_employers == 0
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
        candidates = script.fetch_sparse_candidates(
            client, max_age_days=30, limit=10, min_current_employers=1
        )
        urls = [c["linkedin_url"] for c in candidates]
        assert "https://www.linkedin.com/in/sparse-1" in urls
        assert "https://www.linkedin.com/in/sparse-2" in urls
        assert "https://www.linkedin.com/in/dense-1" not in urls

    def test_min_current_employers_zero_excludes_one(self, script):
        client = FakeSupabaseClient(rows=[
            self._row("https://www.linkedin.com/in/sparse-1", "Sparse One", 60, 1),
            self._row("https://www.linkedin.com/in/sparse-2", "Sparse Two", 60, 0),
        ])
        candidates = script.fetch_sparse_candidates(
            client, max_age_days=30, limit=10, min_current_employers=0
        )
        urls = [c["linkedin_url"] for c in candidates]
        assert urls == ["https://www.linkedin.com/in/sparse-2"]

    def test_limit_is_respected(self, script):
        rows = [
            self._row(f"https://www.linkedin.com/in/p{i}", f"P {i}", 60, 1)
            for i in range(20)
        ]
        client = FakeSupabaseClient(rows=rows)
        candidates = script.fetch_sparse_candidates(
            client, max_age_days=30, limit=3, min_current_employers=1
        )
        assert len(candidates) == 3

    def test_empty_when_zero_limit(self, script):
        client = FakeSupabaseClient(rows=[
            self._row("https://www.linkedin.com/in/sparse-1", "Sparse One", 60, 1),
        ])
        assert script.fetch_sparse_candidates(
            client, max_age_days=30, limit=0, min_current_employers=1
        ) == []

    def test_cutoff_filter_passed_to_select(self, script):
        now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
        client = FakeSupabaseClient(rows=[])
        script.fetch_sparse_candidates(
            client, max_age_days=10, limit=5, min_current_employers=1, now=now
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

        candidates = script.fetch_sparse_candidates(
            client, max_age_days=30, limit=5, min_current_employers=1
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
