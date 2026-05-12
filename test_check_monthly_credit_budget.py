"""Tests for scripts/check_monthly_credit_budget.py.

All external services are mocked. No real network calls.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Module loader (the script lives under scripts/, not on sys.path by default)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_monthly_credit_budget.py"


def _load_script():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "check_monthly_credit_budget", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


# ---------------------------------------------------------------------------
# Fake Supabase client
# ---------------------------------------------------------------------------


class FakeSupabaseClient:
    """Returns a fixed list of usage rows. Records the filters it was asked for."""

    def __init__(self, rows):
        self.rows = rows
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
        return list(self.rows)


def _rows_for_total(total_credits):
    """Build a single row holding ``total_credits`` so fetch_mtd_spend returns it."""
    return [{"credits_used": total_credits, "created_at": "2026-05-01T00:00:00Z",
             "provider": "crustdata"}]


# ---------------------------------------------------------------------------
# Pure evaluate() — no Supabase needed
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_zero_mtd_under_cap_proceeds(self, script):
        result = script.evaluate(mtd_spend=0, cap=22000, projected_spend=750)
        assert result["should_proceed"] is True
        assert result["projected_total"] == 750
        assert result["mtd_spend"] == 0
        assert result["cap"] == 22000

    def test_high_mtd_still_fits_proceeds(self, script):
        # 21000 + 750 = 21750 < 22000 -> proceed
        result = script.evaluate(mtd_spend=21000, cap=22000, projected_spend=750)
        assert result["should_proceed"] is True
        assert result["projected_total"] == 21750

    def test_projected_exceeds_cap_skips(self, script):
        # 21500 + 750 = 22250 > 22000 -> skip
        result = script.evaluate(mtd_spend=21500, cap=22000, projected_spend=750)
        assert result["should_proceed"] is False
        assert result["projected_total"] == 22250
        assert "skip" in result["reason"].lower()

    def test_exactly_at_cap_proceeds(self, script):
        # Boundary: equal-to-cap is allowed (cap is "do not exceed").
        result = script.evaluate(mtd_spend=21250, cap=22000, projected_spend=750)
        assert result["should_proceed"] is True
        assert result["projected_total"] == 22000

    def test_zero_cap_refuses(self, script):
        result = script.evaluate(mtd_spend=0, cap=0, projected_spend=750)
        assert result["should_proceed"] is False
        assert "cap" in result["reason"].lower()


# ---------------------------------------------------------------------------
# month_start_utc()
# ---------------------------------------------------------------------------


class TestMonthStart:
    def test_returns_first_of_month_utc(self, script):
        now = datetime(2026, 5, 12, 18, 30, 0, tzinfo=timezone.utc)
        start = script.month_start_utc(now=now)
        assert start.year == 2026
        assert start.month == 5
        assert start.day == 1
        assert start.hour == 0
        assert start.minute == 0
        assert start.tzinfo is not None


# ---------------------------------------------------------------------------
# fetch_mtd_spend()
# ---------------------------------------------------------------------------


class TestFetchMtdSpend:
    def test_sums_credits_used(self, script):
        client = FakeSupabaseClient([
            {"credits_used": 100, "created_at": "2026-05-02T01:00:00Z", "provider": "crustdata"},
            {"credits_used": 250, "created_at": "2026-05-03T01:00:00Z", "provider": "crustdata"},
            {"credits_used": 50.5, "created_at": "2026-05-04T01:00:00Z", "provider": "crustdata"},
        ])
        total = script.fetch_mtd_spend(client, provider="crustdata")
        assert total == pytest.approx(400.5)

    def test_skips_null_and_malformed(self, script):
        client = FakeSupabaseClient([
            {"credits_used": 100, "created_at": "2026-05-02T01:00:00Z", "provider": "crustdata"},
            {"credits_used": None, "created_at": "2026-05-03T01:00:00Z", "provider": "crustdata"},
            {"credits_used": "not-a-number", "created_at": "2026-05-04T01:00:00Z", "provider": "crustdata"},
            {"created_at": "2026-05-05T01:00:00Z", "provider": "crustdata"},  # no credits field
        ])
        total = script.fetch_mtd_spend(client, provider="crustdata")
        assert total == 100

    def test_filters_passed_correctly(self, script):
        client = FakeSupabaseClient([])
        now = datetime(2026, 5, 12, 18, 30, 0, tzinfo=timezone.utc)
        script.fetch_mtd_spend(client, provider="crustdata", now=now)
        call = client.select_calls[0]
        assert call["table"] == "api_usage_logs"
        assert call["filters"]["provider"] == "eq.crustdata"
        # Must filter from the first instant of the current month.
        assert call["filters"]["created_at"] == "gte.2026-05-01T00:00:00Z"

    def test_empty_rows_returns_zero(self, script):
        client = FakeSupabaseClient([])
        total = script.fetch_mtd_spend(client, provider="crustdata")
        assert total == 0


# ---------------------------------------------------------------------------
# main() integration — patches get_supabase_client + env
# ---------------------------------------------------------------------------


def _run_main(script, argv, *, env, rows=None, client=None, raise_on_select=None):
    """Run script.main() with patched env + Supabase client. Returns (exit_code, stdout, stderr)."""
    import os

    old_env = {k: os.environ.get(k) for k in ("SUPABASE_URL", "SUPABASE_KEY")}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

        if client is None and rows is not None:
            client = FakeSupabaseClient(rows)
            if raise_on_select:
                def _raise(*a, **kw):
                    raise raise_on_select
                client.select = _raise  # type: ignore[assignment]

        orig = script.get_supabase_client
        script.get_supabase_client = lambda: client  # type: ignore[assignment]

        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = script.main(argv)
        finally:
            script.get_supabase_client = orig  # type: ignore[assignment]
        return code, stdout.getvalue(), stderr.getvalue()
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestMainIntegration:
    def test_proceeds_with_zero_mtd(self, script):
        code, out, _err = _run_main(
            script,
            ["--cap", "22000", "--projected-spend", "750"],
            env={"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_KEY": "fake-key"},
            rows=[],
        )
        payload = json.loads(out.strip().splitlines()[-1])
        assert code == 0
        assert payload["should_proceed"] is True
        assert payload["mtd_spend"] == 0

    def test_proceeds_when_high_mtd_still_fits(self, script):
        code, out, _err = _run_main(
            script,
            ["--cap", "22000", "--projected-spend", "750"],
            env={"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_KEY": "fake-key"},
            rows=_rows_for_total(21000),
        )
        payload = json.loads(out.strip().splitlines()[-1])
        assert code == 0
        assert payload["should_proceed"] is True
        assert payload["projected_total"] == 21750

    def test_skips_when_projected_exceeds_cap(self, script):
        code, out, _err = _run_main(
            script,
            ["--cap", "22000", "--projected-spend", "750"],
            env={"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_KEY": "fake-key"},
            rows=_rows_for_total(21500),
        )
        payload = json.loads(out.strip().splitlines()[-1])
        assert code == 75  # EX_TEMPFAIL — workflow uses this to decide to skip
        assert payload["should_proceed"] is False
        assert payload["projected_total"] == 22250

    def test_zero_cap_refuses(self, script):
        code, out, _err = _run_main(
            script,
            ["--cap", "0", "--projected-spend", "750"],
            env={"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_KEY": "fake-key"},
            rows=[],
        )
        payload = json.loads(out.strip().splitlines()[-1])
        assert code == 75
        assert payload["should_proceed"] is False

    def test_missing_env_errors_clearly(self, script):
        code, _out, err = _run_main(
            script,
            ["--cap", "22000", "--projected-spend", "750"],
            env={"SUPABASE_URL": None, "SUPABASE_KEY": None},
            rows=[],
        )
        assert code == 2
        assert "SUPABASE_URL" in err and "SUPABASE_KEY" in err

    def test_partial_env_errors_clearly(self, script):
        code, _out, err = _run_main(
            script,
            ["--cap", "22000", "--projected-spend", "750"],
            env={"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_KEY": None},
            rows=[],
        )
        assert code == 2
        assert "SUPABASE" in err

    def test_supabase_client_none_errors(self, script):
        code, _out, err = _run_main(
            script,
            ["--cap", "22000", "--projected-spend", "750"],
            env={"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_KEY": "fake-key"},
            client=None,  # forces get_supabase_client() to return None
        )
        # Note: with no rows and no client supplied, our helper passes client=None.
        # In that case _run_main injects ``get_supabase_client = lambda: None``.
        assert code == 2
        assert "Supabase" in err

    def test_supabase_read_error_is_reported(self, script):
        code, _out, err = _run_main(
            script,
            ["--cap", "22000", "--projected-spend", "750"],
            env={"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_KEY": "fake-key"},
            rows=[],
            raise_on_select=RuntimeError("network down"),
        )
        assert code == 2
        assert "api_usage_logs" in err
        assert "network down" in err

    def test_negative_args_rejected(self, script):
        code, _out, err = _run_main(
            script,
            ["--cap", "-1", "--projected-spend", "750"],
            env={"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_KEY": "fake-key"},
            rows=[],
        )
        assert code == 2
        assert "non-negative" in err
