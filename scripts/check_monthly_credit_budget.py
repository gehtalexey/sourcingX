"""
Monthly Crustdata credit budget preflight check.

Used by the daily ``db-refresh`` GitHub Actions workflow to decide whether the
next auto-refresh run is allowed to spend credits. Reads month-to-date (MTD)
spend from Supabase's ``api_usage_logs`` table and compares it against the
configured monthly cap.

Why this script is conservative
-------------------------------
``api_usage_logs`` records every Crustdata credit charged anywhere in the
system — Tab 0 search, Tab 1 enrich, the dashboard, and this auto-refresh
job all write to the same table. We do NOT scope the MTD sum to the
auto-refresh job specifically; we sum **all** Crustdata credits month-to-
date and treat the cap as a shared budget. That is intentional:

  * It's the only spend signal we have that's authoritative across paths.
  * Auto-refresh is the low-priority consumer; if humans burn the monthly
    budget on manual searches, the auto-refresh should yield, not race.

If, later, we want a separate auto-refresh sub-budget, add an
``operation`` column filter (e.g. ``operation=eq.reenrich_sparse``) once
the writer tags its rows that way. Today the writer in
``usage_tracker.py`` doesn't set a distinct operation for this job, so a
sub-budget would silently always read zero — worse than the conservative
shared-budget behaviour we have here.

Schema reference
----------------
From ``db.py`` / ``usage_tracker.py``:

  * Table: ``api_usage_logs``
  * Columns used: ``provider`` (lowercased string, e.g. ``"crustdata"``),
    ``credits_used`` (numeric), ``created_at`` (ISO timestamp).

CLI
---
    python scripts/check_monthly_credit_budget.py \\
        --cap 22000 --projected-spend 750 [--provider crustdata]

Output (stdout, JSON):
    {
      "mtd_spend": 12345.0,
      "cap": 22000,
      "projected_spend": 750,
      "projected_total": 13095.0,
      "should_proceed": true,
      "reason": "..."
    }

Exit codes
----------
  * 0   — proceed (``should_proceed=true``)
  * 75  — skip (``should_proceed=false``); ``EX_TEMPFAIL`` from sysexits.h.
          The workflow checks for this and bails out of the run.
  * 2   — bad CLI args or missing env / Supabase connection failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Make repo root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db import get_supabase_client  # noqa: E402


# Exit code aliases for readability. ``EX_TEMPFAIL`` (75) matches the BSD
# sysexits.h convention for "try again later" — the workflow uses this exit
# code as the "skip the run" signal, distinct from a hard failure (2).
EXIT_PROCEED = 0
EXIT_SKIP = 75
EXIT_ERROR = 2


def month_start_utc(now: Optional[datetime] = None) -> datetime:
    """Return the first instant of the current calendar month in UTC."""
    base = now if now is not None else datetime.now(timezone.utc)
    return datetime(base.year, base.month, 1, tzinfo=timezone.utc)


def fetch_mtd_spend(client, provider: str, now: Optional[datetime] = None) -> float:
    """Sum ``credits_used`` for ``provider`` since the start of this month (UTC).

    Reads via the project's ``SupabaseClient.select`` because the REST API
    doesn't expose SUM() directly. We pull the rows for the current month
    (a single-digit-thousands volume) and sum client-side.
    """
    cutoff = month_start_utc(now=now)
    cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = client.select(
        "api_usage_logs",
        columns="credits_used,created_at,provider",
        filters={
            "provider": f"eq.{provider.lower()}",
            "created_at": f"gte.{cutoff_iso}",
        },
        # Same defensive ceiling pattern as get_usage_summary in db.py.
        limit=10000,
        order_by="created_at.asc",
    )

    total = 0.0
    for row in rows or []:
        credits = row.get("credits_used")
        if credits is None:
            continue
        try:
            total += float(credits)
        except (TypeError, ValueError):
            # Skip malformed rows rather than crashing the preflight.
            continue
    return total


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_monthly_credit_budget",
        description=(
            "Preflight: decide whether the next Crustdata auto-refresh run "
            "fits inside the monthly credit cap. Exits 0 to proceed, 75 to "
            "skip, 2 on misconfiguration."
        ),
    )
    parser.add_argument(
        "--cap",
        type=int,
        required=True,
        help="Monthly cap in Crustdata credits (e.g. 22000).",
    )
    parser.add_argument(
        "--projected-spend",
        type=int,
        required=True,
        help="Credits this run is expected to spend (e.g. 750 for limit=250).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="crustdata",
        help="Provider name in api_usage_logs to scope the SUM to (default: crustdata).",
    )
    return parser


def evaluate(
    mtd_spend: float,
    cap: int,
    projected_spend: int,
) -> Dict[str, Any]:
    """Pure decision logic — easy to unit-test without a Supabase client."""
    projected_total = mtd_spend + projected_spend
    if cap <= 0:
        return {
            "mtd_spend": mtd_spend,
            "cap": cap,
            "projected_spend": projected_spend,
            "projected_total": projected_total,
            "should_proceed": False,
            "reason": f"cap is {cap}; refusing to proceed",
        }
    if projected_total > cap:
        return {
            "mtd_spend": mtd_spend,
            "cap": cap,
            "projected_spend": projected_spend,
            "projected_total": projected_total,
            "should_proceed": False,
            "reason": (
                f"mtd_spend ({mtd_spend:.0f}) + projected ({projected_spend}) "
                f"= {projected_total:.0f} > cap ({cap}); skip"
            ),
        }
    return {
        "mtd_spend": mtd_spend,
        "cap": cap,
        "projected_spend": projected_spend,
        "projected_total": projected_total,
        "should_proceed": True,
        "reason": (
            f"mtd_spend ({mtd_spend:.0f}) + projected ({projected_spend}) "
            f"= {projected_total:.0f} <= cap ({cap}); proceed"
        ),
    }


def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.cap < 0 or args.projected_spend < 0:
        print(
            "ERROR: --cap and --projected-spend must both be non-negative.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    # Validate Supabase env up front — clearer error than letting select() 500.
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_KEY"):
        print(
            "ERROR: SUPABASE_URL and SUPABASE_KEY must both be set in the environment.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    client = get_supabase_client()
    if client is None:
        print(
            "ERROR: Could not build Supabase client (missing or invalid SUPABASE_URL/SUPABASE_KEY).",
            file=sys.stderr,
        )
        return EXIT_ERROR

    try:
        mtd_spend = fetch_mtd_spend(client, provider=args.provider)
    except Exception as e:
        # Don't echo the URL/key in the error message — they're in env.
        print(
            f"ERROR: failed to read api_usage_logs from Supabase: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return EXIT_ERROR

    result = evaluate(
        mtd_spend=mtd_spend,
        cap=args.cap,
        projected_spend=args.projected_spend,
    )

    # JSON to stdout so the workflow can parse it.
    print(json.dumps(result))

    return EXIT_PROCEED if result["should_proceed"] else EXIT_SKIP


if __name__ == "__main__":
    sys.exit(main())
