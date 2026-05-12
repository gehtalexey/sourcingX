"""
Re-enrich sparse profiles in Supabase.

Targets profiles whose stored ``raw_data.current_employers`` only had <=1 entry
at the time of enrichment AND whose ``enriched_at`` is older than N days.
These are the rows that PR #22's SQL backfill could NOT fix, because the
backfill reads ``raw_data`` and there is nothing to pick from when only one
employer was captured. Real examples:

  - Gil Gitlin: stored as "Unit 8200"; today's Crustdata shows Cytactic
    (Senior Software Engineer since 2025-05).
  - Barak Ben Shimon: stored as "Vesttoo"; today's Crustdata shows monday.com
    (Senior Software Engineer since 2023-10).

For those rows we have to re-pull from Crustdata. This script:

  1) Selects candidates from Supabase (filter on enriched_at + array length).
  2) DRY-RUN by default — prints the count, the first 10, and the credit cost.
  3) On --execute, re-pulls each candidate from Crustdata in small batches,
     writes the new raw_data + indexed columns back via db.save_enriched_profile.

Safety defaults:
  - --limit is REQUIRED. The script refuses to run without it.
  - --execute is OPT-IN. Without it nothing is written and no Crustdata
    credits are spent.

Example invocations
-------------------

Dry-run (no credits, no writes):

    python scripts/reenrich_sparse_profiles.py --max-age-days 30 --limit 10

Real run (writes, spends credits):

    python scripts/reenrich_sparse_profiles.py --max-age-days 30 --limit 2 \
        --execute

Adjust how aggressive the sparsity filter is (default 1 catches Gil/Barak):

    python scripts/reenrich_sparse_profiles.py --max-age-days 30 --limit 100 \
        --min-current-employers 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

# Make repo root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_helpers import get_rate_limiter  # noqa: E402
from db import get_supabase_client, save_enriched_profile  # noqa: E402
from normalizers import normalize_crustdata_profile, pick_current_employer  # noqa: E402


CRUSTDATA_ENRICH_ENDPOINT = "https://api.crustdata.com/screener/person/enrich"

# Crustdata pricing for people-enrich. The repo's usage tracker records
# 3 credits/profile (see usage_tracker.CRUSTDATA_PRICING). This is the rate
# we use for the dry-run estimate; the live `crustdata_credits_check` endpoint
# is the source of truth if Crustdata ever changes pricing.
CREDITS_PER_ENRICH_PROFILE = 3

# Crustdata's enrich endpoint accepts a comma-separated batch. We keep the
# batch small for this maintenance job — the cost is the same, and small
# batches make per-profile logging cleaner.
DEFAULT_BATCH_SIZE = 5


# =============================================================================
# CONFIG / API KEY
# =============================================================================


def _load_crustdata_api_key() -> str:
    """Load the Crustdata API key from config.json or CRUSTDATA_API_KEY.

    Mirrors the pattern in ``enrich.py`` and ``crustdata_search.py``. Never
    prints the key.
    """
    config_path = REPO_ROOT / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                api_key = config.get("api_key")
                if api_key and api_key != "YOUR_CRUSTDATA_API_KEY_HERE":
                    return api_key
        except (json.JSONDecodeError, IOError):
            pass

    api_key = os.environ.get("CRUSTDATA_API_KEY")
    if api_key:
        return api_key

    raise RuntimeError(
        "No Crustdata API key found. Add `api_key` to config.json or set "
        "the CRUSTDATA_API_KEY environment variable."
    )


# =============================================================================
# CANDIDATE SELECTION
# =============================================================================


def _cutoff_iso(max_age_days: int, now: Optional[datetime] = None) -> str:
    """Return the ISO-8601 cutoff timestamp used in the Supabase filter."""
    base = now if now is not None else datetime.now(timezone.utc)
    cutoff = base - timedelta(days=max_age_days)
    # Supabase REST accepts ISO 8601 with timezone or 'Z'. Use 'Z' form for
    # consistency with how timestamps appear in the dashboard.
    return cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")


def _current_employers_count(raw_data: Any) -> int:
    """Count current_employers entries on a Crustdata raw_data blob.

    Tolerant of missing/None/malformed inputs. Returns 0 if the field is
    absent or not a list. Only counts dict entries (matches what
    pick_current_employer treats as "real" rows).
    """
    if not isinstance(raw_data, dict):
        return 0
    employers = raw_data.get("current_employers")
    if not isinstance(employers, list):
        return 0
    return sum(1 for e in employers if isinstance(e, dict))


def fetch_sparse_candidates(
    client,
    max_age_days: int,
    limit: int,
    min_current_employers: int,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Return profiles eligible for re-enrichment.

    Filters applied:
      - ``enriched_at < now() - max_age_days``
      - ``raw_data.current_employers`` length <= ``min_current_employers``
        (including missing/null, which count as 0).

    We do the array-length filter client-side because the Supabase REST API
    doesn't expose ``jsonb_array_length`` as a filter operator. We
    over-fetch by a small multiplier so we usually still hit the requested
    limit after filtering.
    """
    if limit <= 0:
        return []

    cutoff = _cutoff_iso(max_age_days, now=now)

    # Over-fetch to compensate for the post-filter pass. Cap to a sane
    # ceiling so we don't accidentally pull the whole table.
    fetch_limit = min(max(limit * 5, 50), 5000)

    rows = client.select(
        "profiles",
        columns="linkedin_url,name,enriched_at,raw_data",
        filters={"enriched_at": f"lt.{cutoff}"},
        limit=fetch_limit,
        order_by="enriched_at.asc",
    )

    candidates: List[Dict[str, Any]] = []
    for row in rows or []:
        count = _current_employers_count(row.get("raw_data"))
        if count <= min_current_employers:
            candidates.append({
                "linkedin_url": row.get("linkedin_url"),
                "name": row.get("name"),
                "enriched_at": row.get("enriched_at"),
                "current_employers_count": count,
            })
        if len(candidates) >= limit:
            break

    return candidates


# =============================================================================
# CRUSTDATA CALL
# =============================================================================


def crustdata_enrich_batch(
    urls: List[str],
    api_key: str,
    timeout: int = 120,
) -> List[Dict[str, Any]]:
    """Call Crustdata's people-enrich endpoint for a batch of LinkedIn URLs.

    Uses the same endpoint + auth header as enrich.py / dashboard.enrich_batch.
    We don't reuse those callers because enrich.py is a CLI main() and
    dashboard.enrich_batch depends on Streamlit session state.

    Returns the list of profile dicts as Crustdata returns them. On error,
    raises an exception so the caller can decide how to handle.
    """
    if not urls:
        return []

    limiter = get_rate_limiter("crustdata")
    limiter.wait_if_needed()
    try:
        response = requests.get(
            CRUSTDATA_ENRICH_ENDPOINT,
            params={"linkedin_profile_url": ",".join(urls)},
            headers={"Authorization": f"Token {api_key}"},
            timeout=timeout,
        )
    finally:
        limiter.record_request()

    if response.status_code != 200:
        # Don't echo the auth header. response.text is body only.
        raise RuntimeError(
            f"Crustdata enrich failed: HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    data = response.json()
    if isinstance(data, list):
        return data
    return [data] if isinstance(data, dict) else []


def _match_response_to_input(
    response: List[Dict[str, Any]],
    input_url: str,
) -> Optional[Dict[str, Any]]:
    """Find the Crustdata result that matches a given input URL.

    Crustdata echoes the input in ``query_linkedin_profile_urn_or_slug``. We
    extract the slug from the input URL and look for the response item whose
    echo matches. Falls back to ``linkedin_flagship_url`` slug.
    """
    def slug(url: Optional[str]) -> Optional[str]:
        if not url or "/in/" not in str(url).lower():
            return None
        return str(url).lower().split("/in/")[-1].rstrip("/").split("?")[0]

    target = slug(input_url)
    if not target:
        # Without a slug we can't match anything reliably.
        return response[0] if len(response) == 1 else None

    for item in response:
        if not isinstance(item, dict):
            continue
        query = item.get("query_linkedin_profile_urn_or_slug") or []
        if isinstance(query, list) and query:
            if str(query[0]).lower() == target:
                return item
        # Fallback: match on flagship URL slug.
        flag_slug = slug(item.get("linkedin_flagship_url") or item.get("linkedin_url"))
        if flag_slug and flag_slug == target:
            return item

    # Last resort: if there is exactly one response item, take it.
    if len(response) == 1 and isinstance(response[0], dict):
        return response[0]
    return None


# =============================================================================
# PROCESSING
# =============================================================================


def _summarize_employer(emp: Optional[Dict[str, Any]]) -> str:
    if not isinstance(emp, dict):
        return "—"
    title = emp.get("employee_title") or emp.get("title") or "—"
    name = emp.get("employer_name") or emp.get("company_name") or "—"
    return f"{title} @ {name}"


def process_candidates(
    client,
    candidates: List[Dict[str, Any]],
    api_key: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Re-enrich candidates and write the new data back.

    Returns a stats dict:
      {
        'processed': int,         # candidates we attempted
        'updated': int,           # rows where current_company actually changed
        'unchanged': int,         # re-enriched but current_company same as before
        'failed': int,            # API or DB errors
        'failures': [ {url, error}, ... ],
        'changes': [ {url, name, before, after}, ... ],
      }
    """
    stats: Dict[str, Any] = {
        "processed": 0,
        "updated": 0,
        "unchanged": 0,
        "failed": 0,
        "failures": [],
        "changes": [],
    }

    batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
    total_batches = len(batches)

    for batch_idx, batch in enumerate(batches, start=1):
        urls = [c["linkedin_url"] for c in batch if c.get("linkedin_url")]
        if not urls:
            continue

        if verbose:
            print(f"[batch {batch_idx}/{total_batches}] enriching {len(urls)} profile(s)...")

        try:
            response = crustdata_enrich_batch(urls, api_key)
        except Exception as e:
            for c in batch:
                stats["failed"] += 1
                stats["failures"].append({
                    "linkedin_url": c.get("linkedin_url"),
                    "name": c.get("name"),
                    "error": f"crustdata call failed: {str(e)[:200]}",
                })
            continue

        for c in batch:
            stats["processed"] += 1
            url = c["linkedin_url"]
            name = c.get("name") or "(no name)"

            matched = _match_response_to_input(response, url)
            if not matched or matched.get("error"):
                err = (matched or {}).get("error") if matched else "no match in response"
                stats["failed"] += 1
                stats["failures"].append({
                    "linkedin_url": url,
                    "name": name,
                    "error": f"no enrichment data: {err}",
                })
                if verbose:
                    print(f"  FAIL {name}: {err}")
                continue

            # Pull current_company "before" for change tracking.
            before_row = client.select(
                "profiles",
                columns="current_company,current_title,raw_data",
                filters={"linkedin_url": f"eq.{url}"},
                limit=1,
            )
            before_company = None
            before_title = None
            if before_row:
                before_company = before_row[0].get("current_company")
                before_title = before_row[0].get("current_title")

            # Compute "after" from the new Crustdata payload.
            new_emp = pick_current_employer(matched.get("current_employers"))
            after_company = (new_emp or {}).get("employer_name") or (new_emp or {}).get("company_name")
            after_title = (new_emp or {}).get("employee_title") or (new_emp or {}).get("title")

            try:
                save_enriched_profile(client, url, matched, original_url=url)
            except Exception as e:
                stats["failed"] += 1
                stats["failures"].append({
                    "linkedin_url": url,
                    "name": name,
                    "error": f"db write failed: {str(e)[:200]}",
                })
                if verbose:
                    print(f"  FAIL {name}: db write {e}")
                continue

            changed = bool(after_company) and (before_company != after_company)
            if changed:
                stats["updated"] += 1
                stats["changes"].append({
                    "linkedin_url": url,
                    "name": name,
                    "before": f"{before_title or '—'} @ {before_company or '—'}",
                    "after": _summarize_employer(new_emp),
                })
                if verbose:
                    print(f"  UPDATED {name}: {before_company!r} -> {after_company!r}")
            else:
                stats["unchanged"] += 1
                if verbose:
                    print(f"  unchanged {name} (current_company still {before_company!r})")

        # Be polite between batches (the rate limiter already throttles, but
        # this gives external services a beat).
        if batch_idx < total_batches:
            time.sleep(0.5)

    return stats


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reenrich_sparse_profiles",
        description=(
            "Re-enrich profiles whose stored raw_data.current_employers has "
            "<=N entries and whose enriched_at is older than --max-age-days. "
            "Dry-run by default; pass --execute to actually call Crustdata "
            "and write the results back. --limit is REQUIRED."
        ),
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Re-enrich profiles whose enriched_at is older than this many "
             "days (default: 30).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="REQUIRED. Cap how many profiles to process. The script refuses "
             "to run without this — there is no default.",
    )
    parser.add_argument(
        "--min-current-employers",
        type=int,
        default=1,
        help="Only re-enrich profiles whose raw_data.current_employers has at "
             "most this many entries (default: 1 catches the Gil/Barak class).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call Crustdata and write to Supabase. Without this "
             "flag the script lists candidates and prints the estimated "
             "credit cost only.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Crustdata batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-profile progress during --execute.",
    )
    return parser


def _print_dry_run(candidates: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    print(f"\n[DRY-RUN] Candidates matching filters:")
    print(f"  --max-age-days        = {args.max_age_days}")
    print(f"  --min-current-employers = {args.min_current_employers}")
    print(f"  --limit               = {args.limit}")
    print(f"  matched               = {len(candidates)} profile(s)")

    estimated_credits = len(candidates) * CREDITS_PER_ENRICH_PROFILE
    print(
        f"\nEstimated credit cost: {estimated_credits} credits "
        f"({CREDITS_PER_ENRICH_PROFILE} credits/profile, "
        f"confirm rate with `crustdata_credits_check`)."
    )

    if not candidates:
        print("\nNothing to do.")
        return

    print("\nFirst 10 candidates:")
    print(f"  {'name':<32} {'enriched_at':<22} {'#emp':<5} linkedin_url")
    for c in candidates[:10]:
        name = (c.get("name") or "(no name)")[:30]
        ts = (c.get("enriched_at") or "")[:19]
        cnt = c.get("current_employers_count", 0)
        url = c.get("linkedin_url") or ""
        print(f"  {name:<32} {ts:<22} {cnt:<5} {url}")

    print("\nRe-run with --execute to actually re-enrich these profiles.")


def _print_summary(stats: Dict[str, Any]) -> None:
    print("\n[SUMMARY]")
    print(f"  Processed: {stats['processed']}")
    print(f"  Updated (current_company changed): {stats['updated']}")
    print(f"  Unchanged: {stats['unchanged']}")
    print(f"  Failed: {stats['failed']}")

    if stats["changes"]:
        print("\nChanges:")
        for ch in stats["changes"][:25]:
            print(f"  {ch['name']}: {ch['before']}  ->  {ch['after']}")
        if len(stats["changes"]) > 25:
            print(f"  ...and {len(stats['changes']) - 25} more.")

    if stats["failures"]:
        print("\nFailures:")
        for f in stats["failures"][:10]:
            print(f"  {f.get('name') or f.get('linkedin_url')}: {f.get('error')}")
        if len(stats["failures"]) > 10:
            print(f"  ...and {len(stats['failures']) - 10} more.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.limit <= 0:
        print("ERROR: --limit must be a positive integer.", file=sys.stderr)
        return 2

    client = get_supabase_client()
    if client is None:
        print(
            "ERROR: Could not build Supabase client. Set supabase_url + "
            "supabase_key in config.json or SUPABASE_URL/SUPABASE_KEY env vars.",
            file=sys.stderr,
        )
        return 2

    if args.verbose:
        print(f"Fetching candidates older than {args.max_age_days} days "
              f"with current_employers <= {args.min_current_employers}...")

    candidates = fetch_sparse_candidates(
        client,
        max_age_days=args.max_age_days,
        limit=args.limit,
        min_current_employers=args.min_current_employers,
    )

    if not args.execute:
        _print_dry_run(candidates, args)
        return 0

    if not candidates:
        print("No candidates matched the filters; nothing to do.")
        return 0

    # Confirm key is available BEFORE we tell the user we're going to spend
    # credits. We do this lazily so the dry-run path doesn't need a key.
    try:
        api_key = _load_crustdata_api_key()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(
        f"[EXECUTE] Re-enriching {len(candidates)} profile(s) — "
        f"~{len(candidates) * CREDITS_PER_ENRICH_PROFILE} credits."
    )

    stats = process_candidates(
        client,
        candidates,
        api_key=api_key,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    _print_summary(stats)
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
