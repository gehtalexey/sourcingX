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
  3) On --execute, bulk-pulls candidates from Crustdata via the people_search_db
     endpoint (3 credits per 100 results — ~1/80th the cost of per-profile
     enrich), writes the new raw_data + indexed columns back via
     ``db.save_enriched_profile``.

Search vs enrich
----------------

Earlier versions of this script called the per-profile enrich endpoint, which
costs 3 credits PER PROFILE. The search endpoint returns the same profile
shape (``current_employers``, ``past_employers``, ``skills``, ``all_employers``,
``all_titles``, ``all_schools``, etc.) at 3 credits per 100 results — i.e. a
single batched call refreshes up to 100 profiles for the same 3 credits one
enrich call would burn. The cost ratio is ~80x in our favour.

The one trade-off: search returns NO personal email. That's fine — emails are
populated by a separate flow (CSV import or `people_enrich`), and the daily
refresh is about keeping current employer/title fresh, not about email. The
write path therefore PRESERVES any existing email already in the row.

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


# Make repo root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crustdata_search import bulk_search_by_flagship_urls  # noqa: E402
from db import get_supabase_client, save_enriched_profile  # noqa: E402
from normalizers import normalize_linkedin_url, pick_current_employer  # noqa: E402


# Crustdata pricing for people_search_db: 3 credits per 100 results, billed
# per request (not per result returned). One batch of <=100 URLs costs 3
# credits regardless of how many of them resolve.
CREDITS_PER_SEARCH_BATCH = 3
SEARCH_RESULTS_PER_BATCH = 100

# Default search batch size — keep at 100 so each batch maps to exactly one
# 3-credit billing unit and response payloads stay manageable.
DEFAULT_BATCH_SIZE = 100


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
# COST ESTIMATION
# =============================================================================


def _estimate_credits(num_candidates: int, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """Estimate credits for a bulk search refresh.

    Crustdata bills 3 credits per request, regardless of how many of the
    requested URLs resolved. We chunk into ``batch_size``-sized requests,
    so the estimate is ``ceil(N / batch_size) * 3``.
    """
    if num_candidates <= 0:
        return 0
    if batch_size <= 0:
        batch_size = SEARCH_RESULTS_PER_BATCH
    return math.ceil(num_candidates / batch_size) * CREDITS_PER_SEARCH_BATCH


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
    """Re-enrich candidates via bulk search and write the new data back.

    Flow (see module docstring for the cost rationale):

      1. Collect candidate URLs.
      2. Call ``bulk_search_by_flagship_urls`` — one search request per batch
         of up to ``batch_size`` URLs (default 100). Each request costs
         exactly 3 credits.
      3. For each candidate URL:
         - If absent from the search response → record
           ``Failed: not found in Crustdata``.
         - Else → upsert via ``save_enriched_profile`` and compare
           ``current_company`` to the stored value to classify as Updated
           or Unchanged.

    The write path PRESERVES any existing ``email`` on the row. We never
    overwrite the email column with "" or null — the search payload has no
    email, and ``save_enriched_profile`` only writes columns we pass.

    Returns a stats dict:
      {
        'processed': int,         # candidates we attempted
        'updated': int,           # rows where current_company actually changed
        'unchanged': int,         # re-enriched but current_company same as before
        'failed': int,            # API or DB errors (or "not found")
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

    # Build the URL list once. Use the stored DB URL as the key — that's the
    # public ``flagship_profile_url`` form Crustdata will echo back.
    urls = [c["linkedin_url"] for c in candidates if c.get("linkedin_url")]
    if not urls:
        return stats

    # Chunk into batches so we can show per-batch progress in verbose mode and
    # bound any single failure to one billing unit.
    batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
    total_batches = len(batches)

    # Merge all batch results into one map keyed by flagship URL.
    search_results: Dict[str, Dict[str, Any]] = {}
    # URLs whose batch raised an exception — we record them as failed up front
    # and skip them in the per-candidate resolution loop so we don't double-count.
    batch_failed_urls: Dict[str, str] = {}
    candidates_by_url = {c["linkedin_url"]: c for c in candidates if c.get("linkedin_url")}

    for batch_idx, batch_urls in enumerate(batches, start=1):
        if verbose:
            print(f"[batch {batch_idx}/{total_batches}] searching {len(batch_urls)} profile(s)...")

        try:
            batch_results = bulk_search_by_flagship_urls(
                batch_urls,
                api_key=api_key,
                batch_size=len(batch_urls),
            )
        except Exception as e:
            # Whole batch failed — mark every URL as failed in a dedicated
            # set so the resolution loop skips them (avoids double-counting).
            err_msg = f"crustdata search failed: {str(e)[:200]}"
            for url in batch_urls:
                batch_failed_urls[url] = err_msg
                if verbose:
                    cand = candidates_by_url.get(url, {})
                    print(f"  FAIL {cand.get('name') or url}: search {e}")
            # Be polite between batches; the rate limiter already throttles.
            if batch_idx < total_batches:
                time.sleep(0.5)
            continue

        # Also key by normalized form so URLs that came back in a slightly
        # different shape still match. We trust Crustdata's
        # `flagship_profile_url` to be the canonical public URL.
        for key, profile in batch_results.items():
            search_results[key] = profile
            norm = normalize_linkedin_url(key)
            if norm and norm not in search_results:
                search_results[norm] = profile

        if batch_idx < total_batches:
            time.sleep(0.5)

    # Now resolve each candidate against the merged map.
    for c in candidates:
        stats["processed"] += 1
        url = c.get("linkedin_url")
        name = c.get("name") or "(no name)"

        if not url:
            stats["failed"] += 1
            stats["failures"].append({
                "linkedin_url": url,
                "name": name,
                "error": "candidate has no linkedin_url",
            })
            continue

        # If this URL's whole batch errored out, count it once here.
        if url in batch_failed_urls:
            stats["failed"] += 1
            stats["failures"].append({
                "linkedin_url": url,
                "name": name,
                "error": batch_failed_urls[url],
            })
            continue

        matched = search_results.get(url)
        if matched is None:
            norm = normalize_linkedin_url(url)
            if norm:
                matched = search_results.get(norm)
        if matched is None:
            stats["failed"] += 1
            stats["failures"].append({
                "linkedin_url": url,
                "name": name,
                "error": "not found in Crustdata",
            })
            if verbose:
                print(f"  FAIL {name}: not found in Crustdata")
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
            # NOTE: save_enriched_profile does NOT touch the email column
            # when called without an email argument — see _prepare_profile_row
            # in db.py. That's the contract that lets the daily refresh keep
            # current_company/title fresh without nuking emails the email-flow
            # populated.
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
        help=f"Crustdata search batch size (default: {DEFAULT_BATCH_SIZE}). "
             "Each batch costs 3 credits regardless of size.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-batch progress during --execute.",
    )
    return parser


def _print_dry_run(candidates: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    print(f"\n[DRY-RUN] Candidates matching filters:")
    print(f"  --max-age-days        = {args.max_age_days}")
    print(f"  --min-current-employers = {args.min_current_employers}")
    print(f"  --limit               = {args.limit}")
    print(f"  --batch-size          = {args.batch_size}")
    print(f"  matched               = {len(candidates)} profile(s)")

    estimated_credits = _estimate_credits(len(candidates), batch_size=args.batch_size)
    num_batches = math.ceil(len(candidates) / args.batch_size) if candidates else 0
    print(
        f"\nEstimated credit cost: {estimated_credits} credits "
        f"(ceil({len(candidates)} / {args.batch_size}) = {num_batches} search "
        f"request(s) x {CREDITS_PER_SEARCH_BATCH} credits/request — "
        f"people_search_db pricing is 3 credits per 100 results, not 3 "
        f"credits per profile)."
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

    estimated = _estimate_credits(len(candidates), batch_size=args.batch_size)
    print(
        f"[EXECUTE] Re-enriching {len(candidates)} profile(s) via bulk search "
        f"— ~{estimated} credits."
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
