"""
Backfill profile embeddings.

Reads enriched profiles from Supabase that don't yet have an ``embedding``
(or whose source text has changed), builds the embedding input text, calls
OpenAI's embeddings API in batches, and writes the resulting vectors back.

Resumable
---------
Re-running the script picks up where it left off because:
1. The partial index ``idx_profiles_embedding_missing`` (migration 019)
   makes "next rows missing an embedding" an O(missing) lookup.
2. Each row is written back with ``embedded_at``, ``embedding_model``,
   and ``embedding_input_hash`` so the next run skips rows whose text
   hasn't changed.

Usage
-----
    python backfill_embeddings.py                       # all missing rows
    python backfill_embeddings.py --limit 500           # smoke test
    python backfill_embeddings.py --dry-run             # cost preview only
    python backfill_embeddings.py --re-embed-changed    # also re-embed rows
                                                        # whose input text changed

Cost preview
------------
``--dry-run`` reads N profiles, builds their embedding text, and prints
the estimated token count + USD cost without calling OpenAI. Use this
before the first full run.

This script is read-only against the existing data — it only writes the
new ``embedding`` / ``embedded_at`` / ``embedding_model`` /
``embedding_input_hash`` columns introduced by migration 019.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

from openai import OpenAI

# Local imports
from db import get_supabase_client, SupabaseClient
from embeddings import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    build_embedding_text,
    compute_input_hash,
    embed_texts,
    estimate_embedding_cost,
    estimate_token_count,
)


# Columns we fetch from Supabase. Includes raw_data so the embedding text
# can pull headline/summary/education detail. raw_data is heavy (~20-50KB)
# but we only need it during embedding, so we drop it before write-back.
FETCH_COLUMNS = (
    "linkedin_url,name,current_title,current_company,location,"
    "all_titles,all_employers,all_schools,skills,raw_data,"
    "embedding_input_hash"
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_openai_key() -> str:
    """Load OpenAI key from config.json (same source as the rest of the app)."""
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                key = config.get("openai_api_key", "").strip()
                if key:
                    return key
        except Exception as exc:  # pragma: no cover
            print(f"Warning: could not read config.json: {exc}", file=sys.stderr)
    import os
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            "OpenAI API key not found. Set 'openai_api_key' in config.json "
            "or export OPENAI_API_KEY."
        )
    return key


# ---------------------------------------------------------------------------
# Profile fetching
# ---------------------------------------------------------------------------
def fetch_missing_batch(
    client: SupabaseClient,
    batch_size: int,
    re_embed_changed: bool,
) -> list[dict]:
    """Fetch the next batch of profiles needing embeddings.

    Always includes rows with ``embedding IS NULL``. When
    ``re_embed_changed`` is True, callers handle drift via the hash check
    after building the text — this function still selects only NULL rows
    (drift is rare and that path is opt-in below).
    """
    # PostgREST: `is.null` matches NULL. We restrict to enriched rows so we
    # don't waste tokens embedding rows that never received Crustdata data.
    filters = {
        "embedding": "is.null",
        "enrichment_status": "eq.enriched",
        "raw_data": "not.is.null",
    }
    return client.select(
        table="profiles",
        columns=FETCH_COLUMNS,
        filters=filters,
        limit=batch_size,
        order_by="enriched_at.desc",
    )


def fetch_all_for_drift_check(
    client: SupabaseClient,
    batch_size: int,
    cursor: str | None,
) -> list[dict]:
    """Page through *all* enriched rows for the drift-detection mode.

    Used only when ``--re-embed-changed`` is set. Returns one keyset page;
    the caller advances the cursor by the last row's ``enriched_at``.
    """
    filters = {
        "enrichment_status": "eq.enriched",
        "raw_data": "not.is.null",
    }
    if cursor:
        filters["enriched_at"] = f"lt.{cursor}"
    return client.select(
        table="profiles",
        columns=FETCH_COLUMNS + ",embedded_at",
        filters=filters,
        limit=batch_size,
        order_by="enriched_at.desc",
    )


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------
def write_embeddings(
    client: SupabaseClient,
    rows: list[dict],
    vectors: list[list[float]],
    hashes: list[str],
    model: str,
) -> int:
    """Upsert embedding columns for each row. Returns count written."""
    now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    payload = []
    for row, vector, h in zip(rows, vectors, hashes):
        payload.append({
            "linkedin_url": row["linkedin_url"],
            "embedding": vector,
            "embedded_at": now,
            "embedding_model": model,
            "embedding_input_hash": h,
        })
    if not payload:
        return 0
    client.upsert_batch("profiles", payload, on_conflict="linkedin_url")
    return len(payload)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_backfill(
    limit: int | None,
    dry_run: bool,
    re_embed_changed: bool,
    page_size: int,
) -> None:
    client = get_supabase_client()
    if client is None:
        raise SystemExit(
            "Supabase client unavailable. Check 'supabase_url' / 'supabase_key' "
            "in config.json."
        )

    openai_client = None if dry_run else OpenAI(api_key=load_openai_key())

    total_processed = 0
    total_written = 0
    total_tokens_est = 0
    cursor: str | None = None
    start = time.time()

    while True:
        if limit is not None and total_processed >= limit:
            break

        page_target = page_size
        if limit is not None:
            page_target = min(page_size, limit - total_processed)

        if re_embed_changed:
            batch = fetch_all_for_drift_check(client, page_target, cursor)
        else:
            batch = fetch_missing_batch(client, page_target, re_embed_changed=False)

        if not batch:
            break

        # Build embedding texts + filter out empty ones / unchanged ones.
        rows_to_embed: list[dict] = []
        texts: list[str] = []
        hashes: list[str] = []

        for row in batch:
            text = build_embedding_text(row)
            if not text:
                continue
            h = compute_input_hash(text)

            if re_embed_changed:
                # Skip rows whose text matches what we already embedded.
                existing_hash = row.get("embedding_input_hash")
                if row.get("embedded_at") and existing_hash == h:
                    continue

            rows_to_embed.append(row)
            texts.append(text)
            hashes.append(h)
            total_tokens_est += estimate_token_count(text)

        if not rows_to_embed:
            # Advance the cursor for the drift path even when nothing matched.
            if re_embed_changed and batch:
                cursor = batch[-1].get("enriched_at") or cursor
                total_processed += len(batch)
                continue
            break

        if dry_run:
            total_processed += len(batch)
            print(
                f"[dry-run] would embed {len(rows_to_embed)} / {len(batch)} rows "
                f"(running total: {total_processed} scanned, "
                f"~{total_tokens_est:,} tokens, "
                f"~${estimate_embedding_cost(total_tokens_est):.4f})"
            )
            if re_embed_changed and batch:
                cursor = batch[-1].get("enriched_at") or cursor
            continue

        # Embed in OpenAI-sized chunks (96) to keep payloads small.
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            chunk = texts[i : i + EMBEDDING_BATCH_SIZE]
            try:
                chunk_vectors = embed_texts(openai_client, chunk, model=EMBEDDING_MODEL)
            except Exception as exc:
                print(
                    f"OpenAI call failed for chunk starting at {i}: {exc}. "
                    f"Sleeping 5s and retrying once.",
                    file=sys.stderr,
                )
                time.sleep(5)
                chunk_vectors = embed_texts(openai_client, chunk, model=EMBEDDING_MODEL)
            vectors.extend(chunk_vectors)

        written = write_embeddings(
            client, rows_to_embed, vectors, hashes, EMBEDDING_MODEL
        )
        total_written += written
        total_processed += len(batch)

        if re_embed_changed and batch:
            cursor = batch[-1].get("enriched_at") or cursor

        elapsed = time.time() - start
        rate = total_written / elapsed if elapsed > 0 else 0
        print(
            f"Embedded {written} (cumulative {total_written}); "
            f"scanned {total_processed}; "
            f"{rate:.1f} rows/sec; "
            f"est cost so far ${estimate_embedding_cost(total_tokens_est):.4f}"
        )

    elapsed = time.time() - start
    print()
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"Scanned:     {total_processed} profiles")
    print(f"Embedded:    {total_written} profiles")
    print(f"Tokens (est): {total_tokens_est:,}")
    print(f"Cost (est):   ${estimate_embedding_cost(total_tokens_est):.4f}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of profiles to scan (default: all).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=500,
        help="How many rows to fetch per Supabase page (default: 500).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate token count and cost without calling OpenAI.",
    )
    parser.add_argument(
        "--re-embed-changed",
        action="store_true",
        help="Page through all enriched rows and re-embed any whose input text changed.",
    )
    args = parser.parse_args()

    run_backfill(
        limit=args.limit,
        dry_run=args.dry_run,
        re_embed_changed=args.re_embed_changed,
        page_size=args.page_size,
    )


if __name__ == "__main__":
    main()
