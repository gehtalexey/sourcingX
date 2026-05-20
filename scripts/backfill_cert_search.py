"""
Backfill search_text for profiles that have certifications in raw_data.

Migration 019 added certifications to the search_text trigger, but existing
profiles were not automatically updated (bulk backfill timed out via MCP).
This script fetches those profiles and re-saves them, which fires the trigger.

No Crustdata credits are spent — we re-save what's already in the DB.

Selection strategy:
    We page through `profiles` by primary key `id` (keyset pagination) and
    project only the certifications JSON path in `select=`. Filtering for
    non-empty certifications is done client-side. This avoids any JSON
    expression in the WHERE clause, which on the live table caused
    Supabase statement timeouts (PostgREST 500 / code 57014) when an
    earlier version of this script paged by `raw_data->>certifications`.

Usage:
    # Dry-run (default) — shows count only:
    python scripts/backfill_cert_search.py

    # Actually run it:
    python scripts/backfill_cert_search.py --execute

    # Limit batch size (default 500):
    python scripts/backfill_cert_search.py --execute --batch-size 200
"""

import argparse
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import SupabaseClient


# How many rows to scan per keyset page when looking for non-empty
# certifications. The projection is tiny (just the cert path), so 1000
# is comfortably under any timeout.
SCAN_PAGE_SIZE = 1000

# How many linkedin_urls to inline in a PostgREST `?linkedin_url=in.(...)`
# GET request when fetching full rows. URLs average ~50 chars, and
# PostgREST/Supabase rejects requests once the URL query exceeds its
# limit (empirically 500 URLs → 400 Bad Request, 450 still works).
# 200 keeps us safely under that ceiling regardless of the user's
# --batch-size for upserts.
URL_FETCH_CHUNK_SIZE = 200


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    with open(config_path) as f:
        return json.load(f)


def find_profiles_with_certs(client: SupabaseClient) -> list:
    """Scan profiles by id (keyset pagination) and return linkedin_urls
    of those with a non-empty certifications array.

    We never put a JSON expression in the WHERE clause — only in the
    projection — so the query stays on the `id` primary key index and
    can't time out scanning JSON for filterable rows.
    """
    url = f"{client.url}/rest/v1/profiles"
    last_id = None
    found = []
    scanned = 0

    while True:
        params = {
            'select': 'id,linkedin_url,certifications:raw_data->certifications',
            'order': 'id.asc',
            'limit': SCAN_PAGE_SIZE,
        }
        if last_id is not None:
            params['id'] = f'gt.{last_id}'

        resp = requests.get(url, headers=client.headers, params=params, timeout=60)
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break

        scanned += len(page)
        for row in page:
            certs = row.get('certifications')
            if isinstance(certs, list) and len(certs) > 0:
                found.append(row['linkedin_url'])

        last_id = page[-1]['id']
        print(f"  scanned {scanned} rows, matches so far: {len(found)}")

        if len(page) < SCAN_PAGE_SIZE:
            break

    return found


def backfill(execute: bool, batch_size: int):
    try:
        config = load_config()
        supabase_url = config.get('supabase_url', os.environ.get('SUPABASE_URL', ''))
        supabase_key = config.get('supabase_key', os.environ.get('SUPABASE_KEY', ''))
    except FileNotFoundError:
        supabase_url = os.environ.get('SUPABASE_URL', '')
        supabase_key = os.environ.get('SUPABASE_KEY', '')

    if not supabase_url or not supabase_key:
        print("ERROR: Missing SUPABASE_URL / SUPABASE_KEY")
        sys.exit(1)

    client = SupabaseClient(supabase_url, supabase_key)
    profiles_url = f"{client.url}/rest/v1/profiles"

    print("Scanning profiles for non-empty certifications (keyset paging by id)...")
    all_urls = find_profiles_with_certs(client)
    print(f"\nFound {len(all_urls)} profiles with non-empty certifications.")

    if not execute:
        print("DRY-RUN — pass --execute to actually update.")
        return

    # Re-save each profile by doing a no-op upsert that fires the trigger.
    # We fetch the full row and upsert it back — no Crustdata credits, no data change.
    updated = 0
    errors = 0
    total = len(all_urls)

    for i in range(0, total, batch_size):
        chunk = all_urls[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}: rows {i+1}–{min(i+batch_size, total)} of {total}...")

        # Fetch full rows. The IN-list goes through the GET query string,
        # which PostgREST caps in length — so we sub-chunk the URLs even
        # when the upsert batch is larger. The upsert below sends rows in
        # the POST body, where URL length doesn't apply.
        rows = []
        for j in range(0, len(chunk), URL_FETCH_CHUNK_SIZE):
            sub = chunk[j:j + URL_FETCH_CHUNK_SIZE]
            rows_resp = requests.get(
                profiles_url,
                headers=client.headers,
                params={
                    'select': '*',
                    'linkedin_url': f"in.({','.join(sub)})",
                    'limit': len(sub),
                },
                timeout=60
            )
            rows_resp.raise_for_status()
            rows.extend(rows_resp.json())

        if not rows:
            continue

        # Upsert back — fires the BEFORE UPDATE trigger which rebuilds search_text
        upsert_resp = requests.post(
            profiles_url,
            headers={**client.headers, 'Prefer': 'resolution=merge-duplicates'},
            params={'on_conflict': 'linkedin_url'},
            json=rows,
            timeout=60
        )
        if upsert_resp.status_code in (200, 201):
            updated += len(rows)
        else:
            print(f"    WARNING: batch failed ({upsert_resp.status_code}): {upsert_resp.text[:200]}")
            errors += len(rows)

        time.sleep(0.5)  # avoid overwhelming the DB

    print(f"\nDone. Updated: {updated}, Errors: {errors}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backfill search_text for profiles with certifications')
    parser.add_argument('--execute', action='store_true', help='Actually run the backfill (default: dry-run)')
    parser.add_argument('--batch-size', type=int, default=500, help='Rows per batch (default: 500)')
    args = parser.parse_args()
    backfill(execute=args.execute, batch_size=args.batch_size)
