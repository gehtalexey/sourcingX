# Supabase Egress Analysis - February 2026

## Issue

Received email from Supabase on Feb 21, 2026:
- Organization "AddedValue" exceeded free tier usage quota
- Need to reduce egress bandwidth below 5.5 GB
- Grace period until March 23, 2026
- After that, Fair Use Policy applies (potential restrictions)

## Root Cause

`SELECT *` queries fetch `raw_data` column (~20-50KB per profile) unnecessarily.

### Top Egress Sources in `db.py`

| Function | Line | Issue | Est. Egress per Call |
|----------|------|-------|---------------------|
| `get_profiles_by_status()` | 406 | Fetches 500 profiles with `raw_data` | 15-25 MB |
| `get_profiles_by_fit_level()` | 411 | Fetches 1000 profiles with `raw_data` | 30-50 MB |
| `get_profile()` | 395 | Fetches single profile with full `raw_data` | 20-50 KB |
| `search_profiles()` | 444 | Fetches up to 100 profiles with `raw_data` | 3-5 MB |

### Usage in `dashboard.py`

```
Line 4646: get_profiles_by_status(db_client, "enriched", limit=500)  ← loads for screening
Line 8150: get_profiles_by_status(db_client, "enriched", limit=500)  ← loads for screening
Line 4396: get_profile(db_client, url)  ← fallback fetch during screening
Line 7604: get_profile(db_client, url)  ← individual profile lookup
```

### Egress Estimates

| Scenario | Egress |
|----------|--------|
| 1 screening session (500 profiles) | ~15 MB |
| 10 screening sessions/week | ~600 MB/month |
| 30 screening sessions/month + browsing | 2-3 GB/month |
| Heavy usage with multiple users | 5+ GB (exceeds limit) |

## Options

### Option 1: Upgrade to Pro ($25/month)

- 250 GB egress (50x more than free tier)
- Daily backups
- 8 GB database storage
- Zero risk, immediate fix

### Option 2: Code Fix (Risky)

Add `include_raw_data=False` parameter to:
- `get_profiles_by_status()`
- `get_profiles_by_fit_level()`
- `get_profile()` → or create `get_profile_metadata()`
- `search_profiles()`

**Risks:**
- Screening needs `raw_data` - changing defaults breaks existing calls
- Would need to audit every call site in dashboard.py
- Silent failures if a call site is missed

**Pattern already exists:**
```python
# get_all_profiles() already has this pattern (db.py:414)
def get_all_profiles(client, limit=10000, include_raw_data=False):
    if include_raw_data:
        return client.select('profiles', '*', limit=limit)
    else:
        columns = 'linkedin_url,original_url,name,location,...'
        return client.select('profiles', columns, limit=limit)
```

## Recommendation

**Upgrade now, optimize later.**

The code fix is the right long-term solution but carries risk. For $25/month, the Pro plan removes urgency and allows careful optimization without deadline pressure.

## Related PR

PR #1 (database-migration) adds migration tracking and transaction support - unrelated to egress issue.

## Resolution (2026-02-24)

### Actions Taken

1. **Upgraded to Pro plan** ($25/month)
   - 250 GB egress (vs 5 GB free tier)
   - Removes immediate urgency

2. **Upgraded instance from Nano to Micro** (no additional cost on Pro)
   - More RAM (prevents memory exhaustion)
   - I/O capacity: 250 → 500 IOps
   - 30 min/day burst compute for peak loads

### Post-Upgrade Monitoring

- Watch IOps usage in Supabase dashboard
- If consistently hitting 500 IOps, consider:
  - Adding pagination to large queries
  - Batching database writes
  - Using server-side filtering (already added in commit `4c8f63a`)

### Future Optimization (Optional)

The `include_raw_data=False` pattern can still be implemented to reduce egress further, but is no longer urgent with the Pro plan's 250 GB limit.
