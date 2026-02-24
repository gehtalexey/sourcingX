# Supabase Egress Analysis

**Date:** 2026-02-24
**Status:** Analyzed - Decided to upgrade to Pro plan

## Current Usage (Free Plan)

| Resource | Used | Limit | % Used |
|----------|------|-------|--------|
| Egress | 10.87 GB | 5 GB | 217% |
| Database Size | 127 MB | 500 MB | 25% |

Grace period until: March 23, 2026

## High-Egress Patterns Identified

### Critical (P1)

| Issue | Location | Data per Call | Can Optimize? |
|-------|----------|---------------|---------------|
| Export tab fetches raw_data | `dashboard.py:6566` | 100-250 MB | No - breaks standalone export |
| Pipeline refresh uses SELECT * | `dashboard.py:5116` | ~175 MB | Yes - safe |
| Screening batch DB fetch | `dashboard.py:3228` | ~15 MB per run | No - legitimately needed |

### High (P2)

| Issue | Location | Data per Call | Can Optimize? |
|-------|----------|---------------|---------------|
| Usage logs fetch all columns | `db.py:572, 646` | 100 KB - 1 MB | Yes - safe |
| Search profiles uses SELECT * | `db.py:444-447` | ~3.5 MB per search | Maybe - needs verification |

## Root Causes

1. **`SELECT *` instead of specific columns** - Fetching `raw_data` JSONB (20-50KB per profile) when not needed
2. **Data fetched then discarded** - Pipeline refresh fetches raw_data then calls `p.pop('raw_data', None)`
3. **No server-side filtering** - Client-side filtering after heavy data transfer

## Safe Optimizations (Not Implemented)

These are safe but only save ~30-40% egress:

1. **Pipeline refresh (dashboard.py:5116)** - Specify column list instead of `*`
2. **Usage logs (db.py:572, 646)** - Select only aggregation-needed columns

## Risky Optimizations (Not Recommended)

1. **Export tab** - Removing raw_data fetch breaks:
   - Standalone export (no screening first)
   - Export after page refresh (session cache cleared)
   - Export of profiles not in current screening batch

2. **Search profiles** - May break profile detail view

## Decision

**Upgrade to Supabase Pro ($25/mo)** rather than risk breaking working features.

Pro plan provides:
- 250 GB egress (vs 5 GB free)
- 8 GB database (vs 500 MB free)

Database will hit free limit at ~10,000-15,000 profiles (at 20-50KB raw_data per profile).

## Future Considerations

If egress becomes a cost concern on Pro:
- Implement local caching layer for raw_data
- Add compression for raw_data JSONB
- Consider storing raw_data in Supabase Storage instead of database column
