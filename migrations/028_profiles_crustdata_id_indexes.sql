-- Migration 028: lookup indexes on public.profiles by Crustdata id.
--
-- STATUS: APPLIED 2026-11-03 by hand (built via a one-off pg_cron job because
-- the MCP session cuts statements at 2 min; each built in ~3 min,
-- CONCURRENTLY, no writes blocked). Must run outside a transaction
-- (CREATE INDEX CONCURRENTLY cannot run inside one).
--
-- WHY
-- Migrations 027 and 029 (sourcingx_upsert_profiles dedupe) look up an
-- existing row by Crustdata id on every save. Without an index that lookup
-- scans the whole profiles table. 027's lookup matches idx_profiles_crustdata_id;
-- 029's lookup repeats the expression and predicate of
-- idx_profiles_crustdata_id_any EXACTLY, so the planner uses it. This
-- migration must be applied BEFORE the 027/029 functions are relied on.
--
-- The Crustdata id lives in one of three places in raw_data:
--   raw_data.person_id, raw_data.crustdata_person_id, or (new-API profiles)
--   raw_data._new_api.crustdata_person_id.
--
-- idx_profiles_crustdata_id (two keys) is superseded by
-- idx_profiles_crustdata_id_any (three keys). It can be dropped later, but
-- only with Alexey's OK.
--
-- Same content as agent-kalamata migrations/059_profiles_crustdata_id_indexes.sql
-- (shared database; both projects rely on the same two indexes).

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_profiles_crustdata_id ON public.profiles ((coalesce(nullif(raw_data->>'person_id',''), nullif(raw_data->>'crustdata_person_id',''))))
  WHERE coalesce(nullif(raw_data->>'person_id',''), nullif(raw_data->>'crustdata_person_id','')) IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_profiles_crustdata_id_any ON public.profiles ((coalesce(nullif(raw_data->>'person_id',''), nullif(raw_data->>'crustdata_person_id',''), nullif(raw_data->'_new_api'->>'crustdata_person_id',''))))
  WHERE coalesce(nullif(raw_data->>'person_id',''), nullif(raw_data->>'crustdata_person_id',''), nullif(raw_data->'_new_api'->>'crustdata_person_id','')) IS NOT NULL;
