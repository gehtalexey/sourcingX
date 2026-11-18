-- Migration 031 (manual): lookup index on public.supanova_person_ids(person_id).
--
-- STATUS: NOT YET APPLIED. Written 2026-11-18. Apply by hand, BEFORE
-- migrations/031_sourcingx_upsert_profiles_reads_supanova_person_ids.sql.
--
-- Lives in migrations/manual/ because CREATE INDEX CONCURRENTLY cannot run
-- inside a transaction, so the automatic runner (db_migrations.py) must skip it.
--
-- WHY
-- public.supanova_person_ids is Supanova's table (linkedin_url -> Crustdata
-- person_id, recovered free on 2026-11-17). Its only index today is the unique
-- key on linkedin_url. Migration 031 makes sourcingx_upsert_profiles look a
-- person up by person_id on every save, so it needs this index.
-- The table has no migration file in the Supanova repo (it was created by a
-- one-off job), so the index lives here, with SourcingX's save-check change.
-- Measured live 2026-11-18: 40,993 rows, so the build takes seconds.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_supanova_person_ids_person_id
  ON public.supanova_person_ids (person_id);
