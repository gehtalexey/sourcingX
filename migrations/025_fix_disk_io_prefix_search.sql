-- Migration 025: Fix Disk IO Budget warning — add prefix-search indexes,
-- drop confirmed-dead indexes.
--
-- CONTEXT (2026-07-20):
-- Supabase emailed a "running out of Disk IO Budget" warning for this project.
-- Investigated with pg_stat_statements + pg_stat_user_indexes on the live DB.
--
-- ROOT CAUSE:
-- dashboard.py's dedup pre-check batches up to 100 LinkedIn URL stems and
-- searches with a prefix wildcard pattern:
--     linkedin_url LIKE 'https://www.linkedin.com/in/<stem>-*'
--     original_url LIKE 'https://www.linkedin.com/in/<stem>-*'
-- (see dashboard.py, ~line 594-609). This database's collation is
-- en_US.UTF-8, not C — so a plain btree index (what idx_profiles_linkedin_url
-- and idx_profiles_original_url are today) CANNOT accelerate a prefix LIKE
-- query. Every one of these batched checks falls back to reading nearly the
-- entire profiles table from disk.
--
-- Verified via pg_stat_statements: this single query pattern accounted for
-- ~690,000 of the ~1.19M disk block reads on the profiles table (roughly half
-- of all disk reads happening anywhere in the shared database).
--
-- FIX: add text_pattern_ops indexes, which DO support prefix LIKE regardless
-- of collation. Built CONCURRENTLY (and dropped CONCURRENTLY below) because
-- this table is live and written to continuously by four sibling pipelines —
-- CONCURRENTLY avoids taking a write lock during the build/drop.
--
-- Idempotent: IF NOT EXISTS / IF EXISTS throughout.
--
-- *** DO NOT RUN THIS FILE AS ONE PASTED BLOCK / ONE TRANSACTION. ***
-- CONCURRENTLY cannot run inside a transaction — Postgres rejects it with
-- "CREATE INDEX CONCURRENTLY cannot run inside a transaction block". Any tool
-- that submits this whole file as a single statement or wraps it in
-- BEGIN...COMMIT (a naive migration runner, or pasting the whole file into a
-- SQL editor that auto-wraps) will fail and silently apply NONE of the
-- statements below. Each statement below must be run separately, with
-- autocommit on (Supabase SQL Editor's default "Run" behaviour is fine one
-- statement at a time; so is `psql -f` with default autocommit; this repo's
-- db_migrations.py runner does not yet execute CONCURRENTLY-safe statement-
-- by-statement, so this migration must be applied manually, one statement at
-- a time, until that runner is updated to support it).

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_profiles_linkedin_url_pattern
  ON public.profiles (linkedin_url text_pattern_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_profiles_original_url_pattern
  ON public.profiles (original_url text_pattern_ops);

-- ============================================================
-- DROP CONFIRMED-DEAD INDEXES
-- ============================================================
-- These three were never scanned once since the database was created
-- (pg_stat_database.stats_reset is NULL — the counters cover the DB's full
-- lifetime, ~5.5 months). Checked against actual code: the only queries
-- against name/current_company/location use ILIKE '%term%' (two-sided
-- wildcard, dashboard.py ~line 1355-1357 and the similarity-search RPC in
-- migration 021), which a plain btree index can never accelerate anyway —
-- so these were dead weight from the day they were created, not recently
-- unused.
--
-- Deliberately NOT touched here (investigated and kept):
--   idx_profiles_embedding_ivfflat  — confirmed live via EXPLAIN, backs the
--                                     "Find Similar Profiles" feature.
--   idx_profiles_search             — full-text search RPC is actively
--                                     maintained across many migrations;
--                                     0 scans is unexplained and needs its
--                                     own investigation, not a blind drop.
--   idx_profiles_skills_blob_trgm,
--   idx_profiles_titles_blob_trgm   — built for agent-kalamata's talent-pool
--                                     title/skills pre-filter, which is
--                                     still status "PLAN" as of its planning
--                                     doc (docs/plans/talent-pool-server-side
--                                     -title-filter.md) — schema exists ahead
--                                     of the feature, not abandoned code.
--   idx_profiles_embedding_missing  — small (6MB), backs the embedding
--                                     backfill worker's batch query; not
--                                     worth the risk for the space saved.

DROP INDEX CONCURRENTLY IF EXISTS public.idx_profiles_name;
DROP INDEX CONCURRENTLY IF EXISTS public.idx_profiles_current_company;
DROP INDEX CONCURRENTLY IF EXISTS public.idx_profiles_location;
