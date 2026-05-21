-- Migration 021: Fix index bloat and RLS performance issues
-- Context: bulk backfill exhausted disk I/O and RAM due to duplicate/unused indexes
-- and per-row auth re-evaluation in RLS policies.
--
-- Changes:
--   1. Drop duplicate indexes (double write cost on every upsert)
--   2. Drop unused indexes on profiles (never queried, waste RAM + I/O)
--   3. Fix RLS policies: wrap auth.uid() in (SELECT ...) so it evaluates once per
--      query instead of once per row
--   4. Grant EXECUTE on is_workspace_member to authenticator (stops repeated
--      "permission denied" errors that fire every ~60 seconds from PostgREST)
--
-- Fresh-DB safety: index drops use IF EXISTS throughout. RLS and GRANT changes
-- are wrapped in DO $$ blocks with existence checks because top_companies,
-- similar_search_state, company_similarity_reviews, and is_workspace_member
-- are not yet represented in the checked-in migration chain — the blocks are
-- no-ops on a fresh DB and apply correctly on production.

-- ============================================================
-- 1. DROP DUPLICATE INDEXES
-- ============================================================

-- profiles: idx_profiles_search_text is an out-of-band production duplicate of
-- idx_profiles_search (created in migration 009). Drop the untracked one.
DROP INDEX IF EXISTS public.idx_profiles_search_text;

-- pipeline_candidates: two duplicate pairs
DROP INDEX IF EXISTS public.idx_pc_linkedin;
DROP INDEX IF EXISTS public.idx_pc_position;

-- pipeline_runs: idx_pr_position duplicates idx_pipeline_runs_position_id
DROP INDEX IF EXISTS public.idx_pr_position;

-- screening_results: idx_screening_results_linkedin duplicates idx_screening_results_linkedin_url
DROP INDEX IF EXISTS public.idx_screening_results_linkedin;

-- ============================================================
-- 2. DROP UNUSED INDEXES ON profiles
-- ============================================================

DROP INDEX IF EXISTS public.idx_profiles_all_employers;
DROP INDEX IF EXISTS public.idx_profiles_all_titles;
DROP INDEX IF EXISTS public.idx_profiles_all_schools;
DROP INDEX IF EXISTS public.idx_profiles_skills;

-- ============================================================
-- 3. FIX RLS POLICIES: (SELECT auth.uid()) instead of auth.uid()
--    Existence-guarded because these tables are not in the migration chain.
-- ============================================================

DO $$ BEGIN

  -- top_companies: SELECT policy
  IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = 'top_companies') THEN
    DROP POLICY IF EXISTS top_companies_read ON public.top_companies;
    CREATE POLICY top_companies_read ON public.top_companies
      FOR SELECT
      USING (
        (workspace_id IS NULL)
        OR (workspace_id IN (
          SELECT workspace_members.workspace_id
          FROM workspace_members
          WHERE workspace_members.user_id = (SELECT auth.uid())
        ))
      );

    -- top_companies: ALL policy (write/admin)
    DROP POLICY IF EXISTS top_companies_write ON public.top_companies;
    CREATE POLICY top_companies_write ON public.top_companies
      FOR ALL
      USING (
        (
          (workspace_id IS NULL)
          AND (EXISTS (
            SELECT 1 FROM user_profiles
            WHERE user_profiles.id = (SELECT auth.uid())
              AND user_profiles.role = 'admin'::text
          ))
        )
        OR (workspace_id IN (
          SELECT workspace_members.workspace_id
          FROM workspace_members
          WHERE workspace_members.user_id = (SELECT auth.uid())
            AND workspace_members.role = 'admin'::text
        ))
      );
  END IF;

  -- similar_search_state
  IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = 'similar_search_state') THEN
    DROP POLICY IF EXISTS similar_search_state_access ON public.similar_search_state;
    CREATE POLICY similar_search_state_access ON public.similar_search_state
      FOR ALL
      USING (
        conversation_id IN (
          SELECT conversations.id
          FROM conversations
          WHERE conversations.workspace_id IN (
            SELECT workspace_members.workspace_id
            FROM workspace_members
            WHERE workspace_members.user_id = (SELECT auth.uid())
          )
        )
      );
  END IF;

  -- company_similarity_reviews
  IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = 'company_similarity_reviews') THEN
    DROP POLICY IF EXISTS company_similarity_reviews_access ON public.company_similarity_reviews;
    CREATE POLICY company_similarity_reviews_access ON public.company_similarity_reviews
      FOR ALL
      USING (
        workspace_id IN (
          SELECT workspace_members.workspace_id
          FROM workspace_members
          WHERE workspace_members.user_id = (SELECT auth.uid())
        )
      );
  END IF;

END $$;

-- ============================================================
-- 4. FIX is_workspace_member PERMISSION ERROR
--    Existence-guarded because the function is not in the migration chain.
-- ============================================================

DO $$ BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_proc p
    JOIN pg_namespace n ON n.oid = p.pronamespace
    WHERE p.proname = 'is_workspace_member' AND n.nspname = 'public'
  ) THEN
    GRANT EXECUTE ON FUNCTION public.is_workspace_member(uuid) TO authenticator;
  END IF;
END $$;
