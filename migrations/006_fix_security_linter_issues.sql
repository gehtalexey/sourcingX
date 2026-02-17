-- Migration: Fix Supabase Security Linter Issues
-- Run this in Supabase SQL Editor

-- ============================================
-- 1. FIX SECURITY DEFINER VIEWS
-- Recreate views with SECURITY INVOKER (default)
-- ============================================

DROP VIEW IF EXISTS profiles_needing_screening;
DROP VIEW IF EXISTS pipeline_stats;

-- Recreate views (SECURITY INVOKER is the default, but explicit for clarity)
CREATE VIEW profiles_needing_screening
WITH (security_invoker = true) AS
SELECT *
FROM profiles
WHERE status = 'enriched'
  AND screening_score IS NULL;

CREATE VIEW pipeline_stats
WITH (security_invoker = true) AS
SELECT
  COUNT(*) FILTER (WHERE status = 'enriched' AND screening_score IS NULL) AS pending_screening,
  COUNT(*) FILTER (WHERE status = 'screened') AS screened,
  COUNT(*) FILTER (WHERE status = 'contacted') AS contacted,
  COUNT(*) FILTER (WHERE screening_fit_level = 'Strong Fit') AS strong_fit,
  COUNT(*) FILTER (WHERE screening_fit_level = 'Good Fit') AS good_fit,
  COUNT(*) FILTER (WHERE screening_fit_level = 'Partial Fit') AS partial_fit,
  COUNT(*) FILTER (WHERE screening_fit_level = 'Not a Fit') AS not_a_fit,
  COUNT(*) AS total
FROM profiles;

-- ============================================
-- 2. FIX FUNCTION SEARCH PATH
-- Set immutable search_path to prevent injection
-- ============================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = ''
AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;

-- ============================================
-- 3. ENABLE RLS ON ALL TABLES
-- ============================================

ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_usage_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE search_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;

-- ============================================
-- 4. CREATE RLS POLICIES
-- Adjust these based on your auth requirements
-- ============================================

-- PROFILES: Allow authenticated users full access
-- (If this is a single-user/team app, this is appropriate)
CREATE POLICY "Allow authenticated access to profiles" ON profiles
    FOR ALL TO authenticated
    USING (true)
    WITH CHECK (true);

-- API_USAGE_LOGS: Allow authenticated users to view and insert logs
CREATE POLICY "Allow authenticated access to api_usage_logs" ON api_usage_logs
    FOR ALL TO authenticated
    USING (true)
    WITH CHECK (true);

-- SETTINGS: Allow authenticated users full access
CREATE POLICY "Allow authenticated access to settings" ON settings
    FOR ALL TO authenticated
    USING (true)
    WITH CHECK (true);

-- SEARCH_HISTORY: Allow authenticated users full access
CREATE POLICY "Allow authenticated access to search_history" ON search_history
    FOR ALL TO authenticated
    USING (true)
    WITH CHECK (true);

-- SESSIONS: Allow authenticated users full access
CREATE POLICY "Allow authenticated access to sessions" ON sessions
    FOR ALL TO authenticated
    USING (true)
    WITH CHECK (true);

-- ============================================
-- 5. FIX SCREENING_PROMPTS POLICY (optional)
-- Replace overly permissive policy with role-specific one
-- ============================================

-- Drop the overly permissive policy
DROP POLICY IF EXISTS "Allow all" ON screening_prompts;
DROP POLICY IF EXISTS "Allow all for authenticated" ON screening_prompts;

-- Create a more specific policy
CREATE POLICY "Allow authenticated access to screening_prompts" ON screening_prompts
    FOR ALL TO authenticated
    USING (true)
    WITH CHECK (true);

