-- Migration: Fix api_usage_logs RLS to allow anon access
-- The Streamlit app uses anon key, so we need to allow anon role

-- Drop existing policy if exists and recreate with anon access
DROP POLICY IF EXISTS "Allow anon access to api_usage_logs" ON api_usage_logs;

CREATE POLICY "Allow anon access to api_usage_logs" ON api_usage_logs
    FOR ALL TO anon
    USING (true)
    WITH CHECK (true);
