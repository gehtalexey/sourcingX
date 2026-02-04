-- Migration: Simplify schema to raw_data storage
-- Keep only essential indexed columns, everything else in raw_data JSONB
--
-- Run in Supabase SQL Editor

-- Drop views first
DROP VIEW IF EXISTS profiles_needing_screening;
DROP VIEW IF EXISTS pipeline_stats;

-- Drop columns we no longer need (data is in raw_data)
ALTER TABLE profiles DROP COLUMN IF EXISTS first_name;
ALTER TABLE profiles DROP COLUMN IF EXISTS last_name;
ALTER TABLE profiles DROP COLUMN IF EXISTS headline;
ALTER TABLE profiles DROP COLUMN IF EXISTS location;
ALTER TABLE profiles DROP COLUMN IF EXISTS summary;
ALTER TABLE profiles DROP COLUMN IF EXISTS current_years_in_role;
ALTER TABLE profiles DROP COLUMN IF EXISTS current_years_at_company;
ALTER TABLE profiles DROP COLUMN IF EXISTS positions;
ALTER TABLE profiles DROP COLUMN IF EXISTS education_list;
ALTER TABLE profiles DROP COLUMN IF EXISTS skills_list;
ALTER TABLE profiles DROP COLUMN IF EXISTS skills;
ALTER TABLE profiles DROP COLUMN IF EXISTS education;
ALTER TABLE profiles DROP COLUMN IF EXISTS connections_count;
ALTER TABLE profiles DROP COLUMN IF EXISTS followers_count;
ALTER TABLE profiles DROP COLUMN IF EXISTS profile_picture_url;

-- Keep these columns:
-- linkedin_url (PK)
-- raw_data (JSONB - full Crustdata response)
-- current_title (indexed for filtering)
-- current_company (indexed for filtering)
-- screening_score, screening_fit_level, screening_summary, screening_reasoning
-- email, email_source
-- status
-- created_at, updated_at, enriched_at, screened_at, contacted_at

-- Recreate views
CREATE VIEW profiles_needing_screening AS
SELECT *
FROM profiles
WHERE status = 'enriched'
  AND screening_score IS NULL;

CREATE VIEW pipeline_stats AS
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

-- Create GIN index on raw_data for JSONB queries
CREATE INDEX IF NOT EXISTS idx_profiles_raw_data ON profiles USING GIN (raw_data);
