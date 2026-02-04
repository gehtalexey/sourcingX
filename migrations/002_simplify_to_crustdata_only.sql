-- Migration: Simplify profiles table to store only Crustdata enriched data
-- PhantomBuster data will only be used in UI session state, not stored in DB
--
-- Run this in Supabase SQL Editor
-- BACKUP YOUR DATA FIRST!

-- Step 1: Drop PhantomBuster-specific columns
ALTER TABLE profiles DROP COLUMN IF EXISTS phantombuster_data;
ALTER TABLE profiles DROP COLUMN IF EXISTS source_search_id;

-- Step 2: Update status constraint - remove 'scraped' status
-- First, update any 'scraped' profiles to NULL (they shouldn't be in DB anymore)
DELETE FROM profiles WHERE status = 'scraped' AND enriched_at IS NULL;

-- Update the constraint
ALTER TABLE profiles DROP CONSTRAINT IF EXISTS profiles_status_check;
ALTER TABLE profiles ADD CONSTRAINT profiles_status_check
  CHECK (status IN ('enriched', 'screened', 'contacted', 'archived'));

-- Step 3: Rename crustdata_data to raw_data for clarity
ALTER TABLE profiles RENAME COLUMN crustdata_data TO raw_data;

-- Step 4: Add new columns for full Crustdata arrays (for advanced querying)
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS positions JSONB;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS education_list JSONB;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS skills_list JSONB;

-- Step 5: Add columns that Crustdata provides but we weren't storing
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS profile_picture_url TEXT;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS connections_count INTEGER;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS followers_count INTEGER;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS current_years_at_company NUMERIC;

-- Step 6: Update existing records - extract arrays from raw_data
UPDATE profiles
SET
  positions = raw_data->'positions',
  education_list = raw_data->'education',
  skills_list = raw_data->'skills'
WHERE raw_data IS NOT NULL AND positions IS NULL;

-- Step 7: Create index on positions for company/title searches
CREATE INDEX IF NOT EXISTS idx_profiles_positions ON profiles USING GIN (positions);
CREATE INDEX IF NOT EXISTS idx_profiles_skills ON profiles USING GIN (skills_list);

-- Step 8: Update the view for profiles needing screening
DROP VIEW IF EXISTS profiles_needing_screening;
CREATE VIEW profiles_needing_screening AS
SELECT *
FROM profiles
WHERE status = 'enriched'
  AND screening_score IS NULL;

-- Step 9: Drop the old view that referenced 'scraped' status
DROP VIEW IF EXISTS profiles_needing_enrichment;

-- Step 10: Update pipeline stats view
DROP VIEW IF EXISTS pipeline_stats;
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

-- Done! New schema:
-- profiles (
--   linkedin_url TEXT PRIMARY KEY,
--   first_name, last_name, headline, location, summary TEXT,
--   current_title, current_company TEXT,
--   current_years_in_role, current_years_at_company NUMERIC,
--   positions JSONB,        -- Full array from Crustdata
--   education_list JSONB,   -- Full array from Crustdata
--   skills_list JSONB,      -- Full array from Crustdata
--   skills TEXT,            -- Comma-separated for display
--   education TEXT,         -- Most recent school for display
--   connections_count, followers_count INTEGER,
--   profile_picture_url TEXT,
--   raw_data JSONB,         -- Full Crustdata response
--   screening_score, screening_fit_level, screening_summary, screening_reasoning,
--   email, email_source,
--   status TEXT CHECK (status IN ('enriched', 'screened', 'contacted', 'archived')),
--   enriched_at, screened_at, contacted_at, created_at, updated_at TIMESTAMPTZ
-- )
