-- Migration: Add education text column for display
-- The education_list JSONB exists but we need a simple text column for display

ALTER TABLE profiles ADD COLUMN IF NOT EXISTS education TEXT;

-- Backfill from education_list if it exists
UPDATE profiles
SET education = education_list->0->>'school_name'
WHERE education IS NULL
  AND education_list IS NOT NULL
  AND jsonb_array_length(education_list) > 0;
