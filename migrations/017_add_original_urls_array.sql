-- Migration: Add original_urls array for multi-source URL tracking
-- This allows storing multiple input URLs when the same profile is uploaded
-- from different sources (PhantomBuster, Jam CSV, GEM, etc.)

-- 1. Add array column (keeps original_url for backward compatibility)
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS original_urls TEXT[] DEFAULT '{}';

-- 2. Populate from existing original_url (only if not already populated)
UPDATE profiles
SET original_urls = ARRAY[original_url]
WHERE original_url IS NOT NULL
  AND (original_urls IS NULL OR original_urls = '{}');

-- 3. Create GIN index for fast ANY() lookups
CREATE INDEX IF NOT EXISTS idx_profiles_original_urls
ON profiles USING GIN (original_urls);

-- 4. Add comment for documentation
COMMENT ON COLUMN profiles.original_urls IS
'Array of all input URLs from different sources (PhantomBuster, Jam, CSV, etc.). Used for deduplication across sources.';

COMMENT ON COLUMN profiles.original_url IS
'Most recent input URL (kept for backward compatibility). Use original_urls for multi-source matching.';
