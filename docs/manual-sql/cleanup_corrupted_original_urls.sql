-- MANUAL ONE-TIME CLEANUP — not an automated migration.
-- Run these steps by hand in Supabase SQL editor when investigating
-- profiles where original_url points at the wrong person.
-- The destructive UPDATE in STEP 3 is intentionally commented out so it
-- only runs after a human has reviewed the preview/count output.

-- STEP 1: PREVIEW - Run this first to see what will be cleaned (READ-ONLY)
-- This shows corrupted profiles WITHOUT making any changes

SELECT
    linkedin_url,
    original_url,
    name,
    REGEXP_REPLACE(LOWER(SPLIT_PART(linkedin_url, '/in/', 2)), '-[a-z0-9]{5,}$', '', 'g') as linkedin_base,
    REGEXP_REPLACE(LOWER(SPLIT_PART(original_url, '/in/', 2)), '-[a-z0-9]{5,}$', '', 'g') as original_base
FROM profiles
WHERE original_url IS NOT NULL
  AND original_url != linkedin_url
  AND NOT (
    REGEXP_REPLACE(LOWER(SPLIT_PART(linkedin_url, '/in/', 2)), '-[a-z0-9]{5,}$', '', 'g')
    =
    REGEXP_REPLACE(LOWER(SPLIT_PART(original_url, '/in/', 2)), '-[a-z0-9]{5,}$', '', 'g')
  )
LIMIT 50;

-- Expected: ~971 rows with mismatched linkedin_base vs original_base
-- Example: linkedin_base='adam-gorkoz', original_base='meirkoen' (WRONG - different people)


-- STEP 2: COUNT - See exactly how many will be affected (READ-ONLY)

SELECT COUNT(*) as corrupted_count
FROM profiles
WHERE original_url IS NOT NULL
  AND original_url != linkedin_url
  AND NOT (
    REGEXP_REPLACE(LOWER(SPLIT_PART(linkedin_url, '/in/', 2)), '-[a-z0-9]{5,}$', '', 'g')
    =
    REGEXP_REPLACE(LOWER(SPLIT_PART(original_url, '/in/', 2)), '-[a-z0-9]{5,}$', '', 'g')
  );

-- Expected: ~971


-- STEP 3: UPDATE - Only run this AFTER verifying Steps 1 and 2 look correct
-- This ONLY sets original_url to NULL - does NOT delete any profiles or other data

/*
UPDATE profiles
SET original_url = NULL
WHERE original_url IS NOT NULL
  AND original_url != linkedin_url
  AND NOT (
    REGEXP_REPLACE(LOWER(SPLIT_PART(linkedin_url, '/in/', 2)), '-[a-z0-9]{5,}$', '', 'g')
    =
    REGEXP_REPLACE(LOWER(SPLIT_PART(original_url, '/in/', 2)), '-[a-z0-9]{5,}$', '', 'g')
  );
*/

-- STEP 4: VERIFY - Run after update to confirm cleanup worked

-- SELECT COUNT(*) as remaining_corrupted FROM profiles WHERE original_url IS NOT NULL AND original_url != linkedin_url;
-- Expected: Should be much lower (only valid cases where original_url differs but same person)
