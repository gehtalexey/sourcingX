-- ============================================================================
-- Backfill: correct current_company / current_title for multi-current-employer
-- profiles enriched before the pick_current_employer fix.
-- ============================================================================
--
-- Background
-- ----------
-- Before PR #21 (merged 2026-05-11), normalize_crustdata_profile picked
-- current_employers[0] blindly. When Crustdata returned a profile with
-- multiple active employers (advisor + full-time, two parallel jobs,
-- IDF reserve + civilian role), the WRONG entry was often stored in
-- the indexed columns current_company and current_title.
--
-- Diagnostic counts (2026-05-11 against linkedin-enricher / public.profiles):
--   - 29,481 total profiles
--   -    797 have 0 current_employers
--   - 22,401 have 1 current_employer        (cannot be wrong)
--   -  6,283 have >= 2 current_employers    (potentially affected)
--   -  1,900 of those 6,283 have a stored value that DIFFERS from the
--           most-recent entry by start_date — these will be corrected.
--
-- Concrete bad cases that this query fixes:
--   - Hadar Zilbershtein:  "MDI Health"      -> "Forter"
--   - Raga Saleh:          "Unity"           -> "Taboola"
--   - Adi Levy:            "triptip.ai"      -> "Optibus"
--   - Petar Galic:         "NjiaPay"         -> "FLYR Labs"
--   - Abir Gitlin:         "Hebrew University" -> "INSS Israel"
--
-- Safety
-- ------
-- - Only updates rows where the new value actually differs.
-- - Does NOT modify raw_data (the source of truth stays intact).
-- - Date format is verified: all 14,586 non-null start_dates in the DB are
--   full ISO 8601 (YYYY-MM-DD...). Lexicographic DESC sort on ISO 8601 strings
--   matches datetime ordering, so plain SQL is correct. NULL start_dates sort
--   LAST via NULLS LAST, matching pick_current_employer's behavior.
-- - Not in scope: profiles where the stored raw_data has only 1 current_employer
--   because Crustdata was sparse at enrichment time (Gil Gitlin, Barak Ben Shimon
--   class). Those need re-enrichment from Crustdata, handled in a separate PR.
--
-- Operational notes
-- -----------------
-- 1. Run the DRY-RUN block first to confirm the row count and sample.
-- 2. Run the UPDATE inside a transaction so it can be rolled back.
-- 3. Run the verification block to confirm zero remaining diffs.
-- ============================================================================


-- --------------------------------------------------------------------------
-- DRY-RUN PREVIEW: count + first 20 rows that will change
-- --------------------------------------------------------------------------
WITH multi AS (
  SELECT
    linkedin_url,
    name,
    current_company AS stored_company,
    current_title   AS stored_title,
    raw_data->'current_employers' AS emps
  FROM profiles
  WHERE jsonb_typeof(raw_data->'current_employers') = 'array'
    AND jsonb_array_length(raw_data->'current_employers') >= 2
),
picked AS (
  SELECT
    m.linkedin_url,
    m.name,
    m.stored_company,
    m.stored_title,
    (SELECT e->>'employer_name'
       FROM jsonb_array_elements(m.emps) e
       ORDER BY (e->>'start_date') DESC NULLS LAST
       LIMIT 1) AS new_company,
    (SELECT COALESCE(e->>'employee_title', e->>'title')
       FROM jsonb_array_elements(m.emps) e
       ORDER BY (e->>'start_date') DESC NULLS LAST
       LIMIT 1) AS new_title
  FROM multi m
)
SELECT
  COUNT(*)                                                                AS would_change_rows,
  COUNT(*) FILTER (WHERE stored_company IS DISTINCT FROM new_company)     AS company_changes,
  COUNT(*) FILTER (WHERE stored_title   IS DISTINCT FROM new_title)       AS title_changes
FROM picked
WHERE stored_company IS DISTINCT FROM new_company
   OR stored_title   IS DISTINCT FROM new_title;


-- --------------------------------------------------------------------------
-- THE BACKFILL — wrap in a transaction so you can ROLLBACK if the count
-- doesn't match the dry-run above.
-- --------------------------------------------------------------------------
BEGIN;

WITH multi AS (
  SELECT
    linkedin_url,
    raw_data->'current_employers' AS emps
  FROM profiles
  WHERE jsonb_typeof(raw_data->'current_employers') = 'array'
    AND jsonb_array_length(raw_data->'current_employers') >= 2
),
picked AS (
  SELECT
    m.linkedin_url,
    (SELECT e->>'employer_name'
       FROM jsonb_array_elements(m.emps) e
       ORDER BY (e->>'start_date') DESC NULLS LAST
       LIMIT 1) AS new_company,
    (SELECT COALESCE(e->>'employee_title', e->>'title')
       FROM jsonb_array_elements(m.emps) e
       ORDER BY (e->>'start_date') DESC NULLS LAST
       LIMIT 1) AS new_title
  FROM multi m
)
UPDATE profiles p
SET
  current_company = picked.new_company,
  current_title   = picked.new_title
FROM picked
WHERE p.linkedin_url = picked.linkedin_url
  AND (p.current_company IS DISTINCT FROM picked.new_company
    OR p.current_title   IS DISTINCT FROM picked.new_title);

-- Inspect the row count printed by the UPDATE. If it matches the dry-run
-- (~1,900), COMMIT. Otherwise ROLLBACK and investigate.
-- COMMIT;
-- ROLLBACK;


-- --------------------------------------------------------------------------
-- POST-RUN VERIFICATION: should return 0 rows once the COMMIT lands.
-- --------------------------------------------------------------------------
WITH multi AS (
  SELECT
    linkedin_url,
    current_company AS stored_company,
    current_title   AS stored_title,
    raw_data->'current_employers' AS emps
  FROM profiles
  WHERE jsonb_typeof(raw_data->'current_employers') = 'array'
    AND jsonb_array_length(raw_data->'current_employers') >= 2
),
picked AS (
  SELECT
    m.linkedin_url,
    m.stored_company,
    m.stored_title,
    (SELECT e->>'employer_name'
       FROM jsonb_array_elements(m.emps) e
       ORDER BY (e->>'start_date') DESC NULLS LAST
       LIMIT 1) AS new_company,
    (SELECT COALESCE(e->>'employee_title', e->>'title')
       FROM jsonb_array_elements(m.emps) e
       ORDER BY (e->>'start_date') DESC NULLS LAST
       LIMIT 1) AS new_title
  FROM multi m
)
SELECT COUNT(*) AS remaining_diffs
FROM picked
WHERE stored_company IS DISTINCT FROM new_company
   OR stored_title   IS DISTINCT FROM new_title;
