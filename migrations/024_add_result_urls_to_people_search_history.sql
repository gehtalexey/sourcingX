-- Migration: store the people each People-DB search returned
-- Adds result_urls to people_search_history so the Search tab can offer a
-- "Load saved results (free)" button: re-open a past search and pull the exact
-- profiles it found straight from the `profiles` table, with no Crustdata call.
--
-- Backward-compatible: the column is nullable with no default. Older rows (and
-- searches recorded before this ships) simply have no saved links, so the free
-- button is hidden for them and the existing refill-filters + Search flow still
-- works unchanged.

ALTER TABLE people_search_history
    ADD COLUMN IF NOT EXISTS result_urls JSONB;  -- normalized LinkedIn URLs the last run loaded

COMMENT ON COLUMN people_search_history.result_urls IS 'Normalized LinkedIn URLs of the profiles this search loaded, for one-click free reload from the profiles table. NULL for searches recorded before this column existed.';
