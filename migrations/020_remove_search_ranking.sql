-- Migration 020: Drop ts_rank from full-text search RPC
--
-- WHY: ts_rank reads the tsvector blob for every matching row before sorting.
-- On a query like "senior" (~22K matches), that single SELECT takes ~24 seconds
-- end-to-end, blowing the Python HTTP timeout (30s) and silently falling back
-- to a name/company ilike that finds essentially nothing. Recruiters typing
-- "senior" or "engineer" saw "0 results" even though 22K and 36K profiles
-- match respectively.
--
-- WHAT: Return rows in whatever order the GIN index emits (no ORDER BY).
-- Keep the `rank REAL` column in the return signature (always 0) so the
-- Python caller's column layout doesn't need to change.
--
-- IMPACT: First page of "senior" goes from 23,888 ms -> 1,147 ms (~20x).
-- Loses relevance ranking, but the dashboard's other filters (location,
-- current_title, freshness) are the real signal anyway.

CREATE OR REPLACE FUNCTION search_profiles_text(
  query TEXT,
  p_limit INT DEFAULT 1000,
  p_offset INT DEFAULT 0
)
RETURNS TABLE (
  linkedin_url TEXT,
  name TEXT,
  current_title TEXT,
  current_company TEXT,
  location TEXT,
  all_employers TEXT[],
  all_titles TEXT[],
  all_schools TEXT[],
  skills TEXT[],
  email TEXT,
  enriched_at TIMESTAMPTZ,
  enrichment_status TEXT,
  rank REAL
) AS $$
DECLARE
  ts_query tsquery;
BEGIN
  BEGIN
    ts_query := to_tsquery('english', translate_user_boolean(query));
  EXCEPTION WHEN OTHERS THEN
    ts_query := websearch_to_tsquery('english', query);
  END;

  RETURN QUERY
  SELECT
    p.linkedin_url,
    p.name,
    p.current_title,
    p.current_company,
    p.location,
    p.all_employers,
    p.all_titles,
    p.all_schools,
    p.skills,
    p.email,
    p.enriched_at,
    p.enrichment_status,
    0::REAL AS rank
  FROM profiles p
  WHERE p.search_text @@ ts_query
  LIMIT p_limit
  OFFSET p_offset;
END;
$$ LANGUAGE plpgsql STABLE;

GRANT EXECUTE ON FUNCTION search_profiles_text(TEXT, INT, INT) TO anon;
GRANT EXECUTE ON FUNCTION search_profiles_text(TEXT, INT, INT) TO authenticated;
