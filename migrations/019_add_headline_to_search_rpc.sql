-- Migration: return raw_data->>'headline' from search_profiles_text.
--
-- WHY:
--   The DB-tab "Current Title" column filter is applied client-side over
--   the rows the RPC returns. Today the RPC does NOT return headline, so
--   the filter only matches against the structured current_title column.
--   Many candidates have their effective title only in their headline
--   (e.g. "Senior Python Engineer at Wiz"), and they get dropped from the
--   funnel even when the full-text phase matched them via search_text
--   (which already indexes headline — see migration 018).
--
--   Returning headline lets the client OR the title filter across
--   current_title and headline, the same shape PR #44 ships for the
--   Crustdata side.
--
--   As of writing, 30,982 / 32,540 profiles (~95%) have a non-null
--   raw_data->>'headline'.
--
-- This migration DROPs and recreates the RPC because the return type is
-- changing (a new column). The function body is otherwise byte-identical
-- to migration 018's definition.

DROP FUNCTION IF EXISTS public.search_profiles_text(TEXT, INT, INT);

CREATE OR REPLACE FUNCTION public.search_profiles_text(
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
  headline TEXT,
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
    COALESCE(p.raw_data->>'headline', '') AS headline,
    ts_rank(p.search_text, ts_query) AS rank
  FROM profiles p
  WHERE p.search_text @@ ts_query
  ORDER BY rank DESC
  LIMIT p_limit
  OFFSET p_offset;
END;
$$ LANGUAGE plpgsql STABLE;

GRANT EXECUTE ON FUNCTION public.search_profiles_text(TEXT, INT, INT) TO anon;
GRANT EXECUTE ON FUNCTION public.search_profiles_text(TEXT, INT, INT) TO authenticated;
