-- Migration: Increase full-text search limit and use websearch_to_tsquery for boolean support
-- Run this in Supabase SQL Editor

-- Update the search function with higher limit and boolean search support
CREATE OR REPLACE FUNCTION search_profiles_text(query TEXT, p_limit INT DEFAULT 50000)
RETURNS SETOF profiles AS $$
  SELECT * FROM profiles
  WHERE search_text @@ websearch_to_tsquery('english', query)
  ORDER BY ts_rank(search_text, websearch_to_tsquery('english', query)) DESC
  LIMIT p_limit;
$$ LANGUAGE SQL;

-- Grant access to anon role
GRANT EXECUTE ON FUNCTION search_profiles_text TO anon;
