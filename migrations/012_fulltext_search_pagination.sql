-- Migration: Add pagination + boolean search support to full-text search
-- Run this in Supabase SQL Editor

-- Drop old function and recreate with to_tsquery for boolean support
DROP FUNCTION IF EXISTS search_profiles_text(TEXT, INT, INT);

CREATE OR REPLACE FUNCTION search_profiles_text(query TEXT, p_limit INT DEFAULT 1000, p_offset INT DEFAULT 0)
RETURNS SETOF profiles AS $$
  SELECT * FROM profiles
  WHERE search_text @@ to_tsquery('english', query)
  ORDER BY ts_rank(search_text, to_tsquery('english', query)) DESC
  LIMIT p_limit
  OFFSET p_offset;
$$ LANGUAGE SQL;

-- Grant access to anon role
GRANT EXECUTE ON FUNCTION search_profiles_text TO anon;
