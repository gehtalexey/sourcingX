-- Migration: Add full-text search for profiles
-- Run this in Supabase SQL Editor to enable fast keyword search across all profile data

-- 1. Add column for searchable text (tsvector is PostgreSQL's full-text search type)
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS search_text TSVECTOR;

-- 2. Create GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS idx_profiles_search ON profiles USING GIN(search_text);

-- 3. Backfill existing profiles with searchable text
-- This combines: name, title, company, location, skills, employers, titles, schools
UPDATE profiles
SET search_text = to_tsvector('english',
  COALESCE(name, '') || ' ' ||
  COALESCE(current_title, '') || ' ' ||
  COALESCE(current_company, '') || ' ' ||
  COALESCE(location, '') || ' ' ||
  COALESCE(array_to_string(skills, ' '), '') || ' ' ||
  COALESCE(array_to_string(all_employers, ' '), '') || ' ' ||
  COALESCE(array_to_string(all_titles, ' '), '') || ' ' ||
  COALESCE(array_to_string(all_schools, ' '), '')
)
WHERE raw_data IS NOT NULL;

-- 4. Create trigger function to auto-update search_text on insert/update
CREATE OR REPLACE FUNCTION update_search_text()
RETURNS TRIGGER AS $$
BEGIN
  NEW.search_text := to_tsvector('english',
    COALESCE(NEW.name, '') || ' ' ||
    COALESCE(NEW.current_title, '') || ' ' ||
    COALESCE(NEW.current_company, '') || ' ' ||
    COALESCE(NEW.location, '') || ' ' ||
    COALESCE(array_to_string(NEW.skills, ' '), '') || ' ' ||
    COALESCE(array_to_string(NEW.all_employers, ' '), '') || ' ' ||
    COALESCE(array_to_string(NEW.all_titles, ' '), '') || ' ' ||
    COALESCE(array_to_string(NEW.all_schools, ' '), '')
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 5. Create trigger to run on insert/update
DROP TRIGGER IF EXISTS trg_search_text ON profiles;
CREATE TRIGGER trg_search_text
  BEFORE INSERT OR UPDATE ON profiles
  FOR EACH ROW EXECUTE FUNCTION update_search_text();

-- 6. Create RPC function for full-text search (called from Python)
CREATE OR REPLACE FUNCTION search_profiles_text(query TEXT, p_limit INT DEFAULT 500)
RETURNS SETOF profiles AS $$
  SELECT * FROM profiles
  WHERE search_text @@ plainto_tsquery('english', query)
  ORDER BY ts_rank(search_text, plainto_tsquery('english', query)) DESC
  LIMIT p_limit;
$$ LANGUAGE SQL;

-- Grant access to anon role (for Streamlit app)
GRANT EXECUTE ON FUNCTION search_profiles_text TO anon;
