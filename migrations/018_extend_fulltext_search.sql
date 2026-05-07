-- Migration: Extend full-text search to index raw_data fields + lighten RPC payload
--
-- WHY:
--   1. The original full-text index (migration 009) only covered 8 flat columns.
--      Free-text fields inside raw_data (headline, summary, past job descriptions,
--      education degree/field, languages) were never searchable. This migration
--      indexes them so users can find profiles by content like "fintech" in a
--      summary or "Kubernetes" mentioned only in a past job description.
--
--   2. The previous RPC `search_profiles_text` returned `SETOF profiles`, which
--      includes the heavy `raw_data` JSONB column (~20-50 KB per row). At
--      limit=50000, that meant up to ~1.5 GB transferred to the Streamlit app
--      (free tier, 1 GB RAM hard cap) just to be discarded client-side.
--      The RPC now returns only the lightweight columns the dashboard displays.
--
-- This migration is idempotent (CREATE OR REPLACE).

-- 1. Trigger function: build search_text from flat columns + raw_data extracts
CREATE OR REPLACE FUNCTION update_search_text()
RETURNS TRIGGER AS $$
DECLARE
  raw_text TEXT := '';
BEGIN
  IF NEW.raw_data IS NOT NULL THEN
    raw_text :=
      COALESCE(NEW.raw_data->>'headline', '') || ' ' ||
      COALESCE(NEW.raw_data->>'summary', '') || ' ' ||
      -- past job descriptions + titles (titles included for completeness in case all_titles is stale)
      COALESCE((
        SELECT string_agg(
          COALESCE(elem->>'employee_description', '') || ' ' ||
          COALESCE(elem->>'employee_title', ''),
          ' '
        )
        FROM jsonb_array_elements(COALESCE(NEW.raw_data->'past_employers', '[]'::jsonb)) AS elem
      ), '') || ' ' ||
      -- current role description
      COALESCE((
        SELECT string_agg(elem->>'employee_description', ' ')
        FROM jsonb_array_elements(COALESCE(NEW.raw_data->'current_employers', '[]'::jsonb)) AS elem
        WHERE elem->>'employee_description' IS NOT NULL
      ), '') || ' ' ||
      -- education: degree + field of study
      COALESCE((
        SELECT string_agg(
          COALESCE(elem->>'degree_name', '') || ' ' || COALESCE(elem->>'field_of_study', ''),
          ' '
        )
        FROM jsonb_array_elements(COALESCE(NEW.raw_data->'education_background', '[]'::jsonb)) AS elem
      ), '') || ' ' ||
      -- languages (array of strings)
      COALESCE((
        SELECT string_agg(lang, ' ')
        FROM jsonb_array_elements_text(COALESCE(NEW.raw_data->'languages', '[]'::jsonb)) AS lang
      ), '');
  END IF;

  NEW.search_text := to_tsvector('english',
    COALESCE(NEW.name, '') || ' ' ||
    COALESCE(NEW.current_title, '') || ' ' ||
    COALESCE(NEW.current_company, '') || ' ' ||
    COALESCE(NEW.location, '') || ' ' ||
    COALESCE(array_to_string(NEW.skills, ' '), '') || ' ' ||
    COALESCE(array_to_string(NEW.all_employers, ' '), '') || ' ' ||
    COALESCE(array_to_string(NEW.all_titles, ' '), '') || ' ' ||
    COALESCE(array_to_string(NEW.all_schools, ' '), '') || ' ' ||
    raw_text
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 2. RPC: drop old SETOF profiles signature, return only lightweight columns
DROP FUNCTION IF EXISTS search_profiles_text(TEXT, INT, INT);
DROP FUNCTION IF EXISTS search_profiles_text(TEXT, INT);

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
    ts_rank(p.search_text, websearch_to_tsquery('english', query)) AS rank
  FROM profiles p
  WHERE p.search_text @@ websearch_to_tsquery('english', query)
  ORDER BY rank DESC
  LIMIT p_limit
  OFFSET p_offset;
$$ LANGUAGE SQL STABLE;

GRANT EXECUTE ON FUNCTION search_profiles_text(TEXT, INT, INT) TO anon;
GRANT EXECUTE ON FUNCTION search_profiles_text(TEXT, INT, INT) TO authenticated;

-- 3. Backfill: rebuild search_text for every profile using the new logic.
--    Triggers BEFORE UPDATE on profiles, so each row gets the new tsvector.
--    NOTE: this is a single UPDATE on ~30k rows. It will take 30-90 seconds and
--    briefly hold row locks. Run during low-traffic time.
UPDATE profiles SET updated_at = updated_at;
