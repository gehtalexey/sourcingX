-- Migration 019: Index certifications in full-text search
--
-- WHY: Crustdata returns certifications (name + authority) in raw_data but
-- migration 018's trigger didn't extract them. Recruiters searching for
-- "AWS", "CKA", "GCP", "PMP", etc. get zero results even when the profile
-- has the cert. This adds certifications to the tsvector and backfills.
--
-- WHAT: Extends update_search_text() to append cert names + authorities.
-- The RPC (search_profiles_text) is unchanged — it queries search_text,
-- so no RPC changes needed.

-- 1. Replace the trigger function to include certifications
CREATE OR REPLACE FUNCTION update_search_text()
RETURNS TRIGGER AS $$
DECLARE
  raw_text TEXT := '';
BEGIN
  IF NEW.raw_data IS NOT NULL THEN
    raw_text :=
      COALESCE(NEW.raw_data->>'headline', '') || ' ' ||
      COALESCE(NEW.raw_data->>'summary', '') || ' ' ||
      COALESCE((
        SELECT string_agg(
          COALESCE(elem->>'employee_description', '') || ' ' ||
          COALESCE(elem->>'employee_title', ''),
          ' '
        )
        FROM jsonb_array_elements(COALESCE(NEW.raw_data->'past_employers', '[]'::jsonb)) AS elem
      ), '') || ' ' ||
      COALESCE((
        SELECT string_agg(elem->>'employee_description', ' ')
        FROM jsonb_array_elements(COALESCE(NEW.raw_data->'current_employers', '[]'::jsonb)) AS elem
        WHERE elem->>'employee_description' IS NOT NULL
      ), '') || ' ' ||
      COALESCE((
        SELECT string_agg(
          COALESCE(elem->>'degree_name', '') || ' ' || COALESCE(elem->>'field_of_study', ''),
          ' '
        )
        FROM jsonb_array_elements(COALESCE(NEW.raw_data->'education_background', '[]'::jsonb)) AS elem
      ), '') || ' ' ||
      COALESCE((
        SELECT string_agg(lang, ' ')
        FROM jsonb_array_elements_text(COALESCE(NEW.raw_data->'languages', '[]'::jsonb)) AS lang
      ), '') || ' ' ||
      COALESCE((
        SELECT string_agg(
          COALESCE(elem->>'name', '') || ' ' || COALESCE(elem->>'authority', ''),
          ' '
        )
        FROM jsonb_array_elements(COALESCE(NEW.raw_data->'certifications', '[]'::jsonb)) AS elem
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

-- 2. Backfill: touch every row so the new trigger fires and rebuilds search_text.
--    Run in batches to avoid timeout. Four UPDATEs of ~10k rows each:
--
--    WITH batch AS (SELECT id FROM profiles ORDER BY id LIMIT 10000 OFFSET 0)
--    UPDATE profiles p SET updated_at = updated_at FROM batch b WHERE p.id = b.id;
--
--    WITH batch AS (SELECT id FROM profiles ORDER BY id LIMIT 10000 OFFSET 10000)
--    UPDATE profiles p SET updated_at = updated_at FROM batch b WHERE p.id = b.id;
--
--    ... repeat with OFFSET 20000, 30000 as needed.
--
-- OR run once if the table is small enough not to timeout:
--    UPDATE profiles SET updated_at = updated_at;
