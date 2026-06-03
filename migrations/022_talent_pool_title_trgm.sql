-- Migration 022: trigram pre-filter columns/indexes on profiles (talent-pool title scan)
--
-- ALREADY APPLIED to the shared database (via the israel-sourcing-autopilot pipeline,
-- its migration 009) on 2026-06-03. Recorded HERE because SourcingX is the canonical
-- project and `profiles` + the `update_search_text()` trigger are SHARED across all
-- sibling pipelines — this file keeps SourcingX's history truthful and lets a
-- from-scratch DB rebuild reproduce the live state. It is IDEMPOTENT (IF NOT EXISTS /
-- CREATE OR REPLACE / fills only NULLs), so re-running is a safe no-op.
--
-- WHAT: the Israel pipeline added a server-side title/skills pre-filter for its
-- talent-pool scan. PostgREST can ILIKE the scalar `current_title` directly but
-- cannot substring-match inside the `skills`/`all_titles` ARRAY columns, so two
-- flattened text columns + trigram indexes were added, and the shared trigger was
-- extended to maintain them. **`search_text` is untouched** — SourcingX's full-text
-- search behaves exactly as before. SourcingX does not query the new columns; they
-- are additive for the sibling pipeline.

-- 1. Trigram matching support
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 2. Flattened, lower-cased searchable text for the array columns (nullable add)
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS skills_blob text;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS titles_blob text;

-- 3. Extend the shared trigger to maintain the two blobs on insert/update.
--    This is the CURRENT LIVE update_search_text() body, captured verbatim via
--    pg_get_functiondef, with only the two NEW.*_blob assignments added before
--    RETURN NEW. The search_text expression is byte-for-byte the live one and is
--    deliberately NOT changed here.
--    NB: the live body includes `current_company` TWICE (a pre-existing quirk
--    introduced by a migration after 019_add_certifications_to_search.sql) — that
--    duplicate is preserved on purpose. Removing it would CHANGE the live trigger's
--    search_text output for all future writes (a real behaviour change to shared
--    full-text search), which is explicitly out of scope for this mirror/record.
CREATE OR REPLACE FUNCTION public.update_search_text()
RETURNS trigger
LANGUAGE plpgsql
AS $function$
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
    COALESCE(NEW.current_company, '') || ' ' ||
    COALESCE(NEW.location, '') || ' ' ||
    COALESCE(array_to_string(NEW.skills, ' '), '') || ' ' ||
    COALESCE(array_to_string(NEW.all_employers, ' '), '') || ' ' ||
    COALESCE(array_to_string(NEW.all_titles, ' '), '') || ' ' ||
    COALESCE(array_to_string(NEW.all_schools, ' '), '') || ' ' ||
    raw_text
  );

  -- trigram pre-filter blobs (mirror talent_pool._profile_text_values in the
  -- israel-sourcing-autopilot pipeline)
  NEW.skills_blob := lower(COALESCE(array_to_string(NEW.skills, ' '), ''));
  NEW.titles_blob := lower(COALESCE(array_to_string(NEW.all_titles, ' '), ''));

  RETURN NEW;
END;
$function$;

-- 4. Backfill existing rows (already done on the live shared DB; no-op if re-run
--    since blobs are populated — kept here so a fresh rebuild is complete).
UPDATE profiles SET
  skills_blob = lower(COALESCE(array_to_string(skills, ' '), '')),
  titles_blob = lower(COALESCE(array_to_string(all_titles, ' '), ''))
WHERE skills_blob IS NULL OR titles_blob IS NULL;

-- 5. Trigram GIN indexes for fast ILIKE '%term%'.
CREATE INDEX IF NOT EXISTS idx_profiles_current_title_trgm
  ON profiles USING gin (current_title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_profiles_skills_blob_trgm
  ON profiles USING gin (skills_blob gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_profiles_titles_blob_trgm
  ON profiles USING gin (titles_blob gin_trgm_ops);
