-- Migration: Extend full-text search to index raw_data fields,
--            lighten the RPC payload, and add LinkedIn-style Boolean syntax.
--
-- WHY:
--   1. The original index (migration 009) only covered 8 flat columns.
--      Free-text inside raw_data (headline, summary, past job descriptions,
--      education, languages) was not searchable. We now index those too.
--
--   2. The previous RPC `search_profiles_text` returned `SETOF profiles`,
--      including the heavy `raw_data` JSONB column (~20-50 KB per row).
--      At limit=50000 the Streamlit app (1 GB cap, free tier) had to parse
--      ~1.5 GB of JSON just to discard it client-side. The RPC now returns
--      only the lightweight columns the dashboard actually displays.
--
--   3. The previous RPC used `websearch_to_tsquery`, which doesn't support
--      parentheses, the `NOT` keyword, or comma-as-OR. We now translate
--      LinkedIn-style Boolean input (AND / OR / NOT / parens / comma /
--      dash-prefix / implicit-AND) into `to_tsquery` syntax. On parse
--      failure we fall back to `websearch_to_tsquery` so a malformed query
--      never breaks the search.
--
-- This migration is idempotent (CREATE OR REPLACE / DROP IF EXISTS).

-- ----------------------------------------------------------------------
-- 1. Trigger: build search_text from flat columns + raw_data extracts
-- ----------------------------------------------------------------------
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

-- ----------------------------------------------------------------------
-- 2. Translator: LinkedIn-style Boolean -> to_tsquery syntax
--    Supported user input:
--      AND, OR, NOT     (case-insensitive)
--      &, |, !, ()      (raw operators also accepted)
--      ,                (comma = OR)
--      - prefix         (-word = NOT word)
--      adjacent words   (implicit AND, like LinkedIn / Google)
-- ----------------------------------------------------------------------
CREATE OR REPLACE FUNCTION translate_user_boolean(input TEXT)
RETURNS TEXT AS $$
DECLARE
  result TEXT;
  prev TEXT;
BEGIN
  IF input IS NULL OR length(trim(input)) = 0 THEN
    RETURN '';
  END IF;

  result := input;

  -- Keyword translations (case-insensitive, word-boundary)
  result := regexp_replace(result, '\s+AND\s+', ' & ', 'gi');
  result := regexp_replace(result, '\s+OR\s+',  ' | ', 'gi');
  result := regexp_replace(result, '\s+NOT\s+', ' & !', 'gi');
  result := regexp_replace(result, '^NOT\s+',   '!',   'i');

  -- Comma as OR
  result := replace(result, ',', ' | ');

  -- Dash-prefix as NOT (only at start or after space/operator)
  result := regexp_replace(result, '(^|[\s&|(])-(\S)', '\1!\2', 'g');

  -- Implicit AND between adjacent tokens; lookahead to avoid consuming
  -- the next token's first char. Loop until no more substitutions.
  LOOP
    prev := result;
    result := regexp_replace(result, '([^\s&|!(])\s+(?=[^\s&|)])', '\1 & ', 'g');
    EXIT WHEN result = prev;
  END LOOP;

  -- Collapse whitespace
  result := regexp_replace(result, '\s+', ' ', 'g');
  result := trim(result);

  RETURN result;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ----------------------------------------------------------------------
-- 3. RPC: lightweight return shape + Boolean translation with fallback
-- ----------------------------------------------------------------------
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
DECLARE
  ts_query tsquery;
BEGIN
  -- Try full Boolean syntax via to_tsquery
  BEGIN
    ts_query := to_tsquery('english', translate_user_boolean(query));
  EXCEPTION WHEN OTHERS THEN
    -- Fall back to websearch (forgiving parser) on any syntax error
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
    ts_rank(p.search_text, ts_query) AS rank
  FROM profiles p
  WHERE p.search_text @@ ts_query
  ORDER BY rank DESC
  LIMIT p_limit
  OFFSET p_offset;
END;
$$ LANGUAGE plpgsql STABLE;

GRANT EXECUTE ON FUNCTION search_profiles_text(TEXT, INT, INT) TO anon;
GRANT EXECUTE ON FUNCTION search_profiles_text(TEXT, INT, INT) TO authenticated;

-- ----------------------------------------------------------------------
-- 4. Backfill: rebuild search_text for every profile under the new trigger.
--    Run in batches outside this file (the MCP timeout cuts a single
--    UPDATE on 30k rows). See the four batch UPDATEs that were applied:
--      WITH batch AS (SELECT id FROM profiles ORDER BY id LIMIT 10000 OFFSET <0|10000|20000|30000>)
--      UPDATE profiles p SET updated_at = updated_at FROM batch b WHERE p.id = b.id;
-- ----------------------------------------------------------------------
