-- Migration: Add an optional location filter to the similarity-search RPC.
--
-- WHY:
--   The "Find Similar Profiles" tab searches the whole database. Recruiters
--   often want "similar people, but only in Israel" (or only in a given
--   city). This adds an optional list of location substrings the match must
--   contain — so the search can be scoped to a country/city without changing
--   how similarity itself is computed.
--
-- WHAT IT ADDS:
--   A 4th parameter to match_profiles_by_embedding:
--       location_terms text[] DEFAULT '{}'
--   When empty/NULL the function behaves EXACTLY as before (no filter), so
--   existing 3-argument callers are unaffected. When populated, a profile
--   only qualifies if its free-text `location` contains (case-insensitive)
--   at least one of the terms.
--
-- TERM EXPANSION HAPPENS IN PYTHON:
--   The app (geo_terms.py) turns one user input ("Israel" / "Tel Aviv") into
--   the full list of substrings ("israel", "tel aviv", "tel aviv-yafo",
--   "herzliya", ...). The database just does the OR-contains match. Keeping
--   the term list in Python means we can grow coverage without a migration.
--
-- ACCURACY NOTE (filtered nearest-neighbour search):
--   The HNSW index returns approximate nearest neighbours and applies the
--   location filter *during* the index walk. Without help, a selective
--   filter (e.g. a single city) can make the walk run out of candidates and
--   return FEWER than match_count rows even when more qualify — or miss the
--   truly closest ones. pgvector 0.8.0 fixes this with iterative index
--   scans, which keep walking until the LIMIT is satisfied. We enable it at
--   the function level with `SET hnsw.iterative_scan = 'strict_order'`:
--   'strict_order' keeps results in exact similarity order (no reshuffling),
--   so the ranking the user sees stays correct. This is off by default in
--   0.8.0, hence the explicit SET. (Confirmed: vector extension is 0.8.0.)
--
-- Idempotent: CREATE OR REPLACE. We DROP the old 3-arg signature so there is
-- exactly one function and no overload ambiguity for PostgREST.

DROP FUNCTION IF EXISTS match_profiles_by_embedding(vector, int, float);
DROP FUNCTION IF EXISTS match_profiles_by_embedding(vector, int, float, text[]);

CREATE OR REPLACE FUNCTION match_profiles_by_embedding(
  query_embedding vector(1536),
  match_count int DEFAULT 20,
  min_similarity float DEFAULT 0.0,
  location_terms text[] DEFAULT '{}'
)
RETURNS TABLE (
  linkedin_url text,
  name text,
  current_title text,
  current_company text,
  location text,
  all_employers text[],
  all_titles text[],
  all_schools text[],
  skills text[],
  email text,
  enriched_at timestamptz,
  enrichment_status text,
  similarity float
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
    1 - (p.embedding <=> query_embedding) AS similarity
  FROM profiles p
  WHERE p.embedding IS NOT NULL
    AND 1 - (p.embedding <=> query_embedding) >= min_similarity
    AND (
      location_terms IS NULL
      OR cardinality(location_terms) = 0
      OR EXISTS (
        SELECT 1
        FROM unnest(location_terms) AS term
        WHERE p.location ILIKE '%' || term || '%'
      )
    )
  ORDER BY p.embedding <=> query_embedding
  LIMIT match_count;
$$ LANGUAGE sql STABLE
  SET hnsw.iterative_scan = 'strict_order';

GRANT EXECUTE ON FUNCTION match_profiles_by_embedding(vector, int, float, text[]) TO anon;
GRANT EXECUTE ON FUNCTION match_profiles_by_embedding(vector, int, float, text[]) TO authenticated;
