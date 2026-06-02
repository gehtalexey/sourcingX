-- Migration: Add an optional location filter to the similarity-search RPC.
--
-- WHY:
--   The "Find Similar Profiles" tab searches the whole database. Recruiters
--   often want "similar people, but only in Israel" — and to narrow further
--   to a city ("only in Tel Aviv"). This adds two optional groups of location
--   substrings the match must contain, scoping the search without changing
--   how similarity itself is computed.
--
-- WHAT IT ADDS:
--   Two array parameters to match_profiles_by_embedding:
--       country_terms text[] DEFAULT '{}'
--       city_terms    text[] DEFAULT '{}'
--   When both are empty/NULL the function behaves EXACTLY as before (no
--   filter). The two groups combine as AND, each group internally as OR:
--       (location matches ANY country term) AND (location matches ANY city term)
--   So country="Israel" + city="Tel Aviv" means "in Israel AND in Tel Aviv"
--   (i.e. city narrows within country) rather than "Israel OR Tel Aviv".
--   An empty group is treated as "no constraint from this group".
--
-- TERM EXPANSION HAPPENS IN PYTHON:
--   The app (geo_terms.py) turns one user input ("Israel" / "Tel Aviv") into
--   the full list of substrings ("israel", "tel aviv", "tel aviv-yafo",
--   "herzliya", ...). The database just does the OR-contains match per group.
--   Keeping the term lists in Python means we can grow coverage without a
--   migration.
--
-- ACCURACY NOTE (filtered nearest-neighbour search):
--   The HNSW index returns approximate nearest neighbours and applies the
--   location filter *during* the index walk. With a very selective filter
--   (e.g. a single small-town city in a sparse region) the walk can run out
--   of candidates and return FEWER than match_count rows even when more
--   qualify. pgvector 0.8.0 has iterative index scans to fix this
--   (`hnsw.iterative_scan = 'strict_order'`), but on this hosted database the
--   migration/SQL role is NOT permitted to pin that GUC onto a function
--   (`ERROR 42501: permission denied to set parameter`). So we do NOT attach
--   it here. In practice this is a non-issue for the primary use case: the
--   database is Israel-heavy, so an Israel/city filter still fills the full
--   match_count with correct results.
--
--   IF narrow-filter under-fill ever matters, enable iterative scans on the
--   PostgREST role that actually invokes this RPC. This app connects with the
--   Supabase service-role key (SUPABASE_KEY), so its requests run as the
--   `service_role` Postgres role — that is the one to set:
--       ALTER ROLE service_role SET hnsw.iterative_scan = 'strict_order';
--   (If the dashboard ever calls this RPC with the anon/authenticated keys,
--   set it on those roles too.) Needs elevated DB privileges, and the
--   database is SHARED by four projects, so coordinate first. It only affects
--   vector-index scans (i.e. this RPC), nothing else.
--   (Confirmed: vector extension is 0.8.0.)
--
-- Idempotent: CREATE OR REPLACE. We DROP older signatures so there is exactly
-- one function and no overload ambiguity for PostgREST.

DROP FUNCTION IF EXISTS match_profiles_by_embedding(vector, int, float);
DROP FUNCTION IF EXISTS match_profiles_by_embedding(vector, int, float, text[]);
DROP FUNCTION IF EXISTS match_profiles_by_embedding(vector, int, float, text[], text[]);

CREATE OR REPLACE FUNCTION match_profiles_by_embedding(
  query_embedding vector(1536),
  match_count int DEFAULT 20,
  min_similarity float DEFAULT 0.0,
  country_terms text[] DEFAULT '{}',
  city_terms text[] DEFAULT '{}'
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
    -- Country group (OR within). Empty group = no constraint.
    AND (
      country_terms IS NULL
      OR cardinality(country_terms) = 0
      OR EXISTS (
        SELECT 1
        FROM unnest(country_terms) AS term
        WHERE p.location ILIKE '%' || term || '%'
      )
    )
    -- City group (OR within), ANDed with the country group. Empty = no constraint.
    AND (
      city_terms IS NULL
      OR cardinality(city_terms) = 0
      OR EXISTS (
        SELECT 1
        FROM unnest(city_terms) AS term
        WHERE p.location ILIKE '%' || term || '%'
      )
    )
  ORDER BY p.embedding <=> query_embedding
  LIMIT match_count;
$$ LANGUAGE sql STABLE;

GRANT EXECUTE ON FUNCTION match_profiles_by_embedding(vector, int, float, text[], text[]) TO anon;
GRANT EXECUTE ON FUNCTION match_profiles_by_embedding(vector, int, float, text[], text[]) TO authenticated;
