-- Migration: Add semantic-search embeddings to profiles.
--
-- WHY:
--   Full-text search (migration 018) matches exact words. We want to find
--   profiles that are *similar in meaning* — same kind of career, even when
--   the words differ ("Backend Engineer" ≈ "Server-side Developer"). pgvector
--   stores an embedding vector per profile; an HNSW index makes nearest-
--   neighbour search return matches in milliseconds even at 40k+ rows.
--
-- WHAT IT ADDS:
--   1. `vector` extension (already installed at v0.8.0 in this project,
--      ensured here for portability).
--   2. Columns on profiles:
--        embedding              vector(1536)   -- OpenAI text-embedding-3-small
--        embedded_at            timestamptz    -- when this row was embedded
--        embedding_model        text           -- which model produced the vector
--        embedding_input_hash   text           -- md5 of the text we embedded;
--                                                lets the backfill skip rows that
--                                                haven't changed and detect drift
--   3. HNSW index on `embedding` using `vector_cosine_ops`.
--   4. `match_profiles_by_embedding(query, k, min_similarity)` RPC that
--      returns the k nearest profiles, ordered by cosine similarity.
--
-- DIMENSION CHOICE:
--   1536 dims matches OpenAI `text-embedding-3-small` (cheap, fast, good
--   quality) and the older `text-embedding-ada-002`. If we ever move to
--   `text-embedding-3-large` (3072 dims) we add a new column rather than
--   resize this one — pgvector dimension changes require a rewrite.
--
-- This migration is idempotent (CREATE EXTENSION IF NOT EXISTS,
-- ADD COLUMN IF NOT EXISTS, CREATE INDEX IF NOT EXISTS,
-- CREATE OR REPLACE FUNCTION).

-- ----------------------------------------------------------------------
-- 1. Extension
-- ----------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;

-- ----------------------------------------------------------------------
-- 2. Columns
-- ----------------------------------------------------------------------
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS embedding vector(1536);
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS embedded_at timestamptz;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS embedding_model text;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS embedding_input_hash text;

-- ----------------------------------------------------------------------
-- 3. HNSW index for fast cosine-similarity search.
--    HNSW chosen over IVFFlat: no training step required, robust at our
--    row count (~40k now, growing), and recall is high out of the box.
-- ----------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_profiles_embedding_hnsw
  ON profiles
  USING hnsw (embedding vector_cosine_ops);

-- Partial index on rows still missing an embedding makes the backfill
-- worker's "give me the next batch" query O(missing rows), not O(table).
CREATE INDEX IF NOT EXISTS idx_profiles_embedding_missing
  ON profiles (linkedin_url)
  WHERE embedding IS NULL;

-- ----------------------------------------------------------------------
-- 4. Similarity-search RPC
--    Returns lightweight columns (no raw_data) ordered by cosine
--    similarity descending. `min_similarity` lets callers cut off weak
--    matches; pass 0.0 to disable.
-- ----------------------------------------------------------------------
DROP FUNCTION IF EXISTS match_profiles_by_embedding(vector, int, float);

CREATE OR REPLACE FUNCTION match_profiles_by_embedding(
  query_embedding vector(1536),
  match_count int DEFAULT 20,
  min_similarity float DEFAULT 0.0
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
  ORDER BY p.embedding <=> query_embedding
  LIMIT match_count;
$$ LANGUAGE sql STABLE;

GRANT EXECUTE ON FUNCTION match_profiles_by_embedding(vector, int, float) TO anon;
GRANT EXECUTE ON FUNCTION match_profiles_by_embedding(vector, int, float) TO authenticated;
