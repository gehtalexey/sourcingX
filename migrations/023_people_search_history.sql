-- Migration: People DB search history (per-user)
-- Records each Crustdata People-DB search a user runs, so the Search tab can
-- show "your recent searches" and warn before re-running an identical one.
-- Separate from the existing `search_history` table, which logs PhantomBuster
-- launches (agent_id / csv_name) and is unrelated to the Search-tab filters.
--
-- Privacy is per-user at the app layer: the app always filters on `username`.
-- RLS is enabled with no policy, matching the other app tables (profiles,
-- search_history, api_usage_logs). The app connects with the service-role key,
-- which bypasses RLS; enabling RLS keeps anon/authenticated locked out by default.

CREATE TABLE IF NOT EXISTS people_search_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT NOT NULL,
    filters_hash TEXT NOT NULL,        -- stable hash of the matching filters (sort/limit excluded)
    filters JSONB NOT NULL,            -- the full filter bundle, for one-click reload
    summary TEXT,                      -- short human label, e.g. "team lead · Tel Aviv · Senior"
    result_count INTEGER,              -- total matching profiles the last run reported
    run_count INTEGER NOT NULL DEFAULT 1,
    first_run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (username, filters_hash)
);

-- Recent-searches list: newest first, scoped to the logged-in user.
CREATE INDEX IF NOT EXISTS idx_people_search_history_user_recent
    ON people_search_history (username, last_run_at DESC);

-- RLS on, no policy (service-role key bypasses it) — same posture as profiles/search_history.
ALTER TABLE people_search_history ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE people_search_history IS 'Per-user history of Crustdata People-DB searches (Search tab). Deduped by (username, filters_hash).';
COMMENT ON COLUMN people_search_history.filters_hash IS 'sha256 of canonicalized matching filters; sort/limit excluded so they do not count as a different search.';
COMMENT ON COLUMN people_search_history.filters IS 'Full filter bundle keyed by Search-tab widget keys, for one-click reload.';
COMMENT ON COLUMN people_search_history.run_count IS 'How many times this exact search has been run (bumped on repeat).';
