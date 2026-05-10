-- Migration: Add exact URL matching function (safer than fuzzy matching)
-- This function matches input URLs against linkedin_url OR original_url exactly
-- (with ID suffix stripping only, no name reversal or hyphen removal)

-- ============================================================================
-- EXACT MATCHING FUNCTION
-- ============================================================================

CREATE OR REPLACE FUNCTION match_profiles_by_urls_exact(
    input_urls TEXT[],
    enriched_after TIMESTAMPTZ DEFAULT NULL
)
RETURNS TABLE (
    linkedin_url TEXT,
    original_url TEXT,
    name TEXT,
    location TEXT,
    current_title TEXT,
    current_company TEXT,
    all_employers TEXT[],
    all_titles TEXT[],
    all_schools TEXT[],
    skills TEXT[],
    email TEXT,
    email_source TEXT,
    status TEXT,
    enriched_at TIMESTAMPTZ,
    screening_score INTEGER,
    screening_fit_level TEXT,
    screening_summary TEXT,
    screening_reasoning TEXT,
    screened_at TIMESTAMPTZ
) AS $$
DECLARE
    url_variations TEXT[] := '{}';
    u TEXT;
    username TEXT;
    base_username TEXT;
BEGIN
    -- Build URL variations from input (only exact + ID-stripped versions)
    FOREACH u IN ARRAY input_urls LOOP
        IF u IS NULL OR u = '' THEN
            CONTINUE;
        END IF;

        username := extract_linkedin_username(u);
        IF username IS NULL THEN
            CONTINUE;
        END IF;

        -- Add exact username
        url_variations := array_append(url_variations, username);

        -- Add username with ID suffix stripped (to match profiles stored with different suffix)
        base_username := strip_linkedin_id_suffix(username);
        IF base_username IS NOT NULL AND base_username != username THEN
            url_variations := array_append(url_variations, base_username);
        END IF;
    END LOOP;

    -- Remove duplicates
    SELECT array_agg(DISTINCT v) INTO url_variations FROM unnest(url_variations) v;

    -- Match against profiles - EXACT match on username extracted from URLs
    -- No name reversal, no hyphen removal (safer matching)
    RETURN QUERY
    SELECT DISTINCT ON (p.linkedin_url)
        p.linkedin_url,
        p.original_url,
        p.name,
        p.location,
        p.current_title,
        p.current_company,
        p.all_employers,
        p.all_titles,
        p.all_schools,
        p.skills,
        p.email,
        p.email_source,
        p.status,
        p.enriched_at,
        p.screening_score,
        p.screening_fit_level,
        p.screening_summary,
        p.screening_reasoning,
        p.screened_at
    FROM profiles p
    WHERE (
        -- Exact match on linkedin_url username
        extract_linkedin_username(p.linkedin_url) = ANY(url_variations)
        -- Or match on linkedin_url with ID stripped
        OR strip_linkedin_id_suffix(extract_linkedin_username(p.linkedin_url)) = ANY(url_variations)
        -- Exact match on original_url username
        OR extract_linkedin_username(p.original_url) = ANY(url_variations)
        -- Or match on original_url with ID stripped
        OR strip_linkedin_id_suffix(extract_linkedin_username(p.original_url)) = ANY(url_variations)
    )
    AND (enriched_after IS NULL OR p.enriched_at >= enriched_after)
    ORDER BY p.linkedin_url, p.enriched_at DESC;
END;
$$ LANGUAGE plpgsql STABLE;

-- Grant permissions
GRANT EXECUTE ON FUNCTION match_profiles_by_urls_exact TO anon;
