-- Migration: Server-side URL matching for 100k+ profile scale
-- Run this in Supabase SQL Editor to enable efficient profile matching
--
-- This migration adds:
-- 1. Helper functions for URL processing (extract username, strip ID suffix, etc.)
-- 2. Main RPC function match_profiles_by_urls that matches input URLs against DB profiles
-- 3. Index on original_url column for faster lookups

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Extract username from LinkedIn URL (e.g., 'https://linkedin.com/in/john-doe' -> 'john-doe')
CREATE OR REPLACE FUNCTION extract_linkedin_username(url TEXT)
RETURNS TEXT AS $$
DECLARE
    username TEXT;
BEGIN
    IF url IS NULL OR url = '' THEN
        RETURN NULL;
    END IF;

    -- Extract portion after /in/
    IF position('/in/' in lower(url)) > 0 THEN
        username := split_part(lower(url), '/in/', 2);
        -- Remove trailing slash
        username := rtrim(username, '/');
        -- Remove query parameters
        username := split_part(username, '?', 1);
        RETURN username;
    END IF;

    RETURN NULL;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Strip numeric ID suffix from username (e.g., 'john-doe-12345' -> 'john-doe')
-- Only strips if suffix looks like an ID (all digits, or alphanumeric with >=50% digits)
CREATE OR REPLACE FUNCTION strip_linkedin_id_suffix(username TEXT)
RETURNS TEXT AS $$
DECLARE
    parts TEXT[];
    suffix TEXT;
    digit_count INT;
BEGIN
    IF username IS NULL OR username = '' THEN
        RETURN NULL;
    END IF;

    -- If no hyphen, return as-is
    IF position('-' in username) = 0 THEN
        RETURN username;
    END IF;

    -- Split on last hyphen
    parts := regexp_split_to_array(username, '-');
    IF array_length(parts, 1) < 2 THEN
        RETURN username;
    END IF;

    suffix := parts[array_length(parts, 1)];

    -- Check if suffix is all digits
    IF suffix ~ '^[0-9]+$' THEN
        -- Return everything except the last part
        RETURN array_to_string(parts[1:array_length(parts, 1)-1], '-');
    END IF;

    -- Check if suffix is alphanumeric with >=50% digits and length >= 5
    IF length(suffix) >= 5 AND suffix ~ '^[a-z0-9]+$' THEN
        digit_count := length(regexp_replace(suffix, '[^0-9]', '', 'g'));
        IF digit_count::float / length(suffix) >= 0.5 THEN
            RETURN array_to_string(parts[1:array_length(parts, 1)-1], '-');
        END IF;
    END IF;

    RETURN username;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Reverse name order for matching (e.g., 'john-doe' -> 'doe-john')
CREATE OR REPLACE FUNCTION reverse_linkedin_username(username TEXT)
RETURNS TEXT AS $$
DECLARE
    parts TEXT[];
BEGIN
    IF username IS NULL OR username = '' THEN
        RETURN NULL;
    END IF;

    -- Only works for two-part names
    parts := string_to_array(username, '-');
    IF array_length(parts, 1) = 2 THEN
        RETURN parts[2] || '-' || parts[1];
    END IF;

    RETURN NULL;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Remove hyphens from username for fuzzy matching (e.g., 'john-doe' -> 'johndoe')
CREATE OR REPLACE FUNCTION hyphen_free_username(username TEXT)
RETURNS TEXT AS $$
BEGIN
    IF username IS NULL OR username = '' THEN
        RETURN NULL;
    END IF;

    RETURN replace(username, '-', '');
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- ============================================================================
-- MAIN MATCHING FUNCTION
-- ============================================================================

-- Match profiles by URLs with fuzzy matching (handles ID suffixes, name reversals, etc.)
-- Returns profiles WITHOUT raw_data to save bandwidth
CREATE OR REPLACE FUNCTION match_profiles_by_urls(
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
    reversed TEXT;
    hyphen_free TEXT;
BEGIN
    -- Build comprehensive set of URL variations from input
    FOREACH u IN ARRAY input_urls LOOP
        -- Skip nulls and empty strings
        IF u IS NULL OR u = '' THEN
            CONTINUE;
        END IF;

        -- Extract username
        username := extract_linkedin_username(u);
        IF username IS NULL THEN
            CONTINUE;
        END IF;

        -- Add base username
        url_variations := array_append(url_variations, username);

        -- Add username with ID suffix stripped
        base_username := strip_linkedin_id_suffix(username);
        IF base_username IS NOT NULL AND base_username != username THEN
            url_variations := array_append(url_variations, base_username);
        END IF;

        -- Add reversed name version (for both original and base)
        reversed := reverse_linkedin_username(username);
        IF reversed IS NOT NULL THEN
            url_variations := array_append(url_variations, reversed);
        END IF;

        IF base_username IS NOT NULL THEN
            reversed := reverse_linkedin_username(base_username);
            IF reversed IS NOT NULL THEN
                url_variations := array_append(url_variations, reversed);
            END IF;
        END IF;

        -- Add hyphen-free versions (catches 'johndoe' vs 'john-doe')
        hyphen_free := hyphen_free_username(username);
        IF hyphen_free IS NOT NULL AND hyphen_free != username THEN
            url_variations := array_append(url_variations, hyphen_free);
        END IF;

        IF base_username IS NOT NULL THEN
            hyphen_free := hyphen_free_username(base_username);
            IF hyphen_free IS NOT NULL AND hyphen_free != base_username THEN
                url_variations := array_append(url_variations, hyphen_free);
            END IF;
        END IF;
    END LOOP;

    -- Remove duplicates
    SELECT array_agg(DISTINCT v) INTO url_variations FROM unnest(url_variations) v;

    -- Match against profiles (linkedin_url and original_url columns)
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
        -- Match by extracting username from linkedin_url
        extract_linkedin_username(p.linkedin_url) = ANY(url_variations)
        OR strip_linkedin_id_suffix(extract_linkedin_username(p.linkedin_url)) = ANY(url_variations)
        OR hyphen_free_username(extract_linkedin_username(p.linkedin_url)) = ANY(url_variations)
        -- Match by extracting username from original_url
        OR extract_linkedin_username(p.original_url) = ANY(url_variations)
        OR strip_linkedin_id_suffix(extract_linkedin_username(p.original_url)) = ANY(url_variations)
        OR hyphen_free_username(extract_linkedin_username(p.original_url)) = ANY(url_variations)
    )
    -- Optional date filter
    AND (enriched_after IS NULL OR p.enriched_at >= enriched_after)
    ORDER BY p.linkedin_url, p.enriched_at DESC;

END;
$$ LANGUAGE plpgsql STABLE;


-- ============================================================================
-- INDEXES
-- ============================================================================

-- Index on original_url for faster lookups (linkedin_url is already indexed)
CREATE INDEX IF NOT EXISTS idx_profiles_original_url ON profiles(original_url);


-- ============================================================================
-- PERMISSIONS
-- ============================================================================

-- Grant execute to anon role (for Streamlit app)
GRANT EXECUTE ON FUNCTION extract_linkedin_username TO anon;
GRANT EXECUTE ON FUNCTION strip_linkedin_id_suffix TO anon;
GRANT EXECUTE ON FUNCTION reverse_linkedin_username TO anon;
GRANT EXECUTE ON FUNCTION hyphen_free_username TO anon;
GRANT EXECUTE ON FUNCTION match_profiles_by_urls TO anon;
