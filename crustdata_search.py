"""
Crustdata People Search Database API Client

This module provides functions to search Crustdata's 100M+ professional database.
Used by the Search tab in dashboard.py to find candidates before enrichment.

API Endpoint: POST https://api.crustdata.com/screener/persondb/search
Cost: 3 credits per 100 results

Usage:
    from crustdata_search import search_people_db, build_filters, normalize_search_results_to_df

    # Build filters from UI inputs
    filters = build_filters(
        title="Backend Engineer",
        company="Google",
        location="Israel",
        experience_min=3,
        experience_max=10
    )

    # Search
    results = search_people_db(filters, limit=100)

    # Convert to DataFrame for pipeline
    df = normalize_search_results_to_df(results['profiles'])
"""

import json
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any

from api_helpers import get_rate_limiter, RateLimitExceeded
from error_handling import (
    retry_with_backoff,
    ExternalServiceError,
    RateLimitError,
    AuthenticationError,
    ServiceUnavailableError,
    classify_http_error,
)
from normalizers import normalize_linkedin_url, clean_value, is_nan_or_none, pick_current_employer


# =============================================================================
# CONSTANTS
# =============================================================================

CRUSTDATA_SEARCH_ENDPOINT = "https://api.crustdata.com/screener/persondb/search"
CRUSTDATA_CREDITS_ENDPOINT = "https://api.crustdata.com/account/credits"

# Natural-language ("semantic") people search — Crustdata's newer v2025-11-01
# API. Same /person/search dataset as the filter search above, but ranks
# people by how well their whole profile matches a plain-language query
# instead of exact filter conditions. Verified against the live Crustdata
# docs (docs.crustdata.com/person-docs/search/introduction) on 2026-07-20 —
# request body is {"search": {"query", "mode"}, "mode": "exact"|"managed",
# "limit"}, auth is "Authorization: Bearer <key>" + "x-api-version" header
# (both different from the legacy endpoint above, which still uses "Token").
CRUSTDATA_SEMANTIC_SEARCH_ENDPOINT = "https://api.crustdata.com/person/search"
CRUSTDATA_API_VERSION = "2025-11-01"
CREDITS_PER_RESULT_SEMANTIC = 0.03

# Seniority levels supported by Crustdata.
# Canonical values verified via crustdata_autocomplete_person on 2026-05-17.
# Previously this list used "Manager", "Senior", "Entry", "Training" which
# Crustdata silently does not recognize, so those filters returned zero matches.
SENIORITY_LEVELS = [
    "Entry Level",
    "Entry Level Manager",
    "Senior",
    "Experienced Manager",
    "Director",
    "Vice President",
    "CXO",
    "Owner / Partner",
    "In Training",
    "Strategic",
]

# Company headcount ranges supported by Crustdata.
# Note the comma in "10,001+" — Crustdata writes the top bucket that way and
# the value sent in filters must match exactly. Previously this list used
# "10001+" (no comma), so the largest-company filter silently missed matches.
HEADCOUNT_RANGES = [
    "1-10",
    "11-50",
    "51-200",
    "201-500",
    "501-1000",
    "1001-5000",
    "5001-10000",
    "10,001+",
]

# Job function categories (current_employers.function_category).
# Values verified via crustdata_autocomplete_person on 2026-05-17.
FUNCTION_CATEGORIES = [
    "Engineering",
    "Product Management",
    "Sales",
    "Marketing",
    "Operations",
    "Finance",
    "Consulting",
    "Human Resources",
    "Research",
    "Legal",
    "Customer Success and Support",
    "Arts and Design",
]

# Industry values for current_employers.company_industries (curated top values).
# Verified via crustdata_autocomplete_person on 2026-05-17.
COMPANY_INDUSTRIES = [
    "Software Development",
    "Technology, Information and Internet",
    "Technology, Information and Media",
    "IT Services and IT Consulting",
    "Financial Services",
    "Capital Markets",
    "Business Consulting and Services",
    "Professional Services",
    "Manufacturing",
    "Hospitals and Health Care",
    "Retail",
    "Education",
    "Real Estate",
    "Advertising Services",
    "Marketing Services",
    "Media and Telecommunications",
    "Government Administration",
    "Non-profit Organizations",
    "Construction",
    "Transportation, Logistics, Supply Chain and Storage",
    "Legal Services",
    "Accounting",
    "Architecture and Planning",
    "Design Services",
    "Consumer Services",
]

# Credits per 100 results
CREDITS_PER_100_RESULTS = 3


# =============================================================================
# CONFIG LOADING
# =============================================================================

def _load_api_key() -> str:
    """Load Crustdata API key from config.json or environment.

    The returned key is .strip()ed so a stray trailing newline (common when a
    secret is set via ``gh secret set --body``) doesn't poison the HTTP header
    layer with errors like "Invalid leading whitespace, reserved character(s),
    or return character(s) in header value".
    """
    config_path = Path(__file__).parent / 'config.json'

    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_key = config.get('api_key')
                if api_key and api_key != "YOUR_CRUSTDATA_API_KEY_HERE":
                    return api_key.strip()
        except (json.JSONDecodeError, IOError):
            pass

    # Fallback to environment variable
    import os
    api_key = os.environ.get('CRUSTDATA_API_KEY')
    if api_key:
        return api_key.strip()

    raise AuthenticationError(
        "Crustdata",
        message="No Crustdata API key found. Set in config.json or CRUSTDATA_API_KEY env var."
    )


# =============================================================================
# FILTER BUILDERS
# =============================================================================

def _parse_keyword_boolean(expr: str) -> List[List[str]]:
    """
    Parse boolean keyword expression into AND groups of OR terms.

    Syntax:
        (node OR node.js) AND (react OR react.js)
        node OR node.js  (single OR group)
        node, node.js    (comma = OR for simple cases)
        kubernetes       (single term)

    Returns:
        List of groups, where each group is a list of OR terms.
        Groups are combined with AND logic.
        E.g., [["node", "node.js"], ["react", "react.js"]]
    """
    import re

    if not expr or not expr.strip():
        return []

    expr = expr.strip()

    # Check if it contains AND keyword (case insensitive)
    if re.search(r'\bAND\b', expr, re.IGNORECASE):
        # Split by AND (case insensitive, word boundary)
        and_parts = re.split(r'\s+AND\s+', expr, flags=re.IGNORECASE)
        groups = []
        for part in and_parts:
            part = part.strip()
            # Remove surrounding parentheses if present
            if part.startswith('(') and part.endswith(')'):
                part = part[1:-1].strip()
            # Split by OR (case insensitive) or comma
            if re.search(r'\bOR\b', part, re.IGNORECASE):
                or_terms = re.split(r'\s+OR\s+', part, flags=re.IGNORECASE)
            else:
                or_terms = part.split(',')
            # Clean up terms
            or_terms = [t.strip() for t in or_terms if t.strip()]
            if or_terms:
                groups.append(or_terms)
        return groups

    # No AND - check for OR
    if re.search(r'\bOR\b', expr, re.IGNORECASE):
        # Remove surrounding parentheses if present
        if expr.startswith('(') and expr.endswith(')'):
            expr = expr[1:-1].strip()
        or_terms = re.split(r'\s+OR\s+', expr, flags=re.IGNORECASE)
        or_terms = [t.strip() for t in or_terms if t.strip()]
        return [or_terms] if or_terms else []

    # No AND/OR - treat commas as OR (backwards compatibility)
    if ',' in expr:
        terms = [t.strip() for t in expr.split(',') if t.strip()]
        return [terms] if terms else []

    # Single term
    return [[expr.strip()]]


def build_filters(
    title: str = None,
    company: str = None,
    location: str = None,
    seniority: List[str] = None,
    headcount: List[str] = None,
    experience_min: int = None,
    experience_max: int = None,
    skills: List[str] = None,
    skills_and: bool = False,
    skill_groups: List[str] = None,
    keywords: str = None,
    past_companies: str = None,
    past_titles: str = None,
    school: str = None,
    recently_changed_jobs: bool = None,
    has_verified_email: bool = None,
    function_categories: List[str] = None,
    industries: List[str] = None,
    country: str = None,
    continent: str = None,
    geo_city: str = None,
    geo_radius_km: int = None,
    min_connections: int = None,
    exact_company: bool = False,
    not_relevant_companies: List[str] = None,
    blacklist_companies: List[str] = None,
) -> Dict[str, Any]:
    """
    Build Crustdata filter object from UI inputs.

    All conditions are combined with AND logic.
    Keywords search across headline + summary + skills with OR logic.

    Filter operators:
        [.] = substring match (contains)
        (.) = fuzzy match
        in / not_in = set membership (value must be array)
        >, <, >=, <= = numeric comparison

    Args:
        title: Job title (substring match). Comma-separated for OR logic.
        company: Company name (substring match). Comma-separated for OR logic.
        location: Location/region (substring match)
        seniority: List of seniority levels (in operator)
        headcount: List of headcount ranges (in operator)
        experience_min: Minimum years of experience (>= operator)
        experience_max: Maximum years of experience (<= operator)
        skills: List of skills (legacy, use skill_groups instead)
        skills_and: If True, require ALL skills (AND). If False, require ANY skill (OR)
        skill_groups: List of comma-separated skill strings. Each group is OR, groups combined with AND.
                      E.g., ["aws, gcp", "docker, kubernetes"] means (AWS OR GCP) AND (Docker OR Kubernetes)
        keywords: Boolean keyword expression. E.g., "(node OR node.js) AND (react OR react.js)"
                      Supports AND, OR keywords (case insensitive) and parentheses. Comma = OR for simple cases.
        past_companies: Comma-separated past company names (substring match)
        past_titles: Comma-separated past job titles (substring match)
        school: School/university name (substring match)
        recently_changed_jobs: If True, filter for job changes in last 90 days
        has_verified_email: If True, filter for verified business email

    Returns:
        Filter dict ready for Crustdata API: {"op": "and", "conditions": [...]}
    """
    conditions = []

    # Title filter (substring match on current job title OR headline)
    # Supports comma-separated values for OR logic.
    #
    # Why OR across two columns:
    # Crustdata persondb evaluates AND across nested-array columns
    # (current_employers.*) per-array-element. That makes
    # title=X AND company=Y collapse to "must be in the SAME current
    # employment entry" — see PR #39 / docs/research/crustdata-filter-semantics.md
    # for the per-element AND collapse and live probe transcript.
    #
    # Mirroring each title onto `headline` (a top-level profile field
    # evaluated per-profile, not per-array-element) keeps everything the
    # previous shape already matched and adds all profiles whose headline
    # mentions the title — measured lift on the PR #28 repro:
    # title="software developer" + 4 companies + region=Israel: 5 -> 12-149.
    if title and title.strip():
        title_values = [t.strip() for t in title.split(",") if t.strip()]
        if title_values:
            title_conditions = []
            for t in title_values:
                title_conditions.append(
                    {"column": "current_employers.title", "type": "[.]", "value": t}
                )
                title_conditions.append(
                    {"column": "headline", "type": "[.]", "value": t}
                )
            conditions.append({
                "op": "or",
                "conditions": title_conditions,
            })

    # Current company filter (comma-separated for OR logic)
    if company and company.strip():
        company_values = [c.strip() for c in company.split(",") if c.strip()]
        match_type = "=" if exact_company else "[.]"
        if len(company_values) == 1:
            conditions.append({
                "column": "current_employers.name",
                "type": match_type,
                "value": company_values[0]
            })
        elif len(company_values) > 1:
            company_conditions = [
                {"column": "current_employers.name", "type": match_type, "value": c}
                for c in company_values
            ]
            conditions.append({
                "op": "or",
                "conditions": company_conditions
            })

    # Location filter (comma-separated for OR logic)
    if location and location.strip():
        location_values = [l.strip() for l in location.split(",") if l.strip()]
        if len(location_values) == 1:
            conditions.append({
                "column": "region",
                "type": "[.]",
                "value": location_values[0]
            })
        elif len(location_values) > 1:
            location_conditions = [
                {"column": "region", "type": "[.]", "value": l}
                for l in location_values
            ]
            conditions.append({
                "op": "or",
                "conditions": location_conditions
            })

    # Seniority filter (set membership)
    if seniority and len(seniority) > 0:
        # Validate seniority levels
        valid_seniority = [s for s in seniority if s in SENIORITY_LEVELS]
        if valid_seniority:
            conditions.append({
                "column": "current_employers.seniority_level",
                "type": "in",
                "value": valid_seniority
            })

    # Headcount filter (set membership)
    if headcount and len(headcount) > 0:
        # Validate headcount ranges
        valid_headcount = [h for h in headcount if h in HEADCOUNT_RANGES]
        if valid_headcount:
            conditions.append({
                "column": "current_employers.company_headcount_range",
                "type": "in",
                "value": valid_headcount
            })

    # Experience range filters
    # Note: API doesn't support >= or <=, so we use > and < with adjusted values
    if experience_min is not None and experience_min > 0:
        conditions.append({
            "column": "years_of_experience_raw",
            "type": ">",
            "value": experience_min - 1  # ">= 3" becomes "> 2"
        })

    if experience_max is not None and experience_max > 0:
        conditions.append({
            "column": "years_of_experience_raw",
            "type": "<",
            "value": experience_max + 1  # "<= 10" becomes "< 11"
        })

    # Skill groups filter (each group is OR, groups combined with AND)
    # E.g., ["aws, gcp", "docker, kubernetes"] means (AWS OR GCP) AND (Docker OR Kubernetes)
    if skill_groups and len(skill_groups) > 0:
        for group in skill_groups:
            if not group or not group.strip():
                continue
            group_skills = [s.strip() for s in group.split(',') if s.strip()]
            if len(group_skills) == 1:
                # Single skill in group - simple condition
                conditions.append({
                    "column": "skills",
                    "type": "[.]",
                    "value": group_skills[0]
                })
            elif len(group_skills) > 1:
                # Multiple skills in group - OR condition
                group_conditions = [
                    {"column": "skills", "type": "[.]", "value": s}
                    for s in group_skills
                ]
                conditions.append({
                    "op": "or",
                    "conditions": group_conditions
                })

    # Legacy skills filter (AND or OR based on skills_and flag) - for backwards compatibility
    elif skills and len(skills) > 0:
        skill_values = [s.strip() if isinstance(s, str) else str(s) for s in skills]
        skill_values = [s for s in skill_values if s]
        if len(skill_values) == 1:
            conditions.append({
                "column": "skills",
                "type": "[.]",
                "value": skill_values[0]
            })
        elif len(skill_values) > 1:
            skill_conditions = [
                {"column": "skills", "type": "[.]", "value": s}
                for s in skill_values
            ]
            if skills_and:
                # AND mode: add each skill as separate condition (all must match)
                conditions.extend(skill_conditions)
            else:
                # OR mode: wrap in OR condition (any must match)
                conditions.append({
                    "op": "or",
                    "conditions": skill_conditions
                })

    # Keywords filter (searches headline, summary, skills)
    # Boolean syntax: (node OR node.js) AND (react OR react.js)
    # Also supports: node, node.js (comma = OR for simple cases)
    if keywords and keywords.strip():
        # Parse boolean expression
        keyword_groups = _parse_keyword_boolean(keywords)

        for group in keyword_groups:
            if group:
                # Build OR condition for keywords in this group
                keyword_conditions = []
                for kw in group:
                    # Search in headline
                    keyword_conditions.append({
                        "column": "headline",
                        "type": "[.]",
                        "value": kw
                    })
                    # Search in summary
                    keyword_conditions.append({
                        "column": "summary",
                        "type": "[.]",
                        "value": kw
                    })
                    # Search in skills
                    keyword_conditions.append({
                        "column": "skills",
                        "type": "[.]",
                        "value": kw
                    })

                # Wrap group in OR condition, add to main conditions (AND)
                conditions.append({
                    "op": "or",
                    "conditions": keyword_conditions
                })

    # Past companies filter
    if past_companies and past_companies.strip():
        company_list = [c.strip() for c in past_companies.split(',') if c.strip()]
        if company_list:
            # OR condition for past companies
            past_company_conditions = []
            for pc in company_list:
                past_company_conditions.append({
                    "column": "past_employers.name",
                    "type": "[.]",
                    "value": pc
                })
            conditions.append({
                "op": "or",
                "conditions": past_company_conditions
            })

    # Past titles filter
    if past_titles and past_titles.strip():
        title_list = [t.strip() for t in past_titles.split(',') if t.strip()]
        if title_list:
            # OR condition for past titles
            past_title_conditions = []
            for pt in title_list:
                past_title_conditions.append({
                    "column": "past_employers.title",
                    "type": "[.]",
                    "value": pt
                })
            conditions.append({
                "op": "or",
                "conditions": past_title_conditions
            })

    # School filter (comma-separated for OR logic)
    if school and school.strip():
        school_values = [s.strip() for s in school.split(",") if s.strip()]
        if len(school_values) == 1:
            conditions.append({
                "column": "education_background.institute_name",
                "type": "[.]",
                "value": school_values[0]
            })
        elif len(school_values) > 1:
            school_conditions = [
                {"column": "education_background.institute_name", "type": "[.]", "value": s}
                for s in school_values
            ]
            conditions.append({
                "op": "or",
                "conditions": school_conditions
            })

    # Recently changed jobs filter
    if recently_changed_jobs:
        conditions.append({
            "column": "recently_changed_jobs",
            "type": "=",
            "value": True
        })

    # Has verified email filter
    if has_verified_email:
        conditions.append({
            "column": "current_employers.business_email_verified",
            "type": "=",
            "value": True
        })

    # Function category filter (set membership on current job function)
    if function_categories:
        valid_functions = [f for f in function_categories if f in FUNCTION_CATEGORIES]
        if valid_functions:
            conditions.append({
                "column": "current_employers.function_category",
                "type": "in",
                "value": valid_functions
            })

    # Industry filter (set membership on company industries)
    if industries:
        conditions.append({
            "column": "current_employers.company_industries",
            "type": "in",
            "value": list(industries)
        })

    # Country filter (exact match — values are case-sensitive)
    if country and country.strip():
        conditions.append({
            "column": "location_country",
            "type": "=",
            "value": country.strip()
        })

    # Continent filter (exact match)
    if continent and continent.strip():
        conditions.append({
            "column": "location_continent",
            "type": "=",
            "value": continent.strip()
        })

    # Geo radius filter ("within N km of CITY")
    if geo_city and geo_city.strip() and geo_radius_km and geo_radius_km > 0:
        conditions.append({
            "column": "region",
            "type": "geo_distance",
            "value": {"location": geo_city.strip(), "distance": geo_radius_km, "unit": "km"}
        })

    # Min connections filter (=> is Crustdata's "greater than or equal to" operator)
    if min_connections and min_connections > 0:
        conditions.append({
            "column": "num_of_connections",
            "type": "=>",
            "value": min_connections
        })

    # Exclude not-relevant and blacklisted companies (current employer only).
    # Both lists are merged into a single not_in to avoid ambiguity if the API
    # treats two conditions on the same column with OR rather than AND semantics.
    # not_in is exact/case-sensitive; the post-search fuzzy filter handles variants.
    _excl_set = set()
    if not_relevant_companies:
        _excl_set.update(n.strip().strip('"').strip() for n in not_relevant_companies if n and n.strip())
    if blacklist_companies:
        _excl_set.update(n.strip().strip('"').strip() for n in blacklist_companies if n and n.strip())
    if _excl_set:
        conditions.append({
            "column": "current_employers.name",
            "type": "not_in",
            "value": sorted(_excl_set)
        })

    # Return combined filter
    if not conditions:
        # Return empty filter that matches everything
        return {}

    if len(conditions) == 1:
        # Single condition doesn't need wrapper
        return {"filters": conditions[0]}

    # Multiple conditions with AND
    return {
        "filters": {
            "op": "and",
            "conditions": conditions
        }
    }


# =============================================================================
# API FUNCTIONS
# =============================================================================

@retry_with_backoff(
    max_retries=3,
    base_delay=2.0,
    retryable_exceptions=(RateLimitError, ServiceUnavailableError, ConnectionError, TimeoutError),
)
def search_people_db(
    filters: Dict[str, Any],
    limit: int = 100,
    cursor: str = None,
    sorts: List[Dict[str, str]] = None,
    api_key: str = None,
    exclude_profiles: List[str] = None,
) -> Dict[str, Any]:
    """
    Search Crustdata's people database.

    Args:
        filters: Filter dict from build_filters() or raw filter object
        limit: Results per page (max 1000, default 100)
        cursor: Pagination cursor from previous response
        sorts: Optional sorting list, e.g., [{"column": "years_of_experience_raw", "order": "desc"}]
        api_key: Optional API key (if not provided, loads from config.json or env var)

    Returns:
        {
            "profiles": [...],       # List of profile dicts
            "cursor": "...",         # Next page cursor (None if no more results)
            "total_count": N,        # Total matching profiles
            "credits_used": N        # Credits consumed
        }

    Raises:
        AuthenticationError: Invalid API key
        RateLimitError: Rate limit exceeded
        ExternalServiceError: API error
    """
    if not api_key:
        api_key = _load_api_key()
    limiter = get_rate_limiter('crustdata')

    # Build request body.
    # compact=false is the load-bearing parameter that makes search return the
    # FULL profile (past_employers with descriptions, certifications, summary,
    # flagship_profile_url, etc.). Crustdata's default is compact=true which
    # silently strips nested data — that's what created the original (incorrect)
    # assumption that enrichment was needed after every search.
    # Verified via Crustdata founder call + live test against Ami Blonder
    # on 2026-05-15. See .planning equivalent docs / GitHub issue #68.
    body = {
        "limit": min(limit, 1000),  # Cap at 1000
        "compact": False,
    }

    # Add filters if provided
    if filters:
        if "filters" in filters:
            body["filters"] = filters["filters"]
        elif "op" in filters or "column" in filters:
            body["filters"] = filters

    # Add pagination cursor
    if cursor:
        body["cursor"] = cursor

    # Exclude specific LinkedIn profiles (past candidates).
    # Crustdata nests this under post_processing, not at the top level.
    # Normalize URLs to canonical form so scheme/www/trailing-slash variants match.
    if exclude_profiles:
        clean_urls = [normalize_linkedin_url(u) for u in exclude_profiles if u and str(u).strip()]
        clean_urls = [u for u in clean_urls if u]
        if clean_urls:
            body["post_processing"] = {"exclude_profiles": clean_urls}

    # Add sorting
    if sorts:
        body["sorts"] = sorts

    # Rate limiting
    try:
        limiter.wait_if_needed()
    except RateLimitExceeded as e:
        raise RateLimitError("Crustdata", message=str(e))

    start_time = time.time()

    try:
        response = requests.post(
            CRUSTDATA_SEARCH_ENDPOINT,
            json=body,
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )

        limiter.record_request()

        # Handle HTTP errors
        if response.status_code == 401:
            raise AuthenticationError("Crustdata")
        elif response.status_code == 429:
            retry_after = response.headers.get('Retry-After')
            raise RateLimitError(
                "Crustdata",
                retry_after=float(retry_after) if retry_after else None
            )
        elif response.status_code >= 500:
            raise ServiceUnavailableError(
                "Crustdata",
                status_code=response.status_code,
                response_body=response.text[:500]
            )
        elif response.status_code >= 400:
            raise ExternalServiceError(
                "Crustdata",
                message=f"Search failed: {response.text[:500]}",
                status_code=response.status_code,
                response_body=response.text
            )

        data = response.json()

        # Extract results - API returns "profiles" not "data", "next_cursor" not "cursor"
        profiles = data.get("profiles", [])
        next_cursor = data.get("next_cursor")
        total_count = data.get("total_count", len(profiles))

        # Calculate credits used (3 credits per 100 results)
        credits_used = (len(profiles) // 100 + (1 if len(profiles) % 100 > 0 else 0)) * CREDITS_PER_100_RESULTS

        return {
            "profiles": profiles,
            "cursor": next_cursor,
            "total_count": total_count,
            "credits_used": credits_used,
            "response_time_ms": int((time.time() - start_time) * 1000),
        }

    except requests.exceptions.Timeout:
        raise ExternalServiceError(
            "Crustdata",
            message="Search request timed out",
            status_code=504
        )
    except requests.exceptions.ConnectionError as e:
        raise ServiceUnavailableError(
            "Crustdata",
            message=f"Connection error: {str(e)[:200]}"
        )


@retry_with_backoff(
    max_retries=2,
    base_delay=1.0,
    retryable_exceptions=(RateLimitError, ServiceUnavailableError, ConnectionError),
)
def check_credits(api_key: str = None) -> Dict[str, Any]:
    """
    Check remaining Crustdata credits.

    Args:
        api_key: Optional API key (if not provided, loads from config.json or env var)

    Returns:
        {
            "remaining": N,      # Credits remaining
            "used": N,           # Credits used (if available)
            "total": N           # Total credits (if available)
        }

    Raises:
        AuthenticationError: Invalid API key
        ExternalServiceError: API error
    """
    if not api_key:
        api_key = _load_api_key()
    limiter = get_rate_limiter('crustdata')

    try:
        limiter.wait_if_needed()
    except RateLimitExceeded as e:
        raise RateLimitError("Crustdata", message=str(e))

    try:
        response = requests.get(
            CRUSTDATA_CREDITS_ENDPOINT,
            headers={"Authorization": f"Token {api_key}"},
            timeout=30,
        )

        limiter.record_request()

        if response.status_code == 401:
            raise AuthenticationError("Crustdata")
        elif response.status_code >= 400:
            raise ExternalServiceError(
                "Crustdata",
                message=f"Credits check failed: {response.text[:200]}",
                status_code=response.status_code
            )

        data = response.json()

        return {
            "remaining": data.get("credits_remaining", data.get("remaining", 0)),
            "used": data.get("credits_used", data.get("used", 0)),
            "total": data.get("credits_total", data.get("total", 0)),
        }

    except requests.exceptions.Timeout:
        raise ExternalServiceError(
            "Crustdata",
            message="Credits check timed out",
            status_code=504
        )
    except requests.exceptions.ConnectionError as e:
        raise ServiceUnavailableError(
            "Crustdata",
            message=f"Connection error: {str(e)[:200]}"
        )


@retry_with_backoff(
    max_retries=3,
    base_delay=2.0,
    retryable_exceptions=(RateLimitError, ServiceUnavailableError, ConnectionError, TimeoutError),
)
def search_people_semantic(
    query: str,
    limit: int = 20,
    cursor: str = None,
    search_mode: str = "hybrid",
    recall_mode: str = "managed",
    api_key: str = None,
) -> Dict[str, Any]:
    """
    Natural-language ("search by description") people search — beta.

    Instead of building filter conditions, pass a plain-language description
    of who you're looking for (a role, a persona, or a pasted JD) and get
    back people ranked by how well their whole profile matches it. Each
    result carries a "fit" tier (strong/possible/weak) in the raw profile —
    read it to judge quality; total_count is the size of the ranked pool,
    not a count of good matches.

    Args:
        query: Plain-language description, e.g. "founding engineers at
            developer-tools startups in Israel".
        limit: Results to return (1-100, default 20).
        cursor: Pagination cursor from a previous response.
        search_mode: "hybrid" (default, keyword+vector), "lexical" (exact
            terms only), or "semantic" (vector/meaning only).
        recall_mode: "managed" (default — query is the main signal) or
            "exact" (only used if filters are added later; kept here so
            callers can opt in without a signature change).
        api_key: Optional API key (loads from config.json / env var if omitted).

    Returns:
        {
            "profiles": [...],   # raw nested v2 profile dicts (see
                                  # semantic_profile_to_legacy_shape() to adapt
                                  # them for the rest of the pipeline)
            "cursor": "...",
            "total_count": N,
            "credits_used": N,   # 0.03 credits per result returned
            "response_time_ms": N,
        }
    """
    if not query or not query.strip():
        raise ValueError("query is required for semantic search")

    if not api_key:
        api_key = _load_api_key()
    limiter = get_rate_limiter('crustdata')

    body = {
        "search": {"query": query.strip(), "mode": search_mode},
        "limit": max(1, min(limit, 100)),
    }
    if recall_mode == "exact":
        body["mode"] = "exact"
    if cursor:
        body["cursor"] = cursor

    try:
        limiter.wait_if_needed()
    except RateLimitExceeded as e:
        raise RateLimitError("Crustdata", message=str(e))

    start_time = time.time()

    try:
        response = requests.post(
            CRUSTDATA_SEMANTIC_SEARCH_ENDPOINT,
            json=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "x-api-version": CRUSTDATA_API_VERSION,
            },
            timeout=60,
        )

        limiter.record_request()

        if response.status_code == 401:
            raise AuthenticationError("Crustdata")
        elif response.status_code == 429:
            retry_after = response.headers.get('Retry-After')
            raise RateLimitError(
                "Crustdata",
                retry_after=float(retry_after) if retry_after else None
            )
        elif response.status_code >= 500:
            raise ServiceUnavailableError(
                "Crustdata",
                status_code=response.status_code,
                response_body=response.text[:500]
            )
        elif response.status_code >= 400:
            raise ExternalServiceError(
                "Crustdata",
                message=f"Semantic search failed: {response.text[:500]}",
                status_code=response.status_code,
                response_body=response.text
            )

        data = response.json()
        profiles = data.get("profiles", [])
        next_cursor = data.get("next_cursor")
        total_count = data.get("total_count", len(profiles))
        credits_used = round(len(profiles) * CREDITS_PER_RESULT_SEMANTIC, 2)

        return {
            "profiles": profiles,
            "cursor": next_cursor,
            "total_count": total_count,
            "credits_used": credits_used,
            "response_time_ms": int((time.time() - start_time) * 1000),
        }

    except requests.exceptions.Timeout:
        raise ExternalServiceError(
            "Crustdata",
            message="Semantic search request timed out",
            status_code=504
        )
    except requests.exceptions.ConnectionError as e:
        raise ServiceUnavailableError(
            "Crustdata",
            message=f"Connection error: {str(e)[:200]}"
        )


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_search_result(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a Crustdata search result to pipeline DataFrame format.

    With ``compact=false`` (now the default in ``search_people_db``), Crustdata
    search returns the FULL profile: flagship_profile_url, past_employers with
    full descriptions and dates, education, certifications, summary, skills,
    languages, etc. The only field NOT returned by search is ``emails`` (SourcingX
    uses SalesQL for emails separately). A follow-up enrichment call is no
    longer needed in the default pipeline — see GitHub issue #68 / the
    Crustdata-founder call notes.

    Args:
        profile: Raw profile dict from search results

    Returns:
        Normalized dict ready for pipeline DataFrame with fields:
            - linkedin_url
            - name, first_name, last_name
            - headline, location, summary
            - current_company, current_title, seniority, company_size
            - skills (comma-separated string)
            - years_experience
            - _source = 'crustdata_search'
            - _needs_enrichment = False  (search response is complete with compact=false)
    """
    if not profile:
        return None

    # Extract LinkedIn URL - use flagship (clean URL) not linkedin_profile_url (URN format)
    linkedin_url = (
        profile.get('flagship_profile_url') or
        profile.get('linkedin_flagship_url') or
        profile.get('linkedin_profile_url') or
        profile.get('linkedin_url')
    )
    linkedin_url = normalize_linkedin_url(linkedin_url)

    if not linkedin_url:
        return None

    # Name fields
    full_name = profile.get('name', '')
    first_name = profile.get('first_name', '')
    last_name = profile.get('last_name', '')

    # Parse full name if first/last not available
    if not first_name and not last_name and full_name:
        parts = full_name.split(' ', 1)
        first_name = parts[0] if parts else ''
        last_name = parts[1] if len(parts) > 1 else ''

    # Build full name if not available
    if not full_name and (first_name or last_name):
        full_name = f"{first_name} {last_name}".strip()

    # Current employer info — pick the most recent when multiple are present
    current_company = ''
    current_title = ''
    seniority = ''
    company_size = ''
    current_years_in_role = None
    current_years_at_company = None
    all_current = profile.get('current_employers') or []
    emp = pick_current_employer(all_current)
    if emp:
        current_company = emp.get('employer_name') or emp.get('name') or ''
        current_title = emp.get('employee_title') or emp.get('title') or ''
        seniority = emp.get('seniority_level') or emp.get('seniority') or ''
        company_size = emp.get('company_headcount_range') or emp.get('headcount') or ''

        # Tenure in current ROLE (most recent entry's start_date)
        raw_start = emp.get('start_date')
        if raw_start:
            from normalizers import _parse_start_date_sort_key
            parseable, dt = _parse_start_date_sort_key(raw_start)
            if parseable:
                from datetime import datetime
                current_years_in_role = round((datetime.now() - dt).days / 365.25, 1)

        # Tenure at COMPANY — find earliest start_date for the same company
        current_years_at_company = current_years_in_role
        if current_company and isinstance(all_current, list) and len(all_current) > 1:
            from normalizers import _parse_start_date_sort_key as _psd
            from datetime import datetime as _dt
            company_lower = current_company.lower().strip()
            earliest = None
            for entry in all_current:
                if not isinstance(entry, dict):
                    continue
                name = (entry.get('employer_name') or entry.get('name') or '').lower().strip()
                if name != company_lower:
                    continue
                ok, d = _psd(entry.get('start_date'))
                if ok and (earliest is None or d < earliest):
                    earliest = d
            if earliest:
                current_years_at_company = round((_dt.now() - earliest).days / 365.25, 1)

    # Fallback to top-level fields
    if not current_title:
        current_title = profile.get('title', '')
    if not current_company:
        current_company = profile.get('company', '')

    # Skills
    skills = profile.get('skills', [])
    if isinstance(skills, list):
        skills_str = ', '.join(str(s) for s in skills[:50] if s)
    elif skills:
        skills_str = str(skills)
    else:
        skills_str = ''

    # All employers / titles / schools — handle both flat strings (enrich endpoint)
    # and objects (compact=false search endpoint, mirrors db._prepare_profile_row logic)
    def _extract_names_titles(raw):
        names, titles = [], []
        for item in (raw or []):
            if isinstance(item, dict):
                n = item.get('name') or item.get('employer_name') or ''
                t = item.get('title') or item.get('employee_title') or ''
                if n: names.append(n)
                if t: titles.append(t)
            elif isinstance(item, str) and item:
                names.append(item)
        return names, titles

    def _extract_schools(raw):
        schools = []
        for item in (raw or []):
            if isinstance(item, dict):
                s = item.get('institute_name') or item.get('school') or item.get('name') or ''
                if s: schools.append(s)
            elif isinstance(item, str) and item:
                schools.append(item)
        return schools

    _emp_names, _emp_titles = _extract_names_titles(profile.get('all_employers'))
    if not _emp_names:
        _fb_names, _fb_titles = _extract_names_titles(profile.get('past_employers'))
        _emp_names = _fb_names
        _emp_titles = _emp_titles or _fb_titles

    _raw_titles = profile.get('all_titles') or []
    _titles = [str(x) for x in _raw_titles if x] if _raw_titles else _emp_titles
    _schools = _extract_schools(profile.get('all_schools')) or _extract_schools(profile.get('education_background'))

    all_employers_str = ', '.join(_emp_names)
    all_titles_str = ', '.join(_titles)
    all_schools_str = ', '.join(_schools)

    # Connections
    connections_count = profile.get('num_of_connections') or profile.get('connections_count')

    # Experience
    years_exp = profile.get('years_of_experience_raw')
    if years_exp is None:
        years_exp = profile.get('years_of_experience')

    return {
        # Core identifiers (snake_case to match pipeline)
        'linkedin_url': linkedin_url,
        'name': clean_value(full_name) or '',
        'first_name': clean_value(first_name) or '',
        'last_name': clean_value(last_name) or '',
        # Profile content
        'headline': clean_value(profile.get('headline', '')) or '',
        'location': clean_value(profile.get('region') or profile.get('location', '')) or '',
        'summary': clean_value(profile.get('summary', '')) or '',
        # Current employment
        'current_company': clean_value(current_company) or '',
        'current_title': clean_value(current_title) or '',
        'seniority': clean_value(seniority) or '',
        'company_size': clean_value(company_size) or '',
        # Tenure
        'current_years_in_role': current_years_in_role,
        'current_years_at_company': current_years_at_company,
        # Skills and experience
        'skills': skills_str,
        'years_experience': years_exp,
        # Filter tab fields
        'all_employers': all_employers_str,
        'all_titles': all_titles_str,
        'all_schools': all_schools_str,
        'connections_count': connections_count,
        # Metadata
        '_source': 'crustdata_search',
        '_needs_enrichment': False,  # compact=false returns full profile
        '_raw_search_result': profile,  # Keep raw for debugging
    }


def semantic_profile_to_legacy_shape(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt one nested v2025-11-01 person-search profile (as returned by
    search_people_semantic) to the flat legacy profile shape that
    normalize_search_result() and the Search tab's results table already
    know how to read (current_employers/past_employers as lists of
    {name, title, ...}, top-level headline/region/summary, etc.).

    This lets semantic search results flow through the exact same
    display, selection, CSV export, and pipeline code as the regular
    filter search — no parallel code path needed.

    Verified response field paths against the live Crustdata docs on
    2026-07-20 (basic_profile.location.{raw,country}, experience
    .employment_details.{current,past}, social_handles
    .professional_network_identifier.profile_url, education.schools,
    skills.professional_network_skills, professional_network.connections).
    """
    if not profile:
        return {}

    basic = profile.get('basic_profile') or {}
    location = basic.get('location') or {}
    employment = (profile.get('experience') or {}).get('employment_details') or {}
    schools = (profile.get('education') or {}).get('schools') or []
    skills = (profile.get('skills') or {}).get('professional_network_skills') or []
    connections = (profile.get('professional_network') or {}).get('connections')
    social_id = (profile.get('social_handles') or {}).get('professional_network_identifier') or {}

    def _employer_entries(raw_entries):
        # v2 uses "company_headcount_latest"; the legacy code this profile
        # feeds into reads "company_headcount_range" — alias it across so
        # company size still shows up without touching that shared code.
        out = []
        for entry in (raw_entries or []):
            if not isinstance(entry, dict):
                continue
            entry = dict(entry)
            if 'company_headcount_range' not in entry and 'company_headcount_latest' in entry:
                entry['company_headcount_range'] = entry['company_headcount_latest']
            out.append(entry)
        return out

    return {
        'name': basic.get('name', ''),
        'headline': basic.get('headline', ''),
        'region': location.get('raw', ''),
        'location_country': location.get('country', ''),
        'summary': basic.get('summary', ''),
        'flagship_profile_url': social_id.get('profile_url'),
        'current_employers': _employer_entries(employment.get('current')),
        'past_employers': _employer_entries(employment.get('past')),
        'skills': skills,
        'num_of_connections': connections,
        'years_of_experience_raw': profile.get('years_of_experience_raw'),
        'all_schools': [
            s.get('school') for s in schools if isinstance(s, dict) and s.get('school')
        ],
        'crustdata_person_id': profile.get('crustdata_person_id'),
        # Crustdata's relevance tier for this result: strong/possible/weak.
        # Not part of the legacy shape — carried through as an extra field
        # so the results table can show it.
        '_fit': profile.get('fit', ''),
        # The shim above only maps the fields the legacy shape/pipeline knows
        # about — it drops some of the nested v2 response (contact flags,
        # normalized_title, professional_network handle, etc.). Keep the
        # original nested profile so a "full details" export can still show
        # everything Crustdata actually returned for this person, the same
        # way normalize_search_result() keeps _raw_search_result for the
        # regular filter search.
        '_raw_semantic_result': profile,
    }


def normalize_search_results_to_df(profiles: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Batch normalize search results and return DataFrame ready for pipeline.

    Args:
        profiles: List of raw profile dicts from search_people_db()

    Returns:
        pandas DataFrame with normalized columns matching pipeline format
    """
    if not profiles:
        return pd.DataFrame()

    normalized = []
    for profile in profiles:
        result = normalize_search_result(profile)
        if result:
            normalized.append(result)

    if not normalized:
        return pd.DataFrame()

    df = pd.DataFrame(normalized)

    # Ensure consistent column order (snake_case to match pipeline)
    column_order = [
        'linkedin_url',
        'name',
        'first_name',
        'last_name',
        'headline',
        'current_title',
        'current_company',
        'location',
        'seniority',
        'company_size',
        'years_experience',
        'skills',
        'summary',
        '_source',
        '_needs_enrichment',
    ]

    # Reorder columns (keep any extra columns at the end)
    existing_cols = [c for c in column_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in column_order]
    df = df[existing_cols + extra_cols]

    return df


# =============================================================================
# USAGE TRACKING HELPER
# =============================================================================

def log_search_usage(
    tracker,
    profiles_found: int,
    credits_used: int,
    status: str = 'success',
    error_message: str = None,
    response_time_ms: int = None,
) -> Optional[Dict]:
    """
    Log Crustdata search usage to the usage tracker.

    Args:
        tracker: UsageTracker instance
        profiles_found: Number of profiles returned
        credits_used: Credits consumed
        status: 'success' or 'error'
        error_message: Error details if status is 'error'
        response_time_ms: API response time

    Returns:
        Logged record or None
    """
    if not tracker:
        return None

    return tracker.log_usage(
        provider='crustdata',
        operation='search',
        request_count=1,
        credits_used=credits_used,
        cost_usd=credits_used * 0.01,  # $0.01 per credit
        status=status,
        error_message=error_message,
        response_time_ms=response_time_ms,
        metadata={'profiles_found': profiles_found}
    )


# =============================================================================
# AI-ASSISTED EXPANSION
# =============================================================================

_EXPANSION_PROMPTS = {
    'title': (
        'You are a senior tech recruiter searching LinkedIn. Given the job title "{term}", '
        'list 5-10 alternative titles that REAL PEOPLE actually use on their LinkedIn profiles '
        'for the same role.\n'
        'Rules:\n'
        '- Every title you suggest MUST be something you would realistically find on LinkedIn\n'
        '- Do NOT invent creative combinations by mixing the input word with random nouns '
        '(e.g., "team guide", "team overseer", "team architect", "team chief" are NOT real titles)\n'
        '- Focus on: how different companies name the same job, abbreviations, seniority prefixes\n'
        '- Do NOT include titles at a higher seniority (e.g., director, VP) unless the input is at that level\n'
        'Return ONLY a valid JSON array of lowercase strings.\n'
        'Example for "team leader": ["team lead", "tech lead", "engineering lead", "r&d team lead"]\n'
        'Example for "devops engineer": ["devops developer", "cloud engineer", '
        '"infrastructure engineer", "site reliability engineer", "sre", "platform engineer"]'
    ),
    'skill': (
        'Given the technical skill "{term}", list 5-10 alternative names, abbreviations, '
        'or very closely related tools that recruiters treat as interchangeable. '
        'Only include skills that someone searching for "{term}" would also want to match. '
        'Do NOT include loosely related or adjacent technologies. '
        'Return ONLY a valid JSON array of lowercase strings. '
        'Example for "kubernetes": ["k8s", "docker", "helm", "container orchestration", "openshift"]'
    ),
    'company': (
        'Given the description "{term}"{geo_clause}, list 5-10 specific company names that match. '
        'The input may be a category (e.g., "SaaS startups in Tel Aviv"), a single company '
        '(e.g., "Wiz" → suggest similar companies), or a concept (e.g., "big tech"). '
        'Return ONLY a valid JSON array of strings with proper company name capitalization. '
        'Example: ["Wiz", "Monday.com", "Gong", "Snyk", "Fireblocks"]'
    ),
    'location': (
        'Given the location "{term}", list 5-10 specific locations useful for a LinkedIn '
        'people search. Rules:\n'
        '- If input is a country (e.g., "Israel"), return the country name plus its major cities\n'
        '- If input is a region (e.g., "west coast usa"), return specific cities in that region\n'
        '- If input is a city, return nearby cities and the metro area name\n'
        '- Use proper city names as they appear on LinkedIn profiles (not airport codes)\n'
        '- Include the original input if it is already a valid location\n'
        'Return ONLY a valid JSON array of strings.\n'
        'Example for "west coast usa": ["San Francisco", "Los Angeles", "Seattle", "San Diego", "Portland"]'
    ),
    'school': (
        'Given the description "{term}"{geo_clause}, list 5-10 specific university or school names '
        'that match. The input may be a category (e.g., "top tech universities in London"), '
        'a country, or a single school (suggest similar ones). '
        'Return ONLY a valid JSON array of strings with proper capitalization. '
        'Example: ["Imperial College London", "UCL", "King\'s College London"]'
    ),
    'keywords': (
        'Given the keyword or concept "{term}", list 5-10 closely related keywords, '
        'technologies, or terms that recruiters would search for together. '
        'Return ONLY a valid JSON array of lowercase strings. '
        'Example: ["microservices", "kubernetes", "docker", "service mesh", "api gateway"]'
    ),
}


def expand_variations(
    term: str,
    field_type: str = 'title',
    openai_api_key: str = None,
    exclude: List[str] = None,
    geo_context: str = None,
) -> List[str]:
    """
    Use OpenAI gpt-4o-mini to expand a term into common variations.

    Args:
        term: The user's input (e.g., "team leader", "SaaS startups in Tel Aviv")
        field_type: 'title', 'skill', 'company', 'location', 'school', or 'keywords'
        openai_api_key: OpenAI API key
        exclude: List of already-suggested values to exclude from results
        geo_context: Optional geographic scope (e.g. "Israel", "Tel Aviv") read
            from the user's Location filter. Only injected into prompts for
            field types where region matters in practice — currently 'company'
            and 'school'. Has no effect for 'location' (would be circular) or
            for skill/keyword expansions (region-neutral).

    Returns:
        List of 5-10 variations (always includes the original term for title/skill types).
        On failure, returns [term] (graceful fallback).
    """
    if not term or not term.strip():
        return []

    term = term.strip()

    if not openai_api_key:
        return [term]

    # Only field types whose prompt template references {geo_clause} get the
    # geographic scope. Other prompts ignore the parameter.
    if geo_context and field_type in ('company', 'school'):
        geo_clause = (
            f' (only suggest options with significant presence in {geo_context.strip()} — '
            f'companies/schools with offices, engineering teams, or hiring in that region)'
        )
    else:
        geo_clause = ''

    prompt_template = _EXPANSION_PROMPTS.get(field_type, _EXPANSION_PROMPTS['title'])
    # Templates that don't reference {geo_clause} ignore it via .format kwargs.
    try:
        prompt = prompt_template.format(term=term, geo_clause=geo_clause)
    except KeyError:
        # Defensive fallback for any template that doesn't accept geo_clause.
        prompt = prompt_template.format(term=term)

    if exclude:
        prompt += f'\nDo NOT include any of these (already suggested): {json.dumps(exclude)}'

    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )

        text = response.choices[0].message.content.strip()
        # Handle markdown code fences
        if text.startswith('```'):
            text = text.split('\n', 1)[1] if '\n' in text else text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            if text.startswith('json'):
                text = text[4:].strip()

        variations = json.loads(text)
        if not isinstance(variations, list):
            return [term]

        # For title/skill/keywords: lowercase and include original term
        # For company/location/school: preserve capitalization
        preserve_case = field_type in ('company', 'location', 'school')
        if preserve_case:
            variations = [v.strip() for v in variations if isinstance(v, str) and v.strip()]
        else:
            variations = [v.strip().lower() for v in variations if isinstance(v, str) and v.strip()]
            if term.lower() not in variations:
                variations.insert(0, term.lower())

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for v in variations:
            if v not in seen:
                seen.add(v)
                unique.append(v)

        return unique

    except Exception:
        return [term]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'SENIORITY_LEVELS',
    'HEADCOUNT_RANGES',
    'CREDITS_PER_100_RESULTS',
    'CREDITS_PER_RESULT_SEMANTIC',
    # Main functions
    'search_people_db',
    'build_filters',
    'check_credits',
    'search_people_semantic',
    # Normalization
    'normalize_search_result',
    'normalize_search_results_to_df',
    'semantic_profile_to_legacy_shape',
    # Usage tracking
    'log_search_usage',
    # AI expansion
    'expand_variations',
]
