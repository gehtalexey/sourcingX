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
from normalizers import normalize_linkedin_url, clean_value, is_nan_or_none


# =============================================================================
# CONSTANTS
# =============================================================================

CRUSTDATA_SEARCH_ENDPOINT = "https://api.crustdata.com/screener/persondb/search"
CRUSTDATA_CREDITS_ENDPOINT = "https://api.crustdata.com/account/credits"

# Seniority levels supported by Crustdata
SENIORITY_LEVELS = [
    "CXO",
    "Vice President",
    "Director",
    "Manager",
    "Senior",
    "Entry",
    "Training",
    "Owner / Partner",
]

# Company headcount ranges supported by Crustdata
HEADCOUNT_RANGES = [
    "1-10",
    "11-50",
    "51-200",
    "201-500",
    "501-1000",
    "1001-5000",
    "5001-10000",
    "10001+",
]

# Credits per 100 results
CREDITS_PER_100_RESULTS = 3


# =============================================================================
# CONFIG LOADING
# =============================================================================

def _load_api_key() -> str:
    """Load Crustdata API key from config.json or environment."""
    config_path = Path(__file__).parent / 'config.json'

    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_key = config.get('api_key')
                if api_key and api_key != "YOUR_CRUSTDATA_API_KEY_HERE":
                    return api_key
        except (json.JSONDecodeError, IOError):
            pass

    # Fallback to environment variable
    import os
    api_key = os.environ.get('CRUSTDATA_API_KEY')
    if api_key:
        return api_key

    raise AuthenticationError(
        "Crustdata",
        message="No Crustdata API key found. Set in config.json or CRUSTDATA_API_KEY env var."
    )


# =============================================================================
# FILTER BUILDERS
# =============================================================================

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
        keywords: Comma-separated keywords (OR across headline/summary/skills)
        past_companies: Comma-separated past company names (substring match)
        past_titles: Comma-separated past job titles (substring match)
        school: School/university name (substring match)
        recently_changed_jobs: If True, filter for job changes in last 90 days
        has_verified_email: If True, filter for verified business email

    Returns:
        Filter dict ready for Crustdata API: {"op": "and", "conditions": [...]}
    """
    conditions = []

    # Title filter (substring match on current job title)
    # Supports comma-separated values for OR logic
    if title and title.strip():
        title_values = [t.strip() for t in title.split(",") if t.strip()]
        if len(title_values) == 1:
            # Single title - simple condition
            conditions.append({
                "column": "current_employers.title",
                "type": "[.]",
                "value": title_values[0]
            })
        elif len(title_values) > 1:
            # Multiple titles - OR condition
            title_conditions = [
                {"column": "current_employers.title", "type": "[.]", "value": t}
                for t in title_values
            ]
            conditions.append({
                "op": "or",
                "conditions": title_conditions
            })

    # Current company filter (comma-separated for OR logic)
    if company and company.strip():
        company_values = [c.strip() for c in company.split(",") if c.strip()]
        if len(company_values) == 1:
            conditions.append({
                "column": "current_employers.name",
                "type": "[.]",
                "value": company_values[0]
            })
        elif len(company_values) > 1:
            company_conditions = [
                {"column": "current_employers.name", "type": "[.]", "value": c}
                for c in company_values
            ]
            conditions.append({
                "op": "or",
                "conditions": company_conditions
            })

    # Location filter
    if location and location.strip():
        conditions.append({
            "column": "region",
            "type": "[.]",
            "value": location.strip()
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

    # Keywords filter (OR across headline, summary, skills)
    if keywords and keywords.strip():
        keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
        if keyword_list:
            # Build OR condition for keywords
            keyword_conditions = []
            for kw in keyword_list:
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

            # Wrap in OR condition
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

    # School filter
    if school and school.strip():
        conditions.append({
            "column": "education_background.institute_name",
            "type": "[.]",
            "value": school.strip()
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

    # Build request body
    body = {
        "limit": min(limit, 1000),  # Cap at 1000
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


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_search_result(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a Crustdata search result to pipeline DataFrame format.

    Search results have partial data (no full employment history).
    Enrichment is recommended for full details.

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
            - _needs_enrichment = True
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

    # Current employer info
    current_company = ''
    current_title = ''
    current_employers = profile.get('current_employers', [])
    if current_employers and isinstance(current_employers, list) and len(current_employers) > 0:
        emp = current_employers[0]
        if isinstance(emp, dict):
            current_company = emp.get('employer_name') or emp.get('name') or ''
            current_title = emp.get('employee_title') or emp.get('title') or ''

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

    # Experience
    years_exp = profile.get('years_of_experience_raw')
    if years_exp is None:
        years_exp = profile.get('years_of_experience')

    # Extract seniority from current employer
    seniority = ''
    if current_employers and isinstance(current_employers, list) and len(current_employers) > 0:
        emp = current_employers[0]
        if isinstance(emp, dict):
            seniority = emp.get('seniority_level') or emp.get('seniority') or ''

    # Extract company size from current employer
    company_size = ''
    if current_employers and isinstance(current_employers, list) and len(current_employers) > 0:
        emp = current_employers[0]
        if isinstance(emp, dict):
            company_size = emp.get('company_headcount_range') or emp.get('headcount') or ''

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
        # Skills and experience
        'skills': skills_str,
        'years_experience': years_exp,
        # Metadata
        '_source': 'crustdata_search',
        '_needs_enrichment': True,
        '_raw_search_result': profile,  # Keep raw for debugging
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

_TITLE_PROMPT = (
    'Given the job title "{term}", list 5-10 common alternative titles for the '
    'same or very similar role. Include spelling variations (Lead vs Leader), '
    'abbreviations, and industry-standard equivalents at the same seniority level. '
    'Return ONLY a valid JSON array of lowercase strings. '
    'Example: ["team lead", "tech lead", "engineering manager"]'
)

_SKILL_PROMPT = (
    'Given the technical skill "{term}", list 5-10 closely related skills, tools, '
    'or technologies that someone with this skill would likely know or that '
    'recruiters search for interchangeably. Include abbreviations and alternative names. '
    'Return ONLY a valid JSON array of lowercase strings. '
    'Example: ["k8s", "docker", "helm", "container orchestration"]'
)


def expand_variations(
    term: str,
    field_type: str = 'title',
    openai_api_key: str = None,
) -> List[str]:
    """
    Use OpenAI gpt-4o-mini to expand a term into common variations.

    Args:
        term: The user's input (e.g., "team leader" or "Kubernetes")
        field_type: 'title' for job titles, 'skill' for technical skills
        openai_api_key: OpenAI API key

    Returns:
        List of 5-10 variations (always includes the original term).
        On failure, returns [term] (graceful fallback).
    """
    if not term or not term.strip():
        return []

    term = term.strip()

    if not openai_api_key:
        return [term]

    prompt_template = _TITLE_PROMPT if field_type == 'title' else _SKILL_PROMPT
    prompt = prompt_template.format(term=term)

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

        # Ensure original term is included, deduplicate
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
    # Main functions
    'search_people_db',
    'build_filters',
    'check_credits',
    # Normalization
    'normalize_search_result',
    'normalize_search_results_to_df',
    # Usage tracking
    'log_search_usage',
    # AI expansion
    'expand_variations',
]
