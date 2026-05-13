"""
Unified Data Normalizers for LinkedIn Enricher

Single source of truth for all field mappings between:
- PhantomBuster (LinkedIn scraper) → Canonical format
- Crustdata (enrichment API) → Canonical format
- Canonical format → Supabase database

All normalization logic lives here. Both dashboard.py and db.py import from this module.
"""

import re
import json
import math
from typing import Optional, Any
from datetime import datetime

# ============================================================================
# FIELD MAPPING DOCUMENTATION
# ============================================================================

# Canonical field names used in Supabase
CANONICAL_FIELDS = [
    'linkedin_url',      # Primary key - normalized URL
    'first_name',
    'last_name',
    'headline',
    'location',
    'summary',
    'current_title',
    'current_company',
    'current_years_in_role',    # Numeric (years as float)
    'current_years_at_company', # Numeric (years as float)
    'skills',                   # Comma-separated string (display) / TEXT[] (DB)
    'education',                # Most recent school name
    'all_employers',            # Comma-separated string (display) / TEXT[] (DB)
    'all_titles',               # Comma-separated string (display) / TEXT[] (DB)
    'all_schools',              # Comma-separated string (display) / TEXT[] (DB)
    'connections_count',
    'followers_count',
    'profile_picture_url',
]

# PhantomBuster field name variations → canonical field
PHANTOMBUSTER_FIELD_MAP = {
    # URL fields (try in order)
    'linkedin_url': ['linkedin_url', 'defaultProfileUrl', 'profileUrl', 'linkedInProfileUrl', 'public_url', 'profileLink'],
    'public_identifier': ['publicIdentifier', 'public_identifier'],

    # Name fields
    'first_name': ['first_name', 'firstName'],
    'last_name': ['last_name', 'lastName'],
    'full_name': ['full_name', 'fullName', 'name'],

    # Profile fields
    'headline': ['headline', 'title'],
    'location': ['location'],
    'summary': ['summary'],

    # Current job fields
    'current_title': ['current_title', 'jobTitle', 'currentJobTitle', 'title'],
    'current_company': ['current_company', 'companyName', 'company', 'currentCompanyName'],

    # Duration fields (text, needs parsing)
    'duration_in_role': ['current_years_in_role', 'durationInRole'],
    'duration_at_company': ['current_years_at_company', 'durationInCompany'],
}

# Crustdata field name variations → canonical field
CRUSTDATA_FIELD_MAP = {
    # URL fields - linkedin_flagship_url has the clean format, linkedin_profile_url is encoded
    'linkedin_url': ['linkedin_flagship_url', 'linkedin_profile_url', 'linkedin_url'],

    # Name fields
    'first_name': ['first_name'],
    'last_name': ['last_name'],
    'full_name': ['name'],

    # Profile fields
    'headline': ['headline'],
    'location': ['location'],
    'summary': ['summary'],

    # Position fields (from positions[0])
    'position_title': ['title', 'job_title'],
    'position_company': ['company_name', 'company', 'organization'],
    'position_duration_role': ['duration_in_role', 'years_in_role'],
    'position_duration_company': ['duration_at_company', 'years_at_company'],

    # Other fields - Crustdata uses num_of_connections
    'connections': ['num_of_connections', 'connections_count', 'connections'],
    'followers': ['followers_count', 'followers'],
    'profile_pic': ['profile_picture_url', 'profile_pic_url', 'profile_picture_permalink'],
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_nan_or_none(value: Any) -> bool:
    """Check if value is NaN, None, empty, or pandas NA."""
    if value is None:
        return True

    # Check for pandas NA type (not JSON serializable)
    if type(value).__name__ == 'NAType':
        return True

    # Try pandas isna
    try:
        import pandas as pd
        if pd.isna(value):
            return True
    except (ImportError, TypeError, ValueError):
        pass

    # Check float NaN/Inf
    if isinstance(value, float):
        try:
            if math.isnan(value) or math.isinf(value):
                return True
        except (TypeError, ValueError):
            pass

    # Check empty string
    if isinstance(value, str) and value.strip() == '':
        return True

    return False


def clean_value(value: Any) -> Any:
    """Clean a single value - convert NaN/None to None, strip strings."""
    if is_nan_or_none(value):
        return None
    if isinstance(value, str):
        return value.strip() or None
    return value


def clean_dict(data: dict, keep_none: bool = False) -> dict:
    """Clean all values in a dict. Optionally keep None values (for batch upsert)."""
    cleaned = {}
    for key, value in data.items():
        if isinstance(value, dict):
            cleaned[key] = clean_dict(value, keep_none)
        elif isinstance(value, list):
            cleaned[key] = [clean_value(v) for v in value if not is_nan_or_none(v)]
        else:
            clean_val = clean_value(value)
            if keep_none or clean_val is not None:
                cleaned[key] = clean_val
    return cleaned


def get_first_valid(data: dict, field_names: list) -> Any:
    """Get first non-null value from a list of possible field names."""
    for field in field_names:
        value = data.get(field)
        if not is_nan_or_none(value):
            return clean_value(value)
    return None


# ============================================================================
# URL NORMALIZATION
# ============================================================================

def normalize_linkedin_url(url: str) -> Optional[str]:
    """
    Normalize LinkedIn URL to canonical format for consistent matching.

    Transformations:
    - Add https:// if missing
    - Remove query parameters
    - Remove trailing slashes
    - Convert to lowercase
    - Validate it's a regular profile URL (not Sales Navigator, company, etc.)

    Returns None if URL is invalid.
    """
    if is_nan_or_none(url):
        return None

    url = str(url).strip()
    if not url:
        return None

    # Add protocol if missing
    if url.startswith('www.'):
        url = 'https://' + url
    elif not url.startswith('http'):
        url = 'https://' + url

    # Remove query parameters
    if '?' in url:
        url = url.split('?')[0]

    # Normalize www prefix - always use www.linkedin.com
    url = url.replace('://linkedin.com', '://www.linkedin.com')

    # Remove trailing slashes
    url = url.rstrip('/')

    # Validate it's a LinkedIn profile URL (case-insensitive check)
    if 'linkedin.com' not in url.lower():
        return None

    # Reject Sales Navigator URLs
    if '/sales/' in url.lower():
        return None

    # Should contain /in/ for personal profiles
    if '/in/' not in url.lower():
        return None

    # Lowercase handling: preserve case for obfuscated URLs (ACoAAA... pattern)
    # These are case-sensitive Crustdata internal IDs — lowercasing breaks enrichment
    slug = url.split('/in/')[-1] if '/in/' in url else ''
    is_obfuscated = slug.lower().startswith('aco')

    if is_obfuscated:
        # Lowercase only the domain portion, preserve the slug case
        in_idx = url.lower().index('/in/')
        url = url[:in_idx].lower() + url[in_idx:]
    else:
        # Lowercase domain, then lowercase slug BUT preserve percent-encoding case
        # LinkedIn URLs with emojis use %F0%9F... — lowercasing to %f0%9f breaks them
        in_idx = url.lower().index('/in/')
        domain = url[:in_idx].lower()
        slug = url[in_idx:]

        if '%' in slug:
            # Preserve percent-encoded sequences (%XX) in uppercase, lowercase the rest
            result = []
            i = 0
            while i < len(slug):
                if slug[i] == '%' and i + 2 < len(slug):
                    result.append('%' + slug[i+1:i+3].upper())
                    i += 3
                else:
                    result.append(slug[i].lower())
                    i += 1
            slug = ''.join(result)
        else:
            slug = slug.lower()

        url = domain + slug

    return url


def extract_linkedin_url(data: dict, field_map: dict = None) -> Optional[str]:
    """
    Extract and normalize LinkedIn URL from data using field map.
    Falls back to constructing from publicIdentifier if URL fields are empty.
    """
    if field_map is None:
        field_map = PHANTOMBUSTER_FIELD_MAP

    # Try URL fields
    url_fields = field_map.get('linkedin_url', ['linkedin_url'])
    for field in url_fields:
        url = data.get(field)
        if url:
            normalized = normalize_linkedin_url(url)
            if normalized:
                return normalized

    # Fallback: construct from publicIdentifier
    id_fields = field_map.get('public_identifier', ['publicIdentifier'])
    for field in id_fields:
        public_id = data.get(field)
        if public_id and public_id != 'null' and not is_nan_or_none(public_id):
            constructed = f"https://www.linkedin.com/in/{public_id}"
            return normalize_linkedin_url(constructed)

    return None


# ============================================================================
# DURATION PARSING
# ============================================================================

def parse_duration(duration_str: Any) -> Optional[float]:
    """
    Parse duration text to numeric years.

    Examples:
    - "8 months" → 0.67
    - "2 years" → 2.0
    - "1 year 6 months" → 1.5
    - "2.5" → 2.5 (already numeric)
    - None → None
    """
    if is_nan_or_none(duration_str):
        return None

    # Already numeric
    if isinstance(duration_str, (int, float)):
        return float(duration_str) if not math.isnan(duration_str) else None

    duration_str = str(duration_str).lower().strip()
    if not duration_str:
        return None

    # Try direct float conversion first
    try:
        return float(duration_str)
    except (ValueError, TypeError):
        pass

    # Parse text like "8 months", "2 years", "1 year 3 months"
    years = 0
    months = 0

    year_match = re.search(r'(\d+(?:\.\d+)?)\s*year', duration_str)
    month_match = re.search(r'(\d+(?:\.\d+)?)\s*month', duration_str)

    if year_match:
        years = float(year_match.group(1))
    if month_match:
        months = float(month_match.group(1))

    if years or months:
        return round(years + months / 12, 2)

    return None


# ============================================================================
# NAME PARSING
# ============================================================================

def parse_full_name(full_name: str) -> tuple[Optional[str], Optional[str]]:
    """
    Split full name into first and last name.

    Returns (first_name, last_name) tuple.
    """
    if is_nan_or_none(full_name):
        return None, None

    full_name = str(full_name).strip()
    if not full_name:
        return None, None

    parts = full_name.split(' ', 1)
    first_name = parts[0] if parts else None
    last_name = parts[1] if len(parts) > 1 else None

    return first_name, last_name


def _parse_start_date_sort_key(raw_date: Any) -> tuple:
    """Convert a start_date value into a sortable tuple.

    Returns (parseable_flag, datetime). `parseable_flag` is 1 for a real date
    and 0 for missing/unparseable. With reverse=True sort, entries with
    parseable_flag=0 always sort LAST (i.e. treated as oldest) regardless of
    the datetime in the tuple — Python's tuple comparison stops at the first
    differing element, so the datetime is only compared between two
    parseable entries.

    Handles:
    - datetime instances (passthrough, tz stripped)
    - ISO 8601 strings ("2025-05-01", "2025-05-01T00:00:00+00:00", "...Z")
    - Year-month strings ("2025-05")
    - Year-only strings ("2025")

    Anything else — "Present", "May 2025", localized month names, garbage —
    is unparseable and sorts last.
    """
    if raw_date is None:
        return (0, datetime.min)
    if isinstance(raw_date, datetime):
        return (1, raw_date.replace(tzinfo=None) if raw_date.tzinfo else raw_date)

    s = str(raw_date).strip()
    if not s:
        return (0, datetime.min)

    try:
        dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
        return (1, dt.replace(tzinfo=None) if dt.tzinfo else dt)
    except ValueError:
        pass

    for fmt in ('%Y-%m', '%Y'):
        try:
            return (1, datetime.strptime(s, fmt))
        except ValueError:
            continue

    return (0, datetime.min)


def pick_current_employer(current_employers: Any) -> Optional[dict]:
    """
    Pick the most recent entry from a current_employers array.

    Crustdata can return multiple entries in current_employers when a person
    has parallel/active roles (advisor + full-time job, two current jobs after
    a switch, IDF reserve + civilian role). The list is not guaranteed to be
    ordered by recency, so taking [0] blindly can silently surface the WRONG
    current employer — e.g. a reserve unit instead of the actual current job.

    Sorts by start_date descending and returns the most recent. start_date is
    parsed as a real datetime (see _parse_start_date_sort_key) so that
    non-ISO and unparseable values ("Present", "May 2025", garbage) sort
    LAST regardless of how their string would compare lexicographically.

    When two entries have the same start_date (or both unparseable), Python's
    stable sort preserves input order — so the tie-break is deterministic.

    Returns None when the input is not a list, is empty, or contains no dicts.
    """
    if not isinstance(current_employers, list):
        return None
    valid = [e for e in current_employers if isinstance(e, dict)]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    return sorted(
        valid,
        key=lambda e: _parse_start_date_sort_key(e.get('start_date')),
        reverse=True,
    )[0]


# ============================================================================
# PHANTOMBUSTER NORMALIZATION
# ============================================================================

def normalize_phantombuster_profile(raw: dict) -> Optional[dict]:
    """
    Normalize a PhantomBuster profile to canonical format.

    Input: Raw PhantomBuster CSV row as dict
    Output: Canonical profile dict ready for Supabase, or None if invalid
    """
    if not raw:
        return None

    # Clean the input
    raw = clean_dict(raw)

    # Extract LinkedIn URL (required)
    linkedin_url = extract_linkedin_url(raw, PHANTOMBUSTER_FIELD_MAP)
    if not linkedin_url:
        return None

    # Extract name
    first_name = get_first_valid(raw, PHANTOMBUSTER_FIELD_MAP['first_name'])
    last_name = get_first_valid(raw, PHANTOMBUSTER_FIELD_MAP['last_name'])

    # Fallback to full name parsing
    if not first_name and not last_name:
        full_name = get_first_valid(raw, PHANTOMBUSTER_FIELD_MAP['full_name'])
        first_name, last_name = parse_full_name(full_name)

    # Extract other fields
    headline = get_first_valid(raw, PHANTOMBUSTER_FIELD_MAP['headline'])
    location = get_first_valid(raw, ['location'])
    summary = get_first_valid(raw, ['summary'])

    # Current job
    current_title = get_first_valid(raw, PHANTOMBUSTER_FIELD_MAP['current_title'])
    current_company = get_first_valid(raw, PHANTOMBUSTER_FIELD_MAP['current_company'])

    # Duration (parse text to numeric)
    duration_role_text = get_first_valid(raw, PHANTOMBUSTER_FIELD_MAP['duration_in_role'])
    duration_company_text = get_first_valid(raw, PHANTOMBUSTER_FIELD_MAP['duration_at_company'])

    return {
        'linkedin_url': linkedin_url,
        'first_name': first_name,
        'last_name': last_name,
        'headline': headline,
        'location': location,
        'summary': summary,
        'current_title': current_title,
        'current_company': current_company,
        'current_years_in_role': parse_duration(duration_role_text),
        'current_years_at_company': parse_duration(duration_company_text),
        'raw_phantombuster': raw,  # Keep original for JSONB storage
    }


# ============================================================================
# CRUSTDATA NORMALIZATION
# ============================================================================

def normalize_crustdata_profile(raw: dict, original_url: str = None) -> Optional[dict]:
    """
    Normalize a Crustdata API response to canonical format.

    Input: Raw Crustdata API response dict
    Output: Canonical profile dict ready for Supabase, or None if invalid

    Args:
        raw: The Crustdata API response for a single profile
        original_url: The LinkedIn URL we sent to the API (for matching)
    """
    if not raw:
        return None

    # Store original URL if provided (for matching back)
    if original_url:
        raw['_original_linkedin_url'] = original_url

    # Clean the input
    raw = clean_dict(raw)

    # Extract LinkedIn URL
    linkedin_url = extract_linkedin_url(raw, CRUSTDATA_FIELD_MAP)
    if not linkedin_url and original_url:
        linkedin_url = normalize_linkedin_url(original_url)

    if not linkedin_url:
        return None

    # Extract name
    first_name = get_first_valid(raw, CRUSTDATA_FIELD_MAP['first_name'])
    last_name = get_first_valid(raw, CRUSTDATA_FIELD_MAP['last_name'])

    # Fallback to full name parsing
    if not first_name and not last_name:
        full_name = get_first_valid(raw, CRUSTDATA_FIELD_MAP['full_name'])
        first_name, last_name = parse_full_name(full_name)

    # Basic fields
    headline = get_first_valid(raw, CRUSTDATA_FIELD_MAP['headline'])
    location = get_first_valid(raw, CRUSTDATA_FIELD_MAP['location'])
    summary = get_first_valid(raw, CRUSTDATA_FIELD_MAP['summary'])

    # Extract current position from current_employers array (Crustdata format)
    current_title = None
    current_company = None

    emp = pick_current_employer(raw.get('current_employers'))
    current_start_date = None
    current_years_in_role = None
    if emp:
        current_title = emp.get('employee_title') or emp.get('title')
        current_company = emp.get('employer_name') or emp.get('company_name')
        # Capture tenure of the current role so post-enrichment filters
        # (Filter+, DB search) can filter on "minimum X years at current
        # company" without re-loading raw_crustdata. start_date is preserved
        # as the raw string; current_years_in_role is computed against today
        # using the same parser pick_current_employer uses for sorting, so
        # unparseable "Present" / "May 2025" strings leave it None instead of
        # producing nonsensical multi-millennium values.
        raw_start = emp.get('start_date')
        if raw_start is not None and str(raw_start).strip():
            current_start_date = str(raw_start).strip()
            parseable, dt = _parse_start_date_sort_key(raw_start)
            if parseable:
                current_years_in_role = round((datetime.now() - dt).days / 365.25, 1)

    # Fallback to top-level fields
    if not current_title:
        current_title = raw.get('title') or raw.get('headline', '').split(' at ')[0] if ' at ' in raw.get('headline', '') else raw.get('title')
    if not current_company:
        current_company = raw.get('company') or raw.get('company_name')

    # Skills - convert array to comma-separated string
    skills = raw.get('skills', [])
    if isinstance(skills, list):
        skills_str = ', '.join(str(s) for s in skills[:50] if s)
    elif skills:
        skills_str = str(skills)
    else:
        skills_str = None

    # Past positions - format work history for preview
    past_employers = raw.get('past_employers', [])
    past_positions_parts = []
    for emp in past_employers[:10]:  # Limit to 10
        if isinstance(emp, dict):
            title = emp.get('employee_title') or emp.get('title') or ''
            company = emp.get('employer_name') or emp.get('company_name') or ''
            if title and company:
                past_positions_parts.append(f"{title} at {company}")
            elif title or company:
                past_positions_parts.append(title or company)
    past_positions_str = ' | '.join(past_positions_parts) if past_positions_parts else None

    # Pre-flattened arrays from Crustdata (convert to comma-separated for display)
    all_employers = raw.get('all_employers', [])
    all_employers_str = ', '.join(str(x) for x in all_employers if x) if isinstance(all_employers, list) else None

    all_titles = raw.get('all_titles', [])
    all_titles_str = ', '.join(str(x) for x in all_titles if x) if isinstance(all_titles, list) else None

    all_schools = raw.get('all_schools', [])
    all_schools_str = ', '.join(str(x) for x in all_schools if x) if isinstance(all_schools, list) else None

    # Other fields
    connections = get_first_valid(raw, CRUSTDATA_FIELD_MAP['connections'])
    followers = get_first_valid(raw, CRUSTDATA_FIELD_MAP['followers'])
    profile_pic = get_first_valid(raw, CRUSTDATA_FIELD_MAP['profile_pic'])

    # Combine name for display
    name = f"{first_name or ''} {last_name or ''}".strip() or None

    return {
        'linkedin_url': linkedin_url,
        'name': name,
        'first_name': first_name,
        'last_name': last_name,
        'headline': headline,
        'location': location,
        'summary': summary,
        'current_title': current_title,
        'current_company': current_company,
        'skills': skills_str,
        'all_employers': all_employers_str,
        'all_titles': all_titles_str,
        'all_schools': all_schools_str,
        'past_positions': past_positions_str,
        'connections_count': connections,
        'current_start_date': current_start_date,
        'current_years_in_role': current_years_in_role,
        # Note: raw_crustdata NOT stored here to save memory - fetch from DB when needed
    }


# ============================================================================
# BATCH NORMALIZATION
# ============================================================================

def normalize_phantombuster_batch(profiles: list[dict]) -> list[dict]:
    """
    Normalize a batch of PhantomBuster profiles.
    Returns list of valid normalized profiles (skips invalid ones).
    """
    normalized = []
    for raw in profiles:
        profile = normalize_phantombuster_profile(raw)
        if profile:
            normalized.append(profile)
    return normalized


def normalize_crustdata_batch(profiles: list[dict], original_urls: list[str] = None) -> list[dict]:
    """
    Normalize a batch of Crustdata API responses.
    Returns list of valid normalized profiles (skips invalid ones).

    Args:
        profiles: List of Crustdata API responses
        original_urls: Optional list of original URLs (same order as profiles)
    """
    normalized = []
    for i, raw in enumerate(profiles):
        original_url = original_urls[i] if original_urls and i < len(original_urls) else None
        profile = normalize_crustdata_profile(raw, original_url)
        if profile:
            normalized.append(profile)
    return normalized


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def profile_to_display_dict(profile: dict) -> dict:
    """
    Convert canonical profile to display-friendly dict for UI.
    Combines first/last name, formats fields nicely.
    """
    first = profile.get('first_name') or ''
    last = profile.get('last_name') or ''
    name = f"{first} {last}".strip() or 'Unknown'

    # Extract past_positions from raw_crustdata - full JSON for preview
    raw = profile.get('raw_crustdata') or profile.get('raw_data') or {}
    past_employers = raw.get('past_employers', [])
    past_positions = json.dumps(past_employers, ensure_ascii=False) if past_employers else ''

    return {
        'name': name,
        'first_name': first,
        'last_name': last,
        'current_company': profile.get('current_company') or '',
        'current_title': profile.get('current_title') or '',
        'headline': profile.get('headline') or '',
        'location': profile.get('location') or '',
        'linkedin_url': profile.get('linkedin_url') or '',
        'skills': profile.get('skills') or '',
        'all_employers': profile.get('all_employers') or '',
        'all_titles': profile.get('all_titles') or '',
        'all_schools': profile.get('all_schools') or '',
        'past_positions': past_positions,
        'summary': profile.get('summary') or '',
        'connections_count': profile.get('connections_count'),
    }


def profiles_to_display_df(profiles: list[dict]):
    """
    Convert list of canonical profiles to a pandas DataFrame for display.
    """
    import pandas as pd

    if not profiles:
        return pd.DataFrame()

    display_data = [profile_to_display_dict(p) for p in profiles]
    return pd.DataFrame(display_data)


# ============================================================================
# ISRAELI MILITARY SERVICE DETECTION
# ============================================================================
#
# Israeli military service is mandatory (ages 18-21) and MUST NOT count toward
# any user-stated "max N years experience" / "reject >N years" rule. Single
# source of truth — used by AI Screen prompt pre-computation in dashboard.py
# (compute_role_durations) AND by the regression test
# test_ai_screen_excludes_army.py. Keep them in sync.
#
# Match rule: a position is military if any keyword appears as a substring of
# the lowercased employer name OR the lowercased role title. The list is
# intentionally generous (Hebrew + English + bare unit numbers) so the
# detection works against Crustdata's inconsistent formatting (sometimes
# "Unit 8200", sometimes just "8200", sometimes "IDF - 8200").
#
# Hebrew tokens are matched on the raw Hebrew string (no transliteration).
# - צה"ל / צה״ל = IDF
# - צבא = army
# - יחידה = unit
# - מודיעין = intelligence
#
# DO NOT add civilian "security" or "defense" companies here — that would
# wrongly exclude legitimate civilian work (e.g. Check Point, Palo Alto).

MILITARY_KEYWORDS = frozenset({
    # English — generic
    "idf",
    "israel defense forces",
    "israeli defense forces",
    "israeli army",
    "israeli military",
    "israeli air force",
    "israeli navy",
    "iaf",
    "israeli intelligence",
    "idf intelligence",
    "intelligence corps",
    "military service",
    "mandatory service",

    # English — well-known IDF tech / intelligence units
    "unit 8200",
    "8200",
    "unit 9900",
    "9900",
    "unit 81",
    "mamram",
    "talpiot",
    "matzov",
    "c4i",
    "j6 & cyber defense",
    "cyber defense directorate",
    "ofek unit",

    # English — common combat / special units (recruiters sometimes leave
    # them in the work history and they still must not count as industry yoe)
    "sayeret matkal",
    "shaldag",
    "duvdevan",
    "shayetet 13",
    "egoz",
    "maglan",
    "givati",
    "golani",
    "nahal",
    "paratroopers",

    # Hebrew
    'צה"ל',
    "צה״ל",
    "צבא",
    "צבא ההגנה לישראל",
    "8200 יחידה",
    "יחידה 8200",
    "יחידה 9900",
    "ממר\"ם",
    "ממר״ם",
    "ממרם",
    "תלפיות",
    "מודיעין",
})


def is_military_position(title: Optional[str], company: Optional[str]) -> bool:
    """Return True if a role title / employer name looks like Israeli military
    service. Conservative substring match against MILITARY_KEYWORDS.

    Used by AI Screen's pre-computed EXPERIENCE SUMMARY block to exclude
    mandatory IDF / 8200 / Mamram / Talpiot / C4I / etc. service from
    INDUSTRY EXPERIENCE so the model does not count it toward user-stated
    "max N years" / "reject >N years" rules.
    """
    title_l = (title or "").lower()
    company_l = (company or "").lower()
    if not title_l and not company_l:
        return False
    for kw in MILITARY_KEYWORDS:
        # Hebrew keywords are not lowercased (case is meaningless for Hebrew
        # consonant text). We still match against the lowercased haystack
        # because mixed-case Hebrew in English text is uncommon and
        # .lower() is a no-op on Hebrew characters.
        if kw in company_l or kw in title_l:
            return True
        # Also check the raw (non-lowercased) original strings so Hebrew
        # quote-mark variants (״ vs ") match regardless of input casing
        # decisions upstream.
        if kw in (company or "") or kw in (title or ""):
            return True
    return False
