"""
Supabase Database Module for LinkedIn Enricher (v2 - Crustdata Only)

Stores only Crustdata-enriched profiles. PhantomBuster data stays in UI session state.
Uses REST API directly - no supabase package required.
"""

import os
import json
import re
import requests
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
import pandas as pd

from normalizers import normalize_linkedin_url

# Refresh threshold for re-enriching stale profiles
ENRICHMENT_REFRESH_MONTHS = 3


class SupabaseClient:
    """Simple Supabase REST API client."""

    def __init__(self, url: str, key: str):
        self.url = url.rstrip('/')
        self.key = key
        self.headers = {
            'apikey': key,
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            'Prefer': 'return=representation'
        }

    def _request(self, method: str, endpoint: str, params: dict = None, json_data: dict = None) -> dict:
        """Make a request to Supabase REST API."""
        url = f"{self.url}/rest/v1/{endpoint}"
        response = requests.request(
            method,
            url,
            headers=self.headers,
            params=params,
            json=json_data,
            timeout=30
        )
        response.raise_for_status()
        if response.text:
            return response.json()
        return {}

    def select(self, table: str, columns: str = '*', filters: dict = None, limit: int = 5000) -> list:
        """Select rows from a table. Auto-paginates past Supabase 1000-row server limit.
        Default limit reduced to 5000 to prevent OOM on large datasets."""
        PAGE_SIZE = 1000
        params = {'select': columns}
        if filters:
            for key, value in filters.items():
                params[key] = value

        # Small requests don't need pagination
        if limit <= PAGE_SIZE:
            params['limit'] = limit
            return self._request('GET', table, params=params)

        # Paginate to get all results
        all_results = []
        offset = 0
        while offset < limit:
            page_limit = min(PAGE_SIZE, limit - offset)
            page_params = {**params, 'limit': page_limit, 'offset': offset}
            page = self._request('GET', table, params=page_params)
            if not page:
                break
            all_results.extend(page)
            if len(page) < page_limit:
                break
            offset += PAGE_SIZE
        return all_results

    def insert(self, table: str, data: dict) -> list:
        """Insert a row into a table."""
        return self._request('POST', table, json_data=data)

    def upsert(self, table: str, data: dict, on_conflict: str = None) -> list:
        """Upsert (insert or update) a row."""
        headers = self.headers.copy()
        if on_conflict:
            headers['Prefer'] = f'resolution=merge-duplicates,return=representation'
        url = f"{self.url}/rest/v1/{table}"
        params = {}
        if on_conflict:
            params['on_conflict'] = on_conflict
        # Pre-serialize JSON to handle NaN values
        json_str = json.dumps(data, allow_nan=True)
        json_str = json_str.replace(': NaN', ': null').replace(':NaN', ':null')
        json_str = json_str.replace(': Infinity', ': null').replace(':Infinity', ':null')
        json_str = json_str.replace(': -Infinity', ': null').replace(':-Infinity', ':null')
        response = requests.post(url, headers=headers, params=params, data=json_str, timeout=30)
        if response.status_code >= 400:
            error_msg = f"{response.status_code}: {response.text}"
            raise requests.HTTPError(error_msg)
        if response.text:
            return response.json()
        return []

    def upsert_batch(self, table: str, rows: list, on_conflict: str = None) -> list:
        """Upsert multiple rows in a single request (much faster than individual upserts)."""
        if not rows:
            return []
        headers = self.headers.copy()
        if on_conflict:
            headers['Prefer'] = f'resolution=merge-duplicates,return=representation'
        url = f"{self.url}/rest/v1/{table}"
        params = {}
        if on_conflict:
            params['on_conflict'] = on_conflict
        json_str = json.dumps(rows, allow_nan=True)
        json_str = json_str.replace(': NaN', ': null').replace(':NaN', ':null')
        json_str = json_str.replace(': Infinity', ': null').replace(':Infinity', ':null')
        json_str = json_str.replace(': -Infinity', ': null').replace(':-Infinity', ':null')
        response = requests.post(url, headers=headers, params=params, data=json_str, timeout=60)
        if response.status_code >= 400:
            error_msg = f"{response.status_code}: {response.text}"
            raise requests.HTTPError(error_msg)
        if response.text:
            return response.json()
        return []

    def update(self, table: str, data: dict, filters: dict) -> list:
        """Update rows matching filters."""
        params = {}
        for key, value in filters.items():
            params[key] = f'eq.{value}'
        return self._request('PATCH', table, params=params, json_data=data)

    def delete(self, table: str, filters: dict) -> list:
        """Delete rows matching filters."""
        params = {}
        for key, value in filters.items():
            params[key] = f'eq.{value}'
        return self._request('DELETE', table, params=params)

    def count(self, table: str, filters: dict = None) -> int:
        """Count rows in a table."""
        headers = self.headers.copy()
        headers['Prefer'] = 'count=exact'
        headers['Range-Unit'] = 'items'
        url = f"{self.url}/rest/v1/{table}"
        params = {'select': 'id'}
        if filters:
            for key, value in filters.items():
                params[key] = value
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        content_range = response.headers.get('Content-Range', '*/0')
        total = content_range.split('/')[-1]
        return int(total) if total != '*' else 0


def get_supabase_client() -> Optional[SupabaseClient]:
    """Get Supabase client from config.json, Streamlit secrets, or environment."""
    url = None
    key = None

    # Try config.json first (local development)
    try:
        config_path = Path(__file__).parent / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                url = config.get('supabase_url')
                key = config.get('supabase_key')
    except Exception:
        pass

    # Try Streamlit secrets (cloud deployment)
    if not url or not key:
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                url = url or st.secrets.get('supabase_url')
                key = key or st.secrets.get('supabase_key')
        except Exception:
            pass

    # Fall back to environment variables
    if not url:
        url = os.environ.get('SUPABASE_URL')
    if not key:
        key = os.environ.get('SUPABASE_KEY')

    if url and key:
        return SupabaseClient(url, key)
    return None


# ============================================================================
# PROFILE OPERATIONS (Crustdata Enriched Profiles Only)
# ============================================================================

def save_enriched_profile(client: SupabaseClient, linkedin_url: str, crustdata_response: dict, original_url: str = None) -> dict:
    """Save a Crustdata-enriched profile to the database.

    Simplified approach: Store raw_data as-is, extract only title/company for indexing.
    All other fields are extracted at display time from raw_data.

    Args:
        client: SupabaseClient instance
        linkedin_url: The LinkedIn URL (used as primary key, typically from Crustdata)
        crustdata_response: Raw response from Crustdata API
        original_url: The original input URL (for matching with loaded data)

    Returns:
        The saved profile record
    """
    linkedin_url = normalize_linkedin_url(linkedin_url)
    if not linkedin_url:
        raise ValueError("Valid linkedin_url is required")

    # Also normalize original_url for matching
    original_url = normalize_linkedin_url(original_url) if original_url else None

    cd = crustdata_response or {}

    # Extract name for indexed column
    name = cd.get('name') or ''
    if not name:
        first_name = cd.get('first_name') or ''
        last_name = cd.get('last_name') or ''
        name = f"{first_name} {last_name}".strip()

    # Extract location
    location = cd.get('location') or ''

    # Extract only title/company for indexed filtering
    current_title = None
    current_company = None

    # Try current_employers first (Crustdata format)
    current_employers = cd.get('current_employers') or []
    if current_employers and isinstance(current_employers, list):
        emp = current_employers[0] if current_employers else {}
        if isinstance(emp, dict):
            current_title = emp.get('employee_title') or emp.get('title')
            current_company = emp.get('employer_name') or emp.get('company_name')

    # Fallback: extract from headline (e.g., "CEO at Company")
    if not current_title or not current_company:
        headline = cd.get('headline', '')
        if headline and ' at ' in headline:
            parts = headline.split(' at ', 1)
            if not current_title:
                current_title = parts[0].strip()
            if not current_company and len(parts) > 1:
                current_company = parts[1].split('/')[0].strip()

    # Extract pre-flattened arrays from Crustdata (already provided by API)
    all_employers = cd.get('all_employers') or []
    all_titles = cd.get('all_titles') or []
    all_schools = cd.get('all_schools') or []
    skills = cd.get('skills') or []

    # Ensure they're lists of strings
    all_employers = [str(x) for x in all_employers if x] if isinstance(all_employers, list) else []
    all_titles = [str(x) for x in all_titles if x] if isinstance(all_titles, list) else []
    all_schools = [str(x) for x in all_schools if x] if isinstance(all_schools, list) else []
    skills = [str(x) for x in skills if x] if isinstance(skills, list) else []

    # Data to save - indexed fields + raw_data for everything else
    data = {
        'linkedin_url': linkedin_url,
        'original_url': original_url,  # For matching with loaded data
        'raw_data': crustdata_response,
        'name': name if name else None,
        'location': location if location else None,
        'current_title': current_title,
        'current_company': current_company,
        'all_employers': all_employers if all_employers else None,
        'all_titles': all_titles if all_titles else None,
        'all_schools': all_schools if all_schools else None,
        'skills': skills if skills else None,
        'status': 'enriched',
        'enriched_at': datetime.utcnow().isoformat(),
    }

    # Remove None values
    data = {k: v for k, v in data.items() if v is not None}

    result = client.upsert('profiles', data, on_conflict='linkedin_url')
    return result[0] if result else None


def save_enriched_profiles_batch(client: SupabaseClient, profiles: list[tuple[str, dict]]) -> dict:
    """Save multiple enriched profiles in a batch.

    Args:
        client: SupabaseClient instance
        profiles: List of (linkedin_url, crustdata_response) tuples

    Returns:
        Stats dict with 'saved' and 'errors' counts
    """
    stats = {'saved': 0, 'errors': 0}

    for linkedin_url, crustdata_response in profiles:
        try:
            save_enriched_profile(client, linkedin_url, crustdata_response)
            stats['saved'] += 1
        except Exception as e:
            print(f"[DB] Error saving {linkedin_url}: {e}")
            stats['errors'] += 1

    return stats


# Backwards compatibility alias
def update_profile_enrichment(client: SupabaseClient, linkedin_url: str, crustdata_response: dict, original_url: str = None) -> dict:
    """Alias for save_enriched_profile (backwards compatibility)."""
    return save_enriched_profile(client, linkedin_url, crustdata_response, original_url=original_url)


def update_profile_screening(client: SupabaseClient, linkedin_url: str, score: int, fit_level: str,
                              summary: str, reasoning: str) -> dict:
    """Update profile with AI screening results."""
    linkedin_url = normalize_linkedin_url(linkedin_url)

    data = {
        'linkedin_url': linkedin_url,
        'screening_score': score,
        'screening_fit_level': fit_level,
        'screening_summary': summary,
        'screening_reasoning': reasoning,
        'screened_at': datetime.utcnow().isoformat(),
        'status': 'screened',
    }

    result = client.upsert('profiles', data, on_conflict='linkedin_url')
    return result[0] if result else None


def update_profile_screening_batch(client: SupabaseClient, results: list) -> dict:
    """Batch update profiles with AI screening results (single DB call instead of N).

    Args:
        client: SupabaseClient instance
        results: List of dicts with keys: linkedin_url, score, fit_level, summary, reasoning
    Returns:
        Stats dict with 'saved' and 'errors' counts
    """
    if not results:
        return {'saved': 0, 'errors': 0}

    now = datetime.utcnow().isoformat()
    rows = []
    for r in results:
        url = normalize_linkedin_url(r.get('linkedin_url', ''))
        if not url:
            continue
        rows.append({
            'linkedin_url': url,
            'screening_score': r.get('score'),
            'screening_fit_level': r.get('fit_level'),
            'screening_summary': r.get('summary'),
            'screening_reasoning': r.get('reasoning'),
            'screened_at': now,
            'status': 'screened',
        })

    if not rows:
        return {'saved': 0, 'errors': 0}

    try:
        client.upsert_batch('profiles', rows, on_conflict='linkedin_url')
        return {'saved': len(rows), 'errors': 0}
    except Exception as e:
        print(f"[DB] Batch screening save failed: {e}")
        return {'saved': 0, 'errors': len(rows)}


def update_profile_email(client: SupabaseClient, linkedin_url: str, email: str, source: str = 'salesql') -> dict:
    """Update profile with email from SalesQL or other source."""
    linkedin_url = normalize_linkedin_url(linkedin_url)
    client.update('profiles', {'email': email, 'email_source': source}, {'linkedin_url': linkedin_url})
    return {'linkedin_url': linkedin_url, 'email': email}


# ============================================================================
# QUERY OPERATIONS
# ============================================================================

def get_profile(client: SupabaseClient, linkedin_url: str) -> Optional[dict]:
    """Get a single profile by LinkedIn URL."""
    linkedin_url = normalize_linkedin_url(linkedin_url)
    result = client.select('profiles', '*', {'linkedin_url': f'eq.{linkedin_url}'})
    return result[0] if result else None


def get_profiles_needing_screening(client: SupabaseClient, limit: int = 100) -> list:
    """Get enriched profiles that haven't been screened yet."""
    return client.select('profiles', '*', {'status': 'eq.enriched', 'screening_score': 'is.null'}, limit=limit)


def get_profiles_by_status(client: SupabaseClient, status: str, limit: int = 1000) -> list:
    """Get profiles by pipeline status."""
    return client.select('profiles', '*', {'status': f'eq.{status}'}, limit=limit)


def get_profiles_by_fit_level(client: SupabaseClient, fit_level: str, limit: int = 1000) -> list:
    """Get profiles by screening fit level."""
    return client.select('profiles', '*', {'screening_fit_level': f'eq.{fit_level}'}, limit=limit)


def get_all_profiles(client: SupabaseClient, limit: int = 10000, include_raw_data: bool = False) -> list:
    """Get all profiles.

    Args:
        client: SupabaseClient instance
        limit: Maximum number of profiles to return
        include_raw_data: If False (default), excludes raw_data to save memory.
                         Set to True only when raw_data is needed (e.g., screening).
    """
    if include_raw_data:
        return client.select('profiles', '*', limit=limit)
    else:
        # Select only indexed columns to save memory (exclude raw_data)
        columns = 'linkedin_url,original_url,name,location,current_title,current_company,all_employers,all_titles,all_schools,skills,status,enriched_at,screening_score,screening_fit_level,screening_summary,screening_reasoning,screened_at,email'
        return client.select('profiles', columns, limit=limit)


def get_profiles_by_urls(client: SupabaseClient, urls: list, include_raw_data: bool = True) -> list:
    """Fetch profiles by LinkedIn URLs.

    Args:
        client: SupabaseClient instance
        urls: List of LinkedIn URLs to fetch
        include_raw_data: If True, includes raw_data JSONB (needed for screening)

    Returns:
        List of profile dicts
    """
    if not urls:
        return []

    # Normalize URLs
    normalized = [normalize_linkedin_url(u) for u in urls if u]
    normalized = [u for u in normalized if u]

    if not normalized:
        return []

    results = []
    batch_size = 50  # Supabase URL length limits

    for i in range(0, len(normalized), batch_size):
        batch = normalized[i:i+batch_size]
        # Use 'in' filter for multiple URLs
        url_list = ','.join(batch)
        filters = {'linkedin_url': f'in.({url_list})'}

        if include_raw_data:
            columns = '*'
        else:
            columns = 'linkedin_url,name,location,current_title,current_company,all_employers,all_titles,all_schools,skills,email,enriched_at'

        batch_results = client.select('profiles', columns, filters, limit=len(batch))
        results.extend(batch_results)

    return results


def get_pipeline_stats(client: SupabaseClient) -> dict:
    """Get pipeline funnel statistics."""
    try:
        result = client.select('pipeline_stats', '*')
        if result:
            return result[0]
    except:
        pass
    return {}


def search_profiles(client: SupabaseClient, query: str, limit: int = 100) -> list:
    """Search profiles by name, company, or title (legacy - use search_profiles_fulltext for better results)."""
    results = client.select('profiles', '*', {'current_company': f'ilike.%{query}%'}, limit=limit)
    if len(results) < limit:
        more = client.select('profiles', '*', {'name': f'ilike.%{query}%'}, limit=limit - len(results))
        results.extend(more)
    return results


def _convert_to_tsquery(query: str) -> str:
    """Convert user-friendly boolean query to PostgreSQL tsquery syntax.

    Converts:
        OR, |, comma -> |
        AND, & -> &
        NOT, -, ! -> !
        "phrase" -> phrase (keeps as single term)

    Examples:
        "node OR node.js" -> "node | node.js"
        "python AND django" -> "python & django"
        "engineer NOT junior" -> "engineer & !junior"
    """
    import re

    # Handle quoted phrases - replace spaces with <-> for phrase search
    def handle_phrase(match):
        phrase = match.group(1)
        # Convert spaces to <-> for tsquery phrase matching
        words = phrase.split()
        if len(words) > 1:
            return ' <-> '.join(words)
        return phrase

    result = re.sub(r'"([^"]+)"', handle_phrase, query)

    # Convert boolean operators (case insensitive)
    result = re.sub(r'\bOR\b', '|', result, flags=re.IGNORECASE)
    result = re.sub(r'\bAND\b', '&', result, flags=re.IGNORECASE)
    result = re.sub(r'\bNOT\b', '& !', result, flags=re.IGNORECASE)

    # Convert symbols
    result = result.replace(',', ' | ')

    # Handle - and ! at word boundaries (NOT operator)
    result = re.sub(r'(?<![a-zA-Z0-9])-(?=[a-zA-Z])', '& !', result)
    result = re.sub(r'(?<![a-zA-Z0-9])!(?=[a-zA-Z])', '& !', result)

    # Clean up multiple spaces and operators
    result = re.sub(r'\s+', ' ', result).strip()
    result = re.sub(r'\|\s*\|', '|', result)  # || -> |
    result = re.sub(r'&\s*&', '&', result)    # && -> &
    result = re.sub(r'^\s*[&|]\s*', '', result)  # Remove leading operators
    result = re.sub(r'\s*[&|]\s*$', '', result)  # Remove trailing operators

    return result


def search_profiles_fulltext(client: SupabaseClient, query: str, limit: int = 5000) -> list:
    """Full-text search across all profile data with pagination.

    Searches: name, title, company, location, skills, all employers, all titles, all schools.
    Requires the search_profiles_text RPC function (see migrations/009_add_fulltext_search.sql).
    Uses pagination to bypass Supabase 1000 row limit.

    Supports boolean syntax:
        - OR: node OR node.js, node | node.js, node, node.js
        - AND: python AND django, python & django
        - NOT: engineer NOT junior, engineer -junior

    Args:
        client: SupabaseClient instance
        query: Search query (e.g., "node OR node.js", "python AND kubernetes")
        limit: Maximum results to return (default 5000, supports up to ~10k)

    Returns:
        List of matching profile dicts, ranked by relevance (without raw_data to save memory)
    """
    # Fields to keep (exclude raw_data and search_text to save memory)
    KEEP_FIELDS = {'linkedin_url', 'name', 'current_title', 'current_company', 'location',
                   'all_employers', 'all_titles', 'all_schools', 'skills', 'email',
                   'enriched_at', 'screening_score', 'screening_fit_level', 'status'}

    try:
        all_results = []
        page_size = 1000  # Supabase max rows per request
        offset = 0

        while len(all_results) < limit:
            remaining = limit - len(all_results)
            batch_size = min(page_size, remaining)

            response = requests.post(
                f"{client.url}/rest/v1/rpc/search_profiles_text",
                headers=client.headers,
                json={"query": query, "p_limit": batch_size, "p_offset": offset},
                timeout=30
            )
            response.raise_for_status()
            batch = response.json()

            if not batch:
                break  # No more results

            # Strip large fields to save memory
            batch = [{k: v for k, v in p.items() if k in KEEP_FIELDS} for p in batch]
            all_results.extend(batch)

            if len(batch) < batch_size:
                break  # Last page

            offset += len(batch)

        return all_results
    except Exception as e:
        print(f"[DB] Full-text search error: {e}")
        # Fallback to basic search
        return search_profiles(client, query, limit)


# ============================================================================
# BOOLEAN QUERY PARSER
# ============================================================================

def parse_boolean_query(query: str, column: str) -> str:
    """Parse boolean query string into PostgREST filter syntax.

    Supports:
        - Quoted phrases: "fullstack developer"
        - OR: term1 OR term2, term1 | term2, term1, term2
        - AND: term1 AND term2, term1 & term2, +term
        - NOT: NOT term, -term, !term
        - Grouping: (term1 OR term2)

    Examples:
        '"fullstack developer" OR "fullstack engineer"'
        -> or(current_title.ilike.*fullstack developer*,current_title.ilike.*fullstack engineer*)

        '"full stack" AND (lead OR leader) NOT director'
        -> and(current_title.ilike.*full stack*,or(current_title.ilike.*lead*,current_title.ilike.*leader*),current_title.not.ilike.*director*)

    Args:
        query: Boolean query string
        column: Database column name to search

    Returns:
        PostgREST filter string (without the column= prefix)
    """
    # Tokenize the query
    tokens = _tokenize_boolean_query(query)

    # Parse tokens into AST
    ast = _parse_boolean_tokens(tokens)

    # Convert AST to PostgREST syntax
    return _ast_to_postgrest(ast, column)


def _tokenize_boolean_query(query: str) -> list:
    """Tokenize boolean query into list of tokens.

    Token types: 'PHRASE', 'WORD', 'AND', 'OR', 'NOT', 'LPAREN', 'RPAREN'
    """
    tokens = []
    i = 0
    query = query.strip()

    while i < len(query):
        # Skip whitespace
        if query[i].isspace():
            i += 1
            continue

        # Quoted phrase
        if query[i] == '"':
            end = query.find('"', i + 1)
            if end == -1:
                end = len(query)
            phrase = query[i+1:end]
            tokens.append(('PHRASE', phrase))
            i = end + 1
            continue

        # Parentheses
        if query[i] == '(':
            tokens.append(('LPAREN', '('))
            i += 1
            continue
        if query[i] == ')':
            tokens.append(('RPAREN', ')'))
            i += 1
            continue

        # Operators as symbols
        if query[i] == '|':
            tokens.append(('OR', 'OR'))
            i += 1
            continue
        if query[i] == '&':
            tokens.append(('AND', 'AND'))
            i += 1
            continue
        if query[i] == '!' or query[i] == '-':
            # Check if it's at start of word (NOT operator)
            if i == 0 or query[i-1].isspace() or query[i-1] in '(&|':
                tokens.append(('NOT', 'NOT'))
                i += 1
                continue
        if query[i] == '+':
            # Implicit AND (ignored, AND is default between terms)
            i += 1
            continue
        if query[i] == ',':
            # Comma as OR
            tokens.append(('OR', 'OR'))
            i += 1
            continue

        # Word (including operators as words)
        word_match = re.match(r'[\w\-]+', query[i:])
        if word_match:
            word = word_match.group()
            upper_word = word.upper()
            if upper_word == 'AND':
                tokens.append(('AND', 'AND'))
            elif upper_word == 'OR':
                tokens.append(('OR', 'OR'))
            elif upper_word == 'NOT':
                tokens.append(('NOT', 'NOT'))
            else:
                tokens.append(('WORD', word))
            i += len(word)
            continue

        # Unknown character, skip
        i += 1

    return tokens


def _parse_boolean_tokens(tokens: list) -> dict:
    """Parse tokens into an AST (Abstract Syntax Tree).

    Grammar (simplified):
        expr     -> or_expr
        or_expr  -> and_expr (OR and_expr)*
        and_expr -> not_expr (AND? not_expr)*
        not_expr -> NOT? primary
        primary  -> PHRASE | WORD | '(' expr ')'

    Returns AST as nested dicts:
        {'type': 'OR', 'children': [...]}
        {'type': 'AND', 'children': [...]}
        {'type': 'NOT', 'child': {...}}
        {'type': 'TERM', 'value': '...'}
    """
    pos = [0]  # Use list for mutable closure

    def peek():
        if pos[0] < len(tokens):
            return tokens[pos[0]]
        return None

    def consume():
        token = peek()
        pos[0] += 1
        return token

    def parse_or():
        left = parse_and()
        while peek() and peek()[0] == 'OR':
            consume()  # consume OR
            right = parse_and()
            if left.get('type') == 'OR':
                left['children'].append(right)
            else:
                left = {'type': 'OR', 'children': [left, right]}
        return left

    def parse_and():
        left = parse_not()
        while peek():
            token = peek()
            # Explicit AND
            if token[0] == 'AND':
                consume()
                right = parse_not()
            # Implicit AND (two terms next to each other)
            elif token[0] in ('WORD', 'PHRASE', 'NOT', 'LPAREN'):
                right = parse_not()
            else:
                break

            if left.get('type') == 'AND':
                left['children'].append(right)
            else:
                left = {'type': 'AND', 'children': [left, right]}
        return left

    def parse_not():
        if peek() and peek()[0] == 'NOT':
            consume()
            child = parse_primary()
            return {'type': 'NOT', 'child': child}
        return parse_primary()

    def parse_primary():
        token = peek()
        if not token:
            return {'type': 'TERM', 'value': ''}

        if token[0] == 'LPAREN':
            consume()  # consume (
            expr = parse_or()
            if peek() and peek()[0] == 'RPAREN':
                consume()  # consume )
            return expr

        if token[0] in ('WORD', 'PHRASE'):
            consume()
            return {'type': 'TERM', 'value': token[1]}

        # Unexpected token, skip
        consume()
        return parse_primary()

    if not tokens:
        return {'type': 'TERM', 'value': ''}

    return parse_or()


def _ast_to_postgrest(ast: dict, column: str) -> str:
    """Convert AST to PostgREST filter syntax.

    Args:
        ast: Parsed AST from _parse_boolean_tokens
        column: Database column name

    Returns:
        PostgREST filter string
    """
    node_type = ast.get('type')

    if node_type == 'TERM':
        value = ast.get('value', '').strip()
        if not value:
            return ''
        # Escape special characters in the value
        value = value.replace('%', r'\%').replace('_', r'\_')
        return f"{column}.ilike.*{value}*"

    if node_type == 'NOT':
        child_filter = _ast_to_postgrest(ast['child'], column)
        if not child_filter:
            return ''
        # PostgREST NOT syntax: column.not.operator.value
        # Convert "column.ilike.*val*" to "column.not.ilike.*val*"
        if '.ilike.' in child_filter:
            return child_filter.replace('.ilike.', '.not.ilike.')
        return f"not.{child_filter}"

    if node_type == 'OR':
        children = [_ast_to_postgrest(c, column) for c in ast['children']]
        children = [c for c in children if c]  # Remove empty
        if not children:
            return ''
        if len(children) == 1:
            return children[0]
        return f"or({','.join(children)})"

    if node_type == 'AND':
        children = [_ast_to_postgrest(c, column) for c in ast['children']]
        children = [c for c in children if c]  # Remove empty
        if not children:
            return ''
        if len(children) == 1:
            return children[0]
        return f"and({','.join(children)})"

    return ''


def search_profiles_boolean(client: SupabaseClient, filters: dict, limit: int = 5000) -> list:
    """Search profiles with full boolean query support.

    Supports complex boolean expressions in each field:
        - Quoted phrases: "fullstack developer"
        - OR: term1 OR term2, term1 | term2, term1, term2
        - AND: term1 AND term2, term1 & term2
        - NOT: NOT term, -term
        - Grouping: (term1 OR term2)

    Args:
        client: SupabaseClient instance
        filters: Dict with optional keys:
            - name: Boolean query for name
            - current_title: Boolean query for current title
            - current_company: Boolean query for current company
            - location: Boolean query for location
            - has_email: Boolean, filter for profiles with email
            - date_after: ISO date string, enriched_at >= date
            - date_before: ISO date string, enriched_at <= date
        limit: Maximum results (default 5000)

    Examples:
        # Search for fullstack variations
        search_profiles_boolean(client, {
            'current_title': '"fullstack developer" OR "fullstack engineer" OR "full stack"'
        })

        # Complex boolean search
        search_profiles_boolean(client, {
            'current_title': '"full stack" AND (lead OR leader) NOT director',
            'location': 'israel OR "tel aviv"'
        })

    Returns:
        List of profile dicts
    """
    params = {}
    and_conditions = []

    # Process string columns with boolean parser
    string_columns = {
        'name': 'name',
        'current_title': 'current_title',
        'current_company': 'current_company',
        'location': 'location'
    }

    for filter_key, column in string_columns.items():
        query = filters.get(filter_key)
        if query:
            postgrest_filter = parse_boolean_query(query, column)
            if postgrest_filter:
                and_conditions.append(postgrest_filter)

    # Boolean filters
    if filters.get('has_email'):
        params['email'] = 'not.is.null'

    # Date filters
    if filters.get('date_after') and filters.get('date_before'):
        and_conditions.append(f"enriched_at.gte.{filters['date_after']}")
        and_conditions.append(f"enriched_at.lte.{filters['date_before']}")
    elif filters.get('date_after'):
        params['enriched_at'] = f"gte.{filters['date_after']}"
    elif filters.get('date_before'):
        params['enriched_at'] = f"lte.{filters['date_before']}"

    # Combine all AND conditions
    if and_conditions:
        if len(and_conditions) == 1:
            # Single condition - check if it's already wrapped
            cond = and_conditions[0]
            if cond.startswith('or('):
                params['or'] = f"({cond[3:-1]})"  # Extract inner, wrap in parens
            elif cond.startswith('and('):
                params['and'] = f"({cond[4:-1]})"
            else:
                # Single term condition - add directly as column filter
                # Format: column.ilike.*value* -> column=ilike.*value*
                parts = cond.split('.', 1)
                if len(parts) == 2:
                    params[parts[0]] = parts[1]
        else:
            # Multiple conditions - wrap in and()
            params['and'] = f"({','.join(and_conditions)})"

    # Select without raw_data for performance
    columns = 'linkedin_url,name,current_title,current_company,all_employers,all_titles,all_schools,skills,location,email,enriched_at'

    return client.select('profiles', columns, params, limit=limit)


def search_profiles_filtered(client: SupabaseClient, filters: dict, limit: int = 5000) -> list:
    """Server-side filtered search for profiles.

    Within each filter: OR logic (e.g., "wiz, monday" matches wiz OR monday)
    Between filters: AND logic (e.g., company AND location must both match)

    Args:
        client: SupabaseClient instance
        filters: Dict with optional keys:
            - name: Search in name field
            - current_title: Search in current_title only
            - past_titles: Search in all_titles array
            - current_company: Search in current_company only
            - past_companies: Search in all_employers
            - location: Search in location
            - skills: Search in skills array
            - schools: Search in all_schools
            - has_email: Boolean, filter for profiles with email
            - date_after: ISO date string, enriched_at >= date
            - date_before: ISO date string, enriched_at <= date
        limit: Maximum results

    Returns:
        List of profile dicts
    """
    # Build params - each filter passed directly (implicitly ANDed by PostgREST)
    params = {}
    or_groups = []  # Collect OR groups for complex AND/OR combinations

    def add_string_filter(column: str, terms_str: str):
        """Add filter for a string column. Single term -> column param, multiple -> or()."""
        terms = [t.strip() for t in terms_str.split(',') if t.strip()]
        if len(terms) == 1:
            # Single term: direct column filter
            params[column] = f"ilike.*{terms[0]}*"
        else:
            # Multiple terms: create OR group
            # PostgREST nested syntax: or(col.ilike.*term1*,col.ilike.*term2*)
            conditions = ','.join([f"{column}.ilike.*{t}*" for t in terms])
            or_groups.append(f"or({conditions})")

    # String columns - direct ilike filter or OR conditions
    if filters.get('name'):
        add_string_filter('name', filters['name'])

    if filters.get('current_title'):
        add_string_filter('current_title', filters['current_title'])

    if filters.get('current_company'):
        add_string_filter('current_company', filters['current_company'])

    if filters.get('location'):
        add_string_filter('location', filters['location'])

    # Array columns - use 'ov' (overlaps) for OR matching (any of the terms)
    # Note: This requires exact term match within array, not partial match
    if filters.get('past_titles'):
        terms = [t.strip() for t in filters['past_titles'].split(',') if t.strip()]
        params['all_titles'] = f"ov.{{{','.join(terms)}}}"

    if filters.get('past_companies'):
        terms = [t.strip() for t in filters['past_companies'].split(',') if t.strip()]
        params['all_employers'] = f"ov.{{{','.join(terms)}}}"

    if filters.get('skills'):
        terms = [t.strip() for t in filters['skills'].split(',') if t.strip()]
        params['skills'] = f"ov.{{{','.join(terms)}}}"

    if filters.get('schools'):
        terms = [t.strip() for t in filters['schools'].split(',') if t.strip()]
        params['all_schools'] = f"ov.{{{','.join(terms)}}}"

    # Boolean and date filters
    if filters.get('has_email'):
        params['email'] = 'not.is.null'

    # Handle date range - add to and_conditions if both dates provided
    and_conditions = []
    if filters.get('date_after') and filters.get('date_before'):
        # Both dates: add as AND condition
        and_conditions.append(f"enriched_at.gte.{filters['date_after']}")
        and_conditions.append(f"enriched_at.lte.{filters['date_before']}")
    elif filters.get('date_after'):
        params['enriched_at'] = f"gte.{filters['date_after']}"
    elif filters.get('date_before'):
        params['enriched_at'] = f"lte.{filters['date_before']}"

    # Combine OR groups with AND logic between them
    # PostgREST: Multiple OR groups need and=(or(...),or(...)) to AND them together
    if or_groups:
        if len(or_groups) == 1 and not and_conditions:
            # Single OR group, no other AND conditions: use 'or' parameter directly
            inner = or_groups[0][3:-1]  # Remove 'or(' prefix and ')' suffix
            params['or'] = f"({inner})"
        else:
            # Multiple OR groups or mixed with AND conditions: combine in 'and'
            and_conditions.extend(or_groups)

    # Set final 'and' parameter if we have conditions
    if and_conditions:
        params['and'] = f"({','.join(and_conditions)})"

    # Select without raw_data for performance (large field)
    columns = 'linkedin_url,name,current_title,current_company,all_employers,all_titles,all_schools,skills,location,email,enriched_at'

    return client.select('profiles', columns, params, limit=limit)


# ============================================================================
# DEDUPLICATION (for skipping already-enriched URLs)
# ============================================================================

def get_enriched_urls(client: SupabaseClient) -> set:
    """Get all LinkedIn URLs that have been enriched.

    Used to skip profiles that are already in the database when enriching.
    """
    result = client.select('profiles', 'linkedin_url', limit=50000)
    urls = set()
    for p in result:
        url = p.get('linkedin_url')
        if url:
            urls.add(normalize_linkedin_url(url))
    return urls


def get_all_linkedin_urls(client: SupabaseClient) -> list:
    """Get all LinkedIn URLs from database."""
    result = client.select('profiles', 'linkedin_url', limit=50000)
    return [p['linkedin_url'] for p in result if p.get('linkedin_url')]


def get_recently_enriched_urls(client: SupabaseClient, months: int = 6) -> list:
    """Get LinkedIn URLs enriched within the last N months.
    Returns both linkedin_url and original_url for better matching.
    Uses pagination to bypass Supabase 1000 row limit."""
    cutoff_date = (datetime.utcnow() - timedelta(days=months * 30)).isoformat()

    # Paginate to get all results (Supabase has 1000 row server limit)
    all_results = []
    offset = 0
    page_size = 1000
    while True:
        filters = {'enriched_at': f'gte.{cutoff_date}', 'offset': str(offset)}
        result = client.select('profiles', 'linkedin_url,original_url', filters, limit=page_size)
        if not result:
            break
        all_results.extend(result)
        if len(result) < page_size:
            break
        offset += page_size

    urls = []
    for p in all_results:
        if p.get('linkedin_url'):
            urls.append(p['linkedin_url'])
        if p.get('original_url') and p.get('original_url') != p.get('linkedin_url'):
            urls.append(p['original_url'])
    return urls


def get_dedup_stats(client: SupabaseClient) -> dict:
    """Get stats about profiles in database for dedup preview."""
    total = client.count('profiles')
    cutoff_date = (datetime.utcnow() - timedelta(days=ENRICHMENT_REFRESH_MONTHS * 30)).isoformat()
    recently_enriched = client.count('profiles', {'enriched_at': f'gte.{cutoff_date}'})

    return {
        'total_profiles': total,
        'recently_enriched': recently_enriched,
        'will_skip': recently_enriched,
    }


# ============================================================================
# DATAFRAME CONVERSION
# ============================================================================

def profiles_to_dataframe(profiles: list) -> pd.DataFrame:
    """Convert list of profile dicts to DataFrame.

    Uses helpers.profiles_to_display_df to extract fields from raw_data.
    """
    from helpers import profiles_to_display_df
    return profiles_to_display_df(profiles)


def dataframe_to_profiles(df: pd.DataFrame) -> list:
    """Convert DataFrame back to list of profile dicts."""
    return df.to_dict('records')


# ============================================================================
# UTILITY
# ============================================================================

def check_connection(client: SupabaseClient) -> bool:
    """Check if Supabase connection is working."""
    if not client:
        return False
    try:
        client.select('profiles', 'linkedin_url', limit=1)
        return True
    except Exception:
        return False


# ============================================================================
# API USAGE TRACKING
# ============================================================================

def log_api_usage(client: SupabaseClient, data: dict) -> Optional[dict]:
    """Insert a usage record into api_usage_logs table."""
    try:
        result = client.insert('api_usage_logs', data)
        return result[0] if result else None
    except Exception as e:
        print(f"[DB] Failed to log API usage: {e}")
        return None


def get_usage_summary(client: SupabaseClient, days: int = None) -> dict:
    """Get aggregated usage stats by provider."""
    filters = {}
    if days:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        filters['created_at'] = f'gte.{cutoff}'

    try:
        logs = client.select('api_usage_logs', '*', filters, limit=10000)
    except Exception as e:
        print(f"[DB] Failed to fetch usage logs: {e}")
        return {}

    summary = {
        'crustdata': {'credits': 0, 'cost_usd': 0.0, 'requests': 0, 'errors': 0},
        'salesql': {'lookups': 0, 'requests': 0, 'errors': 0},
        'openai': {'cost_usd': 0.0, 'tokens_input': 0, 'tokens_output': 0, 'requests': 0, 'errors': 0},
        'phantombuster': {'runs': 0, 'profiles_scraped': 0, 'errors': 0},
    }

    for log in logs:
        provider = log.get('provider', '').lower()
        if provider not in summary:
            continue

        summary[provider]['requests'] = summary[provider].get('requests', 0) + (log.get('request_count') or 1)

        if log.get('status') == 'error':
            summary[provider]['errors'] = summary[provider].get('errors', 0) + 1

        if provider == 'crustdata':
            summary[provider]['credits'] += log.get('credits_used') or 0
            summary[provider]['cost_usd'] += log.get('cost_usd') or 0
        elif provider == 'salesql':
            summary[provider]['lookups'] += log.get('credits_used') or 0
        elif provider == 'openai':
            summary[provider]['cost_usd'] += log.get('cost_usd') or 0
            summary[provider]['tokens_input'] += log.get('tokens_input') or 0
            summary[provider]['tokens_output'] += log.get('tokens_output') or 0
        elif provider == 'phantombuster':
            summary[provider]['runs'] += 1
            metadata = log.get('metadata') or {}
            summary[provider]['profiles_scraped'] += metadata.get('profiles_scraped') or 0

    return summary


def get_usage_logs(client: SupabaseClient, provider: str = None, days: int = None,
                   limit: int = 100, offset: int = 0) -> list:
    """Get detailed usage logs with optional filtering."""
    filters = {}

    if provider:
        filters['provider'] = f'eq.{provider.lower()}'

    if days:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        filters['created_at'] = f'gte.{cutoff}'

    try:
        params = {'select': '*', 'order': 'created_at.desc'}
        if filters:
            params.update(filters)
        if limit:
            params['limit'] = limit
        if offset:
            params['offset'] = offset

        url = f"{client.url}/rest/v1/api_usage_logs"
        response = requests.get(url, headers=client.headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json() if response.text else []
    except Exception as e:
        print(f"[DB] Failed to fetch usage logs: {e}")
        return []


def get_usage_by_date(client: SupabaseClient, days: int = 30) -> list:
    """Get usage aggregated by date for charting."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    try:
        logs = client.select('api_usage_logs', '*', {'created_at': f'gte.{cutoff}'}, limit=10000)
    except Exception as e:
        print(f"[DB] Failed to fetch usage logs: {e}")
        return []

    by_date = {}
    for log in logs:
        created_at = log.get('created_at', '')
        if not created_at:
            continue
        date_str = created_at[:10]
        provider = log.get('provider', '').lower()

        if date_str not in by_date:
            by_date[date_str] = {'date': date_str, 'crustdata': 0, 'salesql': 0, 'openai': 0, 'phantombuster': 0}

        if provider == 'crustdata':
            by_date[date_str]['crustdata'] += log.get('credits_used') or 0
        elif provider == 'salesql':
            by_date[date_str]['salesql'] += log.get('credits_used') or 0
        elif provider == 'openai':
            by_date[date_str]['openai'] += log.get('cost_usd') or 0
        elif provider == 'phantombuster':
            by_date[date_str]['phantombuster'] += 1

    result = sorted(by_date.values(), key=lambda x: x['date'])
    return result


# ============================================================================
# SETTINGS (key-value store for shared config like system prompts)
# ============================================================================

def get_setting(client: SupabaseClient, key: str) -> Optional[str]:
    """Get a setting value by key from the settings table."""
    try:
        result = client.select('settings', 'value', {'key': f'eq.{key}'}, limit=1)
        if result:
            return result[0].get('value')
    except Exception as e:
        print(f"[DB] Failed to get setting '{key}': {e}")
    return None


def save_setting(client: SupabaseClient, key: str, value: str) -> bool:
    """Save a setting value. Creates or updates."""
    try:
        client.upsert('settings', {
            'key': key,
            'value': value,
            'updated_at': datetime.utcnow().isoformat(),
        }, on_conflict='key')
        return True
    except Exception as e:
        print(f"[DB] Failed to save setting '{key}': {e}")
        return False


# ============================================================================
# SCREENING PROMPTS
# ============================================================================

def get_screening_prompts(client: SupabaseClient) -> list:
    """Get all screening prompts from the database."""
    try:
        result = client.select('screening_prompts', '*', limit=100)
        return result if result else []
    except Exception as e:
        print(f"[DB] Failed to get screening prompts: {e}")
        return []


def get_screening_prompt_by_role(client: SupabaseClient, role_type: str) -> Optional[dict]:
    """Get a screening prompt by role type."""
    try:
        result = client.select('screening_prompts', '*', {'role_type': f'eq.{role_type}'}, limit=1)
        if result:
            return result[0]
    except Exception as e:
        print(f"[DB] Failed to get prompt for role '{role_type}': {e}")
    return None


def get_default_screening_prompt(client: SupabaseClient) -> Optional[dict]:
    """Get the default screening prompt (is_default=true)."""
    try:
        result = client.select('screening_prompts', '*', {'is_default': 'eq.true'}, limit=1)
        if result:
            return result[0]
    except Exception as e:
        print(f"[DB] Failed to get default prompt: {e}")
    return None


def save_screening_prompt(client: SupabaseClient, role_type: str, prompt_text: str,
                          keywords: list = None, is_default: bool = False, name: str = None) -> bool:
    """Save or update a screening prompt."""
    try:
        data = {
            'role_type': role_type.lower().strip(),
            'prompt_text': prompt_text,
            'keywords': keywords or [],
            'is_default': is_default,
            'name': name or role_type.title(),
            'updated_at': datetime.utcnow().isoformat(),
        }
        client.upsert('screening_prompts', data, on_conflict='role_type')
        return True
    except Exception as e:
        print(f"[DB] Failed to save prompt '{role_type}': {e}")
        return False


def delete_screening_prompt(client: SupabaseClient, role_type: str) -> bool:
    """Delete a screening prompt by role type."""
    try:
        client.delete('screening_prompts', {'role_type': role_type})
        return True
    except Exception as e:
        print(f"[DB] Failed to delete prompt '{role_type}': {e}")
        return False


def match_prompt_by_keywords(client: SupabaseClient, text: str) -> Optional[dict]:
    """Find the best matching prompt based on keywords in the text.

    Scans the job description text for keywords and returns the prompt
    with the most keyword matches. Leadership keywords get bonus points.
    """
    # Leadership keywords get extra weight to prioritize lead roles
    leadership_keywords = ['team lead', 'team leader', 'tech lead', 'tech leader',
                          'engineering lead', 'technical lead', 'lead engineer',
                          'engineering manager', 'manager', 'director', 'vp', 'head of']

    try:
        prompts = get_screening_prompts(client)
        if not prompts:
            return None

        text_lower = text.lower()
        best_match = None
        best_score = 0

        for prompt in prompts:
            keywords = prompt.get('keywords', [])
            if not keywords:
                continue

            # Score keyword matches with word-boundary matching
            score = 0
            for kw in keywords:
                kw_lower = kw.lower()
                is_leadership = any(lk in kw_lower for lk in leadership_keywords) or kw_lower in leadership_keywords

                if ' ' in kw_lower:
                    # Multi-word phrase: substring match
                    if kw_lower in text_lower:
                        score += 3 if is_leadership else 2
                else:
                    # Single word: word-boundary match to avoid false positives
                    if re.search(r'\b' + re.escape(kw_lower) + r'\b', text_lower):
                        score += 2 if is_leadership else 1

            if score > best_score:
                best_score = score
                best_match = prompt

        # Return match if score >= 2 (a phrase match, or 2+ single-word matches)
        if best_score >= 2:
            return best_match

        return None
    except Exception as e:
        print(f"[DB] Failed to match prompt: {e}")
        return None


# ============================================================================
# SEARCH HISTORY (PhantomBuster runs)
# ============================================================================

def get_search_history(client: SupabaseClient, agent_id: str = None) -> list:
    """Get search history from the search_history table.

    Args:
        client: SupabaseClient instance
        agent_id: If provided, filter history to this agent only

    Returns:
        List of search history entries, most recent first
    """
    try:
        params = {'select': '*', 'order': 'launched_at.desc'}
        if agent_id:
            params['agent_id'] = f'eq.{agent_id}'
        url = f"{client.url}/rest/v1/search_history"
        response = requests.get(url, headers=client.headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json() if response.text else []
    except Exception as e:
        print(f"[DB] Failed to get search history: {e}")
        return []


def save_search_history_entry(client: SupabaseClient, agent_id: str, csv_name: str,
                               search_url: str = None, profiles_requested: int = None,
                               search_name: str = None) -> bool:
    """Save a search entry to the search_history table.

    Args:
        client: SupabaseClient instance
        agent_id: PhantomBuster agent ID
        csv_name: Name of the output CSV file
        search_url: LinkedIn search URL
        profiles_requested: Number of profiles requested
        search_name: Optional human-readable name for the search

    Returns:
        True if saved successfully
    """
    try:
        data = {
            'agent_id': str(agent_id),
            'csv_name': csv_name,
            'launched_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M'),
            'profiles_requested': profiles_requested,
            'search_name': search_name,
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        client.insert('search_history', data)
        return True
    except Exception as e:
        print(f"[DB] Failed to save search history: {e}")
        return False


def delete_search_history_entry(client: SupabaseClient, agent_id: str, csv_name: str) -> bool:
    """Delete a search entry from the search_history table.

    Args:
        client: SupabaseClient instance
        agent_id: PhantomBuster agent ID
        csv_name: Name of the output CSV file to delete

    Returns:
        True if deleted successfully
    """
    try:
        client.delete('search_history', {'agent_id': str(agent_id), 'csv_name': csv_name})
        return True
    except Exception as e:
        print(f"[DB] Failed to delete search history entry: {e}")
        return False


# ============================================================================
# BACKWARDS COMPATIBILITY - Deprecated functions
# ============================================================================

def upsert_profiles_from_phantombuster(client: SupabaseClient, profiles: list, search_id: str = None) -> dict:
    """DEPRECATED: PhantomBuster data is no longer stored in DB.

    This function is kept for backwards compatibility but does nothing.
    PhantomBuster data should only be used in UI session state.
    """
    print("[DB] WARNING: upsert_profiles_from_phantombuster is deprecated. PB data is no longer stored in DB.")
    return {'inserted': 0, 'updated': 0, 'skipped': len(profiles), 'errors': 0, 'debug': ['DEPRECATED - PB data not stored']}
