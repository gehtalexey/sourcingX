# Supabase Patterns for SourcingX

## Overview
This skill covers Supabase-specific patterns, best practices, and lessons learned from the SourcingX project. Use this when working with Supabase REST API, migrations, transactions, or optimizing database operations.

## Key Files
- `db.py` - Custom REST API client (~1,100 lines)
- `db_migrations.py` - Migration management with checksums (~250 lines)
- `db_transactions.py` - Application-level transaction patterns (~340 lines)
- `supabase_setup.sql` - Full database schema

## REST API Client Patterns

### Custom Client (Not Official SDK)
SourcingX uses a custom REST client instead of the official Supabase Python SDK for better control:

```python
class SupabaseClient:
    def __init__(self, url: str, key: str):
        self.url = url.rstrip('/')
        self.headers = {
            'apikey': key,
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            'Prefer': 'return=representation'
        }
```

### Pagination (1000-Row Limit)
Supabase REST API has a 1000-row server limit. Handle with built-in pagination:

```python
def select(self, table: str, columns: str = '*', filters: dict = None, limit: int = 5000):
    PAGE_SIZE = 1000
    all_results = []
    offset = 0
    while offset < limit:
        page_params = {**params, 'limit': min(PAGE_SIZE, limit - offset), 'offset': offset}
        page = self._request('GET', table, params=page_params)
        if not page:
            break
        all_results.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return all_results
```

**IMPORTANT**: Never pass `offset` as a filter - it must be a query parameter.

### JSONB Handling
When upserting JSONB data, handle NaN/Infinity values:

```python
def clean_for_json(obj):
    """Replace NaN/Infinity with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    return obj
```

## Migration Management

### Checksum Verification
Track migrations with checksums to detect unauthorized changes:

```python
def compute_checksum(sql_content: str) -> str:
    """Compute SHA-256 checksum of migration content."""
    normalized = sql_content.strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]
```

### Migration Table Schema
```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    checksum VARCHAR(64),
    applied_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Transaction Simulation

Supabase REST API doesn't support true transactions. Use application-level patterns:

```python
class BatchOperation:
    """Track operations for potential rollback."""
    def __init__(self, client):
        self.client = client
        self.operations = []
        self.committed = False

    def add_operation(self, table, operation, data, rollback_data=None):
        self.operations.append({
            'table': table, 'operation': operation,
            'data': data, 'rollback_data': rollback_data
        })

    def rollback(self):
        """Best-effort rollback - may leave partial state."""
        for op in reversed(self.operations):
            if op['rollback_data']:
                # Attempt to restore previous state
                self.client.upsert(op['table'], op['rollback_data'])
```

## RPC Functions

### Server-Side URL Matching
For large-scale profile matching, use PostgreSQL functions:

```sql
CREATE OR REPLACE FUNCTION match_profiles_by_urls_exact(
    input_urls TEXT[],
    enriched_after TIMESTAMPTZ DEFAULT NULL
) RETURNS TABLE (...) AS $$
-- Server-side matching is 10x faster than client-side for 1000+ URLs
```

Call via REST:
```python
response = requests.post(
    f"{client.url}/rest/v1/rpc/match_profiles_by_urls_exact",
    headers=client.headers,
    json={"input_urls": urls, "enriched_after": cutoff}
)
```

## Cost Optimization

### Egress Costs
Supabase charges for data egress. For large JSONB fields (20-50KB per profile):

1. **Select only needed columns**: `select('profiles', 'linkedin_url,name')` not `select('profiles', '*')`
2. **Use RPC to filter server-side**: Return only matching rows, not all data
3. **Trim large fields**: Remove logos, long descriptions before storing

### Pro Plan Triggers
- Free tier: 2GB egress/month
- At ~50K profiles with raw_data, expect $25-50/month on Pro plan

## Common Pitfalls

### 1. Offset as Filter (BUG)
```python
# WRONG - offset is treated as a column filter
filters = {'enriched_at': f'gte.{date}', 'offset': str(offset)}

# CORRECT - offset as query parameter
params = {'offset': offset, 'limit': page_size}
```

### 2. Index-Based Matching (DATA CORRUPTION)
Never assume API results preserve input order:
```python
# WRONG - causes data corruption
if len(results) == len(inputs):
    for i, result in enumerate(results):
        result['original_input'] = inputs[i]  # ORDER NOT GUARANTEED!

# CORRECT - match by unique identifier
for result in results:
    url = result.get('linkedin_url')
    if url in input_url_map:
        result['original_input'] = input_url_map[url]
```

### 3. Corrupted original_url Data
~10% of profiles may have `original_url` pointing to wrong person due to past index-based matching bug. Always validate:
```python
def is_original_url_valid(linkedin_url, original_url):
    """Check if original_url matches linkedin_url (same person)."""
    linkedin_base = get_base_username(linkedin_url)
    original_base = get_base_username(original_url)
    return linkedin_base == original_base
```

## Schema Patterns

### Profiles Table
```sql
CREATE TABLE profiles (
    linkedin_url TEXT PRIMARY KEY,
    original_url TEXT,  -- Input URL (may differ from linkedin_url)
    raw_data JSONB,     -- Full Crustdata response (20-50KB)
    name TEXT,
    current_title TEXT,
    current_company TEXT,
    all_employers TEXT[],  -- GIN indexed for array search
    skills TEXT[],         -- GIN indexed
    enriched_at TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'enriched'
);

-- Important indexes
CREATE INDEX idx_profiles_enriched_at ON profiles(enriched_at);
CREATE INDEX idx_profiles_all_employers ON profiles USING GIN(all_employers);
```

### Views for Pipeline Stats
```sql
CREATE VIEW pipeline_stats AS
SELECT
    COUNT(*) FILTER (WHERE status = 'enriched') as pending,
    COUNT(*) FILTER (WHERE status = 'screened') as screened,
    COUNT(*) FILTER (WHERE screening_fit_level = 'strong') as strong_fits
FROM profiles;
```

## Testing Patterns

### Mock Supabase Client
```python
class MockSupabaseClient:
    def __init__(self):
        self.data = {}

    def select(self, table, columns='*', filters=None, limit=1000):
        return list(self.data.get(table, {}).values())[:limit]

    def upsert(self, table, data, on_conflict='id'):
        if table not in self.data:
            self.data[table] = {}
        key = data.get(on_conflict)
        self.data[table][key] = data
        return [data]
```

## Debugging

### Check for Data Corruption
```sql
-- Find profiles with mismatched original_url
SELECT linkedin_url, original_url, name
FROM profiles
WHERE original_url IS NOT NULL
  AND original_url != linkedin_url
  AND SPLIT_PART(linkedin_url, '/in/', 2) NOT LIKE
      '%' || SPLIT_PART(original_url, '/in/', 2) || '%';
```

### Verify Pagination
```python
# Always verify you got all expected rows
result = client.select('profiles', 'linkedin_url', limit=50000)
actual_count = client.count('profiles')
assert len(result) == actual_count, f"Pagination issue: got {len(result)}, expected {actual_count}"
```
