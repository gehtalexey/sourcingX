# Memory Optimization Skill for Streamlit + Supabase Apps

Use this skill when debugging memory issues, optimizing Streamlit apps for Community Cloud, or moving operations server-side to Supabase.

---

## Streamlit Memory Model

### How Memory Works

1. **Script re-execution**: Streamlit re-runs the entire Python script on every user interaction
2. **Session state**: Each user gets isolated `st.session_state` (per-user memory cost)
3. **Shared cache**: `@st.cache_data` and `@st.cache_resource` are shared across ALL users
4. **No automatic cleanup**: Objects persist until explicitly deleted or TTL expires

### Streamlit Community Cloud Limits

| Resource | Limit |
|----------|-------|
| Memory | ~1GB per app (hard limit) |
| Recommended | Stay under 800MB to avoid throttling |
| Concurrent users | 2-3 on free tier practical limit |
| CPU | Shared, can be throttled |

### Memory Formula

```
Total Memory = (Users × Session State Size) + Cache Size + Base App
```

Example for 3 users with 50MB session state each:
- 50MB × 3 users = 150MB session state
- Shared cache = ~400MB
- Base app = ~100MB
- **Total: ~650MB**

---

## Session State vs Caching

| Aspect | `session_state` | `@st.cache_data` | `@st.cache_resource` |
|--------|-----------------|------------------|----------------------|
| Scope | Per-user | All users (shared) | All users (shared) |
| Returns | Same object | **Copy** each call | Same object (reference) |
| Use for | User-specific state, form inputs | DataFrames, API results, computed data | DB connections, ML models, clients |
| Memory | Per-user cost (expensive) | Shared (efficient) | Shared (most efficient) |
| Mutation safe | Yes | Yes (returns copy) | **No** (shared reference) |

### Critical Cache Parameters

```python
@st.cache_data(
    ttl=3600,           # Expire after 1 hour (REQUIRED for Cloud)
    max_entries=100,    # Cap cached results (REQUIRED for Cloud)
    show_spinner=False, # Optional: hide spinner
    persist="disk"      # Optional: survive app restarts
)
def load_data():
    return expensive_operation()
```

### Cache Resource (for connections)

```python
@st.cache_resource(ttl=300)  # 5 min TTL for connections
def get_supabase_client():
    return SupabaseClient(url, key)
```

---

## DataFrame Memory Optimization

### 1. Downcast Data Types (50-90% reduction)

```python
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce DataFrame memory by downcasting types."""
    for col in df.columns:
        col_type = df[col].dtype

        if col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'object':
            # Use category for low-cardinality string columns
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')

    return df
```

### 2. Load Only Needed Columns

```python
# Bad - loads everything
df = pd.read_csv('large_file.csv')

# Good - loads only needed columns
df = pd.read_csv('large_file.csv', usecols=['name', 'email', 'company'])
```

### 3. Avoid Unnecessary `.copy()` Calls

```python
# Bad - creates full duplicate
filtered_df = df[df['status'] == 'active'].copy()

# Good - use view when only reading
filtered_df = df[df['status'] == 'active']

# Only use .copy() when you NEED to modify the result
# and must preserve the original
```

### 4. Don't Store Same Data Multiple Times

```python
# Bad - 3x memory usage
st.session_state['results_df'] = df
st.session_state['passed_candidates_df'] = df.copy()
st.session_state['original_results_df'] = df.copy()

# Good - store once, derive others
st.session_state['results_df'] = df
# Only store original if filters will be applied
if 'original_results_df' not in st.session_state:
    st.session_state['original_results_df'] = df.copy()
# Don't store passed_candidates_df - derive from results_df
```

---

## Memory Cleanup Pattern

```python
import gc

def cleanup_memory():
    """Aggressive memory cleanup - call after major operations."""

    # Remove heavy columns from DataFrames
    heavy_cols = ['raw_data', 'raw_json', 'embeddings']
    for df_key in ['results_df', 'enriched_df']:
        if df_key in st.session_state:
            df = st.session_state[df_key]
            if isinstance(df, pd.DataFrame):
                cols_to_drop = [c for c in heavy_cols if c in df.columns]
                if cols_to_drop:
                    st.session_state[df_key] = df.drop(columns=cols_to_drop)

    # Clear temporary keys
    temp_keys = ['_debug', '_temp', 'processing_batch']
    for key in list(st.session_state.keys()):
        if any(key.startswith(prefix) for prefix in temp_keys):
            del st.session_state[key]

    # Force garbage collection
    gc.collect()
```

### When to Call Cleanup

```python
# After screening batch completes
if screening_complete:
    cleanup_memory()

# After export/download
if download_clicked:
    cleanup_memory()

# On tab change (for heavy tabs)
if current_tab != previous_tab:
    cleanup_memory()
```

---

## Supabase Server-Side Operations

### Why Move Operations to Supabase?

| Client-Side (Python) | Server-Side (PostgreSQL) |
|---------------------|-------------------------|
| Loads all data into memory | Returns only filtered results |
| Network transfer of full dataset | Minimal network transfer |
| Python loop = slow | SQL with indexes = fast |
| Memory scales with data size | Memory stays constant |

### PostgREST Filter Operators

Use these in your Supabase queries to filter server-side:

```python
# Equal
params['status'] = 'eq.active'

# Not equal
params['status'] = 'neq.archived'

# Greater/less than
params['score'] = 'gte.7'
params['created_at'] = 'lt.2025-01-01'

# Pattern matching (case-insensitive)
params['name'] = 'ilike.*john*'

# Array contains
params['skills'] = 'cs.{python,sql}'

# Array overlaps (ANY match)
params['all_employers'] = 'ov.{Google,Meta,Microsoft}'

# IN list
params['status'] = 'in.(active,pending)'

# IS NULL / NOT NULL
params['email'] = 'not.is.null'

# OR conditions
params['or'] = '(status.eq.active,score.gte.8)'

# AND conditions (default - just add multiple params)
params['status'] = 'eq.active'
params['score'] = 'gte.7'
```

### Creating PostgreSQL Functions

```sql
-- Function to match profiles against company list
CREATE OR REPLACE FUNCTION match_profiles_by_companies(
    target_companies TEXT[],
    p_limit INT DEFAULT 1000
)
RETURNS TABLE (
    linkedin_url TEXT,
    name TEXT,
    current_company TEXT,
    match_count INT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.linkedin_url,
        p.name,
        p.current_company,
        (SELECT COUNT(*)::INT FROM unnest(p.all_employers) e
         WHERE lower(e) = ANY(SELECT lower(c) FROM unnest(target_companies) c))
    FROM profiles p
    WHERE p.all_employers && target_companies  -- && = overlap operator
    ORDER BY match_count DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;
```

### Calling RPC Functions from Python

```python
def match_by_companies(client, companies: list, limit: int = 1000) -> list:
    """Find profiles matching target companies (server-side)."""
    response = requests.post(
        f"{client.url}/rest/v1/rpc/match_profiles_by_companies",
        headers=client.headers,
        json={
            'target_companies': companies,
            'p_limit': limit
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()
```

### Materialized Views for Aggregations

```sql
-- Pre-computed dashboard statistics
CREATE MATERIALIZED VIEW mv_pipeline_stats AS
SELECT
    COUNT(*) as total_profiles,
    COUNT(*) FILTER (WHERE status = 'screened') as screened,
    COUNT(*) FILTER (WHERE screening_fit_level = 'Strong Fit') as strong_fit,
    AVG(screening_score) as avg_score,
    NOW() as refreshed_at
FROM profiles;

-- Refresh on demand
REFRESH MATERIALIZED VIEW mv_pipeline_stats;

-- Query from Python (instant, no computation)
stats = client.select('mv_pipeline_stats', '*', limit=1)
```

### Cursor-Based Pagination

```python
def get_profiles_cursor(client, cursor=None, limit=100):
    """Efficient pagination for large datasets."""
    params = {
        'limit': limit,
        'order': 'enriched_at.desc,id.desc'
    }

    if cursor:
        enriched_at, profile_id = cursor.split('::')
        # Get records AFTER cursor position
        params['or'] = f"(enriched_at.lt.{enriched_at}," \
                       f"(enriched_at.eq.{enriched_at},id.lt.{profile_id}))"

    result = client.select('profiles', 'linkedin_url,name,current_company', params)

    if result:
        last = result[-1]
        next_cursor = f"{last['enriched_at']}::{last['id']}"
    else:
        next_cursor = None

    return result, next_cursor
```

---

## Common Memory Pitfalls

### 1. Loading Data on Every Rerun

```python
# Bad - loads on every interaction
df = pd.read_csv('large_file.csv')

# Good - cache it
@st.cache_data(ttl=3600, max_entries=5)
def load_data():
    return pd.read_csv('large_file.csv')

df = load_data()
```

### 2. Unbounded Cache Growth

```python
# Bad - cache grows forever
@st.cache_data
def fetch_profile(url):
    return api.get(url)

# Good - limit cache size
@st.cache_data(max_entries=1000, ttl=3600)
def fetch_profile(url):
    return api.get(url)
```

### 3. Storing Large Objects in Session State

```python
# Bad - raw_data is 20-50KB per profile
st.session_state['profiles'] = profiles_with_raw_data

# Good - exclude heavy fields
st.session_state['profiles'] = [
    {k: v for k, v in p.items() if k != 'raw_data'}
    for p in profiles
]
```

### 4. Not Cleaning Up After Operations

```python
# Bad - screening results accumulate
results = screen_batch(profiles)
st.session_state['screening_results'].extend(results)

# Good - clean up after export
if export_clicked:
    download_results()
    del st.session_state['screening_results']
    gc.collect()
```

### 5. Multiple Copies of Same DataFrame

```python
# Bad - creating unnecessary copies
view_df = df.copy()
display_df = view_df.copy()
export_df = display_df.copy()

# Good - use views or single copy
view_df = df[display_columns]  # View, not copy
export_df = df.copy() if need_modification else df
```

---

## Quick Checklist for Memory Issues

### Immediate Fixes
- [ ] Add `max_entries` to ALL `@st.cache_data` decorators
- [ ] Add `ttl` to all cache decorators
- [ ] Remove duplicate DataFrame storage in session_state
- [ ] Remove unnecessary `.copy()` calls
- [ ] Exclude `raw_data` / heavy columns when not needed

### Medium-Term Fixes
- [ ] Move filtering operations to Supabase (server-side)
- [ ] Implement cursor-based pagination for large datasets
- [ ] Create materialized views for dashboard stats
- [ ] Add automatic `cleanup_memory()` calls after major operations

### Architecture Fixes
- [ ] Load raw_data on-demand only during screening
- [ ] Use lightweight profile views by default
- [ ] Store screening results in Supabase instead of session_state

---

## Memory Budget Example (SourcingX)

For 2000 profiles on Streamlit Cloud (1GB limit):

| Component | Before | After Optimization |
|-----------|--------|-------------------|
| Session state (3 DataFrames) | 150 MB | 50 MB (single DF) |
| Raw data in memory | 100 MB | 20 MB (on-demand) |
| Unnecessary copies | 30 MB | 10 MB |
| Server-side filtering | 50 MB | 5 MB |
| **Total** | **330 MB** | **85 MB** |

**Savings: 74%**
