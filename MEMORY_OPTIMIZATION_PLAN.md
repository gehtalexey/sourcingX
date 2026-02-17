# Memory Optimization Plan for Streamlit Cloud

## Problem
Streamlit Cloud has a **1GB RAM limit**. Our app exceeds this when handling large profile datasets.

## Root Causes Identified
1. **Duplicate data storage**: Same profiles stored as both list (`results`) AND DataFrame (`results_df`)
2. **filtered_out**: Storing full copies of filtered profiles
3. **Session persistence**: Serializing large data to JSON doubles memory during save
4. **No cache limits**: Unbounded caches grow indefinitely
5. **Heavy columns**: `raw_crustdata`, `raw_data` columns contain large JSON blobs

---

## Architecture Change: DB-First Approach

### Current (Memory-Heavy)
```
Load Profiles → Session State → Filter → Session State → Screen → Session State → Display
```

### Proposed (DB-First)
```
Load Profiles → DB → Query on Demand → Display (minimal session state)
```

---

## Implementation Plan

### Phase 1: Eliminate Duplicate Storage (HIGH PRIORITY)
**Goal**: Store data ONCE, derive views on demand

1. **Remove list/DataFrame duplication**
   - Keep ONLY `results_df` (DataFrame)
   - Create property/function to get list: `get_results_list()` that returns `results_df.to_dict('records')`
   - Remove: `results`, `enriched_results` from session state

2. **Remove redundant copies**
   - Remove: `passed_candidates_df` (same as `results_df` after filtering)
   - Remove: `original_results_df` (reload from DB if needed)
   - Remove: `enriched_df` (merge enrichment into `results_df`)

3. **Single source of truth**
   ```python
   # BEFORE: 4 copies of same data
   st.session_state['results'] = profiles          # list
   st.session_state['results_df'] = df             # DataFrame
   st.session_state['enriched_results'] = profiles # list (duplicate)
   st.session_state['enriched_df'] = df            # DataFrame (duplicate)

   # AFTER: 1 copy
   st.session_state['profiles_df'] = df  # Single DataFrame
   # Lists derived on demand: df.to_dict('records')
   ```

### Phase 2: Move Data to Database (HIGH PRIORITY)
**Goal**: Session state only stores IDs/references, not full data

1. **Store profiles in DB immediately on load**
   ```python
   def load_profiles(source):
       profiles = fetch_from_source(source)
       save_to_db(profiles)  # Store in Supabase
       # Only keep lightweight reference in session
       st.session_state['profile_urls'] = [p['linkedin_url'] for p in profiles]
       st.session_state['profile_count'] = len(profiles)
   ```

2. **Query profiles on demand for display**
   ```python
   @st.cache_data(ttl=300, max_entries=5)
   def get_profiles_for_display(page: int, page_size: int = 50):
       """Load only what's needed for current view."""
       return db.query_profiles(offset=page * page_size, limit=page_size)
   ```

3. **Store screening results in DB only**
   - Don't keep `screening_results` in session state
   - Query from DB when displaying results
   - Session only stores: `screening_complete: bool`, `screened_count: int`

4. **Filter results stored in DB**
   - Add `filter_status` column to profiles table
   - Values: 'active', 'filtered_past_candidate', 'filtered_blacklist', etc.
   - Query filtered profiles from DB instead of storing copies

### Phase 3: Implement Proper Caching (MEDIUM PRIORITY)
**Goal**: Bounded, efficient caches

1. **Add limits to all caches**
   ```python
   @st.cache_data(ttl=300, max_entries=10)  # 5 min TTL, max 10 entries
   def load_config():
       ...

   @st.cache_resource(ttl=300)  # 5 min TTL for DB client
   def get_db_client():
       ...
   ```

2. **Use session-scoped caches for user data**
   ```python
   @st.cache_data(ttl=600, max_entries=3, scope="session")
   def get_user_profiles(user_id, page):
       ...
   ```

3. **Clear caches aggressively**
   ```python
   def on_new_search():
       st.cache_data.clear()  # Clear all data caches
       cleanup_session_state()
   ```

### Phase 4: Optimize DataFrame Storage (MEDIUM PRIORITY)
**Goal**: Minimize DataFrame memory footprint

1. **Drop heavy columns immediately on load**
   ```python
   HEAVY_COLUMNS = ['raw_crustdata', 'raw_data', 'education_details',
                    'certifications', 'past_positions_raw']

   def load_profiles(source):
       df = pd.read_csv(source)
       df = df.drop(columns=[c for c in HEAVY_COLUMNS if c in df.columns])
       return df
   ```

2. **Optimize dtypes**
   ```python
   def optimize_df_memory(df):
       # Convert object columns to category where appropriate
       for col in df.select_dtypes(include=['object']).columns:
           if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique
               df[col] = df[col].astype('category')

       # Downcast numeric types
       for col in df.select_dtypes(include=['int64']).columns:
           df[col] = pd.to_numeric(df[col], downcast='integer')
       for col in df.select_dtypes(include=['float64']).columns:
           df[col] = pd.to_numeric(df[col], downcast='float')

       return df
   ```

3. **Paginate large displays**
   - Never load more than 100 rows into display at once
   - Use `st.dataframe` pagination

### Phase 5: Session State Minimization (MEDIUM PRIORITY)
**Goal**: Session state < 10MB

1. **Allowed in session state (lightweight)**
   ```python
   ALLOWED_SESSION_KEYS = {
       'current_page': int,
       'filter_settings': dict,  # Just settings, not data
       'screening_job_description': str,
       'profile_count': int,
       'last_action': str,
       'user_preferences': dict,
   }
   ```

2. **NOT allowed in session state (move to DB)**
   - Profile data (any list/DataFrame of profiles)
   - Screening results
   - Filtered candidates
   - Any list > 100 items

3. **Implement session state validator**
   ```python
   def validate_session_state():
       """Run on each rerun to prevent memory bloat."""
       for key in list(st.session_state.keys()):
           value = st.session_state[key]
           if isinstance(value, pd.DataFrame) and len(value) > 100:
               st.warning(f"Large DataFrame in session: {key}")
               del st.session_state[key]
           if isinstance(value, list) and len(value) > 100:
               st.warning(f"Large list in session: {key}")
               del st.session_state[key]
   ```

### Phase 6: Remove Session Persistence to JSON (LOW PRIORITY)
**Goal**: Don't serialize large data to files

1. **Remove local session file saving**
   - Session data should be in DB, not JSON files
   - JSON serialization doubles memory during save

2. **Keep only essential state in cookies/URL params**
   ```python
   # Use query params for shareable state
   st.query_params['view'] = 'screening'
   st.query_params['filter'] = 'strong_fit'
   ```

---

## Database Schema Changes

### New: `user_sessions` table
```sql
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    current_search_id UUID,  -- Reference to active search
    filter_settings JSONB,   -- Lightweight filter config
    last_active TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### New: `searches` table
```sql
CREATE TABLE searches (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    source TEXT,  -- 'phantombuster', 'csv_upload', etc.
    source_ref TEXT,  -- Agent ID or filename
    profile_count INT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Update: `profiles` table
```sql
ALTER TABLE profiles ADD COLUMN search_id UUID REFERENCES searches(id);
ALTER TABLE profiles ADD COLUMN filter_status TEXT DEFAULT 'active';
-- Values: 'active', 'filtered_past_candidate', 'filtered_blacklist', etc.
```

---

## Migration Steps

### Step 1: Immediate (Today)
- [x] Reduce load limits to 500
- [x] Add cleanup_memory() function
- [x] Don't store filtered_out DataFrames
- [x] Clear screening_batch_state after completion

### Step 2: This Week
- [ ] Remove `results` list (keep only `results_df`)
- [ ] Remove `enriched_results` list (keep only `enriched_df`)
- [ ] Merge `enriched_df` into `results_df` (single DataFrame)
- [ ] Add `max_entries` to all `@st.cache_data` decorators

### Step 3: Next Week
- [ ] Create `searches` table in DB
- [ ] Store profiles in DB on load (not session state)
- [ ] Implement paginated profile loading from DB
- [ ] Remove `results_df` from session state (query from DB)

### Step 4: Following Week
- [ ] Move screening results to DB-only
- [ ] Move filter status to DB column
- [ ] Remove session file persistence
- [ ] Implement session state validator

---

## Monitoring

### Add memory logging
```python
import sys

def log_session_memory():
    """Log session state memory usage."""
    total = 0
    for key, value in st.session_state.items():
        size = sys.getsizeof(value)
        if isinstance(value, pd.DataFrame):
            size = value.memory_usage(deep=True).sum()
        elif isinstance(value, list):
            size = sum(sys.getsizeof(item) for item in value)
        print(f"[Memory] {key}: {size / 1024 / 1024:.2f} MB")
        total += size
    print(f"[Memory] Total session state: {total / 1024 / 1024:.2f} MB")
```

### Target metrics
- Session state: < 10 MB
- Total app memory: < 500 MB (50% of limit)
- Profile load: < 500 at a time
- Cache entries: < 20 total

---

## Sources
- [Streamlit Resource Limits](https://docs.streamlit.io/knowledge-base/deploy/resource-limits)
- [st.cache_data Documentation](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)
- [Session State Documentation](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)
- [Streamlit Forum - Memory Limits](https://discuss.streamlit.io/t/hitting-memory-limit-of-streamlit-app/68389)
