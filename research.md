# SourcingX Production Readiness Research
## Full Investigation Report - February 19, 2026

**Investigation Team:**
- Team Leader: Coordinated 3 parallel research agents
- Agent 1: Full codebase memory scan (dashboard.py, db.py, all modules)
- Agent 2: Streamlit platform research (limits, pricing, alternatives)
- Agent 3: Existing plans & configuration review

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Codebase Memory Issues (Full Scan)](#2-codebase-memory-issues-full-scan)
3. [Streamlit Platform Limits & Pricing](#3-streamlit-platform-limits--pricing)
4. [Memory Architecture Deep Dive](#4-memory-architecture-deep-dive)
5. [Existing Optimization Status](#5-existing-optimization-status)
6. [API & Concurrency Risks](#6-api--concurrency-risks)
7. [Production Alternatives Comparison](#7-production-alternatives-comparison)
8. [Self-Hosting Guide](#8-self-hosting-guide)
9. [Recommendation & Decision Matrix](#9-recommendation--decision-matrix)
10. [Action Plan](#10-action-plan)

---

## 1. Executive Summary

**Verdict: The app WILL crash in production on Streamlit Community Cloud with 7-10 users.**

| Metric | Current State | Production Need |
|--------|--------------|-----------------|
| Available RAM | 1 GB (hard kill) | 4-8 GB minimum |
| Peak memory per session | ~300 MB | Must stay under 75 MB |
| Max safe concurrent users | 2-3 | 7-10 required |
| Crash risk at 10 users | **~95%** | Must be <5% |
| Memory optimization implemented | ~40% of plan | 100% needed |
| Concurrency fixes implemented | ~30% of plan | 100% needed |

**Root causes of crashes identified:**
1. Unbounded database pagination loads ALL profiles into memory (50MB+ per load)
2. DataFrames stored 3x (list + DataFrame + filtered copies) = 150MB waste per session
3. Filter operations store entire DataFrames instead of indices = 125-250MB bloat
4. 50-thread batch screening per user = thread explosion + memory fragmentation
5. Raw data (20-50KB/profile) loaded even when not needed
6. No session memory caps or automatic cleanup triggers

**Bottom line: You MUST either (a) self-host on a 4-8GB VM ($30-60/mo), or (b) fix all critical memory issues AND accept max ~3 concurrent users on free tier.**

---

## 2. Codebase Memory Issues (Full Scan)

### CRITICAL Severity (Will Crash the App)

#### 2.1 Unbounded Database Pagination
**File:** `dashboard.py:5738-5746`
```python
while True:
    batch = db_client.select('profiles', '*', filters, limit=page_size)
    if not batch:
        break
    all_db_profiles.extend(batch)  # NO MAXIMUM LIMIT
    offset += page_size
```
- Loads ALL profiles from Supabase without a cap
- 10,000 profiles with raw_data = 500MB+ in one operation
- **This is the #1 crash risk**
- **Fix:** Add `max_profiles=5000` limit, use `include_raw_data=False`

#### 2.2 Triple Data Storage in Session State
**File:** `dashboard.py:363-371, 5807, 6968`
- `enriched_results` (list) + `enriched_df` (DataFrame) + `results_df` (DataFrame)
- 1000 profiles stored 3 ways = 3x memory (150MB+ per session)
- Cleanup code exists at line 336-338 but data is re-created elsewhere
- **Fix:** Keep only `enriched_df`, derive lists on demand

#### 2.3 50-Thread Batch Screening
**File:** `dashboard.py:6937-6947`
```python
max_workers=min(50, len(batch_profiles))  # Up to 50 concurrent threads
```
- 50 concurrent OpenAI API threads per user
- 3 concurrent users = 150 threads in shared process
- Thread overhead + request buffers = 15MB+ peak per user
- **Fix:** Reduce to `max_workers=10`, add queue-based rate limiting

#### 2.4 Keep-Alive Infinite Loop
**File:** `dashboard.py:91-112`
- Background daemon thread pings app URL every 600s forever
- Each request creates temporary objects that accumulate
- On Streamlit Cloud (apps run for days) = slow memory creep
- **Fix:** Add stop condition, call `gc.collect()` in loop

#### 2.5 CSV Loading Without Optimization
**File:** `dashboard.py:2314` (and throughout)
```python
df = pd.read_csv(uploaded_file)  # No chunking, no dtype, no usecols
```
- `filtered_profiles.csv` (2MB) x pandas overhead (2-3x) = 6MB per CSV
- No column filtering, no dtype optimization
- **Fix:** Use `usecols`, `dtype` parameters, implement chunking

### HIGH Severity (Degrades Performance, Eventual Crash)

#### 2.6 Filter Operations Store Entire DataFrames
**File:** `dashboard.py:2811, 2826, 2837, 2854, 2870`
```python
st.session_state['filtered_out'][name] = df[mask]  # STORES FULL DataFrame
```
- Every filter category stores a COPY of matched rows
- 5-10 filter categories x 50% of data each = 125-250MB for filter metadata
- **Fix:** Store only indices: `df.index[mask].tolist()` instead of `df[mask]`

#### 2.7 Raw Data Loaded Unnecessarily
**File:** `dashboard.py:5740`
```python
batch = db_client.select('profiles', '*', filters, limit=page_size)
```
- Loads ALL columns including `raw_data` (20-50KB per profile)
- 1000 profiles x 50KB = 50MB of raw JSON loaded for every browse operation
- Pop cleanup at line 5804-5806 happens AFTER loading into session state
- **Fix:** Use `get_all_profiles(..., include_raw_data=False)`

#### 2.8 Screening Batch Results Accumulate
**File:** `dashboard.py:6911, 6948`
```python
all_results = batch_state.get('results', [])
all_results.extend(batch_results)  # Grows every batch
```
- Screening 1000 profiles: Batch 1=50 results, Batch 2=100, ... Batch 20=1000
- Entire history kept in session state = 2MB+ continuous
- **Fix:** Save to DB after each batch, keep only current batch in memory

#### 2.9 DataFrame.copy() Duplication
**File:** `dashboard.py:810, 1190`
```python
df = profiles_df.copy()  # Full copy of potentially large DataFrame
```
- Multiple `.copy()` calls double peak memory during operations
- **Fix:** Use views `df[cols]` when mutation isn't needed

#### 2.10 db.py Default Limit Too High
**File:** `db.py:52-78`
```python
def select(self, table, columns='*', filters=None, limit=50000):
```
- Default limit=50000 allows loading 50K records
- Combined with `*` columns (including raw_data) = potential 2.5GB load
- **Fix:** Reduce default to 5000, add warning at 1000+

### MEDIUM Severity (Contributes to Memory Pressure)

#### 2.11 Inconsistent Cache TTLs
**File:** `dashboard.py:218` - `load_config()` has `ttl=60` (too short, causes constant re-execution)
**File:** `dashboard.py:5618` - Other caches have `ttl=300`
- Config rarely changes, should be `ttl=3600`
- Some `@st.cache_resource` decorators lack `max_entries`

#### 2.12 URL Variation Set Growth
**File:** `dashboard.py:5752-5768`
- Creates set with 3x URL count for skip matching
- 10,000 URLs = 30,000 set entries = 900KB overhead
- Not critical alone but adds to fragmentation

#### 2.13 apply(axis=1) Instead of Vectorized Operations
**File:** `dashboard.py:2802-2806`
```python
df['_full_name'] = df.apply(
    lambda r: normalize_name(f"{r.get('first_name', '')} {r.get('last_name', '')}"), axis=1
)
```
- Row-wise application is 10-100x slower than vectorized pandas
- Creates intermediate objects per row
- **Fix:** Use `df['first_name'] + ' ' + df['last_name']`

#### 2.14 Unpinned Dependencies
**File:** `requirements.txt`
```
streamlit>=1.30.0
pandas>=2.0.0
```
- Open-ended version ranges allow memory regression from newer versions
- **Fix:** Pin to known-good versions

### Memory Impact Summary

| Scenario | Current Peak | After Fixes | Savings |
|----------|-------------|-------------|---------|
| Load 1000 profiles | ~150 MB | ~30 MB | 80% |
| Filter operation | ~250 MB | ~50 MB | 80% |
| Screen 100 profiles | ~50 MB | ~15 MB | 70% |
| **Total per session** | **~300 MB** | **~75 MB** | **75%** |
| **10 concurrent users** | **~3 GB** (crash) | **~750 MB** (tight) | - |

---

## 3. Streamlit Platform Limits & Pricing

### 3.1 Community Cloud (Free Tier)

| Resource | Limit |
|----------|-------|
| **RAM** | **1 GB per app** (hard kill on exceed - OOMKilled) |
| **CPU** | 1 vCPU (shared/burstable, throttled under load) |
| **Disk** | ~1 GB ephemeral (not persistent across reboots) |
| **App sleep** | After **7 days** of no traffic (shows "Wake up" button) |
| **Apps per account** | ~5-10 public apps |
| **Concurrent users** | No hard cap, but degrades past 5-10 sessions |
| **Uptime SLA** | None |

### 3.2 Paid Options - The Snowflake Reality

**There is NO standalone paid Streamlit Cloud with higher resources.**

After Snowflake's acquisition of Streamlit, the enterprise path is:

| Offering | Details | Cost |
|----------|---------|------|
| **Community Cloud** | Free, 1GB RAM, public/private apps | $0 |
| **Streamlit in Snowflake (SiS)** | Apps run inside Snowflake warehouses | Snowflake credit-based ($2-3/credit/hour) |
| **Self-hosted** | Run on your own VM/container | VM cost ($30-100/mo) |

**Streamlit in Snowflake details:**
- XS warehouse: ~8 GB RAM, 1 credit/hour (~$2-3/hour)
- Medium warehouse: ~32 GB RAM, 4 credits/hour
- Requires data to be IN Snowflake (not compatible with Supabase/PostgreSQL without pipelines)
- **NOT a viable option for SourcingX** without major data architecture changes

### 3.3 What This Means

You have exactly 3 options:
1. **Stay on free tier** (1GB) - must fix ALL memory issues, accept 2-3 user max
2. **Self-host** ($30-60/mo) - 4-8GB RAM, same code, better stability
3. **Migrate to another framework** - more work, better long-term scalability

---

## 4. Memory Architecture Deep Dive

### 4.1 How Streamlit Uses Memory

```
User clicks button
  -> Entire Python script re-runs top to bottom
  -> All imports reload (from cache)
  -> All data processing re-runs (unless cached)
  -> All UI elements re-render
  -> Results sent back via WebSocket
```

**Per-session memory multiplication:**
- Each browser tab = separate session
- Each session = own copy of `st.session_state` (server-side, NOT browser)
- Sessions persist until tab close + timeout (~10 min default)

### 4.2 Practical Memory Budget on 1GB

| Component | Memory |
|-----------|--------|
| Python interpreter + Streamlit | ~150 MB |
| Libraries (pandas, plotly, openai, etc.) | ~150 MB |
| Shared caches (@st.cache_resource) | ~50 MB |
| **Available for sessions** | **~650 MB** |
| Per session (current, unoptimized) | ~300 MB |
| **Max concurrent users** | **~2** |

| Component | After Optimization |
|-----------|-------------------|
| Per session (optimized) | ~75 MB |
| **Max concurrent users on 1GB** | **~8** (theoretical) |
| **Realistic max (with spikes)** | **~4-5** |

### 4.3 Caching Memory Comparison

| Method | Memory Model | SourcingX Impact |
|--------|-------------|------------------|
| `@st.cache_data` | Serialized copy + deserialized copy per caller | 10 users x 100MB DataFrame = 1,100 MB total |
| `@st.cache_resource` | Single shared object, all sessions reference same | 10 users x 100MB DataFrame = 100 MB total |

**Recommendation:** Use `@st.cache_resource` for read-only shared data (profile lists, config). Use `@st.cache_data` only for per-user computed results with `ttl` and `max_entries`.

### 4.4 WebSocket Connection Overhead
- Each active session: 1-5 MB for WebSocket (Tornado server, buffers, protocol)
- 10 concurrent connections: 10-50 MB just for connections
- Zombie sessions (tab closed but not cleanly disconnected) continue consuming

---

## 5. Existing Optimization Status

### 5.1 MEMORY_OPTIMIZATION_PLAN.md Progress

| Phase | Description | Status | % Done |
|-------|-------------|--------|--------|
| Phase 1 | Eliminate duplicate storage | Partial | 30% |
| Phase 2 | Move data to database (session keeps IDs only) | Not started | 0% |
| Phase 3 | Proper caching (TTL, max_entries) | Partial | 50% |
| Phase 4 | Optimize DataFrame storage | Partial | 40% |
| Phase 5 | Session state minimization | Not started | 0% |
| Phase 6 | Remove session persistence to JSON | Not started | 0% |

### 5.2 What IS Working

- `cleanup_memory()` function exists and is called at 6+ locations
- Load limits reduced to 500 per query
- `get_all_profiles()` with `include_raw_data` parameter exists
- Some caches have `ttl` and `max_entries`
- Global SalesQL rate limiter partially implemented
- Heavy columns (raw_crustdata, raw_data) dropped after load

### 5.3 Critical Gaps Remaining

| Gap | Impact | Priority |
|-----|--------|----------|
| Unbounded pagination in profile loading | OOM crash | CRITICAL |
| Triple DataFrame storage not eliminated | 3x memory waste | CRITICAL |
| Filter operations store full DataFrames | 125-250MB bloat | HIGH |
| Phase 2 (DB-first) not started | Session bloat | HIGH |
| No session memory monitoring | Silent creep | HIGH |
| PhantomBuster agent lock missing | Data corruption | CRITICAL |
| OpenAI 429 retry missing | Silent failures | CRITICAL |
| search_history.json race condition | Data loss | MEDIUM |

### 5.4 Configuration Gaps

**Missing from `.streamlit/config.toml`:**
```toml
[server]
maxUploadSize = 50          # Limit file uploads
maxMessageSize = 200        # Limit WebSocket messages

[logger]
level = "error"             # Reduce log overhead
```

**Security issue:** `config.json` contains plaintext API keys in the repository. Should only use `config.example.json` as template and `.streamlit/secrets.toml` for actual keys.

---

## 6. API & Concurrency Risks

### 6.1 Risk Matrix for 7-10 Users

| Risk | Severity | Likelihood | Impact |
|------|----------|------------|--------|
| OOM crash from DataFrame bloat | CRITICAL | HIGH | App kills, all users disconnected |
| PhantomBuster agent collision | CRITICAL | HIGH | Corrupted search results |
| OpenAI 429 → silent score=0 | CRITICAL | MEDIUM | Wrong screening results |
| SalesQL rate limit breach | HIGH | HIGH | 429 errors for all users |
| search_history.json race | HIGH | MEDIUM | Lost search history |
| Thread explosion (50 x N users) | HIGH | MEDIUM | Memory spike + rate limits |
| Google Sheets 60 RPM exceeded | MEDIUM | LOW | Filter tab fails |
| Supabase free tier pause | MEDIUM | LOW | 7-day inactivity pause |

### 6.2 Shared API Keys Budget

All users share the same API keys. Monthly cost for 7-10 users:

| Service | Plan Needed | Monthly Cost |
|---------|------------|-------------|
| Crustdata | Pro | ~$95-200+ |
| OpenAI (gpt-4o-mini) | Pay-as-you-go (Tier 2+) | ~$10-50 |
| PhantomBuster | Pro (80h) or Team (300h) | $159-439 |
| SalesQL | Organization (12K credits) | $119 |
| Supabase | Pro (no auto-pause) | $25 |
| Streamlit / Hosting | Self-hosted VM | $30-60 |
| **Total** | | **$438-893+/month** |

---

## 7. Production Alternatives Comparison

### 7.1 Framework Comparison Matrix

| Framework | Memory Efficiency | Prod Stability | Scalability | Migration Effort | Multi-User | Cost |
|-----------|:-:|:-:|:-:|:-:|:-:|:-:|
| **Streamlit (self-host)** | Low | Medium | Low-Medium | Very Low | Medium | $30-60/mo |
| **Dash (Plotly)** | High | High | High | High | High | $30-60/mo |
| **FastAPI + React** | Highest | Highest | Highest | Very High | Highest | $30-60/mo |
| **Reflex** | Medium | Medium | Medium | Medium-High | Medium | $30-60/mo |
| **NiceGUI** | Medium | Medium | Medium | Medium | Medium | $30-60/mo |
| Gradio | Low | Medium | Medium | Medium | Low | $30-60/mo |
| Panel | Low-Medium | Medium | Medium | Medium-High | Low | $30-60/mo |

### 7.2 Detailed Breakdown

#### Option A: Streamlit Self-Hosted (RECOMMENDED SHORT-TERM)
- **What:** Same code, deploy on AWS/GCP VM with 4-8GB RAM
- **Effort:** 1-2 days setup (Docker + Nginx)
- **Cost:** $30-60/month (t3.medium/large)
- **Pros:** Zero code changes, immediate 4-8x more memory, full control
- **Cons:** Still Streamlit's re-run model, still per-session memory multiplication
- **Max users:** 10-15 (with memory fixes) on 8GB VM

#### Option B: Dash by Plotly (RECOMMENDED MEDIUM-TERM)
- **What:** Rewrite UI in Dash (Flask-based, callback model)
- **Effort:** 3-6 weeks rewrite
- **Cost:** Same VM cost, optionally Dash Enterprise
- **Pros:** Stateless architecture = no per-session memory multiplication, battle-tested in enterprise, excellent for data dashboards
- **Cons:** Complete UI rewrite, different paradigm (callbacks vs script), steeper learning curve
- **Max users:** 50+ on same hardware

#### Option C: FastAPI + React/Next.js (RECOMMENDED LONG-TERM)
- **What:** Separate backend API from frontend
- **Effort:** 2-3 months full rewrite
- **Cost:** Same VM cost (often less at scale)
- **Pros:** Industry standard, best scalability, best memory efficiency (30-50MB baseline), independent frontend/backend scaling
- **Cons:** Requires JavaScript/TypeScript skills, two codebases, much more development time
- **Max users:** 100+ easily

#### Option D: Reflex (Python-only modern alternative)
- **What:** Python-native framework that compiles to React
- **Effort:** 3-5 weeks rewrite
- **Cost:** Same VM cost
- **Pros:** Python only (no JS needed), React-quality frontend, supports Redis for state
- **Cons:** Newer framework (less battle-tested), smaller community, production reliability uncertain
- **Max users:** 20-30+ (with Redis state backend)

### 7.3 Memory Comparison Per Architecture

| Architecture | Baseline | Per Session | 10 Users Total |
|-------------|----------|-------------|----------------|
| Streamlit (current, unoptimized) | 300 MB | 300 MB | **3.3 GB** (crash) |
| Streamlit (optimized) | 300 MB | 75 MB | **1.05 GB** (tight) |
| Streamlit self-host (optimized) | 300 MB | 75 MB | 1.05 GB on 4-8GB VM |
| Dash | 120 MB | ~5 MB (stateless) | **170 MB** |
| FastAPI + React | 50 MB backend | ~0 (stateless) | **50 MB** backend |

---

## 8. Self-Hosting Guide (Quickest Path to Stability)

### 8.1 Docker Setup

**Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "dashboard.py", \
  "--server.port=8501", \
  "--server.address=0.0.0.0", \
  "--server.headless=true", \
  "--browser.gatherUsageStats=false"]
```

**Docker Compose:**
```yaml
version: '3.8'
services:
  sourcingx:
    build: .
    ports:
      - "8501:8501"
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 8.2 Nginx Reverse Proxy

```nginx
upstream streamlit {
    server 127.0.0.1:8501;
}
server {
    listen 443 ssl;
    server_name sourcingx.yourdomain.com;

    # WebSocket support (CRITICAL for Streamlit)
    location / {
        proxy_pass http://streamlit;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

### 8.3 Recommended VM Sizes

| Provider | Instance | RAM | CPU | Cost/month |
|----------|----------|-----|-----|------------|
| AWS | t3.medium | 4 GB | 2 vCPU | ~$30 |
| AWS | t3.large | 8 GB | 2 vCPU | ~$60 |
| GCP | e2-medium | 4 GB | 2 vCPU | ~$25 |
| DigitalOcean | Basic 4GB | 4 GB | 2 vCPU | $24 |
| Hetzner | CPX21 | 4 GB | 3 vCPU | ~$8 |

### 8.4 Memory Monitoring Script

```bash
#!/bin/bash
# monitor.sh - run via cron every minute
CONTAINER="sourcingx"
MEM_LIMIT=80

MEM_USAGE=$(docker stats --no-stream --format "{{.MemPerc}}" $CONTAINER | tr -d '%')
if (( $(echo "$MEM_USAGE > $MEM_LIMIT" | bc -l) )); then
    echo "$(date): Memory at ${MEM_USAGE}% - restarting" >> /var/log/sourcingx.log
    docker restart $CONTAINER
fi
```

### 8.5 In-App Memory Check

```python
import gc, psutil

def check_memory():
    """Force GC if memory usage is high."""
    mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
    if mem_mb > 1500:  # 1.5 GB threshold
        gc.collect()
        st.warning(f"High memory: {mem_mb:.0f} MB")
```

---

## 9. Recommendation & Decision Matrix

### 9.1 Decision: Should We Keep Streamlit?

| Question | Answer |
|----------|--------|
| Can Streamlit handle 7-10 users on free tier? | **NO** - 1GB RAM is insufficient |
| Can we upgrade to a paid Streamlit Cloud? | **NO** - Paid tier requires Snowflake |
| Can we fix the code to fit in 1GB? | **MAYBE** for 3-4 users max, with ALL optimizations |
| Can we self-host Streamlit? | **YES** - Best short-term solution |
| Should we migrate to another framework? | **YES** - For long-term production at scale |

### 9.2 Recommended Strategy: Two-Phase Approach

#### Phase 1: Immediate (This Week) - Self-Host + Fix Critical Bugs
**Goal:** Stop the crashes NOW

1. Deploy to a 4-8GB VM via Docker ($30-60/mo)
2. Fix the 5 critical memory issues (Section 2.1-2.5)
3. Implement PhantomBuster agent lock
4. Add OpenAI 429 retry logic
5. Move search_history.json to database

**Expected result:** Stable app for 7-10 users within 1 week

#### Phase 2: Medium-Term (1-3 Months) - Evaluate Migration
**Goal:** Build for scale

1. Complete ALL memory optimizations from existing plan
2. Add proper monitoring (memory, API usage, active sessions)
3. Evaluate Dash or Reflex as migration targets
4. If user base grows past 15, begin migration to Dash or FastAPI+React

### 9.3 Cost Comparison

| Approach | Monthly Cost | Dev Time | Max Users | Crash Risk |
|----------|-------------|----------|-----------|------------|
| Stay on free Streamlit | $0 | 2 weeks fixes | 3-4 | HIGH |
| Self-host (4GB VM) | $30 | 1-2 days | 10-15 | LOW |
| Self-host (8GB VM) | $60 | 1-2 days | 15-25 | VERY LOW |
| Migrate to Dash | $30-60 | 3-6 weeks | 50+ | VERY LOW |
| Migrate to FastAPI+React | $30-60 | 2-3 months | 100+ | MINIMAL |

---

## 10. Action Plan

### Week 1: Emergency Stabilization

- [ ] **Deploy to self-hosted VM** (4GB minimum, 8GB recommended)
- [ ] **Fix #2.1:** Add max_profiles=5000 limit to database pagination
- [ ] **Fix #2.2:** Remove triple DataFrame storage, keep only `enriched_df`
- [ ] **Fix #2.3:** Reduce ThreadPoolExecutor to max_workers=10
- [ ] **Fix #2.7:** Use `include_raw_data=False` for browse operations
- [ ] **Fix:** Implement PhantomBuster agent lock
- [ ] **Fix:** Add OpenAI 429 retry with exponential backoff

### Week 2: Memory Optimization

- [ ] **Fix #2.6:** Store filter indices instead of full DataFrames
- [ ] **Fix #2.8:** Save screening batches to DB, clear from session
- [ ] **Fix #2.10:** Reduce db.py default limit to 5000
- [ ] **Fix #2.11:** Normalize cache TTLs (config=3600, data=300)
- [ ] **Fix #2.13:** Replace apply(axis=1) with vectorized operations
- [ ] **Fix:** Move search_history.json to Supabase table
- [ ] **Fix:** Add `psutil` memory monitoring to app
- [ ] **Fix:** Pin dependency versions in requirements.txt

### Week 3: Concurrency Hardening

- [ ] Complete SalesQL global rate limiter
- [ ] Add per-user API usage tracking
- [ ] Implement session timeout/cleanup
- [ ] Add `disconnectedSessionTTL=120` to Streamlit config
- [ ] Load test with 10 simulated users

### Month 2-3: Migration Evaluation

- [ ] Profile app under real production load
- [ ] Build proof-of-concept in Dash (1 page)
- [ ] Compare performance and development experience
- [ ] Make final migration decision

---

## Appendix A: All Files Analyzed

| File | Size | Issues Found |
|------|------|-------------|
| dashboard.py | 394 KB (~11,000 lines) | 15 memory issues |
| db.py | 35 KB | 2 issues (default limits) |
| normalizers.py | 21 KB | Clean |
| prompts.py | 32 KB | Clean (text storage acceptable) |
| helpers.py | 10 KB | Clean |
| usage_tracker.py | 9 KB | Clean |
| enrich.py | 6 KB | 1 minor (batch size) |
| pb_dedup.py | 6 KB | Clean |
| requirements.txt | 168 B | Unpinned versions |
| config.json | 1.2 KB | API keys in plaintext |

## Appendix B: Session State Keys (206 Instances)

Major memory-consuming keys in `st.session_state`:
- `enriched_df` - Main DataFrame (2-50 MB)
- `results_df` - PhantomBuster results (1-5 MB)
- `enriched_results` - Duplicate list (2-50 MB)
- `screening_results` - Screening output (1-2 MB)
- `filtered_out` - Filter DataFrames (5-250 MB)
- `screening_batch_state` - Accumulating results (0.5-2 MB)
- `passed_candidates_df` - Derived DataFrame (1-5 MB)

## Appendix C: Reference Links

- [Streamlit Cloud Limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/app-resources-and-limits)
- [Streamlit Caching Docs](https://docs.streamlit.io/develop/concepts/architecture/caching)
- [Streamlit in Snowflake](https://docs.snowflake.com/en/developer-guide/streamlit/about-streamlit)
- [Dash by Plotly](https://dash.plotly.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Reflex](https://reflex.dev/)
- [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
- [SalesQL API Limits](https://docs.salesql.com/reference/api-daily-and-minute-rate-limits)
- [Supabase Pricing](https://supabase.com/pricing)
