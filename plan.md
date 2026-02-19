# Multi-User Concurrency Plan (7-10 Users)

Based on findings in `research.md`. Ordered by priority (critical fixes first).

---

## Phase 1: Critical Fixes (Prevent Data Corruption & API Bans)

### 1.1 Global SalesQL Rate Limiter
**Problem:** Each user has independent 0.35s delay. Two concurrent users = 2x the rate = 429 errors.
**Fix:** Add a global (cross-session) rate limiter using `st.cache_resource` to share a thread-safe token bucket across all users.

```
File: dashboard.py
- Create a shared rate limiter via @st.cache_resource
- Use threading.Lock + time tracking for a simple token bucket
- Target: max 150 req/min globally (below 180 limit with safety margin)
- Each user's enrich_profiles_with_salesql() acquires from the shared bucket
```

### 1.2 PhantomBuster Agent Lock
**Problem:** Two users can launch the same PhantomBuster agent simultaneously, corrupting results.
**Fix:** Add a lightweight lock mechanism per agent_id.

```
File: dashboard.py
- Use @st.cache_resource to create a shared dict of locks per agent_id
- Before launching: check if agent is already running (fetch status first)
- Show warning: "Agent X is currently running (launched by another user). Wait or use a different agent."
- Alternative: assign dedicated agents per user (if plan supports enough phantoms)
```

### 1.3 OpenAI 429 Retry Logic
**Problem:** Rate limit errors silently produce score=0 results with "Screen error" message.
**Fix:** Add exponential backoff retry on 429/rate-limit errors.

```
File: dashboard.py, function: screen_profile() (~line 2690)
- Wrap OpenAI call in retry loop (max 3 retries)
- On 429: wait 2^attempt seconds, then retry
- On other errors: fail immediately as before
- Also reduce max_workers based on number of active screening sessions
```

---

## Phase 2: Resource Protection (Prevent OOM & Throttling)

### 2.1 Reduce OpenAI Concurrent Workers
**Problem:** `max_workers=50` per user. With 3+ concurrent screeners, that's 150+ threads.
**Fix:** Dynamic worker count based on active sessions.

```
File: dashboard.py (~line 5831)
- Track active screening sessions via @st.cache_resource counter
- Formula: max_workers = max(5, 50 // active_screening_sessions)
- When screening starts: increment counter
- When screening ends (or errors): decrement counter
- Minimum 5 workers per user to maintain reasonable speed
```

### 2.2 Memory-Conscious DataFrame Handling
**Problem:** 10 users × large DataFrames in session_state could exceed 1GB.
**Fix:** Limit what's stored in session and add monitoring.

```
File: dashboard.py
- Add a session memory estimate display in sidebar (debug info)
- When loading all profiles from DB, limit columns to what's needed for display
  (don't load full raw_data for browsing — load on demand for detail view)
- Cap profiles loaded per session to ~5000 (with pagination for more)
- Clear stale session data on login (enriched_results from previous sessions)
```

### 2.3 Supabase Connection Resilience
**Problem:** Single cached client; if it dies, all users are affected until TTL expires.
**Fix:** Add connection health check and auto-reconnect.

```
File: dashboard.py
- Add try/except around DB operations with auto-reconnect on connection errors
- Reduce cache TTL from 300s to 120s for faster recovery
- Add a "DB Status" indicator in sidebar
```

---

## Phase 3: Usage Controls (Cost & Fairness)

### 3.1 Per-User API Usage Tracking
**Problem:** Can't tell which user consumed API credits.
**Fix:** Extend usage_tracker to include username.

```
File: usage_tracker.py + db.py
- Add 'username' field to api_usage_logs table
- Pass st.session_state.get('username') to tracker calls
- Show per-user breakdown in the Usage tab
```

### 3.2 Per-User Daily Limits (Optional)
**Problem:** One user could exhaust all SalesQL credits (5000/day) leaving none for others.
**Fix:** Add configurable per-user daily limits.

```
File: dashboard.py
- Add settings for per-user daily limits:
  - SalesQL: 500-1000 lookups/user/day
  - OpenAI: 500 screenings/user/day
  - Crustdata: 500 enrichments/user/day
- Check against usage_tracker before allowing operation
- Show remaining quota in UI
```

### 3.3 Active Users Display
**Problem:** Users don't know who else is online or what operations are running.
**Fix:** Add lightweight presence indicator.

```
File: dashboard.py sidebar
- Use @st.cache_resource with a shared dict tracking last-seen timestamps
- Show "X users active" in sidebar
- Show "Screening in progress by [user]" warnings near action buttons
```

---

## Phase 4: Infrastructure Improvements (If Needed)

### 4.1 Upgrade Supabase to Pro ($25/mo)
**When:** If 500MB database limit is approached or 7-day pause becomes an issue.
**Benefits:** 8GB database, no pausing, daily backups, 250GB bandwidth.

### 4.2 Upgrade Streamlit or Self-Host
**When:** If 1GB RAM is consistently hit with 10 users.
**Options:**
- Streamlit Teams (paid): More resources, no sleep
- Self-host on a small VPS ($5-10/mo): 2-4GB RAM, full control
- Railway/Render free tier: Similar to Streamlit Cloud but different limits

### 4.3 Move to OpenAI Tier 2+
**When:** If screening bottleneck becomes frequent.
**How:** Spend $50+ on OpenAI to auto-upgrade to Tier 2 (5,000 RPM, 2M TPM).

---

## Implementation Order

| Step | Effort | Impact |
|------|--------|--------|
| 1.3 OpenAI retry logic | 30 min | Stops silent screening failures |
| 1.1 Global SalesQL limiter | 45 min | Prevents API ban |
| 1.2 PhantomBuster agent lock | 45 min | Prevents data corruption |
| 2.1 Dynamic OpenAI workers | 30 min | Prevents rate limit cascade |
| 2.2 Memory-conscious DataFrames | 1 hr | Prevents OOM crashes |
| 2.3 DB connection resilience | 30 min | Prevents cascade failures |
| 3.1 Per-user usage tracking | 1 hr | Enables cost accountability |
| 3.3 Active users display | 30 min | Better user experience |
| 3.2 Per-user daily limits | 1 hr | Prevents credit hogging |

**Total estimated changes:** ~6 hours of work, all in `dashboard.py` + minor `db.py` changes.

---

## What Does NOT Need Changing

- **Session state isolation**: Already per-user via `st.session_state` + per-user files
- **DB upserts**: Already use `ON CONFLICT` — concurrent writes are safe (last-write-wins)
- **Authentication**: Already has `streamlit_authenticator` with per-user login
- **Crustdata API**: Low concurrency risk (batch requests, few calls)
- **DB pagination**: Already fixed (earlier today)
