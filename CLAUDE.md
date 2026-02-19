# SourcingX — AI-Powered Recruiting Automation

## Project Overview

SourcingX is a Streamlit-based recruiting automation platform that screens LinkedIn profiles against job descriptions using AI. It integrates with Crustdata (profile enrichment), PhantomBuster (LinkedIn scraping), SalesQL (email lookup), and OpenAI (AI screening).

## Tech Stack

- **Frontend/Backend**: Streamlit (Python) — single-file app in `dashboard.py`
- **Database**: Supabase (PostgreSQL via REST API) — stores enriched profiles, NOT screening results
- **AI Screening**: OpenAI `gpt-4o-mini` (default) or `gpt-4o`
- **Profile Data**: Crustdata API (enrichment), PhantomBuster (scraping)
- **Email Lookup**: SalesQL API
- **Config**: Google Sheets (company lists, universities, blacklists)

## Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `dashboard.py` | Main Streamlit app — UI, screening logic, batch processing | ~8000+ |
| `db.py` | Supabase REST client — profile storage, queries | ~900 |
| `prompts.py` | 20+ role-specific screening prompts (Israel + Global) | ~1400 |
| `normalizers.py` | Data normalization between PhantomBuster/Crustdata formats | ~600 |
| `helpers.py` | Display helpers, field extraction | ~270 |
| `usage_tracker.py` | API cost tracking (Crustdata, OpenAI, SalesQL, PhantomBuster) | ~280 |
| `enrich.py` | LinkedIn profile enrichment utilities | ~180 |
| `config.json` | API keys and Google Sheets URLs (DO NOT commit) |  |

## Architecture Decisions

### Screening is ALWAYS fresh per JD
- Screening results are NOT stored in Supabase — each JD gets a fresh evaluation
- Profile enrichment data (raw Crustdata JSON) IS stored in Supabase
- Session-level results are kept in `st.session_state['screening_results']`

### Experience calculation is pre-computed in Python
- Total career experience is calculated in `dashboard.py:3526-3596`
- Military service (IDF) is detected and excluded from "reject >X years" checks
- A `✅/🚫 EXPERIENCE LIMIT CHECK` verdict is pre-computed and injected into the prompt
- The AI should NEVER recalculate total experience — it must use the pre-computed value

### Israeli military service handling
- Israeli military is mandatory (age 18-21) and excluded from experience limits
- Military keywords: idf, mamram, unit 8200, talpiot, israeli air force, c4i, etc.
- Two metrics: TOTAL CAREER SPAN (with military) and INDUSTRY EXPERIENCE (without military)
- The AI uses INDUSTRY EXPERIENCE for "reject >X years" rules

## Critical Functions in dashboard.py

| Function | Line | Purpose |
|----------|------|---------|
| `screen_profile()` | ~3085 | Screens a single profile against JD via OpenAI |
| `screen_profiles_batch()` | ~4147 | Parallel screening with ThreadPoolExecutor |
| `build_raw_data_index()` | ~4070 | Indexes raw data by LinkedIn URL for fast lookup |
| `fetch_raw_data_for_batch()` | ~4085 | Loads raw data from DB for a batch (memory efficient) |

## Screening Prompt Architecture

The screening prompt sent to OpenAI has these sections (in order):
1. **Role-specific rubric** from `prompts.py` (e.g., FULLSTACK_ISRAEL)
2. **Company description analysis** instructions
3. **Anti-hallucination rules**
4. **Assessment rules** (signal check, tiered scoring)
5. **User prompt** with JD + pre-calculated experience + raw JSON

### Pre-calculated sections injected into user prompt:
- Current employer (title, company, start date, duration)
- Lead/management experience (months calculated, consulting excluded)
- Fullstack experience (title + skills analysis)
- DevOps experience (title + skills analysis)
- Total career experience with EXPERIENCE LIMIT CHECK verdict

## Common Pitfalls

### AI hallucinating experience numbers
The AI (especially gpt-4o-mini) may try to recalculate total experience from raw JSON and get it wrong. The fix: pre-compute the comparison in Python and inject a clear `EXPERIENCE LIMIT CHECK: X years <= Y years -> PASSES` verdict that the AI must follow.

### "Already screened" profiles
Screening is always fresh per JD. Never skip profiles based on previous scores. Different JDs = different scores.

### Memory management
- Raw Crustdata JSON is 20-50KB per profile
- Batch processing loads raw data per-batch, not all at once
- Session state is cleaned after screening completes
- Max ~2-3 concurrent Streamlit users on free tier

### Overqualification rules
Only reject titles the JD EXPLICITLY lists. "Team Lead" is NOT overqualified unless the JD says so. Read the JD's reject list literally.

## Skills (Claude Code)

| Skill | Path | Purpose |
|-------|------|---------|
| `/screen` | `.claude/skills/screen/SKILL.md` | Team leader: coordinates parallel screening agents |
| `/email-opener` | `.claude/skills/email-opener/SKILL.md` | Generate personalized outreach emails |
| `/pre-filter-candidates` | `.claude/skills/pre-filter-candidates/SKILL.md` | Pre-filter profiles before screening |
| `/filter-csv-columns` | `.claude/skills/filter-csv-columns/SKILL.md` | CSV column filtering |

## Prompt Templates (prompts.py)

### Israel Engineering
`BACKEND_ISRAEL`, `FRONTEND_ISRAEL`, `FULLSTACK_ISRAEL`, `DEVOPS_ISRAEL`, `TEAMLEAD_ISRAEL`, `PRODUCT_ISRAEL`, `MOBILE_ISRAEL`

### Global GTM
`SALES_GLOBAL`, `SDR_GLOBAL`, `MARKETING_GLOBAL`, `CUSTOMER_SUCCESS_GLOBAL`, `SOLUTIONS_ENGINEER_GLOBAL`

### Other
`ENGINEERING_GLOBAL`, `MANAGER`, `VP`, `DATASCIENCE`, `AUTOMATION`, `AI_ENGINEER`, `GENERAL`

## Database Schema (Supabase)

### `profiles` table (enrichment data only)
- `linkedin_url` (PK), `raw_data` (JSONB), `name`, `current_title`, `current_company`
- `all_employers[]`, `all_titles[]`, `all_schools[]`, `skills[]` (indexed arrays)
- `email`, `email_source`, `status` (enriched/contacted/archived)
- `enriched_at`, `contacted_at` (timestamps)

### `api_usage_logs` table
- Provider, operation, credits, tokens, cost tracking

## Conventions

- All screening happens through OpenAI API (not Claude) for the dashboard
- Claude Code skills (`/screen`, `/email-opener`) use Claude for agent-based workflows
- Config lives in `config.json` — never commit API keys
- CSV exports use UTF-8-BOM for Excel compatibility
- LinkedIn URLs are normalized before storage (`normalize_linkedin_url()` in db.py)
