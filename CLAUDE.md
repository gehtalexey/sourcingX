# SourcingX — AI-Powered Recruiting Automation

## Workflow Rules

- **Desktop (Claude Code CLI):** Always ask before making changes on `master` branch. Offer to create a feature branch first.
- **Mobile (Claude app):** Branches are created automatically - safe by default.

## Project Overview

SourcingX is a Streamlit-based recruiting automation platform that screens LinkedIn profiles against job descriptions using AI. It integrates with Crustdata (profile enrichment), PhantomBuster (LinkedIn scraping), SalesQL (email lookup), and OpenAI (AI screening).

## Tech Stack

- **Frontend/Backend**: Streamlit (Python) — single-file app in `dashboard.py`
- **Database**: Supabase (PostgreSQL via REST API) — stores enriched profiles, NOT screening results
- **AI Screening**: OpenAI `gpt-4o-mini` (default) or `gpt-4o`
- **Profile Data**: Crustdata API (enrichment), PhantomBuster (scraping)
- **Email Lookup**: SalesQL API
- **Config**: Google Sheets (company lists, universities, blacklists) + local CSV filters

## Project Structure

```
sourcingX/
├── dashboard.py              # Main Streamlit app (UI, screening, batch processing)
├── db.py                     # Supabase REST client (profiles, queries, usage)
├── prompts.py                # 20 role-specific screening prompt templates
├── normalizers.py            # Data normalization (PhantomBuster/Crustdata formats)
├── helpers.py                # Display helpers, field extraction
├── usage_tracker.py          # API cost tracking across all providers
├── enrich.py                 # Standalone Crustdata enrichment CLI
├── error_handling.py         # Exception hierarchy, retry logic, circuit breaker
├── api_helpers.py            # Rate limiting (token bucket), safe API call wrappers
├── security.py               # Input validation, secrets protection
├── db_transactions.py        # Batch operation safety patterns
├── db_migrations.py          # Database migration management
├── pb_dedup.py               # PhantomBuster deduplication (post-scrape filtering)
├── config.json               # API keys (DO NOT commit — use config.example.json)
├── config.example.json       # Config template with placeholder keys
├── requirements.txt          # 9 dependencies
├── supabase_setup.sql        # Initial database schema
├── conftest.py               # Pytest fixtures and mock helpers
├── test_screening.py         # AI screening tests
├── test_mocking_services.py  # External service mocking tests
├── test_screening_fixes.py   # Regression tests for screening helpers
├── test_screening_parameterized.py  # Parameterized screening tests
├── .claude/skills/           # Claude Code skills (5 skills)
├── skills/                   # Additional skills (email-subject-line, email-personal-note)
├── .devcontainer/            # Dev container config (Python 3.11, port 8501)
├── .streamlit/               # Streamlit config (dark theme)
├── filters/                  # CSV filter files (blacklist, companies, universities)
├── migrations/               # SQL migration files (11 migrations)
├── sql/                      # SQL scripts (screening_prompts table)
└── docs/                     # Analysis documents
```

## Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `dashboard.py` | Main Streamlit app — UI, screening logic, batch processing | ~7,300 |
| `db.py` | Supabase REST client — profile storage, queries, settings | ~1,100 |
| `prompts.py` | 20 role-specific screening prompts (Israel + Global) | ~1,400 |
| `error_handling.py` | Exception hierarchy, retry with backoff, circuit breaker | ~660 |
| `normalizers.py` | Data normalization between PhantomBuster/Crustdata formats | ~590 |
| `api_helpers.py` | Token bucket rate limiting, safe API call wrappers | ~570 |
| `security.py` | Input validation, secrets protection, rate limiting decorators | ~360 |
| `db_transactions.py` | Safe batch operations with error handling | ~340 |
| `usage_tracker.py` | API cost tracking (Crustdata, OpenAI, SalesQL, PhantomBuster) | ~280 |
| `helpers.py` | Display helpers, field extraction from Crustdata responses | ~270 |
| `db_migrations.py` | Migration management with checksum verification | ~250 |
| `pb_dedup.py` | PhantomBuster post-scrape deduplication against database | ~190 |
| `enrich.py` | Standalone Crustdata enrichment CLI utility | ~180 |
| `config.json` | API keys and Google Sheets URLs (DO NOT commit) | |

## Architecture Decisions

### Screening is ALWAYS fresh per JD
- Screening results are NOT stored in Supabase — each JD gets a fresh evaluation
- Profile enrichment data (raw Crustdata JSON) IS stored in Supabase
- Session-level results are kept in `st.session_state['screening_results']`

### Experience calculation is pre-computed in Python
- Total career experience is calculated in `dashboard.py` (search for `calculate_total_experience` or the experience limit check block)
- Military service (IDF) is detected and excluded from "reject >X years" checks
- A `EXPERIENCE LIMIT CHECK` verdict is pre-computed and injected into the prompt
- The AI should NEVER recalculate total experience — it must use the pre-computed value

### Israeli military service handling
- Israeli military is mandatory (age 18-21) and excluded from experience limits
- Military keywords: idf, mamram, unit 8200, talpiot, israeli air force, c4i, etc.
- Two metrics: TOTAL CAREER SPAN (with military) and INDUSTRY EXPERIENCE (without military)
- The AI uses INDUSTRY EXPERIENCE for "reject >X years" rules

### Error handling & resilience
- Custom exception hierarchy in `error_handling.py` (see section below)
- Retry with exponential backoff via `retry_with_backoff()` decorator
- Circuit breaker pattern prevents cascade failures to external services
- Token bucket rate limiting in `api_helpers.py` protects API quotas
- Input validation in `security.py` for LinkedIn URLs, text inputs, API keys

## Critical Functions in dashboard.py

| Function | Purpose |
|----------|---------|
| `screen_profile()` | Screens a single profile against JD via OpenAI |
| `screen_profiles_batch()` | Parallel screening with ThreadPoolExecutor |
| `build_raw_data_index()` | Indexes raw data by LinkedIn URL for fast lookup |
| `fetch_raw_data_for_batch()` | Loads raw data from DB for a batch (memory efficient) |
| `enrich_batch()` | Batch enrich profiles via Crustdata API |
| `apply_pre_filters()` | Apply pre-filtering rules before screening |
| `normalize_phantombuster_columns()` | Normalize PhantomBuster CSV columns |
| `launch_phantombuster_agent()` | Launch a PhantomBuster scraping agent |

## Error Handling & Resilience Patterns

### Exception hierarchy (`error_handling.py`)
```
ApplicationError (base)
├── ExternalServiceError  — API errors (service name, status code, response body)
│   └── RateLimitError    — Rate limit errors (includes retry_after)
├── ValidationError       — Input validation errors
├── DatabaseError         — Database operation errors
└── CircuitBreakerError   — Circuit breaker is open (service temporarily unavailable)
```

### Retry decorator
Use `retry_with_backoff()` from `error_handling.py` for API calls that may fail transiently. It supports configurable max retries, base delay, and exception filtering.

### Circuit breaker
`get_service_circuit()` returns a circuit breaker per external service. After a threshold of failures, the circuit opens and fast-fails requests for a cooldown period, preventing cascade failures.

### Rate limiting
`api_helpers.py` provides a `RateLimiter` class (token bucket algorithm) and `rate_limited()` decorator. Use `get_rate_limiter()` to get a limiter for a specific provider.

### Input validation
`security.py` provides validation functions: `validate_linkedin_url()`, `validate_google_sheets_url()`, `validate_text_input()`, `validate_api_key_format()`, `validate_job_description()`. Use these at system boundaries (user input, external data).

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

### Duplicate functions
`normalize_crustdata_profile()` exists in both `normalizers.py` (canonical) and `dashboard.py` (inline copy). The canonical version in `normalizers.py` is the source of truth.

## Skills (Claude Code)

| Skill | Path | Purpose |
|-------|------|---------|
| `/screen` | `.claude/skills/screen/SKILL.md` | Team leader: coordinates parallel screening agents |
| `/email-opener` | `.claude/skills/email-opener/SKILL.md` | Generate personalized outreach emails |
| `/pre-filter-candidates` | `.claude/skills/pre-filter-candidates/SKILL.md` | Pre-filter profiles before screening |
| `/filter-csv-columns` | `.claude/skills/filter-csv-columns/SKILL.md` | CSV column filtering for recruiter review |
| `/phantom-pre-filter` | `.claude/skills/phantom-pre-filter/SKILL.md` | Pre-filter raw PhantomBuster CSV before enrichment |
| (email-personal-note) | `skills/email-personal-note/SKILL.md` | Generate personal notes for follow-up emails |
| (email-subject-line) | `skills/email-subject-line/SKILL.md` | Generate email subject lines with angle variation |

## Prompt Templates (prompts.py)

### Israel Engineering
`BACKEND_ISRAEL`, `FRONTEND_ISRAEL`, `FULLSTACK_ISRAEL`, `DEVOPS_ISRAEL`, `TEAMLEAD_ISRAEL`, `PRODUCT_ISRAEL`, `MOBILE_ISRAEL`

### Global GTM
`SALES_GLOBAL`, `SDR_GLOBAL`, `MARKETING_GLOBAL`, `CUSTOMER_SUCCESS_GLOBAL`, `SOLUTIONS_ENGINEER_GLOBAL`

### Other
`ENGINEERING_GLOBAL`, `MANAGER`, `VP`, `DATASCIENCE`, `AUTOMATION`, `AI_ENGINEER`, `GENERAL`

### Prompt selection
- `DEFAULT_PROMPTS` dict maps role keys to prompt templates
- `DEFAULT_SCREENING_PROMPT` uses `GENERAL` as fallback
- `db.py:match_prompt_by_keywords()` finds the best prompt for a JD based on keyword matching
- Prompts can also be stored/customized in Supabase via `screening_prompts` table

## Database Schema (Supabase)

### `profiles` table (enrichment data only)
- `linkedin_url` (UNIQUE), `raw_data` (JSONB), `name`, `current_title`, `current_company`
- `all_employers[]`, `all_titles[]`, `all_schools[]`, `skills[]` (GIN-indexed arrays)
- `email`, `email_source`, `status` (enriched/screened/contacted/archived)
- `enriched_at`, `screened_at`, `contacted_at` (timestamps)
- Screening fields: `screening_score`, `screening_fit_level`, `screening_summary`, `screening_reasoning`

### `api_usage_logs` table
- Provider, operation, credits, tokens, cost tracking
- Indexed on provider and created_at

### `screening_prompts` table
- `role_type`, `name`, `prompt_text`, `keywords[]`, `is_default`
- Managed via `sql/screening_prompts.sql`

### Views
- `profiles_needing_screening` — enriched profiles awaiting screening
- `pipeline_stats` — funnel metrics (pending, screened, contacted, fit levels)

### Migrations
11 migration files in `migrations/` (002-010 + create_api_usage_logs). Managed by `db_migrations.py` with checksum verification. Initial schema in `supabase_setup.sql`.

## Testing

Run tests with: `pytest`

| File | Purpose |
|------|---------|
| `conftest.py` | Pytest fixtures: MockOpenAIResponse, MockOpenAIClient, shared profile/JD fixtures |
| `test_screening.py` | AI screening logic tests with sample JDs and profiles |
| `test_mocking_services.py` | External service mocking patterns, error handling, concurrency |
| `test_screening_fixes.py` | Regression tests for `format_past_positions()` and `format_education()` |
| `test_screening_parameterized.py` | Parameterized tests with `pytest.mark.parametrize` |

## Development Setup

1. **DevContainer** (recommended): Python 3.11, auto-forwards port 8501
2. **Dependencies**: `pip install -r requirements.txt` (streamlit, pandas, requests, openai, gspread, google-auth, streamlit-authenticator, bcrypt, plotly)
3. **Config**: Copy `config.example.json` to `config.json` and fill in API keys (Crustdata, OpenAI, PhantomBuster, Google credentials)
4. **Run**: `streamlit run dashboard.py`
5. **Filter data**: CSV files in `filters/` (blacklist.csv, not_relevant_companies.csv, target_companies.csv, universities.csv)

## Conventions

- All screening happens through OpenAI API (not Claude) for the dashboard
- Claude Code skills (`/screen`, `/email-opener`) use Claude for agent-based workflows
- Config lives in `config.json` — never commit API keys
- CSV exports use UTF-8-BOM for Excel compatibility
- LinkedIn URLs are normalized before storage (`normalize_linkedin_url()` in `normalizers.py`)
- Security validation should use `security.py` functions for user input at system boundaries
- API calls to external services should use `api_helpers.py` wrappers for rate limiting
- Filter CSVs live in `filters/` directory
- Streamlit theme: dark mode with purple primary (`#615fff`)
