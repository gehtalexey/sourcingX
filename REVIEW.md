# Code Review Criteria — SourcingX

This is the shared rubric for **AI reviewers** (Codex) and **human reviewers**
(Alexey, occasional collaborators) when evaluating a PR.

It exists so reviewers are looking at the **same** axes and so Claude can
self-review its own changes before opening the PR. If a section does not
apply to the change, skip it with a one-line note in the PR comments.

---

## 1. Correctness

- The change does what the PR description says — no more, no less.
- Every database write/read uses columns that currently exist in the live
  schema. New SQL migrations have a matching numbered file under
  `migrations/` and a sane rollback story.
- Error paths are handled, not swallowed. Bare `except: pass` is a red flag.
- External API calls use the project's rate-limiter / retry / circuit-breaker
  helpers (`api_helpers.py`, `error_handling.py`) — they don't bypass them
  with raw `requests`.

## 2. Tests

- Behavior-changing code has a focused test. Bug fixes have a regression
  test that fails before the fix and passes after.
- Tests use the project's mocks (`conftest.py`, `MockOpenAIClient`, fake
  Supabase clients). They never hit real APIs.
- `test_*.py` filenames match the area they cover; one assertion per logical
  expectation, not one mega-test per file.
- New tests are listed in `.github/workflows/test.yml` if they should run
  in CI (or there is a deliberate note explaining why not).

## 3. Security & secrets

- No API keys, tokens, or credentials anywhere in the diff. `config.json` and
  `google_credentials.json` stay gitignored.
- User-supplied strings (LinkedIn URLs, job descriptions, search queries) are
  validated before they reach DB filters or external APIs (`security.py`).
- New endpoints / Streamlit inputs do not introduce SQL injection,
  command injection, or prompt-injection footguns.
- Secrets that CI needs go in GitHub Actions secrets, referenced by name —
  never inlined.

## 4. Performance & cost

- Loops over profiles use batch helpers (`upsert_batch`, `update_profile_*_batch`)
  rather than per-row API/DB calls.
- New AI calls have an explicit model choice and an estimated cost-per-run
  in the PR description if the call runs in a loop.
- Streamlit caches (`@st.cache_data`) have a sensible `ttl` and `max_entries`;
  caches that hold raw profiles are bounded to prevent OOM.

## 5. Repo hygiene

- Diff contains only intentional source / test / docs / migrations / CI.
- No `.agent/`, `.agents/`, `.continue/`, `.claude/settings.local.json`,
  `*.tmp`, `nul`, `_*` scratch files, or `debug_*.py` ever land in the PR.
- Line endings are LF (the repo's `.gitattributes` enforces this — the
  diff should never show 1000-line CRLF-only churn).
- Conventional Commits format on every commit.

## 6. UX & migration safety

- Streamlit changes do not break the resume-from-DB flow.
- Schema changes are backwards-compatible during the deploy window
  (add column → backfill → switch reads), or the PR description states
  the planned cutover.
- Dropped columns must be removed from every read/write/filter site in
  the same PR (see `test_profiles_status_dropped.py` for the pattern).

---

## Reviewer instructions

- **Codex (AI):** Read the diff, check each section above. Report findings
  grouped by severity: BLOCKER, NIT, OPTIONAL. Don't edit files — only
  report. Alexey relays findings to Claude.
- **Claude (self-review before push):** Walk this list before opening the
  PR. If a section is N/A, note it in the PR description.
- **Alexey (final merge):** Skim sections 1, 3, 5 even when Codex says LGTM.
  Click merge yourself — no auto-merge.
