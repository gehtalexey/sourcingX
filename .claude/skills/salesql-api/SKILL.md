---
name: salesql-api
description: SalesQL API reference — email/phone finder by LinkedIn URL. Read before touching SalesQL code.
allowed-tools: Bash, Read, Write, Edit, Grep
---

# SalesQL API — Reference for All Projects

SalesQL is the LinkedIn email finder used across our sourcing projects (`smartlead-sourcing-autopilot`, `daily-sourcing-autopilot-e2e`, `SourcingX`, `claude-terminal-Slack sourcing-agent`, `sourcing-in-terminal`). This skill is the **single source of truth** about how the SalesQL API behaves. Read it before changing email-finding code or debugging hit-rate problems.

Full docs lived in the Obsidian vault: `wiki/sources/2026-05-20-salesql-api-reference.md` and `wiki/entities/salesql.md`.

## API basics

- **Base URL:** `https://api-public.salesql.com/v1`
- **Auth:** `Authorization: Bearer <SALESQL_API_KEY>` (header)
- **Official docs (Swagger):** https://api-public.salesql.com/docs
- **Help center:** https://help.salesql.com/

## Endpoints we use

### `GET /persons/enrich/` — find email by LinkedIn URL

**Trailing slash is required.**

Parameters:

| Param | Type | Notes |
|-------|------|-------|
| `linkedin_url` | string (URI) | The LinkedIn profile URL — see URL Format Rules below |
| `first_name` | string | Optional alternative input |
| `last_name` | string | Optional alternative input |
| `full_name` | string | Optional alternative input |
| `organization_name` | string | Optional — improves accuracy when name is ambiguous |
| `organization_domain` | string | Optional |
| `match_if_direct_email` | bool | `true` = only return Direct (personal) email, no work email |
| `match_if_direct_phone` | bool | `true` = only return Direct phone |

Response (HTTP 200): JSON with `emails: [{email, type, status}]`, `phones`, profile data. `type` is `Direct` (personal) or `Professional` (work).

Status codes:
- `200` — match found
- `404` — no match (treat as "no result", not an error)
- `429` — rate limit exceeded

### `GET /account/credits` — remaining credits

No trailing slash. Returns `{credits, balance}` (one of those keys depending on plan).

## URL Format Rules — CRITICAL

**SalesQL accepts BOTH vanity URLs and hash-suffix URLs.** Do NOT strip the suffix.

| URL form | Example | Works? |
|----------|---------|--------|
| Vanity | `https://www.linkedin.com/in/edsioufi` | Yes |
| Hash-suffix | `https://www.linkedin.com/in/jordan-weir-35007150` | Yes — confirmed in production |
| Numeric+letter suffix | `https://www.linkedin.com/in/jose-leandro-torres-sicilia-aba038114` | Yes — confirmed in production |
| Obfuscated (`ACoAA...`) | `https://www.linkedin.com/in/ACoAAAlC_rYBHnh7F_LpS04vhecIjCN7y4rlmd8` | **No.** Resolve to flagship URL first (see Crustdata `linkedin_flagship_url` / `flagship_profile_url`) |

The hash suffix is part of LinkedIn's canonical URL when the vanity name is taken — it is NOT padding. Stripping it produces either a 404 or a URL pointing to a **different person**. Do not normalize it away.

## Why a lookup returns no email

Per SalesQL's official FAQ:

> "If there's any doubt about the accuracy of the results, it won't show them."

SalesQL is precision-over-recall. A "no result" means SalesQL has no high-confidence match — not that the URL is malformed and not that URL normalization will help. The realistic lift for hit rate is a **second provider** (Apollo, ContactOut, Hunter, Prospeo), not URL rewriting.

Observed hit rate in production (2026-05-20, `owner-staff-agents-eng-sf` position): **~19% (6/31)** on US engineering candidates. Some hits had hash suffixes; some misses had clean vanity URLs.

## Credits and rate limits

- **Credits charged only when results return.** No result → no credit.
- Phone numbers: not charged.
- Duplicate enrichment of the same profile: not charged again.
- **Daily rate limit is consumed on every call** including those that return nothing.

Plan tiers (2026):

| Plan | API access | Daily calls | Per-minute |
|------|-----------|-------------|------------|
| Basic | No | — | — |
| Professional | Yes | 5,000 | 180/min |
| Organization | Yes | 20,000 | 300/min |

429 returned when the daily cap is hit.

## Personal email filter (our pipeline default)

We push personal emails to cold-email tools (Smartlead, Instantly), not work emails. Set `match_if_direct_email=true` on every request — that filters server-side to Direct (personal) addresses.

**Do NOT layer a local personal-domain whitelist on top.** Earlier versions of this pipeline had a 19-domain allowlist in `email_step.py` that silently rejected legitimate Direct emails at `fastmail.com`, `hey.com`, `tutanota.com`, `pm.me`, `zoho.com`, and other less-common providers. Trust SalesQL's classification — if it says `type=Direct`, treat the email as personal. If you want defence in depth, use a **denylist** of obvious corporate domains (microsoft.com, google.com, etc.), not an allowlist.

When matching the `type` field in code, do it **case-insensitive** (`str(e.get('type','')).lower() == 'direct'`) — the docs don't contract a casing.

When multiple emails come back, **prefer `status=Valid`** over `Risky`/`Catch-all`/`Unknown`.

## Gotchas

1. **404 is normal** — skip and continue, do not retry.
2. **Trailing slash matters** — `/persons/enrich/` requires it; `/account/credits` does not.
3. **Obfuscated ACoAA URLs** — SalesQL 404s on these. Resolve to a flagship URL via Crustdata `linkedin_flagship_url` / `flagship_profile_url` first.
4. **No batch endpoint** — enrich one at a time with throttling (~1 req/sec is safe).
5. **Coverage gap, not URL bug** — do not "fix" missing emails by mangling URLs. Add a fallback provider instead.
6. **Stale emails possible** — for projects that look up old LinkedIn URLs, the returned email may be from a previous employer. Filter accordingly.
7. **`credits` response shape varies** — current API returns `{"credits": {"emails_and_phones": <int>, ...}}` (nested). Older shapes used a flat int. Code that reads credits must handle both.
8. **Stop the batch on first 429** — SalesQL counts every 429 against the daily quota, so continuing the loop after the cap is hit just burns headroom for nothing.
9. **Many-to-one URL resolution** — multiple candidate URLs (obfuscated + canonical for the same person) can resolve to the same flagship URL. When mapping SalesQL results back, update every original row that mapped in, not just one — a flat reverse dict will collapse them.
