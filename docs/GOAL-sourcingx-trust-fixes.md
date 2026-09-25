# SourcingX: make screening trustworthy, show every cost, fix openers, measure it blind

## 1. Outcome
SourcingX no longer rejects people just because LinkedIn doesn't prove a must-have (they
land in a visible "Needs verification" group instead), every paid click shows its cost
first, openers cite something concrete the person built, and a blind test by two outside
experts says how far SourcingX's verdicts can now be trusted, all merged and shown on the
existing audit page.

## 2. Why it matters
Alexey runs sourcing on SourcingX. The September audit
(`docs/sourcingx-audit-2026-09.md`, page https://claude.ai/artifact/Sfn6q6sQuR5qNzxYjWZctf)
found it rejects most strong candidates: of 20 people the sister pipeline (kalamata) had
approved for Dwelly, SourcingX passed 2; with one must-have moved to nice-to-haves, 14.
It also spends Crustdata/SalesQL credits without showing them, and its openers are one
generic sentence. Both experts (Claude Fable 5.1, GPT-6 Astra) named these as the next
steps. Alexey will use the blind-test result to decide whether to trust SourcingX's NO GO
on real searches or keep reviewing them by hand.

## 3. Inputs
- Repo `C:\Users\gehta\projects\sourcingX`, master `152e5c9` (includes PRs #139-#143).
- The audit list: `docs/sourcingx-audit-2026-09.md` (items 1, 2, 3, 4, 8 are this goal).
- Previous goal file with full history: `docs/GOAL-sourcingx-fix-and-expert-audit.md`.
- Shared Supabase DB, project `ciyyvbzblogtbwabhbmh`: `profiles`, `pipeline_candidates`,
  `pipeline_positions`, `screening_results`, `api_usage_logs`.
- Sister code, read-only: `C:\Users\gehta\projects\agent-kalamata` (does it read
  `screening_results.screening_fit_level`?).
- The live app: `.claude/launch.json` entry `sourcingx`, driven with Playwright.
- Experts: the `consult` skill (Fable via `consult-expert`, Astra via `codex exec`).

## 4. Constraints
- Code changes: feature branch per build item, PR against master, real Codex review via
  `codex-pr-review` (4 rounds max), CI green.
- **Never change how rows are written to shared tables without asking first.** Adding a new
  value to `screening_results.screening_fit_level` (e.g. "Needs verification") counts: check
  whether agent-kalamata or Supanova read that column first. If anything reads it, store
  an existing value (e.g. "Maybe") plus the new detail in a column nobody reads, or ask.
- Keep the hiring manager's requirement strict (Astra's version): an unproven must-have is
  "needs verification", not "met". A must-have the profile CONTRADICTS is still NO GO.
- Money: up to **$3 model spend and 10 Crustdata + SalesQL credits** for the whole goal
  (Alexey, 2026-09-24). Track the running total in block 8; stop and ask before crossing.
  The re-screens and blind test use stored, already-enriched profiles only (0 credits);
  credits are for one small live check of the new cost display.
- Experts are read-only. Nothing sent to them may contain candidate names, emails or
  LinkedIn URLs; strip and check with a text search before sending. Company names are fine.
- Openers are generated and shown only. Never send an email, push a lead or touch SmartLead.
- Can't edit `.github/workflows/*.yml`; put new tests in files CI already runs.
- Plain English for Alexey; one question at a time; never delete without asking.

**What can run in parallel:**

| Build item | Files it owns |
|---|---|
| A. "Needs verification" for unproven must-haves | `screening_policy.py`, `dashboard.py` (screening + results view: `screen_profile()`, `_verdicts_force_no_go()`, `_decision_to_fit_label()`, GO/MAYBE/NO GO display), one CI-covered test file |
| B. Cost shown before every paid click | `dashboard.py` (AI Screen estimate, SalesQL button + post-lookup message, Load tab labels, search call sites calling `log_search_usage()`), `usage_tracker.py`, one CI-covered test file |
| C. Opener prompt | `email_generator.py`, `test_pick_current_employer.py` or another CI-covered test file |
| D. Measurements (re-screen 20, blind test) | read-only + the running app; no code |

A and B both own `dashboard.py`: serial, A first. C owns no shared file: runs alongside A.
D's re-screen waits for A merged; the blind test's expert labelling (D2) is read-only and can
start at once. Max three agents in flight. Merge one at a time. No `git stash`.

## 5. Definition of done
- [ ] Unproven must-haves no longer reject. The model can answer met / not met /
      needs verification per must-have; any "not met" = NO GO; no "not met" but at least one
      "needs verification" = a visible "Needs verification" group (not NO GO). Proven by:
      new tests in a CI-covered file for all three cases, CI green, PR MERGED with Codex review.
- [ ] Shared-table check done before merging A. Proven by: a grep of agent-kalamata and
      Supanova for `screening_fit_level`, shown in chat, and the chosen storage stated.
- [ ] Re-screen of the same 20 kalamata-approved Dwelly candidates with the ORIGINAL brief
      (production-AI line kept as a must-have). Proven by: a query on `screening_results`
      for this run showing the GO / Needs verification / NO GO counts (was 2 / – / 18), and
      the NO GO reasons all citing a contradiction, not "not established".
- [ ] Blind test done. Proven by: a results file with one row per profile (target 40, both
      positions Dwelly + Owner, a mix of kalamata-approved, kalamata-rejected and
      SourcingX-screened), each with Fable's and Astra's blind label (outreach / reject /
      needs verification, given the brief and the anonymised profile, never any verdict),
      SourcingX's new verdict, and the headline numbers: of profiles both experts call
      "outreach", how many SourcingX says NO GO; of profiles both call "reject", how many
      SourcingX says GO.
- [ ] Every paid click shows its cost first: AI Screen shows "N profiles need a top-up =
      N Crustdata credits" plus the AI estimate built from the last run's real cost per
      candidate; the email lookup button shows "N lookups"; every search is written to
      `api_usage_logs`; enrichment labels say 1 credit. Proven by: tests, CI green, PR
      MERGED with Codex review, and one small live check (a ≤10-result search + screening
      ≤3 thin profiles) with a screenshot of the cost line and the matching
      `api_usage_logs` rows.
- [ ] The email lookup never says "already have emails" when none were found; it says
      "found N of M". Proven by: a test reproducing the false message, and the live check.
- [ ] Openers cite one concrete thing from the person's profile, use the sender and company
      from the position (no hard-coded "Israeli tech company"), never start with "Your",
      and never use "aligns well" / "mission". Proven by: tests on the prompt, CI green,
      PR MERGED with Codex review, and 3 openers generated in the app for Dwelly GO
      candidates, shown side by side with 3 kalamata openers (anonymised).
- [ ] Audit page updated in place (same link) with a "What changed" section: new counts,
      blind-test numbers, cost display, opener examples; `docs/sourcingx-audit-2026-09.md`
      marks items 1, 2, 3, 4, 8 as fixed or improved. Proven by: opening the page.
- [ ] Spend reported. Proven by: `api_usage_logs` totals for the goal window.

## 6. Stop conditions
Decide alone: the exact prompt/policy wording, how "Needs verification" looks in the app,
test design, the blind-sample composition, page layout.

Pre-approved by Alexey for this goal only (2026-09-24): merging every build-item PR and a
docs PR once Codex review is clean and CI is green.

Come back and ask, every time:
- Before crossing $3 model spend or 10 credits.
- Before changing what is written to a shared table (see constraints).
- Before sending anything outside (emails, Slack, SmartLead) — never in this goal.
- Before deleting or rewriting any file.

**A question to Alexey never stops the whole goal.** Before asking, start everything that
does not depend on the answer, in background subagents. While other work remains, ask
without blocking: a one-line `PushNotification` with the options, the same line in chat,
and the item marked "waiting on Alexey" in block 8. No answer within 10 minutes means that
item is parked: carry on with the rest and raise it again when he is back. Use a blocking
pop-up only when nothing else is left to do. Anything hard to undo still waits for his answer.

## 7. Deliverable
- Three merged PRs against master (A, B, C) plus a docs PR if the audit file changes.
- The blind-test results file in the scratchpad and summarised on the audit page.
- The existing audit page updated in place: https://claude.ai/artifact/Sfn6q6sQuR5qNzxYjWZctf
- A phone-sized final summary: the new 20-candidate numbers, the blind-test headline
  (how many expert-approved people SourcingX still rejects), what to do next.

## 8. Handover — where things stand right now
Mid-step handover 2026-09-25: see docs/HANDOFF.md
**Written 2026-09-24 at the end of the fix-and-audit goal.**

- Done last goal (all merged, Codex-reviewed): #139 screening reads both profile formats
  (`_emp_field()` in `dashboard.py`); #140 openers read both formats; #141 new Crustdata
  endpoints only, credits display fixed, thin search rows no longer saved to `profiles`,
  legacy search + `enrich.py` deleted; #142 combined search (description + filters,
  checkbox "Also apply the filters below"); #143 audit list.
- Where the rule lives: `screening_policy.py` line ~33 ("if the profile lacks evidence it is
  satisfied, treat it as a fail"), ~59 ("A must-have is missing..."), binary `met` in the
  structured-output section (~157-217); `dashboard.py` `_verdicts_force_no_go()` ~4611
  applied ~4889-4902; `_tri()` already returns "unknown" for unrecognised values;
  `GO_CONFIDENCE_THRESHOLD = 7` ~4560; `_decision_to_fit_label()` maps GO/NO GO + score to
  "Good Fit" / "Maybe" / "Not a Fit". The policy's startup-experience paragraph (~75-78)
  already says "absent data is never evidence for OR against" for one criterion.
- The 20-candidate test: first 20 of kalamata's 86 pushed Dwelly candidates
  (`pipeline_candidates`, `position_id='dwelly-ai-applied-eng-eu'`, `smartlead_pushed`),
  ordered by `md5(linkedin_url || '20260924')`. All have fresh stored profiles with
  skills/summary (0 credits). Results so far: original brief 2 GO / 18 NO GO; production-AI
  line moved to nice-to-haves 14 GO / 6 NO GO. Every rejection said "does not verify /
  establish", none cited a contradiction.
- The Dwelly brief used (keep identical for comparability):
  Role: Applied AI Engineer at Dwelly (AI-first UK lettings and property management
  platform, $170M Series B), fully remote in the UK or Europe on UK hours, building
  production agentic systems in TypeScript/Node and Python.
  Must: At least 3 years of hands-on software engineering, backend (TypeScript/Node.js or
  Python) / Has built and shipped AI or agentic systems that ran in production (tool use,
  orchestration, structured outputs, evals, cost/latency/reliability), not just demos,
  courses or side projects / Based in the UK or Europe / Degree in Computer Science or a
  related technical field, or a strong engineering track record with another science degree.
  Nice: Came from an AI-native company or a product startup / Technical founder history
  (past founder now employed as an engineer) / Based in London, Spain, Portugal or Poland /
  Uses coding agents in their own daily engineering work / PostgreSQL, LangGraph /
  LangChain, LLM eval tooling.
  Exclude: Data scientist, ML researcher, academic or research-heavy profile, or an "Applied
  AI" title that is really classic ML / model training / Career mostly at IT consultancies,
  outsourcing or software houses / Own company or solo freelancing is the only current job /
  Currently at Dwelly.
  The Owner brief (for the blind test) is in `docs/GOAL-sourcingx-fix-and-expert-audit.md` block 8.
- Screening is not repeatable today: both Maybes of the recruiter run became NO GO on
  re-screen with the same data (gpt-5.6-luna path sets no temperature/seed). Expect some
  noise in counts; run the 20-candidate test twice if a number looks borderline.
- Cost facts: every new-search result is thin (no skills/summary) so AI Screen tops it up at
  1 Crustdata credit each via `enrich_thin_profiles_for_batch()` (~dashboard.py 10206);
  the AI Screen estimate (~9947) ignores that and is ~4x low on tokens (2 model calls per
  candidate, ~$0.0025-0.0035 each on luna). `crustdata_search.log_search_usage()` exists but
  nothing calls it. `usage_tracker.log_salesql()` logs every lookup as a credit, misses included.
- Openers: `email_generator.py` `build_email_prompt()` ~213 hard-codes "at an Israeli tech
  company"; rule "NEVER start opener with 'Your'" at ~220 is ignored in practice; model
  gpt-4o-mini. Kalamata's openers (in `pipeline_candidates.email_opener`) cite a concrete
  built thing and end with an observation — use them as the quality bar.
- Crustdata account quirks: `/person/search` refuses `years_of_experience_raw` and
  `recently_changed_jobs` in `fields` (they work as filters). Failed calls cost nothing.
- App-driving quirks: Playwright uploads only from inside the repo (`.playwright-mcp/`,
  git-excluded); if Playwright says "browser already in use", another idle session holds it.
  The built-in browser is not signed in to claude.ai, so it cannot view artifact pages.
  Streamlit checkboxes inside expanders need a real mouse click in automation. Screenshots
  with candidate names must be blurred/cropped before going on a page.
- The SalesQL "All profiles already have emails!" message after a lookup that found none:
  reproduced, root cause not pinned (mask ~dashboard.py 10712-10720 on rerun).
- Leftovers for repo-tidy: folders `C:\Users\gehta\_sx_migrate`, `_sx_combined`, `_sx_opener`,
  `.claude/worktrees/agent-a9ad4863dc2ffed8e` (unregistered, Windows-locked).
- Deliberately NOT in this goal: making combined search the default, multi-country filter,
  experience-maths fixes, hidden generic hard filters, CI allow-list — later goals.
- Alexey's answers (2026-09-24): scope = everything the experts suggested; blind test
  labelled by the two experts (not recruiters; note it is AI judging AI); budget $3 + 10
  credits; merge when Codex is clean and CI green.
- Running spend for this goal: $0.00, 0 credits.

### Progress log (session 1 of this goal, 2026-09-24)
- Shared-table check: agent-kalamata `core/db.py:1617` filters `latest_screening` by
  `screening_fit_level`, `:1987` counts by it; Supanova `db_core/db.py:961` filters by it.
  Nobody reads `screening_results.screening_notes`. Chosen storage: needs-verification rows
  write `screening_fit_level='Maybe'` and `screening_notes='Needs verification: a; b'`.
- Launched (Sonnet, background): A on `feat/needs-verification-verdict`, C on
  `fix/opener-concrete-detail`, blind-sample prep into scratchpad `blind/`.
- A opened PR #144. My review: flip NO GO -> NEEDS VERIFICATION would also override
  NO GOs from generic hard filters / stability / experience-limit; sent back to add
  `hard_filter_failed` + respect the Python verdicts. Also `db.py` batch save now writes
  `screening_notes` (was silently dropped) — that is the sanctioned storage.
- Blind sample built: 40 profiles (Dwelly 7/7/3/3, Owner 7/7/1/5 — only 1 SourcingX Good
  Fit exists for Owner). Fixed 2 over-masked files (P01 "San Francisco", P33 "PhD").
  Pack `blind/label_pack.md` (160k chars), 0 links/emails. Fable + Astra labelling started.
- C's first agent wrote nothing in 15 min: stopped, relaunched with a tighter brief.
- Blind labels done: `blind/fable_labels.json` (7 outreach / 20 NV / 13 reject),
  `blind/astra_labels.md` (gpt-6-astra: 4 / 24 / 12). Agree 31/40; both-outreach 3,
  both-reject 10, both-NV 18. SourcingX's new verdicts still to run (after A merges).
- PR #144 Codex round 1: 3 real findings (GO + stability fail passes; stale notes on
  re-screen; NV rows leak into Maybe filter/download). Posted; fixes sent to A's agent.
- PR #144 MERGED (b94de2c) after 3 Codex rounds (round 3: legacy freeform schema only,
  out of scope — app always passes a structured brief); CI green; 787 passed on my run.
- PR #145 (openers): 4 Codex rounds used; round-4 fix (blank opener if retry still breaks
  rules) in progress. WAITING ON ALEXEY (asked ~20:35): allow a 5th Codex round then merge?
- B launched (Opus) on `feat/cost-before-paid-clicks`. Re-screen of 20 Dwelly + 40 blind
  launched (Opus), scratchpad `rescreen/`, cap $1.00, 0 credits.
- #145: round-4 fix pushed (770f492), 748 passed locally, CI green. PARKED (no answer in
  10 min): 5th Codex round + merge waits for Alexey.
- B opened PR #146. Root causes of the false "already have emails": empty group, result
  erased by immediate rerun, filtered rows dropped from session, Load tab counted one table
  and enriched another. SalesQL: misses free (global salesql-api skill) -> logged 0 credits,
  status not_found. All 5 Crustdata search call sites now log. Codex r1: 1 finding (Usage
  tab lookups metric summed credits) -> sent back.
- PR #146 MERGED (f38b45b): Codex r2 clean, CI green, 808 passed on my run.
- RE-SCREEN DONE (run start 2026-09-24T18:32:32Z, gpt-5.6-luna, 121 model requests,
  $0.2247, 0 credits; Dwelly brief without trailing period so jd_hash matches older rows).
  Set 1 (20 Dwelly, original brief): GO 2 / NEEDS VERIFICATION 15 / NO GO 3 (was 2/-/18),
  confirmed by SQL on screening_results (fit 'Maybe' + notes 'Needs verification:%').
  All 3 NO GOs cite evidence: consultancy exclusion; backend shown only as PHP/frontend;
  research-heavy exclusion. Files: scratchpad `rescreen/`.
- BLIND TEST DONE: `blind/blind_results.csv|json` (40 rows: fable, astra, sourcingx_new,
  sourcingx_old, kalamata, group). SourcingX new: GO 2 / NV 24 / NO GO 14. Both-outreach 3
  -> SX NO GO 0 (2 GO, 1 NV). Both-reject 10 -> SX GO 0 (7 NO GO, 3 NV). SX matches the
  expert consensus on 23/31. SX NO GO where neither expert rejects: 4 (P10 stability rule,
  P36 all-big-company vs startup must-have, P07+P30 still "not evidenced" -> old habit leaks).
- LIVE CHECK DONE (18:48-18:51 UTC): description search 10 results -> log row crustdata
  search 0.3 credits; AI Screen cost line "2 profiles need a top-up = 2 Crustdata credits ·
  AI ≈ $0.01 for 3" -> log batch_enrich 2 credits + 3 openai screen rows $0.0039; SalesQL
  "Enrich 1 profiles with Emails (1 lookups)" -> "SalesQL found 0 of 1 emails (1 lookups,
  0 credits)", log salesql email_lookup 0 credits status not_found. Found "$" pairs rendered
  as LaTeX -> PR #147 (escape) MERGED (cffc2bb), Codex clean, CI green, verified in app.
- Running spend: ~$0.229 model, 2.3 Crustdata credits, 0 SalesQL credits.
- SPEND (api_usage_logs since 17:40 UTC): openai screen 124 requests $0.2286; crustdata
  search 0.3 + batch_enrich 2 credits; salesql 1 lookup 0 credits. Expert/Codex runs are
  subscription, not logged.
- Audit page updated in place (version 2): "What changed" section, openers marked waiting.
  `docs/sourcingx-audit-2026-09.md` has a "What changed" section (items 1,2,3,4,8).
- GOAL STATUS: WAITING. Open: #145 needs Alexey's call (5th Codex round / merge / leave),
  then 3 Dwelly GO openers in the app side by side with 3 kalamata openers, page + docs
  update for item 4. Pop-up asked 2026-09-24 was dismissed; asked again in chat 2026-09-25.
