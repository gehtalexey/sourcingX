# SourcingX audit, September 2026

Ranked by how much each item changes what a recruiter gets out of the app. Found by two
outside experts (Claude Fable 5.1 and GPT-6 Astra, independently, two rounds: first the
code, then a written record of a real recruiter run) and by Claude during the fix, the
bake-off re-run and the Dwelly recruiter run of 2026-09-24. Every item was checked by
Claude before it went on the list: **verified** = confirmed by a file read, a query, a test
or a live run; **expert opinion** = could not be settled that way.

## The one decision to make first

**Does a must-have mean "proven on the LinkedIn profile" or "not contradicted by it"?**
SourcingX answers "proven"; kalamata answers "not contradicted". Measured on 20 candidates
kalamata had already approved and pushed to outreach for Dwelly:

| Brief | SourcingX says yes |
|---|---|
| "Shipped production AI/agentic systems" as a must-have | 2 of 20 |
| Same line moved to nice-to-haves, nothing else changed | 14 of 20 |

All 18 rejections said the profile "does not verify / does not establish" production
shipping; none pointed at anything that contradicted it. The bake-off re-run tells the same
story: after the data fix, agreement with kalamata only moved from 69% to 70%.

Both experts agree this is #1. They differ on the cure: Fable would make "absent is not a
fail, infer from title/company/skills" the general rule; Astra would keep the requirement
strict but add a third answer, "needs verification", and send those people to review
instead of rejection. Either way the rejection stops being automatic.

## The list

| # | What is wrong | Where | Evidence | Fix | Effort | Status |
|---|---|---|---|---|---|---|
| 1 | A must-have with no evidence counts as failed, so good people are rejected unseen | `screening_policy.py` line 33 ("if the profile lacks evidence it is satisfied, treat it as a fail"), line 59, binary `met` in the structured output; `dashboard.py` `_verdicts_force_no_go()` ~4611 applied ~4889 | Flip test above (2/20 -> 14/20); 115 of 149 Owner rejections in the bake-off cite an unproven must-have | Add "unclear / needs verification"; route those to Maybe or review; remove the line-33 clause | small-medium | verified |
| 2 | Same person, same brief, different answer on a second run | `screen_profile()`; gpt-5 path sets no temperature/seed (~4703-4714); GO/Maybe cut at score 7 (`GO_CONFIDENCE_THRESHOLD` ~4560) | Recruiter run: both Maybes became NO GO when re-screened | Mostly fixed by #1; also show "unclear" verdicts instead of collapsing them | small | verified (run); cause partly expert opinion |
| 3 | Money is spent without being shown or logged | AI Screen top-up ~10206-10212 not in the estimate (~9947, also ~4x low on tokens); SalesQL button shows no cost; `crustdata_search.log_search_usage()` is never called; Load tab "3 credits each" (~7758) and help "3 credits/profile" (~12617) though enrichment now costs 1; `usage_tracker.log_salesql()` records every lookup as a credit, misses included | Run: screening 10 search results showed "$0.007" and spent 9 Crustdata credits; search credits (0.75/0.30/0.30) appear only on screen | Show "N thin profiles = N credits" and "N lookups" before each button; log every search; fix labels | small | verified |
| 4 | Openers are generic and written as if from an Israeli company | `email_generator.py` `build_email_prompt()` ~213 ("You are ... at an Israeli tech company"), Israeli examples ~229-243, "NEVER start opener with 'Your'" ~220 is ignored, "aligns well" not banned, model gpt-4o-mini | Run: both openers were one sentence starting "Your experience ... aligns well with ..."; kalamata's cite a concrete fact ("replacing a PHP backend with Kafka and Kubernetes") | Require one concrete thing the person built; sender and company from the position; ban filler; check the rules before showing drafts | medium | verified |
| 5 | The structured filter search is a weak front door; the new combined search is much better | Title filter also matches headlines (`crustdata_search.build_filters()` ~357-381); default sort by connections (~6603); Country is single-choice (~6335); smallest limit 25 (~6498); description search shows a meaningless total ("9 of 24,682,821") | Run: structured 25 results, ~8 real AI engineers, 0 of 10 screened passed; combined 10 of 10 real AI engineers in 7 s | Make description + filters the default; relevance sort; multi-country; hide the total for description searches | small-medium | verified |
| 6 | Hidden rules the recruiter never typed, and two definitions of a short stint | `screening_policy.py` Hard Filters (~53-59: telecom/banking/outsourcing careers, 8+ years at one company, 3+ stints under 2 years) vs `compute_role_durations()` (~4383, under 12 months) | Both texts read | Show the generic rules as optional switches; one stint definition | medium | verified (rules); impact expert opinion |
| 7 | Experience maths: gaps between jobs count as experience; unknown dates show as "0m" | `compute_role_durations()` ~4420-4453 (first job to today, minus military), ~4325 | Astra reproduced ~2y8m of jobs shown as 16y8m; "? - ? = 0m" seen in tests | Sum real job periods; say "unknown" and leave it out of stint counts | medium | verified (code); reproduction by Astra |
| 8 | Workflow traps that lose work or say the wrong thing | "Send to Filter Tab" clears screened results with no choice (~7070-7073); top banner always says "uploaded from CSV" (~5608); after a SalesQL lookup that found nothing, "All profiles already have emails!" | Run: screened results vanished on the next send; banner wrong; 2 lookups, 0 emails, message said the opposite | Ask add-or-replace; banner from the real source; "found N of M" after lookups | small | verified (run); SalesQL message cause not pinned |
| 9 | Failures look like verdicts or are hidden | Failed enrichment still screened, then "insufficient data" NO GO (~5127 then ~10206); "Error" rows hidden by the default filter (~10540); background save errors swallowed (~6873); SalesQL "failed" and "no email" look the same (~1908-1926); batch retry never fires because errors become rows (~4966/~5377) | Code read by both experts | An "incomplete" state with retry; error count next to GO/MAYBE/NO GO | medium | expert opinion (paths read) |
| 10 | CI runs 32 of 49 test files; skipped ones include the live-database guard and the wrong-person identity check | `.github/workflows/test.yml` allow-list | 17 files never run, incl. `test_no_live_db_in_tests.py`, `test_linkedin_identity_matching.py`, `test_structured_screening_v2.py` | Run everything, skip only the live/e2e files | small | verified |
| 11 | "Maybe" means different things in different places; the shared table stores only the label | `_decision_to_fit_label()` (Maybe = low-score GO) vs comment ~10532 ("borderline NO GO") vs policy ~88 | Texts read | One definition; store GO/NO GO too | small | verified |
| 12 | Found emails may not be saved to the shared table | `db.py` `update_profile_emails_batch()` ~1272 uses the upsert route migration 025 says rejects existing identities | Not testable today (the run found no emails) | Save through the supported route; show failures | medium | expert opinion |
| 13 | Paid top-up even when a richer copy is already stored; headline-only profiles still get a paid screening | `fetch_raw_data_for_batch()` ~5028, `enrich_thin_profiles_for_batch()`, `screen_profile()` ~4814 | Code read | Check the stored copy first; skip profiles with no job history | small-medium | expert opinion |
| 14 | A GO with no per-must-have answers still counts as Good Fit | `_verdicts_force_no_go()` accepts empty verdicts (deliberate per its docstring) | Astra reproduced with a mocked reply; Fable ranks it low (a false GO is easy to spot) | Require an answer per must-have or mark incomplete | medium | FIXED, PR #150 (a missing or repeated answer is now NO GO) |
| 15 | Nice-to-haves cost a second full model call per candidate | `screen_profile()` ~4856-4865 | 2 calls per candidate in `api_usage_logs` (592 for 296) | Experts split: merge into one call (Fable) vs keep separate so bonuses cannot affect rejection (Astra); compare quality first | small | verified (cost); fix is a judgement call |
| 16 | New-format profiles: blank Location column; degree and field lost from search results | `screen_profiles_batch()` ~5422 reads only `location`; translator emits only school names | Seen in the app; code read | Fall back to `region`; emit full education | small | verified / expert opinion |

## Fixed during this goal (2026-09-24)

| Problem | Fix |
|---|---|
| Screening could not read ~88% of stored profiles (no jobs, no location); durations cache could mix up two candidates; tenure rule skipped | PR #139 — "Insufficient data" verdicts 37 -> 6 on the bake-off re-run |
| Openers crashed on new-format profiles | PR #140 |
| Default filter search was the legacy endpoint that stops working 2026-09-30; credits display broken; the new search failed on every call (asked for two fields this account cannot read); thin search rows saved to the shared table as "enriched"; legacy search and `enrich.py` removed | PR #141 |
| No way to combine a description with filters | PR #142 — combined search, used in the run (10/10 relevant, 7 s) |

## What changed (trust-fixes goal, 2026-09-24)

| Item | Status | Change |
|---|---|---|
| 1 | FIXED, PR #144 | Each must-have is now met / not met / needs verification. Only a contradiction rejects. Unproven must-haves go to a visible "Needs verification" group with its own download, kept out of outreach. Stored as "Maybe" plus `screening_notes` "Needs verification: ...", because agent-kalamata and Supanova filter on `screening_fit_level`. |
| 2 | IMPROVED, PR #144 | The biggest source of flips (unproven = fail) is gone. Repeatability itself was not re-measured. |
| 3 | FIXED, PRs #146, #147 | AI Screen shows "N profiles need a top-up = N Crustdata credits · AI ≈ $X" from the last run's real cost. Email buttons say "N lookups". All 5 search call sites write to `api_usage_logs`. Enrichment labels say 1 credit. SalesQL misses are logged as 0 credits (SalesQL does not charge for them). |
| 4 | PR #145, waiting to merge | Openers cite one concrete thing from the profile, take the company from the position, and are checked in code for banned starts and words; an opener that breaks the rules twice is dropped. |
| 8 | PARTLY FIXED, PR #146 | After a lookup the app says "SalesQL found N of M emails". Four causes fixed: empty group, result erased by an instant reload, hidden rows dropped from the session, Load tab counting one table and looking up another. "Send to Filter Tab" and the wrong banner are still open. |

**The 20 kalamata-approved Dwelly candidates, original brief (must-have kept):**
before 2 GO / 18 NO GO; now **2 GO / 15 Needs verification / 3 NO GO**. All 3 NO GOs cite
evidence: a consultancy career, backend only in PHP, a research-heavy profile.

**Blind test (40 anonymised profiles, Dwelly and Owner).** Fable 5.1 and GPT-6 Astra each
labelled every profile outreach / needs verification / reject from the brief alone. They
agreed on 31 of 40. This is AI judging AI, not recruiters.
- Both experts say outreach (3): SourcingX says NO GO to **0** (2 GO, 1 needs verification).
- Both experts say reject (10): SourcingX says GO to **0** (7 NO GO, 3 needs verification).
- SourcingX matches the experts' shared answer on 23 of 31.
- SourcingX says NO GO where neither expert rejects: 4. One is its own short-stint rule,
  one is a real contradiction, and 2 still reject for "not shown" (the old habit, about 1 in 20).
- The "both say outreach" group is small, so treat the 0 as encouraging, not proven.

**Live check (18:48-18:51 UTC):** a 10-result search logged 0.3 credits; screening 3 profiles
predicted 2 top-up credits and charged 2; one email lookup found nothing, said "found 0 of 1"
and logged 0 credits.

## What changed (automatic GO / NO GO goal, 2026-09-25)

**Decision (Alexey, 2026-09-25): no manual "Needs verification" bucket. Every candidate ends
GO or NO GO on its own.** This replaces the "Needs verification" group added in PR #144. PR #150.

The rule, decided in code from the per-criterion answers plus the score (the model's own
`decision` field is never trusted):
- A contradicted must-have, a matched exclusion or a hard filter is NO GO.
- Otherwise GO only at a score of 7 or more (`GO_CONFIDENCE_THRESHOLD`, one constant). A
  must-have that is not shown but clearly implied by the visible career keeps the score up and
  gives GO with a "Verify in call" note. One that is not shown and not clearly implied keeps
  the score at 6 or below, so NO GO with the reason "Not shown and the visible career doesn't
  clearly imply: X".
- Stored fit labels stay "Good Fit" and "Not a Fit". No new value is written, because
  agent-kalamata reads that column. Old "Needs verification" rows display as NO GO until they
  are screened again.

**How answers are tied to criteria.** The prompt numbers every criterion (M1, M2 for
must-haves, E1, E2 for exclusions) and the model must send the id back. A criterion is
answered only by exactly one verdict carrying its id. A missing, repeated, unknown-id or
invalid-valued answer makes the screening incomplete: NO GO, with "Screening incomplete" or
"not judged" in the reason. A must-have with no answer is no longer treated as "needs
verification" (audit item 14). Matching by wording was tried first and dropped: Codex rounds
2 to 5 each found another way two similar requirements could be mixed up (Python and Java,
C++ and C#, .NET and NET). Fable 5.1 and GPT-6 Astra, asked separately, both recommended ids.
After the change, the last two Codex rounds found no problem with the matching.

**Re-screen, 100 stored profiles (96 unique person and brief pairs), model gpt-5.6-luna,
0 Crustdata credits, $0.32.** 0 of 100 answers were incomplete.

| Role | Both experts: outreach | SourcingX NO GO (missed) | Both experts: reject | SourcingX GO (wrongly approved) |
|---|---|---|---|---|
| Dwelly | 2 | 1 | 6 | 1 |
| Owner | 1 | 0 | 4 | 0 |
| Autofleet | 1 | 0 | 2 | 0 |
| ScaleOps | 5 | 1 | 8 | 0 |
| **All 80 blind profiles** | **9** | **2** | **20** | **1** |

- Both misses scored exactly 6. The wrongly approved profile scored 8.
- Bar sensitivity: at 6, 0 missed but 6 of the 20 clear rejects approved; at 7, 2 missed and
  1 approved. The bar stays at 7.
- The 20 candidates kalamata approved for Dwelly: **5 GO / 15 NO GO** (was 2 GO / 15 needs
  verification / 3 NO GO). 14 of the 15 NO GOs scored exactly 6 because the shipped-AI
  must-have is not shown and not clearly implied; 1 is a contradiction. This is the direct
  cost of removing the manual bucket: kalamata approves these people, SourcingX does not.
- Caveats: the experts are AI, not recruiters; only 9 profiles were "both say outreach";
  screening is not perfectly repeatable, so borderline counts can move by 1 or 2; the blind
  id lists were rebuilt from the label packs by matching profile text (80 of 80 matched, 2
  differed only in a redacted name).
- Results were saved to the shared `screening_results` table as Good Fit or Not a Fit only
  (96 rows). Dwelly and Owner used the same brief text as 2026-09-24, so their older rows for
  the same people are replaced.

## What the experts said

**Fable 5.1:** SourcingX can be trusted today to find people (the combined search is good and
fast) and to say no to obvious mismatches, but not to say yes: it rejects most good
candidates because LinkedIn rarely spells out what a brief asks for, and it can give the same
person a different answer tomorrow. Until the "unproven is not failed" change is made, treat
every NO GO whose reason says "not established" or "does not verify" as unread, not rejected.
Fix the silent credit spend before the next real run, and rewrite the opener prompt before
anyone sends one.

**GPT-6 Astra:** SourcingX can support supervised searches today, but its NO GO decisions are
not yet reliable enough to discard candidates automatically. The first change should separate
an unproven requirement from a proven mismatch while keeping the hiring manager's requirement
intact. Then check a mixed sample of approved, rejected and incomplete profiles against
recruiters' own judgements, made without seeing either system's answer; agreement with
kalamata alone cannot establish correctness.

## Suggested next goals
1. Item 1 (tri-state must-haves), then re-screen the same 20 kalamata-approved Dwelly
   candidates with the must-have kept: 12+ passing means the policy was the cause.
2. Items 3 and 8 together (cost shown before every paid click; no silent loss of work).
3. Item 4 (opener prompt) and item 5 (combined search as the default).
4. Decide who sets the bar for "shipped AI to production": SourcingX now says NO GO to 15 of the
   20 people kalamata pushed. Repeat the blind test with recruiters instead of AI experts.

Raw expert answers and the run record: session scratchpad (`consult-*-audit*.md`, `run-record.md`).
