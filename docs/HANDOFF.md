Written: 2026-09-25 ~09:50 (Israel) on Alexey_Geht
Branch: wip/auto-decide-blindtest-2026-09-25 (this note + blind-test files). Code under review: PR #150 on `feat/auto-decide-unproven`.

## Where things stand
- Merged on master (2026-09-24/25): #144 three-state must-haves, #145 openers (concrete detail,
  company from position, rule check + retry, 6 Codex rounds), #146 cost shown before every paid
  click + honest logging, #147 "$" escape. (#149 nul-strip came from another session.)
- Alexey decided (2026-09-25): NO manual "Needs verification" bucket. Every candidate must end
  GO or NO GO automatically. He approved removing the NV decision, its results section, filter
  option, download and 4th counter.
- Experts consulted on the rule (brief + Astra answer in `docs/blindtest/consult-*.md`). They
  split: Astra = 4 states (met/implied/not_met/unproven); Fable = keep 3 states, let the score
  decide (GO only at >= 7), plus guard sentences in the prompt. Took Fable's (tunable constant
  `GO_CONFIDENCE_THRESHOLD`), with Astra's wording on what may count as implied.
- PR #150 (`feat/auto-decide-unproven`) implements it: contradiction / exclusion / hard filter
  -> NO GO; else GO iff score >= 7 with "Verify in call: X" in `screening_notes`; else NO GO
  "Not shown and the visible career doesn't clearly imply: X". Fit labels only Good Fit / Not a
  Fit (no new "Maybe"). Missing must-have or exclusion verdicts never pass. Prompt: "insufficient
  data" removed as a score reason; "defining requirement unproven -> score max 6"; "company
  implies a skill only when its product IS the skill".
- Codex round 1 on #150: 3 findings, all fixed in `dbf2a49` (885 passed per the agent;
  NOT yet re-run by me). Codex round 2 NOT run yet.
- Second blind sample built and labelled: 40 anonymised profiles, Autofleet full stack
  (`autofleet-fullstack-senior-il`) + ScaleOps backend (`scaleops-backend-industry-il`), ids
  Q01-Q40. Fable: 14 outreach / 15 NV / 11 reject. Astra: 6 / 23 / 11. Agree 30/40
  (6 both-outreach, 10 both-reject, 14 both-NV).
- First blind sample (P01-P40, Dwelly + Owner) labels from 2026-09-24 are in
  `docs/blindtest/set1_dwelly_owner/`.

## Still in flight
Nothing running. PR #150 is open, waiting for Codex round 2.

## The exact next step
1. Re-run CI tests on #150 locally, then Codex round 2 (`codex-pr-review` skill,
   `--base master`). Max 4 rounds; merge when clean + CI green (Alexey asked for this work).
2. After merge: re-screen with the NEW code, stored profiles only, 0 credits, about $0.45 of
   gpt-5.6-luna (Alexey OK'd ~$0.40): the 20 kalamata-approved Dwelly people (first 20 of
   `pipeline_candidates` position `dwelly-ai-applied-eng-eu`, `smartlead_pushed`, ordered by
   md5(linkedin_url || '20260924')), blind set 1 (P01-P40) and blind set 2 (Q01-Q40). Use the
   approach in `docs/blindtest/rescreen_scripts/` (call `screen_profile()` directly with stored
   raw profiles, never the batch flow that tops up thin profiles; save through
   `update_profile_screening_batch()`). Briefs: Dwelly/Owner in
   `docs/GOAL-sourcingx-fix-and-expert-audit.md` block 8 and `docs/GOAL-sourcingx-trust-fixes.md`
   block 8 (Dwelly role line WITHOUT a trailing period so jd_hash matches); Autofleet + ScaleOps
   in `docs/blindtest/set2_autofleet_scaleops/label_pack.md`.
3. Report per role: of "both experts: outreach", how many SourcingX NO GO (missed); of "both
   experts: reject", how many SourcingX GO (wrongly approved). If misses are high, lower
   nothing blindly: look at the scores first (the threshold is one constant).
4. Update the audit page in place (https://claude.ai/artifact/Sfn6q6sQuR5qNzxYjWZctf) and
   `docs/sourcingx-audit-2026-09.md` (docs PR #148 is open, never reviewed; fold into it).

## Watch-outs
- The blind manifests (id -> LinkedIn URL -> kalamata verdict) were NOT committed (they hold
  URLs). They are on the dev PC scratchpad only. The build scripts are deterministic
  (md5 ordering), so re-running `build_blind_sample.py` / `build_blind_sample2.py` against the
  shared DB recreates them; check the ids match the label packs before scoring.
- Yesterday's stored scores cannot be reused for the new rule: the old prompt squeezed unproven
  people to 5-6 (one "both experts: outreach" profile scored 6). Must re-screen.
- Kalamata reads `screening_results.screening_fit_level`: never write a new value there.
- Codex on Windows: set `USERPROFILE=C:\_cxhome` and `CODEX_HOME=<real .codex>` for the run.
- Screening is not perfectly repeatable (luna, no seed); borderline counts can move by 1-2.

## Waiting on Alexey
Nothing blocking. (Docs PR #148: he interrupted its Codex run; ask before reviewing.)

## Did not travel
- Blind manifests and re-screen outputs with URLs (dev PC scratchpad).
- Leftover review worktrees `C:\Users\gehta\_cr144`..`_cr150` and agent worktrees under
  `.claude/worktrees/` (repo-tidy).
- The Streamlit app preview started on the dev PC (port 8501).
