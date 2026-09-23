# Make SourcingX's screening model swappable, and get Luna + Jev working

## 1. Outcome

**Correction, 2026-09-23, mid-goal:** SourcingX already switched its live screening
from `gpt-4.1-mini` to `gpt-5.6-luna` on 2026-09-17 (PR #131, merged before this goal
was written) — including the parameter fixes (`max_completion_tokens`, no
`temperature`), a retry on empty responses, and per-call cost logging. None of that was
known when this file was first drafted; the rest of this document has been corrected to
match. See the handover section for the full story.

SourcingX can screen candidates with any of three models — `gpt-4.1-mini`, the live
`gpt-5.6-luna`, or TypeSafe's Jev — by changing one setting, not editing code (today the
model is a literal string typed into ~4 places in `dashboard.py`). Jev actually works
end to end (client built from kalamata's real code — done, see handover). On top of
that: a report screening 200 real profiles with both Luna and Jev, laid next to how
agent-kalamata already screened those same profiles, so Alexey can see where the
verdicts agree or disagree with kalamata's own pipeline.

**Parked, not part of this goal (Alexey's call, 2026-09-23):** a proper "screening
incomplete" outcome for ambiguous model answers. The current code already makes a
deliberate, commented choice here (`_verdicts_force_no_go` in `dashboard.py`, ~line
4571) — an ambiguous/missing verdict never overrides an existing GO, on purpose, so a
malformed answer can't wrongly flip a real qualified candidate to a reject. That's a
real product tradeoff, not an obvious bug, so it's not being touched by this goal.

## 2. Why it matters

Alexey wants to test whether a cheaper model (Jev) can do the same job as Luna, and
compare both against what kalamata's own pipeline already decided on the same
candidates — real evidence, not another AI's opinion of the code. Today the model
choice is hardcoded in several places, which makes that comparison harder to run
cleanly and makes any future model change (a fourth option, a per-position choice, etc.)
a code edit instead of a setting. This goal makes that swap safe and gets the actual
comparison data in front of Alexey.

## 3. Inputs

**SourcingX** (`C:\Users\admin\Claude Projects\SourcingX`, this repo):
- `dashboard.py` — `_screening_api_call()` (~line 4604, already handles `gpt-5.x`
  parameter differences and retry-on-empty — read it before changing it),
  `screen_profile()` signature (~line 4734), live screening call site currently a
  literal `ai_model = "gpt-5.6-luna"` (~line 9891), batch screening (~line 10105, same
  literal), email generation (~line 10894, `gpt-4o-mini` literal — leave as-is, out of
  scope). Build item A's real job now: replace those literal strings with one settings
  lookup, NOT re-fix parameters/retry/logging that already work.
- `structured_screening.py` — a separate, currently **dead** screening module
  (Anthropic `claude-haiku-4-5-20251001` default); reachable only from its own UI file,
  which nothing else imports. Leave it alone — don't delete it, don't treat it as the
  live path.
- `usage_tracker.py` — PR #131's second commit already added a `gpt-5.6-luna` pricing
  entry and fixed flex-tier cost tracking (Codex found this gap and it was fixed before
  merge). Check whether an *unknown* model (e.g. Jev, once wired in) still silently
  falls back to `gpt-4o-mini` pricing or now warns — verify before assuming either way.
- `compare_screening_modes.py` — an existing model-comparison script; it currently
  exercises the *legacy freeform* screening prompt, not the live per-requirement prompt
  the dashboard actually uses (lines ~124-127) — that's a real bug, not just a gap. Also
  check whether its own model list/pricing still references `gpt-4.1-mini` as if it
  were live — it isn't, anymore.

**agent-kalamata** (`C:\Users\admin\Claude Projects\agent-kalamata`, read-only
reference — do not change anything in this repo):
- `pipeline/screen_engine.py` — `DEFAULT_MODEL`, `_DEFAULT_MODEL_BY_PROVIDER`,
  `get_screen_model()` (~lines 750-794: provider default < `config.json` <
  per-position override < explicit argument), `_call_openai_model` (~line 2541+:
  `max_completion_tokens` not `max_tokens`, no `temperature`, fixed seed, retries),
  usage-event logging (`_make_usage_event`, ~line 2125)
- `pipeline/prescreen_pass.py` (lines 101-108) and `pipeline/db_helpers.py` (lines
  328-333) — confirm, verified by reading the actual code this session, that
  `gpt-5.6-luna` is kalamata's real production screening model today, at roughly
  $0.00088/candidate
- Jev's client code does **not** exist on this repo's checked-out `master`. It only
  exists on the unmerged branch `origin/feat/jev-phase2-shadow-runner`
  (`scripts/jev_shadow_runner.py`, `scripts/jev_compare.py`). Read it with
  `git show origin/feat/jev-phase2-shadow-runner:scripts/jev_shadow_runner.py` from
  inside the agent-kalamata folder — do not check that branch out, do not touch
  agent-kalamata's working tree at all.

**The 200 test profiles:** must be candidates agent-kalamata has already screened for
one of its own positions, so there's a real kalamata verdict to compare against — not a
random sample from SourcingX's `profiles` table. Kalamata's own verdicts live in its
`pipeline_candidates` table (agent-kalamata repo, `core/db.py`); confirmed this session
that kalamata also writes a copy into the shared `screening_results` table with
`source_project='autopilot'` (SourcingX writes its own as `source_project='sourcingx'`,
see `db.py` ~lines 1052-1184) — read from whichever is easier, they should be the same
data. Screen each profile against the **same job description kalamata used** for that
candidate, pulled from kalamata's own position config, not a generic SourcingX JD —
otherwise the comparison isn't apples-to-apples. First check how many candidates
actually satisfy "kalamata already screened them" — there may be fewer than 200 that
qualify; report the real number rather than forcing exactly 200.

**Prior work this session:** two independent outside experts (Fable 5.1 and GPT-6
Astra) were separately consulted on how SourcingX's and kalamata's screening differ.
Both agreed Luna needs different OpenAI call parameters, both independently flagged the
`compare_screening_modes.py` wrong-prompt bug above, both said Jev needs a client built
from kalamata's real code, not guessed at. Don't re-run that consult — the answer is
already known and summarized in this file; re-read it only if something below turns out
to be wrong.

## 4. Constraints

- Alexey is not a developer. Every update to him is in plain English, no jargon.
- Follow SourcingX's existing repo workflow (its own `CLAUDE.md`): feature branch, PR
  against `master`, real Codex review via the `codex-pr-review` skill, fixes applied on
  the same branch. Alexey merges — or Claude may merge only on his direct
  "merge it"/"merge now" said in that chat turn. Never merge because CI is green or
  Codex's review is clean; that's a signal, not permission.
- SourcingX, agent-kalamata, and Supanova share one Supabase database. As of the
  2026-09-23 pull, `dashboard.py` now writes completed batch results to a shared
  `screening_results` table that agent-kalamata's pipeline reads. Nothing in this goal
  may change what gets written there or how, beyond it correctly recording whichever
  model actually ran.
- No paid API call runs without Alexey's direct go-ahead, stated in dollars, every
  single time — including the first real test call proving Jev actually works, even
  though it will cost a fraction of a cent. "Just testing" is not implicit permission.
  A `--dry-run` flag, if one gets added, must mean no spend — verify that before relying
  on it.
- The 200-profile run is real money on top of the small parameter-check calls above —
  roughly 200 profiles x 2 models x ~2 calls each (main + nice-to-have, matching
  SourcingX's existing per-requirement shape) is on the order of 800 calls. Compute the
  actual estimate from real Luna and Jev pricing once both clients exist, state it in
  dollars, and get a separate explicit yes before running it — this is its own approval,
  not covered by the earlier go-ahead to build the code.
- The kalamata side of the comparison uses verdicts kalamata **already has stored** —
  nothing in this goal re-screens anything through kalamata or spends against its
  pipeline.
- Never delete or rewrite `structured_screening.py` or any other existing file without
  asking first and describing exactly what would be deleted and why.
- Jev's real request/response shape is not documented anywhere in this session — build
  its SourcingX client from kalamata's actual branch code (see Inputs), not from
  guessing at an API shape.
- Keep `gpt-5.6-luna` as the live default — that's already the production choice
  (PR #131). Nothing about today's live screening behavior changes until Alexey
  explicitly flips the setting.
- The "screening incomplete" outcome idea is explicitly OUT of scope for this goal
  (Alexey's call, 2026-09-23) — the current ambiguous-answer handling in
  `_verdicts_force_no_go` is a deliberate, commented design choice, not a bug to fix
  here. Don't touch that function.

## 5. Definition of done

- [ ] One place decides which model/provider screens a candidate, read from settings,
      not hardcoded per call site. Proven by: grepping `dashboard.py`'s live screening
      and email-generation call sites for literal model-name strings (`"gpt-4.1-mini"`,
      `"gpt-4o-mini"`, etc.) finds none outside the new settings module.
- [ ] Switching the screening model is a one-setting change, not a code edit. Proven
      by: changing the setting and restarting shows the live call site picking up the
      new model — a pasted before/after diff of the effective request body.
- [ ] An unknown or new model (Jev, once wired in) can't be silently mispriced. Proven
      by: `usage_tracker` raises a visible warning instead of falling back to
      `gpt-4o-mini` pricing when asked to price a model it doesn't recognize.
- [x] Jev has a working client function in SourcingX, built from kalamata's real Jev
      call code, that takes a candidate + JD and returns a verdict. **Done 2026-09-23** —
      `jev_client.py` + `test_jev_client.py` (34 mocked tests), PR #133, not yet merged.
- [ ] `compare_screening_modes.py` exercises the live, per-requirement screening
      prompt — the one the dashboard actually uses — not the legacy freeform path.
      Proven by: reading its call site shows it calling the same function, with the
      same prompt-building, as `dashboard.py`'s live path.
- [ ] The one real test call proving each new model returns a complete, parseable
      verdict happens only after Alexey has been told the exact cost and said yes.
      Proven by: the chat shows the ask and the yes before the call, and the call's
      actual response shown to Alexey afterward.
- [ ] Nothing about today's live `gpt-5.6-luna` screening changes behavior — this goal
      only changes *how* the model gets picked, never the live default itself. Proven
      by: the existing `pytest` suite still passes, unchanged, plus the live call site's
      default settings resolve to `gpt-5.6-luna` exactly as they do today.
- [ ] The change is on GitHub as a merged PR, reviewed by real Codex, merged only on
      Alexey's direct word. Proven by: the PR URL, Codex's review comments, Alexey's
      explicit "merge it" in chat, and the `gh pr merge` confirmation.
- [ ] The real count of eligible test profiles (candidates kalamata has already
      screened) is known before anything is spent. Proven by: a plain-English number
      reported to Alexey — "X profiles qualify" — before the spend ask below.
- [ ] 200 profiles (or the real eligible count if smaller) are screened by both Luna
      and Jev in SourcingX, against the same JD kalamata used per candidate. Proven by:
      a results file with one row per profile, per model, showing verdict + reasoning.
- [ ] Those results are laid next to kalamata's own stored verdict for the same
      candidate. Proven by: a comparison table/report showing, per profile, what
      today's model / Luna / Jev / kalamata each decided, with disagreements called out.
- [ ] The 200-profile run happened only after Alexey saw the dollar estimate and said
      yes, separately from the earlier code-build approval. Proven by: chat shows the
      ask and the yes before the run starts.

## 6. Stop conditions

Decide alone: file/module layout inside SourcingX, the exact settings-table schema,
which existing test file to extend, how the mocked tests are structured, code comment
wording.

Come back and ask, every time, no exceptions:
- Before any real (paid) API call to `gpt-5.6-luna` or Jev — state the dollar cost
  first, however small.
- Before the 200-profile run specifically — its own separate dollar estimate and yes,
  even after the small parameter-check calls have already been approved once.
- Before merging the PR to `master`.
- Before deleting or rewriting `structured_screening.py` or any other existing file —
  describe exactly what and why, per the never-delete rule.
- If kalamata's `feat/jev-phase2-shadow-runner` branch content contradicts what's
  written in this file (e.g. Jev's real call shape turns out materially different) —
  that changes the client's design, so check before building around a guess.

**A question never stalls the whole goal.** Before asking, start everything that
doesn't depend on the answer, in background subagents if useful. While other work
remains, ask without blocking: one line in chat (plus a `PushNotification` if Alexey's
away), and mark the item "waiting on Alexey" in this file's handover section below. No
answer within 10 minutes means that item is parked — carry on with the rest and raise
it again when he's back. Use a blocking pop-up only when nothing else is left to do.
The spend and merge approvals above are the exception: those always wait, full stop,
regardless of what else could proceed.

## 7. Deliverable

Two PRs against SourcingX's `master`, each Codex-reviewed: PR #133 (Jev client — done,
awaiting review) and a new PR for the model-settings module/table plus the fixed
`compare_screening_modes.py`. A short plain-English chat update when each is ready for
review, again when Codex's review comes back, and a final one-line summary once merged
(or parked, naming what's blocking).

Once that's merged: a second, separate deliverable — the 200-profile comparison report
(Luna vs. Jev vs. today's model vs. kalamata's stored verdict, per candidate, with
disagreements called out in plain English). This only happens after its own spend
approval (see constraints/stop conditions) and does not block the PR above from being
finished and merged first.

## 8. Handover — where things stand right now

**The mid-goal correction (read this first):** this goal was originally written on the
premise that SourcingX's live screening model was `gpt-4.1-mini` and needed a Luna
migration built from scratch (parameter fixes, retry-on-empty, per-call cost logging).
That premise was wrong by the time the goal started — SourcingX had already switched to
`gpt-5.6-luna` in PR #131, merged 2026-09-17, six days before this goal was written. The
mistake: after pulling latest changes into the local checkout mid-session, the pull's
file list was skimmed but the live model string itself was never re-checked, so the
stale "still on gpt-4.1-mini" belief carried through the rest of the session, into the
two independent expert consults, and into this file's first draft. A background agent
(build item A) was launched to build the Luna migration before this was caught; it was
stopped at the ~12-minute mark having only set up its test environment — no code
written, nothing lost. Alexey chose (2026-09-23) to drop the duplicate work entirely and
keep the rest of the goal (model-settings abstraction, Jev, the bake-off), and to leave
the "screening incomplete" idea out of scope rather than build it now. **Lesson for next
time:** after any `git pull`, re-verify facts already stated as true in an open goal
file or an in-progress plan, not just skim the changed-files list.

**Already done:**
- Two independent experts (Fable 5.1, GPT-6 Astra) were consulted on how SourcingX's
  screening differs from agent-kalamata's. Both agreed on the `compare_screening_modes.py`
  wrong-prompt bug; they disagreed on whether kalamata's live screener is Sonnet or
  Luna — verified by reading the actual code (`pipeline/prescreen_pass.py:101-108`,
  `pipeline/db_helpers.py:328-333`) that it's `gpt-5.6-luna`, confirming Fable's reading.
  (Their shared belief that SourcingX itself still ran `gpt-4.1-mini` was wrong, per the
  correction above — that wasn't something either expert could have caught, since it
  wasn't in what they were told to read.)
- **Jev client — DONE and MERGED.** PR #133, squash-merged to `master` 2026-09-23 as
  `99ec2ef`. `jev_client.py` + `test_jev_client.py` (40 tests, 1 intentional skip, no
  real API calls). Built from kalamata's real branch code: SDK is `typesafe_sdk`
  (now pinned `>=0.7.0` in `requirements.txt`), one call —
  `client.system_one(state=..., questions=...)` — three typed question kinds (`Noul`,
  `Score`, `Choice`), no free-text reasoning field at all (Jev returns typed decisions
  only). Real constants ported: `MIN_CONFIDENCE = 0.60`, `QUALIFY_SCORE_FLOOR = 6`.
  Went through 6 rounds of real Codex review (the standard 4 plus 2 extra Alexey
  explicitly approved) — 7 real bugs found and fixed: exclusions not enforced in the
  hard-filter decision, missing `requirements.txt` entry, dropped `region`/
  `company_name`/`employee_description` fields, the SDK's own retry stacking with the
  client's outer retry loop (9 requests instead of 3), and thin-raw-data/string-skills
  data loss. One Codex suggestion (folding nice-to-haves into the fit score) was
  declined — it would have broken parity with `dashboard.py`'s own deliberate design,
  where nice-to-haves are display-only and never touch score. Fable independently
  verified the final state against the real installed SDK before merge (not just the
  test mocks) and called it ready. **`jev_client.py` is not imported by `dashboard.py`
  or anything else yet** — it exists but nothing calls it.
  **Correction to this file's own earlier note:** item A does NOT wire Jev into
  `screen_profile()`'s live dispatch. Jev's real behavior on a low-confidence answer
  is to defer (`screening_result: None`, its own docstring says "read this as
  screening_incomplete"), and `dashboard.py` has no incomplete outcome for
  `screen_profile()` to return into — that's the parked work. `screening_models.py`'s
  `get_screen_model()` deliberately only ever resolves to `openai`/`anthropic`
  providers for the live path; Jev is listed in its table (for a shared display name
  other code can reference) but excluded from what `config.json` can select. The
  200-profile bake-off (item D) calls `jev_client.screen_with_jev()` directly, not
  through `screen_profile()`.

- **Item A — DONE, PR #134 open, 2 clean Codex rounds, awaiting Alexey's merge word.**
  `screening_models.py` (new) + `get_screen_model()`, wired into `dashboard.py`'s live
  call site, batch-resume fallback, and both `screen_profile()`/`screen_profiles_batch()`
  signature defaults. Fixed a Codex-found P1 along the way: `load_config()`'s Streamlit
  Cloud secrets override never copied `screen_model`, so the setting had no effect on a
  deployed app. **Known constraint hit and worked around:** the GitHub token available
  in this environment lacks `workflow` scope, so `.github/workflows/test.yml` cannot be
  edited by Claude here — Alexey explicitly declined to touch anything himself
  ("I don't touch anything... find a workaround"). Workaround used: new tests were
  folded into existing CI-covered files (`test_mocking_services.py`,
  `test_api_key_strip.py`) instead of adding new files to the workflow's focused list.
  **This constraint applies to any future PR that adds a new test file** — either fold
  new tests into an existing CI-covered file, or flag it for Alexey. `test_jev_client.py`
  (from PR #133) still isn't in CI's list either — a known, not-yet-fixed gap, left alone
  since folding a 642-line file into another would be worse than the problem.

- **Item C — DONE, PR #135 open, 3 clean/fixed Codex rounds, awaiting Alexey's merge
  word.** Fixed the wrong-prompt-path bug (now passes `screening_brief`, matching the
  live dashboard). Codex also caught two real side effects of that fix: the "quick" vs
  "detailed" combo comparison was silently comparing a combo against an identical copy
  of itself (mode has no effect under the structured path — dropped "haiku / quick"),
  and the nice-to-have bonus pass doubles the real call count per profile (added
  `CALLS_PER_PROFILE`). Round 2 caught that a "doubled but still stale" cost estimate
  was still a wrong number dressed as precise — `cost_summary()` no longer prints any
  dollar figure; it states plainly what's known and what's needed for a real one
  (wiring `tracker=` into the worker to report actual measured cost from a real run).

**In flight:** nothing right now. Next up: item D (the 200-profile bake-off) — blocked
on Alexey approving real spend, per this file's constraints. Not started.

**Facts established the hard way:**
- SourcingX has exactly one live screening path (`dashboard.py`'s own
  `_screening_api_call()`). `structured_screening.py` is dead code — don't mistake it
  for live, don't delete it.
- Kalamata's own provisional verdict (2026-09-22, not finalized) is that Jev can only
  clear about a third of Luna's rejects within Alexey's safety bar, with small savings
  — so this goal builds the *capability* to test Jev, it does not assume Jev wins.
- SourcingX pulled a change on 2026-09-23 where `dashboard.py` now writes completed
  batch results to a shared `screening_results` table that agent-kalamata's pipeline
  reads — this write path must keep working unchanged.

**Deliberately NOT being done:** no real paid test calls happen without a separate,
explicit ask each time; no switch away from `gpt-5.6-luna` as the live default (that's a
later decision, made from this goal's 200-profile results); no changes to
agent-kalamata's code or data; no fresh human grading of all 200 by Alexey (kalamata's
own stored verdict is the comparison point); no "screening incomplete" outcome work
(parked, Alexey's call 2026-09-23 — see Constraints).

## Parallel work

| Build item | Files it owns |
|---|---|
| A. Model settings (narrowed scope — just the lookup, not parameters/retry/logging, those already exist) | `dashboard.py` (only the ~4 literal-model-string call sites), new `screening_models.py` |
| B. Jev client | `jev_client.py`, its test file — **done, PR #133** |
| C. Fix `compare_screening_modes.py` | `compare_screening_modes.py` |
| D. 200-profile comparison run + report | new script (e.g. `bakeoff_vs_kalamata.py`), a results file/report — no shared files with A/B/C |

B is done. A and C both depend on knowing PR #133's final shape (the settings table
needs a Jev entry, `compare_screening_modes.py` needs to call it) — so do A after #133
merges, not in parallel with it. C depends on A's settings module too. Sequence:
review+merge #133 → build A → build C → (spend approval) → D.

D depends on A and B both being merged (it needs the working Luna/Jev call paths) and
on its own separate spend approval — it does not start until both are true. Counting
how many eligible profiles exist (definition-of-done item above) can happen any time
after B is merged, in parallel with C, since it only reads data and spends nothing.
