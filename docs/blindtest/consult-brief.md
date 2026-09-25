# Consult brief: how should SourcingX decide "unproven must-haves" automatically?

## 1. Goal
SourcingX (Streamlit recruiting app, repo C:\Users\gehta\projects\sourcingX) screens LinkedIn
profiles against a recruiter's brief (role, must-haves, nice-to-haves, exclusions) with an LLM
(gpt-5.6-luna) and returns GO / NO GO. The owner (a sourcing lead, not a developer) wants it to
run fully automatically: every candidate ends as GO or NO GO, no manual-review bucket. It must
stop rejecting strong people just because LinkedIn doesn't spell out a must-have, while still
rejecting people who genuinely don't fit. The result must work across very different roles:
a hard one (Applied AI / agents engineer, Dwelly) and simple ones (Senior Full Stack at
Autofleet, Backend at ScaleOps, both Israel).

## 2. Where we are
- Old rule: "if the profile lacks evidence a must-have is satisfied, treat it as a fail".
  On 20 candidates the sister pipeline (agent-kalamata) had approved for Dwelly: 2 GO / 18 NO GO,
  every rejection said "does not verify / establish", none cited a contradiction.
- Yesterday (PR #144, merged) we made each must-have three-state: met / not_met /
  needs_verification. Any not_met, matched exclusion or hard filter (incl. a Python-computed
  stability verdict) => NO GO; otherwise any needs_verification => a "NEEDS VERIFICATION"
  decision stored as fit "Maybe" with notes. Result on the same 20: 2 GO / 15 NV / 3 NO GO,
  all 3 NO GOs cite evidence.
- Blind test (40 anonymised Dwelly+Owner profiles, labelled by two AI experts): of 3 both call
  outreach, SourcingX rejected 0; of 10 both call reject, SourcingX approved 0. But 24 of 40
  ended in the NV bucket.
- The owner now says: no NV bucket, decide automatically, like kalamata does.
- How kalamata decides (read its files): binary GO/NO_GO; per-position rulebooks calibrated
  with the owner on named examples; e.g. for Dwelly "an engineering seat at a top AI-native
  product company counts as production AI even if the profile says nothing about AI"; general
  rule "a thin profile is NOT evidence against the candidate... if titles + companies + career
  trajectory clearly fit the role -> GO with a note 'thin profile - verify stack in call'; if
  titles + companies are ambiguous AND skills can't be verified -> NO GO"; GO only at score >= 7;
  "marginal / limited evidence" on the key requirement -> 6 = NO GO unless at a client-wanted
  company. SourcingX has NO per-position rulebook: only the brief the recruiter types.

## 3. My planned design (unverified — attack it)
Per must-have four states: met (shown) / implied (not stated, but the career clearly implies
it: titles, company type e.g. AI-native, what they built) / not_met (contradicted) /
unproven (neither shown nor implied). Decision: any not_met, matched exclusion or hard filter
-> NO GO. Any unproven -> NO GO, reason names the must-have. All met or implied and score >= 7
-> GO, with a note "verify in call: <implied items>". Store GO as "Good Fit" (+ note in
screening_notes), NO GO as "Not a Fit". Remove the NV decision, its UI group and download.

## 4. Assumptions I have NOT checked
- That the model will actually use "implied" generously enough; it may fall back to
  "unproven" and we're back to 2/18.
- That "implied" won't become a loophole that passes weak people (the model is lenient on
  generic signals like a big-company name).
- That one generic instruction can work across roles without per-position calibration.
- That 4 states is better than, say, keeping 3 states and letting a second rule decide NV
  (e.g. NV counts as met when score >= X, or NV on at most one must-have).
- That score >= 7 is the right GO line for SourcingX's scoring.

## 5. The question
What decision rule should turn "not shown on the profile" into an automatic GO or NO GO, so it
(a) stops the over-rejection, (b) doesn't pass weak candidates, and (c) works on both a hard
AI-agents role and simple full-stack / backend roles, with only the recruiter's typed brief?
Is my 4-state design right, or is there a better rule? Be concrete: the exact states, the exact
decision rule, and the key sentences of prompt wording.

## 6. Real code (this is where I looked, not a boundary)
Project root: C:\Users\gehta\projects\sourcingX
- screening_policy.py — the policy text and the structured JSON schema the model answers in
  (search "Must-Have Verdict: three states" and "hard_filter_failed").
- dashboard.py — `_must_have_verdict_state`, `_verdicts_force_no_go`,
  `_verdicts_needs_verification`, `_resolve_three_state_decision`, `_decision_to_fit_label`,
  `screen_profile` (search these names; the file is ~11k lines, read only those functions).
Sister project (read-only reference): C:\Users\gehta\projects\agent-kalamata
- .claude/skills/screening/SKILL.md — general rulebook (see "Step 5: Must-Have Verification",
  "Thin profiles").
- .claude/skills/screening-dwelly-ai-applied-eng-eu/SKILL.md — a calibrated position rulebook.
Please read only these, not the whole repos.

## 7. Constraints
- The owner maintains this by reading plain English; the rule must be explainable in 3 lines.
- Shared Supabase table `screening_results`: other projects filter on `screening_fit_level`
  ("Good Fit" / "Maybe" / "Not a Fit"); don't invent new values there. `screening_notes` is free.
- Cost: ~2 model calls per candidate on gpt-5.6-luna (~$0.004); don't add more calls without
  a strong reason.
- We will measure the result on a blind test (Dwelly, Owner, Autofleet, ScaleOps) labelled by
  you two experts from brief + anonymised profile, plus the 20 kalamata-approved Dwelly people.

If the question itself is wrong, or we are solving the wrong problem, say that first. And end
with the one check that would settle this.
