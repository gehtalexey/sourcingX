# PR-D — Career-break / maternity-leave detection

**Status**: research-only. No code change in this PR.
**Owners**: `dashboard.py:compute_role_durations`, `screening_policy.py`.
**Branch (proposed)**: `feat/career-break-detection-proposal`

---

## What Shiri reported

> "Something else I noticed: it's not [reacting to] the fact that this profile
> hasn't been working for two years because of maternity leave. Maybe it's
> worth adding 'career break' to the exclude list?"

She wants the platform to handle career breaks (maternity leave, sabbatical,
caregiving, etc.) differently from "stale candidate" or "current role <6
months" patterns that the stability verdict currently penalises.

## What the code does today

A grep across the repo for `career break|maternity|leave|gap|sabbatical`
returns **zero** screening / prompt hits. There is no career-break detection
anywhere.

The closest existing logic is in `dashboard.py:compute_role_durations`
(`3884-4085`), which computes stability at the **company level**:

- A `STABILITY VERDICT: FAIL` fires when:
  - 3+ short-stint companies (<12 months each, excluding internships and
    early-career roles), OR
  - Current company tenure < 6 months.

A maternity-leave candidate looks like:
- A gap (no employer entry between two real roles), or
- A current employer with a recent `start_date` (post-return), giving < 6
  months tenure even though her prior career was long and stable.

The second pattern collides with the stability verdict — the candidate
gets capped at MAX SCORE 5 even when she's a strong fit.

## Findings / hypotheses

### Gap detection is feasible without new data

`compute_role_durations` already iterates over `past_employers` +
`current_employers` and parses `start_date` / `end_date`. We can compute the
sorted timeline of `(start, end)` intervals and identify any gap > N months
as a candidate "career break".

### Two distinct UX intents

1. **Inform the AI** about the gap so it doesn't penalise the candidate's
   short current tenure (which is just "back from leave"). This needs a new
   pre-computed signal injected into `durations_text`, and a policy clause
   that tells the model how to interpret it.
2. **Filter on career breaks** at the recruiter's discretion. Could be in
   Filter+ or DB search ("include career breaks" / "exclude career breaks").

Shiri specifically asked for the second case ("add career break to the
exclude list"), but the AI-informing case is the more impactful one — if the
AI knows about the gap, the stability verdict can be tempered.

### Gap definition

What counts as a "career break"?

- Minimum gap length: 6 months? 12 months? The 6-month threshold lines up
  with the existing `current role < 6 months` stability rule.
- Ending recency: only flag gaps that ended in the last N months (since an
  old gap doesn't change "current tenure" perception).
- Inclusivity: should partial overlaps count (e.g., the candidate took a
  freelance role during her leave)? The conservative answer is no.

Without external signals (Crustdata does not expose leave reasons), we can
only detect "no recorded employer for N months." We **cannot** distinguish
maternity leave from sabbatical from unemployment from "did contract work
they didn't list."

## Options

### Option D1 — Inject a CAREER BREAK DETECTED line into `durations_text`

Add a helper in `compute_role_durations` that emits a deterministic line:

```
CAREER BREAK DETECTED: 24 months between Jun 2022 and May 2024
  → The candidate's current role started after this gap.
  → Do NOT apply the "current role < 6 months → MAX SCORE 5" verdict
    in this case.
```

Then in `screening_policy.py` Pre-Computed Blocks section, add:

```
- CAREER BREAK DETECTED — when present, suspend the
  "current role < 6 months → MAX SCORE 5" stability cap. The candidate's
  short current tenure is explained by the gap, not by job-hopping.
```

This is the lightest viable touch. Maintains the deterministic-verdict
pattern that `STABILITY VERDICT` and `EXPERIENCE LIMIT CHECK` already use.

### Option D2 — Add `career_break_months` to EXPERIENCE SUMMARY

Same logic, but instead of a separate verdict, extend the existing
EXPERIENCE SUMMARY block:

```
EXPERIENCE SUMMARY (pre-calculated — DO NOT recalculate):
  TOTAL CAREER SPAN: 12y 3m
  CAREER BREAK: 24m ending May 2024 (excluded from stability count)
  INDUSTRY EXPERIENCE: 10y 0m
```

Pros: keeps the verdict surface area small.
Cons: harder for the AI to anchor on — the deterministic verdicts work because
they have `>>> ... <<<` delimiters; embedding it inline weakens that.

### Option D3 — Filter+ / DB toggle ("Exclude career-break candidates")

Add a binary filter to Filter+ and DB search that drops profiles with any
detected gap > N months. Per Shiri's literal ask.

Pros: gives the recruiter explicit control.
Cons: most teams want the opposite — keep career-break candidates so they
don't get rejected on stability. We should probably ship D1 first and only
add a filter if recruiters actually want to exclude them.

### Option D4 — Inverse filter ("Show career-break candidates")

Same widget as D3 but flipped: surface profiles WITH career breaks (for
returnship programs or specific outreach campaigns).

### Option D5 — Both — surface in screening AND filter

D1 + D3 + D4. Most complete but more code.

## Trade-offs to consider

- **False positives**: the gap is just "missing data" half the time (a
  freelance role the candidate didn't list, an unpaid leave the platform
  doesn't know about). We need a high enough threshold (12+ months?) to
  avoid noise.
- **False negatives**: a candidate who took 6 months of leave but updated
  her LinkedIn to show it ("Maternity Leave" as an employer) will not be
  detected by our pure date-gap logic.
- **No reason information**: we cannot infer "maternity" vs "sabbatical"
  from dates alone. The recruiter sees "career break detected" — they're
  responsible for the rest.

## Recommendation (for the PR description, not for shipping)

**D1** as the first ship, optionally followed by **D3+D4** if recruiters
actively want the filter. Rationale: the highest-value change is preventing
the AI from rejecting strong returners on the 6-month stability rule.
Filtering on top of that is additive UX.

## Open questions for Shiri / Alexey

1. **Direction of the bias**: do you want career-break candidates included
   (with a context note to the AI so they don't get auto-rejected) or
   excluded (auto-rejected as not interested)? Shiri's wording says "add
   to the exclude list" but the intent seems to be "don't drop them just
   because their current tenure is short." Which is it?

2. **Threshold**: what counts as a career break? 6 months? 12 months?
   Anything longer than a typical job change?

3. **Recency**: only flag breaks that ended recently (e.g. in the last 24
   months)? Older breaks are usually irrelevant.

4. **Treat as "fresh-hire reset"?**: if a candidate returned from a 24-month
   break and has been at her current company for 4 months, should we
   pretend her tenure is 4 months (current rule applies) or 4 months but
   "explained" (rule suspended)?

## Verification

When this lands:

1. Build a synthetic raw_data fixture with a 24-month gap.
2. Run `compute_role_durations` and assert the verdict appears.
3. Mock a screening call and assert the AI doesn't NO GO on the candidate
   purely because of the short current tenure.
4. Manual smoke: Shiri runs screening on a profile she knows has a
   maternity gap. Confirm the AI's reasoning explicitly mentions the gap
   and does not cite the 6-month rule.

## File refs

- `dashboard.py:3884-4085` — `compute_role_durations` (where to add the
  gap detection helper).
- `dashboard.py:4000-4005` — STABILITY VERDICT emission (where to add the
  conditional suspension).
- `screening_policy.py:43-52` — Pre-Computed Blocks clause (where to
  document the new verdict for the AI).
- `tenure_constraint_validator.py` — may need a small change so the
  deterministic tenure validator also respects a detected career break.
