# PR-C — AI screening audit: skills, tenure verdict drift, "Lead" downgrade

**Status**: research-only. No code change in this PR. Each candidate fix below
is a tiny diff stub that we want Codex / Alexey to weigh in on before
applying.
**Owners**: `screening_policy.py`, `dashboard.py` (`screen_profile`,
`compute_role_durations`, `trim_raw_profile`), `prompts.py`.
**Branch (proposed)**: `chore/screening-policy-audit`

This is the highest-leverage PR of the five. Three of Shiri's symptoms point
at one architectural seam: the unified screening policy that landed in commit
`ee77326` (March 2026, #42) and the user-stated hard constraints clause that
landed in commit `d536416` (Feb 2026, #16). The three symptoms are:

- **A**: "every candidate is 0 go" when she adds a skills requirement.
- **B**: tenure / military verdicts drift to wrong NO GOs when she edits the
  recruiter request — even though she didn't change the verdict-injecting code.
- **C**: profiles with "Lead" / "Team Lead" titles get downgraded even when
  her include-titles filter explicitly says "lead" is OK.

---

## What Shiri reported (verbatim summary)

> "I added [skills required] to the AI prompt. Ran on the first 10, got zero
> matches. When I went into the explanation and then the profile itself, I
> realised it didn't catch the issue of skills or the detail under the
> experience itself — because it rejected candidates that are worth reaching
> out to."
>
> "I changed it, but it didn't manage to understand and again I got
> everyone as NO GO on the skills issue."
>
> "In the end I removed the skills part and left only the tenure +
> military calculation. In the previous search that was unsuccessful
> because of skills, it computed it perfectly. Now when I left only this
> part, it completely messed up its calculation (even though I didn't
> change the text) and again gave me 0 go when I go into profiles and see
> that they're fine."
>
> "I got 4 GOs for a change, but the rest it didn't compute correctly,
> like the examples above + it brought in parameters it wasn't supposed to
> reference, like one profile that became 'Lead' and was dropped because
> of that (by the way in titles in filters I put 'lead' to include)."
>
> Example bad reasoning quoted on
> `https://www.linkedin.com/in/oren-samuel/`.

## Prompt assembly trace

Order delivered to the model (OpenAI or Anthropic):

1. **System prompt** = `screening_policy.SCREENING_POLICY`, ~2,300 tokens.
   Returned by `get_system_prompt()` with `{today}` substituted.
2. **User prompt** = `build_user_prompt(user_request, durations_text,
   trimmed_raw)` at `screening_policy.py:130-149`:

   ```python
   durations_header = f"{durations_text}\n\n" if durations_text else ""
   return f"""{durations_header}## Recruiter Request
   {user_request}

   ## Candidate Profile (raw JSON)
   ```json
   {json.dumps(trimmed_raw, indent=2, default=str)}
   ```

   Evaluate this candidate against the recruiter request using the policy. Return ONLY the JSON object."""
   ```

So the user-side message is, in order:
- Pre-computed `ROLE DURATIONS`, `STABILITY SUMMARY`, `STABILITY VERDICT`,
  `EXPERIENCE SUMMARY`, `EXPERIENCE LIMIT CHECK` (from
  `dashboard.py:compute_role_durations` lines 3884-4085).
- `## Recruiter Request\n{user_request}`
- `## Candidate Profile (raw JSON)\n```json\n{trimmed profile}\n``` `

`compute_role_durations` emits the verdict lines with literal `>>> ... <<<`
delimiters, e.g. `dashboard.py:4000-4005`:

```python
if len(short_companies) >= 3:
    lines.append(f'>>> STABILITY VERDICT: FAIL — {len(short_companies)} short-stint companies >= 3 → MAX SCORE 4 <<<')
elif current_company_months is not None and current_company_months < 6:
    lines.append(f'>>> STABILITY VERDICT: FAIL — current role {_fmt_duration(current_company_months)} < 6 months → MAX SCORE 5 <<<')
else:
    lines.append('>>> STABILITY VERDICT: PASS <<<')
```

The system policy at `screening_policy.py:43-52` tells the model to use those
verdicts as authoritative. But — and this is important — there is **no
explicit separator** between the verdict block and `## Recruiter Request`
beyond two newlines, and **no clause** that says "the Recruiter Request below
does not override the verdicts above."

## System prompt audit (relevant clauses)

### Hard-constraint clause (`screening_policy.py:23-38`)

```
## User-Stated Hard Constraints (HIGHEST PRIORITY)
Treat any condition the recruiter writes in the request as a HARD
CONSTRAINT — a binary filter, not a preference. This includes:
- Tenure minimums ...
- Tenure maximums ...
- Current employer / industry filters ...
- Categorical exclusions ...
- Geographic / language constraints ...
- Any explicit numeric threshold the recruiter states ...

Rules:
1. If the candidate clearly violates any user-stated hard constraint → return NO GO with score 1-2.
2. Name the violated constraint in the reasoning ...
3. Do NOT soft-score around hard constraints. ...
4. Do NOT give the benefit of the doubt on user-stated constraints — be strict and literal. ...
5. User-stated constraints OVERRIDE the generic Hard Filters below when they are stricter ...
6. Tenure at the current company is always measured at the COMPANY level ...
```

This clause is **highest priority** — the policy explicitly says recruiter
constraints OVERRIDE the generic filters. Combined with rule 4 ("if the
profile lacks evidence that the constraint is satisfied, treat it as a fail"),
this is the engine behind Symptom A.

### Skills clause (`screening_policy.py:74-76`)

```
## Skills Match
Normalize first. Exact synonyms = direct match. Adjacent tech = partial
match only with profile evidence of transferability. Buzzwords ≠ depth.
Don't reject for non-1:1 stack if fundamentals are strong and transfer is
credible. Do reject if the gap is too large for near-term fit.
```

Vague about **where** to look for skills evidence. The trimmed profile
(`dashboard.py:trim_raw_profile`, lines 4088-4128) carries:

- `skills` — Crustdata's array of skill tags.
- `summary` — top-level summary.
- `past_employers[*].employee_description`, `current_employers[*].employee_description`
  — role descriptions.
- `headline`.

The policy mentions "the profile" generically but never says "skills can also
appear inside role descriptions." Combine that with rule 4 above (strict
literal interpretation) and the result is exactly Symptom A.

### IC vs Leadership Filter (`screening_policy.py:54-55`)

```
For IC searches (Senior SWE, Backend, Full Stack, hands-on Tech Lead):
return NO GO when the profile is primarily leadership-oriented.
Leadership-heavy titles (CTO, Founder, VP, Director, Team Leader,
Head of Eng, R&D Manager) are a negative signal for IC searches —
but do NOT exclude on title alone. Exclude only when title AND
description show leadership scope without recent hands-on execution.
For leadership searches these titles are relevant.
```

Two issues:

1. "Team Leader" is hardcoded. There is no way for the model to know that
   the recruiter explicitly included "lead" in her filter — the filter's
   include-titles list is not passed into the prompt.
2. "IC search" is inferred by the model, not pinned by the recruiter or
   the platform. So the rule fires whenever the model decides the search is
   IC-ish, regardless of recruiter intent.

This is Symptom C. CLAUDE.md already documents this pitfall verbatim:

> "Only reject titles the JD EXPLICITLY lists. 'Team Lead' is NOT
> overqualified unless the JD says so. Read the JD's reject list literally."

So the policy contradicts CLAUDE.md.

## Root-cause hypotheses

### Symptom A — "0 go on skills"

When Shiri writes "must have X skill" in the recruiter request:

1. The hard-constraint clause (`screening_policy.py:23-38`) catches the
   phrase and binds the model to literal enforcement.
2. The skills clause (`74-76`) doesn't tell the model **where** to find
   evidence — only that it should "normalize first."
3. Crustdata's `skills` array is the easiest signal: it's short, structured,
   and explicit. The model defaults to it.
4. Rule 4 says "if the profile lacks evidence that the constraint is
   satisfied, treat it as a fail" — even when the skill is mentioned
   in role descriptions.

Net: skills-required + shallow skills array → universal NO GO.

### Symptom B — tenure / military verdict drifts when recruiter request changes

The verdict block (`>>> STABILITY VERDICT: ... <<<` and `>>> EXPERIENCE LIMIT
CHECK: ... <<<`) is emitted by `compute_role_durations`. It does not change
when Shiri edits her recruiter request — that's pure Python.

But the **AI's interpretation** of the verdict can shift because:

1. The verdict block and `## Recruiter Request` are separated by only two
   newlines.
2. There's no clause that says "the request below does not modify the
   verdicts above."
3. If the recruiter request now contains tenure-adjacent wording (e.g.
   "must have at least 1 year"), the deterministic post-screening tenure
   validator (`dashboard.py:4321-4329`, calling
   `tenure_constraint_validator.enforce_tenure_constraint`) parses it and
   may **further override** the model's decision to NO GO.

So removing the skills sentence but leaving tenure phrasing in place can:
- Let the validator's parser pick up a stricter rule than before.
- Or change which sentence is "load-bearing" in the AI's reasoning — the
  model may now anchor on a different phrase and produce a different score.

In short: the verdict block isn't fortified against neighbouring text, and
the deterministic validator runs unconditionally on whatever's in
`user_request + job_description`.

### Symptom C — "Lead" downgrade

The IC vs Leadership clause fires whenever the model classifies the search
as IC. "Team Leader" is hardcoded. The recruiter's filter include-list is
not visible to the model. CLAUDE.md and `screening_policy.py:54-55` are in
direct conflict.

## Recent commits that correlate with the regression

| Commit | When | What changed | Why it matters |
|---|---|---|---|
| `d536416` (#16) | Feb 2026 | Added "User-Stated Hard Constraints (HIGHEST PRIORITY)" clause to the policy + deterministic `tenure_constraint_validator`. | Makes the model strict-literal on recruiter constraints — the substrate for Symptom A and Symptom B. |
| `ee77326` (#42) | Mar 2026 | Consolidated all screening into one policy path. Removed the per-role prompt dropdown and `role_prompt` argument. Dropped tenure threshold from 2y → 1y. | Per-role nuance from `prompts.py` (e.g., `TEAMLEAD_ISRAEL` knew how to read "Lead" titles in a leadership search) is no longer in the prompt the model sees. The IC vs Leadership rule now applies globally. |

Neither commit was wrong — but together they removed the "this search is for
a Team Lead" / "this search is IC" signal from the prompt while
simultaneously raising the strictness of literal interpretation. That's how
the policy ended up over-rejecting on skills and titles.

## Candidate fixes (do NOT ship without review)

### For Symptom A — skills

**A1: Update the policy text to mention where to look for skills evidence.**

```diff
 ## Skills Match
-Normalize first. Exact synonyms = direct match. Adjacent tech = partial match only with profile evidence of transferability. Buzzwords ≠ depth. Don't reject for non-1:1 stack if fundamentals are strong and transfer is credible. Do reject if the gap is too large for near-term fit.
+Normalize first. Exact synonyms = direct match. Adjacent tech = partial match only with profile evidence of transferability. Buzzwords ≠ depth. Don't reject for non-1:1 stack if fundamentals are strong and transfer is credible. Do reject if the gap is too large for near-term fit.
+
+### Where to find skills evidence
+When the recruiter requires a specific skill, check ALL of:
+1. The `skills` array (primary signal, but often sparse).
+2. `employee_description` on past_employers and current_employers (where
+   real evidence usually lives).
+3. `summary` and `headline`.
+Do NOT mark a skill as missing solely because the `skills` array doesn't
+contain it. Treat the skill as present when role descriptions credibly
+demonstrate hands-on use.
```

Cheapest fix, no Python work. Risk: GPT-4o-mini might still anchor on the
`skills` array.

**A2: Pre-compute a SKILLS COVERAGE verdict in Python.** Add a function
similar to `compute_role_durations` that scans `skills + employee_description
+ summary` for each user-requested skill, returns a deterministic
`SKILLS COVERAGE: <found / missing>` block. Inject it into `durations_text`
so the model sees it as authoritative.

```python
def compute_skills_coverage(raw, required_skills: list[str]) -> str:
    """Search skills array + all role descriptions + summary for each required
    skill; return a deterministic verdict block the AI must respect."""
    ...
```

Highest reliability. Cost: needs us to parse the recruiter request for
"must have X" / "X required" phrases, or pass `required_skills` from the UI.

**A3: Make skills informational, never reject.** Edit the hard-constraint
clause to exclude "missing skill" from the binary-filter list. The model
treats skills as a soft signal only; tenure / stability / explicit reject
titles remain hard.

Riskiest semantically — recruiters genuinely use the "must have X" phrasing
for hard skills.

### For Symptom B — tenure verdict bleed

**B1: Insert an explicit barrier in `build_user_prompt`** between the
durations block and the recruiter request:

```diff
-durations_header = f"{durations_text}\n\n" if durations_text else ""
+if durations_text:
+    sep = "\n\n" + ("=" * 60) + "\nEND PRE-COMPUTED BLOCKS — Recruiter Request Below\n" + ("=" * 60) + "\n\n"
+    durations_header = durations_text + sep
+else:
+    durations_header = ""
```

One-line change. Doesn't fix the deterministic validator — but stops the
model from reading the recruiter request as if it amends the verdicts.

**B2: Add an explicit policy clause** after the Pre-Computed Blocks section
(`screening_policy.py:43-52`):

```diff
 - EXPERIENCE LIMIT CHECK — when present, restates the rule above and is binding...
+
+CRITICAL: The Pre-Computed Blocks above are IMMUTABLE system facts. The
+Recruiter Request section that follows them may add NEW constraints (e.g.
+"must have skill X", "must be in Tel Aviv") but DOES NOT modify or override
+any pre-computed verdict. Apply both:
+- Pre-computed STABILITY VERDICT and EXPERIENCE LIMIT CHECK = score caps.
+- Recruiter constraints = additional binary filters.
+If they conflict, the pre-computed verdicts win.
```

**B3: Move the recruiter request ABOVE the durations** in `build_user_prompt`.
The model reads the durations last so they're freshest in working memory.

```diff
-return f"""{durations_header}## Recruiter Request
-{user_request}
+return f"""## Recruiter Request
+{user_request}
+
+{durations_text}

 ## Candidate Profile (raw JSON)
 ...
```

Trade-off: changes reading order; needs a re-evaluation of the
hard-constraint clause's "HIGHEST PRIORITY" framing.

**B4: Tighten the tenure validator's input.** Today
`tenure_constraint_validator.enforce_tenure_constraint` parses
`user_request + job_description` joined together. If the recruiter writes
something like "minimum 1 year" in either field, the validator overrides
the model. This is intentional (introduced by `d536416` Codex review). But
if she edits unrelated parts of the request, the parsed threshold might
shift unintentionally. Worth a regression test that pins the parser's
behaviour on the exact phrases Shiri uses.

### For Symptom C — "Lead" downgrade

**C1: Pass the include-titles list into the prompt.** When the recruiter
runs screening from inside Filter+ (which has an "Include keywords" field
at `dashboard.py:7999`), forward the list as an explicit "Acceptable
titles" note inside the recruiter request:

```python
def screen_profile(..., include_titles: list[str] = None):
    ...
    if include_titles:
        accept_note = f"\n\nAcceptable titles (do NOT downgrade for these): {', '.join(include_titles)}"
        effective_user_request = (user_request or '') + accept_note
```

Then in the IC vs Leadership clause:

```diff
-For IC searches (Senior SWE, Backend, Full Stack, hands-on Tech Lead): return NO GO when the profile is primarily leadership-oriented. Leadership-heavy titles (CTO, Founder, VP, Director, Team Leader, Head of Eng, R&D Manager) are a negative signal for IC searches — but do NOT exclude on title alone. Exclude only when title AND description show leadership scope without recent hands-on execution. For leadership searches these titles are relevant.
+For IC searches (Senior SWE, Backend, Full Stack, hands-on Tech Lead): return NO GO when the profile is primarily leadership-oriented, UNLESS the recruiter request includes the title in its "Acceptable titles" list. Leadership-heavy titles (CTO, Founder, VP, Director, Team Leader, Head of Eng, R&D Manager) are a negative signal for IC searches — but do NOT exclude on title alone, and never downgrade a candidate whose current title is in the "Acceptable titles" list. Exclude only when title AND description show leadership scope without recent hands-on execution AND the recruiter did NOT list the title as acceptable. For leadership searches these titles are relevant.
```

**C2: Drop "Team Leader" from the hardcoded list.** Smallest possible
change. Risk: actual IC searches stop rejecting Team Leader candidates who
have stopped coding.

**C3: Make CLAUDE.md the source of truth and update the policy to match.**
The pitfall is already documented: "Only reject titles the JD EXPLICITLY
lists." We could literally bake that rule into the policy and remove the
heuristic list entirely.

## Recommendation (for the PR description, not for shipping)

If we ship one cluster of fixes:
- **A2** (deterministic skills coverage) — most reliable, but most work.
- **B1** + **B2** (separator + explicit "verdicts win" clause) — cheap and
  fixes the bleed.
- **C1** (pass include-titles + amend the policy) — clean fix; aligns
  policy with CLAUDE.md.

If we want minimum effort: **A1** + **B2** + **C3**. All three are
policy-text-only.

## Open questions for Codex / Alexey

1. **Skills**: do we want the deterministic verdict path (A2) or the
   prompt-only path (A1)? A2 is more reliable; A1 is shippable in 10
   minutes.
2. **Verdict bleed**: ship the separator (B1) AND the explicit policy
   clause (B2)? Or just the separator?
3. **Lead title**: do we plumb the include-titles list into the prompt
   (C1), or do we trust the policy text alone (C3)?
4. **Tenure validator**: should we add a `test_tenure_validator_regression`
   that pins the parser's behaviour on the exact wordings Shiri uses?

## Open questions for Shiri

1. **Can you screenshot the exact recruiter-request text** in two
   different runs — one where tenure worked, one where it broke? We need
   the prompt to reproduce.
2. **The Oren Samuel example** — can you share the AI's full reasoning
   text? We want to see what verdict it cited.

## Verification

Each candidate fix has its own verification path. We should not ship
without:

1. A regression test that pins the change in `screening_policy.py` or the
   prompt assembler.
2. A manual rerun on the same 10 profiles Shiri used. The fix is good if
   the GO count moves from 0 → reasonable (not necessarily 10).

If we ship A2, that's a `test_skills_coverage_verdict.py` plus an end-to-end
mock screening test.

If we ship B1/B2, that's an extension to
`test_screening_policy_consolidation.py` to assert the separator and the
new clause are present.

If we ship C1/C3, that's an addition to
`test_hard_constraints_prompt.py` or a new test that mocks an IC search
with an include-titles list containing "team lead".

## File refs

- `screening_policy.py:23-38` — User-Stated Hard Constraints clause.
- `screening_policy.py:43-52` — Pre-Computed Blocks clause.
- `screening_policy.py:54-55` — IC vs Leadership Filter.
- `screening_policy.py:74-76` — Skills Match clause.
- `screening_policy.py:130-149` — `build_user_prompt`.
- `dashboard.py:3884-4085` — `compute_role_durations` (verdict emission).
- `dashboard.py:4088-4128` — `trim_raw_profile` (what the AI sees).
- `dashboard.py:4167-4329` — `screen_profile` (prompt assembly + validator).
- `tenure_constraint_validator.py` — deterministic post-screening override.
- CLAUDE.md "Common Pitfalls → Overqualification rules".
- Commits `d536416` (#16) and `ee77326` (#42).
