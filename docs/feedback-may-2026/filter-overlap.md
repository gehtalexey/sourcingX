# PR-B — Filter ↔ Filter+ overlap, tenure-filter parity, OR/AND scope confusion

**Status**: research + minimal code.
**Owners**: dashboard.py Filter tab (Tab 2), Filter+ tab (Tab 4), `normalizers.py`.
**Branch (proposed)**: `refactor/filter-plus-audit-tenure-parity`

This PR ships three things:

1. A small change to `normalize_crustdata_profile` so enriched profiles carry
   `current_start_date` + `current_years_in_role`. Used by 2 and by PR-E.
2. A new **tenure filter** in Filter+ (min/max years at current company), the
   parity fix Shiri asked for.
3. A safer default for the **Required keywords scope** radio (defaults to
   "Full profile" instead of "Skills only").

Everything else (which redundant filters to remove, whether to add an "auto-skip
duplicated filters" mode) is documented but **not** shipped — left as numbered
options for Codex / Alexey.

---

## What Shiri reported

> "In the Filter+ tab, the same filters from the Filter tab are checked
> again, and after Filter had already removed a lot, Filter+ filters more
> profiles — in the end very few remain. Regarding keywords, I understand
> it's after enrichment, but it's weird that it removes so many profiles for
> irrelevant companies, after a large quantity was already removed before."
>
> "I was surprised that suddenly it removed a lot for keywords. I set it with
> OR because one of them is enough. […] It's not logical to me. What does
> 'any' mean? Is there a state where it actually considers AND and not OR?"
>
> "I'm suddenly catching that before in Filter+ there was the option to
> filter by tenure, or am I confused and it's only in the Crustdata search?"

Three issues:

1. Filter+ duplicates several filters that Filter already runs.
2. Required keywords with OR-logic behaved like AND from her POV.
3. Tenure filter is in Filter and Crustdata Search but not in Filter+ (she
   remembers it being there).

## What the code does today

### Filter (Tab 2, `dashboard.py:6157`+, `apply_pre_filters` at `3487`)

Filters supported:

- Past candidates (LinkedIn URL / name match).
- Blacklist companies (current company only).
- Not-relevant companies (current company only).
- Exclude title keywords (preset categories + free-text).
- Include title keywords (free-text).
- Duration filters: **min/max months in role** and **min/max months at
  company** (`dashboard.py:6389-6421`). Active only when the CSV has
  PhantomBuster duration columns (`durationInRole`, `durationInCompany`,
  `current_years_in_role`, etc.).

### Filter+ (Tab 4, `dashboard.py:7888`+)

Filters supported:

- Past candidates (Google Sheet, by full name).
- Blacklist companies (current company only).
- Not-relevant companies (**all employers**, not just current).
- Target companies — Off / Prioritize / Require (all employers).
- Target universities — Off / Prioritize / Require.
- Client wanted companies — Off / Prioritize / Require + scope toggle.
- Job title include / exclude keywords.
- **Required keywords** with AND/OR logic + scope toggle (Skills only / Full
  profile). Logic (`dashboard.py:8528-8531`):

  ```python
  if skills_logic == "AND":
      return all(kw in combined_text for kw in skills_list)
  else:  # OR
      return any(kw in combined_text for kw in skills_list)
  ```

  Help text for the logic radio: `"AND = must have ALL keywords, OR = must
  have at least ONE"`. Help text for scope: `"Skills = skills column only |
  Full profile = everything including job descriptions"`.

- **No tenure / duration filter** prior to this PR.

### Overlap

| Filter (Tab 2) | Filter+ (Tab 4) | Status |
|---|---|---|
| Past candidates | Past candidates | Identical (Sheet vs file source) |
| Blacklist (current) | Blacklist (current) | Identical (redundant) |
| Not-relevant (current) | Not-relevant (**all employers**) | Filter+ is a strict superset |
| Exclude title keywords | Exclude title keywords | Identical (redundant) |
| Include title keywords | Include title keywords | Identical (redundant) |
| Duration in role / company | **— (gap)** | Missing in Filter+ until this PR |

## Findings / hypotheses

### Why Filter+ removes more on "irrelevant companies" even after Filter ran it

The two filters look the same but operate on different scopes:

- **Filter (Tab 2)** matches "Not relevant" only against `current_company`.
- **Filter+ (Tab 4)** matches against ALL employers — current + past
  (`dashboard.py:8176-8210` reads `df['all_employers']`).

So a profile that passed Filter (current company is fine) can fail Filter+
because it had a past job at a not-relevant company. From the recruiter's POV,
that looks like "the same filter ran twice and removed even more." From the
code's POV, they're two distinct gates with different breadth — and Filter+ is
deliberately stricter.

### OR-of-skills returning zero (likely her "any but acting like AND")

When `search_scope = "Skills only"`, the keyword match is restricted to the
`skills` column. After Crustdata enrichment that column is often:

- Empty (Crustdata didn't return a skills array)
- Or a thin comma-separated list like `"Python, Docker"` — way narrower than
  what the candidate actually has in role descriptions.

So `any(kw in skills for kw in [react, vue, angular])` returns False for most
profiles even though they clearly use those frameworks in their role
descriptions. The recruiter reads this as "OR is acting like AND".

The fix is **not** to change the logic — it's correct — but to default the
scope to "Full profile" and warn explicitly that "Skills only" is brittle.

### Tenure filter missing in Filter+

`enriched_df` is built by `flatten_for_csv(successful)` which calls
`normalize_crustdata_profile`. The normalizer never emitted a tenure field, so
even if we wanted to add a Filter+ widget, there was nothing to filter on.

This PR fixes that by adding `current_start_date` (raw string) and
`current_years_in_role` (float) to the normalizer's output. Both are derived
from `current_employers[*]` via the existing `pick_current_employer` +
`_parse_start_date_sort_key` helpers — same parse path the screening prompt
uses, so values are consistent end-to-end.

## What this PR ships (code)

### 1. Normalizer carries tenure fields — `normalizers.py:543-558`

```python
emp = pick_current_employer(raw.get('current_employers'))
current_start_date = None
current_years_in_role = None
if emp:
    current_title = emp.get('employee_title') or emp.get('title')
    current_company = emp.get('employer_name') or emp.get('company_name')
    raw_start = emp.get('start_date')
    if raw_start is not None and str(raw_start).strip():
        current_start_date = str(raw_start).strip()
        parseable, dt = _parse_start_date_sort_key(raw_start)
        if parseable:
            current_years_in_role = round(
                (datetime.now() - dt).days / 365.25, 1
            )
```

Returned alongside the existing keys (`normalizers.py:594-611`):

```python
'current_start_date': current_start_date,
'current_years_in_role': current_years_in_role,
```

Tests pin the contract: `test_normalize_current_tenure.py`. CI runs them.

### 2. Filter+ tenure widget — `dashboard.py:8115-8147`

Below the "Required keywords" row, two `st.number_input` controls:

- Min years at current company
- Max years at current company (0 = no max)

Both are `disabled=not has_tenure_col` so older session-state DataFrames (from
sessions before this PR) render the widget greyed out with a caption
explaining why. No silent breakage.

### 3. Filter+ tenure filter logic — `dashboard.py:8541-8559`

Runs after the keyword filter, before the result summary:

```python
if 'current_years_in_role' in df.columns:
    if min_years_at_company > 0:
        tenure_series = pd.to_numeric(df['current_years_in_role'], errors='coerce')
        below_min_mask = tenure_series < min_years_at_company  # NaN is False
        ...
```

Rows where `current_years_in_role` is NaN/None are **kept**, not dropped — a
profile with an unparseable `start_date` shouldn't be punished for a data
quality issue.

### 4. Scope radio defaults to "Full profile" — `dashboard.py:8101-8113`

Added `index=1` to the radio so "Full profile" is selected by default.
Expanded help text:

> Full profile = skills + job titles + employer names + summary + headline +
> raw Crustdata JSON. Skills only = the `skills` column ONLY — this column
> is often shallow after enrichment, so OR-of-shallow-skills can still drop
> most profiles. Prefer 'Full profile' unless you deliberately want a strict
> skills-tag match.

## What this PR does NOT ship (left for review)

### Option B1 — Remove the redundant filters from Filter+
Drop blacklist (current), exclude/include titles, past-candidates from
Filter+ since Filter already covers them. Filter+ becomes additive-only
(target companies, target universities, client wanted, required keywords,
tenure).

- Pros: no more "same filter ran twice"; less recruiter confusion.
- Cons: breaks the workflow where someone uses Filter+ *without* having run
  Filter first (e.g. profiles loaded from DB straight into Filter+).

### Option B2 — Keep them, document why
Add help text on each redundant filter explaining "this also runs in the
Filter tab, but here we check ALL past employers, not just current." Don't
change behaviour.

- Pros: zero risk of breaking the DB-only workflow.
- Cons: doesn't fix the perception problem.

### Option B3 — Auto-skip if Filter already ran
Track in `session_state` which filters Filter applied this session, then have
Filter+ skip exact duplicates and re-run only the broader ones.

- Pros: best UX, no behaviour loss.
- Cons: more state to manage; "filter already ran" is brittle when the
  recruiter loads new data mid-flow.

### Option B4 — Surface the breadth difference inline
Show a tooltip on each filter result: "Filter removed 12 on current-company
blacklist; Filter+ removed 7 more on past-employer blacklist." This requires
the funnel-breakdown logic to know which filters are scope-extensions of
others.

## Open questions for Shiri

1. **For "Required keywords with OR" — was scope set to "Skills only" or
   "Full profile"?** If "Skills only", the new default in this PR should
   solve the perception. If "Full profile", we need to dig further.

2. **The "any" wording she mentions** — was that the **help text**
   ("must have at least ONE") or the **caption** in the post-filter summary
   that prints something like `"Missing Keywords in skills (any)"` at
   `dashboard.py:8535`? If the latter, the word "any" reads strangely there
   too — we could rename to "match-any" / "match-all".

3. **The "Filter+ removed many on irrelevant companies"** — when she clicks
   into the funnel breakdown, does the count appear under
   `"Not Relevant Companies"` (Filter+ broader scope) or under some other
   bucket? If "Not Relevant", the answer is option B2 (document the broader
   scope). If something else, we have another bug.

## Options matrix (for Codex)

| Topic | Shipped now | Optional next steps |
|---|---|---|
| Tenure filter parity | Filter+ widget + normalizer fields | DB-search parity (PR-E) |
| OR/AND confusion | Default = "Full profile" + clearer help | Reword "any" → "match-any" in funnel breakdown caption |
| Redundant filters | Nothing | Choose between B1 / B2 / B3 / B4 |

## Verification

### Unit tests (CI)

`test_normalize_current_tenure.py` — five cases pin:
- Tenure fields pulled from most recent current employer (not raw[0]).
- Unparseable `start_date` ("Present") leaves `current_years_in_role` as
  `None` (no 2025-year garbage).
- Empty `current_employers` returns None on both fields.
- Year-only ("2024") `start_date` parses to ~Jan 1 of that year.
- Missing `start_date` returns None on both fields.

Added to `.github/workflows/test.yml` so CI runs them on every PR.

### Manual smoke (for Shiri)

1. Enrich a small batch (10 profiles).
2. In Filter+, set min years at current company = 2.
3. Confirm that profiles with `<2y` current-company tenure are dropped, and
   the funnel breakdown lists e.g. `"Below min tenure (2y): N removed"`.
4. Set Required keywords = `react, vue, angular` with OR logic.
5. With the new default "Full profile" scope, confirm OR-matches return a
   reasonable count, not zero.

## File refs

- `normalizers.py:543-611` — derived tenure fields.
- `dashboard.py:8093-8164` — Filter+ tenure widget + scope-default change.
- `dashboard.py:8541-8559` — Filter+ tenure filter logic.
- `test_normalize_current_tenure.py` — regression tests.
- `.github/workflows/test.yml` — CI registration.
