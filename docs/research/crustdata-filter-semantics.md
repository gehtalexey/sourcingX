# Crustdata persondb/search: multi-predicate filter semantics

## Question

When Tab 0 Search in SourcingX combines a title filter with a current-company filter, the result count is far lower than the recruiter expects given the size of each filter's standalone candidate pool. The bug report (closed [PR #28](https://github.com/gehtalexey/sourcingX/pull/28)) showed `title=full stack developer + company OR(wiz, wix.com, forter, monday.com) + region=Israel` collapsing from 6,610 (title alone) and 5,446 (companies alone) down to 120, and `title=software developer + same companies + Israel` collapsing to a handful.

The leading hypothesis, captured in [Codex's diagnosis on PR #28](https://github.com/gehtalexey/sourcingX/pull/28#issuecomment-4421046994), is that `current_employers.title [.] X AND current_employers.name [.] Y` is evaluated **per array element** — i.e. both conditions must match the **same** `current_employers` entry, not any two entries on the profile. So if a person is "Software Developer at Forter" they match, but if they are "Marketing Director at Wix" with an advisory role labelled "Software Developer at Side Co", they do NOT match — even though both conditions are individually satisfied by the profile as a whole.

This document verifies that hypothesis against the live Crustdata `POST /screener/persondb/search` endpoint, checks whether a same-element override syntax exists, quantifies the result-count cost for Tab 0 Search, and recommends a path forward.

## Findings

**Crustdata `persondb/search` evaluates AND across nested-array columns (`current_employers.*`, `all_employers.*`, etc.) per-array-element by default. This is the *only* supported behavior — there is no documented syntax to switch it to cross-element AND, and the official docs explicitly call it out.**

The authoritative passage from Crustdata's own reference (skill: `C:/Users/admin/.claude/skills/crustdata-api/crustdata-full-reference.txt`, lines 3054 - 3065):

> :::warning Important: AND Operator with Nested Fields
> When using AND with nested array fields (like honors, employers, education), ALL conditions must match within the SAME array object.
>
> **Example:**
> - `current_employers.title = "Software Engineer" AND current_employers.name = "Capital One"`
>   - Matches if a SINGLE employment object has both title="Software Engineer" AND company name="Capital One"
>   - Does NOT match if "Software Engineer" is at one company and "Capital One" is a different employment entry

The docs also describe a separate "nested AND" wrap pattern (lines 3067 - 3220) — but that pattern is designed for the **opposite** problem (forcing two predicates onto *different* array elements, e.g. "worked at Hyperscan AND worked at Antler"). It does not, and cannot, change same-element AND across two different columns into cross-element AND. There is no escape hatch.

Live API probes (transcript below) reproduce the bug exactly, show that every documented re-wrapping leaves the count unchanged, and confirm that the only way to grow the result set is to relax one side of the AND — typically by broadening the title pattern or dropping it altogether.

## Evidence (probe transcript)

All probes use `POST /screener/persondb/search` via the Crustdata MCP, `limit=1` (3 credits each). Filters anchor on `region=Israel` for parity with PR #28's numbers.

### Probe 1 — baseline: title alone

```json
{"op": "and", "conditions": [
  {"column": "current_employers.title", "type": "[.]", "value": "software developer"},
  {"column": "region",                  "type": "[.]", "value": "Israel"}
]}
```

- `total_count`: **7,346**
- Interpretation: large pool with that title substring in Israel.

### Probe 2 — baseline: companies alone

```json
{"op": "and", "conditions": [
  {"op": "or", "conditions": [
    {"column": "current_employers.name", "type": "[.]", "value": "wiz"},
    {"column": "current_employers.name", "type": "[.]", "value": "monday"},
    {"column": "current_employers.name", "type": "[.]", "value": "wix"},
    {"column": "current_employers.name", "type": "[.]", "value": "forter"}
  ]},
  {"column": "region", "type": "[.]", "value": "Israel"}
]}
```

- `total_count`: **797**
- Interpretation: this is the absolute ceiling for any AND that uses these four companies in Israel.

### Probe 3 — current SourcingX shape (the bug)

This is the exact shape `build_filters` in `crustdata_search.py` emits when the recruiter types title=`software developer` and company=`wiz, monday, wix, forter`:

```json
{"op": "and", "conditions": [
  {"column": "current_employers.title", "type": "[.]", "value": "software developer"},
  {"op": "or", "conditions": [
    {"column": "current_employers.name", "type": "[.]", "value": "wiz"},
    {"column": "current_employers.name", "type": "[.]", "value": "monday"},
    {"column": "current_employers.name", "type": "[.]", "value": "wix"},
    {"column": "current_employers.name", "type": "[.]", "value": "forter"}
  ]},
  {"column": "region", "type": "[.]", "value": "Israel"}
]}
```

- `total_count`: **5**
- Interpretation: 5 / 797 = 0.6% of the company pool. The collapse is real and reproduces exactly.

### Probe 4 — wrap title in its own nested AND (docs' "multi-employer" trick, applied to one column)

```json
{"op": "and", "conditions": [
  {"op": "and", "conditions": [
    {"column": "current_employers.title", "type": "[.]", "value": "software developer"}
  ]},
  {"op": "or", "conditions": [ /* same 4 companies */ ]},
  {"column": "region", "type": "[.]", "value": "Israel"}
]}
```

- `total_count`: **5** — identical to Probe 3.
- Interpretation: wrapping a single-condition predicate in an extra `op=and` is a logical no-op. Confirmed.

### Probe 5 — wrap BOTH sides in separate nested-AND groups (does this force "different element" matching?)

```json
{"op": "and", "conditions": [
  {"op": "and", "conditions": [
    {"column": "current_employers.title", "type": "[.]", "value": "software developer"}
  ]},
  {"op": "and", "conditions": [
    {"op": "or", "conditions": [ /* same 4 companies */ ]}
  ]},
  {"column": "region", "type": "[.]", "value": "Israel"}
]}
```

- `total_count`: **5** — identical.
- Interpretation: the docs' "wrap each in its own AND" trick only changes behavior when both conditions filter on the **same column** (e.g. two `all_employers.name` predicates). It does not toggle same-element vs cross-element evaluation for predicates on different columns. There is no syntactic switch.

### Probe 6 — broaden title to `engineer` to estimate elasticity

```json
{"op": "and", "conditions": [
  {"column": "current_employers.title", "type": "[.]", "value": "engineer"},
  {"op": "or", "conditions": [ /* same 4 companies */ ]},
  {"column": "region", "type": "[.]", "value": "Israel"}
]}
```

- `total_count`: **213** (vs Probe 3's 5)
- Interpretation: just swapping "software developer" for "engineer" multiplies the result set 42x. Israeli companies overwhelmingly title these roles "Software Engineer", not "Software Developer". The title-string mismatch compounds the same-element constraint, but broadening to `engineer` still only reaches 213 / 797 = 26.7% of the company pool — the same-element AND is still costing two thirds of the candidates.

### Probe 7 — broaden title to the prefix `software` (matches `software engineer`, `software developer`, `software architect`, ...)

```json
{"column": "current_employers.title", "type": "[.]", "value": "software"}
```
combined with the same company OR and region=Israel.

- `total_count`: **173**
- Interpretation: even the most permissive software-flavored title prefix only reaches 173 / 797 = 21.7% of the company pool. The remaining ~78% of profiles at wiz/monday/wix/forter in Israel have NO current_employers element where the title contains "software" — which is consistent with same-element AND, because those people's title-bearing employer entry is "Engineer" or "Backend Engineer" or "Tech Lead" (no "software" substring) even though they all work at software companies.

### Probe 8 — switch to `all_employers.*` (does past-role data help?)

```json
{"op": "and", "conditions": [
  {"column": "all_employers.title", "type": "[.]", "value": "software developer"},
  {"op": "or", "conditions": [
    {"column": "all_employers.name", "type": "[.]", "value": "wiz"},
    {"column": "all_employers.name", "type": "[.]", "value": "monday"},
    {"column": "all_employers.name", "type": "[.]", "value": "wix"},
    {"column": "all_employers.name", "type": "[.]", "value": "forter"}
  ]},
  {"column": "region", "type": "[.]", "value": "Israel"}
]}
```

- `total_count`: **19** (vs Probe 3's 5)
- Interpretation: 3.8x improvement by including past roles. But the same-element constraint still applies (`all_employers` is also a nested array), so this only helps when a past employment entry at one of the named companies happened to carry the literal "software developer" title.

### Probe 9 — sanity check on a single profile

```json
{"op": "and", "conditions": [
  {"column": "current_employers.title", "type": "[.]", "value": "software developer"},
  {"column": "current_employers.name",  "type": "[.]", "value": "wix"},
  {"column": "region",                  "type": "[.]", "value": "Israel"}
]}
```

Returned exactly 1 profile, `Bar Boaron`. The full `current_employers` array has a single element:

```json
{"name": "Wix", "title": "Software Developer", "seniority_level": "Entry Level"}
```

A single element satisfying both conditions. Consistent with same-element AND.

## Implications for SourcingX Tab 0 Search

`crustdata_search.py:build_filters` (lines 240 - 278) emits exactly the Probe 3 shape: a top-level AND that combines a title OR-group with a current-company OR-group. Given Crustdata's same-element semantics, this means every profile returned must have **one single `current_employers` array element** whose `title` contains the recruiter's title substring AND whose `name` contains one of the recruiter's company substrings.

That excludes, by design, all of these:

- "Software Engineer at Wix" (title doesn't contain "developer" substring) — the dominant case
- "Tech Lead at Forter" who was previously a Software Developer at a non-named company — title bearer and company bearer are different entries
- "Software Developer at Side Co, also Advisor at Wiz" — both predicates satisfied by the profile, but by different array elements

The cost is severe. For the PR #28 repro (`software developer` at wiz/monday/wix/forter in Israel):

- Title alone in Israel: 7,346 candidates
- Companies alone in Israel: 797 candidates
- Same-element AND of both: 5 candidates (Probe 3)
- Same-element AND with broader `engineer` substring: 213 candidates (Probe 6)
- Cross-element AND, if it existed, would be bounded above by `min(7346, 797) = 797` and in practice should land in the high hundreds. The current behavior is delivering ~0.6% of that.

Even when the recruiter "fixes" their query by broadening the title (Codex's PR #28 evidence: 3-variant title OR → 120, 6-variant title OR → 883), the result remains a same-element-AND artifact — the broader title list just increases the chance that *one* employment record at one of the named companies happens to use that exact wording. This is fragile and depends entirely on how each company chose to title roles on LinkedIn.

## Recommended path forward

**Recommendation: option (b) plus a UI tweak. Restructure `build_filters` AND set recruiter expectations.**

Option-by-option assessment:

### (a) Accept the limit and educate users in the UI

A short helper text under the title and company inputs in Tab 0 explaining: *"Title + company must both match the same job entry on the profile. Use the broadest reasonable title pattern (e.g. `engineer` instead of `software developer`) to maximize coverage."*

This is free and prevents the recruiter from chasing missing candidates. Should ship regardless of (b).

### (b) Restructure `build_filters` to maximize coverage within Crustdata's constraints

There is no API syntax that unlocks cross-element AND, but there are several behavioural improvements that respect the constraint while delivering more candidates:

1. **Drop the title from same-array-element AND when a company set is present.** Instead of pushing title into `current_employers.title`, push it into `headline` (a top-level field, evaluated per-profile, not per-array-element). Pseudocode:

   ```json
   {"op": "and", "conditions": [
     {"op": "or", "conditions": [
       {"column": "headline", "type": "(.)", "value": "software developer"},
       {"column": "headline", "type": "(.)", "value": "software engineer"}
     ]},
     {"op": "or", "conditions": [/* company list on current_employers.name */]},
     {"column": "region", "type": "[.]", "value": "Israel"}
   ]}
   ```

   This decouples the title check (now profile-level) from the company check (still per-element on current_employers), trading some precision for a step-change in recall. `headline` typically holds the public-facing role, so it captures intent without requiring the role to literally appear in the LinkedIn-employer-entry title string. `skills` is another top-level field that is, in principle, a candidate for the same trick — but no probe was run against it during this investigation, so its lift is unverified. Treat it as a follow-up spike if `headline` alone underperforms in production.

2. **Add a "broaden title" toggle** that, when on, replaces the recruiter's title text with the inferred head-noun (`software developer` → `developer`, `frontend engineer` → `engineer`) for the `current_employers.title` predicate. Keep the narrow form for display/screening. This is a small, contained change to `build_filters` and `_effective_val`.

3. **Move the seniority and function_category filters into the same per-element group as title and company** so they continue to compose correctly. Today seniority is already on `current_employers.seniority_level`, which inherits the same-element semantics — that's the right behavior because seniority is *meant* to apply to the matched employment, not to "any element."

**Measured impact on the PR #28 repro** — three follow-up probes against the exact recruiter query (`software developer` + companies `[wiz, monday, wix, forter]` + `region [.] Israel`), substituting `headline` for `current_employers.title`:

| Probe | Filter substitution | `total_count` | Lift vs baseline |
|---|---|---:|---:|
| Baseline | `current_employers.title [.] software developer` | **5** | 1.0× |
| Reroute (exact term) | `headline [.] software developer` | **12** | 2.4× |
| Broaden head-noun | `headline [.] developer` | **37** | 7.4× |
| Multi-variant OR | `headline [.]` any of `software developer / software engineer / full stack / fullstack` | **149** | **29.8×** |

The reroute alone (`software developer` → `headline`) buys ~2.4×. The bigger lever is combining the reroute with the title-variant OR pattern the recruiter likely intends; that compounds to ~30× on this query. Below the recruiter's expectation of "hundreds" by ~25%, but the right order of magnitude for the use case.

Note: no probe in this investigation tested filtering on the `skills` field directly, so the lift from a `skills`-based reroute is unverified. The recommendation above is grounded in the three measured `headline` probes only.

### (c) Switch to a different endpoint (`/screener/person/search` realtime)

The realtime endpoint accepts `CURRENT_COMPANY` + `CURRENT_TITLE` as separate `filter_type`s and supports a `strict_title_and_company_match` post-processing flag. It is unclear from the docs whether non-strict mode evaluates them per-element or per-profile, but its `fuzzy_match` mode is documented as mutually exclusive with strict mode, implying non-strict mode is more lenient. Worth a single follow-up probe, but it's a 1-credit-per-profile endpoint with 15 RPM, so it's a poor fit for the Tab 0 "skim hundreds of results quickly" use case. Reserve for spot-checks, not the main search.

### (d) Other

Not pursuing. The combination of (a) + (b) lands the practical improvement without re-architecting around a different endpoint.

**Recommended landing order:**

1. Land (a) UI helper text in a small docs/UX PR — gives recruiters an immediate, accurate mental model.
2. Land (b1) `headline` rerouting for title — measurable recall jump on the same query Alexey reported. (`skills` is a follow-up spike if `headline` alone underperforms; not validated here.)
3. Defer (b2) "broaden title" toggle and (c) realtime probe until (b1) is in production and the recruiter team has 1-2 weeks of feedback.

## References

- Crustdata API skill: `C:/Users/admin/.claude/skills/crustdata-api/SKILL.md`
- Crustdata full reference (authoritative same-element semantics): `C:/Users/admin/.claude/skills/crustdata-api/crustdata-full-reference.txt` (lines 3054 - 3220)
- Closed PR #28: https://github.com/gehtalexey/sourcingX/pull/28
- Codex's diagnosis comment: https://github.com/gehtalexey/sourcingX/pull/28#issuecomment-4421046994
- Current SourcingX filter builder: `crustdata_search.py:build_filters` lines 240 - 278
