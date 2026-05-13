# PR-E — DB search: sparse-profile filter + tenure parity

**Status**: research + minimal code.
**Owners**: dashboard.py DB search tab (Tab 7), `helpers.py`.
**Branch (proposed)**: `feat/db-search-quality-filters`

This PR ships one thing: a **Hide sparse profiles** toggle in Tab 7 so the
recruiter can skip profiles too thin for personalised outreach. Tenure
parity with Filter+ is documented as a follow-up — the code path needs an
extra hop through `helpers.extract_display_fields` that's worth its own PR.

---

## What Shiri reported

> "Yesterday when I was working on the search from the DB, at this stage I
> wanted to filter the profiles that don't have detail about them — for
> whom I don't do personalisation — but it also wasn't efficient. I didn't
> get to keep working on it because the computer restarted and it was
> deleted."

> "Suddenly I'm catching that before in Filter+ there was the option to
> filter by tenure, or am I confused and it's only in the Crustdata
> search? […] For some reason I remember there were fields for this in the
> other filters too. Maybe even in DB?"

Two issues:

1. No way to drop "thin" profiles (sparse summary / employment history)
   from a DB-search result. She'd planned to filter them manually but lost
   her work to a computer restart.
2. Tenure filter exists in Filter (Tab 2) and Crustdata Search (Tab 0), but
   not in DB Search (Tab 7) — she remembers it being there.

## What the code does today

### DB search tab (Tab 7, `dashboard.py:10193`+)

Server-side filters supported (via `db.search_profiles_boolean` or
`db.search_profiles_fulltext`):

- Full-text query (boolean syntax with AND / OR / NOT / phrases).
- Name, Location, Current Title, Past Titles.
- Current Company, Past Companies.
- Skills, Schools.
- Freshness (Fresh <6mo / Stale >6mo).
- Has Email.
- Enriched After date.

After server fetch, client-side filters that further narrow:

- Freshness (re-applied) at `dashboard.py:10711-10721`.
- has_email re-applied at `dashboard.py:10723-10725`.

**No filter for**:

- Tenure at current company (`current_years_in_role`).
- Profile completeness / sparseness.
- Career break gaps (see PR-D).

The "sparse profile" question is well defined in the existing display:
`profile_to_display_row` (`helpers.py:166-249`) emits `summary` and
`past_positions` columns. Sparseness is roughly `len((summary + " " +
past_positions).split()) < N`.

### Tenure-by-DB technical hurdle

For the DB-search tenure filter to work, we need
`current_years_in_role` available on each DB-loaded profile. PR-B adds the
field to `normalize_crustdata_profile`, but that path is used only at
enrichment time (`flatten_for_csv` in Tab 3). DB profiles come through
`profiles_to_dataframe` → `helpers.profile_to_display_row` →
`extract_display_fields` — which does NOT compute tenure today.

So porting the Filter+ tenure widget to DB search requires:

1. Extracting the tenure helper out of `normalize_crustdata_profile` (PR-B)
   into a small reusable function in `normalizers.py`.
2. Calling it from `extract_display_fields` in `helpers.py`.
3. Propagating `current_start_date` + `current_years_in_role` through
   `profile_to_display_row`.
4. Adding the widget + filter logic to the DB tab.

That's three files touched, in addition to the widget. It's the right
shape — but big enough to belong in a separate PR after Codex has weighed
in on PR-B's approach.

## What this PR ships (code)

### Sparse-profile filter — `dashboard.py:10727+`

Added directly under the existing freshness / email post-filters in Tab 7:

```python
sparse_col1, sparse_col2 = st.columns([1, 2])
with sparse_col1:
    hide_sparse = st.checkbox(
        "Hide sparse profiles",
        value=False,
        key="db_hide_sparse",
        help=(
            "Drop profiles where the combined word count of "
            "`summary` + `past_positions` is below the threshold on the right. "
            "Useful when you only want profiles with enough detail to "
            "personalise outreach."
        ),
    )
with sparse_col2:
    sparse_threshold = st.slider(
        "Min words (summary + past_positions)",
        min_value=10, max_value=300, value=40, step=5,
        key="db_sparse_threshold",
        disabled=not hide_sparse,
    )

if hide_sparse:
    def _profile_word_count(row):
        parts = []
        for col in ('summary', 'past_positions'):
            val = row.get(col)
            if val is None:
                continue
            try:
                if pd.isna(val):
                    continue
            except (TypeError, ValueError):
                pass
            parts.append(str(val))
        return len(' '.join(parts).split())

    _pre_sparse_count = len(filtered_df)
    _word_counts = filtered_df.apply(_profile_word_count, axis=1)
    filtered_df = filtered_df[_word_counts >= sparse_threshold].reset_index(drop=True)
    _dropped_sparse = _pre_sparse_count - len(filtered_df)
    if _dropped_sparse > 0:
        st.caption(
            f"Hid **{_dropped_sparse}** sparse profiles "
            f"(< {sparse_threshold} words across summary + past_positions)."
        )
```

Design points:

- **Client-side filter**, applied after the server fetch. Toggling does not
  re-query Supabase — the search cache is reused on rerun and only the
  post-filter mask changes.
- **Default off**, so the existing flow is unchanged until the recruiter
  opts in.
- **Slider** is disabled when the checkbox is off, so the control is
  visible but inert. Lower bound 10 (clearly thin), upper bound 300 (most
  profiles have <300 words of summary+history).
- **Default threshold 40** — picked from a back-of-the-envelope of what a
  real summary + 3-5 past role descriptions looks like; revisit if the
  default is too aggressive.
- `pd.isna` raises on list / array values; the try/except catches that and
  falls through to `str(val)`. This matches the pattern used elsewhere in
  the same tab.

## What this PR does NOT ship

### Tenure filter in DB search (E1)
The walk-through above is the recipe. Mainly a refactor:

1. Lift the tenure-extraction snippet from
   `normalize_crustdata_profile` (PR-B, `normalizers.py:543-558`) into a
   public helper, e.g. `normalizers.extract_current_tenure(current_employers)
   -> (current_start_date, current_years_in_role)`.
2. Call it from `helpers.extract_display_fields`. Add the two fields to
   the returned dict (`helpers.py:43-75`).
3. Pass them through `helpers.profile_to_display_row`
   (`helpers.py:166-249`).
4. Render the same widget as PR-B in Tab 7; reuse the same NaN-tolerant
   filter logic.

Why not now: the refactor in step 1 affects more than DB search. PR-B
already ships the change for the enrichment path. Codex should look at
PR-B's normalizer change in isolation before we generalise it.

### "Personalization-ready" composite score (E2)
A boolean column / sort key that combines `has_email AND has_summary
AND num_past_employers >= 3`. Useful for outreach prioritisation. Not
shipped; depends on the recruiter wanting it (Shiri's framing was "hide
the thin ones", not "rank by personalisability").

### Server-side push-down (E3)
The sparse filter runs client-side after fetch. For very large result sets
(>5,000 profiles) that's noticeable. A server-side equivalent would need
either:

- A computed column on `profiles` (`word_count_summary + past_positions`)
  populated by a trigger, or
- A view that pre-computes it.

Both are migrations. Not shipped; relevant only if the client-side filter
proves too slow.

## Open questions for Shiri / Alexey

1. **Default threshold**: 40 words is a guess. After Shiri uses it for a
   real session, we should tune it. Should the default be higher (e.g.
   60) so the toggle feels more aggressive?

2. **What counts as "detail"**: only summary + past_positions, or should
   we also factor in current_title (length), skills count, education
   present, etc.? The current implementation deliberately uses only the
   two text fields she mentioned ("no detail to personalise from").

3. **Same widget in Filter+?**: she only mentioned DB search, but the
   same sparse filter would help Filter+ too. Worth a follow-up?

## Verification

### Smoke test (for Shiri)

1. Search the DB for current_title = "Software Engineer", location =
   "Israel". Note the count.
2. Toggle "Hide sparse profiles" on. Confirm the count drops and the
   caption shows N hidden.
3. Move the slider. Confirm the count updates without re-clicking
   Search.
4. Toggle it off. Confirm the count returns to the original.

### Code health

`python -m compileall -q dashboard.py` is clean. No new tests because the
filter is a pure client-side post-filter with no fixed semantics worth
pinning. If the threshold default changes, that's a UX decision, not a
regression.

## File refs

- `dashboard.py:10727+` — sparse-profile filter (this PR).
- `dashboard.py:10193`+ — DB search tab.
- `helpers.py:43-75` — `extract_display_fields` (where tenure fields
  would go in the E1 follow-up).
- `helpers.py:166-249` — `profile_to_display_row`.
- `normalizers.py:543-558` — tenure extraction (introduced in PR-B).
