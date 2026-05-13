# PR-A — Enrich tab: partial-batch behaviour

**Status**: research-only. No code change in this PR.
**Owners**: dashboard.py Enrich tab (Tab 3) + Email enrich (Tab 6).
**Branch (proposed)**: `chore/enrich-ux-investigation`

---

## What Shiri reported

> "I'm working on files I uploaded in Load. In the Enrich step I write the
> number of profiles to enrich, but it doesn't do all of them. It does them in
> parts, and each time I have to refresh and re-write the remaining count. I
> just want to confirm this is OK. The same happens at the 'enrich emails'
> step — it doesn't do all the quantity, only part, and you have to run it
> several times. Also, after each run a notification comes up about some
> profiles."

So three sub-complaints:

1. **Crustdata enrich** (Tab 3) requested-count > delivered-count per click.
2. **Email enrich** (Tab 6) same shape.
3. **After-run notification** about "some profiles" — unclear what it means.

## What the code does today

### Crustdata loop (Tab 3)

`dashboard.py:7551-7586`:

```python
max_profiles = st.number_input(
    "Number of profiles to enrich",
    min_value=1,
    max_value=len(urls_for_enrichment),
    value=min(10, len(urls_for_enrichment)),
    ...
)
batch_size = 25
...
if st.button("Start Enrichment", type="primary", ...):
    urls_to_process = urls_for_enrichment[:max_profiles]
    ...
    for i in range(0, len(urls_to_process), batch_size):
        batch = urls_to_process[i:i + batch_size]
        batch_results = enrich_batch(batch, api_key, tracker=tracker)
        results.extend(batch_results)
        ...
        if i + batch_size < len(urls_to_process):
            time.sleep(2)
```

So the loop **does process all `max_profiles` URLs**. There is no early
break. Per click, the user gets exactly `max_profiles` API attempts (split into
25-URL batches).

### Where URLs disappear

Two places silently reduce the visible delta:

**1. `enrich_batch()` URL matching cascade (`dashboard.py:3247-3276`)**:

```python
# PRIMARY: Use query_linkedin_profile_urn_or_slug (Crustdata echoes our input)
query_slugs = item.get('query_linkedin_profile_urn_or_slug', [])
...
# FALLBACK: Try matching via linkedin_flagship_url
...
if not matched:
    result_url = item.get('linkedin_flagship_url') or item.get('linkedin_url', '')
    unmatched.append(extract_slug(result_url) or 'NO_SLUG')
```

Unmatched results are tracked separately and later saved as "failed
enrichment" markers so they don't re-appear in the next click
(`dashboard.py:7822-7835`):

```python
save_failed_enrichment(db_client, norm,
    error_message='URL matching failed - profile may exist with different URL',
    original_url=norm)
```

**2. Crustdata API errors per profile (`dashboard.py:7603-7610`)**:

```python
errors = [r for r in results if 'error' in r]
successful = [r for r in results if 'error' not in r]
...
if errors:
    st.warning(f"Errors: {[e.get('error', 'unknown')[:100] for e in errors[:3]]}")
```

Errors are saved to DB as failed enrichments (`dashboard.py:7787-7796`) so the
next click sees fewer "URLs to enrich".

### After-run notification

Two messages are emitted:

- `st.warning("Errors: ...")` — partial-fail toast (line 7610).
- `st.info(f"Tracked {unmatched_saved} unmatched URLs (won't show as 'to enrich' again)")` (line 7837).
- Final: `st.session_state['enrichment_message'] = "warning:Enriched X profiles in Y. Z failed: ..."` (line 7869) — rendered after `st.rerun()`.

This is almost certainly the "notification about some profiles" Shiri mentions.

### Email enrich (Tab 6)

`dashboard.py:6933-6970` + `enrich_profiles_with_salesql()` at `dashboard.py:1526-1637`.
Same pattern: the user sets `enrich_count`, the function processes all of them in
a `ThreadPoolExecutor` (max 10 concurrent), and finishes with
`st.success(f"Done! {new_emails} profiles now have emails.")`.

**Key**: SalesQL returns "no email found" as a non-error response. Those
profiles get marked as "lookup attempted" but contribute zero to `new_emails`.
The user sees `Done! 12 profiles now have emails.` when they ran on 50 →
perceived "only 12 worked".

## Findings / hypotheses

1. **Most likely root cause**: each click does honour the requested count, but
   the visible **delta** is smaller because:
   - Crustdata returns some entries as errors / unmatched.
   - Those errors are persisted to DB so the next click's `urls_for_enrichment`
     list is shorter.
   - SalesQL "no email found" responses are 100% non-errors but produce no
     visible enrichment.

   From the recruiter's POV: "I asked for 50, only 47 changed" feels like the
   tool stopped early. From the code's POV: 50 were attempted, 3 produced no
   usable data and were quietly logged.

2. **Confounding factor**: the `time.sleep(2)` between batches plus the
   per-profile Crustdata round-trip can take ~30s for 50 profiles. If
   Streamlit's connection times out (free tier limit), the run silently
   truncates. We have no observability for this.

3. **UI ambiguity**: the post-run toast says "Enriched X profiles" — the recruiter
   reads it as "of the N I asked for". Wording could say
   `"Attempted N, enriched X, Y unmatched, Z errors — total credits used: 3·N"`.

## Options

### Option A1 — Diagnostic logging only (cheapest)
Add an admin-only post-run table:

| Phase | Count |
|---|---|
| Requested (max_profiles) | 50 |
| Sent to Crustdata | 50 |
| API errors | 1 |
| Unmatched in response | 2 |
| Matched + saved | 47 |
| Already-enriched-or-failed (subtracted from next run) | 50 |

Implementation: just expose what `_enrich_debug` and `_enrich_match_debug`
already carry; show it as a closeable expander for non-admin users too.

### Option A2 — Rewrite the toast
Replace the current `"Enriched N profiles in T"` message with a breakdown the
recruiter can act on:

> Enriched 47 / 50 requested (3 credits each = $1.50). 1 Crustdata error, 2
> not found on LinkedIn. 'Remaining to enrich' has dropped by 50, not 47 —
> the 3 unmatched were logged so we don't retry them.

No diagnostic data needed; just rewrite the message.

### Option A3 — Retry-in-place for unmatched
When Crustdata's response has unmatched / errored profiles, retry them once
inside the same click (different batch grouping). Costs: slightly more
credits, slightly more wait. Benefit: the recruiter doesn't have to re-click.

Risk: if a URL is truly bad it will fail twice, doubling the cost on
unrecoverable inputs.

### Option A4 — Streaming progress with per-URL status (heaviest)
Replace the single progress bar with a live table of "Slug | Status |
Error". Lets Shiri see what's happening profile-by-profile. Implementation is
non-trivial (Streamlit rerun semantics make live updates awkward).

## Recommendation (for the PR description, not for shipping)

If we ship code in a follow-up: **A1 + A2** is the strongest pair.

- A1 ships observability and is reversible.
- A2 fixes the perception gap without changing behaviour.
- A3 is appealing but should wait until A1/A2 confirm the actual failure modes.

## Open questions for Shiri

1. **When you click Enrich a second / third time, does the "remaining to
   enrich" count eventually reach zero, or does it stay stuck above zero
   regardless of how many times you click?** This separates "partial Crustdata
   responses" (count keeps dropping → reaches zero) from "URL-matching
   degradation" (count plateaus → unmatched URLs keep coming back).

2. **In Tab 6 (email enrich), is the perception that "only some get enriched"
   driven by SalesQL not finding an email, or by genuine partial-run
   behaviour?** Useful diagnostic: compare `new_emails` (toast) vs the count
   she actually requested.

3. **The post-run toast text she sees — is it the green success one, the
   yellow warning one, or both? Can you screenshot one example?** We have
   three different messages in the code path (success, warning, info) and
   knowing which one is the "notification about some profiles" pins down
   whether it's an error case or the unmatched-URL info.

## Verification

When the follow-up code PR lands:

1. Repro: load a CSV with 50 fresh URLs, click Enrich with N=50, screenshot
   the post-run breakdown.
2. Run a second click — confirm "remaining" went from 50 → 0 (not 50 → 47).
3. Repeat for Email enrich with 50 profiles, half of which we know have no
   public email.

## File refs

- `dashboard.py:3204-3329` — `enrich_batch()`
- `dashboard.py:6933-6970` — Email enrich UI
- `dashboard.py:7551-7878` — Crustdata enrich UI + loop + toast
- `dashboard.py:1526-1637` — `enrich_profiles_with_salesql()`
