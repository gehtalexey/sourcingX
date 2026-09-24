"""bakeoff_kalamata.py

Screening bake-off: SourcingX (Luna + Jev) vs agent-kalamata
==============================================================

Background: docs/GOAL-screening-bakeoff-vs-kalamata.md. Builds the sample of
candidates kalamata has already screened, runs the same candidates through
Jev (by script) and through SourcingX's real dashboard (Luna, by hand in the
browser -- not this script), then puts all three verdicts side by side.

Three subcommands:
    sample  -- pick the candidate sample from kalamata's pipeline_candidates
               + profiles, write sample.json + one upload CSV per position.
    jev     -- screen the sample through Jev. Without --yes this is a dry
               run (zero Jev calls, just a call-count + rough size estimate).
               With --yes it makes real (paid) calls, resumable.
    report  -- pull Luna's saved verdicts from screening_results, join with
               the Jev CSV and the kalamata sample, write bakeoff.csv +
               bakeoff.json (summary, disagreements, a 30-row review sample).

NO PAID CALLS FROM ANYWHERE IN THIS FILE EXCEPT `jev --yes`. `sample` and
`report` only ever do read-only Supabase selects (profiles, pipeline_candidates,
screening_results). `jev` without --yes builds no Jev client and never calls
jev_client.screen_with_jev() -- see cmd_jev() below.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from db import SupabaseClient, compute_jd_hash  # noqa: E402
from normalizers import normalize_linkedin_url  # noqa: E402
import jev_client  # noqa: E402


# ============================================================================
# Pass/fail rules -- one constant each, with the file:line they were read
# from, per the goal doc's requirement to cite the source of truth rather
# than invent a threshold.
# ============================================================================

# Luna pass rule -- dashboard.py:4527-4551 (_decision_to_fit_label). A GO
# decision (must-haves met, no exclusion matched) buckets into "Good Fit"
# (score >= GO_CONFIDENCE_THRESHOLD, 7) or "Maybe" (score < 7); only a NO GO
# decision becomes "Not a Fit". screening_results.screening_fit_level only
# ever stores this fit label (the raw GO/NO GO decision isn't saved), so
# "Luna passed" here means the underlying decision was GO, i.e. fit_level in
# {"Good Fit", "Maybe"}. (dashboard.py:10493's comment calling "Maybe" a
# "borderline NO GO" is stale next to the authoritative code at 4544-4551;
# the code wins, not that comment.)
LUNA_PASS_FIT_LEVELS = frozenset({"Good Fit", "Maybe"})
LUNA_FAIL_FIT_LEVELS = frozenset({"Not a Fit"})

# Jev pass rule -- jev_client.propose_verdict(), jev_client.py:484-571.
# screening_result == "qualified" when the interpolated fit score clears
# QUALIFY_SCORE_FLOOR (jev_client.py:559-564); "not_qualified" on a failed
# hard filter or a low fit score; None whenever Jev fails open (low
# confidence / malformed answer -- jev_client.py:474-481, 508-549). None is
# a missing verdict here ("screening incomplete"), never a "no".
JEV_PASS_RESULTS = frozenset({"qualified"})
JEV_FAIL_RESULTS = frozenset({"not_qualified"})

UNENRICHABLE_PREFIX = "[Unenrichable]"
PRESCREEN_PREFIX = "[Prescreen]"

DEFAULT_PROFILE_CHUNK = 150

# fetch_profiles_by_urls' OR filter repeats the quoted URL list 3x
# (linkedin_url.in, original_url.in, original_urls.ov) -- same batch size as
# dashboard.py:596-601's "exact pass" for the same reason (PostgREST GET
# query strings are rejected past ~20k chars).
PROFILE_OR_CHUNK = 60


# ============================================================================
# Pure helpers -- no I/O, fully unit-testable.
# ============================================================================

def brief_to_job_description(brief: dict) -> str:
    """Build the freeform job_description string EXACTLY the way
    dashboard.py's four-box screening form does (dashboard.py:9873-9878),
    from a screening_brief dict with role_context/must_haves/nice_to_haves/
    exclusions. Used so Jev sees the same JD text Luna's UI would build from
    the same brief."""
    role_context = (brief.get("role_context") or "").strip()
    must_haves = brief.get("must_haves") or []
    nice_to_haves = brief.get("nice_to_haves") or []
    exclusions = brief.get("exclusions") or []
    parts = [
        f"Role: {role_context}" if role_context else "",
        ("Must-haves:\n" + "\n".join(f"- {m}" for m in must_haves)) if must_haves else "",
        ("Nice-to-haves:\n" + "\n".join(f"- {n}" for n in nice_to_haves)) if nice_to_haves else "",
        ("Exclusions:\n" + "\n".join(f"- {e}" for e in exclusions)) if exclusions else "",
    ]
    return "\n".join(p for p in parts if p)


def _clean(value):
    """Coerce NaN / empty-string / None to None, mirroring dashboard.py's
    clean_value() closely enough for the thin-profile check below (a
    DataFrame-derived NaN, or an empty string, both count as "missing")."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, (list, dict)) and not value:
        return None
    return value


def _as_raw_dict(raw_data) -> dict:
    if isinstance(raw_data, str):
        try:
            raw_data = json.loads(raw_data)
        except (ValueError, TypeError):
            return {}
    return raw_data if isinstance(raw_data, dict) else {}


def is_eligible_profile(profile: dict) -> bool:
    """A candidate is eligible for the bake-off only when profiles.raw_data
    is present AND has a non-empty top-level `skills` or `summary` --
    mirrors the thin-profile check dashboard.py's enrich_thin_profiles_for_batch
    uses to decide whether AI Screen would pay to top up a profile
    (dashboard.py:5147-5177: `if not skills and not summary: ... enrich`).
    We must never hand the dashboard a profile missing both, or screening
    it through the real app would silently trigger a paid Crustdata call."""
    if not isinstance(profile, dict):
        return False
    raw = _as_raw_dict(profile.get("raw_data"))
    if not raw:
        return False
    skills = _clean(raw.get("skills"))
    summary = _clean(raw.get("summary"))
    return bool(skills) or bool(summary)


def is_excluded_kalamata_row(screening_notes: Optional[str], screening_score) -> bool:
    """True when a pipeline_candidates row must be dropped outright, before
    any grouping: a null screening_score, or notes starting with
    "[Unenrichable]"."""
    if screening_score is None:
        return True
    if (screening_notes or "").startswith(UNENRICHABLE_PREFIX):
        return True
    return False


def classify_kalamata_group(screening_result: Optional[str], screening_notes: Optional[str]) -> Optional[str]:
    """'yes' / 'no_fullscreen' / 'no_prescreen', or None if the row belongs
    to none of the three groups (e.g. screening_result is neither 'qualified'
    nor 'not_qualified')."""
    notes = screening_notes or ""
    if screening_result == "qualified":
        return "yes"
    if screening_result == "not_qualified":
        return "no_prescreen" if notes.startswith(PRESCREEN_PREFIX) else "no_fullscreen"
    return None


def kalamata_reason(row: dict) -> str:
    """First 200 chars of screening_notes, or failing that, a summary-ish
    field pulled out of screening_detail (jsonb)."""
    notes = (row.get("screening_notes") or "").strip()
    if notes:
        return notes[:200]
    detail = row.get("screening_detail")
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except (ValueError, TypeError):
            detail = {}
    if isinstance(detail, dict):
        for key in ("summary", "reason", "reasoning", "notes"):
            val = detail.get(key)
            if val:
                return str(val)[:200]
    return ""


def quote_url_list(urls) -> str:
    """Comma-joined, double-quoted values for a PostgREST `in.(...)` (or
    `ov.{...}`) filter -- matches dashboard.py:601
    (`','.join(f'"{u}"' for u in batch)`). Without quoting, a URL is sent to
    PostgREST as a bare, unescaped token; that's how dashboard.py's own fix at
    that line came to exist."""
    return ','.join(f'"{u}"' for u in urls)


def preferred_match_url(row: dict) -> Optional[str]:
    """The URL to match profiles / Luna verdicts against for a sample row --
    prefer `profile_url` (the profiles.linkedin_url the row was actually
    matched to, and what got written into the upload CSV Luna screened), and
    fall back to the pipeline `linkedin_url` only when profile_url is missing
    (e.g. an older sample.json written before profile_url existed)."""
    raw = row.get("profile_url") or row.get("linkedin_url")
    return normalize_linkedin_url(raw) or raw


def deterministic_sample(candidates, k: int, seed: int) -> list:
    """Sort candidates, then random.Random(seed).sample(...) -- so the same
    seed + input always picks the same k items, regardless of call order
    upstream."""
    ordered = sorted(set(candidates))
    if k >= len(ordered):
        return ordered
    return random.Random(seed).sample(ordered, k)


def luna_verdict(fit_level: Optional[str]) -> Optional[str]:
    if fit_level in LUNA_PASS_FIT_LEVELS:
        return "yes"
    if fit_level in LUNA_FAIL_FIT_LEVELS:
        return "no"
    return None


def jev_verdict_from_result(screening_result: Optional[str]) -> Optional[str]:
    if screening_result in JEV_PASS_RESULTS:
        return "yes"
    if screening_result in JEV_FAIL_RESULTS:
        return "no"
    return None


def _score_value(value):
    """Keep a real score of 0 as 0 -- only None or "" (missing) become "".
    A plain `value or ""` treats 0 as falsy and would silently blank out a
    genuine zero score."""
    if value is None or value == "":
        return ""
    return value


def compute_agreement(rows: list, key_a: str, key_b: str) -> dict:
    """Agreement rate for one pair of verdict columns, counting only rows
    where BOTH verdicts are present ('yes'/'no'; 'missing' rows are excluded
    from the denominator, not treated as a disagreement)."""
    matches = 0
    total = 0
    for r in rows:
        a, b = r.get(key_a), r.get(key_b)
        if a in ("yes", "no") and b in ("yes", "no"):
            total += 1
            if a == b:
                matches += 1
    rate = round(matches / total, 4) if total else None
    return {"matches": matches, "total": total, "rate": rate}


def pick_review_sample(rows: list, seed: int, limit: int = 30) -> list:
    """30 rows (all of them if fewer), sorted by linkedin_url first so the
    random.Random(seed).sample() pick is reproducible, then re-sorted by
    linkedin_url for a stable display order."""
    ordered = sorted(rows, key=lambda r: r["linkedin_url"])
    if len(ordered) <= limit:
        return ordered
    picked = random.Random(seed).sample(ordered, limit)
    return sorted(picked, key=lambda r: r["linkedin_url"])


def estimate_jev_prompt_chars(profile: dict, job_description: str, brief: Optional[dict]) -> int:
    """Rough (chars, NOT tokens-from-the-SDK) size estimate for one Jev
    call: jev_client.build_questions() needs typesafe_sdk (not installed in
    this repo, see jev_client.py's module docstring), so we can't build the
    real Noul/Score/Choice objects to measure them without the SDK. This
    counts the same text that would go into the call instead: the candidate
    text once (it's the `state=` argument, sent once per call) plus the JD
    block three times (it's repeated in all three questions' instructions,
    per jev_client.build_questions()) plus the brief's must-haves/exclusions
    text (only embedded once, in the hard-filter question)."""
    candidate_text = jev_client.build_candidate_text(profile)
    brief = brief or {}
    jd_block = job_description or brief.get("role_context") or ""
    must_haves = brief.get("must_haves") or []
    exclusions = brief.get("exclusions") or []
    extra = "\n".join(str(m) for m in must_haves) + "\n".join(str(e) for e in exclusions)
    return len(candidate_text) + 3 * len(jd_block) + len(extra)


# ============================================================================
# I/O helpers
# ============================================================================

def _load_supabase_client() -> SupabaseClient:
    """Same config.json loading pattern as compare_screening_modes.py.
    Never logs or returns the key itself."""
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        print("ERROR: config.json not found -- can't reach Supabase.")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    url = (config.get("supabase_url") or "").strip()
    key = (config.get("supabase_key") or "").strip()
    if not url or not key:
        print("ERROR: supabase_url / supabase_key not set in config.json.")
        sys.exit(1)
    return SupabaseClient(url, key)


def _fetch_pipeline_candidates(client: SupabaseClient, position_id: str) -> list:
    rows = client.select(
        "pipeline_candidates",
        "position_id,linkedin_url,screening_result,screening_score,"
        "screening_notes,screening_detail,screened_at",
        filters={"position_id": f"eq.{position_id}"},
        limit=200000,
        order_by="linkedin_url.asc",
    )
    return dedupe_candidates_by_url(rows)


def dedupe_candidates_by_url(rows: list) -> list:
    """Offset-paginated selects with no stable sort can hand back the same
    linkedin_url more than once (or skip one) across pages. Keep exactly one
    row per linkedin_url -- the one with the latest screened_at (missing/None
    screened_at loses to any row that has one)."""
    best = {}
    for row in rows:
        url = row.get("linkedin_url")
        if not url:
            continue
        prev = best.get(url)
        if prev is None or (row.get("screened_at") or "") >= (prev.get("screened_at") or ""):
            best[url] = row
    return list(best.values())


def dedupe_sample_rows_by_profile_url(rows: list) -> list:
    """Collapse sample rows that resolved to the same profile_url. kalamata's
    pipeline_candidates can hold more than one LinkedIn URL alias for the
    same person (a rescrape under a slightly different slug, say), and those
    aliases can land in different kalamata groups after matching -- one
    person must appear once, in one group, in the sample. Deterministic
    rule: keep the row with the latest screened_at; on a tie, keep the row
    whose (pipeline) linkedin_url sorts lowest, so the result never depends
    on input order."""
    best = {}
    for row in rows:
        url = row.get("profile_url")
        if not url:
            continue
        prev = best.get(url)
        if prev is None:
            best[url] = row
            continue
        row_screened = row.get("screened_at") or ""
        prev_screened = prev.get("screened_at") or ""
        if row_screened > prev_screened:
            best[url] = row
        elif row_screened == prev_screened and (row.get("linkedin_url") or "") < (prev.get("linkedin_url") or ""):
            best[url] = row
    return list(best.values())


def fetch_profiles_by_urls(client: SupabaseClient, urls, chunk_size: int = PROFILE_OR_CHUNK) -> dict:
    """Look up profiles by linkedin_url OR original_url OR original_urls --
    pipeline URLs from kalamata are often aliases of the profile's stored
    canonical URL, not the canonical URL itself (dashboard.py:596-613,
    _fetch_candidate_profiles_for_urls). Mirrors that OR filter shape here,
    and indexes the result under every normalized alias (linkedin_url,
    original_url, each original_urls[] entry) so a lookup by any alias finds
    the same profile row."""
    normalized = sorted({normalize_linkedin_url(u) or u for u in urls if u})
    by_url = {}
    columns = "linkedin_url,raw_data,name,current_title,current_company,original_url,original_urls"
    for i in range(0, len(normalized), chunk_size):
        chunk = normalized[i:i + chunk_size]
        quoted = quote_url_list(chunk)
        filters = {
            "or": (
                f"(linkedin_url.in.({quoted}),"
                f"original_url.in.({quoted}),"
                f"original_urls.ov.{{{quoted}}})"
            ),
        }
        try:
            rows = client.select("profiles", columns, filters=filters, limit=len(chunk) * 5)
        except Exception:
            # Fallback if the combined OR is rejected (e.g. original_urls
            # column missing on an un-migrated DB) -- plain linkedin_url
            # lookup, matching dashboard.py:611-613.
            rows = client.select(
                "profiles", columns,
                filters={"linkedin_url": f"in.({quoted})"},
                limit=len(chunk),
            )
        for row in rows:
            aliases = set()
            for raw in (row.get("linkedin_url"), row.get("original_url"), *(row.get("original_urls") or [])):
                norm = normalize_linkedin_url(raw) or raw
                if norm:
                    aliases.add(norm)
            for alias in aliases:
                by_url[alias] = row
    return by_url


def _fetch_screening_results(client: SupabaseClient, jd_hash: str, urls) -> list:
    """screening_results rows for source_project='sourcingx', this JD, and
    these URLs. Paged in chunks to stay clear of PostgREST's URL-length/row
    limits, same chunking as fetch_profiles_by_urls."""
    all_rows = []
    normalized = sorted({normalize_linkedin_url(u) or u for u in urls if u})
    for i in range(0, len(normalized), DEFAULT_PROFILE_CHUNK):
        chunk = normalized[i:i + DEFAULT_PROFILE_CHUNK]
        quoted = quote_url_list(chunk)
        rows = client.select(
            "screening_results",
            "linkedin_url,screening_score,screening_fit_level,screening_summary,"
            "screening_reasoning,ai_model,screened_at",
            filters={
                "source_project": "eq.sourcingx",
                "jd_hash": f"eq.{jd_hash}",
                "linkedin_url": f"in.({quoted})",
            },
            limit=len(chunk) * 5,
        )
        all_rows.extend(rows)
    return all_rows


def _read_jev_csv(path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[(row.get("position_id"), row.get("linkedin_url"))] = row
    return out


# ============================================================================
# sample
# ============================================================================

def cmd_sample(args):
    client = _load_supabase_client()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    picked_rows = []
    eligible_counts = {}

    for position_id in args.positions:
        candidates = _fetch_pipeline_candidates(client, position_id)
        print(f"[{position_id}] fetched {len(candidates)} pipeline_candidates row(s)")

        kept = []
        needed_urls = set()
        for row in candidates:
            url = row.get("linkedin_url")
            if not url:
                continue
            if is_excluded_kalamata_row(row.get("screening_notes"), row.get("screening_score")):
                continue
            group = classify_kalamata_group(row.get("screening_result"), row.get("screening_notes"))
            if group is None:
                continue
            kept.append((group, row))
            needed_urls.add(url)

        profiles_by_url = fetch_profiles_by_urls(client, needed_urls)

        all_enriched = []
        for group, row in kept:
            norm = normalize_linkedin_url(row["linkedin_url"]) or row["linkedin_url"]
            profile = profiles_by_url.get(norm)
            if profile and is_eligible_profile(profile):
                enriched = dict(row)
                enriched["profile_url"] = (
                    normalize_linkedin_url(profile.get("linkedin_url")) or profile.get("linkedin_url")
                )
                enriched["group"] = group
                all_enriched.append(enriched)

        # One person (profile_url) must appear once, in one group -- kalamata
        # can hold more than one alias URL for the same profile, and those
        # aliases can fall into different groups. Dedupe BEFORE the seeded
        # per-group sampling below, so the sample never double-counts a person.
        deduped = dedupe_sample_rows_by_profile_url(all_enriched)
        dropped = len(all_enriched) - len(deduped)
        print(f"[{position_id}] dropped {dropped} alias duplicate(s) by profile_url")

        groups = defaultdict(list)
        for enriched in deduped:
            groups[enriched["group"]].append(enriched)

        pos_eligible = {g: len(groups.get(g, [])) for g in ("yes", "no_fullscreen", "no_prescreen")}
        eligible_counts[position_id] = pos_eligible
        print(f"[{position_id}] eligible: " + ", ".join(f"{g}={n}" for g, n in pos_eligible.items()))

        pos_picked = {}
        for group in ("yes", "no_fullscreen", "no_prescreen"):
            rows = groups.get(group, [])
            by_url = {r["linkedin_url"]: r for r in rows}
            picked_urls = deterministic_sample(list(by_url.keys()), args.per_group, args.seed)
            for u in picked_urls:
                r = by_url[u]
                picked_rows.append({
                    "position_id": position_id,
                    "linkedin_url": u,
                    "profile_url": r.get("profile_url") or u,
                    "group": group,
                    "kalamata_verdict": "yes" if group == "yes" else "no",
                    "kalamata_score": r.get("screening_score"),
                    "kalamata_reason": kalamata_reason(r),
                })
            pos_picked[group] = len(picked_urls)
        print(f"[{position_id}] picked: " + ", ".join(f"{g}={n}" for g, n in pos_picked.items()))

    sample_path = out_dir / "sample.json"
    sample_path.write_text(json.dumps({
        "seed": args.seed,
        "per_group": args.per_group,
        "positions": args.positions,
        "eligible_counts": eligible_counts,
        "rows": picked_rows,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {sample_path} ({len(picked_rows)} row(s))")

    # Upload CSVs carry profile_url, not the pipeline linkedin_url: that's the
    # profiles.linkedin_url the row matched to, and it's what the dashboard
    # will load, screen, and save results under.
    by_position = defaultdict(set)
    for r in picked_rows:
        by_position[r["position_id"]].add(r["profile_url"])
    for position_id, urls in by_position.items():
        csv_path = out_dir / f"upload_{position_id}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["linkedin_url"])
            for u in sorted(urls):
                writer.writerow([u])
        print(f"Wrote {csv_path} ({len(urls)} url(s))")


# ============================================================================
# jev
# ============================================================================

def cmd_jev(args):
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    briefs = json.loads(Path(args.briefs).read_text(encoding="utf-8"))
    rows = sample["rows"]
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    existing_rows = []
    done = set()
    if out_path.exists():
        with open(out_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                key = (row.get("position_id"), row.get("linkedin_url"))
                has_error = bool((row.get("error") or "").strip())
                if has_error:
                    # Retry this row -- drop it from existing_rows so it isn't
                    # written twice, and leave it out of `done` so it lands in todo.
                    continue
                # No error but an unknown/blank jev_verdict is Jev's normal
                # fail-open "incomplete" outcome (see jev_client), not a
                # failure -- keep it as done, don't re-call. cmd_report
                # reports it as jev_status="incomplete", distinct from a
                # genuinely missing row.
                existing_rows.append(row)
                done.add(key)

    todo = [r for r in rows if (r["position_id"], r["linkedin_url"]) not in done]

    client = _load_supabase_client()
    needed_urls = {r["linkedin_url"] for r in todo} | {r["profile_url"] for r in todo if r.get("profile_url")}
    profiles_by_url = fetch_profiles_by_urls(client, needed_urls)

    if not args.yes:
        total_chars = 0
        for r in todo:
            brief = briefs.get(r["position_id"]) or {}
            jd = brief_to_job_description(brief)
            profile = profiles_by_url.get(preferred_match_url(r))
            if profile:
                total_chars += estimate_jev_prompt_chars(profile, jd, brief)
        print(f"DRY RUN -- would make {len(todo)} Jev call(s), 0 made.")
        print(f"({len(existing_rows)} row(s) already in {out_path} would be skipped/resumed.)")
        if todo:
            print(
                f"Rough estimated total input size: ~{total_chars} chars "
                f"(~{total_chars // 4} tokens, chars/4 approximation -- "
                "jev_client.build_questions() needs typesafe_sdk, which isn't "
                "installed, so a real SDK token count isn't available here)."
            )
        return

    jev_real_client = jev_client.build_client()
    fieldnames = [
        "position_id", "linkedin_url", "jev_verdict", "jev_score", "jev_fit_level",
        "jev_reason", "jev_model", "input_tokens", "output_tokens", "error",
    ]

    def process(row):
        brief = briefs.get(row["position_id"]) or {}
        jd = brief_to_job_description(brief)
        profile = profiles_by_url.get(preferred_match_url(row))
        base = {"position_id": row["position_id"], "linkedin_url": row["linkedin_url"]}
        if not profile:
            return {**base, "jev_verdict": "", "jev_score": "", "jev_fit_level": "",
                    "jev_reason": "", "jev_model": "", "input_tokens": "", "output_tokens": "",
                    "error": "profile not found in profiles table"}
        try:
            verdict = jev_client.screen_with_jev(profile, jd, client=jev_real_client, screening_brief=brief)
        except Exception as e:  # noqa: BLE001 -- one bad candidate must not kill the batch
            return {**base, "jev_verdict": "", "jev_score": "", "jev_fit_level": "",
                    "jev_reason": "", "jev_model": "", "input_tokens": "", "output_tokens": "",
                    "error": f"{type(e).__name__}: {e}"}
        result = verdict.get("screening_result")
        reason = verdict.get("reason") or ""
        # jev_client.screen_with_jev() does NOT raise on an API failure -- it
        # fails open with a deferral verdict whose reason is prefixed
        # "Jev call failed: ..." and jev_model is None (jev_client.py:682-687).
        # That's a call that didn't actually happen and must be retried on
        # resume, so it goes into `error`, same as an exception. A genuine
        # low-confidence deferral (any other _defer() reason) is a real,
        # completed Jev answer -- "incomplete" but done, not an error.
        if reason.startswith("Jev call failed:"):
            return {**base, "jev_verdict": "", "jev_score": "", "jev_fit_level": result or "",
                    "jev_reason": reason[:200], "jev_model": verdict.get("jev_model") or "",
                    "input_tokens": verdict.get("input_tokens"), "output_tokens": verdict.get("output_tokens"),
                    "error": reason}
        return {
            **base,
            "jev_verdict": jev_verdict_from_result(result) or "",
            "jev_score": verdict.get("screening_score"),
            "jev_fit_level": result or "",
            "jev_reason": reason[:200],
            "jev_model": verdict.get("jev_model") or "",
            "input_tokens": verdict.get("input_tokens"),
            "output_tokens": verdict.get("output_tokens"),
            "error": "",
        }

    lock = threading.Lock()
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process, r) for r in todo]
            for fut in as_completed(futures):
                result_row = fut.result()
                with lock:
                    writer.writerow(result_row)
                    f.flush()

    print(f"Wrote {out_path} ({len(existing_rows) + len(todo)} total row(s))")


# ============================================================================
# report
# ============================================================================

def cmd_report(args):
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    briefs = json.loads(Path(args.briefs).read_text(encoding="utf-8"))
    jev_rows = _read_jev_csv(args.jev)

    client = _load_supabase_client()
    all_urls = {r["linkedin_url"] for r in sample["rows"]} | {
        r["profile_url"] for r in sample["rows"] if r.get("profile_url")
    }
    profiles_by_url = fetch_profiles_by_urls(client, all_urls)

    luna_by_key = {}
    for position_id in sample["positions"]:
        brief = briefs.get(position_id) or {}
        jd = brief_to_job_description(brief)
        jd_hash = compute_jd_hash(jd)
        pos_rows = [r for r in sample["rows"] if r["position_id"] == position_id]
        # Luna may have saved verdicts under either the matched profile_url or
        # the raw pipeline linkedin_url -- query for both variants.
        pos_urls = {r["linkedin_url"] for r in pos_rows} | {
            r["profile_url"] for r in pos_rows if r.get("profile_url")
        }
        results = _fetch_screening_results(client, jd_hash, pos_urls)

        latest = {}
        for row in results:
            url = normalize_linkedin_url(row.get("linkedin_url")) or row.get("linkedin_url")
            prev = latest.get(url)
            if prev is None or (row.get("screened_at") or "") >= (prev.get("screened_at") or ""):
                latest[url] = row
        for url, row in latest.items():
            luna_by_key[(position_id, url)] = row

    bakeoff_rows = []
    for r in sample["rows"]:
        norm_url = preferred_match_url(r)
        profile = profiles_by_url.get(norm_url) or {}

        # Luna's saved row is looked up by normalized profile_url first (the
        # URL the dashboard actually screened and saved under), and only
        # falls back to the normalized pipeline linkedin_url when that
        # misses -- some rows were screened before profile_url existed, or
        # Luna was run against the raw pipeline URL directly.
        profile_norm = normalize_linkedin_url(r.get("profile_url")) or r.get("profile_url")
        pipeline_norm = normalize_linkedin_url(r.get("linkedin_url")) or r.get("linkedin_url")
        luna_row = luna_by_key.get((r["position_id"], profile_norm)) if profile_norm else None
        if luna_row is None and pipeline_norm and pipeline_norm != profile_norm:
            luna_row = luna_by_key.get((r["position_id"], pipeline_norm))

        jev_row = jev_rows.get((r["position_id"], r["linkedin_url"]))

        luna_v = luna_verdict(luna_row.get("screening_fit_level")) if luna_row else None
        jev_raw_verdict = (jev_row or {}).get("jev_verdict") or None
        jev_v = jev_raw_verdict if jev_raw_verdict in ("yes", "no") else None
        jev_error = (jev_row or {}).get("error") or ""
        if jev_v:
            jev_status = jev_v
        elif jev_row and not jev_error:
            # Row exists, no error, but no yes/no verdict -- Jev's fail-open
            # "incomplete" outcome, not a missing call.
            jev_status = "incomplete"
        else:
            jev_status = "missing"
        kal_v = r["kalamata_verdict"]

        bakeoff_rows.append({
            "position_id": r["position_id"],
            "group": r["group"],
            "linkedin_url": r["linkedin_url"],
            "name": profile.get("name") or "",
            "current_title": profile.get("current_title") or "",
            "current_company": profile.get("current_company") or "",
            "luna_verdict": luna_v or "missing",
            "luna_score": luna_row.get("screening_score") if luna_row else "",
            "luna_reason": ((luna_row or {}).get("screening_summary") or "")[:200],
            "jev_verdict": jev_v or "missing",
            "jev_status": jev_status,
            "jev_score": _score_value((jev_row or {}).get("jev_score")),
            "jev_reason": (jev_row or {}).get("jev_reason") or "",
            "kalamata_verdict": kal_v,
            "kalamata_score": r["kalamata_score"],
            "kalamata_reason": r["kalamata_reason"],
            "luna_vs_kalamata_disagree": bool(luna_v and luna_v != kal_v),
            "jev_vs_kalamata_disagree": bool(jev_v and jev_v != kal_v),
            "luna_vs_jev_disagree": bool(luna_v and jev_v and luna_v != jev_v),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "bakeoff.csv"
    fieldnames = list(bakeoff_rows[0].keys()) if bakeoff_rows else [
        "position_id", "group", "linkedin_url", "name", "current_title", "current_company",
        "luna_verdict", "luna_score", "luna_reason", "jev_verdict", "jev_status", "jev_score", "jev_reason",
        "kalamata_verdict", "kalamata_score", "kalamata_reason",
        "luna_vs_kalamata_disagree", "jev_vs_kalamata_disagree", "luna_vs_jev_disagree",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bakeoff_rows)
    print(f"Wrote {csv_path} ({len(bakeoff_rows)} row(s))")

    summary = {}
    for position_id in list(sample["positions"]) + ["overall"]:
        rows_for = bakeoff_rows if position_id == "overall" else [
            r for r in bakeoff_rows if r["position_id"] == position_id
        ]
        summary[position_id] = {
            "count": len(rows_for),
            "luna_kalamata_agreement": compute_agreement(rows_for, "luna_verdict", "kalamata_verdict"),
            "jev_kalamata_agreement": compute_agreement(rows_for, "jev_verdict", "kalamata_verdict"),
            "luna_jev_agreement": compute_agreement(rows_for, "luna_verdict", "jev_verdict"),
            "missing_luna": sum(1 for r in rows_for if r["luna_verdict"] == "missing"),
            "missing_jev": sum(1 for r in rows_for if r["jev_verdict"] == "missing"),
            "incomplete_jev": sum(1 for r in rows_for if r.get("jev_status") == "incomplete"),
        }

    disagreements = [
        r for r in bakeoff_rows
        if r["luna_vs_kalamata_disagree"] or r["jev_vs_kalamata_disagree"] or r["luna_vs_jev_disagree"]
    ]
    luna_kalamata_disagreements = [r for r in bakeoff_rows if r["luna_vs_kalamata_disagree"]]
    review_sample = pick_review_sample(luna_kalamata_disagreements, args.seed, 30)

    json_path = out_dir / "bakeoff.json"
    json_path.write_text(json.dumps({
        "seed": args.seed,
        "summary": summary,
        "rows": bakeoff_rows,
        "disagreements": disagreements,
        "review_sample": review_sample,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample", help="Pick the candidate sample from kalamata's data")
    p_sample.add_argument("--positions", nargs="+", required=True)
    p_sample.add_argument("--per-group", type=int, default=50)
    p_sample.add_argument("--seed", type=int, required=True)
    p_sample.add_argument("--out-dir", required=True)
    p_sample.set_defaults(func=cmd_sample)

    p_jev = sub.add_parser("jev", help="Screen the sample through Jev (dry run unless --yes)")
    p_jev.add_argument("--sample", required=True)
    p_jev.add_argument("--briefs", required=True)
    p_jev.add_argument("--out", required=True)
    p_jev.add_argument("--yes", action="store_true")
    p_jev.add_argument("--limit", type=int, default=None)
    p_jev.add_argument("--workers", type=int, default=4)
    p_jev.set_defaults(func=cmd_jev)

    p_report = sub.add_parser("report", help="Build bakeoff.csv / bakeoff.json")
    p_report.add_argument("--sample", required=True)
    p_report.add_argument("--jev", required=True)
    p_report.add_argument("--briefs", required=True)
    p_report.add_argument("--out-dir", required=True)
    p_report.add_argument("--seed", type=int, required=True)
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
