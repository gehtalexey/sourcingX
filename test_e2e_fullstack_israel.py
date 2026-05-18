# -*- coding: utf-8 -*-
"""
End-to-end test: Senior Full Stack Developer in Israel
Tech stack: React, Node.js, AWS, Docker/Kubernetes, Claude/AI

Flow mirrors the SourcingX dashboard:
  STEP 1 - Search    (crustdata_search.py: build_filters + search_people_db)
  STEP 2 - Normalize (crustdata_search.py: normalize_search_results_to_df)
  STEP 3 - Filter    (title keywords, boolean search -- each tested one by one)
  STEP 4 - AI Screen (structured_screening.py via Anthropic)

Run:
    python test_e2e_fullstack_israel.py
"""

import io
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

# Force UTF-8 output so profile names with Hebrew/emoji don't crash
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from crustdata_search import (
    build_filters,
    search_people_db,
    normalize_search_results_to_df,
)

from structured_screening import (
    screen_profile_structured,
    parse_requirements,
)

import anthropic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DIVIDER = "-" * 70


def _sep(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def _print_filter_result(label, before, after):
    removed = before - after
    pct = round((after / before) * 100) if before else 0
    print(f"  {label:<50}  {before:>3} -> {after:>3}  (-{removed}, {pct}% kept)")


def _load_config():
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Inline filter helpers (mirrors dashboard.py apply_pre_filters logic)
# ---------------------------------------------------------------------------

def _title_contains(title_str, keywords):
    """Return True if title contains any of the keywords (case-insensitive)."""
    if not title_str:
        return False
    t = str(title_str).lower()
    return any(kw.lower() in t for kw in keywords)


def filter_exclude_titles(df, keywords):
    """Remove rows whose current_title contains any of the keywords."""
    col = "current_title" if "current_title" in df.columns else None
    if not col:
        return df
    mask = df[col].apply(lambda t: _title_contains(t, keywords))
    return df[~mask]


def filter_include_titles(df, keywords):
    """Keep only rows whose current_title contains at least one keyword."""
    col = "current_title" if "current_title" in df.columns else None
    if not col:
        return df
    mask = df[col].apply(lambda t: _title_contains(t, keywords))
    return df[mask]


def _profile_text(row, cols):
    """Concatenate profile columns into a single searchable string."""
    parts = []
    for c in cols:
        val = row.get(c)
        if val is None:
            continue
        if isinstance(val, (list, dict)):
            parts.append(json.dumps(val))
        else:
            parts.append(str(val))
    return " ".join(parts).lower()


def filter_boolean_fullprofile(df, query):
    """Filter by boolean query across skills, headline, summary, all_titles, all_employers."""
    try:
        from boolean_query import match_boolean_query as mbq
    except ImportError:
        print("  (boolean_query module not available -- skipping boolean filter)")
        return df

    cols = [c for c in ["skills", "headline", "summary", "all_titles", "all_employers"] if c in df.columns]
    if not cols:
        return df
    mask = df.apply(lambda r: mbq(query, _profile_text(r.to_dict(), cols)), axis=1)
    return df[mask]


def filter_boolean_skills(df, query):
    """Filter by boolean query on skills column only."""
    try:
        from boolean_query import match_boolean_query as mbq
    except ImportError:
        print("  (boolean_query module not available -- skipping skills boolean filter)")
        return df

    if "skills" not in df.columns:
        return df
    mask = df["skills"].fillna("").astype(str).apply(lambda t: mbq(query, t.lower()))
    return df[mask]


# ---------------------------------------------------------------------------
# STEP 1 - SEARCH
# ---------------------------------------------------------------------------
def step1_search(api_key):
    _sep("STEP 1 - SEARCH  (100 profiles, Full Stack Israel — expanded synonyms)")

    print("  Filters applied:")
    print("    Title      : Full Stack / Fullstack / Full-Stack / Software Engineer / Web Developer")
    print("    Seniority  : (none — relying on experience filter in Step 3)")
    print("    Country    : Israel")
    print("    Skills     : (React OR React.js OR ReactJS OR Next.js) AND (Node.js OR Node OR NodeJS OR Express OR NestJS)")
    print("    Headcount  : 11-50, 51-200, 201-500  (startups)")
    print("    Sort       : Connections desc")
    print("    Limit      : 100")

    filters = build_filters(
        title=(
            "Full Stack Developer, Full Stack Engineer, "
            "Fullstack Developer, Fullstack Engineer, "
            "Full-Stack Developer, Full-Stack Engineer, "
            "Software Engineer, Software Developer, "
            "Web Developer, Web Engineer"
        ),
        country="Israel",
        skill_groups=[
            "React, React.js, ReactJS, Next.js",
            "Node.js, Node, NodeJS, Express, NestJS",
        ],
        headcount=["11-50", "51-200", "201-500"],
    )

    sorts = [{"column": "num_of_connections", "order": "desc"}]

    print("\n  Calling Crustdata API...")
    t0 = time.time()
    results = search_people_db(filters, limit=100, sorts=sorts, api_key=api_key)
    elapsed = round(time.time() - t0, 1)

    profiles = results.get("profiles", [])
    total_available = results.get("total_count", len(profiles))
    credits_used = results.get("credits_used", 0)

    print(f"\n  OK  Got {len(profiles)} profiles in {elapsed}s")
    print(f"      Total matching in DB : {total_available:,}")
    print(f"      Credits used         : {credits_used}")
    sample = [p.get("name") or p.get("first_name", "?") for p in profiles[:5]]
    print(f"      First 5              : {', '.join(sample)}")

    return profiles


# ---------------------------------------------------------------------------
# STEP 2 - NORMALIZE
# ---------------------------------------------------------------------------
def step2_normalize(profiles):
    _sep("STEP 2 - NORMALIZE  (raw API -> DataFrame)")

    df = normalize_search_results_to_df(profiles)

    print(f"  Shape   : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns : {', '.join(df.columns.tolist())}")

    for col in ["name", "current_title", "current_company", "skills", "all_employers", "years_experience"]:
        if col in df.columns:
            filled = df[col].notna().sum()
            print(f"  {col:<30} {filled}/{len(df)} filled")

    return df


# ---------------------------------------------------------------------------
# STEP 3 - FILTER
# ---------------------------------------------------------------------------
def step3_filter(df):
    _sep("STEP 3 - FILTER  (each filter tested individually, then combined)")

    base = len(df)
    print(f"\n  Starting pool: {base} profiles\n")

    # -- A: Exclude Junior titles
    fa = filter_exclude_titles(df, ["junior", "intern", "student"])
    _print_filter_result("A) Exclude: junior / intern / student", base, len(fa))

    # -- B: Exclude Leadership titles
    fb = filter_exclude_titles(df, ["vp", "vice president", "director", "manager", "head of", "product manager"])
    _print_filter_result("B) Exclude: vp / director / manager / head of", base, len(fb))

    # -- C: Exclude non-FS technical roles
    fc = filter_exclude_titles(df, ["qa", "automation", "data", "machine learning", "ml", "mobile", "ios", "android", "devops", "security", "embedded"])
    _print_filter_result("C) Exclude: qa / data / ml / mobile / devops", base, len(fc))

    # -- D: Include only FS-flavored titles
    fd = filter_include_titles(df, ["developer", "engineer", "fullstack", "full stack", "full-stack"])
    _print_filter_result("D) Include only: developer / engineer / fullstack", base, len(fd))

    # -- E: Boolean full-profile: AWS
    fe = filter_boolean_fullprofile(df, "AWS OR Amazon Web Services OR amazon")
    _print_filter_result("E) Full-profile boolean: AWS OR Amazon Web Services", base, len(fe))

    # -- F: Skills boolean: Docker OR Kubernetes
    ff = filter_boolean_skills(df, "docker OR kubernetes OR container")
    _print_filter_result("F) Skills boolean: docker OR kubernetes OR container", base, len(ff))

    # -- G: Skills boolean: Claude OR OpenAI OR LLM (AI stack)
    fg = filter_boolean_skills(df, "claude OR openai OR langchain OR llm OR gpt OR ai agent")
    _print_filter_result("G) Skills boolean: Claude / OpenAI / LLM / AI agents", base, len(fg))

    # -- COMBINED: A+B+C+D, then E on top
    print(f"\n  --- COMBINED (A+B+C+D then boolean AWS) ---")
    combined = df.copy()
    combined = filter_exclude_titles(combined, [
        "junior", "intern", "student",
        "vp", "vice president", "director", "manager", "head of", "product manager",
        "qa", "automation", "data", "machine learning", "ml",
        "mobile", "ios", "android", "devops", "security", "embedded",
    ])
    combined = filter_include_titles(combined, ["developer", "engineer", "fullstack", "full stack", "full-stack"])
    combined = filter_boolean_fullprofile(combined, "AWS OR Amazon Web Services OR amazon")

    print(f"\n  Start    : {base}")
    print(f"  Remain   : {len(combined)}  (removed {base - len(combined)})")

    if not combined.empty:
        print(f"\n  Profiles passing all filters:")
        for _, row in combined.head(10).iterrows():
            name = row.get("name") or "?"
            title = row.get("current_title") or "?"
            company = row.get("current_company") or "?"
            yoe = row.get("years_experience") or "?"
            print(f"    {name:<32}  {str(title)[:38]:<38}  @ {str(company)[:25]}  ({yoe}y)")

    return combined


# ---------------------------------------------------------------------------
# STEP 4 - AI SCREEN
# ---------------------------------------------------------------------------
def step4_screen(df, config):
    _sep("STEP 4 - AI SCREEN  (Anthropic claude-haiku-4-5)")

    anthropic_key = config.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("  No Anthropic API key in config.json or ANTHROPIC_API_KEY env var.")
        print("  Skipping AI screen.")
        return

    client = anthropic.Anthropic(api_key=anthropic_key)

    JOB_DESCRIPTION = """
Role: Senior Full Stack Developer at an Israeli startup
Must-haves:
- React or React.js or Next.js (frontend)
- Node.js backend experience
- At least 4 years of professional software experience
Nice-to-haves:
- AWS or cloud experience (1 pt)
- Docker or Kubernetes (1 pt)
- AI/LLM tools: Claude, OpenAI, LangChain, MCP (1 pt)
- Startup company (11-200 employees) (1 pt)
Exclusions:
- Currently serving in IDF military only (no civilian product role)
- Pure DevOps / infrastructure engineer (no product development)
"""

    requirements = parse_requirements(JOB_DESCRIPTION, client)
    print("  Parsed requirements:")
    for category, reqs in requirements.items():
        if reqs:
            print(f"    {category}: {', '.join(r.description for r in reqs)}")

    # Screen top 20 profiles from filtered pool
    screen_pool = df.head(20)
    print(f"\n  Screening {len(screen_pool)} profiles (top 20 from filtered pool)...")
    print(f"  Model: claude-haiku-4-5-20251001\n")

    results_list = []
    for i, (_, row) in enumerate(screen_pool.iterrows()):
        # Build profile dict the screener understands.
        # _raw_search_result = full Crustdata API response (compact=False).
        raw = row.get("_raw_search_result") or {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = {}
        profile = {**raw}
        # Normalize field names used by structured_screening checkers
        profile["name"] = row.get("name") or raw.get("name") or "?"
        # check_experience_years reads total_experience_years, not years_of_experience_raw
        profile["total_experience_years"] = (
            raw.get("years_of_experience_raw")
            or raw.get("years_of_experience")
            or row.get("years_experience")
            or 0
        )
        # Ensure skills is a list
        skills = row.get("skills") or raw.get("skills") or []
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]
        profile["skills"] = skills

        name = profile.get("name") or f"Profile {i+1}"
        # Sanitize name for printing
        safe_name = name.encode("ascii", "replace").decode("ascii")

        try:
            result = screen_profile_structured(profile, requirements, client)
            results_list.append(result)
            if result.fit in ["Strong Fit", "Good Fit"]:
                verdict = "GO   "
            elif result.fit == "Partial Fit":
                verdict = "MAYBE"
            else:
                verdict = "NO GO"
            summary = (result.summary or "")[:90].encode("ascii", "replace").decode("ascii")
            print(f"  {i+1:>2}. [{verdict}]  {safe_name:<33}  score={result.score}/10  {result.fit}")
            if summary:
                print(f"       {summary}")
        except Exception as e:
            err = str(e).encode("ascii", "replace").decode("ascii")
            print(f"  {i+1:>2}. [ERROR] {safe_name}: {err}")

    _sep("SCREENING SUMMARY")
    go = [r for r in results_list if r.fit in ["Strong Fit", "Good Fit"]]
    maybe = [r for r in results_list if r.fit == "Partial Fit"]
    no_go = [r for r in results_list if r.fit == "Not a Fit"]

    print(f"  Screened : {len(results_list)}")
    pct_go = round(len(go) / len(results_list) * 100) if results_list else 0
    print(f"  GO       : {len(go)}  ({pct_go}%)")
    print(f"  MAYBE    : {len(maybe)}")
    print(f"  NO GO    : {len(no_go)}")

    if go:
        print(f"\n  Top GO candidates:")
        for r in sorted(go, key=lambda x: x.score, reverse=True)[:5]:
            safe = r.profile_name.encode("ascii", "replace").decode("ascii")
            print(f"    {safe:<35}  score={r.score}/10  [{r.fit}]")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("  SourcingX E2E Test -- Senior Full Stack Developer, Israel")
    print("  Stack: React, Node.js, AWS, Docker/K8s, Claude/AI")
    print("=" * 70)

    config = _load_config()
    api_key = config.get("api_key") or os.environ.get("CRUSTDATA_API_KEY")
    if not api_key:
        print("No Crustdata API key found. Check config.json.")
        sys.exit(1)

    profiles = step1_search(api_key)
    if not profiles:
        print("No profiles returned. Aborting.")
        sys.exit(1)

    df = step2_normalize(profiles)
    filtered_df = step3_filter(df)

    if filtered_df.empty:
        print("\n[!] All profiles filtered out -- skipping AI screen.")
    else:
        step4_screen(filtered_df, config)

    print("\n" + "=" * 70)
    print("  E2E test complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
