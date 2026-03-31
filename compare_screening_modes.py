"""
Screening Mode Comparison Script
=================================
Tests 4 combinations on a sample of real profiles from the DB:
  1. gpt-4o       + detailed
  2. gpt-4o       + quick
  3. gpt-4o-mini  + detailed
  4. gpt-4o-mini  + quick

Outputs a CSV + console summary so you can compare score quality.
Run: python compare_screening_modes.py
"""

import json
import sys
import csv
import subprocess
import tempfile
import os
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -- Load config --------------------------------------------------------------
config_path = Path(__file__).parent / "config.json"
if not config_path.exists():
    print("ERROR: config.json not found")
    sys.exit(1)

with open(config_path) as f:
    config = json.load(f)

OPENAI_KEY    = config.get("openai_api_key", "")
ANTHROPIC_KEY = config.get("anthropic_api_key", "")
SUPABASE_URL  = config.get("supabase_url", "")
SUPABASE_KEY  = config.get("supabase_key", "")

if not OPENAI_KEY:
    print("ERROR: openai_api_key not set in config.json")
    sys.exit(1)

SAMPLE_SIZE = 50
WORKERS = 4

# -- Job Description -----------------------------------------------------------
JD = (
    "We are looking for a VP Marketing with 10+ years of B2B SaaS marketing experience, "
    "including at least 5 years in a VP or Head of Marketing leadership role managing a team of 5 or more. "
    "The candidate must have proven ownership of pipeline generation — demand gen, ABM, or product marketing — "
    "with measurable revenue impact such as MQLs, pipeline contribution, or CAC metrics. "
    "They must have built and scaled a marketing organization across multiple functions "
    "(demand gen, product marketing, content, brand) and worked cross-functionally with Sales, Product, and CS "
    "to drive go-to-market strategy. "
    "Candidates without in-house B2B SaaS experience, or with only B2C, agency, or brand-only backgrounds, "
    "should be rejected. "
    "Nice to have: experience marketing to technical audiences (developers, DevOps, security buyers), "
    "PLG or self-serve funnel experience, and familiarity with modern marketing tools such as "
    "HubSpot, Marketo, 6sense, or Salesforce. "
    "Strong bonus for candidates who have built marketing from early stage to scale at a "
    "well-funded or public B2B SaaS company."
)

COMBOS = [
    {"model": "gpt-4o-2024-08-06",        "mode": "detailed", "label": "gpt-4o / detailed",       "provider": "openai"},
    {"model": "claude-haiku-4-5-20251001","mode": "detailed", "label": "haiku / detailed",          "provider": "anthropic"},
    {"model": "claude-haiku-4-5-20251001","mode": "quick",    "label": "haiku / quick",             "provider": "anthropic"},
]

PRICING = {
    "gpt-4o-2024-08-06":          {"input": 2.50,  "cached_input": 1.25,   "output": 10.00},
    "gpt-4o-mini-2024-07-18":     {"input": 0.15,  "cached_input": 0.075,  "output": 0.60},
    "claude-haiku-4-5-20251001":  {"input": 0.80,  "cached_input": 0.08,   "output": 4.00},
}

# -- Worker script (runs in subprocess, has access to full Streamlit env) ------
PROJECT_DIR = str(Path(__file__).parent)

WORKER_SCRIPT = '''
import json, sys, time, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Project root injected by parent process
PROJECT_DIR = os.environ["SOURCINGX_PROJECT_DIR"]
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

# Streamlit entrypoint shim
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
import streamlit as st

# Import dashboard functions directly
from dashboard import screen_profile, compute_role_durations_cached, trim_raw_profile
from prompts import VP_MARKETING_NYC

ROLE_PROMPT = VP_MARKETING_NYC["prompt"]

def run(args):
    profiles = args["profiles"]
    jd       = args["jd"]
    model    = args["model"]
    mode     = args["mode"]
    key      = args["openai_key"]
    workers  = args["workers"]

    provider = args["provider"]
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=args["anthropic_key"])
        workers = min(workers, 2)  # Haiku has tighter rate limits — cap at 2 concurrent
    else:
        from openai import OpenAI
        client = OpenAI(api_key=key)

    results = []
    def screen_one(profile):
        name = profile.get("name") or "Unknown"
        t0 = time.time()
        result = screen_profile(
            profile=profile,
            job_description=jd,
            client=client,
            mode=mode,
            ai_model=model,
            ai_provider=provider,
            role_prompt=ROLE_PROMPT,
        )
        elapsed = round(time.time() - t0, 2)
        return {
            "name": name,
            "linkedin_url": profile.get("linkedin_url", ""),
            "model": model,
            "mode": mode,
            "score": result.get("score", 0),
            "fit": result.get("fit", ""),
            "summary": result.get("summary", ""),
            "elapsed_s": elapsed,
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(screen_one, p) for p in profiles]
        for fut in as_completed(futures):
            results.append(fut.result())

    print(json.dumps(results))

if __name__ == "__main__":
    args = json.loads(sys.argv[1])
    run(args)
'''

# -- Load profiles from Supabase -----------------------------------------------
def load_profiles(n: int) -> list:
    sys.path.insert(0, str(Path(__file__).parent))
    from db import SupabaseClient

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: No Supabase config in config.json")
        sys.exit(1)

    db = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)

    # Pull VP Marketing profiles in NYC — try several title patterns, merge & dedupe
    vp_title_patterns = [
        'vp%marketing',
        'vice president%marketing',
        'head of marketing',
        'chief marketing officer',
        'cmo',
        'vp%growth',
        'director%marketing',
    ]
    nyc_patterns = ['%new york%', '%nyc%', '%manhattan%', '%brooklyn%']

    print("Searching for VP Marketing / NYC profiles in Supabase...")
    seen = set()
    profiles = []

    for title_pat in vp_title_patterns:
        if len(profiles) >= n * 3:   # fetch more than needed so we can filter
            break
        batch = db.select(
            'profiles', '*',
            filters={
                'current_title': f'ilike.{title_pat}',
                'raw_data': 'not.is.null',
            },
            limit=200
        )
        for p in batch:
            url = p.get('linkedin_url', '')
            if url and url not in seen:
                seen.add(url)
                profiles.append(p)

    print(f"  Found {len(profiles)} VP Marketing profiles total")

    # Prefer NYC profiles, fall back to all if not enough
    nyc = [p for p in profiles if any(
        pat.strip('%').lower() in (p.get('location') or '').lower()
        for pat in ['new york', 'nyc', 'manhattan', 'brooklyn']
    )]
    print(f"  Of which {len(nyc)} are in NYC / New York")

    selected = nyc if len(nyc) >= 10 else profiles
    if len(selected) < 10:
        print(f"  WARNING: only {len(selected)} matching profiles found — using all available")

    result = selected[:n]
    print(f"  Using {len(result)} profiles for the test\n")
    return result

# -- Run one combo via subprocess ----------------------------------------------
def run_combo(profiles: list, combo: dict) -> list:
    label = combo["label"]
    print(f"\n{'-'*55}")
    print(f"  Running: {label}")
    print(f"{'-'*55}")

    # Write worker script + args to temp files (avoids Windows CLI length limit)
    worker_path = Path(tempfile.mktemp(suffix=".py"))
    args_path   = Path(tempfile.mktemp(suffix=".json"))
    worker_path.write_text(WORKER_SCRIPT, encoding="utf-8")
    args_path.write_text(json.dumps({
        "profiles":      profiles,
        "jd":            JD,
        "model":         combo["model"],
        "mode":          combo["mode"],
        "provider":      combo["provider"],
        "openai_key":    OPENAI_KEY,
        "anthropic_key": ANTHROPIC_KEY,
        "workers":       WORKERS,
    }), encoding="utf-8")

    # Worker script reads from file path instead of argv
    worker_src = WORKER_SCRIPT.replace(
        'args = json.loads(sys.argv[1])',
        'args = json.loads(open(sys.argv[1], encoding="utf-8").read())'
    )
    worker_path.write_text(worker_src, encoding="utf-8")

    try:
        env = os.environ.copy()
        env["SOURCINGX_PROJECT_DIR"] = PROJECT_DIR
        proc = subprocess.run(
            [sys.executable, str(worker_path), str(args_path)],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            env=env,
            timeout=600,
        )
        if proc.returncode != 0:
            print(f"  ERROR in worker:\n{proc.stderr[-2000:]}")
            return []

        # Last line of stdout is the JSON results
        lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
        results_line = lines[-1] if lines else "[]"
        results = json.loads(results_line)

        for r in sorted(results, key=lambda x: x.get("name", "")):
            print(f"  {r['name']:<32} score={r['score']}  fit={r['fit']}")

        return results
    finally:
        worker_path.unlink(missing_ok=True)
        args_path.unlink(missing_ok=True)

# -- Comparison table ----------------------------------------------------------
def compare(all_results: dict, profiles: list):
    baseline_label = "gpt-4o / detailed"
    baseline = {r["name"]: r for r in all_results.get(baseline_label, [])}

    print(f"\n{'='*75}")
    print("SCORE COMPARISON  (baseline = gpt-4o / detailed)")
    print(f"{'='*75}")
    col_w = 18
    header = f"{'Name':<30}" + "".join(f"{c['label']:<{col_w}}" for c in COMBOS)
    print(header)
    print("-" * len(header))

    score_diffs    = {c["label"]: [] for c in COMBOS if c["label"] != baseline_label}
    fit_mismatches = {c["label"]: 0   for c in COMBOS if c["label"] != baseline_label}

    names = sorted({r["name"] for results in all_results.values() for r in results})
    for name in names:
        base = baseline.get(name)
        if not base:
            continue
        row = f"{name[:29]:<30}"
        for combo in COMBOS:
            lbl = combo["label"]
            lookup = {r["name"]: r for r in all_results.get(lbl, [])}
            r = lookup.get(name)
            if not r:
                row += f"{'N/A':<{col_w}}"
                continue
            score = r["score"]
            fit_short = r["fit"].replace(" Fit", "").replace("Not a", "No")
            if lbl == baseline_label:
                row += f"{score} {fit_short:<{col_w-3}}"
            else:
                diff = score - base["score"]
                diff_str = f"({'+' if diff >= 0 else ''}{diff})"
                row += f"{score}{diff_str} {fit_short:<{col_w-7}}"
                score_diffs[lbl].append(abs(diff))
                if r["fit"] != base["fit"]:
                    fit_mismatches[lbl] += 1
        print(row)

    n = len(names)
    print(f"\n{'-'*75}")
    print("QUALITY STATS  (vs gpt-4o / detailed baseline)")
    print(f"{'-'*75}")
    print(f"{'Combo':<30} {'Avg diff':>9} {'Max diff':>9} {'Fit mismatch':>14}")
    print("-" * 65)
    for combo in COMBOS:
        lbl = combo["label"]
        if lbl == baseline_label:
            print(f"{lbl:<30} {'—':>9} {'—':>9} {'(baseline)':>14}")
            continue
        diffs = score_diffs[lbl]
        avg_d = round(sum(diffs)/len(diffs), 2) if diffs else 0
        max_d = max(diffs) if diffs else 0
        mm    = fit_mismatches[lbl]
        print(f"{lbl:<30} {avg_d:>9} {max_d:>9} {mm:>8} / {n} ({round(mm/n*100) if n else 0}%)")

# -- Cost table ----------------------------------------------------------------
def cost_summary():
    print(f"\n{'-'*80}")
    print("COST ESTIMATE — 4,000 profiles  (avg input ~3,167 tok based on actual prompt build)")
    print("  OpenAI:    caching automatic (87% hit rate from Mar 30 actuals)")
    print("  Haiku:     caching NOW ACTIVE via cache_control (charged at $0.08/1M after first call)")
    print(f"{'-'*80}")
    print(f"{'Combo':<30} {'Avg in':>8} {'Avg out':>8} {'Cost/4000':>12} {'vs 4o-det':>12}")
    print("-" * 80)

    N = 4000
    AVG_IN       = 3167   # actual measured from prompt build
    SYS_TOKENS   = 2310   # system prompt (cacheable)
    USER_TOKENS  =  857   # user prompt   (never cached — changes per profile)

    baseline_cost = None
    for combo in COMBOS:
        avg_out = 17 if combo["mode"] == "quick" else 100
        p = PRICING[combo["model"]]

        if combo["provider"] == "anthropic":
            # First call: full rate for everything + cache write
            # Remaining N-1: cached rate for system prompt, full rate for user prompt
            first_call   = (AVG_IN * p["input"] + avg_out * p["output"]) / 1_000_000
            cache_write  = (SYS_TOKENS * p["input"]) / 1_000_000   # one-time cache write cost
            subsequent   = ((SYS_TOKENS * p["cached_input"] + USER_TOKENS * p["input"]) * (N-1)
                            + avg_out * (N-1) * p["output"]) / 1_000_000
            cost = first_call + cache_write + subsequent
        else:
            # OpenAI: 87% cache hit on system prompt
            cached_in   = SYS_TOKENS * 0.87
            uncached_in = AVG_IN - cached_in
            cost = (uncached_in * N * p["input"]
                    + cached_in * N * p["cached_input"]
                    + avg_out   * N * p["output"]) / 1_000_000

        if baseline_cost is None:
            baseline_cost = cost
        savings = f"-{round((1 - cost/baseline_cost)*100)}%" if baseline_cost else "—"
        print(f"{combo['label']:<30} {AVG_IN:>8,} {avg_out:>8,} ${cost:>10.2f} {savings:>12}")

# -- Main ----------------------------------------------------------------------
def main():
    print("=" * 75)
    print("SourcingX — Screening Mode Comparison")
    print(f"  Profiles: {SAMPLE_SIZE}  |  Combos: {len(COMBOS)}  |  Total API calls: {SAMPLE_SIZE * len(COMBOS)}")
    print(f"  Role prompt: VP_MARKETING_NYC")
    print("=" * 75)

    profiles = load_profiles(SAMPLE_SIZE)
    if not profiles:
        print("ERROR: No profiles loaded")
        sys.exit(1)

    all_results = {}
    for combo in COMBOS:
        results = run_combo(profiles, combo)
        all_results[combo["label"]] = results

    compare(all_results, profiles)
    cost_summary()

    # Export CSV
    out_path = Path(__file__).parent / "screening_mode_comparison.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["name", "linkedin_url", "model", "mode", "score", "fit", "summary", "elapsed_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for results in all_results.values():
            writer.writerows(results)

    print(f"\nFull results saved to: {out_path}")
    print("Done.\n")

if __name__ == "__main__":
    main()
