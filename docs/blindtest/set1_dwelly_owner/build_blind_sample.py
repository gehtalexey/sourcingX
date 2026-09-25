"""
Build an anonymised blind-test sample of 40 candidate profiles
(20 for 'dwelly', 20 for 'owner'), split kalamata-qualified /
kalamata-not_qualified / sourcingx-good-fit / sourcingx-not-a-fit.

READ-ONLY on the shared Supabase DB. No paid API calls of any kind.
Writes only inside this scratchpad's blind/ folder — never into the repo.
"""
import sys
import os
import re
import json
import hashlib
from datetime import datetime

REPO = r"C:\Users\gehta\projects\sourcingX"
sys.path.insert(0, REPO)
os.chdir(REPO)  # so db.py's config.json lookup (Path(__file__).parent) resolves

import db  # noqa: E402
from normalizers import normalize_linkedin_url  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(OUT_DIR, "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

SALT = "blind20260924"

POSITIONS = {
    "dwelly-ai-applied-eng-eu": "dwelly",
    "owner-staff-agents-eng-sf": "owner",
}
SOURCINGX_PREFIXES = {
    "dwelly": "Role: Applied AI Engineer at Dwelly",
    "owner": "Role: Senior / Staff AI Agents Engineer at Owner",
}

N_PER_GROUP_KALAMATA = 7
N_SOURCINGX = 6
N_SOURCINGX_GOOD_FIT_TARGET = 3


def md5_key(url, salt=SALT):
    return hashlib.md5((url + salt).encode("utf-8")).hexdigest()


def emp_field(emp, field):
    """Read an employer entry field across old/new Crustdata dialects."""
    if not isinstance(emp, dict):
        return None
    if field == "title":
        return emp.get("employee_title") or emp.get("title")
    if field == "company":
        return emp.get("employer_name") or emp.get("name")
    if field == "description":
        return emp.get("employee_description") or emp.get("description")
    return emp.get(field)


def fmt_date(d):
    """Crustdata dates come as ISO datetimes like '2026-07-01T00:00:00' or plain strings."""
    if not d:
        return ""
    s = str(d).strip()
    if not s or s.lower() in ("present", "current", "none", "null"):
        return "present"
    m = re.match(r"^(\d{4})-(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return s


def has_real_work_history(raw):
    if not isinstance(raw, dict):
        return False
    for key in ("past_employers", "current_employers"):
        for emp in (raw.get(key) or []):
            title = (emp_field(emp, "title") or "").strip()
            company = (emp_field(emp, "company") or "").strip()
            if title and company:
                return True
    return False


def has_skills_or_summary(raw):
    if not isinstance(raw, dict):
        return False
    skills = raw.get("skills")
    if isinstance(skills, list) and any(str(s).strip() for s in skills):
        return True
    if isinstance(skills, str) and skills.strip():
        return True
    summary = raw.get("summary")
    if isinstance(summary, str) and summary.strip():
        return True
    return False


def is_eligible(raw):
    return has_real_work_history(raw) and has_skills_or_summary(raw)


# ---------------------------------------------------------------------------
# Fetch screening_results
# ---------------------------------------------------------------------------

client = db.get_supabase_client()
if not client:
    print("ERROR: could not get Supabase client (check config.json exists)")
    sys.exit(1)

kalamata_rows = client.select(
    "screening_results",
    columns="linkedin_url,position_id,screening_result,screening_score,screened_at",
    filters={
        "source_project": "eq.autopilot",
        "position_id": "in.(dwelly-ai-applied-eng-eu,owner-staff-agents-eng-sf)",
    },
    limit=10000,
)

sourcingx_rows_all = client.select(
    "screening_results",
    columns="linkedin_url,jd_title,screening_fit_level,screening_score,screened_at",
    filters={"source_project": "eq.sourcingx"},
    limit=5000,
)

print(f"[fetch] kalamata rows: {len(kalamata_rows)}, sourcingx rows (all jd_titles): {len(sourcingx_rows_all)}")


def dedup_latest(rows, key_fn):
    """Keep the latest row (by screened_at) per key."""
    best = {}
    for r in rows:
        k = key_fn(r)
        if k is None:
            continue
        prev = best.get(k)
        if prev is None or (r.get("screened_at") or "") > (prev.get("screened_at") or ""):
            best[k] = r
    return best


# --- kalamata, split per position ---
kalamata_by_position = {"dwelly": [], "owner": []}
for r in kalamata_rows:
    pos = POSITIONS.get(r.get("position_id"))
    if pos:
        kalamata_by_position[pos].append(r)

kalamata_dedup = {}
for pos, rows in kalamata_by_position.items():
    kalamata_dedup[pos] = dedup_latest(rows, lambda r: normalize_linkedin_url(r.get("linkedin_url")))

# --- sourcingx, split per position by jd_title prefix ---
sourcingx_by_position = {"dwelly": [], "owner": []}
for r in sourcingx_rows_all:
    title = r.get("jd_title") or ""
    for pos, prefix in SOURCINGX_PREFIXES.items():
        if title.startswith(prefix):
            sourcingx_by_position[pos].append(r)
            break

sourcingx_dedup = {}
for pos, rows in sourcingx_by_position.items():
    sourcingx_dedup[pos] = dedup_latest(rows, lambda r: normalize_linkedin_url(r.get("linkedin_url")))

for pos in ("dwelly", "owner"):
    print(f"[fetch] {pos}: kalamata unique urls={len(kalamata_dedup[pos])}, "
          f"sourcingx unique urls={len(sourcingx_dedup[pos])}")


def map_kalamata_result(raw_result):
    if raw_result is None:
        return None
    v = str(raw_result).strip().lower()
    if v in ("go", "qualified"):
        return "qualified"
    if v in ("no_go", "not_qualified"):
        return "not_qualified"
    return v


# ---------------------------------------------------------------------------
# Gather all candidate urls we might need profiles for (union across groups)
# ---------------------------------------------------------------------------

all_candidate_urls = set()
for pos in ("dwelly", "owner"):
    all_candidate_urls.update(kalamata_dedup[pos].keys())
    all_candidate_urls.update(sourcingx_dedup[pos].keys())

print(f"[fetch] total unique candidate urls across both positions: {len(all_candidate_urls)}")

# Fetch profiles: primary pass by linkedin_url
profiles_by_url = {}
raw_profile_rows = db.get_profiles_by_urls(client, list(all_candidate_urls), include_raw_data=True)
for p in raw_profile_rows:
    u = normalize_linkedin_url(p.get("linkedin_url"))
    if u:
        profiles_by_url[u] = p

missing = [u for u in all_candidate_urls if u not in profiles_by_url]
print(f"[fetch] direct linkedin_url match: {len(profiles_by_url)}; missing after direct match: {len(missing)}")

# Fallback pass: try matching missing urls against original_url / original_urls
if missing:
    # Fetch all profiles that have an original_url or non-empty original_urls
    # and try to match client-side (small enough result set expected).
    fallback_rows = client.select(
        "profiles",
        columns="linkedin_url,original_url,original_urls,raw_data",
        filters={"or": "(original_url.not.is.null,original_urls.not.is.null)"},
        limit=20000,
    )
    orig_index = {}
    for p in fallback_rows:
        candidates = set()
        if p.get("original_url"):
            candidates.add(normalize_linkedin_url(p["original_url"]))
        for ou in (p.get("original_urls") or []):
            candidates.add(normalize_linkedin_url(ou))
        for c in candidates:
            if c:
                orig_index[c] = p

    recovered = 0
    for u in missing:
        if u in orig_index:
            profiles_by_url[u] = orig_index[u]
            recovered += 1
    print(f"[fetch] recovered via original_url/original_urls fallback: {recovered}")

still_missing = [u for u in all_candidate_urls if u not in profiles_by_url]
print(f"[fetch] still missing (no stored profile at all): {len(still_missing)}")


def get_raw(url):
    p = profiles_by_url.get(url)
    if not p:
        return None
    return p.get("raw_data")


# ---------------------------------------------------------------------------
# Select 20 per position: 7 kalamata_qualified, 7 kalamata_not_qualified,
# 6 sourcingx (aim 3 good_fit, rest not_a_fit), deterministic md5 order,
# skipping ineligible profiles.
# ---------------------------------------------------------------------------

skip_reasons = {"no_profile": 0, "no_work_history": 0, "no_skills_or_summary": 0}


def eligible_ordered(url_dict):
    """url_dict: {url: row}. Returns urls sorted by md5 key, annotated with eligibility."""
    urls = sorted(url_dict.keys(), key=lambda u: md5_key(u))
    out = []
    for u in urls:
        raw = get_raw(u)
        if raw is None:
            skip_reasons["no_profile"] += 1
            continue
        if not has_real_work_history(raw):
            skip_reasons["no_work_history"] += 1
            continue
        if not has_skills_or_summary(raw):
            skip_reasons["no_skills_or_summary"] += 1
            continue
        out.append(u)
    return out


selection = []  # list of dicts: position, group, linkedin_url, kalamata_result, kalamata_score, sourcingx_fit_level_old

for pos in ("dwelly", "owner"):
    kal_rows = kalamata_dedup[pos]
    qualified_urls = {u: r for u, r in kal_rows.items() if map_kalamata_result(r.get("screening_result")) == "qualified"}
    not_qualified_urls = {u: r for u, r in kal_rows.items() if map_kalamata_result(r.get("screening_result")) == "not_qualified"}

    ordered_qualified = eligible_ordered(qualified_urls)
    ordered_not_qualified = eligible_ordered(not_qualified_urls)

    picked_qualified = ordered_qualified[:N_PER_GROUP_KALAMATA]
    picked_not_qualified = ordered_not_qualified[:N_PER_GROUP_KALAMATA]

    for u in picked_qualified:
        r = qualified_urls[u]
        selection.append({
            "position": pos, "group": "kalamata_qualified", "linkedin_url": u,
            "kalamata_result": r.get("screening_result"), "kalamata_score": r.get("screening_score"),
            "sourcingx_fit_level_old": None,
        })
    for u in picked_not_qualified:
        r = not_qualified_urls[u]
        selection.append({
            "position": pos, "group": "kalamata_not_qualified", "linkedin_url": u,
            "kalamata_result": r.get("screening_result"), "kalamata_score": r.get("screening_score"),
            "sourcingx_fit_level_old": None,
        })

    already_picked = set(picked_qualified) | set(picked_not_qualified)

    sx_rows = sourcingx_dedup[pos]
    good_fit_urls = {u: r for u, r in sx_rows.items()
                     if r.get("screening_fit_level") == "Good Fit" and u not in already_picked}
    not_fit_urls = {u: r for u, r in sx_rows.items()
                    if r.get("screening_fit_level") == "Not a Fit" and u not in already_picked}

    ordered_good_fit = eligible_ordered(good_fit_urls)
    ordered_not_fit = eligible_ordered(not_fit_urls)

    n_good = min(N_SOURCINGX_GOOD_FIT_TARGET, len(ordered_good_fit))
    n_not = N_SOURCINGX - n_good
    picked_good = ordered_good_fit[:n_good]
    picked_not = ordered_not_fit[:n_not]
    # If not enough not-fit to fill remainder, top up from good_fit leftovers
    if len(picked_not) < n_not:
        shortfall = n_not - len(picked_not)
        leftover_good = ordered_good_fit[n_good:n_good + shortfall]
        picked_good = picked_good + leftover_good

    for u in picked_good:
        r = good_fit_urls[u]
        selection.append({
            "position": pos, "group": "sourcingx_good_fit", "linkedin_url": u,
            "kalamata_result": None, "kalamata_score": None,
            "sourcingx_fit_level_old": r.get("screening_fit_level"),
        })
    for u in picked_not:
        r = not_fit_urls[u]
        selection.append({
            "position": pos, "group": "sourcingx_not_a_fit", "linkedin_url": u,
            "kalamata_result": None, "kalamata_score": None,
            "sourcingx_fit_level_old": r.get("screening_fit_level"),
        })

    print(f"[select] {pos}: kalamata_qualified={len(picked_qualified)}/{N_PER_GROUP_KALAMATA} "
          f"(eligible pool {len(ordered_qualified)}/{len(qualified_urls)}), "
          f"kalamata_not_qualified={len(picked_not_qualified)}/{N_PER_GROUP_KALAMATA} "
          f"(eligible pool {len(ordered_not_qualified)}/{len(not_qualified_urls)}), "
          f"sourcingx_good_fit={len(picked_good)} (target {N_SOURCINGX_GOOD_FIT_TARGET}, "
          f"eligible pool {len(ordered_good_fit)}/{len(good_fit_urls)}), "
          f"sourcingx_not_a_fit={len(picked_not)} "
          f"(eligible pool {len(ordered_not_fit)}/{len(not_fit_urls)})")

print(f"[select] TOTAL selected: {len(selection)} (target 40)")
print(f"[select] skip reasons: {skip_reasons}")

# ---------------------------------------------------------------------------
# Shuffle IDs across all 40 by md5(url + 'shuffle')
# ---------------------------------------------------------------------------

for item in selection:
    item["_shuffle_key"] = md5_key(item["linkedin_url"], salt=SALT + "shuffle")

selection.sort(key=lambda item: item["_shuffle_key"])
for i, item in enumerate(selection, start=1):
    item["id"] = f"P{i:02d}"
    del item["_shuffle_key"]

# ---------------------------------------------------------------------------
# Anonymisation helpers
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\-\.\s\(\)]{7,}\d)")
LINKEDIN_WORD_RE = re.compile(r"\blinkedin\b", re.IGNORECASE)
GITHUB_WORD_RE = re.compile(r"\bgithub\b", re.IGNORECASE)
AT_SIGN_RE = re.compile(r"\s*@\s*")


def strip_urls_emails_phones(text):
    if not text:
        return text
    text = URL_RE.sub("[link]", text)
    text = EMAIL_RE.sub("[link]", text)
    text = PHONE_RE.sub("[link]", text)
    # Bare mentions of "LinkedIn"/"GitHub" (not part of a URL, already
    # handled above) still identify a platform profile — scrub them too.
    text = LINKEDIN_WORD_RE.sub("[link]", text)
    text = GITHUB_WORD_RE.sub("[link]", text)
    # "@" is used in headlines as "Title @ Company" shorthand, not just in
    # emails (already stripped above). The verification grep checks for any
    # remaining "@" character, so replace the connector word too.
    text = AT_SIGN_RE.sub(" at ", text)
    return text


def name_tokens(raw):
    tokens = set()
    for field in ("first_name", "last_name", "name"):
        v = raw.get(field)
        if isinstance(v, str):
            for t in re.split(r"[\s,]+", v):
                t = t.strip()
                if len(t) >= 3:
                    tokens.add(t)
    return tokens


def redact_names(text, tokens):
    if not text or not tokens:
        return text
    for t in tokens:
        pattern = re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE)
        text = pattern.sub("[name]", text)
    return text


def clean(text, tokens):
    text = strip_urls_emails_phones(text)
    text = redact_names(text, tokens)
    return text


def location_str(raw):
    ld = raw.get("location_details")
    if isinstance(ld, dict):
        city = (ld.get("city") or "").strip()
        country = (ld.get("country") or "").strip()
        parts = [p for p in (city, country) if p]
        if parts:
            return ", ".join(parts)
    region = raw.get("region") or raw.get("location") or ""
    return region.strip()


def build_profile_text(raw):
    tokens = name_tokens(raw)

    headline = clean(raw.get("headline") or "", tokens)
    location = clean(location_str(raw), tokens)
    summary = clean(raw.get("summary") or "", tokens)

    lines = []
    lines.append(f"HEADLINE: {headline}")
    lines.append(f"LOCATION: {location}")
    lines.append("")
    lines.append("SUMMARY:")
    lines.append(summary if summary else "(none)")
    lines.append("")
    lines.append("EXPERIENCE:")

    roles = []
    for key in ("current_employers", "past_employers"):
        for emp in (raw.get(key) or []):
            title = emp_field(emp, "title") or ""
            company = emp_field(emp, "company") or ""
            if not title and not company:
                continue
            start = fmt_date(emp.get("start_date"))
            end = fmt_date(emp.get("end_date")) if key == "past_employers" else (fmt_date(emp.get("end_date")) or "present")
            if key == "current_employers" and not emp.get("end_date"):
                end = "present"
            desc = emp_field(emp, "description") or ""
            roles.append((start, title, company, end, desc))

    if not roles:
        lines.append("(none)")
    else:
        for start, title, company, end, desc in roles:
            title_c = clean(title, tokens)
            company_c = clean(company, tokens)  # company names kept, but still strip any accidental name/url
            desc_c = clean(desc, tokens)
            lines.append(f"- {title_c} at {company_c} ({start} - {end})")
            if desc_c:
                for dl in desc_c.splitlines():
                    lines.append(f"    {dl}")
    lines.append("")
    lines.append("EDUCATION:")
    edu = raw.get("education_background") or []
    if not edu:
        lines.append("(none)")
    else:
        for e in edu:
            school = clean(e.get("institute_name") or "", tokens)
            degree = clean(e.get("degree_name") or "", tokens)
            field = clean(e.get("field_of_study") or "", tokens)
            start = fmt_date(e.get("start_date"))
            end = fmt_date(e.get("end_date"))
            bits = [b for b in (degree, field) if b]
            degree_field = ", ".join(bits)
            years = f"{start} - {end}".strip(" -")
            line = f"- {school}"
            if degree_field:
                line += f" — {degree_field}"
            if years:
                line += f" ({years})"
            lines.append(line)
    lines.append("")
    lines.append("SKILLS:")
    skills = raw.get("skills")
    if isinstance(skills, list) and skills:
        skills_c = [clean(str(s), tokens) for s in skills]
        lines.append(", ".join(skills_c))
    elif isinstance(skills, str) and skills.strip():
        lines.append(clean(skills, tokens))
    else:
        lines.append("(none)")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Write files
# ---------------------------------------------------------------------------

manifest = []
sizes = []
for item in selection:
    url = item["linkedin_url"]
    raw = get_raw(url)
    text = build_profile_text(raw)
    path = os.path.join(PROFILES_DIR, f"{item['id']}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    sizes.append(len(text))
    manifest.append({
        "id": item["id"],
        "position": item["position"],
        "group": item["group"],
        "linkedin_url": url,
        "kalamata_result": item["kalamata_result"],
        "kalamata_score": item["kalamata_score"],
        "sourcingx_fit_level_old": item["sourcingx_fit_level_old"],
    })

manifest.sort(key=lambda m: m["id"])

manifest_path = os.path.join(OUT_DIR, "manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"[write] wrote {len(selection)} profile files to {PROFILES_DIR}")
print(f"[write] wrote manifest to {manifest_path}")
print(f"[write] avg file size: {sum(sizes)/len(sizes):.0f} chars (min={min(sizes)}, max={max(sizes)})")

# Counts per position/group
from collections import Counter
counts = Counter((m["position"], m["group"]) for m in manifest)
print("[summary] counts per position/group:")
for k in sorted(counts.keys()):
    print(f"  {k[0]:8s} {k[1]:24s} {counts[k]}")

print(f"[summary] still_missing candidate urls with no stored profile at all: {len(still_missing)}")
print(f"[summary] skip reasons across all candidate pools (includes overlap/repeats across groups): {skip_reasons}")
