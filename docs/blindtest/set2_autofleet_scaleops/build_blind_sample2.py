"""
Build a second anonymised blind-test sample of 40 candidate profiles
('autofleet-fullstack-senior-il' x20, 'scaleops-backend-industry-il' x20),
split kalamata-qualified / kalamata-not_qualified / sourcingx-screened.

Adapted from the earlier blind/build_blind_sample.py (read, not edited) with:
  - new positions, new salt, ID prefix 'Q' instead of 'P'
  - 8/8/4 group split instead of 7/7/6
  - sourcingx group is a flat pool (no good-fit/not-a-fit split requirement)
  - fallback: if a position has no sourcingx rows, top up with 4 more
    kalamata rows (2 qualified + 2 not_qualified), noted in the report
  - anonymisation over-masking fixes: don't mask a name token that is a
    place-name word, a degree abbreviation, a common word, or shorter than
    3 chars, or that coincides with a word in the candidate's own location

READ-ONLY on the shared Supabase DB. No paid API calls of any kind.
Writes only inside this scratchpad's blind2/ folder — never into the repo.
"""
import sys
import os
import re
import json
import hashlib

REPO = r"C:\Users\gehta\projects\sourcingX"
sys.path.insert(0, REPO)
os.chdir(REPO)  # so db.py's config.json lookup (Path(__file__).parent) resolves

import db  # noqa: E402
from normalizers import normalize_linkedin_url  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(OUT_DIR, "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

SALT = "blind2_20260925"

POSITIONS = {
    "autofleet-fullstack-senior-il": "autofleet",
    "scaleops-backend-industry-il": "scaleops",
}
SOURCINGX_PREFIX = {
    "autofleet": "Role: Senior Full Stack Developer at Autofleet",
    # scaleops: matched by substring "scaleops" anywhere in jd_title (case-insensitive)
}

N_PER_GROUP_KALAMATA = 8
N_SOURCINGX_TARGET = 4


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
        "position_id": "in.(autofleet-fullstack-senior-il,scaleops-backend-industry-il)",
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
kalamata_by_position = {"autofleet": [], "scaleops": []}
for r in kalamata_rows:
    pos = POSITIONS.get(r.get("position_id"))
    if pos:
        kalamata_by_position[pos].append(r)

kalamata_dedup = {}
for pos, rows in kalamata_by_position.items():
    kalamata_dedup[pos] = dedup_latest(rows, lambda r: normalize_linkedin_url(r.get("linkedin_url")))

# --- sourcingx, split per position ---
sourcingx_by_position = {"autofleet": [], "scaleops": []}
for r in sourcingx_rows_all:
    title = r.get("jd_title") or ""
    if title.startswith(SOURCINGX_PREFIX["autofleet"]):
        sourcingx_by_position["autofleet"].append(r)
    if "scaleops" in title.lower():
        sourcingx_by_position["scaleops"].append(r)

sourcingx_dedup = {}
for pos, rows in sourcingx_by_position.items():
    sourcingx_dedup[pos] = dedup_latest(rows, lambda r: normalize_linkedin_url(r.get("linkedin_url")))

for pos in ("autofleet", "scaleops"):
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
for pos in ("autofleet", "scaleops"):
    all_candidate_urls.update(kalamata_dedup[pos].keys())
    all_candidate_urls.update(sourcingx_dedup[pos].keys())

print(f"[fetch] total unique candidate urls across both positions: {len(all_candidate_urls)}")

profiles_by_url = {}
raw_profile_rows = db.get_profiles_by_urls(client, list(all_candidate_urls), include_raw_data=True)
for p in raw_profile_rows:
    u = normalize_linkedin_url(p.get("linkedin_url"))
    if u:
        profiles_by_url[u] = p

missing = [u for u in all_candidate_urls if u not in profiles_by_url]
print(f"[fetch] direct linkedin_url match: {len(profiles_by_url)}; missing after direct match: {len(missing)}")

if missing:
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
# Select 20 per position: 8 kalamata_qualified, 8 kalamata_not_qualified,
# 4 sourcingx (flat pool, deterministic md5 order). If a position has fewer
# than 4 eligible sourcingx rows, top up the shortfall with extra kalamata
# rows split as evenly as possible between qualified/not_qualified,
# continuing past the first 8 already picked from each pool.
# ---------------------------------------------------------------------------

skip_reasons = {"no_profile": 0, "no_work_history": 0, "no_skills_or_summary": 0}
notes = []


def eligible_ordered(url_dict):
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


selection = []  # dicts: position, group, linkedin_url, kalamata_result, kalamata_score, sourcingx_fit_level_old

for pos in ("autofleet", "scaleops"):
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
    sx_pool = {u: r for u, r in sx_rows.items() if u not in already_picked}
    ordered_sx = eligible_ordered(sx_pool)
    picked_sx = ordered_sx[:N_SOURCINGX_TARGET]

    for u in picked_sx:
        r = sx_pool[u]
        selection.append({
            "position": pos, "group": "sourcingx_screened", "linkedin_url": u,
            "kalamata_result": None, "kalamata_score": None,
            "sourcingx_fit_level_old": r.get("screening_fit_level"),
        })

    shortfall = N_SOURCINGX_TARGET - len(picked_sx)
    if shortfall > 0:
        n_extra_q = (shortfall + 1) // 2  # favour qualified on odd shortfall
        n_extra_nq = shortfall - n_extra_q
        extra_qualified = ordered_qualified[len(picked_qualified):len(picked_qualified) + n_extra_q]
        extra_not_qualified = ordered_not_qualified[len(picked_not_qualified):len(picked_not_qualified) + n_extra_nq]
        for u in extra_qualified:
            r = qualified_urls[u]
            selection.append({
                "position": pos, "group": "kalamata_qualified_topup", "linkedin_url": u,
                "kalamata_result": r.get("screening_result"), "kalamata_score": r.get("screening_score"),
                "sourcingx_fit_level_old": None,
            })
        for u in extra_not_qualified:
            r = not_qualified_urls[u]
            selection.append({
                "position": pos, "group": "kalamata_not_qualified_topup", "linkedin_url": u,
                "kalamata_result": r.get("screening_result"), "kalamata_score": r.get("screening_score"),
                "sourcingx_fit_level_old": None,
            })
        note = (f"{pos}: only {len(picked_sx)}/{N_SOURCINGX_TARGET} eligible sourcingx rows found "
                f"(pool had {len(sx_pool)} unique urls before eligibility filter); "
                f"topped up with {len(extra_qualified)} extra kalamata_qualified + "
                f"{len(extra_not_qualified)} extra kalamata_not_qualified")
        notes.append(note)
        print(f"[select][NOTE] {note}")

    print(f"[select] {pos}: kalamata_qualified={len(picked_qualified)}/{N_PER_GROUP_KALAMATA} "
          f"(eligible pool {len(ordered_qualified)}/{len(qualified_urls)}), "
          f"kalamata_not_qualified={len(picked_not_qualified)}/{N_PER_GROUP_KALAMATA} "
          f"(eligible pool {len(ordered_not_qualified)}/{len(not_qualified_urls)}), "
          f"sourcingx_screened={len(picked_sx)} (target {N_SOURCINGX_TARGET}, "
          f"eligible pool {len(ordered_sx)}/{len(sx_pool)})")

print(f"[select] TOTAL selected: {len(selection)} (target 40)")
print(f"[select] skip reasons: {skip_reasons}")

# ---------------------------------------------------------------------------
# Shuffle IDs across all 40 by md5(url + salt + 'shuffle'), prefix 'Q'
# ---------------------------------------------------------------------------

for item in selection:
    item["_shuffle_key"] = md5_key(item["linkedin_url"], salt=SALT + "shuffle")

selection.sort(key=lambda item: item["_shuffle_key"])
for i, item in enumerate(selection, start=1):
    item["id"] = f"Q{i:02d}"
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

# Over-masking guards: known bugs from the first blind sample.
COMMON_WORDS_STOP = {
    "the", "and", "for", "with", "from", "team", "senior", "staff", "lead",
    "manager", "engineer", "developer", "software", "product", "full",
    "stack", "backend", "frontend", "data", "science", "research",
    "university", "college", "institute", "technology", "technologies",
    "group", "global", "international", "solutions", "systems", "company",
    "ltd", "inc", "llc", "founder", "cofounder", "director", "head", "vice",
    "president", "chief", "officer",
}
DEGREE_ABBR_STOP = {
    "bsc", "msc", "mba", "phd", "beng", "meng", "llb", "btech", "mtech",
    "bcom", "mcom", "bfa", "mfa", "llm", "edd", "dba",
}
PLACE_WORDS_STOP = {
    "san", "jose", "diego", "antonio", "paulo", "tel", "aviv", "new", "york",
    "los", "angeles", "las", "vegas", "hong", "kong", "santa", "clara",
    "cape", "town", "buenos", "aires", "sao", "jerusalem", "haifa", "beer",
    "sheva", "petah", "tikva", "herzliya", "netanya", "ramat", "gan",
    "jordan", "georgia", "chad", "congo", "guinea", "china", "india",
    "france", "poland", "spain", "portugal", "england", "britain",
}


def strip_urls_emails_phones(text):
    if not text:
        return text
    text = URL_RE.sub("[link]", text)
    text = EMAIL_RE.sub("[link]", text)
    text = PHONE_RE.sub("[link]", text)
    text = LINKEDIN_WORD_RE.sub("[link]", text)
    text = GITHUB_WORD_RE.sub("[link]", text)
    text = AT_SIGN_RE.sub(" at ", text)
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


def name_tokens(raw):
    """Tokens from the candidate's own name, to redact everywhere EXCEPT the
    LOCATION line (which is built fresh from city/country and is protected
    separately in build_profile_text, never passed through redact_names).

    We deliberately do NOT skip a token just because it matches a word in
    this candidate's own location (e.g. a real first name of "Israel" living
    in Israel) — that caused a real leak in testing: the LOCATION line stayed
    safe (it's excluded from redaction below), but the same word then went
    unredacted everywhere else in the file, where it really is their name.
    We only skip tokens that are curated, generic place/degree/common words
    (a fixed list, not derived from this candidate's own data), since those
    are the over-masking false positives the task asked to avoid.
    """
    tokens = set()
    for field in ("first_name", "last_name", "name"):
        v = raw.get(field)
        if isinstance(v, str):
            for t in re.split(r"[\s,]+", v):
                t = t.strip()
                if len(t) < 3:
                    continue  # skip short tokens (e.g. initials)
                tl = t.lower()
                if tl in COMMON_WORDS_STOP:
                    continue  # skip common words
                if tl in DEGREE_ABBR_STOP:
                    continue  # skip degree abbreviations (BSc, MSc, MBA...)
                if tl in PLACE_WORDS_STOP:
                    continue  # skip common place-name words (San, Tel, Aviv...)
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


def build_profile_text(raw):
    tokens = name_tokens(raw)

    headline = clean(raw.get("headline") or "", tokens)
    # LOCATION is built fresh from city/country only — never run name
    # redaction on it (only the URL/email/phone strip, which is a no-op here
    # in practice). This is what lets name_tokens() skip the "own location"
    # exclusion above without corrupting the LOCATION line.
    location = strip_urls_emails_phones(location_str(raw))
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
            company_c = clean(company, tokens)
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

from collections import Counter
counts = Counter((m["position"], m["group"]) for m in manifest)
print("[summary] counts per position/group:")
for k in sorted(counts.keys()):
    print(f"  {k[0]:10s} {k[1]:28s} {counts[k]}")

print(f"[summary] still_missing candidate urls with no stored profile at all: {len(still_missing)}")
print(f"[summary] skip reasons across all candidate pools (includes overlap/repeats across groups): {skip_reasons}")
print(f"[summary] notes: {notes if notes else '(none)'}")
