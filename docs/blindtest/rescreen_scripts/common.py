import sys, os, json, hashlib
REPO = r"C:\Users\gehta\projects\sourcingX"
sys.path.insert(0, REPO)
os.chdir(REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
BLIND = os.path.join(os.path.dirname(HERE), "blind")

DWELLY = {
    "role_context": "Applied AI Engineer at Dwelly (AI-first UK lettings and property management platform, $170M Series B), fully remote in the UK or Europe on UK hours, building production agentic systems in TypeScript/Node and Python",
    "must_haves": [
        "At least 3 years of hands-on software engineering, backend (TypeScript/Node.js or Python)",
        "Has built and shipped AI or agentic systems that ran in production (tool use, orchestration, structured outputs, evals, cost/latency/reliability), not just demos, courses or side projects",
        "Based in the UK or Europe",
        "Degree in Computer Science or a related technical field, or a strong engineering track record with another science degree",
    ],
    "nice_to_haves": [
        "Came from an AI-native company or a product startup",
        "Technical founder history (past founder now employed as an engineer)",
        "Based in London, Spain, Portugal or Poland",
        "Uses coding agents in their own daily engineering work",
        "PostgreSQL, LangGraph / LangChain, LLM eval tooling",
    ],
    "exclusions": [
        "Data scientist, ML researcher, academic or research-heavy profile, or an \"Applied AI\" title that is really classic ML / model training",
        "Career mostly at IT consultancies, outsourcing or software houses",
        "Own company or solo freelancing is the only current job",
        "Currently at Dwelly",
    ],
}
OWNER = {
    "role_context": "Senior / Staff AI Agents Engineer at Owner (all-in-one platform for independent restaurants), remote in US or Canada, building production LLM agents end to end",
    "must_haves": [
        "Minimum 5 years of software engineering",
        "Has shipped production LLM or agentic features (tool calling, agent loops), not just a simple LLM API call",
        "Full-stack: works across backend and frontend / UI",
        "Based in the US or Canada",
        "Bachelor's degree in CS or a related STEM field",
        "Worked at a strong startup backed by top-tier VCs",
    ],
    "nice_to_haves": [
        "LLM eval or observability tools (Braintrust, LangSmith, custom evals)",
        "RAG / retrieval systems in production",
        "Restaurant tech, commerce or SMB tools background",
        "Open source work on agent frameworks or LLM tooling",
        "Node.js / TypeScript",
        "Big tech first, then moved to a startup",
    ],
    "exclusions": [
        "Only research or theoretical AI, nothing shipped to production",
        "Pure backend-only or frontend-only specialist",
        "Career mostly at consulting, outsourcing or low-signal companies",
    ],
}


def render_jd(b):
    # Mirrors dashboard.py ~10084 exactly
    return "\n".join(p for p in [
        f"Role: {b['role_context']}" if b['role_context'] else "",
        ("Must-haves:\n" + "\n".join(f"- {m}" for m in b['must_haves'])) if b['must_haves'] else "",
        ("Nice-to-haves:\n" + "\n".join(f"- {n}" for n in b['nice_to_haves'])) if b['nice_to_haves'] else "",
        ("Exclusions:\n" + "\n".join(f"- {e}" for e in b['exclusions'])) if b['exclusions'] else "",
    ] if p)


def load_sets(db):
    from normalizers import normalize_linkedin_url
    rows = db.select('pipeline_candidates', 'linkedin_url',
                     {'position_id': 'eq.dwelly-ai-applied-eng-eu', 'smartlead_pushed': 'is.true'}, limit=5000)
    urls = sorted({r['linkedin_url'] for r in rows if r.get('linkedin_url')},
                  key=lambda u: hashlib.md5((u + '20260924').encode()).hexdigest())
    set1 = urls[:20]
    manifest = json.load(open(os.path.join(BLIND, 'manifest.json'), encoding='utf-8'))
    return set1, manifest


def fetch_profile(db, url):
    from db import get_profile
    from normalizers import normalize_linkedin_url
    p = get_profile(db, url) or get_profile(db, normalize_linkedin_url(url))
    return p
