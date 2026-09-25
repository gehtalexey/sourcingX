import sys, time
from datetime import datetime, timezone
from common import *
import dashboard
from dashboard import screen_profile, _ensure_raw_dict, load_config
from db import get_supabase_client, compute_jd_hash, update_profile_screening_batch
from normalizers import normalize_linkedin_url
from screening_models import get_screen_model
from usage_tracker import UsageTracker
from openai import OpenAI

BUDGET = 1.00
MODE = sys.argv[1]  # one | all
cache_path = os.path.join(HERE, 'results_cache.json')
cache = json.load(open(cache_path, encoding='utf-8')) if os.path.exists(cache_path) else {}
start_path = os.path.join(HERE, 'run_start.txt')
if not os.path.exists(start_path):
    open(start_path, 'w').write(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f'))
START = open(start_path).read().strip()

cfg = load_config()
ai_model, ai_provider = get_screen_model(cfg)
assert ai_provider == 'openai'
client = OpenAI(api_key=cfg.get('openai_api_key'))
db = get_supabase_client()
tracker = UsageTracker(db)

def spend():
    rows = db.select('api_usage_logs', 'cost_usd,provider,request_count', {'created_at': f'gte.{START}', 'provider': 'eq.openai'}, limit=5000)
    return sum(float(r.get('cost_usd') or 0) for r in rows), len(rows), sum(int(r.get('request_count') or 1) for r in rows)

def screen(url, key, brief):
    ck = key + '|' + normalize_linkedin_url(url)
    if ck in cache:
        return cache[ck]
    p = fetch_profile(db, url)
    raw = _ensure_raw_dict(p.get('raw_data'))
    profile = {k: p.get(k) for k in ('linkedin_url', 'name', 'current_title', 'current_company', 'location', 'email', 'all_schools')}
    profile['raw_crustdata'] = raw  # as the AI Screen tab does after fetch_raw_data_for_batch; NO thin top-up
    jd = render_jd(brief)
    r = None
    for attempt in range(3):
        r = screen_profile(profile, jd, client, tracker=tracker, mode='detailed', ai_model=ai_model,
                           ai_provider=ai_provider, user_request=None, screening_brief=brief, use_flex=False)
        if r.get('fit') != 'Error' or 'rate' not in str(r.get('summary','')).lower():
            break
        time.sleep(5 * (attempt + 1))
    r['linkedin_url'] = p.get('linkedin_url') or url
    cache[ck] = r
    json.dump(cache, open(cache_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return r

set1, manifest = load_sets(db)
jobs = [(u, 'dwelly', DWELLY) for u in set1]
for m in manifest:
    jobs.append((m['linkedin_url'], m['position'], DWELLY if m['position'] == 'dwelly' else OWNER))

if MODE == 'one':
    before = spend()
    r = screen(*jobs[0])
    time.sleep(2)
    after = spend()
    per = after[0] - before[0]
    todo = len({j[1] + '|' + normalize_linkedin_url(j[0]) for j in jobs})
    print(f"model={ai_model} decision={r.get('decision')} fit={r.get('fit')}")
    print(f"one profile: log rows={after[1]-before[1]} requests={after[2]-before[2]} cost=${per:.5f}")
    print(f"unique profiles to screen={todo} projected=${per*todo:.4f}")
    sys.exit(0)

errors = []
for i, (u, key, brief) in enumerate(jobs):
    s = spend()[0]
    if s > BUDGET:
        print(f"STOP: spend ${s:.4f} passed budget"); break
    r = screen(u, key, brief)
    if r.get('fit') in ('Error', 'Skipped'):
        errors.append((i, r.get('fit'), str(r.get('summary'))[:120]))
    if i % 10 == 0:
        print(f"{i+1}/{len(jobs)} spend=${s:.4f}", flush=True)

tot = spend()
print(f"done: spend=${tot[0]:.4f} log_rows={tot[1]} requests={tot[2]} errors={errors}")
json.dump({'errors': errors, 'spend': tot}, open(os.path.join(HERE, 'run_summary.json'), 'w'))
