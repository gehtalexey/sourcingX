from collections import Counter
from common import *
from db import get_supabase_client, compute_jd_hash, update_profile_screening_batch
from normalizers import normalize_linkedin_url
from screening_models import DEFAULT_SCREEN_MODEL
db = get_supabase_client()
cache = json.load(open(os.path.join(HERE, 'results_cache.json'), encoding='utf-8'))
set1, manifest = load_sets(db)
get = lambda key, u: cache.get(key + '|' + normalize_linkedin_url(u))

def row(r):  # dashboard.py ~10513 row shape
    return {'linkedin_url': r.get('linkedin_url'), 'score': r.get('score'), 'fit_level': r.get('fit'),
            'summary': r.get('summary'), 'reasoning': r.get('reasoning'),
            'notes': ('Needs verification: ' + '; '.join(r.get('needs_verification'))
                      if r.get('decision') == 'NEEDS VERIFICATION' and r.get('needs_verification') else None)}

def ok(r): return r and r.get('fit') not in ('Error', 'Skipped')

stats = {}
for key, brief in (('dwelly', DWELLY), ('owner', OWNER)):
    jd = render_jd(brief)
    rows = [row(v) for k, v in cache.items() if k.startswith(key + '|') and ok(v)]
    stats[key] = update_profile_screening_batch(db, rows, jd_hash=compute_jd_hash(jd), jd_title=jd[:200], ai_model=DEFAULT_SCREEN_MODEL)
    stats[key]['jd_hash'] = compute_jd_hash(jd)[:12]
print('saved', stats)

s1 = []
for u in set1:
    r = get('dwelly', u) or {}
    s1.append({'linkedin_url': u, 'decision': r.get('decision'), 'score': r.get('score'), 'fit': r.get('fit'),
               'needs_verification': r.get('needs_verification') or [], 'reasoning': r.get('reasoning'),
               'must_have_verdicts': r.get('must_have_verdicts'), 'exclusion_verdicts': r.get('exclusion_verdicts'),
               'hard_filter_failed': r.get('hard_filter_failed')})
c1 = Counter(x['decision'] for x in s1)
json.dump({'counts': dict(c1), 'candidates': s1}, open(os.path.join(HERE, 'set1_dwelly20.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('set1', dict(c1))
b = []
for m in manifest:
    r = get(m['position'], m['linkedin_url']) or {}
    b.append({'id': m['id'], 'position': m['position'], 'decision': r.get('decision'), 'score': r.get('score'), 'fit': r.get('fit'),
              'needs_verification': r.get('needs_verification') or [], 'reasoning': r.get('reasoning')})
json.dump(b, open(os.path.join(HERE, 'blind_sourcingx.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('blind', {p: dict(Counter(x['decision'] for x in b if x['position'] == p)) for p in ('dwelly', 'owner')})
