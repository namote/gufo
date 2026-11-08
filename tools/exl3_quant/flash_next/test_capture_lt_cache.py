"""CPU resource-lifetime contract tests; no HIP calls."""
import hashlib
import json
from pathlib import Path
from capture_lt import BoundedPlans

events=[];live=set();busy=set();peak=0;next_id=0
def sync():events.append('sync');busy.clear()
def destroy(value):
    assert 'sync' in events and value in live and value not in busy
    live.remove(value);events.append(('destroy',value))
def create():
    global next_id,peak
    next_id+=1;live.add(next_id);peak=max(peak,len(live));events.append(('create',next_id));return next_id
cache=BoundedPlans(8,sync,destroy)
for i in range(100):
    value=cache.get(i,create);busy.add(value);assert len(live)<=8
    assert cache.get(i,create)==value
assert peak==8 and cache.stats['evictions']==92 and cache.stats['hits']==100
cache.get(92,create)
cache.get(100,create)
assert 92 in cache.plans and 93 not in cache.plans,'LRU hit must refresh lifetime'
before=len(live)
def failure():raise ValueError('intentional allocation failure')
try:cache.get(101,failure)
except ValueError:pass
else:raise AssertionError('Creation failure hidden')
assert len(live)==before-1 and 101 not in cache.plans
cache.close();assert not live
previous=events.copy();cache.close();assert events==previous
try:cache.get(0,create)
except AssertionError:pass
else:raise AssertionError('Closed cache reused')
for i,event in enumerate(events):
    if isinstance(event,tuple) and event[0]=='destroy':
        assert 'sync' in events[:i]
result=dict(status='pass',gpu_used=False,checks=['100shape_bound','hits_reuse','LRU_refresh',
    'failure_not_cached','destroy_before_next_allocate','synchronize_before_destroy','close_releases_all',
    'close_idempotent','closed_rejected'],peak_plans=peak,workspace_limit_bytes=8*32*1024*1024,
    helper_sha256=hashlib.sha256(Path(__file__).with_name('capture_lt.py').read_bytes()).hexdigest())
p=Path('/output/capture-lt-pipeline/CACHE-CPU.json');p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps(result,indent=2));print(json.dumps(result))
