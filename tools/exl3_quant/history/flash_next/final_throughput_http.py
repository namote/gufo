"""Four representative cold prompts, greedy/sampled512, C1/C2 timing.

Uses the immutable smoke SSE client. Sampled outputs are not an equality oracle.
Complete decode_ms includes the server's draft/verify/rollback DecodeStep path;
request wall and concurrent group wall are retained independently.
"""
from concurrent.futures import ThreadPoolExecutor
import math,time
from final_validation_http import Client,PROMPTS,save,finite


def timing(row):
    u=row['usage'];g=u['gufo'];n=u['completion_tokens']
    assert 0<n<=512 and g['decode_ms']>0 and g['prefill_ms']>=0 and g['queue_ms']>=0
    assert isinstance(u['draft_tokens'],int) and 0<=u['draft_tokens_accepted']<=u['draft_tokens']
    result=dict(completion_tokens=n,decode_ms=g['decode_ms'],prefill_ms=g['prefill_ms'],queue_ms=g['queue_ms'],
        ttft_seconds=row['ttft_seconds'],request_wall_seconds=row['request_wall_seconds'],
        full_decode_tokens_per_second=n/(g['decode_ms']/1000),
        emitted_tokens_per_request_wall=n/row['request_wall_seconds'],draft_tokens=u['draft_tokens'],
        draft_accepted=u['draft_tokens_accepted'],requested_logical_concurrency=g['requested_logical_concurrency'],
        physical_execution_width=g['physical_execution_width'])
    finite(result);return result


def run_suite(url,out,concurrency):
    assert concurrency in (1,2)
    c=Client(url,out,timeout=300)
    models=c.get('/v1/models');save(c.out/'models.json',models)
    assert models['data'] and 'text' in models['data'][0]['architecture']['input_modalities']
    # Retained16-token warmup is excluded from the measured groups.
    c.chat('warmup',[{'role':'user','content':'Count from one to ten.'}],max_tokens=16)
    groups={}
    for sampled in (False,True):
        name='sampled' if sampled else 'greedy'
        def run(label):
            row=c.chat(name+'-'+label,[{'role':'user','content':PROMPTS[label]}],max_tokens=512,sampled=sampled,cache=False)
            return label,row
        start=time.monotonic()
        with ThreadPoolExecutor(max_workers=concurrency) as workers:rows=dict(workers.map(run,PROMPTS))
        wall=time.monotonic()-start
        measurements={k:timing(v) for k,v in rows.items()}
        groups[name]=dict(rows={k:{x:v for x,v in row.items() if x!='events'} for k,row in rows.items()},
            timings=measurements,group_wall_seconds=wall,
            aggregate_emitted_tokens_per_group_wall=sum(v['completion_tokens'] for v in measurements.values())/wall)
        save(c.out/(name+'-SUMMARY.json'),groups[name])
    result=dict(status='passed',concurrency=concurrency,max_tokens=512,cache_prompt=False,prompts=list(PROMPTS),groups=groups,
        timing_scope='Per-request full decode includes draft/verify/rollback, while request/group wall also includes prefill/queue/HTTP; no independent draft-only speed claim',
        sampled_equivalence_claim=False)
    save(c.out/'SUMMARY.json',result);return result
