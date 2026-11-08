"""Frozen throughput groups plus K4V4 identity/cache checks outside timed groups."""
from pathlib import Path
from final_throughput_http import run_suite as measured_suite
from final_validation_http import Client,save,signature,cached
KV='kvarn-k4v4'

def verify_kv(row):
    assert row['usage']['gufo']['component_cache']['kv_cache_type']==KV,'Actual HTTP KV differs'

def run_suite(url,out,concurrency):
    result=measured_suite(url,out,concurrency)
    for group in result['groups'].values():
        for row in group['rows'].values():verify_kv(row)
    c=Client(url,Path(out)/'functionality',timeout=180)
    prefix=[{'role':'system','content':'Follow the last user instruction.\n'+('Record: Alice owns the red bicycle; Bob owns the blue bicycle.\n'*90)}]
    prompt=prefix+[{'role':'user','content':'Reply with only ALPHA.'}]
    cold=c.chat('cold',prompt,cache=True,max_tokens=16)
    repeat=c.chat('repeat',prompt,cache=True,max_tokens=16)
    branch=c.chat('branch',prefix+[{'role':'user','content':'Reply with only BETA.'}],cache=True,max_tokens=16)
    for row in (cold,repeat,branch):verify_kv(row)
    assert signature(cold)==signature(repeat) and cached(repeat['usage'])>0 and cached(branch['usage'])>=128
    assert 'ALPHA' in cold['text'] and 'BETA' in branch['text']
    result['kv_cache_type']=KV
    result['functionality']=dict(status='passed',outside_timed_groups=True,repeat_cached_tokens=cached(repeat['usage']),branch_cached_tokens=cached(branch['usage']))
    save(Path(out)/'KV4-SUMMARY.json',result)
    return result
