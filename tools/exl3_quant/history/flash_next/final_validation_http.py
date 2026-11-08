"""Isolated production HTTP checks. Retains every request/SSE response and timing.

HTTP exposes text/count, not emitted token IDs: exact IDs are checked separately
by the production bench's generated-token SHA. No sampled off/on equality claim.
"""
import base64,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request

PROMPTS={
 'ko':'양자화된 언어 모델에서 검증이 필요한 이유를 한국어로 세 문장으로 설명해 주세요.',
 'en':'Explain why immutable backups are useful, in three concise sentences.',
 'code':'Write a Python function that returns the unique elements of a list while preserving order. Include two small assert examples.',
 'tool':'A read_file tool returned {"remaining_tasks":["test parser","save report"],"tests_passed":false}. Explain the next two actions. Do not claim tests already passed.'}


def save(path,value):
    path=Path(path);assert not path.exists(),'Retain original baseline/request receipts'
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False))


def finite(value):
    import math
    if isinstance(value,float):assert math.isfinite(value),'Nonfinite HTTP response'
    elif isinstance(value,dict):
        for x in value.values():finite(x)
    elif isinstance(value,list):
        for x in value:finite(x)


def cached(usage):
    value=usage.get('cached_tokens',usage.get('prompt_tokens_details',{}).get('cached_tokens'))
    assert isinstance(value,int) and value>=0,'API cached-token evidence unavailable'
    return value


def signature(row):
    return row['text'],row['reasoning'],row['finish'],row['usage']['completion_tokens']


class Client:
    def __init__(self,url,out,*,timeout=180):
        self.url=url;self.out=Path(out);self.out.mkdir();self.timeout=timeout

    def get(self,path):
        with urllib.request.urlopen(self.url+path,timeout=10) as r:return json.load(r)

    def chat(self,label,messages,*,max_tokens=64,cache=False,sampled=False):
        body=dict(model='gufo-final-c03',messages=messages,max_tokens=max_tokens,
                  temperature=1.0 if sampled else 0.0,top_p=.95 if sampled else 1.0,
                  top_k=20 if sampled else 0,min_p=0,repeat_penalty=1.0,
                  presence_penalty=0,frequency_penalty=0,seed=314159,cache_prompt=cache,
                  reasoning_effort='medium' if sampled else 'none',
                  chat_template_kwargs={'enable_thinking':sampled},stream=True,stream_options={'include_usage':True})
        started=time.monotonic();events=[];text='';reasoning='';usage=None;finish=None;first=None;done=False
        record={'label':label,'request':body,'started':time.time(),'events':events}
        try:
            request=urllib.request.Request(self.url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=self.timeout) as response:
                for raw in response:
                    line=raw.decode().strip()
                    if not line.startswith('data:'):continue
                    payload=line[5:].strip();now=time.monotonic()
                    events.append({'seconds':now-started,'data':payload})
                    assert now-started<self.timeout,'HTTP request total deadline'
                    if payload=='[DONE]':done=True;break
                    item=json.loads(payload);finite(item);assert 'error' not in item,item
                    if item.get('usage') is not None:usage=item['usage']
                    for choice in item.get('choices',[]):
                        delta=choice.get('delta',{})
                        chunk=delta.get('content') or '';thought=delta.get('reasoning_content') or delta.get('reasoning') or ''
                        if (chunk or thought) and first is None:first=now
                        text+=chunk;reasoning+=thought
                        if choice.get('finish_reason') is not None:finish=choice['finish_reason']
            wall=time.monotonic()-started
            assert done and usage is not None and first is not None
            assert isinstance(usage['completion_tokens'],int) and usage['completion_tokens']>0
            assert text or reasoning;assert finish in ('stop','length'),finish
            assert not cache or cached(usage)>=0
            if not cache:assert cached(usage)==0,'Cold request reused cached input'
            record.update(status='passed',text=text,reasoning=reasoning,finish=finish,usage=usage,
                ttft_seconds=first-started,request_wall_seconds=wall,
                emitted_tokens_per_request_wall=usage['completion_tokens']/wall,
                post_first_event_seconds=time.monotonic()-first,
                timing_scope='Whole request includes prefill/queue/decode; SSE first-output event is TTFT, not a kernel phase timer')
        except BaseException as error:record.update(status='failed',error=repr(error));raise
        finally:save(self.out/(label+'.json'),record)
        return record


def run_suite(url,out,image_path,phase,concurrency):
    c=Client(url,out);models=c.get('/v1/models');save(c.out/'models.json',models)
    assert models['data'] and {'text','image'}<=set(models['data'][0]['architecture']['input_modalities'])
    rows={}
    selected=('ko','en') if phase=='smoke' else tuple(PROMPTS)
    for label in selected:
        rows[label]=c.chat('greedy-'+label,[{'role':'user','content':PROMPTS[label]}],max_tokens=64 if phase=='smoke' else 256)
    if phase=='smoke':
        arithmetic=c.chat('arithmetic',[{'role':'user','content':'What is 17 plus 25? Reply with only the number.'}],max_tokens=16)
        assert arithmetic['text'].strip()=='42',arithmetic['text']
        image=base64.b64encode(Path(image_path).read_bytes()).decode()
        messages=[{'role':'user','content':[{'type':'text','text':'Name the colors of the left and right halves. Answer briefly.'},
                   {'type':'image_url','image_url':{'url':'data:image/png;base64,'+image}}]}]
        first=c.chat('image-cold',messages,cache=True,max_tokens=48)
        assert 'red' in first['text'].lower() and 'blue' in first['text'].lower()
        repeat=c.chat('image-repeat',messages,cache=True,max_tokens=48)
        assert signature(first)==signature(repeat) and cached(repeat['usage'])>0
        continuation=[*messages,{'role':'assistant','content':first['text']},{'role':'user','content':'Which color is on the left? Answer with one word.'}]
        hist=c.chat('image-history',continuation,cache=True,max_tokens=16)
        assert 'red' in hist['text'].lower()
        after=c.chat('text-after-image',[{'role':'user','content':'What is 9 plus 8? Reply with only the number.'}],max_tokens=16)
        assert after['text'].strip()=='17'
    else:
        # Sampled requests retain identical settings/seed, but not independent-stream equality.
        def sampled(label):return label,c.chat('sampled-'+label,[{'role':'user','content':PROMPTS[label]}],max_tokens=512,sampled=True)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:sampled_rows=dict(pool.map(sampled,PROMPTS))
        save(c.out/'sampled-summary.json',{k:{x:v for x,v in r.items() if x!='events'} for k,r in sampled_rows.items()})
    # One shared prefix, exact repeat, then a diverging child. No client/session ID trick.
    background=('Record: Alice owns the red bicycle; Bob owns the blue bicycle. This is background data.\n'*90)
    prefix=[{'role':'system','content':'Follow the last user instruction.\n'+background}]
    messages=prefix+[{'role':'user','content':'Reply with only ALPHA.'}]
    cold=c.chat('shared-cold',messages,cache=True,max_tokens=16)
    warm=c.chat('shared-repeat',messages,cache=True,max_tokens=16)
    branch=c.chat('shared-branch',prefix+[{'role':'user','content':'Reply with only BETA.'}],cache=True,max_tokens=16)
    assert signature(cold)==signature(warm)
    assert cached(warm['usage'])>0 and cached(branch['usage'])>=128
    assert 'ALPHA' in cold['text'] and 'BETA' in branch['text']
    with urllib.request.urlopen(url+'/metrics',timeout=10) as response:(c.out/'metrics.txt').write_bytes(response.read())
    summary={'status':'passed','phase':phase,'concurrency':concurrency,
             'greedy':{k:{x:v for x,v in r.items() if x!='events'} for k,r in rows.items()},
             'cache':{'repeat':warm['usage'],'branch':branch['usage']},
             'http_identity_scope':'text, reasoning, finish and emitted count; token IDs unavailable through HTTP'}
    save(c.out/'SUMMARY.json',summary);return summary
