"""Bounded real HC-sidecar smoke: C2 branches, cache reuse, MTP and vision.

Runs only the supplied binary/model in an owned child process. The surrounding
GPU window owns device admission and restores the normal user service.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request

IMAGE='iVBORw0KGgoAAAANSUhEUgAAAEAAAAAgCAIAAAAt/+nTAAAAOElEQVR4nO3PsQkAAAzDsPz/dHpFliLwbFCaTBvvu94DAAAAAAAAAAAAAAAAAAAAAAAAAAAAPAQcOkv4asq/aw0AAAAASUVORK5CYII='


def run(binary, model, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    url='http://127.0.0.1:18741'
    command=[binary,'serve','llm','--model',model,'--served-model-name','hc-test','--host','127.0.0.1','--port','18741',
             '--sessions','2','--context','4096','--kv-cache-type','fp16','--kv-pool-bytes','268435456',
             '--cache-ram-bytes','2147483648','--think','off','--speculative','mtp','--mtp-model',model,
             '--draft-tokens','7','--temperature','0','--top-p','1','--top-k','0']
    (output/'ARGV.json').write_text(json.dumps(command,indent=2)+'\n')
    def request(path,payload=None):
        body=None if payload is None else json.dumps(payload).encode()
        req=urllib.request.Request(url+path,data=body,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=180) as response:return json.load(response)
    def chat(prompt,max_tokens=192):
        content=prompt if isinstance(prompt,list) else str(prompt)
        result=request('/v1/chat/completions',{'model':'hc-test','messages':[{'role':'user','content':content}],
            'temperature':0,'max_tokens':max_tokens})
        message=result['choices'][0]['message']['content']
        if not isinstance(message,str) or not message.strip() or result['usage']['completion_tokens']<=0:
            raise RuntimeError('empty response')
        return result
    report={'status':'running','cases':{}}
    def save(): (output/'RESULT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    save()
    with (output/'server.log').open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(180):
                if child.poll() is not None:raise RuntimeError('owned server exited before readiness')
                try:
                    models=request('/v1/models')
                    if models.get('data'):break
                except (urllib.error.URLError,TimeoutError):pass
                time.sleep(1)
            else:raise RuntimeError('server readiness timeout')
            prefix='Background notes: '+('alpha beta gamma delta. '*96)+'\n'
            prompts=[prefix+'Compute 17 + 25. Reply with only the number.',
                     prefix+'Compute 6 * 6. Reply with only the number.']
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:answers=list(pool.map(chat,prompts))
            for i,(answer,expected) in enumerate(zip(answers,['42','36'])):
                report['cases']['branch-'+str(i)]=answer;save()
                if re.findall(r'\d+',answer['choices'][0]['message']['content'])!=[expected]:
                    raise RuntimeError('arithmetic branch mismatch')
            repeated=chat(prompts[0]);report['cases']['repeat']=repeated;save()
            usage=repeated['usage']
            cached=usage.get('cached_tokens',usage.get('prompt_tokens_details',{}).get('cached_tokens',0))
            if cached<64:raise RuntimeError('long shared prefix was not reused')
            counted=chat('Count from 1 to 40 inclusive. Output only the numbers separated by spaces.')
            report['cases']['mtp']=counted;save()
            if re.findall(r'\d+',counted['choices'][0]['message']['content'])!=[str(i) for i in range(1,41)]:
                raise RuntimeError('counting smoke mismatch')
            if counted['usage'].get('draft_tokens',0)<=0 or counted['usage'].get('draft_tokens_accepted',0)<=0:
                raise RuntimeError('MTP did not actually propose and accept drafts')
            visual=chat([{'type':'image_url','image_url':{'url':'data:image/png;base64,'+IMAGE}},
                {'type':'text','text':'The image has two colored halves. Name the color of the left half and the right half, in that order, in two words.'}])
            report['cases']['vision']=visual;save()
            if not re.search(r'red\W+blue',visual['choices'][0]['message']['content'],re.I):
                raise RuntimeError('vision color smoke mismatch')
            report['status']='passed_pending_review';save()
        except BaseException as error:
            report.update(status='failed',error=repr(error));save();raise
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=30)
            report['server_exit_code']=child.returncode;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','model','output'):p.add_argument('--'+name,required=True)
    run(**vars(p.parse_args()))
