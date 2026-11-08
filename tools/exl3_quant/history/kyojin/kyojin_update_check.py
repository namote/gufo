"""Short paired AR/MTP check on the updated resident C1 server."""
import json,time,urllib.request
from pathlib import Path
ROOT=Path('/home/m2on/projects/ai.tools')
OUT=ROOT/'tmp/test-results/kyojin-update-20261103'
PROMPTS=json.loads((ROOT/'test/experiments/gufo_upstream_port_20261023/KYOJIN-LOCAL-PROMPTS.json').read_text())
BASE='http://127.0.0.1:18740'
def main():
 report={'settings':{'context':262144,'sessions':1,'kv':'fp16','temperature':0,'thinking':False,'output_tokens':256,'version':'1.8.6'},'runs':[],'pairs':[],'old_baseline_repeated':False}
 def save(): (OUT/'PAIRED.json').write_text(json.dumps(report,indent=2)+'\n')
 save()
 for spec in [False,True]:
  for item in PROMPTS:
   body={'model':'gufo-flash-next','messages':[{'role':'user','content':item['text']}],'max_tokens':256,'temperature':0,'top_k':1,'top_p':1,'seed':314159,'enable_thinking':False,'cache_prompt':False,'speculative':spec,'return_token_ids':True,'loop_guard':False}
   req=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   started=time.monotonic()
   with urllib.request.urlopen(req,timeout=180) as r:data=json.load(r)
   row={'name':item['id'],'speculative':spec,'seconds':time.monotonic()-started,'request':body,'response':data}
   report['runs'].append(row);save()
   assert data['token_ids'] and data['usage']['completion_tokens']>0,data
   assert data['timings']['speculative']==spec,data['timings']
   print(item['id'],'MTP' if spec else 'AR',data['timings'],flush=True)
 for item in PROMPTS:
  a,b=[r['response'] for r in report['runs'] if r['name']==item['id']]
  x,y=a['token_ids'],b['token_ids']
  first=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),None)
  if first is None and len(x)!=len(y):first=min(len(x),len(y))
  report['pairs'].append({'id':item['id'],'equal_ids':x==y,'equal_text':a['choices'][0]['message']==b['choices'][0]['message'],'first_divergence':first,'AR_tokens':len(x),'MTP_tokens':len(y),'accepted':b['timings'].get('accepted_draft_tokens',0),'rejected':b['timings'].get('rejected_draft_tokens',0)})
 report['actual_drafts']=sum(p['accepted']+p['rejected'] for p in report['pairs'])
 assert report['actual_drafts']>0
 report['status']='MATCH' if all(p['equal_ids'] for p in report['pairs']) else 'OUTPUT_DIFFERENCE'
 save();print(json.dumps({'status':report['status'],'pairs':report['pairs']}),flush=True)
if __name__=='__main__': main()
