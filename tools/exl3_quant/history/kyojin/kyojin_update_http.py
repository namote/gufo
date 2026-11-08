import json,urllib.request,base64,time,concurrent.futures
from pathlib import Path
root=Path('/home/m2on/projects/ai.tools/tmp/test-results/kyojin-update-20261103')
base='http://127.0.0.1:18740'
reports=[]
def ask(name,messages,answer=None,extra=None):
 body={'model':'gufo-flash-next','messages':messages,'max_tokens':96,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
 if extra:body.update(extra)
 start=time.monotonic()
 req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:data=json.load(r)
 text=data['choices'][0]['message'].get('content') or ''
 row={'name':name,'response':data,'seconds':time.monotonic()-start}
 (root/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n')
 if answer:assert answer.lower() in text.lower(),(name,text)
 reports.append(row)
 print(name,round(row['seconds'],2),text,flush=True)
 return data
with urllib.request.urlopen(base+'/health',timeout=5) as r:health=json.load(r)
assert health['source']=='kyojin' and health['vision'] and health['speculative'],health
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 list(pool.map(lambda code:ask('concurrent-'+code,[{'role':'user','content':'Return exactly '+code}],code),['ORANGE-731','VIOLET-482']))
ask('medium',[{'role':'user','content':'What is 17 times 19? Answer with only the number.'}],'323',{'max_tokens':512,'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'medium'}})
img=base64.b64encode(Path('/home/m2on/.local/state/gufo/vision-20261103/two-panels.png').read_bytes()).decode()
d=ask('vision',[{'role':'user','content':[{'type':'text','text':'Name the colors of the left and right halves. Use: Left: color; Right: color.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+img}}]}],'red')
assert 'blue' in d['choices'][0]['message']['content'].lower()
d=ask('tools',[{'role':'user','content':'Call the echo tool with text HELLO-731. Do not answer directly.'}],extra={'tools':[{'type':'function','function':{'name':'echo','description':'Echo the given text','parameters':{'type':'object','properties':{'text':{'type':'string'}},'required':['text']}}}]})
calls=d['choices'][0]['message'].get('tool_calls',[]);assert calls and calls[0]['function']['name']=='echo',d
ask('tool-result',[{'role':'user','content':'Call the echo tool with text HELLO-731. Do not answer directly.'},d['choices'][0]['message'],{'role':'tool','tool_call_id':calls[0]['id'],'content':'HELLO-731'},{'role':'user','content':'What text did the tool return? Reply with that text only.'}],'HELLO-731')
(root/'HTTP.json').write_text(json.dumps({'PASS':True,'health':health,'requests':reports},indent=2)+'\n')
print('HTTP PASS',flush=True)
