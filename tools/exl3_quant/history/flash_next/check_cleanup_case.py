"""CPU-only regression for observed Docker diagnostic case and auto-remove races."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import uuid
from owned_cleanup import cleanup_owned

def check(events, expected_error=False, label='owned'):
    calls=[];waits=[]
    def run(command,**kwargs):
        calls.append(command[1]);event=events.pop(0)
        if isinstance(event,Exception):raise event
        if event=='running':
            return SimpleNamespace(returncode=0,stdout=json.dumps([dict(Id='id',
                Config=dict(Labels={'gufo.quant.run':label}),State=dict(Running=True))]),stderr='')
        return SimpleNamespace(returncode=1,stdout='',stderr=event)
    child=SimpleNamespace(wait=lambda **kwargs:waits.append(True))
    result=cleanup_owned('name','owned',child,lambda:dict(KFD_contexts=[]),run=run)
    assert not events and waits==[True]
    assert bool(result['cleanup_errors'])==expected_error,result
    assert result['returned_KFD_contexts']==[]
    return calls

cases=[]
for message in ['Error: No such object: name\n','error: no such object: name\\n',
                'Error response from daemon: No such container: name',
                'ERROR: NO SUCH OBJECT: name']:
    assert check([message])==['inspect']
    cases.append(message)
missing='error: no such object: id\\n'
assert check(['running',subprocess.TimeoutExpired('stop',20),missing])==['inspect','stop','inspect']
assert check(['running',subprocess.TimeoutExpired('stop',20),'running',missing,missing])==['inspect','stop','inspect','kill','inspect']
for message in ['Cannot connect to the Docker daemon','permission denied',
                'dial unix /missing/docker.sock: no such file or directory']:
    assert check([message],expected_error=True)==['inspect']
assert check(['running'],expected_error=True,label='foreign')==['inspect']
# Exercise the actual installed Docker CLI against a uniquely absent object.
name='gufo-cleanup-probe-'+uuid.uuid4().hex
response=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=15)
assert response.returncode!=0
assert check([response.stderr])==['inspect']
result=dict(status='pass',checks=11,gpu_used=False,observed_docker_stderr=response.stderr,
            owned_cleanup_sha256=hashlib.sha256(Path(__file__).with_name('owned_cleanup.py').read_bytes()).hexdigest())
out=Path('/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104/model-job-c03/repair-003-cleanup-case')
(out/'CPU-CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
