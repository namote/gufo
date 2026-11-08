"""CPU input observer contract: no H mutation, cloning, abort, restoration."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from exllamav3.modules import Linear
from capture_inputs import capture_inputs,InputsCaptured
root=Path('/output/capture-quarter-holdout');checks=[]
old=Linear.capture_H;inputs={};params={'capture':{}}
x=torch.arange(24,dtype=torch.float16).reshape(2,3,4)
with capture_inputs({'a','z'},'z',inputs):
    Linear.capture_H(SimpleNamespace(key='ignored'),x,params)
    Linear.capture_H(SimpleNamespace(key='a'),x,params)
    try:Linear.capture_H(SimpleNamespace(key='z'),x,params)
    except InputsCaptured:pass
    else:raise AssertionError('No terminal abort')
assert Linear.capture_H is old and params=={'capture':{}}
assert all(torch.equal(v.view(torch.uint8),x.view(torch.uint8)) and v.data_ptr()!=x.data_ptr() for v in inputs.values())
checks+=['no_H_or_pending_mutation','cloned_input_bytes_and_shape','terminal_abort_and_restore']
try:
    with capture_inputs({'a','z'},'z',{}):
        Linear.capture_H(SimpleNamespace(key='z'),x,params)
except AssertionError:pass
else:raise AssertionError('Missing earlier input accepted')
assert Linear.capture_H is old;checks.append('missing_input_fails_and_restores')
try:
    with capture_inputs({'z'},'z',{}):raise ValueError('unrelated')
except ValueError:pass
else:raise AssertionError('Error swallowed')
assert Linear.capture_H is old;checks.append('unrelated_failure_propagates')
identity={str(Path('/task')/n):hashlib.sha256((Path('/task')/n).read_bytes()).hexdigest() for n in ('capture_holdout_probe.py','capture_holdout_job.py','capture_inputs.py','test_capture_inputs.py','prepare_quarter_holdout.py')}
for n in ('DATA.json','TEXTS.json','input-ids.safetensors'):identity[str(root/n)]=hashlib.sha256((root/n).read_bytes()).hexdigest()
(root/'CPU.json').write_text(json.dumps(dict(status='pass',gpu_used=False,checks=checks,identity=identity),indent=2));print(checks)
