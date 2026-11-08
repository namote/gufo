"""CPU hook tests; actual observer arithmetic, mocked devices and model classes."""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import types
from contextlib import contextmanager
import torch

TASK=Path(__file__).parent
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=True)
linear_source=Path(sys.argv[2])
checks=[]
class FakeLinear:
    def __init__(self,key):self.key=key
    def capture_H(self,x,params):params.setdefault('seen',[]).append(self.key)
ns=dict(Linear=FakeLinear,contextmanager=contextmanager)
parsed=ast.parse((TASK/'capture_selected.py').read_text())
exec(compile(ast.Module(body=[n for n in parsed.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))],type_ignores=[]),'selected-hook','exec'),ns)
original=FakeLinear.capture_H
params={}
with ns['selected_capture']({'a','z'},'z'):
    FakeLinear('skip').capture_H(None,params);FakeLinear('a').capture_H(None,params)
    try:FakeLinear('z').capture_H(None,params)
    except ns['PrefixCaptured']:pass
    else:raise AssertionError('Did not abort')
assert params['seen']==['a','z'] and FakeLinear.capture_H is original
checks.append('filter_terminal_abort_restores')
try:
    with ns['selected_capture']({'a','z'},'z'):
        FakeLinear('z').capture_H(None,{})
except AssertionError:pass
else:raise AssertionError('Missing projection accepted')
assert FakeLinear.capture_H is original
checks.append('missing_prefix_fails_and_restores')
try:
    with ns['selected_capture']({'z'},'z'):
        raise ValueError('unrelated')
except ValueError:pass
else:raise AssertionError('Exception swallowed')
assert FakeLinear.capture_H is original
checks.append('unrelated_error_propagates')

source=linear_source.read_text();tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='flush_capture_H')
original_ns=dict(torch=torch,_CAP_PROFILE=False,_time=time,FLUSH_STAT={})
exec(compile(ast.Module(body=[node],type_ignores=[]),str(linear_source),'exec'),original_ns)
original_flush=original_ns['flush_capture_H']
class Store:
    index=None
    def __getitem__(self,index):self.index=index;return torch.zeros(1,3,5)
class Block:
    def prepare_for_device(self,x,params):return x
mod=types.ModuleType('exllamav3.modules.transformer');mod.TransformerBlock=Block
sys.modules['exllamav3.modules.transformer']=mod
node=next(n for n in ast.parse((TASK/'capture_profile_probe.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='instrument_original_flush')
class Event:
    count=0
    def __init__(self,**kwargs):self.index=Event.count;Event.count+=1
    def record(self):pass
    def elapsed_time(self,other):return float(other.index-self.index)
class Profile:
    starts=stops=0
    def __init__(self,**kwargs):pass
    def start(self):Profile.starts+=1
    def stop(self):Profile.stops+=1
    def key_averages(self):return []
    def export_chrome_trace(self,path):Path(path).write_text('{}')
ns=dict(torch=torch,inspect=inspect,hashlib=hashlib,time=time,DiskRows=Store,OUT=OUT,
        capture_pack=types.SimpleNamespace(PINNED_FLUSH_SHA256=hashlib.sha256(inspect.getsource(original_flush).encode()).hexdigest()),original_flush=original_flush)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(TASK/'capture_profile_probe.py'),'exec'),ns)
old_event,old_profile=torch.cuda.Event,torch.profiler.profile
old_read,old_prepare=Store.__getitem__,Block.prepare_for_device
try:
    torch.cuda.Event=Event;torch.profiler.profile=Profile;torch.set_num_threads(2)
    flush,report=ns['instrument_original_flush']()
    store=Store();block=Block()
    for row in range(64):
        x=store[row];block.prepare_for_device(x,dict(capture={}))
        params=dict(_cap_pending=[(dict(H=torch.zeros(5,5)),torch.arange(15,dtype=torch.float16).reshape(3,5)/8) for _ in range(65 if row==0 else 1)])
        params['_cap_pending'][0][1][0,0]=float('inf')
        a,b=copy.deepcopy(params),copy.deepcopy(params);aa,bb=a['_cap_pending'],b['_cap_pending']
        original_flush(a);flush(b)
        for (ha,xa),(hb,xb) in zip(aa,bb):
            assert torch.equal(ha['H'].view(torch.uint8),hb['H'].view(torch.uint8))
            assert torch.equal(ha['count_dev'],hb['count_dev'])
            assert torch.equal(xa.view(torch.uint8),xb.view(torch.uint8))
        block.prepare_for_device(x,{})
    r=report();assert len(r['rows'])==len(r['reads'])==64
    assert sum(x['trace_instrumented'] for x in r['rows'])==1 and r['rows'][0]['chunks']==2
    assert Profile.starts==Profile.stops==1
    flush.restore();flush.restore()
    assert Store.__getitem__ is old_read and Block.prepare_for_device is old_prepare
    checks+=['all64_profile_exact_CPU_arithmetic','nonfinite_and_chunk65','reference_prepare_not_counted','one_trace_only','restore_idempotent']
finally:torch.cuda.Event=old_event;torch.profiler.profile=old_profile
# Test strict byte mismatch diagnostics, including signed zero and metadata.
node=next(n for n in ast.parse((TASK/'capture_profile_probe.py').read_text()).body
          if isinstance(n,ast.FunctionDef) and n.name=='numeric_difference')
ns=dict(torch=torch);exec(compile(ast.Module(body=[node],type_ignores=[]),'numeric-difference','exec'),ns)
f=ns['numeric_difference']
assert f({'H':torch.ones(2,2)},{'H':torch.ones(2,2)})['exact']
r=f({'H':torch.ones(2,2)},{'H':torch.zeros(2,2)})
assert not r['exact'] and r['mismatches'][0]['changed_values']==4 and r['mismatches'][0]['relative_l2']==1
assert not f(torch.tensor([0.]),torch.tensor([-0.]))['exact']
assert not f(torch.ones(1),torch.ones(2))['exact']
checks += ['strict_difference_identical_and_changed','signed_zero_remains_byte_mismatch','difference_metadata_fail_closed']
names=('capture_profile_probe.py','capture_selected.py','capture_quarter_probe.py','capture_diagnostic_probe.py','capture_diagnostic_job.py','test_capture_diagnostic.py')
result=dict(status='pass',gpu_used=False,checks=checks,torch_version=torch.__version__,
 identity={str(TASK/n):hashlib.sha256((TASK/n).read_bytes()).hexdigest() for n in names})
# Container paths translated to actual task identities by the host after verification.
(OUT/'CPU.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
