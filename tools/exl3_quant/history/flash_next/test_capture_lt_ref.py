"""CPU contracts for row0 intervention and stop-before-row1."""
import ast,copy,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
p=Path(__file__).with_name('capture_lt_ref_probe.py')
tree=ast.parse(p.read_text());names={'focused_flush','StopAfterRow0','CaptureFinished','compare','h_difference','cpu_tree','tensor_digest','exact'}
nodes=[n for n in tree.body if getattr(n,'name',None) in names]
source=torch.tensor([[[1.,2.]]]);rng=torch.random.get_rng_state();ids=torch.tensor([3])
trace=[];saved={};outputs={};report={};pending_x=torch.tensor([[1.,2.]])
class Module:
    def prepare_for_device(self,x,p):
        assert 'capture' not in p and 'activate_all_experts' not in p
        return x.clone()
    def forward(self,x,p):trace.append('normal');return x+1
class Bar:
    def update(self,i):trace.append(('update',i))
def original(p):trace.append('original');assert len(p.pop('_cap_pending'))==516
def lt(p):trace.append('lt');assert len(p.pop('_cap_pending'))==516
ns=dict(torch=torch,copy=copy,hashlib=hashlib,sys=sys,flush_calls=0,outputs=outputs,report=report,
    case_cpu_rng=rng,case_gpu_rng=rng,original_flush=original,lt_flush=lt,
    native=SimpleNamespace(close=lambda:trace.append('close'),cache=SimpleNamespace(stats={})),
    capture_fast=SimpleNamespace(_active=SimpleNamespace(get=lambda:None)),
    case_dir=Path('/unused'),FAILED=Path('/unused'),original_progress=Bar,
    input_hashes=[hashlib.sha256(source.reshape(-1).view(torch.uint8).numpy()).hexdigest()],
    control={'ref_states':{0:source+1}},write=lambda n,v:saved.update({n:v}))
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
exec('''def fixture():
    i=0
    module=Module()
    state=[source]
    quant_preserves=[{}]
    original_input_ids=[ids.clone()]
    params={'_cap_pending':[(None,pending_x)]*516}
    focused_flush(params)
    ref_states={0:source+1}
    StopAfterRow0().update(1)
    raise AssertionError('row1 must not execute')
''',dict(ns,Module=Module,source=source,ids=ids,pending_x=pending_x),ns)
f=ns['fixture'];f.__globals__['main_body']=f
# focused_flush uses its own execution namespace.
ns['main_body']=f
with patch.object(torch.cuda,'get_rng_state',lambda device:rng),patch.object(torch.cuda,'set_rng_state',lambda *args:None),patch.object(torch,'save',lambda *args:None),patch.object(torch,'load',lambda *args,**kwargs:{'ref_states':{0:source+1}}):
    try:f()
    except ns['CaptureFinished']:pass
    else:raise AssertionError('did not stop')
assert trace==['normal','normal','original','normal','normal','lt','normal','normal','close','normal','normal'],trace
assert all(v['exact'] for v in report['within_process'].values())
assert report['cm_normal_vs_saved']['exact'] and report['source_unchanged']
assert torch.equal(ids,torch.tensor([3])) and torch.equal(source,torch.tensor([[[1.,2.]]]))
result=dict(status='pass',gpu_used=False,checks=['normal_original_lt_close_order','independent_normal_params','pending_reuse','input_preserve_ID_RNG_unchanged','row1_quantization_unreachable'],probe_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
Path(sys.argv[1]).write_text(json.dumps(result,indent=2));print(json.dumps(result))
