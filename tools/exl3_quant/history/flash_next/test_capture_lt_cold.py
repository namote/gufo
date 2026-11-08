"""Ensure cold probe observes first cm reference before adding any work."""
import ast,copy,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
p=Path(__file__).with_name('capture_lt_cold_probe.py');tree=ast.parse(p.read_text())
names={'StopAfterRow0','CaptureFinished','compare','h_difference','cpu_tree','tensor_digest','exact'}
nodes=[n for n in tree.body if getattr(n,'name',None) in names]
source=torch.tensor([[[1.,2.]]]);rng=torch.random.get_rng_state();ids=torch.tensor([3]);trace=[];outputs={};report={}
class Module:
    def prepare_for_device(self,x,p):
        assert 'capture' not in p and 'activate_all_experts' not in p
        return x.clone()
    def forward(self,x,p):trace.append('normal');return x+1
class Bar:
    def update(self,i):trace.append(('update',i))
ns=dict(torch=torch,copy=copy,hashlib=hashlib,sys=sys,outputs=outputs,report=report,
    case_cpu_rng=rng,case_gpu_rng=rng,lengths=[3217],
    native=SimpleNamespace(close=lambda:trace.append('close'),cache=SimpleNamespace(stats={})),
    capture_fast=SimpleNamespace(_active=SimpleNamespace(get=lambda:None)),
    case_dir=Path('/unused'),FAILED=Path('/unused'),original_progress=Bar,base_bar=Bar,
    input_hashes=[hashlib.sha256(source.reshape(-1).view(torch.uint8).numpy()).hexdigest()],
    control={'ref_states':{0:source+1}},write=lambda *args:None)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
exec('''def fixture():
    module=Module();state=[source];quant_preserves=[{}];original_input_ids=[ids.clone()]
    capture_H={str(i):{'count_dev':torch.tensor(3217)} for i in range(516)}
    ref_states={0:source+1}
    StopAfterRow0().update(0)
    StopAfterRow0().update(1)
    raise AssertionError('row1 must not execute')
''',dict(ns,Module=Module,source=source,ids=ids),ns)
f=ns['fixture'];f.__globals__['main_body']=f;ns['main_body']=f
with patch.object(torch.cuda,'get_rng_state',lambda device:rng),patch.object(torch,'save',lambda v,path:trace.append(('save',path.name))),patch.object(torch,'load',lambda *args,**kwargs:{'ref_states':{0:source+1}}):
    try:f()
    except ns['CaptureFinished']:pass
    else:raise AssertionError('no stop')
assert trace[:3]==[('update',0),('save','cm_first.pt'),'normal'],trace
assert trace.count('normal')==4 and trace.count('close')==1
assert all(v['exact'] for v in report['saved_control'].values())
assert report['source_preserve_ID_RNG_unchanged']
# No flush wrapper or intervention is inserted before actual first reference.
assert 'cm._flush_capture_H=lt_flush' in p.read_text()
r=dict(status='pass',gpu_used=False,checks=['first_reference_saved_before_intervention','cold_flush_direct','row1_quantization_unreachable','repeated_normal_inputs_preserved'],probe_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
Path(sys.argv[1]).write_text(json.dumps(r,indent=2));print(json.dumps(r))
