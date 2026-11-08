"""CPU observer contracts: capture five references and reject later mutation."""
import ast,copy,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
p=Path(__file__).with_name('capture_lt_prefix_probe.py');tree=ast.parse(p.read_text())
names={'StopAfterRow0','CaptureFinished','compare','h_difference','cpu_tree','tensor_digest','exact'}
nodes=[n for n in tree.body if getattr(n,'name',None) in names]
sources=[torch.tensor([[[float(i),2.]]]) for i in range(5)];ids=[torch.tensor([i]) for i in range(5)]
rng=torch.random.get_rng_state();trace=[];outputs={};report={};lengths=[3217,7793,4655,3671,512]
class Bar:
    def update(self,i):trace.append(i)
ns=dict(torch=torch,copy=copy,hashlib=hashlib,sys=sys,outputs=outputs,report=report,
    case_cpu_rng=rng,case_gpu_rng=rng,lengths=lengths,
    native=SimpleNamespace(cache=SimpleNamespace(stats={'evictions':5,'peak_plans':8,'max_workspace_bytes':256*2**20})),
    case_dir=Path('/unused'),FAILED=Path('/unused'),base_bar=Bar,
    input_hashes=[hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy()).hexdigest() for x in sources],
    control={'ref_states':{i:x+1 for i,x in enumerate(sources)},'original_input_ids':ids,'quant_preserves':[{} for _ in ids]},write=lambda *args:None)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
exec('''def fixture(mutate=False):
    state=sources;quant_preserves=[{} for _ in sources];original_input_ids=ids
    capture_H={str(i):{'count_dev':torch.tensor(sum(lengths))} for i in range(516)}
    ref_states={}
    bar=StopAfterRow0();bar.update(0)
    for i in range(5):
        ref_states[i]=sources[i]+1
        if mutate and i==4:ref_states[0].add_(1)
        bar.update(i+1)
    raise AssertionError('row5 must not execute')
''',dict(ns,sources=sources,ids=ids),ns)
f=ns['fixture'];f.__globals__['main_body']=f;ns['main_body']=f
with patch.object(torch.cuda,'get_rng_state',lambda device:rng),patch.object(torch.cuda,'max_memory_allocated',lambda:123),patch.object(torch,'save',lambda *args:None),patch.object(torch,'load',lambda *args,**kwargs:ns['control']):
    try:f()
    except ns['CaptureFinished']:pass
    else:raise AssertionError('no stop')
    assert trace==[0,1,2,3,4] and report['H_tokens']==19848
    assert all(v['exact'] for v in report['immediate'].values())
    try:f(True)
    except AssertionError as error:assert str(error)=='CPU reference mutated later'
    else:raise AssertionError('missed mutation')
r=dict(status='pass',gpu_used=False,checks=['five_immediate_reference_copies','stop_before_row5_quantization','reject_later_reference_mutation','516_counts_19848','inputs_preserves_ID_RNG'],probe_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
Path(sys.argv[1]).write_text(json.dumps(r,indent=2));print(json.dumps(r))
