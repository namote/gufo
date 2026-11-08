"""CPU-only protocol, output lifetime and pinned-source contract tests."""
import ast, hashlib, inspect, json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from exllamav3 import Config, Model
from exllamav3.modules.linear import Linear
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from capture_forward_pair import PairState, installed, cleanup_all
from exllamav3.modules.quant.fp16 import LinearFP16
import capture_fast
R=Path('/output/capture-forward-pair');R.mkdir(exist_ok=True)
checks=[]
def pair_state():
    a,b,c,d=[NS(out_dtype=torch.float16) for _ in range(4)]
    w=torch.arange(2*4*3).reshape(2,4,3).half()/16
    def execute(x,w):return torch.stack([x@z for z in w])
    return PairState([(a,b,w),(c,d,w)],execute),(a,b,c,d)
with torch.inference_mode():
    state,(a,b,c,d)=pair_state();x=torch.ones(5,4).half();params={}
    up=state.call(a,x,params,None);gate=state.call(b,x,params,None);saved=up.clone()
    gate.add_(7);assert torch.equal(up,saved)
    up.mul_(3);saved=up.clone() # simulate activation retained by down H
    next_up=state.call(c,x,params,None);state.call(d,x,params,None);next_up.zero_()
    assert torch.equal(up,saved);state.drained()
    checks+=['inference_tensor_without_version','up_gate_disjoint','retained_down_H_input_not_overwritten']
    for action in ['gate_first','second_up','wrong_gate','different_input','different_params','wrong_dtype']:
        state,(a,b,c,d)=pair_state();params={}
        try:
            if action=='gate_first':state.call(b,x,params,None)
            else:
                state.call(a,x,params,None)
                if action=='second_up':state.call(c,x,params,None)
                elif action=='wrong_gate':state.call(d,x,params,None)
                elif action=='different_input':state.call(b,x.clone(),params,None)
                elif action=='different_params':state.call(b,x,{},None)
                else:state.call(b,x,params,torch.float32)
        except AssertionError:pass
        else:raise AssertionError(action+' accepted')
        checks.append(action+'_rejected')
    old=LinearFP16.forward
    def fallback(inner,x,params,out_dtype=None):return 'original'
    fallback._gufo_capture_fast=True;LinearFP16.forward=fallback
    state,(a,b,c,d)=pair_state();down=object();block=NS(mlp=NS(downs=[NS(inner=down)]));params={}
    try:
        with installed(block,state):
            assert LinearFP16.forward(a,x,params)=='original'
            token=capture_fast._active.set((params,frozenset([id(down)])))
            try:
                LinearFP16.forward(a,x,params);LinearFP16.forward(b,x,params)
            finally:capture_fast._active.reset(token)
        assert LinearFP16.forward is fallback
        try:
            with installed(block,state):raise RuntimeError('intentional')
        except RuntimeError:pass
        assert LinearFP16.forward is fallback
    finally:LinearFP16.forward=old
    checks+=['normal_reference_unmodified','scoped_pair_dispatch','restore_on_failure']
contract=json.loads(Path('/task/capture-forward-pair-contract.json').read_text())
for name,fn in [('linear',Linear.forward),('moe',BlockSparseMLP.forward)]:
    assert hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()==contract[name]
checks.append('pinned_adjacent_call_and_outer_source')
seen=[]
def bad_cleanup():
    seen.append('failed');raise ValueError('intentional cleanup fault')
primary=RuntimeError('original kernel failure')
errors=cleanup_all([('bad',bad_cleanup),('later',lambda:seen.append('later'))],primary)
assert seen==['failed','later'] and len(errors)==1 and primary.__notes__
checks.append('cleanup_preserves_primary_runs_all')
old=LinearFP16.forward
try:
    LinearFP16.forward=fallback
    state,(a,b,c,d)=pair_state();params={};scope=None
    try:
        with installed(block,state):
            token=capture_fast._active.set((params,frozenset([id(down)])))
            try:
                LinearFP16.forward(a,x,params)
                raise RuntimeError('between up and gate')
            finally:capture_fast._active.reset(token)
    except RuntimeError as e:assert str(e)=='between up and gate'
    assert state.pending is None and LinearFP16.forward is fallback
finally:LinearFP16.forward=old
checks.append('pending_gate_failure_restores_hook')
cfg=Config.from_directory('/source');model=Model.from_config(cfg);m=model.modules[2].mlp
assert m.gated and m.interm_dtype==torch.float16 and len(m.ups)==len(m.gates)==512
for u,g in zip(m.ups,m.gates):
    for o in (u,g):
        assert not o.is_sliced and not o.trim_padded_out
        assert o.in_features==o.in_features_unpadded and o.out_features==o.out_features_unpadded
        assert o.pre_scale==o.post_scale==1 and o.softcap==0 and not o.lora_a_tensors
        assert o.out_dtype==torch.float16
cfg.stc.close();checks.append('actual_model_pair_metadata')
# Exercise the probe's exact helper without executing its GPU-only top level.
probe_ast=ast.parse(Path('/task/capture_forward_pair_probe.py').read_text())
exact_node=next(n for n in probe_ast.body if isinstance(n,ast.FunctionDef) and n.name=='exact')
ns={'torch':torch};exec(compile(ast.Module(body=[exact_node],type_ignores=[]),'<probe-cpu>','exec'),ns)
ns['exact']({'scalar':torch.tensor(7),'nested':[torch.tensor(1.5)]},{'scalar':torch.tensor(7),'nested':[torch.tensor(1.5)]})
try:ns['exact'](torch.tensor(0.),torch.tensor(-0.))
except AssertionError:pass
else:raise AssertionError('Signed zero byte change was ignored')
checks += ['probe_scalar_nested_exact','probe_signed_zero_strict']
for name in ['capture_forward_pair_probe.py','capture_forward_pair_job.py']:
    compile(Path('/task',name).read_text(),name,'exec')
checks.append('probe_and_runner_compile')
files=['capture_forward_pair.py','test_capture_forward_pair.py','capture-forward-pair-contract.json','capture_forward_pair_probe.py','capture_forward_pair_job.py']
result=dict(status='pass',gpu_used=False,checks=checks,identity={n:hashlib.sha256((Path('/task')/n).read_bytes()).hexdigest() for n in files})
(R/'CPU.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
