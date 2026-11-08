"""CPU operand/alias/source/guard tests; GPU device is not exposed."""
import hashlib,inspect,json,copy
from pathlib import Path
import torch
from exllamav3 import Config,Model
from exllamav3.modules.linear import flush_capture_H,Linear
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from exllamav3.modules.quant.fp16 import LinearFP16
import capture_fast,capture_lt,capture_lean_gram as gram,capture_static_moe as moe
R=Path('/output/capture-static');R.mkdir(exist_ok=True);checks=[]
def exact(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
def old_operands(xs):
    xf=xs.to(torch.float32);finite=torch.isfinite(xf).all(dim=2);xf.masked_fill_(~finite.unsqueeze(2),0.)
    return (xs.masked_fill(~finite.unsqueeze(2),0.) if xs.dtype==torch.float16 else xf),finite
# Exhaust all half bit patterns in independent rows, and rows with mixed bad/good entries.
x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16).reshape(1,-1,1)
for value in (x,x.reshape(1,-1,8)):
    expected,counts=old_operands(value);actual,got=gram.prepare_stack(value.clone());exact(expected,actual);exact(counts,got)
checks.append('all_65536_half_patterns_and_mixed_finite_rows_byte_exact')
for dtype in (torch.float16,torch.bfloat16,torch.float32):
    x=torch.tensor([[[0.,-0.,1.],[float('inf'),1.,2.],[2.,float('nan'),3.],[-2.,.1,7.]]],dtype=dtype)
    expected,counts=old_operands(x.clone());actual,got=gram.prepare_stack(x.clone());exact(expected,actual);exact(counts,got)
checks.append('half_bfloat_float_nonfinite_signedzero_operand_exact')
def product(x):return torch.bmm(x.float().transpose(1,2),x.float())
def old_product(xf,xs,finite):return product(xs.masked_fill(~finite.unsqueeze(2),0.) if xs.dtype==torch.float16 else xf)
old_flush=capture_lt.make_flush(flush_capture_H,old_product);new_flush=gram.make_flush(flush_capture_H,product)
params={'_cap_pending':[]};torch.manual_seed(77)
for dtype in (torch.float16,torch.bfloat16,torch.float32):
    shared={'H':torch.zeros(8,8)}
    for i in range(67):
        x=torch.randn(5,8).to(dtype)
        if i%11==0:x[1,3]=float('nan')
        hd=shared if i%3==0 else {'H':torch.zeros(8,8),'count_dev':torch.tensor(9)}
        params['_cap_pending'].append((hd,x))
before=[x.clone() for h,x in params['_cap_pending']];left=copy.deepcopy(params);right=copy.deepcopy(params)
lh=[h for h,x in left['_cap_pending']];rh=[h for h,x in right['_cap_pending']]
old_flush(left);new_flush(right)
assert left==right=={}
for a,b in zip(lh,rh):
    exact(a['H'],b['H']);exact(a['count_dev'],b['count_dev'])
for (_,x),saved in zip(params['_cap_pending'],before):exact(x,saved)
new_flush({});new_flush({'_cap_pending':[]});checks.append('full_flush_chunk_tail_repeated_H_counts_source_immutable')
for rows in (33,512):
    e=8;selected=torch.arange(e).repeat(rows,1);tokens=torch.arange(rows).repeat_interleave(e)
    order=selected.flatten().argsort(stable=True);grouped=tokens[order];y=torch.randn(rows,16)
    g=moe.Gather(e);copies=[]
    for i in range(e):
        ids=grouped[i*rows:(i+1)*rows];exact(ids,torch.arange(rows))
        copies.append(g.get(y,ids,rows));exact(copies[-1],y)
    assert all(v is copies[0] for v in copies);g.drained();saved=copies[0].clone();y.zero_();exact(copies[0],saved)
checks.append('stable_all_expert_grouping_owning_gather_lifetime')
contract=json.loads(Path('/task/capture-static-contract.json').read_text())
old_inner=LinearFP16.forward;old_moe=BlockSparseMLP.forward
def fallback(*args,**kwargs):return 'original'
fallback._gufo_capture_fast=True
try:
    LinearFP16.forward=fallback
    with moe.installed(contract):
        assert LinearFP16.forward(None,None,{})=='original'
    assert LinearFP16.forward is fallback and BlockSparseMLP.forward is old_moe
    try:
        with moe.installed(contract):raise RuntimeError('intentional')
    except RuntimeError:pass
    assert LinearFP16.forward is fallback and BlockSparseMLP.forward is old_moe
finally:LinearFP16.forward=old_inner
checks.append('pinned_source_transform_install_failure_restore')
cfg=Config.from_directory('/source');model=Model.from_config(cfg);mlp=model.modules[2].mlp
# Only attach metadata fake inners; no weights loaded and no GPU.
for outer in (*mlp.ups,*mlp.gates,*mlp.downs):
    inner=object.__new__(LinearFP16);inner.bias=None;inner._pinned_store=None;inner.out_dtype=outer.out_dtype;outer.inner=inner
from types import SimpleNamespace
# Loader initializes num_experts from this same metadata; no router weights needed here.
mlp.routing_cfg=SimpleNamespace(num_experts=mlp.num_experts)
params={'activate_all_experts':True,'capture':{}}
moe.guards(mlp,params);checks.append('actual_model_metadata_guard')
for field,value in [('routing_gate',None),('latent_in',object()),('num_slices',2),('routing_first',1),('bc',object()),('cpu_offload',True)]:
    saved=getattr(mlp,field);setattr(mlp,field,value)
    try:moe.guards(mlp,params)
    except AssertionError:pass
    else:raise AssertionError(field+' guard accepted')
    finally:setattr(mlp,field,saved)
checks.append('unsupported_metadata_rejected');cfg.stc.close()
# Aggregate repeats exactly, including nonfinite values and preexisting counters.
from capture_repeat_counts import RepeatedCounts
for repeats in (1,2,1024):
    x=torch.tensor([[float('inf'),float('-inf'),float('nan'),0.,1.]])
    target=torch.tensor([7,13],dtype=torch.long);expected=target.clone();calls=[]
    def count(value,out):
        calls.append(1);out.add_(torch.stack((value.isinf().sum(),value.isnan().sum())))
    cache=RepeatedCounts()
    for i in range(repeats):cache.add(x,target,count);expected.add_(torch.tensor([2,1]))
    assert len(calls)==1;cache.commit();exact(target,expected)
    try:cache.add(x,target,count)
    except AssertionError:pass
    else:raise AssertionError('closed cache accepted')
for target in (torch.zeros(2),torch.zeros(3,dtype=torch.long),torch.zeros(4,dtype=torch.long)[::2]):
    try:RepeatedCounts().add(torch.zeros(1),target,lambda *a:None)
    except AssertionError:pass
    else:raise AssertionError('invalid native counter accepted')
checks.append('integer_nonfinite_repeat_counter_exact_and_closed_guard')
# Exercise the cloned real capture_H with CPU count mock; H queue/num_total stays identical.
import exllamav3.modules.linear as linear_module
from types import SimpleNamespace as NS
orig_count=linear_module.ext.count_inf_nan;old_inner=LinearFP16.forward
try:
    linear_module.ext.count_inf_nan=lambda x,out:out.add_(torch.stack((x.isinf().sum(),x.isnan().sum())))
    LinearFP16.forward=fallback
    x=torch.tensor([[float('inf'),1.],[float('nan'),2.],[3.,4.]])
    def hd():return dict(H=torch.zeros(2,2),first_key='up0',count=0,finalized=False,num_total=0,inf_nan=torch.tensor([4,5]),device='cpu')
    outers=[NS(qmap='shared',key='up0' if i==0 else f'other{i}',cap_rkey=None) for i in range(6)]
    baseline={'capture':{'shared':hd()}};candidate={'capture':{'shared':hd()}}
    for outer in outers:Linear.capture_H(outer,x,baseline)
    counts=RepeatedCounts();g=NS(value=x)
    with moe.installed(contract):
        token=moe._current.set((None,candidate,g,counts,frozenset(id(o) for o in outers)))
        try:
            for outer in outers:Linear.capture_H(outer,x,candidate)
        finally:moe._current.reset(token)
        counts.commit()
    assert counts.calls==6 and counts.kernels==1
    for name in ('H','inf_nan'):exact(baseline['capture']['shared'][name],candidate['capture']['shared'][name])
    assert baseline['capture']['shared']['num_total']==candidate['capture']['shared']['num_total']==36
    assert len(baseline['_cap_pending'])==len(candidate['_cap_pending'])==1
    assert candidate['_cap_pending'][0][1].data_ptr()==x.data_ptr()
finally:linear_module.ext.count_inf_nan=orig_count;LinearFP16.forward=old_inner
checks.append('real_capture_H_queue_total_nonfinite_exact')
files=['capture_lean_gram.py','capture_static_moe.py','capture_repeat_counts.py','capture-static-contract.json','test_capture_static.py','capture_static_boundary.py']
for name in ['capture_static_probe.py','capture_static_job.py']:
    if Path('/task',name).exists():compile(Path('/task',name).read_text(),name,'exec');files.append(name)
result=dict(status='pass',gpu_used=False,checks=checks,identity={n:hashlib.sha256(Path('/task',n).read_bytes()).hexdigest() for n in files})
(R/'CPU.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
