"""Two real rows reproduce conditional capture ownership; no production writes."""
import copy,hashlib,json,time
from pathlib import Path
import torch
from exllamav3 import Config,Model
from exllamav3.conversion import convert_model as cm
from exllamav3.loader.safetensors_alt import safe_open
from capture_fast import install as fast_install,_discarded_capture
from capture_lt import LtGram,make_flush
from capture_owner import install
from disk_state import _read_row
from row_checkpoint import open_rows
ROOT=Path('/output');OUT=ROOT/'capture-owner';started=time.monotonic()
def write(n,v):(OUT/n).write_text(json.dumps(v,indent=2))
def cpu(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    return copy.deepcopy(v)
def exact(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
assert json.loads((ROOT/'work/ckpt/job.json').read_text())['next_module_idx']==10
paths,infos=open_rows(ROOT/'work/ckpt/state.safetensors');inputs={};ids={}
for i in (0,4):
    with safe_open(str(paths[i])) as f:inputs[i]=_read_row(f,'tensor')
with safe_open('/task/c03-input-ids.safetensors') as f:
    for i in inputs:ids[i]=_read_row(f,f'input_ids.{i}')
assert inputs[0].shape[1]==3217 and inputs[4].shape[1]==512
config=Config.from_directory('/source');model=Model.from_config(config);module=model.modules[10]
assert module.key=='model.language_model.layers.7' and type(module.attn).__name__=='Attention'
assert module.attn.qsa_indexer.sparse_threshold()==2051
native=LtGram('/task/native/capture-gram.so','original_f16');flush=make_flush(cm._flush_capture_H,native)
records=[];saved={};restore_fast=fast_install(cm,automatic=False)
with torch.inference_mode():
    config.stc.begin_deferred_load()
    try:module.load(torch.device('cuda:0'))
    finally:config.stc.end_deferred_load()
    config.stc.close()
    try:
        for label,order,fix in [('long-first',(0,4),False),('fixed-long-first',(0,4),True),('short-first',(4,0),False)]:
            H={};counts=[];xs={};refs={};restore=install(type(module.attn)) if fix else None
            linear=module.attn.q_proj;orig_forward=linear.forward
            def observe(x,params,out_dtype=None):
                # Capture independent actual q_proj input without altering it.
                if 'capture' in params:xs[active_row]=cpu(x)
                if out_dtype is None:return orig_forward(x,params)
                return orig_forward(x,params,out_dtype=out_dtype)
            linear.forward=observe
            try:
                for active_row in order:
                    assert time.monotonic()-started<240
                    x=inputs[active_row].cuda();params=dict(input_ids=ids[active_row],attn_mode='flash_attn_nc',capture=H,quant_preserve={},activate_all_experts=True)
                    with _discarded_capture(module,params):module.forward(module.prepare_for_device(x,params),params)
                    flush(params);counts.append(int(H['block.attn.input']['count_dev']))
                    normal=dict(input_ids=ids[active_row],attn_mode='flash_attn_nc',quant_preserve={})
                    refs[active_row]=cpu(module.forward(module.prepare_for_device(x,normal),normal))
                    exact(inputs[active_row],cpu(x));del x,params,normal
                captured=cpu(H)
                torch.save(dict(H=captured,inputs=xs,refs=refs),OUT/(label+'.pt'))
                assert len(captured)==516
                assert all(torch.isfinite(v['H']).all() and int(v['inf_nan'].sum())==0 for v in captured.values())
                saved[label]=dict(H=captured,inputs=xs,refs=refs)
                record=dict(label=label,order=order,count_progress=counts,final_counts={k:int(v['count_dev']) for k,v in captured.items()},first_key=captured['block.attn.input']['first_key'])
                records.append(record);write('PROGRESS.json',records)
            finally:
                linear.forward=orig_forward
                if restore:restore()
    finally:restore_fast();native.close();module.unload();config.stc.close()
a=saved['long-first'];b=saved['fixed-long-first'];c=saved['short-first']
assert records[0]['count_progress']==[3217,3217]
assert records[1]['count_progress']==[3217,3729]
assert records[2]['count_progress']==[512,3729]
for i in (0,4):
    exact(a['inputs'][i],b['inputs'][i]);exact(b['inputs'][i],c['inputs'][i]);exact(a['refs'][i],b['refs'][i]);exact(b['refs'][i],c['refs'][i])
for key,v in b['H'].items():
    assert int(v['count_dev'])==3729
    exact(v['H'],c['H'][key]['H'])
    if key!='block.attn.input':exact(v['H'],a['H'][key]['H'])
# Small independent CPU FP64 submatrix, using every row from actual q_proj inputs.
xx=torch.cat([b['inputs'][i].reshape(-1,2560)[:,:16].double() for i in (0,4)])
oracle=xx.T@xx;actual=b['H']['block.attn.input']['H'][:16,:16].double()
relative=float((actual-oracle).norm()/oracle.norm());assert relative<1e-3
write('SUMMARY.json',dict(status='capture_owner_bug_reproduced_fix_pass',records=[{k:v for k,v in r.items() if k!='final_counts'} for r in records],all516_fixed_counts=3729,normal_refs_exact=True,other515_H_unchanged=True,fixed_vs_short_first_H_exact=True,independent_FP64_relative_l2=relative,seconds=time.monotonic()-started,production_changed=False,scope='Actual layer7 two rows3217/512; prior layer3 pack requires separate recovery decision. No full calibration rerun.'))
