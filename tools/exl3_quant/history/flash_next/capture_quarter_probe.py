"""Screen actual quarter-c03 H on two layer0 projections vs retained full2M pack.

Uses archived idx2 states, never the current idx10 state as a layer0 fixture.
Two projections are a diagnostic screen, not model/independent-holdout quality.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
from exllamav3 import Config, Model
from exllamav3.modules import Linear
from exllamav3.modules.linear import flush_capture_H
from exllamav3.modules.quant import fp16, exl3
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.conversion.convert_model import make_quant_args
from exllamav3.modules.quant.exl3_lib.quantize import set_worker_stream
from exllamav3.loader.safetensors_alt import safe_open
from capture_selected import selected_capture, PrefixCaptured
from disk_state import _read_row

OUT=Path('/output/capture-profile-quarter/quarter')
OUT.mkdir()
SOURCE=Path('/output/model-job-c03/repair-002-checkpoint/checkpoint-idx2/state.safetensors')
FULL=Path('/output/work/qtensors/model.language_model.layers.0.safetensors')
CANDIDATE=Path('/output/capture-quarter/CANDIDATE.json')
assert os.environ['EXL3_HIP_F32OUT_VIA_F16']=='0' and not fp16._f32_via_f16 and not exl3._f32_via_f16
assert os.environ['EXL3_RWH']=='0'
assert os.environ['EXL3_ENABLE_GROUPING']=='0'
mounts=[line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines()]
for path in ('/output/work','/output/pack','/output/model-job-c03/repair-002-checkpoint','/output/capture-quarter','/output/capture-speed','/output/capture-regression'):
    assert any(p[4]==path and 'ro' in p[5].split(',') for p in mounts),path
candidate=json.loads(CANDIDATE.read_text())
cal=json.loads(Path('/task/c03-input-ids.json').read_text())
assert candidate['source_sha256']==cal['sha256'] and candidate['status']=='selected_not_applied'
assert candidate['selected_tokens']==503322 and len(candidate['selected_indices'])==577
assert candidate['selector_sha256']==hashlib.sha256(Path('/task/select_capture_quarter.py').read_bytes()).hexdigest()
indices=candidate['selected_indices']; chosen=set(indices)
# One unselected row per source and long-length bucket; diagnostic only, not independent holdout.
eval_groups={}
for i,row in enumerate(cal['sequences']):
    if i not in chosen:
        eval_groups.setdefault((row['source'], row['tokens']>2048),i)
eval_indices=sorted(eval_groups.values())
assert set(eval_indices).isdisjoint(chosen)

def write(name,data):
    p=OUT/name; temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(p)
def stat(path):
    s=path.stat();return [s.st_size,s.st_mtime_ns,s.st_ino]
def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    return copy.deepcopy(value)
def to_device(value):
    if isinstance(value,torch.Tensor):return value.to('cuda:0',copy=True)
    if isinstance(value,dict):return {k:to_device(v) for k,v in value.items()}
    return copy.deepcopy(value)
def exact(a,b):
    if isinstance(a,torch.Tensor):
        assert a.dtype==b.dtype and a.shape==b.shape
        assert torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    else:assert a==b,(a,b)

geometry=[]
write('BACKEND.json',dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
    allow_tf32=torch.backends.cuda.matmul.allow_tf32,
    allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    deterministic=torch.are_deterministic_algorithms_enabled(),torch_version=torch.__version__,hip_version=torch.version.hip))
started=time.monotonic(); protected={str(p):stat(p) for p in (SOURCE,FULL,CANDIDATE)}
config=Config.from_directory('/source');model=Model.from_config(config);block=model.modules[2]
assert block.key=='model.language_model.layers.0' and block.num_slices==1
with torch.inference_mode():
    deferred=block.can_defer_load()
    if deferred:config.stc.begin_deferred_load()
    try:block.load(torch.device('cuda:0'))
    finally:
        if deferred:config.stc.end_deferred_load()
    config.stc.close()
    suffix_bits={'linear_attn.out_proj':6,'mlp.experts.0.down_proj':4}
    selected={m.key:m for m in block if isinstance(m,Linear) and any(m.key.endswith('.'+s) for s in suffix_bits)}
    assert len(selected)==2
    down_keys=[f'{block.key}.mlp.experts.{i}.down_proj' for i in range(64)]
    capture_targets={m.key:m for m in block if isinstance(m,Linear) and (m.key in down_keys or m.key in selected)}
    assert len(capture_targets)==65 and len({m.qmap for m in capture_targets.values()})==65
    terminal=down_keys[-1]
    qmaps={m.qmap for m in selected.values()};assert len(qmaps)==2
    timings=[]
    with safe_open(str(SOURCE)) as reader, safe_open('/task/c03-input-ids.safetensors') as ids_reader:
        assert len(reader.keys())==cal['rows']
        def capture(index,H,inputs=None,prefix=True):
            assert time.monotonic()-started<1050, 'Bounded quarter screen'
            t=time.perf_counter();x=_read_row(reader,f'tensor.{index}')
            ids=_read_row(ids_reader,f'input_ids.{index}')
            assert x.shape==(1,cal['sequences'][index]['tokens'],4,2560) and x.dtype==torch.float32
            assert hashlib.sha256(memoryview(ids.contiguous().view(torch.uint8).numpy().reshape(-1))).hexdigest()==cal['sequences'][index]['token_sha256']
            read_seconds=time.perf_counter()-t
            params=dict(attn_mode='flash_attn_nc',input_ids=ids.long(),quant_preserve={},capture=H,activate_all_experts=True)
            # No changes to block/expert arithmetic; execution terminates at a captured input.
            if prefix:
                captured_inputs={} if inputs is not None else None
                with selected_capture(capture_targets,terminal,captured_inputs):
                    try:block.forward(block.prepare_for_device(x,params),params)
                    except PrefixCaptured:pass
                    else:raise AssertionError('Prefix did not stop')
                if inputs is not None:
                    inputs.update({k:captured_inputs[k] for k in selected})
                pending=params['_cap_pending']
                downs=[(hd,value) for hd,value in pending if hd['first_key'] in down_keys]
                assert [hd['first_key'] for hd,value in downs]==down_keys
                assert all(value.shape==(cal['sequences'][index]['tokens'],640) and value.dtype==torch.float16 and value.device.type=='cuda' for hd,value in downs)
                assert len(pending)==65, 'Exactly one attn-out plus the original first64-down chunk'
            else:
                block.forward(block.prepare_for_device(x,params),params)
            if len(geometry)<6:
                groups={}
                for hd,value in params['_cap_pending']:
                    identity=(tuple(value.shape),tuple(value.stride()),str(value.dtype),str(value.device))
                    groups[identity]=groups.get(identity,0)+1
                geometry.append(dict(index=index,prefix=prefix,groups=[dict(shape=k[0],stride=k[1],dtype=k[2],device=k[3],entries=v,chunks=[min(64,v-start) for start in range(0,v,64)]) for k,v in groups.items()]))
                write('BMM-GEOMETRY.json',geometry)
            flush_capture_H(params)
            return read_seconds
        # Existing full-path oracle, never rebuild its full512-expert baseline.
        initial=torch.load('/output/capture-speed/baseline-2-H.pt',map_location='cpu',weights_only=True)
        parity=[]
        def compare_h(left,right):
            report={}
            for key in sorted(qmaps):
                a,b=left[key],right[key]
                try:exact(a,b);same=True
                except AssertionError:same=False
                x,y=a['H'].float(),b['H'].float();diff=y-x
                report[key]=dict(exact=same,max_abs=float(diff.abs().max()),
                    relative_l2=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(x).clamp_min(1e-30)),
                    finite=bool(torch.isfinite(y).all()),changed_values=int(torch.count_nonzero(x!=y)))
            return report
        for length,index in ((512,4),(8192,364)):
            # A tiny current-process full control qualifies the changed prefix geometry.
            # Old saved control remains visible and is never overwritten/relabelled.
            full_H=to_device(initial)
            capture(index,full_H,prefix=False)
            current={k:cpu_tree(full_H[k]) for k in qmaps};del full_H
            oracle=torch.load(f'/output/capture-regression/control-{length}-H.pt',map_location='cpu',weights_only=True)
            old_comparison=compare_h({k:oracle[k] for k in qmaps},current);del oracle
            repeat_comparison=None
            if length==512 and not all(v['exact'] for v in old_comparison.values()):
                repeat_H=to_device(initial);capture(index,repeat_H,prefix=False)
                repeated={k:cpu_tree(repeat_H[k]) for k in qmaps};del repeat_H
                repeat_comparison=compare_h(current,repeated)
                torch.save(repeated,OUT/'current-original-repeat-512.pt')
            H={k:to_device(v) for k,v in initial.items() if k in {m.qmap for m in capture_targets.values()}}
            capture(index,H)
            observed={k:cpu_tree(H[k]) for k in qmaps};del H
            prefix_comparison=compare_h(current,observed)
            if not all(v['exact'] for v in prefix_comparison.values()) and repeat_comparison is None:
                repeat_H=to_device(initial);capture(index,repeat_H,prefix=False)
                repeated={k:cpu_tree(repeat_H[k]) for k in qmaps};del repeat_H
                repeat_comparison=compare_h(current,repeated)
                torch.save(repeated,OUT/f'current-original-repeat-{length}.pt')
            torch.save(dict(original=current,prefix=observed),OUT/f'prefix-parity-{length}.pt')
            entry=dict(tokens=length,saved_original_comparison=old_comparison,
                       current_original_repeat=repeat_comparison,prefix_comparison=prefix_comparison)
            parity.append(entry);write('PARITY.json',parity)
            assert all(v['exact'] for v in prefix_comparison.values()), 'Current matched prefix parity failed; numeric retained'
            assert repeat_comparison is None or all(v['exact'] for v in repeat_comparison.values()), 'Original repeat unstable; do not begin quarter capture'
        del initial
        H={};torch.cuda.synchronize();capture_start=time.perf_counter();read_total=0.0
        for ordinal,index in enumerate(indices):
            read_total+=capture(index,H)
            if ordinal%50==0:
                write('PROGRESS.json',dict(phase='quarter_capture',rows=ordinal+1,total=len(indices),seconds=time.perf_counter()-capture_start))
        torch.cuda.synchronize();capture_seconds=time.perf_counter()-capture_start
        for hd in H.values():
            assert torch.isfinite(hd['H']).all() and int(hd['inf_nan'].sum())==0
            assert int(hd['count_dev'])==candidate['selected_tokens']
        torch.save(cpu_tree(H),OUT/'quarter-H.pt')
        evaluation={}
        for index in eval_indices:
            inputs={};capture(index,{},inputs)
            # Evenly sample at most256 real activation rows per source/length bin.
            evaluation[index]={k:v.reshape(-1,v.shape[-1])[torch.linspace(0,v.numel()//v.shape[-1]-1,min(256,v.numel()//v.shape[-1]),device=v.device).long()].cpu() for k,v in inputs.items()}
    results=[]
    with safe_open(str(FULL)) as full_reader:
        for key,linear in selected.items():
            original=linear.inner
            tensors={name.removeprefix(key+'.'):_read_row(full_reader,name) for name in full_reader.keys() if name.startswith(key+'.')}
            assert 'trellis' in tensors and 'suh' in tensors and 'svh' in tensors
            def tensor(name):
                v=tensors.get(name)
                return v if v is None or name in ('mcg','mul1') else v.to('cuda:0')
            full=LinearEXL3(config,linear.in_features,linear.out_features,None,
                tensor('su'),tensor('sv'),tensor('suh'),tensor('svh'),tensor('trellis'),tensor('mcg'),tensor('mul1'),tensor('bias'),linear.out_dtype,key=key)
            K=next(v for s,v in suffix_bits.items() if key.endswith('.'+s))
            assert full.K==K
            qa=make_quant_args(dict(apply_out_scales=True,codebook='mul1',work_dir=str(OUT)),2,K,[0])
            try:
                torch.cuda.synchronize()
                stream=torch.cuda.Stream()
                t=time.perf_counter();set_worker_stream(stream)
                try:
                    with torch.cuda.stream(stream):
                        proxy=linear.convert_exl3(copy.deepcopy(H[linear.qmap]),qa)
                    stream.synchronize()
                finally:set_worker_stream(None)
                quarter=linear.inner;assert quarter.K==K and not qa.get('q_fallback') and math.isfinite(float(proxy)) and float(proxy)>=0
                assert math.isfinite(float(qa['g_scale']))
                for packed in (full,quarter):
                    for name in ('suh','svh','bias'):
                        value=getattr(packed,name)
                        assert value is None or torch.isfinite(value).all(),(key,name)
                quant_seconds=time.perf_counter()-t
                # Scales differ legitimately with calibration; compare rotation sign bits separately.
                rotation_same=all(torch.equal(torch.sign(getattr(full,n)),torch.sign(getattr(quarter,n))) for n in ('suh','svh'))
                errors=[]
                for index,inputs in evaluation.items():
                    x=inputs[key].to('cuda:0').unsqueeze(0).contiguous()
                    reference=original.forward(x,{}).float()
                    norm=torch.linalg.vector_norm(reference);assert torch.isfinite(reference).all() and torch.isfinite(norm) and norm>0
                    entry=dict(row=index,source=cal['sequences'][index]['source'])
                    for label,inner in (('full',full),('quarter',quarter)):
                        y=inner.forward(x,{}).float();assert torch.isfinite(y).all()
                        error=float(torch.linalg.vector_norm(y-reference)/norm)
                        assert math.isfinite(error) and error>=0
                        entry[label+'_relative_error']=error
                    errors.append(entry)
                record=dict(key=key,K=K,proxy=float(proxy),quant_seconds=quant_seconds,rotation_signs_same=rotation_same,errors=errors)
                results.append(record);write('RESULTS.json',results)
                torch.save(cpu_tree(quarter.get_tensors(key)),OUT/(key+'.pt'))
            finally:linear.inner=original
    block.unload()
    assert all(stat(Path(p))==s for p,s in protected.items())
    write('SUMMARY.json',dict(status='quarter_projection_screen_complete',parity=parity,results=results,
        candidate_sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path('/task/capture_selected.py').read_bytes()).hexdigest(),
        protected_stats=protected,rows=len(indices),tokens=candidate['selected_tokens'],capture_seconds=capture_seconds,
        read_host_seconds=read_total,elapsed_seconds=time.monotonic()-started,eval_indices=eval_indices,
        production_changed=False,quality_validated=False,
        limitations=['Two layer0 projections only, not a full-model or full-capture timing result.',
        'Unselected c03 evaluation rows are not independent of the full2M baseline.',
        'Rotation differences confound a calibration-only attribution; report them.',
        'Current full/prefix two-row controls qualify extraction; any old-control mismatch remains unresolved and is reported in PARITY.json.',
        'No automatic calibration replacement or checkpoint mutation.']))
