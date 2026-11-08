"""Replay saved real inputs through complete flush, then paired quantization.

All modes use identical samples/weights/worker arithmetic. This is not a
replacement calibration corpus or production installation.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from exllamav3 import Config,Model
from exllamav3.modules import Linear
from exllamav3.modules.linear import flush_capture_H
from exllamav3.conversion.convert_model import make_quant_args
from exllamav3.modules.quant.exl3_lib.quantize import set_worker_stream
from exllamav3.conversion.ngram import prepare_ngram_table_for_conversion
from exllamav3.modules.quant import fp16,exl3
import os
from exllamav3.loader.safetensors_alt import safe_open
from disk_state import _read_row
from capture_inputs import capture_inputs,InputsCaptured
from capture_lt import LtGram,make_flush

ROOT=Path('/output');OUT=ROOT/'capture-lt';OLD=ROOT/'capture-gram'
assert not (OUT/'SUMMARY.json').exists()
assert os.environ['EXL3_HIP_F32OUT_VIA_F16']=='0' and not fp16._f32_via_f16 and not exl3._f32_via_f16
mounts=[line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines()]
for p in ('/output/work','/output/pack','/output/capture-gram','/output/capture-quarter-holdout'):
    assert any(v[4]==p and 'ro' in v[5].split(',') for v in mounts)
started=time.monotonic()
def write(name,x):
    p=OUT/(name+'.tmp');p.write_text(json.dumps(x,indent=2));p.replace(OUT/name)
def cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    return copy.deepcopy(x)
def stats(p):
    s=p.stat();return [s.st_size,s.st_mtime_ns,s.st_ino]
protected={str(p):stats(p) for p in (ROOT/'work/ckpt/job.json',ROOT/'pack/ngram_embedding.safetensors',OLD/'real-inputs.pt')}
assert json.loads((OLD/'REVIEW.json').read_text())['status']=='reviewed_candidates_for_flush_and_matched_quantization'
original_inputs=torch.load(OLD/'real-inputs.pt',map_location='cpu',weights_only=True)
input_sha=hashlib.sha256((OLD/'real-inputs.pt').read_bytes()).hexdigest()
controls={name:torch.load(OLD/(name+'-control.pt'),map_location='cpu',weights_only=True) for name in original_inputs}
for name,x in original_inputs.items():assert torch.equal(x,controls[name]['x'])
def difference(a,b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    d=b.double()-a.double()
    return dict(exact=torch.equal(a.view(torch.uint8),b.view(torch.uint8)),max_abs=float(d.abs().max()),
        relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(a.double()).clamp_min(1e-30)))

native={mode:LtGram(OLD/'gram.so',mode) for mode in ('f32','original_f16')}
flushes={'original':flush_capture_H,**{m:make_flush(flush_capture_H,n) for m,n in native.items()}}
records=[];captured={}
with torch.inference_mode():
    inputs={k:v.cuda() for k,v in original_inputs.items()}
    def new_H():
        return {k:dict(H=torch.zeros(d,d,device='cuda'),count=0,num_total=0,
            inf_nan=torch.zeros(2,dtype=torch.long,device='cuda'),first_key=k,device=torch.device('cuda:0'),finalized=False)
            for k,d in [('dense',6144)]+[(f'expert{i}',640) for i in range(64)]}
    def run(mode,H):
        for name in ('expert64_L512','dense_L512','expert64_L8192'):
            x=inputs[name];keys=['dense'] if name=='dense_L512' else [f'expert{i}' for i in range(64)]
            params={'_cap_pending':[(H[k],x[i]) for i,k in enumerate(keys)]}
            for i,k in enumerate(keys):H[k]['num_total']+=x[i].numel()
            flushes[mode](params)
            assert '_cap_pending' not in params
    expected={k:torch.zeros_like(hd['H']) for k,hd in new_H().items()}
    for name in ('expert64_L512','dense_L512','expert64_L8192'):
        p=controls[name]['reference'].cuda();keys=['dense'] if name=='dense_L512' else [f'expert{i}' for i in range(64)]
        for i,k in enumerate(keys):expected[k].add_(p[i])
    for mode in flushes:
        warm=new_H();run(mode,warm);torch.cuda.synchronize();del warm
    # Two matched ABBA comparisons, same reference data reused for numeric checks.
    for ordinal,mode in enumerate(('original','f32','f32','original','original','original_f16','original_f16','original')):
        assert time.monotonic()-started<500
        H=new_H();torch.cuda.synchronize();begin=time.perf_counter()
        run(mode,H);torch.cuda.synchronize();elapsed=time.perf_counter()-begin
        saved=cpu(H);torch.save(saved,OUT/f'{ordinal}-{mode}-H.pt')
        diffs={}
        for k,hd in H.items():
            assert int(hd['count_dev'])==(512 if k=='dense' else 8704)
            assert int(hd['inf_nan'].sum())==0
            diffs[k]=difference(expected[k],hd['H'])
        if mode=='original':assert all(v['exact'] for v in diffs.values()),'Saved BMM control vs original flush mismatch'
        if mode in captured:
            assert all(torch.equal(saved[k]['H'].view(torch.uint8),captured[mode][k]['H'].view(torch.uint8)) for k in saved),'Repeated candidate unstable'
        captured[mode]=saved
        records.append(dict(ordinal=ordinal,mode=mode,seconds=elapsed,differences=diffs,counts_exact=True))
        write('FLUSH.json',records);del H
    for n in native.values():n.close()
    del expected,inputs,controls
    torch.cuda.empty_cache()
    # Matched quantization: same small H samples in all modes, never full2M-vs-small.
    config=Config.from_directory('/source');model=Model.from_config(config);block=model.modules[2]
    prepare_ngram_table_for_conversion(dict(out_dir='/output/pack',ngram_file='/output/pack/ngram_embedding.safetensors'),config,model)
    config.stc.begin_deferred_load()
    try:block.load(torch.device('cuda:0'))
    finally:config.stc.end_deferred_load()
    config.stc.close()
    selected={m.key:m for m in block if isinstance(m,Linear) and (m.key.endswith('.linear_attn.out_proj') or m.key.endswith('.mlp.experts.0.down_proj'))}
    assert len(selected)==2
    evaluation={}
    with safe_open('/output/capture-quarter-holdout/input-ids.safetensors') as reader:
        # New diagnostics are exactly the existing eight fixed input documents.
        for m in model.modules[:2]:m.load(torch.device('cuda:0'))
        docs=json.loads((ROOT/'capture-quarter-holdout/DATA.json').read_text())['rows']
        terminal=next(k for k in selected if k.endswith('.experts.0.down_proj'))
        for i,doc in enumerate(docs):
            ids=_read_row(reader,f'input_ids.{i}').long();x=ids.clone()
            for m in model.modules[:2]:
                params={'input_ids':ids};x=m.forward(m.prepare_for_device(x,params),params)
            params=dict(input_ids=ids,attn_mode='flash_attn_nc',capture={},activate_all_experts=True,quant_preserve={})
            observed={}
            with capture_inputs(selected,terminal,observed):
                try:block.forward(block.prepare_for_device(x,params),params)
                except InputsCaptured:pass
                else:raise AssertionError('Missing observer boundary')
            evaluation[doc['id']]={}
            for k,v in observed.items():
                v=v.reshape(-1,v.shape[-1]);idx=torch.linspace(0,len(v)-1,min(256,len(v)),device=v.device).long()
                evaluation[doc['id']][k]=v[idx].cpu()
        for m in model.modules[:2]:m.unload()
    torch.save(evaluation,OUT/'evaluation-inputs.pt')
    quant_results=[]
    for key,linear in selected.items():
        original=linear.inner;kind='dense' if key.endswith('out_proj') else 'expert0';K=6 if kind=='dense' else 4
        refs={doc:original.forward(values[key].cuda().unsqueeze(0),{}).float().cpu() for doc,values in evaluation.items()}
        rotation=None
        quant_cpu_rng=torch.random.get_rng_state();quant_gpu_rng=torch.cuda.get_rng_state()
        for mode in ('original','f32','original_f16'):
            assert time.monotonic()-started<540
            torch.random.set_rng_state(quant_cpu_rng);torch.cuda.set_rng_state(quant_gpu_rng)
            hd=copy.deepcopy(captured[mode][kind]);hd['H_swap_device']=torch.device('cuda:0')
            # init metadata records the actual upstream linear, not diagnostic label.
            hd['first_key']=key
            qa=make_quant_args(dict(apply_out_scales=True,codebook='mul1',work_dir=str(OUT)),2,K,[0])
            stream=torch.cuda.Stream();torch.cuda.synchronize();set_worker_stream(stream)
            try:
                with torch.cuda.stream(stream):proxy=linear.convert_exl3(hd,qa)
                stream.synchronize()
            finally:set_worker_stream(None)
            packed=linear.inner
            assert not qa.get('q_fallback') and math.isfinite(float(proxy)) and float(proxy)>=0
            assert math.isfinite(float(qa['g_scale'])) and packed.K==K
            signs=[torch.sign(getattr(packed,n)).cpu() for n in ('suh','svh')]
            if rotation is None:rotation=signs
            same=all(torch.equal(a,b) for a,b in zip(rotation,signs))
            for n in ('suh','svh','bias'):
                v=getattr(packed,n);assert v is None or torch.isfinite(v).all()
            torch.save(cpu(packed.get_tensors(key)),OUT/f'{kind}-{mode}-packed.pt')
            assert same,('Rotation signs differ; paired quantization disqualified',key,mode)
            errors=[]
            for doc,values in evaluation.items():
                x=values[key].cuda().unsqueeze(0);ref=refs[doc].cuda();y=packed.forward(x,{}).float()
                assert torch.isfinite(y).all() and torch.isfinite(ref).all()
                errors.append(dict(doc=doc,relative_error=float(torch.linalg.vector_norm(y-ref)/torch.linalg.vector_norm(ref)),
                    sampled_p99=float(torch.quantile(torch.linalg.vector_norm(y-ref,dim=-1)/torch.linalg.vector_norm(ref,dim=-1).clamp_min(1e-20),.99))))
            quant_results.append(dict(key=key,mode=mode,K=K,proxy=float(proxy),rotation_signs_same=same,errors=errors,
                H_tokens=512 if kind=='dense' else 8704,fallback=False))
            write('QUANT.json',quant_results);linear.inner=original
        linear.inner=original
    block.unload()
    assert all(stats(Path(p))==s for p,s in protected.items())
    write('SUMMARY.json',dict(status='flush_and_paired_quant_complete',flush=records,quant=quant_results,
        elapsed_seconds=time.monotonic()-started,input_sha256=input_sha,protected_stats=protected,production_changed=False,
        scope='Complete original flush on saved65matrices from512/8192 inputs; paired quantization two projections with the same small samples; eight fixed diagnostic docs. Not full2M/all-layer quality or complete-conversion timing. Dense512 H is rank-deficient before damping.'))
