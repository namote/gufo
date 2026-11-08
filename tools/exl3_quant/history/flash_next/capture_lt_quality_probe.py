"""Two layer6 projections: matched64-row H, fixed8 documents, source-BF16 teacher.

Prefix uses source weights through the converter loader plus the existing K8
ngram. This is not a BF16 whole-model/KLD evaluation or a new independent corpus.
"""
import copy,hashlib,json,math,os,time
from pathlib import Path
import torch
from exllamav3 import Config,Model
from exllamav3.modules import Linear
from exllamav3.conversion.convert_model import make_quant_args
from exllamav3.conversion.ngram import prepare_ngram_table_for_conversion
from exllamav3.modules.quant.exl3_lib.quantize import set_worker_stream
from exllamav3.modules.quant import fp16,exl3
from exllamav3.loader.safetensors_alt import safe_open
from disk_state import _read_row
from capture_inputs import capture_inputs,InputsCaptured

ROOT=Path('/output');OUT=ROOT/'capture-lt-quality';EVAL=ROOT/'capture-lt-evaluate'
assert not (OUT/'SUMMARY.json').exists()
assert os.environ['EXL3_HIP_F32OUT_VIA_F16']=='0' and not fp16._f32_via_f16 and not exl3._f32_via_f16
mounts=[x.split() for x in Path('/proc/self/mountinfo').read_text().splitlines()]
for p in ('/output/work','/output/pack','/output/capture-lt-evaluate','/output/capture-pipeline','/output/capture-quarter-holdout'):
    assert any(v[4]==p and 'ro' in v[5].split(',') for v in mounts),p
started=time.monotonic()
def write(n,v):
    q=OUT/(n+'.tmp');q.write_text(json.dumps(v,indent=2));q.replace(OUT/n)
def cpu(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    return copy.deepcopy(v)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while x:=f.read(4*2**20):h.update(x)
    return h.hexdigest()
def source_weight(linear):
    """Read BF16 with explicit direct/fused orientation, no precision fallback."""
    stc=linear.config.stc
    assert linear.weight_scale==1 and linear.inner.bias is None
    if stc.has_tensor(linear.key+'.weight'):
        w=stc.get_tensor(linear.key+'.weight','cpu',allow_bf16=True,no_defer=True,transpose=linear.transposed_load)
        route=dict(key=linear.key+'.weight',transpose=linear.transposed_load)
    else:
        assert linear.fkey and linear.fidx is not None and stc.has_tensor(linear.fkey)
        w=stc.get_tensor(linear.fkey,'cpu',allow_bf16=True,no_defer=True,transpose=linear.transpose_fused_weights,fidx=linear.fidx)
        if linear.frange is not None:
            a,b=linear.frange
            assert linear.frange_dim in (0,1)
            w=w[a:b,:] if linear.frange_dim==0 else w[:,a:b]
        if linear.ftranspose_after_load:w=w.T
        route=dict(key=linear.fkey,fidx=linear.fidx,transpose=linear.transpose_fused_weights,frange=linear.frange,frange_dim=linear.frange_dim,post_transpose=linear.ftranspose_after_load)
    assert w.dtype==torch.bfloat16 and torch.isfinite(w).all()
    assert w.shape==(linear.in_features,linear.out_features),'Selected projections must not require padding'
    loaded=linear.inner.get_weight_tensor().cpu()
    assert torch.equal(w.to(loaded.dtype),loaded),'Loader differs from explicit source conversion'
    route.update(source_dtype=str(w.dtype),loaded_dtype=str(loaded.dtype),source_values_exact=bool(torch.equal(w.float(),loaded.float())),max_load_abs=float((w.float()-loaded.float()).abs().max()))
    return w.contiguous(),route

def error_metrics(y,ref):
    assert y.shape==ref.shape and torch.isfinite(y).all() and torch.isfinite(ref).all()
    delta=y.float()-ref.float()
    return dict(relative_error=float(delta.norm()/ref.float().norm().clamp_min(1e-30)),
        sampled_p99=float(torch.quantile(delta.norm(dim=-1)/ref.float().norm(dim=-1).clamp_min(1e-20),.99)))

paths={'original':ROOT/'capture-pipeline/control.pt'}
for mode in ('f32','original_f16'):
    paths[mode]=Path(json.loads((EVAL/(mode+'-first-timed.json')).read_text())['numeric'])
protected={str(p):digest(p) for p in [ROOT/'work/ckpt/job.json',ROOT/'capture-quarter-holdout/input-ids.safetensors',ROOT/'capture-quarter-holdout/DATA.json']}
H={};provenance={}
for mode,path in paths.items():
    data=torch.load(path,map_location='cpu',weights_only=True)
    H[mode]={k:copy.deepcopy(data['capture_H'][k]) for k in ('block.attn.output','block.mlp.0.down')}
    assert all(int(h['count_dev'])==50056 and not h['finalized'] for h in H[mode].values())
    provenance[mode]=dict(path=str(path),sha256=digest(path));del data
write('PROVENANCE.json',dict(H=provenance,protected=protected,scope=__doc__,advisory_mean_ratio=1.02,advisory_max_ratio=1.05))
torch.save(H,OUT/'H-used.pt')
results=[];states=[];preserves=[];documents=[];evaluation={}
with torch.inference_mode():
    config=Config.from_directory('/source');model=Model.from_config(config)
    prepare_ngram_table_for_conversion(dict(out_dir='/output/pack',ngram_file='/output/pack/ngram_embedding.safetensors'),config,model)
    docs=json.loads((ROOT/'capture-quarter-holdout/DATA.json').read_text())['rows'];assert len(docs)==8
    with safe_open('/output/capture-quarter-holdout/input-ids.safetensors') as reader:
        for i,doc in enumerate(docs):
            ids=_read_row(reader,f'input_ids.{i}').long()
            assert ids.numel()==doc['tokens']
            documents.append(ids);states.append(ids.clone());preserves.append({})
    # One module resident at a time. All eight documents share one weight load.
    for idx,m in enumerate(model.modules[:9]):
        assert time.monotonic()-started<420
        config.stc.begin_deferred_load()
        try:m.load(torch.device('cuda:0'))
        finally:config.stc.end_deferred_load()
        config.stc.close()
        for i,ids in enumerate(documents):
            pr=preserves[i];params=dict(attn_mode='flash_attn_nc',input_ids=ids,quant_preserve=pr);params.update(pr)
            x=m.prepare_for_device(states[i],params);y=m.forward(x,params)
            assert torch.isfinite(y).all()
            states[i]=cpu(y);preserves[i]=cpu(params['quant_preserve'])
            del x,y,params
        torch.save(dict(next_module_idx=idx+1,states=states,preserves=preserves,input_ids=documents),OUT/f'prefix-{idx+1:02d}.pt')
        m.unload();torch.cuda.empty_cache()
        write('PROGRESS.json',dict(phase='prefix',completed_module=idx,seconds=time.monotonic()-started))
    block=model.modules[9];assert block.key=='model.language_model.layers.6'
    config.stc.begin_deferred_load()
    try:block.load(torch.device('cuda:0'))
    finally:config.stc.end_deferred_load()
    config.stc.close()
    selected={m.key:m for m in block if isinstance(m,Linear) and (m.key.endswith('.linear_attn.out_proj') or m.key.endswith('.mlp.experts.0.down_proj'))}
    assert len(selected)==2
    terminal=next(k for k in selected if k.endswith('.experts.0.down_proj'))
    for i,doc in enumerate(docs):
        pr=copy.deepcopy(preserves[i]);params=dict(input_ids=documents[i],attn_mode='flash_attn_nc',capture={},activate_all_experts=True,quant_preserve=pr);params.update(pr)
        observed={}
        with capture_inputs(selected,terminal,observed):
            try:block.forward(block.prepare_for_device(states[i],params),params)
            except InputsCaptured:pass
            else:raise AssertionError('Missing input observer boundary')
        evaluation[doc['id']]={}
        for key,value in observed.items():
            value=value.reshape(-1,value.shape[-1]);rows=torch.linspace(0,len(value)-1,min(256,len(value)),device=value.device).long()
            evaluation[doc['id']][key]=cpu(value[rows])
        del observed,params
    torch.save(evaluation,OUT/'evaluation-inputs.pt');del states,preserves
    for key,linear in selected.items():
        original=linear.inner;kind='dense' if key.endswith('out_proj') else 'expert0';hkey='block.attn.output' if kind=='dense' else 'block.mlp.0.down';K=6 if kind=='dense' else 4
        raw,route=source_weight(linear);torch.save(raw,OUT/(kind+'-source-bf16.pt'))
        weight=raw.float().cuda();refs={};fp64={};ref_checks={};sample_indices={}
        for doc,values in evaluation.items():
            x=values[key].float();ref=(x.cuda()@weight).cpu();refs[doc]=ref
            ri=torch.linspace(0,len(x)-1,min(16,len(x))).long();ci=torch.linspace(0,ref.shape[1]-1,min(64,ref.shape[1])).long()
            truth=x[ri].double()@raw[:,ci].double();fp64[doc]=truth;sample_indices[doc]=(ri,ci)
            diff=ref[ri][:,ci].double()-truth
            ref_checks[doc]=dict(relative_l2=float(diff.norm()/truth.norm().clamp_min(1e-30)),max_abs=float(diff.abs().max()))
        torch.save(dict(reference=refs,fp64=fp64,indices=sample_indices),OUT/(kind+'-teachers.pt'))
        del weight
        rotation=None;qrng=torch.random.get_rng_state();grng=torch.cuda.get_rng_state()
        for mode in ('original','f32','original_f16'):
            assert time.monotonic()-started<540
            torch.random.set_rng_state(qrng);torch.cuda.set_rng_state(grng)
            hd=copy.deepcopy(H[mode][hkey]);hd['H_swap_device']=torch.device('cuda:0');hd['first_key']=key
            qa=make_quant_args(dict(apply_out_scales=True,codebook='mul1',work_dir=str(OUT)),2,K,[0])
            calls=[];chol=torch.linalg.cholesky
            def checked_cholesky(x,*args,**kwargs):
                item=dict(shape=list(x.shape),dtype=str(x.dtype),device=str(x.device));calls.append(item)
                try:r=chol(x,*args,**kwargs)
                except BaseException as error:
                    item['error']=repr(error)
                    torch.save(cpu(x),OUT/f'{kind}-{mode}-failed-cholesky-{len(calls)}.pt')
                    write(f'{kind}-{mode}-CHOLESKY.json',calls)
                    raise
                item['success']=True
                write(f'{kind}-{mode}-CHOLESKY.json',calls)
                return r
            torch.linalg.cholesky=checked_cholesky
            stream=torch.cuda.Stream();torch.cuda.synchronize();set_worker_stream(stream)
            try:
                with torch.cuda.stream(stream):proxy=linear.convert_exl3(hd,qa)
                stream.synchronize()
            finally:set_worker_stream(None);torch.linalg.cholesky=chol
            packed=linear.inner
            assert calls and all(c.get('success') for c in calls),'Cholesky retry/failure'
            assert not hd.get('q_fallback') and not qa.get('q_fallback') and math.isfinite(float(proxy)) and float(proxy)>=0
            assert math.isfinite(float(qa['g_scale'])) and packed.K==K
            signs=[cpu(torch.sign(getattr(packed,n))) for n in ('suh','svh')]
            if rotation is None:rotation=signs
            same=all(torch.equal(a,b) for a,b in zip(rotation,signs))
            packed_tensors=cpu(packed.get_tensors(key))
            torch.save(packed_tensors,OUT/f'{kind}-{mode}-packed.pt')
            for value in packed_tensors.values():
                if isinstance(value,torch.Tensor) and value.is_floating_point():assert torch.isfinite(value).all()
            for name in ('suh','svh','bias'):
                value=getattr(packed,name);assert value is None or torch.isfinite(value).all()
            assert same,'Paired rotation mismatch'
            errors=[]
            for doc,values in evaluation.items():
                y=cpu(packed.forward(values[key].cuda().unsqueeze(0),{}).float()).squeeze(0)
                item=dict(doc=doc,**error_metrics(y,refs[doc]));ri,ci=sample_indices[doc];truth=fp64[doc]
                item['fp64_sample_relative_error']=float((y[ri][:,ci].double()-truth).norm()/truth.norm().clamp_min(1e-30))
                errors.append(item)
            results.append(dict(key=key,kind=kind,mode=mode,K=K,proxy=float(proxy),g_scale=float(qa['g_scale']),fallback=False,rotation_signs_same=same,cholesky=calls,H_tokens=50056,source=route,teacher_fp64_checks=ref_checks,errors=errors))
            write('RESULTS.json',results);linear.inner=original
        linear.inner=original
    block.unload();config.stc.close()
    ratios={}
    for kind in ('dense','expert0'):
        base=next(r for r in results if r['kind']==kind and r['mode']=='original')
        for mode in ('f32','original_f16'):
            r=next(r for r in results if r['kind']==kind and r['mode']==mode)
            v=[a['relative_error']/b['relative_error'] for a,b in zip(r['errors'],base['errors'])]
            ratios[kind+'-'+mode]=dict(mean=sum(v)/len(v),maximum=max(v),mean_flag=sum(v)/len(v)>1.02,max_flag=max(v)>1.05)
    assert all(digest(Path(p))==h for p,h in protected.items())
    write('SUMMARY.json',dict(status='layer6_paired_quality_complete',results=results,ratios=ratios,seconds=time.monotonic()-started,production_changed=False,
        scope=__doc__,limitations='Two projections, fixed8 reused documents, sampled positions. SourceBF16 projection teacher; prefix uses converter loader plus fixedK8 ngram.50056 tokens does not guarantee full rank. Not whole-model/KLD quality.'))
