"""One operator screen: real gate/up+dense inputs, Lt algorithms and WMMA.

Never installs a converter hook. Preserve operands and original outputs for reuse.
Different FP32 reduction rounding is measured against a sampled CPU FP64 oracle.
"""
import hashlib,json,time
from pathlib import Path
import torch
from exllamav3 import Config,Model
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.fp16 import LinearFP16
from exllamav3.loader.safetensors_alt import safe_open
from disk_state import _read_row
from capture_inputs import capture_inputs,InputsCaptured
from capture_forward_ops import Native,Unsupported
O=Path('/output');R=O/'capture-forward';started=time.monotonic()
def write(n,x):(R/n).write_text(json.dumps(x,indent=2))
def compare(a,b):
    d=a.double()-b.double()
    return dict(value_exact=torch.equal(a.double(),b.double()),exact=a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)),finite=bool(torch.isfinite(b).all()),max_abs=float(d.abs().max()),relative_l2=float(d.norm()/a.double().norm().clamp_min(1e-30)))
def timing(fn):
    fn();torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    t=time.perf_counter();a.record()
    for _ in range(5):fn()
    b.record();b.synchronize()
    return dict(gpu_ms=a.elapsed_time(b)/5,wall_ms=(time.perf_counter()-t)*200)
build=json.loads((R/'BUILD.json').read_text())
assert build['source_sha256']==hashlib.sha256(Path('/task/capture_forward.hip').read_bytes()).hexdigest()
assert build['binary_sha256']==hashlib.sha256((R/'forward.so').read_bytes()).hexdigest()
assert not (R/'SUMMARY.json').exists(),'Reuse saved screen'
native=Native(R/'forward.so');records=[]
with torch.inference_mode():
    # WMMA exact test failed previously: retained, not relaxed or rerun.
    rejected=json.loads((R/'WMMA-REVIEW.json').read_text())
    assert rejected['status']=='unqualified_exact_analytic'
    analytic=[]
    gen=torch.Generator().manual_seed(917)
    xc=torch.randint(-2,3,(64,256),generator=gen).half()
    wc=torch.randint(-2,3,(2,256,128),generator=gen).half()
    golden=torch.stack([xc.double()@v.double() for v in wc])
    x=xc.cuda();w=wc.cuda()
    for dtype in (torch.float32,torch.float16):
        storage=torch.full((2*64*128+32,),1234.,device='cuda',dtype=dtype)
        y=storage[:2*64*128].view(2,64,128)
        originals=[LinearFP16(256,128,w[i],None,full_in_features=256,full_out_features=128,out_dtype=dtype) for i in range(2)]
        ref=torch.stack([linear.forward(x,{}) for linear in originals]).cpu()
        torch.save(dict(x=xc,w=wc,golden=golden,baseline=ref),R/f'lt-aligned-baseline-{str(dtype)}.pt')
        assert torch.isfinite(ref).all()
        record=dict(dtype=str(dtype),baseline_FP64=compare(golden,ref),candidates=[])
        analytic.append(record)
        try:plan=native.create(x,w,y)
        except Unsupported as e:record['unsupported']=str(e);continue
        for algorithm in range(plan[2]):
            native.run(plan,algorithm,x,w,y);torch.cuda.synchronize()
            observed=y.cpu().clone()
            torch.save(dict(x=xc,w=wc,golden=golden,baseline=ref,observed=observed,tail=storage[-32:].cpu()),R/f'lt-aligned-analytic-{str(dtype)}-{algorithm}.pt')
            error=compare(golden,observed)
            record['candidates'].append(dict(algorithm=algorithm,error_FP64=error,tail_canary=bool((storage[-32:]==1234).all())))
            write('LT-ANALYTIC.json',analytic)
            assert error['finite'] and (storage[-32:]==1234).all()
        native.close()
    write('LT-ANALYTIC.json',analytic)
    del x,w,y,storage,originals
    if (R/'inputs.pt').exists():data=torch.load(R/'inputs.pt',map_location='cpu',weights_only=True)
    else:
        cfg=Config.from_directory('/source');model=Model.from_config(cfg);block=model.modules[2]
        assert block.key=='model.language_model.layers.0'
        cfg.stc.begin_deferred_load()
        try:block.load(torch.device('cuda:0'))
        finally:cfg.stc.end_deferred_load()
        cfg.stc.close()
        up,gate=block.mlp.ups[0],block.mlp.gates[0];dense=block.attn.o_proj
        assert up.inner.out_dtype==gate.inner.out_dtype==torch.float16
        assert dense.inner.out_dtype==torch.float32
        data={'expert_w':torch.stack([up.inner.weight.cpu(),gate.inner.weight.cpu()]).contiguous(),
              'dense_w':dense.inner.weight.cpu().unsqueeze(0).contiguous()}
        for length,index in [(512,4),(8192,364)]:
            with safe_open('/task/c03-input-ids.safetensors') as f:ids=_read_row(f,f'input_ids.{index}').long()
            with safe_open('/output/model-job-c03/repair-002-checkpoint/checkpoint-idx2/state.safetensors') as f:state=_read_row(f,f'tensor.{index}')
            assert state.shape[1]==length
            params=dict(input_ids=ids,attn_mode='flash_attn_nc',capture={},activate_all_experts=True,quant_preserve={});inputs={}
            keys={up.key,gate.key,dense.key}
            with capture_inputs(keys,gate.key,inputs):
                try:block.forward(block.prepare_for_device(state,params),params)
                except InputsCaptured:pass
                else:raise AssertionError('Observer did not terminate at gate')
            assert params['capture']=={} and '_cap_pending' not in params
            assert torch.equal(inputs[up.key],inputs[gate.key])
            data[f'expert_x{length}']=inputs[up.key].reshape(length,-1).cpu()
            if length==512:data['dense_x512']=inputs[dense.key].reshape(length,-1).cpu()
            del inputs,state,params
        torch.save(data,R/'inputs.pt');block.unload();cfg.stc.close();del block,model;torch.cuda.empty_cache()
    cases=[('expert_33_B2',data['expert_x512'][:33],data['expert_w']),
           ('expert_512_B1',data['expert_x512'],data['expert_w'][:1]),
           ('expert_512_B2',data['expert_x512'],data['expert_w']),
           ('expert_8192_B2',data['expert_x8192'],data['expert_w']),
           ('dense_512',data['dense_x512'],data['dense_w'])]
    for name,xc,wc in cases:
        assert time.monotonic()-started<500
        x=xc.cuda().contiguous();w=wc.cuda().contiguous();b,k,n=w.shape;m=x.shape[0]
        assert x.dtype==w.dtype==torch.float16
        assert torch.isfinite(x).all() and torch.isfinite(w).all(),'Nonfinite original operand'
        dtype=torch.float16 if name.startswith('expert') else torch.float32
        linears=[LinearFP16(k,n,w[i],None,full_in_features=k,full_out_features=n,out_dtype=dtype) for i in range(b)]
        y=torch.empty((b,m,n),device='cuda',dtype=dtype)
        def baseline():
            return [linear.forward(x,{}) for linear in linears]
        ref=torch.stack(baseline());torch.cuda.synchronize();saved_ref=ref.cpu().clone()
        torch.save(dict(reference=saved_ref),R/f'{name}-control.pt')
        assert torch.isfinite(saved_ref).all(),'Nonfinite baseline'
        ri=torch.linspace(0,m-1,min(m,16)).long();ci=torch.linspace(0,n-1,min(n,16)).long()
        oracle=torch.stack([xc[ri].double()@v[:,ci].double() for v in wc])
        original_error=compare(oracle,saved_ref[:,ri][:,:,ci])
        entry=dict(name=name,shape=[b,m,k,n],output_dtype=str(dtype),baseline_sample_FP64=original_error,candidates=[])
        records.append(entry)
        modes=[]
        try:
            plan=native.create(x,w,y)
            modes += [(f'lt_{i}',lambda i=i:native.run(plan,i,x,w,y)) for i in range(plan[2])]
        except Unsupported as e:entry['lt_unsupported']=str(e)
        for label,fn in modes:
            fn();torch.cuda.synchronize();candidate=y.cpu().clone()
            difference=compare(saved_ref,candidate);fp64=compare(oracle,candidate[:,ri][:,:,ci])
            torch.save(candidate,R/f'{name}-{label}.pt')
            rec=dict(mode=label,difference=difference,sample_FP64=fp64)
            entry['candidates'].append(rec);write('PROGRESS.json',records)
            assert difference['finite'],'Nonfinite candidate'
            fn();torch.cuda.synchronize();rec['repeat_exact']=torch.equal(candidate.view(torch.uint8),y.cpu().view(torch.uint8))
            rec['ABBA']=[timing(baseline),timing(fn),timing(fn),timing(baseline)]
            write('PROGRESS.json',records)
        native.close()
        del x,w,y,ref,linears
    native.close()
write('SUMMARY.json',dict(status='forward_operator_screen_complete',excluded_candidates={'wmma':'unqualified_exact_analytic; see WMMA-REVIEW.json'},seconds=time.monotonic()-started,results=records,
    scope='Real layer0 operator operands only; excludes weight packing, capture/H/normal forward. Not full model quality or speed. FP16 operands and actual output dtype retained (expert half, dense float). Baseline allocates outputs each call, candidate reuses output: timings include allocation/launch-gap differences, not pure GEMM. Small weights are cache-hot.',production_changed=False))
