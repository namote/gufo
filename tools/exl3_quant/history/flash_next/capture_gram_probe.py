"""Operator-only Gram candidates: direct FP32 Lt, original FP16 Lt/WMMA.

No subsampling of production calibration and no production installation. Capture
two real inputs once, save inputs/control, then measure candidates on those bytes.
Different reduction order is reported, never silently accepted as exact.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from exllamav3 import Config, Model
from exllamav3.modules import Linear
from exllamav3.modules.quant import fp16, exl3
from exllamav3.loader.safetensors_alt import safe_open
from disk_state import _read_row
from capture_inputs import capture_inputs, InputsCaptured

ROOT=Path('/output'); OUT=ROOT/'capture-gram'
assert not (OUT/'SUMMARY.json').exists()
assert os.environ['EXL3_HIP_F32OUT_VIA_F16']=='0'
assert not fp16._f32_via_f16 and not exl3._f32_via_f16
build=json.loads((OUT/'BUILD.json').read_text())
assert build['source_sha256']==hashlib.sha256(Path('/task/capture_gram.hip').read_bytes()).hexdigest()
assert build['binary_sha256']==hashlib.sha256((OUT/'gram.so').read_bytes()).hexdigest()
mounts=[s.split() for s in Path('/proc/self/mountinfo').read_text().splitlines()]
for path in ('/output/work','/output/pack','/output/model-job-c03/repair-002-checkpoint'):
    assert any(p[4]==path and 'ro' in p[5].split(',') for p in mounts)
started=time.monotonic()
def write(name,value):
    tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(OUT/name)
def stamp(p):
    s=p.stat();return [s.st_size,s.st_mtime_ns,s.st_ino]
protected={str(p):stamp(p) for p in (ROOT/'work/ckpt/job.json',ROOT/'pack/ngram_embedding.safetensors',
    ROOT/'model-job-c03/repair-002-checkpoint/checkpoint-idx2/state.safetensors',Path('/task/c03-input-ids.safetensors'))}
lib=C.CDLL(str(OUT/'gram.so'))
lib.gram_launch.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_void_p];lib.gram_launch.restype=C.c_int
lib.gram_lt_create.argtypes=[C.c_int,C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_int)];lib.gram_lt_create.restype=C.c_void_p
lib.gram_lt_run.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p];lib.gram_lt_run.restype=C.c_int
lib.gram_lt_destroy.argtypes=[C.c_void_p];lib.gram_lt_destroy.restype=None
def stream():return torch.cuda.current_stream().cuda_stream
def wmma(x,y):
    assert x.dtype==torch.float16 and x.is_contiguous() and x.ndim==3 and x.is_cuda
    assert y.dtype==torch.float32 and y.is_contiguous() and y.shape==(x.shape[0],x.shape[2],x.shape[2])
    assert lib.gram_launch(x.data_ptr(),y.data_ptr(),*x.shape,stream())==0
def compare(ref,y):
    assert torch.isfinite(ref).all() and torch.isfinite(y).all()
    delta=y.double()-ref.double()
    return dict(exact=torch.equal(ref.view(torch.uint8),y.view(torch.uint8)),
        max_abs=float(delta.abs().max()),relative_l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(ref.double()).clamp_min(1e-30)))
def timing(fn):
    fn();torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    wall=time.perf_counter();start.record()
    for _ in range(3):fn()
    end.record();end.synchronize()
    return dict(gpu_ms=start.elapsed_time(end)/3,wall_ms=(time.perf_counter()-wall)*1000/3)

class UnsupportedLt(RuntimeError):
    pass
results=[]; traces=[]
@torch.inference_mode()
def test_case(name,cpu_x,trace=False):
    assert time.monotonic()-started<520
    x=cpu_x.to('cuda').contiguous();assert x.dtype in (torch.float16,torch.float32)
    assert torch.isfinite(x).all(), 'Finite real input required; do not mask away bad calibration'
    xf=x.float();batch,rows,dim=x.shape
    ref=torch.bmm(xf.transpose(1,2),xf)
    torch.save(dict(x=cpu_x,reference=ref.cpu()),OUT/(name+'-control.pt'))
    y=torch.empty_like(ref)
    baseline=lambda:torch.bmm(xf.transpose(1,2),xf,out=y)
    pos=torch.linspace(0,dim-1,min(8,dim),device=x.device).long()
    xx=xf[:min(2,batch),:,pos].double()
    teacher=xx.transpose(1,2)@xx
    def sampled_error(z):return compare(teacher,z[:min(2,batch)][:,pos][:,:,pos].double())
    entry=dict(name=name,shape=list(x.shape),dtype=str(x.dtype),baseline_sample_fp64=sampled_error(ref),candidates=[])
    if trace:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            baseline();torch.cuda.synchronize()
        path=OUT/(name+'-bmm-trace.json');prof.export_chrome_trace(str(path))
        trace_data=json.loads(path.read_text())
        kernels=[dict(name=e.get('name'),duration_us=e.get('dur')) for e in trace_data['traceEvents'] if e.get('cat')=='kernel']
        entry['torch_bmm_kernels']=kernels
    candidates=[];plans=[]
    if x.dtype==torch.float16:candidates.append(('wmma_original_f16',lambda:wmma(x,y)))
    for label,source in [('lt_f32',xf)]+([('lt_original_f16',x)] if x.dtype==torch.float16 else []):
        status=C.c_int();found=C.c_int()
        plan=lib.gram_lt_create(batch,rows,dim,int(source.dtype==torch.float16),C.byref(status),C.byref(found))
        if not plan:
            info=dict(mode=label,status='unsupported' if status.value in (-2,7,8) else 'failed',code=status.value)
            entry['candidates'].append(info);write('PROGRESS.json',results+[entry])
            assert info['status']=='unsupported',info
            continue
        plans.append(plan)
        for algo in range(found.value):
            def run(p=plan,a=algo,src=source,mode=label):
                code=lib.gram_lt_run(p,a,src.data_ptr(),y.data_ptr(),stream())
                if code in (7,8):raise UnsupportedLt((mode,a,code))
                if code:
                    write('FAILURE.json',dict(case=name,mode=mode,algorithm=a,lt_status=code))
                    raise RuntimeError((mode,a,code))
            candidates.append((label+'_'+str(algo),run))
    try:
        for label,fn in candidates:
            try:fn();torch.cuda.synchronize()
            except UnsupportedLt as error:
                entry['candidates'].append(dict(mode=label,status='unsupported_at_execution',detail=str(error)))
                write('PROGRESS.json',results+[entry]);continue
            # Preserve output before finite/difference checks, including a failed output.
            torch.save(y.cpu(),OUT/(name+'-'+label+'.pt'))
            try:diff=compare(ref,y);quality=sampled_error(y)
            except BaseException as error:
                write('FAILURE.json',dict(case=name,mode=label,error=repr(error),output=name+'-'+label+'.pt'))
                raise
            spans=[timing(f) for f in (baseline,fn,fn,baseline)]
            a=[spans[i]['gpu_ms'] for i in (0,3)];b=[spans[i]['gpu_ms'] for i in (1,2)]
            candidate=dict(mode=label,difference=diff,sample_fp64=quality,timing_ABBA=spans,
                operator_exact_and_faster=diff['exact'] and max(b)<.95*min(a) and (label!='wmma_original_f16' or all(v['exact_fp64'] for v in analytic)),
                wmma_analytic_exact=all(v['exact_fp64'] for v in analytic) if label=='wmma_original_f16' else None)
            if trace and label in ('wmma_original_f16','lt_f32_0','lt_original_f16_0'):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    fn();torch.cuda.synchronize()
                path=OUT/(name+'-'+label+'-trace.json');prof.export_chrome_trace(str(path))
                td=json.loads(path.read_text());candidate['kernels']=[e.get('name') for e in td['traceEvents'] if e.get('cat')=='kernel']
            entry['candidates'].append(candidate)
            write('PROGRESS.json',results+[entry])
    finally:
        torch.cuda.synchronize()
        for plan in plans:lib.gram_lt_destroy(plan)
    results.append(entry);write('RESULTS.json',results)

with torch.inference_mode():
    torch.manual_seed(437)
    analytic=[]
    # Exact dyadic products: independent CPU FP64 oracle catches layout/tail bugs.
    for b,r,d in ((1,1,1),(3,17,19),(2,33,32),(2,64,64)):
        cpu=(torch.randint(-8,9,(b,r,d)).float()/8).half()
        gold=cpu.double().transpose(1,2)@cpu.double()
        x=cpu.cuda();back=torch.full((b*d*d+32,),float('nan'),device='cuda')
        y=back[:b*d*d].view(b,d,d);wmma(x,y);torch.cuda.synchronize()
        torch.save(dict(input=cpu,reference=gold,output=y.cpu(),canary=back[b*d*d:].cpu()),OUT/f'analytic-{b}-{r}-{d}.pt')
        assert torch.isfinite(y).all() and torch.isnan(back[b*d*d:]).all(), 'Invalid output or tail write'
        observed=y.cpu().double();is_exact=torch.equal(observed,gold)
        analytic.append(dict(shape=[b,r,d],exact_fp64=is_exact,tail_canary=True,max_abs=float((observed-gold).abs().max()),
            status='pass' if is_exact else 'numeric_mismatch_not_qualified'))
        write('ANALYTIC.json',analytic)
    write('ANALYTIC.json',analytic)
    config=Config.from_directory('/source');model=Model.from_config(config);block=model.modules[2]
    assert block.key=='model.language_model.layers.0'
    config.stc.begin_deferred_load()
    try:block.load(torch.device('cuda:0'))
    finally:config.stc.end_deferred_load()
    config.stc.close()
    down=[f'{block.key}.mlp.experts.{i}.down_proj' for i in range(64)]
    dense=block.key+'.linear_attn.out_proj';keys=set(down+[dense]);saved=[]
    for length,index in ((512,4),(8192,364)):
        with safe_open('/task/c03-input-ids.safetensors') as reader:ids=_read_row(reader,f'input_ids.{index}').long()
        with safe_open('/output/model-job-c03/repair-002-checkpoint/checkpoint-idx2/state.safetensors') as reader:state=_read_row(reader,f'tensor.{index}')
        assert ids.shape==(1,length) and state.shape==(1,length,4,2560)
        params=dict(input_ids=ids,attn_mode='flash_attn_nc',capture={},activate_all_experts=True,quant_preserve={})
        inputs={}
        with capture_inputs(keys,down[-1],inputs):
            try:block.forward(block.prepare_for_device(state,params),params)
            except InputsCaptured:pass
            else:raise AssertionError('Input capture did not stop')
        assert params['capture']=={} and '_cap_pending' not in params
        expert=torch.stack([inputs[k].reshape(length,640).cpu() for k in down])
        assert expert.dtype==torch.float16
        saved.append((f'expert64_L{length}',expert))
        if length==512:saved.append(('dense_L512',inputs[dense].reshape(1,length,-1).cpu()))
        del inputs,state,params
    torch.save(dict(saved),OUT/'real-inputs.pt')
    block.unload();config.stc.close();torch.cuda.empty_cache()
    write('BACKEND.json',dict(torch=torch.__version__,git=torch.version.git_version,hip=torch.version.hip,
        blas_preference=str(torch.backends.cuda.preferred_blas_library()),float32_precision=torch.get_float32_matmul_precision(),
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,device=torch.cuda.get_device_name(),
        source_dtype_policy='Never cast FP32 input down. WMMA/half Lt only for already FP16 inputs; outputs accumulate FP32.',
        loaded_blas_libraries=sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'blas' in line and '/' in line))))
    for name,data in saved:test_case(name,data,trace=name=='expert64_L512')
    assert all(stamp(Path(p))==v for p,v in protected.items())
    write('SUMMARY.json',dict(status='gram_operator_screen_complete',elapsed_seconds=time.monotonic()-started,
        results=results,analytic=analytic,protected_stats=protected,production_changed=False,
        wmma_analytic_exact=all(v['exact_fp64'] for v in analytic),
        scope='Two layer0 real rows,64 experts and one dense input. Operator timings exclude packing/masking/H addition. No full-model quality or deployment qualification.'))
