import concurrent.futures
import json
import os
from pathlib import Path
import time
import torch
from safetensors.torch import save_file
from exllamav3.conversion.ngram import NgramSource,NgramTableReader,quantize_rows
from exllamav3.modules.quant.exl3_lib.quantize import get_temp_buffers,set_worker_stream

torch.set_grad_enabled(False)
root=Path('/output/ngram-streams');root.mkdir(exist_ok=True)
source=NgramSource('/source');reader=NgramTableReader('/output/ngram-pilot/large-rows.safetensors')
rows=[];bias=[]
for j in range(16):
    h=j%source.num_heads;lo=int(source.head_offsets[h])+8192*(j//source.num_heads)
    rows.append(source.read_rows(lo,lo+8192).clone());bias.append(reader.head_bias[h].expand(8192,-1).clone())
rows=torch.cat(rows).to('cuda');bias=torch.cat(bias).to('cuda')
edge=torch.stack([torch.zeros(160),torch.ones(160),-torch.ones(160),
                  torch.full((160,),1e-8),torch.full((160,),-1e-8),
                  torch.linspace(-3,3,160),torch.arange(160)%2*2.-1.]).to('cuda')
edge_base=None
for mode in ['base','opt']:
    os.environ['EXL3_QT_KERNEL']=mode;get_temp_buffers.cache_clear()
    q,d=quantize_rows(edge,torch.zeros_like(edge),8)
    assert torch.isfinite(d).all()
    if edge_base is None:edge_base=q.clone()
    else:assert torch.equal(q,edge_base),'edge-case packed mismatch'
os.environ['EXL3_QT_KERNEL']='opt';get_temp_buffers.cache_clear()
quantize_rows(rows[:512],bias[:512],8);torch.cuda.synchronize()
baseline=None;results=[]
for workers in [1,2,4]:
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    def run(worker):
        with torch.inference_mode():
            s=torch.cuda.Stream();set_worker_stream(s)
            outputs=[]
            try:
                with torch.cuda.stream(s):
                    for index in range(worker,4,workers):
                        lo=index*32768;q,d=quantize_rows(rows[lo:lo+32768],bias[lo:lo+32768],8)
                        outputs.append((index,q))
                    s.synchronize()
                return outputs
            finally:set_worker_stream(None)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        parts=[p for group in pool.map(run,range(workers)) for p in group]
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    q=torch.cat([p[1] for p in sorted(parts,key=lambda x:x[0])])
    if baseline is None:
        baseline=q.clone()
        save_file({'packed':q.cpu()},str(root/'opt-reference.safetensors'))
    equal=bool(torch.equal(q,baseline));assert equal
    r=dict(workers=workers,chunk_rows=32768,rows=len(rows),seconds=elapsed,
           rows_per_second=len(rows)/elapsed,compute_projection_hours=320001536/len(rows)*elapsed/3600,
           exact_to_opt_reference=equal,edge_case_exact_to_base=True,
           peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    results.append(r);print(json.dumps(r),flush=True)
    (root/'RESULTS.json').write_text(json.dumps(results,indent=2))
reader.close()
