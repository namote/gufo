"""Same-K8, exact-output comparison of existing kernel families and chunk sizes."""
import json
import os
from pathlib import Path
import time
import torch
from exllamav3.conversion.ngram import NgramSource,NgramTableReader,quantize_rows
from exllamav3.modules.quant.exl3_lib.quantize import get_temp_buffers

torch.set_grad_enabled(False)
source=NgramSource('/source');reader=NgramTableReader('/output/ngram-pilot/large-rows.safetensors')
# Mix contiguous stripes from different hash heads; reuse exactly these inputs in each arm.
stripes=[];biases=[]
for j in range(16):
    h=j%source.num_heads;lo=int(source.head_offsets[h])+8192*(j//source.num_heads)
    stripes.append(source.read_rows(lo,lo+8192).clone())
    biases.append(reader.head_bias[h].expand(8192,-1).clone())
rows=torch.cat(stripes).to('cuda');bias=torch.cat(biases).to('cuda')
assert torch.isfinite(rows).all() and torch.isfinite(bias).all()
results=[];baseline=None
for mode,chunk in [('base',8192),('base',131072),('opt',32768),('opt',131072)]:
    os.environ['EXL3_QT_KERNEL']=mode
    get_temp_buffers.cache_clear()
    quantize_rows(rows[:512],bias[:512],8)
    torch.cuda.synchronize()
    start=time.monotonic();packed=[];error=0.0;norm=0.0
    for lo in range(0,len(rows),chunk):
        q,deq=quantize_rows(rows[lo:lo+chunk],bias[lo:lo+chunk],8)
        packed.append(q)
        error+=(deq-rows[lo:lo+chunk].float()).square().sum()
        norm+=rows[lo:lo+chunk].float().square().sum()
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    q=torch.cat(packed)
    if baseline is None:baseline=q.clone()
    equal=bool(torch.equal(q,baseline))
    rfn=float(torch.sqrt(error/norm))
    row=dict(mode=mode,chunk_rows=chunk,rows=len(rows),seconds=elapsed,
             rows_per_second=len(rows)/elapsed,projected_full_hours=320001536/len(rows)*elapsed/3600,
             bit_exact_to_base=equal,rfn=rfn,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    results.append(row);print(json.dumps(row),flush=True)
    Path('/output/ngram-speed/RESULTS.json').write_text(json.dumps(results,indent=2))
    assert torch.isfinite(deq).all() and rfn<=0.01
reader.close()
