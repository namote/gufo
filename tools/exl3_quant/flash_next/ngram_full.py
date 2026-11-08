"""Resumable BF16 -> K8 conversion, rejecting invalid rows rather than sanitizing them."""
import json
import math
import os
from pathlib import Path
import struct
import threading
import time
import torch
from exllamav3.conversion import ngram as ng

torch.set_grad_enabled(False);torch.manual_seed(0)
assert os.environ.get('EXL3_QT_KERNEL')=='opt'
from exllamav3.modules.quant.exl3_lib.quantize import set_worker_stream
root=Path('/output/ngram-full');root.mkdir(exist_ok=True)
path=Path('/output/pack/ngram_embedding.safetensors');path.parent.mkdir(exist_ok=True)
original=ng.quantize_rows
calls=0;maximum=0.0
local=threading.local()
progress_lock=threading.Lock()

def checked_rows(rows,bias,K,cs=None,cs_search=1):
    global calls,maximum
    if not hasattr(local,'stream'):
        local.stream=torch.cuda.Stream()
        set_worker_stream(local.stream)
    local.stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(local.stream):
        assert K==8 and torch.isfinite(rows).all() and torch.isfinite(bias).all()
        packed,deq=original(rows,bias,K,cs,cs_search)
        assert torch.isfinite(deq).all()
        norm=torch.linalg.vector_norm(rows.float())
        error=torch.linalg.vector_norm(deq.float()-rows.float())
        rfn=float(error/norm) if norm>0 else float(error)
        local.stream.synchronize()
    assert math.isfinite(rfn) and rfn<=0.01, f'K8 chunk error {rfn} exceeds frozen operator bound'
    with progress_lock:
        calls+=1;maximum=max(maximum,rfn)
        if calls%32==0:
            tmp=root/'PROGRESS.tmp'
            tmp.write_text(json.dumps(dict(updated=time.time(),chunks_this_window=calls,
                maximum_chunk_rfn=maximum,output_bytes=path.stat().st_size if path.exists() else 0)))
            tmp.replace(root/'PROGRESS.json')
    return packed,deq

ng.quantize_rows=checked_rows
stats=ng.quantize_ngram_table(model_dir='/source',out_path=str(path),K=8,devices=[0,0],
                            chunk_rows=32768,bias_sample_rows=131072,resume=True)
source=ng.NgramSource('/source');reader=ng.NgramTableReader(str(path))
assert reader.K==8 and reader.num_rows==source.num_rows==320001536
with path.open('rb') as f:
    n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n))
assert path.stat().st_size==8+n+max(v['data_offsets'][1] for k,v in h.items() if k!='__metadata__')
ids=torch.randint(0,reader.num_rows,(4096,)).unique().sort().values
ref=torch.cat([source.read_rows(i,i+1) for i in ids.tolist()]).to('cuda').float()
deq=reader.dequant(ids,'cuda',out_dtype=torch.float)
assert torch.isfinite(ref).all() and torch.isfinite(deq).all()
rfn=float(torch.linalg.vector_norm(deq-ref)/torch.linalg.vector_norm(ref))
assert math.isfinite(rfn) and rfn<=0.01
result=dict(status='complete_operator_verified',source='/source',source_revision='de4b8e4d43b917e7706784d8bb445c9af86a3540',
            source_dtype='BF16',K=8,rows=reader.num_rows,bytes=path.stat().st_size,
            sampled_readback_rows=len(ids),sampled_readback_rfn=rfn,operator_bound=0.01,
            kernel='opt',streams=2,chunk_rows=32768,
            current_window_stats=stats,maximum_chunk_rfn_this_window=maximum,finished=time.time())
(root/'RESULT.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
reader.close()
