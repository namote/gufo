"""Bounded K8 conversion and explicit finite/readback checks, not model quality qualification."""
import json
from pathlib import Path
import struct
import time
import torch
from exllamav3.conversion.ngram import NgramSource,NgramTableReader,quantize_ngram_table

torch.set_grad_enabled(False)
torch.manual_seed(0)
path='/output/ngram-pilot/large-rows.safetensors'
assert not Path(path).exists()
start=time.monotonic()
stats=quantize_ngram_table(model_dir='/source',out_path=path,K=8,devices=[0],
                         chunk_rows=8192,limit_rows=32768,bias_sample_rows=8192,resume=False)
elapsed=time.monotonic()-start
source=NgramSource('/source'); reader=NgramTableReader(path)
assert reader.num_rows==32768 and reader.K==8
with open(path,'rb') as f:
    n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
assert Path(path).stat().st_size==8+n+max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
ids=torch.arange(32768)
ref=source.read_rows(0,32768).to('cuda').float()
out=reader.dequant(ids,'cuda',out_dtype=torch.float)
finite=bool(torch.isfinite(ref).all() and torch.isfinite(out).all())
rfn=float(torch.linalg.vector_norm(out-ref)/torch.linalg.vector_norm(ref))
result=dict(status='operator_pass' if finite and rfn<=0.01 else 'failed',rows=32768,K=8,
            finite=finite,readback_relative_frobenius_error=rfn,acceptance_bar=0.01,
            scope='sampled ngram operator only; full model quality remains unqualified',
            elapsed_seconds=elapsed,inline_stats=stats,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            bias_rows_per_head=8192,chunk_rows=8192)
Path('/output/ngram-pilot/NUMERICAL.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
reader.close()
assert result['status']=='operator_pass',result
