"""Small exact GPU staging check, without loading the model or rerunning baseline."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import torch

runtime=Path('/workspace/test/experiments/gufo_upstream_port_20261023/kyojin_runtime/source')
name='exllamav3.loader.safetensors_alt'
spec=importlib.util.spec_from_file_location(name,runtime/'exllamav3/loader/safetensors_alt.py')
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
from disk_state import DiskRows
from safetensors.torch import load_file

out=Path('/output/model-job-c03/repair-001-eager-state')
with tempfile.TemporaryDirectory(dir=out) as directory:
    root=Path(directory)
    original=DiskRows('/output/work/ckpt/state.safetensors',root)
    long_index=next(i for i,(_,shape) in enumerate(original.infos) if shape[1]==8192)
    expected=[original[i].clone() for i in (0,long_index)]
    original.close()
    path=root/'input.safetensors'
    module.save_file({f'tensor.{i}':t for i,t in enumerate(expected)},str(path))
    rows=DiskRows(path,root)
    # First row follows regular forward's GPU assignment then final CPU output.
    rows[0]=rows[0].to('cuda')
    assert rows[0].is_cuda and not rows.changed
    rows[0]=rows[0].clone().cpu()
    assert torch.equal(rows[0],expected[0])
    # A skipped head row only receives GPU assignment; eviction must still persist.
    rows[1]=rows[1].to('cuda')
    assert rows[1].is_cuda
    assert torch.equal(rows[0],expected[0])
    assert rows.changed=={0,1}
    rows.save(root/'next.safetensors')
    actual=load_file(str(root/'next.safetensors'))
    assert all(torch.equal(actual[f'tensor.{i}'],t) for i,t in enumerate(expected))
    rows.close()
torch.cuda.synchronize()
result=dict(status='pass',source_rows=[0,long_index],checks=['gpu_temporary_cpu_final_exact',
            'gpu_only_row_eviction_exact','streamed_checkpoint_exact'],
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
(out/'GPU-CHECK.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
