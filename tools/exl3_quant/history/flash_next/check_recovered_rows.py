"""Read representative recovered real rows through the converter's installed hook."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import torch
runtime=Path('/workspace/test/experiments/gufo_upstream_port_20261023/kyojin_runtime/source')
name='exllamav3.loader.safetensors_alt'
spec=importlib.util.spec_from_file_location(name,runtime/'exllamav3/loader/safetensors_alt.py')
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
from disk_state import install,_read_row
out=Path('/output/model-job-c03/repair-002-checkpoint')
source=next(Path('/output/model-job-c03/module-004/row-scratch').glob('state-rows-*'))
with tempfile.TemporaryDirectory(dir=out) as directory:
    cm=SimpleNamespace(load_tensor=lambda *a:None,save_tensor=lambda *a:None)
    stores=install(cm,directory)
    rows=cm.load_tensor('ckpt/state.safetensors',dict(work_dir='/output/work',cal_rows=2309))
    samples=[0,next(i for i,(_,shape) in enumerate(rows.infos) if shape[1]==8192),2308]
    for i in samples:
        with module.safe_open(str(source/f'{i}.safetensors')) as reader:
            expected=_read_row(reader,'tensor')
        actual=rows[i]
        assert torch.equal(actual,expected) and actual.dtype==expected.dtype
    for store in stores:store.close()
result=dict(status='pass',rows=2309,sampled_rows=samples,exact=True,gpu_used=False)
(out/'RECOVERED-READ.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
