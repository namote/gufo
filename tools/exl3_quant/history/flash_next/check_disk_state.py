"""CPU-only exact row/state/checkpoint contract, with GPU package init excluded."""
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import tempfile
import torch

runtime = Path('/workspace/test/experiments/gufo_upstream_port_20261023/kyojin_runtime/source')
name = 'exllamav3.loader.safetensors_alt'
spec = importlib.util.spec_from_file_location(name, runtime/'exllamav3/loader/safetensors_alt.py')
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
from disk_state import DiskRows, install
from safetensors.torch import load_file

out = Path('/output/model-job-c03/repair-001-eager-state')
checks = []
with tempfile.TemporaryDirectory(dir=out) as directory:
    root = Path(directory)
    source = root/'source.safetensors'
    expected = [torch.arange(n*12, dtype=torch.float32).reshape(1,n,12).to(dtype)
                for n,dtype in [(8,torch.float16),(3,torch.bfloat16),(19,torch.float32)]]
    module.save_file({f'tensor.{i}':t for i,t in enumerate(expected)}, str(source))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = DiskRows(source, root)
    assert rows.tensor is None and not rows.changed
    for i in (2,0,1,0,2):
        assert torch.equal(rows[i], expected[i])
        assert rows[i].dtype == expected[i].dtype
    checks.append('lazy_ragged_fp16_bf16_fp32_exact')
    for i in range(len(rows)):
        # Same read/set/read/set pattern as upstream advance, without a GPU.
        rows[i] = rows[i].clone()
        rows[i] = (rows[i] + 0.5).contiguous()
        expected[i] = (expected[i] + 0.5).contiguous()
    rows[1] = torch.arange(14,dtype=torch.int64).reshape(1,7,2)
    expected[1] = rows[1].clone()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    checks.append('updates_and_shape_dtype_changes_source_immutable')
    target = root/'next.safetensors'
    rows.save(target)
    actual = load_file(str(target))
    assert all(torch.equal(actual[f'tensor.{i}'],v) and actual[f'tensor.{i}'].dtype==v.dtype
               for i,v in enumerate(expected))
    checks.append('official_reader_streamed_checkpoint_exact')
    source.rename(root/'old.safetensors')
    target.rename(source)
    rows.close()
    resumed = DiskRows(source,root)
    assert all(torch.equal(resumed[i],v) for i,v in enumerate(expected))
    resumed.close()
    checks.append('rename_resume_exact')
    # Unmodified rows remain readable through an open checkpoint handle after rename/unlink.
    stale = DiskRows(source,root)
    source.unlink()
    assert all(torch.equal(stale[i],v) for i,v in enumerate(expected))
    stale.close()
    checks.append('source_handle_survives_checkpoint_rotation')
    bad = root/'bad.safetensors'
    module.save_file({'tensor.1':torch.ones(2)},str(bad))
    try:
        DiskRows(bad,root)
        raise AssertionError('Invalid checkpoint accepted')
    except ValueError:
        pass
    checks.append('noncontiguous_keys_rejected')

# Metadata-only open of actual 76GiB checkpoint proves no eager payload allocation.
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
full = DiskRows('/output/work/ckpt/state.safetensors',out)
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
assert len(full)==2309 and full.tensor is None
assert (after-before)*1024 < 32*2**20
full.close()
checks.append('actual_76gib_checkpoint_open_without_payload')
result = dict(status='pass',checks=checks,full_checkpoint_open_rss_increase_bytes=(after-before)*1024,
              gpu_used=False)
(out/'CPU-CHECK.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
