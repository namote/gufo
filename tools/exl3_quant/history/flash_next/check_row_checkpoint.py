"""CPU contracts for zero-copy checkpoint commit and owned cleanup races."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import torch

runtime=Path('/workspace/test/experiments/gufo_upstream_port_20261023/kyojin_runtime/source')
name='exllamav3.loader.safetensors_alt'
spec=importlib.util.spec_from_file_location(name,runtime/'exllamav3/loader/safetensors_alt.py')
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
from disk_state import DiskRows,install
from row_checkpoint import open_rows,tensor_header
from owned_cleanup import cleanup_owned
out=Path('/output/model-job-c03/repair-002-checkpoint')
checks=[]
with tempfile.TemporaryDirectory(dir=out) as directory:
    root=Path(directory);source=root/'source.safetensors'
    expected=[torch.arange(n*12,dtype=torch.float32).reshape(1,n,12).to(dtype)
              for n,dtype in [(8,torch.float16),(3,torch.bfloat16),(19,torch.float32)]]
    module.save_file({f'tensor.{i}':t for i,t in enumerate(expected)},str(source))
    original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    rows=DiskRows(source,root);rows[1]=rows[1]+1;expected[1]=expected[1]+1
    new=root/'ckpt_new';new.mkdir();rows.save_checkpoint(new/'state.safetensors')
    paths,infos=open_rows(new/'state.safetensors')
    assert all(paths[i].stat().st_ino==rows._path(i).stat().st_ino for i in range(3))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==original_hash
    rows.close();new.rename(root/'ckpt');source.unlink()
    rows=DiskRows(root/'ckpt/state.safetensors',root)
    assert all(torch.equal(rows[i],t) and rows[i].dtype==t.dtype for i,t in enumerate(expected))
    checks.append('aggregate_migration_partial_update_hardlink_exact')
    # Private input hardlinks survive original checkpoint rename and deletion.
    (root/'ckpt').rename(root/'old');shutil.rmtree(root/'old')
    assert all(torch.equal(rows[i],t) for i,t in enumerate(expected))
    rows[0]=torch.ones((1,2,7),dtype=torch.int64);expected[0]=rows[0].clone()
    new.mkdir();rows.save_checkpoint(new/'state.safetensors')
    paths,_=open_rows(new/'state.safetensors');saved=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    rows[0]=rows[0]+5
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]==saved
    checks.append('old_checkpoint_deleted_live_reader_and_saved_snapshot_immutable')
    rows.close();new.rename(root/'ckpt')
    rows=DiskRows(root/'ckpt/state.safetensors',root)
    assert all(torch.equal(rows[i],t) for i,t in enumerate(expected))
    rows.save(root/'export.safetensors')
    from safetensors.torch import load_file
    exported=load_file(str(root/'export.safetensors'))
    assert all(torch.equal(exported[f'tensor.{i}'],t) for i,t in enumerate(expected))
    rows.close();checks.append('row_format_resume_and_standard_export_exact')
    cm=SimpleNamespace(load_tensor=lambda *a:None,save_tensor=lambda *a:None)
    install(cm,root)
    try:
        cm.load_tensor('ckpt/state.safetensors',dict(work_dir=str(root),cal_rows=4))
        raise AssertionError('short calibration accepted')
    except ValueError:
        pass
    checks.append('checkpoint_calibration_row_count_mismatch_rejected')
    broken=root/'ckpt/state.rows/1.safetensors'
    with broken.open('r+b') as stream:stream.truncate(broken.stat().st_size-1)
    try:open_rows(root/'ckpt/state.safetensors');raise AssertionError('truncation accepted')
    except ValueError:pass
    checks.append('truncated_row_rejected')

def scenario(events, label='owned'):
    calls=[]
    def run(command,**kwargs):
        calls.append(command[1]);event=events.pop(0)
        if isinstance(event,Exception):raise event
        if event=='gone':return SimpleNamespace(returncode=1,stdout='',stderr='Error: No such object: id')
        if event=='running':return SimpleNamespace(returncode=0,stdout=json.dumps([dict(Id='id',Config=dict(Labels={'gufo.quant.run':label}),State=dict(Running=True))]),stderr='')
        return SimpleNamespace(returncode=1,stdout='',stderr='already removed')
    child=SimpleNamespace(wait=lambda **kwargs:0)
    result=cleanup_owned('name','owned',child,lambda:dict(KFD_contexts=[]),run=run)
    assert not events
    return result,calls

result,calls=scenario(['running',subprocess.TimeoutExpired('stop',20),'gone'])
assert not result['cleanup_errors'] and 'kill' not in calls
result,calls=scenario(['running',subprocess.TimeoutExpired('stop',20),'running','kill_failed','gone'])
assert not result['cleanup_errors'] and 'kill' in calls
checks.append('stop_timeout_and_kill_autoremove_races_idempotent')
result,calls=scenario(['running'],label='someone_else')
assert result['cleanup_errors'] and calls==['inspect'] and result['returned_KFD_contexts']==[]
checks.append('foreign_owner_not_stopped_cleanup_receipt_retained')
result=dict(status='pass',checks=checks,gpu_used=False,
            code_sha256={n:hashlib.sha256((Path('/task')/n).read_bytes()).hexdigest()
                         for n in ['disk_state.py','row_checkpoint.py','owned_cleanup.py','model_job.py']})
(out/'CPU-CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
