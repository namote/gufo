"""Small standard-library fixtures for timeout-only CPU checkpoint completion."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import time
from unittest.mock import patch
from finish_checkpoint import finish,validate
from row_checkpoint import tensor_header,open_rows

result_dir=Path('/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104/model-job-c03/repair-004-cpu-commit')

def save(path,tensors):
    header={};payload=b''
    for key,(dtype,shape,data) in tensors.items():
        header[key]=dict(dtype=dtype,shape=shape,data_offsets=[len(payload),len(payload)+len(data)])
        payload+=data
    raw=json.dumps(header,separators=(',',':')).encode();path.write_bytes(struct.pack('<Q',len(raw))+raw+payload)

def fixture(root):
    task=root/'task';task.mkdir();out=root/'output';out.mkdir();work=out/'work';work.mkdir()
    for name in ('ckpt','ckpt_new','qtensors'):(work/name).mkdir()
    runs=out/'model-job-c03';runs.mkdir();attempt=runs/'module-001';attempt.mkdir()
    scratch=attempt/'row-scratch/state-rows-fixture';scratch.mkdir(parents=True);(scratch/'base').mkdir()
    sequences=[dict(tokens=2),dict(tokens=3)]
    save(task/'c03-input-ids.safetensors',{f'input_ids.{i}':('I32',[1,s['tokens']],struct.pack('<'+'i'*s['tokens'],*range(s['tokens']))) for i,s in enumerate(sequences)})
    cal=dict(rows=2,sequences=sequences,sha256=hashlib.sha256((task/'c03-input-ids.safetensors').read_bytes()).hexdigest())
    (task/'c03-input-ids.json').write_text(json.dumps(cal))
    (task/'b1-recipe.json').write_text(json.dumps(dict(tensors={'model.layers.0.proj':4})))
    tensors={f'tensor.{i}':('F32',[1,s['tokens'],2],struct.pack('<'+'f'*(s['tokens']*2),*range(s['tokens']*2))) for i,s in enumerate(sequences)}
    save(work/'ckpt/state.safetensors',tensors)
    for i in range(2):save(scratch/f'{i}.safetensors',{'tensor':tensors[f'tensor.{i}']})
    save(work/'ckpt/original_input_ids.safetensors',{f'tensor.{i}':('I64',[1,s['tokens']],struct.pack('<'+'q'*s['tokens'],*range(s['tokens']))) for i,s in enumerate(sequences)})
    save(work/'qtensors/model.layers.0.safetensors',{'model.layers.0.proj.'+suffix:('F32',[1],struct.pack('<f',1.0)) for suffix in ('trellis','suh','svh','mul1')})
    (work/'ckpt/job.json').write_text(json.dumps(dict(next_module_idx=0,bad_rows=[])))
    (work/'ckpt_new/job.json').write_text(json.dumps(dict(next_module_idx=1,bad_rows=[])))
    (out/'model-inventory.json').write_text(json.dumps(dict(modules=[dict(index=0,component='main',type='TransformerBlock',key='model.layers.0')])))
    (runs/'FROZEN.json').write_text(json.dumps({str(task/n):hashlib.sha256((task/n).read_bytes()).hexdigest() for n in ['c03-input-ids.safetensors','c03-input-ids.json','b1-recipe.json']}))
    now=time.time();state=dict(status='failed',error='One-module window exceeded',cleanup_errors=[],returned_KFD_contexts=[],started=now-10,finished=now+10)
    (attempt/'STATE.json').write_text(json.dumps(state));(attempt/'run.log').write_text(' -- Quantized: model.layers.0  bpw: 4.00 [1.0 s]\n -- Saving checkpoint\n')
    return task,out,attempt,scratch

checks=[]
with tempfile.TemporaryDirectory(dir=result_dir) as directory:
    root=Path(directory)
    for failure in ['incomplete_rows','head','wrong_ids','bad_extent','missing_packed','other_failure','cleanup_failed','unfinished_forward']:
        case=root/failure;case.mkdir();task,out,attempt,scratch=fixture(case)
        if failure=='incomplete_rows':(scratch/'1.safetensors').unlink()
        if failure=='head':
            p=out/'model-inventory.json';v=json.loads(p.read_text());v['modules'][0]['type']='Linear';p.write_text(json.dumps(v))
        if failure=='wrong_ids':
            p=out/'work/ckpt/original_input_ids.safetensors';v=bytearray(p.read_bytes());v[-1]=1;p.write_bytes(v)
        if failure=='bad_extent':
            p=scratch/'0.safetensors';p.write_bytes(p.read_bytes()[:-1])
        if failure=='missing_packed':save(out/'work/qtensors/model.layers.0.safetensors',{'x':('F32',[1],struct.pack('<f',1.0))})
        if failure in ('other_failure','cleanup_failed'):
            p=attempt/'STATE.json';v=json.loads(p.read_text())
            if failure=='other_failure':v['error']='Host reserve below8GiB'
            else:v['cleanup_errors']=['KFD remains']
            p.write_text(json.dumps(v))
        if failure=='unfinished_forward':(attempt/'run.log').write_text(' -- Captured: model.layers.0\n')
        rejected=False
        try:validate(task,out,attempt)
        except (AssertionError,ValueError):rejected=True
        assert rejected,failure
        assert json.loads((out/'work/ckpt/job.json').read_text())['next_module_idx']==0
        checks.append('reject_'+failure)
    case=root/'valid';case.mkdir();task,out,attempt,scratch=fixture(case)
    result=finish(task,out,attempt);assert result['status']=='committed'
    assert json.loads((out/'work/ckpt/job.json').read_text())['next_module_idx']==1
    paths,_=open_rows(out/'work/ckpt/state.safetensors')
    assert all(p.stat().st_ino==(scratch/p.name).stat().st_ino for p in paths)
    assert (attempt/'previous-checkpoint/job.json').exists()
    assert (attempt/'partial-checkpoint/job.json').exists()
    assert json.loads((attempt/'CPU-COMMIT.json').read_text())['phase']=='committed'
    checks.append('complete_exact_hardlinks_and_durable_commit_original_preserved')
    case=root/'commit_interrupted';case.mkdir();task,out,attempt,scratch=fixture(case)
    original_rename=Path.rename
    def interrupted_rename(source,target):
        if source.name=='ckpt_cpu_pending':raise OSError('injected commit interruption')
        return original_rename(source,target)
    try:
        with patch.object(Path,'rename',interrupted_rename):finish(task,out,attempt)
        raise AssertionError('commit interruption not exercised')
    except OSError:
        pass
    journal=json.loads((attempt/'CPU-COMMIT.json').read_text())
    assert journal['phase']=='old_archived'
    assert (attempt/'previous-checkpoint/job.json').exists()
    assert (out/'work/ckpt_cpu_pending/job.json').exists()
    assert not (out/'work/ckpt').exists()
    assert len(open_rows(out/'work/ckpt_cpu_pending/state.safetensors')[0])==2
    checks.append('interrupted_commit_retains_original_and_stage_with_admission_blocker')
result=dict(status='pass',checks=checks,gpu_used=False,
            code_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['finish_checkpoint.py','model_job.py']})
(result_dir/'CPU-CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
