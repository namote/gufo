"""Explicit CPU-only recovery of completed attempt004 row outputs. No recompute."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import struct
import array
import sys
from row_checkpoint import tensor_header,link_rows,open_rows,sync_directory

from paths import OUT as out, TASK as task
root=out/'model-job-c03';repair=root/'repair-002-checkpoint';work=out/'work'
guard=(root/'JOB.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['systemctl','--user','show','gufo-quant-model-c03.service','-p','MainPID','--value'],text=True).strip()=='0'
assert subprocess.run(['docker','inspect','gufo-quant-model'],capture_output=True).returncode!=0
assert shutil.disk_usage(out).free>=150*2**30
from host_guard import host,npu,quiet
h=host();assert not h['KFD_contexts'] and quiet(h) and not npu()
old=json.loads((work/'ckpt/job.json').read_text());new=json.loads((work/'ckpt_new/job.json').read_text())
assert old['next_module_idx']==2 and new['next_module_idx']==3 and new['bad_rows']==[]
attempt=root/'module-004';log=(attempt/'run.log').read_text()
assert ' -- Quantized: model.language_model.layers.0 ' in log and log.rstrip().endswith('-- Saving checkpoint')
assert 'Traceback' not in log
run=json.loads((attempt/'STATE.json').read_text());end=json.loads((root/'STATE.json').read_text())['finished']
cal=json.loads((task/'c03-input-ids.json').read_text());assert cal['rows']==2309
assert hashlib.sha256((task/'c03-input-ids.safetensors').read_bytes()).hexdigest()==cal['sha256']
frozen=json.loads((repair/'FROZEN.json').read_text())
for name in ['c03-input-ids.safetensors','c03-input-ids.json','b1-recipe.json']:
    p=task/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen[str(p)]
old_header=tensor_header(work/'ckpt/state.safetensors')
token_header=tensor_header(work/'ckpt/original_input_ids.safetensors')
assert len(old_header)==len(token_header)==cal['rows']
assert sys.byteorder=='little'
def read_ids(path):
    with path.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
        result=[]
        for i in range(cal['rows']):
            key=f'tensor.{i}' if f'tensor.{i}' in header else f'input_ids.{i}'
            info=header[key];assert info['dtype'] in ('I32','I64')
            begin,end=info['data_offsets'];stream.seek(8+length+begin)
            values=array.array('i' if info['dtype']=='I32' else 'q');values.frombytes(stream.read(end-begin))
            result.append(values)
        return result
assert read_ids(work/'ckpt/original_input_ids.safetensors')==read_ids(task/'c03-input-ids.safetensors')
directories=list((attempt/'row-scratch').glob('state-rows-*'));assert len(directories)==1
scratch=directories[0];paths=[scratch/f'{i}.safetensors' for i in range(cal['rows'])]
assert set(scratch.iterdir())==set(paths),'Missing/extra/pending row output'
infos=[];records=[]
for i,(path,sequence) in enumerate(zip(paths,cal['sequences'],strict=True)):
    header=tensor_header(path);assert set(header)=={'tensor'}
    info=header['tensor'];expected=old_header[f'tensor.{i}']
    assert info['dtype']==expected['dtype']=='F32'
    assert info['shape']==expected['shape']==[1,sequence['tokens'],4,2560]
    assert token_header[f'tensor.{i}']['shape']==[1,sequence['tokens']]
    stat=path.stat();assert run['started']<=stat.st_mtime<=end
    infos.append((info['dtype'],tuple(info['shape'])))
    records.append(dict(row=i,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino))
quant=work/'qtensors/model.language_model.layers.0.safetensors';qh=tensor_header(quant)
recipe=json.loads((task/'b1-recipe.json').read_text())
targets=[key for key in recipe['tensors'] if key.startswith('model.language_model.layers.0.')]
assert len(targets)==1542
for key in targets:
    for suffix in ('trellis','suh','svh','mul1'):
        assert key+'.'+suffix in qh
for expert in range(512):
    for projection in ('gate_proj','up_proj','down_proj'):
        for suffix in ('trellis','suh','svh','mul1'):
            assert f'model.language_model.layers.0.mlp.experts.{expert}.{projection}.{suffix}' in qh
assert run['started']<=quant.stat().st_mtime<=end
stage=work/'ckpt_recovered';stage.mkdir()
started=time.time();link_rows(paths,stage/'state.safetensors',infos)
os.link(work/'ckpt/original_input_ids.safetensors',stage/'original_input_ids.safetensors')
with (stage/'job.json').open('x') as stream:
    json.dump(new,stream,indent=2);stream.flush();os.fsync(stream.fileno())
sync_directory(stage)
check_paths,check_infos=open_rows(stage/'state.safetensors');assert check_infos==infos
assert all(p.stat().st_ino==q.stat().st_ino for p,q in zip(paths,check_paths,strict=True))
receipt=dict(status='validated_before_commit',next_module_idx=3,calibration_sha256=cal['sha256'],
             rows=records,qtensors_bytes=quant.stat().st_size,qtensors_header_entries=len(qh),
             original_token_ids_unchanged=True,gpu_used=False,link_seconds=time.time()-started,
             log_sha256=hashlib.sha256(log.encode()).hexdigest(),returned_KFD_contexts=h['KFD_contexts'])
(repair/'RECOVERY.json').write_text(json.dumps(receipt,indent=2))
(work/'ckpt_new').rename(repair/'truncated-ckpt-new')
(work/'ckpt').rename(repair/'checkpoint-idx2')
stage.rename(work/'ckpt');sync_directory(work);sync_directory(repair)
receipt.update(status='committed_idx3',finished=time.time())
(repair/'RECOVERY.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='rows'},indent=2))
