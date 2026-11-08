"""Finish only a fully computed transformer checkpoint after GPU-window expiry.

Called under supervisor ownership locks after child reap and fresh KFD return.
No GPU imports, numerical recomputation, dtype changes, or row fallback.
"""
import array
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import shutil
import sys
import time
from row_checkpoint import tensor_header,open_rows,link_rows,sync_directory


class ModuleWindowTimeout(RuntimeError):
    pass


def durable(path, value):
    temporary=path.with_suffix('.pending')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path);sync_directory(path.parent)


def ids(path, count):
    assert sys.byteorder=='little'
    with path.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
        assert len(header)==count
        result=[]
        for i in range(count):
            key=f'tensor.{i}' if f'tensor.{i}' in header else f'input_ids.{i}'
            info=header[key];assert info['dtype'] in ('I32','I64')
            begin,end=info['data_offsets'];stream.seek(8+length+begin)
            values=array.array('i' if info['dtype']=='I32' else 'q');values.frombytes(stream.read(end-begin))
            result.append(values)
        return result


def validate(task, out, attempt):
    run=json.loads((attempt/'STATE.json').read_text())
    assert 'One-module window exceeded' in run['error']
    assert run['cleanup_errors']==[] and run['returned_KFD_contexts']==[]
    assert run['status']=='failed'
    work=out/'work';old=json.loads((work/'ckpt/job.json').read_text())
    new=json.loads((work/'ckpt_new/job.json').read_text())
    assert old['bad_rows']==new['bad_rows']==[]
    index=old['next_module_idx'];assert new['next_module_idx']==index+1
    layout=json.loads((out/'model-inventory.json').read_text())['modules']
    module=next(x for x in layout if x['component']=='main' and x['index']==index)
    assert module['type']=='TransformerBlock','Only completed transformer blocks are recoverable'
    key=module['key'];log=(attempt/'run.log').read_text()
    assert log.rstrip().endswith('-- Saving checkpoint') and 'Traceback' not in log
    assert re.search(r'^ -- Quantized: '+re.escape(key)+r'\s+bpw:.*\[.* s\]',log,re.M)
    cal=json.loads((task/'c03-input-ids.json').read_text());count=cal['rows']
    frozen=json.loads((out/'model-job-c03/FROZEN.json').read_text())
    for name in ['c03-input-ids.safetensors','c03-input-ids.json','b1-recipe.json']:
        path=task/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen[str(path)]
    assert hashlib.sha256((task/'c03-input-ids.safetensors').read_bytes()).hexdigest()==cal['sha256']
    assert ids(work/'ckpt/original_input_ids.safetensors',count)==ids(task/'c03-input-ids.safetensors',count)
    if (work/'ckpt/state.rows.json').exists():
        _,old_infos=open_rows(work/'ckpt/state.safetensors')
    else:
        header=tensor_header(work/'ckpt/state.safetensors')
        assert len(header)==count
        old_infos=[(header[f'tensor.{i}']['dtype'],tuple(header[f'tensor.{i}']['shape'])) for i in range(count)]
    assert len(old_infos)==count
    directories=list((attempt/'row-scratch').glob('state-rows-*'));assert len(directories)==1
    scratch=directories[0];paths=[scratch/f'{i}.safetensors' for i in range(count)]
    assert {x for x in scratch.iterdir() if x.is_file()}==set(paths),'Incomplete new output rows'
    infos=[];records=[]
    for i,(path,sequence) in enumerate(zip(paths,cal['sequences'],strict=True)):
        header=tensor_header(path);assert set(header)=={'tensor'}
        info=header['tensor'];pair=(info['dtype'],tuple(info['shape']))
        assert pair==old_infos[i] and pair[0] in ('F16','BF16','F32')
        assert pair[1][:2]==(1,sequence['tokens'])
        stat=path.stat();assert run['started']<=stat.st_mtime<=run['finished']
        infos.append(pair);records.append(dict(row=i,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino))
    quant=work/'qtensors'/f'{key}.safetensors';qh=tensor_header(quant)
    targets=[k for k in json.loads((task/'b1-recipe.json').read_text())['tensors'] if k.startswith(key+'.')]
    assert targets
    for target in targets:
        for suffix in ('trellis','suh','svh','mul1'):assert target+'.'+suffix in qh
    assert run['started']<=quant.stat().st_mtime<=run['finished']
    return new,paths,infos,dict(index=index,module=key,rows=records,packed_targets=len(targets),
                              calibration_sha256=cal['sha256'],log_sha256=hashlib.sha256(log.encode()).hexdigest())


def finish(task, out, attempt):
    started=time.time();new,paths,infos,evidence=validate(task,out,attempt)
    work=out/'work';stage=work/'ckpt_cpu_pending';stage.mkdir()
    link_rows(paths,stage/'state.safetensors',infos)
    os.link(work/'ckpt/original_input_ids.safetensors',stage/'original_input_ids.safetensors')
    durable(stage/'job.json',new);check_paths,check_infos=open_rows(stage/'state.safetensors')
    assert check_infos==infos
    assert all(p.stat().st_ino==q.stat().st_ino for p,q in zip(paths,check_paths,strict=True))
    archive=attempt/'previous-checkpoint';partial=attempt/'partial-checkpoint'
    journal=attempt/'CPU-COMMIT.json'
    state=dict(phase='ready',old_index=evidence['index'],new_index=new['next_module_idx'],
               stage=str(stage),archive=str(archive),partial=str(partial),evidence=evidence)
    durable(journal,state)
    (work/'ckpt_new').rename(partial);sync_directory(work);sync_directory(attempt)
    (work/'ckpt').rename(archive);sync_directory(work);sync_directory(attempt)
    state['phase']='old_archived';durable(journal,state)
    stage.rename(work/'ckpt');sync_directory(work)
    state.update(phase='committed',seconds=time.time()-started,finished=time.time())
    durable(journal,state)
    return dict(status='committed',next_module_idx=new['next_module_idx'],seconds=state['seconds'],rows=len(paths))


def run_cpu(attempt):
    """Independent bounded CPU commit window; caller retains ownership locks."""
    started=time.monotonic();state=dict(status='running',started=time.time(),minimum_available_bytes=2**63)
    def check_resources():
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        state['minimum_available_bytes']=min(state['minimum_available_bytes'],available)
        assert available>=8*2**30,'CPU commit host reserve below8GiB'
        assert shutil.disk_usage(attempt).free>=100*2**30,'CPU commit disk reserve below100GiB'
    check_resources()
    with (attempt/'CPU-CHECKPOINT.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(attempt)],stdout=log,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                check_resources()
                assert time.monotonic()-started<120,'CPU commit window exceeded'
                time.sleep(0.25)
            assert child.returncode==0,f'CPU checkpoint commit exit{child.returncode}'
            state['status']='passed'
        except BaseException as error:
            state.update(status='failed',error=repr(error));raise
        finally:
            if child.poll() is None:child.kill()
            child.wait(timeout=10)
            state['finished']=time.time();durable(attempt/'CPU-WINDOW.json',state)
    journal=json.loads((attempt/'CPU-COMMIT.json').read_text());assert journal['phase']=='committed'
    return state


if __name__=='__main__':
    task=Path(__file__).resolve().parent;base=task.parents[2]
    out=base/'tmp/test-results/flash-next-quant-20261104'
    attempt=Path(sys.argv[1]).resolve()
    assert attempt.parent==out/'model-job-c03' and re.fullmatch(r'module-\d+',attempt.name)
    print(json.dumps(finish(task,out,attempt)))
