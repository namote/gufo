"""Continue only clean, progressing time-window boundaries; never retry numerical/resource failures."""
import hashlib
import fcntl
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

TASK=Path(__file__).parent
OUT=Path('/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104')
SOURCE=Path('/run/media/m2on/T5 EVO/AI/model/originals-flash-next-hf-20261030/Qwen-Qwen3.8-Flash-Next')
STAGE=OUT/'ngram-full'
STAGE.mkdir(exist_ok=True)
job_lock=(STAGE/'JOB.lock').open('a')
fcntl.flock(job_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
inventory=json.loads((OUT/'source-inventory.json').read_text())
files=[TASK/'run_stage.py',TASK/'ngram_full.py',TASK/'ngram_job.py',OUT/'source-inventory.json']
runtime=Path('/home/m2on/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/kyojin_runtime')
files+=list((runtime/'source/exllamav3/conversion').glob('*.py'))
files+=list((runtime/'source/exllamav3/modules/quant/exl3_lib').glob('*.py'))
files+=[runtime/'source/exllamav3/modules/quant/exl3.py',runtime/'SOURCE.json']
frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
native=runtime/'source/exllamav3_ext.cpython-312-x86_64-linux-gnu.so'
native_stat=(native.stat().st_size,native.stat().st_mtime_ns,native.stat().st_ino)
identity=dict(code=frozen,native=str(native),native_stat=list(native_stat),
    K=8,kernel='opt',streams=2,chunk_rows=32768,source_revision=inventory['revision'])
if (STAGE/'FROZEN.json').exists():
    assert json.loads((STAGE/'FROZEN.json').read_text())==identity,'Frozen job identity changed; do not resume'
else:
    (STAGE/'FROZEN.json').write_text(json.dumps(identity,indent=2))

def save(data):
    t=STAGE/'JOB.tmp';t.write_text(json.dumps(data,indent=2));t.replace(STAGE/'JOB.json')

def progress():
    p=OUT/'pack/ngram_embedding.safetensors'
    if not p.exists() or p.stat().st_size<8:return 0
    with p.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n))
    start=8+n+min(v['data_offsets'][0] for k,v in h.items() if k.endswith('.trellis'))
    return max(0,(p.stat().st_size-start)//(162*32768))

if (STAGE/'JOB.json').exists():
    job=json.loads((STAGE/'JOB.json').read_text())
    if job['status']=='ngram_complete':
        assert (STAGE/'RESULT.json').exists()
        raise SystemExit(0)
    job.update(status='running',resumed=time.time())
else:
    job=dict(status='running',stage='ngram_K8',started=time.time(),windows=[],
             next_stage='Main model conversion; not yet launched by this ngram-only supervisor')
save(job)
first=1+max([int(p.name.split('-')[1]) for p in STAGE.glob('window-*') if p.is_dir()]+[0])
try:
    for number in range(first,first+8):
        for p,h in frozen.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
        assert (native.stat().st_size,native.stat().st_mtime_ns,native.stat().st_ino)==native_stat
        for f in inventory['files']:
            s=(SOURCE/f['name']).stat();assert s.st_size==f['bytes'] and s.st_mtime_ns==f['mtime_ns'],f['name']
        before=progress();start=time.time()
        job.update(window=number,updated=start);save(job)
        result=subprocess.run([sys.executable,str(TASK/'run_stage.py'),'ngram-full','--seconds','3600'])
        state=json.loads((STAGE/'STATE.json').read_text())
        assert state['started']>=start
        archive=STAGE/f'window-{number:02d}';archive.mkdir()
        for n in ['STATE.json','PROGRESS.json']:
            if (STAGE/n).exists():shutil.copyfile(STAGE/n,archive/n)
        after=progress()
        job['windows'].append(dict(number=number,started=start,finished=time.time(),
             complete_chunks_before=before,complete_chunks_after=after,exit_code=result.returncode))
        if result.returncode==0:
            r=json.loads((STAGE/'RESULT.json').read_text());assert r['status']=='complete_operator_verified'
            job.update(status='ngram_complete',finished=time.time());save(job);break
        assert state.get('status')=='stopped' and state.get('error','').startswith('TimeoutError('),state
        assert state.get('cleanup_status')=='passed' and not state.get('returned_KFD_contexts')
        assert subprocess.run(['docker','inspect','gufo-quant-ngram-full'],capture_output=True).returncode!=0
        assert after>before,'No complete-chunk progress; do not retry'
        job.update(updated=time.time());save(job)
    else:
        raise TimeoutError('Eight-hour ngram budget exhausted; retain output for review')
except BaseException as e:
    job.update(status='stopped',error=repr(e),finished=time.time());save(job);raise
