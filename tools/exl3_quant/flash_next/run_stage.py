"""Pinned, GPU-exclusive conversion stages with retained output and host-memory guard."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from paths import OUT, RUNTIME, SOURCE, TASK
from host_guard import LOCK

# Container mount point of RUNTIME, unchanged so env and receipts keep the c03 values.
CR = '/workspace/test/experiments/gufo_upstream_port_20261023/kyojin_runtime'
IMAGE = 'sha256:79e3e0b7bc23ed230dc90a11e342aeae109086d145bb0b302363ac2069b43126'

def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable unavailable')

def save(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['inventory','ngram-pilot','ngram-full','ngram-speed','ngram-streams','bank-pilot','model-pilot','convert'])
    ap.add_argument('--seconds',type=int,default=900);a=ap.parse_args()
    assert 0<a.seconds<=3600
    name='gufo-quant-'+a.stage
    run_id=uuid.uuid4().hex
    logdir=OUT/a.stage;logdir.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(OUT).free>=150*2**30
    assert available()>=28*2**30
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Runtime imports are allowed only after checking current compute ownership.
        from host_guard import host,npu,quiet
        h=host();assert quiet(h), 'Another GPU owner is active';assert not npu()
        assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0, 'Stage container already exists'
        command=['docker','run','--rm','--name',name,'--pull','never','--network','none',
          '--label','gufo.quant.run='+run_id,
          '--read-only','--user','1000:1000','--memory','100g','--memory-swap','100g',
          '--device','/dev/kfd','--device','/dev/dri','--ipc','host',
          '--tmpfs','/tmp:rw,size=1g,mode=1777',
          '--mount',f'type=bind,src={RUNTIME},dst={CR},readonly',
          '--mount',f'type=bind,src={SOURCE},dst=/source,readonly',
          '--mount',f'type=bind,src={TASK},dst=/task,readonly',
          '--mount',f'type=bind,src={OUT},dst=/output']
        for gid in sorted({Path('/dev/kfd').stat().st_gid,*[p.stat().st_gid for p in Path('/dev/dri').glob('renderD*')]}):
            command+=['--group-add',str(gid)]
        env=dict(PYTHONPATH=CR+'/source:'+CR+'/site',PYTHONUNBUFFERED='1',
          PATH=CR+'/site/bin:/opt/venv/bin:/usr/local/bin:/usr/bin:/bin',HOME='/output/home',
          XDG_CACHE_HOME='/output/cache',TRITON_CACHE_DIR='/output/triton-cache',
          EXL3_ROCM_SDK=CR+'/site/_rocm_sdk_devel',EXL3_HIPCC=CR+'/site/_rocm_sdk_devel/bin/hipcc',
          LD_PRELOAD=CR+'/site/_rocm_sdk_core/lib/libhsa-runtime64.so.1',
          PYTORCH_ROCM_ARCH='gfx1151',MAX_JOBS='4',EXL3_ABLIT_RUNTIME='off',
          EXL3_CKPT_KEEP_OLD='0',EXL3_RWH='0')
        if a.stage=='ngram-full':env['EXL3_QT_KERNEL']='opt'
        for k,v in env.items():command+=['-e',k+'='+v]
        command+=['--workdir',CR+'/source','--entrypoint','/opt/venv/bin/python3',IMAGE,'-u']
        if a.stage=='inventory':
            command+=['/task/model_inventory.py']
        elif a.stage=='bank-pilot':
            command+=['/task/bank_pilot.py']
        elif a.stage=='ngram-speed':
            command+=['/task/ngram_speed.py']
        elif a.stage=='ngram-streams':
            command+=['/task/ngram_streams.py']
        elif a.stage=='ngram-pilot':
            command+=['/task/ngram_pilot.py']
        elif a.stage=='ngram-full':
            gate=json.loads((OUT/'ngram-pilot/NUMERICAL.json').read_text())
            assert gate['status']=='operator_pass' and gate['finite']
            command+=['/task/ngram_full.py']
        else:
            raise RuntimeError('Main model stages require qualified recipe and pilot; not yet enabled')
        state=dict(stage=a.stage,status='running',run_id=run_id,command=command,started=time.time(),pid=os.getpid(),
                   host_admission={'MemAvailable':h['available_bytes'],'KFD_contexts':h['KFD_contexts']},
                   deadline_seconds=a.seconds,minimum_available_bytes=available(),gpu_errors_policy='diagnostic_only')
        save(logdir/'STATE.json',state)
        with (logdir/'run.log').open('a') as logfile:
            child=subprocess.Popen(command,stdout=logfile,stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    mem=available();state['minimum_available_bytes']=min(state['minimum_available_bytes'],mem)
                    state.update(updated=time.time(),available_bytes=mem);save(logdir/'STATE.json',state)
                    if mem<8*2**30:
                        raise RuntimeError('Host MemAvailable below 8 GiB')
                    if time.time()-state['started']>a.seconds:
                        raise TimeoutError('Stage window ended; retain resumable output')
                    if shutil.disk_usage(OUT).free<100*2**30:
                        raise RuntimeError('Internal disk reserve below 100 GiB')
                    time.sleep(2)
                state.update(status='passed' if child.returncode==0 else 'failed',returncode=child.returncode)
            except BaseException as e:
                state.update(status='stopped',error=repr(e))
                raise
            finally:
                # Only stop a container carrying this invocation's unique owner label.
                inspect=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=15)
                if inspect.returncode==0:
                    item=json.loads(inspect.stdout)[0]
                    assert item['Config']['Labels'].get('gufo.quant.run')==run_id
                    cid=item['Id']
                    try:
                        subprocess.run(['docker','stop','--time','10',cid],capture_output=True,timeout=20)
                    except subprocess.TimeoutExpired:
                        subprocess.run(['docker','kill',cid],capture_output=True,timeout=15,check=True)
                    check=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=15)
                    if check.returncode==0 and json.loads(check.stdout)[0]['State']['Running']:
                        subprocess.run(['docker','kill',cid],capture_output=True,timeout=15,check=True)
                child.wait(timeout=30)
                final=host()
                state['returned_KFD_contexts']=final['KFD_contexts']
                state['cleanup_status']='passed' if not final['KFD_contexts'] else 'failed'
                if final['KFD_contexts']:state['status']='cleanup_failed'
                state['finished']=time.time();save(logdir/'STATE.json',state)
                assert not final['KFD_contexts'], 'Compute context remained after stage cleanup'
        print(json.dumps({k:v for k,v in state.items() if k!='command'}))
        if state['returncode']:
            raise SystemExit(state['returncode'])

if __name__=='__main__':
    main()
