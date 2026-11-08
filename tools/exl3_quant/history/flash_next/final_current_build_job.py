"""Bounded CPU-only build supervisor, preserving incremental outputs and evidence."""
import argparse,fcntl,json,os,shutil,signal,subprocess,time,uuid
from pathlib import Path
from run_stage import OUT,TASK,available,save
from mtp_calibration_job import sha
from mtp_calibration_boundary import notify
from owned_cleanup import cleanup_owned
B=OUT/'final-current-nix101'

def command(plan,phase,name,label):
    return ['docker','run','--rm','--pull','never','--network','none','--read-only','--user','1000:1000',
        '--name',name,'--label','gufo.quant.run='+label,'--memory','100g','--memory-swap','100g','--cpus','12',
        '--tmpfs','/tmp:rw,size=2g,mode=1777','--mount','type=volume,src=gufo-nix,dst=/nix,readonly',
        '--mount',f'type=bind,src={TASK},dst=/task,readonly',
        '--mount',f'type=bind,src={B},dst=/plan,readonly',
        '--mount',f'type=bind,src={plan["output"]},dst=/build',
        '--mount',f'type=bind,src={plan["source"]},dst=/build/source,readonly',
        '--mount',f'type=bind,src={B/"work"},dst=/receipt',
        '--entrypoint','/opt/venv/bin/python3',plan['image'],'-u','/task/final_current_build.py','--phase',phase]

def cpu_cleanup(name,label,child,run=subprocess.run):
    # Container has no GPU devices: reuse ownership/stop/kill/child verification,
    # without claiming a GPU host observation or blocking unrelated GPU owners.
    result=cleanup_owned(name,label,child,lambda:{'KFD_contexts':[]},run=run)
    result.pop('returned_KFD_contexts')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('configure','build','cpu'),required=True);a=ap.parse_args()
    plan=json.loads((B/'BUILD-PLAN.json').read_text());review=json.loads((B/'BUILD-REVIEW.json').read_text())
    assert review['status']=='approved_cpu_only_coherent_build' and review['plan_sha256']==sha(B/'BUILD-PLAN.json')
    assert plan['driver_sha256']==sha(TASK/'final_current_build.py')
    assert plan['cleanup_sha256']==sha(TASK/'owned_cleanup.py')
    assert review['supervisor_sha256']==sha(__file__)
    assert plan['source_manifest_sha256']==sha(B/'SOURCE.json')
    assert plan['source']==str(TASK/'final-current-nix101/src') and plan['output']==str(TASK/'final-current-nix101/build')
    assert not plan['gpu'] and not plan['network'] and plan['parallel']==8
    work=Path(plan['output']);work.mkdir(exist_ok=True);(B/'work').mkdir(exist_ok=True)
    status=B/(a.phase.upper()+'-STATE.json');assert not status.exists(),'Preserve phase evidence for explicit recovery'
    child=None;name='gufo-current-nix101-'+uuid.uuid4().hex[:10];label=uuid.uuid4().hex
    state=dict(status='admitting',phase=a.phase,started=time.time(),plan_sha256=sha(B/'BUILD-PLAN.json'),gpu_used=False,
               container=name,label=label,cleanup_errors=[])
    with (B/'BUILD.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            assert json.loads((OUT/'WATCH-CONTROL.json').read_text())['enabled']
            assert available()>28*2**30 and shutil.disk_usage(B).free>=150*2**30
            cmd=command(plan,a.phase,name,label);state.update(status='running',command=cmd);save(status,state)
            def stop(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            deadline=time.monotonic()+plan['timeout_seconds']
            with (B/(a.phase+'-supervisor.log')).open('x') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    assert time.monotonic()<deadline,'CPU build deadline'
                    assert json.loads((OUT/'WATCH-CONTROL.json').read_text())['enabled'],'Latest user stop'
                    mem=available();free=shutil.disk_usage(B).free
                    assert mem>=4*2**30 and free>=100*2**30,'Build resource reserve'
                    state.update(updated=time.time(),available_bytes=mem,disk_free_bytes=free)
                    save(status,state);time.sleep(2)
                state['returncode']=child.returncode;assert child.returncode==0
            result=B/'work'/(a.phase.upper()+'.json');assert json.loads(result.read_text())['status']=='passed_pending_review'
            state.update(status='passed_pending_review',result_sha256=sha(result))
        except BaseException as e:state.update(status='failed',error=repr(e))
        finally:
            if child is not None:
                cleaned=cpu_cleanup(name,label,child)
                state['cleanup_errors']=cleaned['cleanup_errors']
                if state['cleanup_errors']:state['status']='cleanup_failed'
            state['finished']=time.time();save(status,state)
    save(B/(a.phase+'-FOLLOWUP.json'),notify('[Gufo currentHEAD Nix10.1 CPU build '+a.phase+' 종료] '+str(status)+
         '와 work/단계 영수증을 확인하라. 모델 재양자화/기존baseline반복 금지. GPU를 사용하지 않았다. 빌드성공은추론검증이아니며 '
         '구production과새currentHEAD ELF/collector를섞지말고 다음독립검토후단계만진행하라.'))
    assert state['status']=='passed_pending_review',state

if __name__=='__main__':main()
