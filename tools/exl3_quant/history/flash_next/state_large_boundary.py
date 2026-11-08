"""Reserve next natural module boundary; never signal an active converter."""
import fcntl,json,os,signal,subprocess,time
from pathlib import Path
from run_stage import OUT,LOCK,save
R=OUT/'capture-io-large';R.mkdir(exist_ok=True)
def stopped(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
signal.signal(signal.SIGTERM,stopped)
signal.signal(signal.SIGALRM,stopped)
assert not (R/'BOUNDARY.json').exists(),'Duplicate boundary request'
assert not (R/'RELEASE.json').exists(),'Stale boundary release'
state=dict(status='waiting',pid=os.getpid(),started=time.time())
save(R/'BOUNDARY.json',state)
try:
    with LOCK.open('a') as gpu, (OUT/'model-job-c03/JOB.lock').open('a') as job:
        signal.alarm(4*3600)
        try:fcntl.flock(gpu,fcntl.LOCK_EX)
        finally:signal.alarm(0)
        deadline=time.monotonic()+180
        while True:
            try:fcntl.flock(job,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                assert time.monotonic()<deadline,'Supervisor did not release module boundary'
                time.sleep(.2)
        state.update(status='reserved',acquired=time.time(),checkpoint=json.loads((OUT/'work/ckpt/job.json').read_text()))
        save(R/'BOUNDARY.json',state)
        control=json.loads((OUT/'WATCH-CONTROL.json').read_text())
        assert control['enabled'],'Latest stop must take priority'
        msg=('[사용자가 승인한 800행/10GiB 읽기큐 적용 경계 도달] /home/m2on/projects/gufo/plan/exl3/F3-CONVERSION-WATCH.md와 capture-io-large/{BOUNDARY,CPU,REVIEW}.json을 읽어라. output=/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104. 본체 자연 모듈경계 GPU/JOB락을 state_large_boundary.py가 보유중이다. 이전 모듈 성공·실제checkpoint·GPUcleanup을 확인하고 검증된 disk_state_large.py 연결 후보/검사/이전동결보존/새동결로 통합한 뒤 RELEASE.json을 쓰고 락반환을 확인하여 본체를 재개하라. 최신 중지지시 우선. 실패한모듈이면 원인검토부터하고 중복GPU·기존baseline재실행·실행중코드수정금지.')
        sent=subprocess.run(['/home/m2on/.local/bin/codex','queue','--thread',control['thread'],'--message',msg,'--resume'],capture_output=True,text=True,timeout=45)
        save(R/'FOLLOWUP.json',dict(exit=sent.returncode,stdout=sent.stdout,stderr=sent.stderr))
        assert sent.returncode==0,'Boundary followup failed'
        deadline=time.monotonic()+3600
        while not (R/'RELEASE.json').exists():
            assert time.monotonic()<deadline,'Reservation expired'
            assert json.loads((OUT/'WATCH-CONTROL.json').read_text())['enabled'],'Latest stop must take priority'
            time.sleep(1)
        state.update(status='released',finished=time.time());save(R/'BOUNDARY.json',state)
except BaseException as e:
    state.update(status='failed',error=repr(e),finished=time.time());save(R/'BOUNDARY.json',state);raise
