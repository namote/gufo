"""Reserve next natural module boundary; never signal an active converter."""
import fcntl,json,os,signal,subprocess,time
from pathlib import Path
from run_stage import OUT,LOCK,save
R=OUT/'disk-boundary';R.mkdir(exist_ok=True)
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
        msg=('[사용자가 승인한 레이어 시작 전 디스크 정리 적용 경계 도달] /home/m2on/projects/gufo/plan/exl3/F3-CONVERSION-WATCH.md와 disk-boundary/{BOUNDARY,CPU,REVIEW,INTEGRATION-CPU}.json을 읽어라. output=/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104. disk_boundary_reserve.py가 자연경계GPU/JOB락을보유한다. 이전모듈성공/실제checkpoint/cleanup확인후 model_job_disk_candidate.py를검토된동일본으로최소통합하라. 이전433동결보존·새동결·RELEASE후락반환확인·CPUadmission과diskprepare실행후현재checkpoint에서재개하라. 상태파일및단독완료자료만정리, 실패미복구/활성checkpoint/모델/baseline보호. 전체정확성/성능반복금지, 실행중코드수정/중복GPU금지. 최신사용자지시우선.')
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
