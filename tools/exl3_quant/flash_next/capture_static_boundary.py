"""Reserve next natural module boundary; never signal an active converter."""
import fcntl,json,os,signal,subprocess,time
from pathlib import Path
from run_stage import OUT,LOCK,save
R=OUT/'capture-static-boundary';R.mkdir(exist_ok=True)
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
        save(R/'FOLLOWUP.json',dict(status='delegated_to_capture_static_job',time=time.time()))
        deadline=time.monotonic()+3600
        while not (R/'RELEASE.json').exists():
            assert time.monotonic()<deadline,'Reservation expired'
            assert json.loads((OUT/'WATCH-CONTROL.json').read_text())['enabled'],'Latest stop must take priority'
            time.sleep(1)
        state.update(status='released',finished=time.time());save(R/'BOUNDARY.json',state)
except BaseException as e:
    state.update(status='failed',error=repr(e),finished=time.time());save(R/'BOUNDARY.json',state);raise
