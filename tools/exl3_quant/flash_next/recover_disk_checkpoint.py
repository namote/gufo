"""Explicit CPU-only completed-checkpoint recovery after disk reserve stop.

Never used as automatic fallback. Keep original failed STATE; retain every
validation in the frozen timeout helper, permit only the exact disk reserve
exception with a complete forward-phase receipt, all newly written rows and pack.
"""
import fcntl,hashlib,inspect,json,signal,shutil,time
from pathlib import Path
import finish_checkpoint as base
from run_stage import OUT,TASK,LOCK,available


def eligible(run,log,count):
    assert run['status']=='failed' and run['error']=="AssertionError('Disk reserve below 100 GiB')"
    assert not run['cleanup_errors'] and not run['returned_KFD_contexts']
    complete=[]
    for line in log.splitlines():
        if ' -- GUFO_PROGRESS ' in line:
            row=json.loads(line.split(' -- GUFO_PROGRESS ',1)[1])
            if row.get('event')=='phase_end' and row.get('text','').startswith(' -- Forward pass: '):complete.append(row)
    assert len(complete)==1 and complete[0]['result']=='completed'
    assert complete[0]['count']==count and complete[0]['current']==count-1
    assert log.rstrip().endswith('-- Saving checkpoint') and 'Traceback' not in log


def functions():
    frozen=json.loads((OUT/'model-job-c03/FROZEN.json').read_text())
    source=Path(base.__file__)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==frozen[str(source)]
    old="assert 'One-module window exceeded' in run['error']"
    new="eligible(run, (attempt/'run.log').read_text(), json.loads((task/'c03-input-ids.json').read_text())['rows'])"
    text=inspect.getsource(base.validate);assert text.count(old)==1
    namespace=dict(vars(base),eligible=eligible)
    exec(compile(text.replace(old,new),'<disk-recovery-validate>','exec'),namespace)
    exec(compile(inspect.getsource(base.finish),'<disk-recovery-finish>','exec'),namespace)
    return namespace['validate'],namespace['finish']


def main():
    from test.experiments.gufo_upstream_port_20261023.window import host,npu
    attempt=OUT/'model-job-c03/module-016'
    with LOCK.open('a') as gpu,(OUT/'model-job-c03/JOB.lock').open('a') as job:
        fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(job,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert json.loads((OUT/'WATCH-CONTROL.json').read_text())['enabled']
        assert not host()['KFD_contexts'] and not npu()
        assert available()>=8*2**30 and shutil.disk_usage(OUT).free>=100*2**30
        assert json.loads((OUT/'capture-forward-pair/STATE.json').read_text())['status']=='failed'
        assert not (attempt/'CPU-COMMIT.json').exists()
        validate,finish=functions()
        new,paths,infos,evidence=validate(TASK,OUT,attempt)
        assert evidence['index']==7 and new['next_module_idx']==8
        base.durable(attempt/'DISK-RECOVERY-VALIDATION.json',dict(status='validated',time=time.time(),rows=len(paths),source_error="Disk reserve below 100 GiB",calibration_sha256=evidence['calibration_sha256'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        def timeout(signum,frame):raise TimeoutError('CPU recovery exceeded120s')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
        try:result=finish(TASK,OUT,attempt)
        finally:signal.alarm(0)
        base.durable(attempt/'DISK-RECOVERY.json',dict(**result,source_failure_preserved=True,time=time.time(),gpu_used=False))
        print(json.dumps(result))


if __name__=='__main__':main()
