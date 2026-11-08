"""Own only a bounded recovery diagnostic, retaining production admission rules."""
import fcntl
import json
from pathlib import Path
import subprocess
import time
import uuid
from run_stage import OUT, LOCK, available, save
from owned_cleanup import cleanup_owned
from test.experiments.gufo_upstream_port_20261023.window import host, npu

R=OUT/'model-job-c03/repair-015-quant-stall'
with LOCK.open('a') as gpu, (OUT/'model-job-c03/JOB.lock').open('a') as job:
    fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fcntl.flock(job,fcntl.LOCK_EX|fcntl.LOCK_NB)
    h=host();assert not h['KFD_contexts'] and not npu()
    assert available()>28*2**30
    assert json.loads((OUT/'work/ckpt/job.json').read_text())['next_module_idx']==47
    old=json.loads((OUT/'model-job-c03/module-056/STATE.json').read_text())
    assert old['status']=='failed' and old['cleanup_errors']==[] and old['returned_KFD_contexts']==[]
    cmd=old['command'].copy();name='gufo-quant-stall-probe';label=uuid.uuid4().hex
    assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
    cmd[cmd.index('--name')+1]=name;cmd[cmd.index('--label')+1]='gufo.quant.run='+label
    cmd[-1]='/task/quant_stall_probe.py'
    state=dict(status='running',started=time.time(),command=cmd,pre=h,diagnostic_admission='KFD0/NPU0/exclusive locks; existing busy100 allowed solely for bounded fresh-context liveness, not production/timing',deadline_seconds=120)
    save(R/'PROBE-WINDOW.json',state)
    child=None
    try:
        with (R/'probe.log').open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                assert available()>=8*2**30
                assert time.time()-state['started']<120,'Diagnostic timeout'
                time.sleep(1)
            state['exit_code']=child.returncode
            assert child.returncode==0,'Diagnostic failed'
            assert json.loads((R/'PROBE.json').read_text())['status']=='passed'
            state['status']='passed'
    except BaseException as e:
        state.update(status='failed',error=repr(e))
    finally:
        if child is not None:state.update(cleanup_owned(name,label,child,host))
        state['post']=host();state['finished']=time.time();save(R/'PROBE-WINDOW.json',state)
    print(json.dumps({k:v for k,v in state.items() if k not in ['command','pre','post']}),flush=True)
    assert state['status']=='passed' and not state.get('cleanup_errors') and state.get('returned_KFD_contexts')==[]
