"""One bounded IO probe under existing global GPU and conversion leases."""
import fcntl,hashlib,json,subprocess,time,uuid,shutil
from pathlib import Path
from run_stage import OUT,TASK,LOCK,available,save
from owned_cleanup import cleanup_owned
from row_checkpoint import open_rows,link_rows
from test.experiments.gufo_upstream_port_20261023.window import host,npu,quiet
R=OUT/'capture-io'
assert not (R/'RESULTS.json').exists(),'Reuse saved probe; no duplicate baseline'
locks=[]
for p in [LOCK,OUT/'model-job-c03/JOB.lock']:
    f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
h=host();assert not h['KFD_contexts'] and quiet(h) and not npu()
assert available()>28*2**30 and shutil.disk_usage(OUT).free>150*2**30
assert json.loads((OUT/'work/ckpt/job.json').read_text())['next_module_idx']==6
fixture=R/'fixture';fixture.mkdir()
paths,infos=open_rows(OUT/'work/ckpt/state.safetensors')
link_rows(paths[:16],fixture/'state.safetensors',infos[:16])
identity={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'disk_state_async.py',TASK/'test_disk_state_async.py',TASK/'state_io_probe.py',TASK/'state_io_job.py']}
cmd=json.loads((OUT/'model-job-c03/module-014/STATE.json').read_text())['command'].copy()
name='gufo-state-io-probe';label=uuid.uuid4().hex
assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
cmd[cmd.index('--name')+1]=name;cmd[cmd.index('--label')+1]='gufo.quant.run='+label
cmd[-1]='/task/state_io_probe.py'
ix=cmd.index('--workdir');cmd[ix:ix]=['--mount',f'type=bind,src={OUT}/work,dst=/output/work,readonly']
state=dict(status='running_probe',started=time.time(),identity=identity,command=cmd)
save(R/'STATE.json',state)
with (R/'run.log').open('w') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    try:
        while child.poll() is None:
            assert available()>8*2**30 and shutil.disk_usage(OUT).free>100*2**30
            assert time.time()-state['started']<240,'IO probe time limit'
            time.sleep(1)
        assert child.returncode==0,child.returncode
        state['status']='completed_pending_review'
    except BaseException as e:
        state.update(status='failed',error=repr(e));raise
    finally:
        state.update(cleanup_owned(name,label,child,host));state['finished']=time.time();save(R/'STATE.json',state)
