"""One-off reviewed boundary migration; never run during a converter/GPU probe."""
import hashlib,json,os,shutil,time
from pathlib import Path
from run_stage import OUT,TASK,LOCK,save
from test.experiments.gufo_upstream_port_20261023.window import host,npu
ROOT=OUT/'model-job-c03';R=ROOT/'repair-012-lean-disk';L=OUT/'disk-boundary';C=OUT/'capture-lean'
assert not R.exists(),'Migration already attempted; inspect its journal'
lease=json.loads((L/'BOUNDARY.json').read_text());assert lease['status']=='reserved'
os.kill(lease['pid'],0)
assert 'disk_boundary_reserve.py' in Path(f'/proc/{lease["pid"]}/cmdline').read_text()
locks=Path('/proc/locks').read_text().splitlines()
for p in (LOCK,ROOT/'JOB.lock'):
    s=p.stat();dev=f'{os.major(s.st_dev):02x}:{os.minor(s.st_dev):02x}:{s.st_ino}'
    assert any(len(v:=l.split())>=6 and v[1:4]==['FLOCK','ADVISORY','WRITE'] and v[4]==str(lease['pid']) and v[5]==dev for l in locks)
assert json.loads((OUT/'WATCH-CONTROL.json').read_text())['enabled']
state=json.loads((C/'STATE.json').read_text());assert state['status']=='completed_pending_review' and not state['cleanup_errors'] and not state['returned_KFD_contexts']
prior=json.loads((ROOT/'module-017/STATE.json').read_text());assert prior['status']=='passed' and not prior['cleanup_errors'] and not prior['returned_KFD_contexts']
checkpoint=json.loads((OUT/'work/ckpt/job.json').read_text());assert checkpoint==dict(next_module_idx=9,q_strategy=None,bad_rows=[])
assert not host()['KFD_contexts'] and not npu()
review=json.loads((C/'REVIEW.json').read_text());assert review['status']=='qualified_for_exact_capture_integration'
for p,h in review['identity'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
for p,h in review['runtime_identity'].items():assert hashlib.sha256((TASK/p).read_bytes()).hexdigest()==h,p
for p,h in review['evidence_sha256'].items():assert hashlib.sha256((C/p).read_bytes()).hexdigest()==h,p
disk_review=json.loads((L/'REVIEW.json').read_text())
for p,h in disk_review['identity'].items():assert hashlib.sha256((TASK/p).read_bytes()).hexdigest()==h,p
assert '432 passed' in (C/'INFRA.log').read_text()
old=json.loads((ROOT/'FROZEN.json').read_text());assert len(old)==433
for p,h in old.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
R.mkdir()
for name in ('FROZEN.json','STATE.json'):shutil.copy2(ROOT/name,R/('previous-'+name))
for name in ('model_job.py','convert_main.py'):shutil.copy2(TASK/name,R/('previous-'+name))
for name in ('job.json','state.rows.json'):shutil.copy2(OUT/'work/ckpt'/name,R/('checkpoint-'+name))
save(R/'MIGRATION.json',dict(status='prepared',time=time.time(),checkpoint=checkpoint,lease_pid=lease['pid']))
for src,dst in [('model_job_lean_candidate.py','model_job.py'),('convert_main_lean_candidate.py','convert_main.py')]:
    tmp=TASK/(dst+'.new');tmp.write_bytes((TASK/src).read_bytes());os.replace(tmp,TASK/dst)
source=(TASK/'model_job.py').read_text();ns={'__name__':'boundary_validation'}
# Read-only gates before the JOB lock, already owned by the verified lease.
exec(compile(source[:source.index("guard=(ROOT/'JOB.lock')")],str(TASK/'model_job.py'),'exec'),ns)
a=source.index('paths=[TASK/name');b=source.index("if (ROOT/'FROZEN.json')",a)
exec(compile(source[a:b],'<frozen-paths>','exec'),ns)
new=ns['identity'];assert set(old)<=set(new)
changed=[p for p,h in old.items() if new[p]!=h]
assert set(changed)=={str(TASK/'model_job.py'),str(TASK/'convert_main.py')}
save(ROOT/'FROZEN.json',new);save(R/'FROZEN.json',new)
save(R/'MIGRATION.json',dict(status='installed_pending_admission',time=time.time(),checkpoint=checkpoint,previous_count=len(old),new_count=len(new),changed_existing=changed,added=sorted(set(new)-set(old)),lease_pid=lease['pid'],infra='432 passed in18.80s'))
print(json.dumps(dict(status='installed_pending_admission',frozen=len(new),checkpoint=checkpoint)))
