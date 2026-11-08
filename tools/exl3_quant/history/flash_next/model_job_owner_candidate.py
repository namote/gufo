"""Wait for the validated ngram artifact, then advance checkpointed model modules serially."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
from run_stage import OUT,TASK,LOCK,IMAGE,available,save
from test.experiments.gufo_upstream_port_20261023.window import host,npu,quiet
from owned_cleanup import cleanup_owned
from finish_checkpoint import ModuleWindowTimeout,run_cpu

ROOT=OUT/'model-job-c03';ROOT.mkdir(exist_ok=True)
long_gate=json.loads((OUT/'long-row-pilot/SUMMARY.json').read_text())
assert long_gate['status']=='long_row_screen_pass'
assert long_gate['calibration_sha256']==json.loads((TASK/'c03-input-ids.json').read_text())['sha256']
assert long_gate['pilot_sha256']==hashlib.sha256((TASK/'long_row_pilot.py').read_bytes()).hexdigest()
long_state=json.loads((OUT/'long-row-pilot/STATE.json').read_text())
assert long_state['status']=='passed' and long_state['returned_KFD_contexts']==[]
assert long_state['pilot_sha256']==long_gate['pilot_sha256']
speed_root=OUT/'capture-speed'
speed_state=json.loads((speed_root/'STATE.json').read_text())
assert speed_state['status']=='passed' and not speed_state['cleanup_errors']
assert speed_state['returned_KFD_contexts']==[]
speed_gate=json.loads((speed_root/'SUMMARY.json').read_text())
assert speed_gate['status']=='capture_exact_pass'
assert speed_gate['calibration_sha256']==long_gate['calibration_sha256']
assert not speed_gate['fp32_output_shortcut']
assert json.loads((speed_root/'SELECTION.json').read_text())['mode']=='skip_down'
for path,digest in speed_state['identity'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
speed_provenance=json.loads((speed_root/'PROVENANCE.json').read_text())
# Only these two reviewed integration callers change in this migration. The
# arithmetic runtime, calibration, recipe and all other probe dependencies stay pinned.
for path,digest in speed_provenance['frozen'].items():
    if Path(path) not in {TASK/'model_job.py',TASK/'convert_main.py'}:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
cpu_root=OUT/'capture-optimization-cpu-active-runtime'
for receipt,candidate in [('FAST-SCOPE-CPU-UNWRAP.json','capture_fast.py'),
                          ('FOREACH-CPU.json','capture_foreach.py'),
                          ('PROGRESS-CPU.json','capture_progress.py')]:
    cpu=json.loads((cpu_root/receipt).read_text())
    assert cpu['status']=='pass' and not cpu['gpu_used']
    assert cpu['candidate_sha256']==hashlib.sha256((TASK/candidate).read_bytes()).hexdigest()
lt_root=OUT/'capture-lt-support'
lt_gate=json.loads((lt_root/'SUMMARY.json').read_text())
lt_state=json.loads((lt_root/'STATE.json').read_text())
assert lt_gate['status']=='main_shapes_attention_qualified'
assert lt_state['status']=='completed_pending_review' and not lt_state['cleanup_errors'] and not lt_state['returned_KFD_contexts']
assert json.loads((lt_root/'REVIEW.json').read_text())['status']=='qualified_for_idx10_resume'
lt_contract=json.loads((TASK/'capture-lt-contract.json').read_text())
assert lt_contract['status']=='qualified_for_idx10_resume'
assert lt_contract['calibration_sha256']==long_gate['calibration_sha256']==lt_gate['calibration_sha256']
assert lt_contract['allowed_shapes']==lt_gate['allowed_shapes']
assert lt_contract['native_sha256']==hashlib.sha256((TASK/'native/capture-gram.so').read_bytes()).hexdigest()==lt_gate['native_sha256']
assert lt_contract['helper_sha256']==hashlib.sha256((TASK/'capture_lt.py').read_bytes()).hexdigest()==lt_gate['helper_sha256']
assert json.loads((OUT/'capture-lt-quality/REVIEW.json').read_text())['status']=='reviewed_prefer_original_f16_lt_for_next_integration_qualification'
owner_root=OUT/'capture-owner'
owner_review=json.loads((owner_root/'REVIEW.json').read_text())
assert owner_review['status']=='owner_fix_qualified_by_saved_valid_control'
assert owner_review['audit_sha256']==hashlib.sha256((owner_root/'AUDIT.json').read_bytes()).hexdigest()
owner_cpu=json.loads((owner_root/'CPU.json').read_text())
assert owner_cpu['status']=='pass'
assert hashlib.sha256((TASK/'capture_owner.py').read_bytes()).hexdigest()==owner_cpu['identity'][str(TASK/'capture_owner.py')]
prefix_root=OUT/'prefix-recovery'
prefix=json.loads((prefix_root/'SUMMARY.json').read_text())
assert prefix['status']=='prefix_replay_complete' and prefix['mode']=='full'
assert prefix['rows']==2309 and prefix['next_module_idx']==6 and prefix['full_first3_pilot_exact']
prefix_state=json.loads((prefix_root/'STATE.json').read_text())
assert prefix_state['status']=='completed_pending_review' and not prefix_state['cleanup_errors'] and not prefix_state['returned_KFD_contexts']
assert json.loads((prefix_root/'REVIEW.json').read_text())['status']=='qualified_for_idx6_promotion'
restored=json.loads((prefix_root/'PROMOTION.json').read_text());assert restored['status']=='committed_idx6'
current_idx=json.loads((OUT/'work/ckpt/job.json').read_text())['next_module_idx'];assert current_idx>=6
if current_idx==6:
    assert hashlib.sha256((OUT/'work/ckpt/state.rows.json').read_bytes()).hexdigest()==restored['manifest_sha256']
    assert hashlib.sha256((OUT/'work/ckpt/original_input_ids.safetensors').read_bytes()).hexdigest()==restored['ids_sha256']
guard=(ROOT/'JOB.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'work/ckpt_cpu_pending').exists(),'Incomplete CPU checkpoint stage needs recovery'
for journal in ROOT.glob('module-*/CPU-COMMIT.json'):
    assert json.loads(journal.read_text())['phase']=='committed',f'Incomplete CPU commit needs recovery: {journal}'
paths=[TASK/name for name in ('model_job.py','convert_main.py','bank_pilot.py',
    'b1-recipe.json','c02-input-ids.safetensors','run_stage.py','ragged_calibration.py',
    'c03-input-ids.safetensors','c03-input-ids.json','long_row_pilot.py','disk_state.py',
    'row_checkpoint.py','owned_cleanup.py','finish_checkpoint.py',
    'capture_fast.py','capture_foreach.py','capture_progress.py','capture_speed_probe.py','capture_probe_job.py')]
paths += [speed_root/name for name in ('SUMMARY.json','STATE.json','PROVENANCE.json','SELECTION.json')]
paths += [cpu_root/name for name in ('FAST-SCOPE-CPU-UNWRAP.json','FOREACH-CPU.json','PROGRESS-CPU.json')]
paths += [TASK/name for name in ('capture_lt_runtime.py','capture_lt.py','capture_gram.hip','native/capture-gram.so','capture-lt-contract.json')]
paths += [lt_root/name for name in ('SUMMARY.json','STATE.json','REVIEW.json','RUNTIME-CPU.json')]
paths += [OUT/'capture-lt-quality/REVIEW.json',OUT/'capture-lt-evaluate/REVIEW.json',OUT/'capture-gram/BUILD.json']
paths += [TASK/'capture_owner.py']
paths += [owner_root/name for name in ('REVIEW.json','CPU.json','AUDIT.json')]
paths += [prefix_root/name for name in ('INPUTS.json','SUMMARY.json','STATE.json','REVIEW.json','CPU.json','PROMOTION.json','OWNER-INTEGRATION-CPU.json')]
paths += [TASK/name for name in ('prefix_replay.py','prefix_replay_run.py','prefix_replay_job.py')]
paths += [OUT/'long-row-pilot/SUMMARY.json']
paths += [OUT/'long-row-pilot/STATE.json']
runtime=Path('/home/m2on/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/kyojin_runtime')
paths+=sorted((runtime/'source/exllamav3').rglob('*.py'))
paths+=[OUT/'model-inventory.json']
identity={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
if (ROOT/'FROZEN.json').exists():assert json.loads((ROOT/'FROZEN.json').read_text())==identity
else:save(ROOT/'FROZEN.json',identity)
state=dict(status='waiting_for_ngram',started=time.time(),pid=os.getpid(),modules=[])
save(ROOT/'STATE.json',state)

def stop_signal(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
signal.signal(signal.SIGTERM,stop_signal)

def check_identity():
    for p,h in identity.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    frozen=json.loads((OUT/'ngram-full/FROZEN.json').read_text())
    for p,h in frozen['code'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    native=Path(frozen['native']);s=native.stat()
    assert [s.st_size,s.st_mtime_ns,s.st_ino]==frozen['native_stat']
    inv=json.loads((OUT/'source-inventory.json').read_text());source=Path(inv['source'])
    for f in inv['files']:
        s=(source/f['name']).stat();assert (s.st_size,s.st_mtime_ns)==(f['bytes'],f['mtime_ns'])

try:
    while True:
        ng=json.loads((OUT/'ngram-full/JOB.json').read_text())
        assert ng['status']!='stopped',ng
        if ng['status']=='ngram_complete':break
        assert time.time()-state['started']<9*3600,'Ngram wait budget ended'
        state['updated']=time.time();save(ROOT/'STATE.json',state);time.sleep(10)
    number=1+max([int(p.name.split('-')[1]) for p in ROOT.glob('module-*') if p.is_dir()]+[0])
    while not (OUT/'MODEL-CONVERSION.json').exists():
        check_identity()
        assert time.time()-state['started']<96*3600,'Conversion budget ended; checkpoints retained'
        with LOCK.open('a') as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # A just-finished owned window can leave a transient busy sample.
            # Wait for quiet, but never retry a module or admit another KFD owner.
            for admission_attempt in range(30):
                h=host();assert not h['KFD_contexts'],'GPU owner active'
                if quiet(h):break
                time.sleep(2)
            assert quiet(h) and not npu(),'GPU not quiet before module'
            assert available()>=28*2**30 and shutil.disk_usage(OUT).free>=150*2**30
            label=uuid.uuid4().hex;name='gufo-quant-model'
            assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
            command=json.loads((OUT/'ngram-full/STATE.json').read_text())['command'].copy()
            assert IMAGE in command and command[-1]=='/task/ngram_full.py'
            command[command.index('--name')+1]=name
            command[command.index('--label')+1]='gufo.quant.run='+label
            ix=command.index('EXL3_QT_KERNEL=opt');del command[ix-1:ix+1]
            ix=command.index('--workdir');command[ix:ix]=['-e','EXL3_QUANT_THREADS=2','-e','EXL3_ENABLE_GROUPING=0','-e','EXL3_HIP_F32OUT_VIA_F16=0']
            command[-1]='/task/convert_main.py'
            receipt=ROOT/f'module-{number:03d}';receipt.mkdir()
            scratch=receipt/'row-scratch'
            ix=command.index('--workdir')
            command[ix:ix]=['-e',f'GUFO_STATE_SCRATCH=/output/{scratch.relative_to(OUT)}']
            run=dict(started=time.time(),status='running',command=command,minimum_available_bytes=available(),deadline_seconds=7200)
            save(receipt/'STATE.json',run)
            state.update(status='converting_main_model',module_attempt=number,updated=time.time());save(ROOT/'STATE.json',state)
            with (receipt/'run.log').open('w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                window_expired=False
                try:
                    while child.poll() is None:
                        mem=available();run['minimum_available_bytes']=min(run['minimum_available_bytes'],mem)
                        assert mem>=8*2**30,'Host reserve below 8 GiB'
                        assert shutil.disk_usage(OUT).free>=100*2**30,'Disk reserve below 100 GiB'
                        if time.time()-run['started']>run['deadline_seconds']:
                            raise ModuleWindowTimeout('One-module window exceeded; inspect before resuming')
                        run['updated']=time.time();save(receipt/'STATE.json',run);time.sleep(2)
                    assert child.returncode==0,f'Converter exit {child.returncode}'
                    run['status']='passed'
                except ModuleWindowTimeout as e:
                    # Only this explicit time boundary can proceed to CPU-only commit.
                    # The helper independently requires complete, validated block output.
                    window_expired=True;run.update(status='failed',error=repr(e))
                except BaseException as e:
                    run.update(status='failed',error=repr(e));raise
                finally:
                    cleanup=cleanup_owned(name,label,child,host)
                    run.update(cleanup)
                    if cleanup['cleanup_errors']:run['status']='cleanup_failed'
                    run['finished']=time.time();save(receipt/'STATE.json',run)
                    # Preserve the original conversion failure; report cleanup separately.
                    if cleanup['cleanup_errors'] and 'error' not in run:
                        raise RuntimeError(f"Cleanup failed: {cleanup['cleanup_errors']}")
                    # Failed attempts retain complete rows for explicit recovery.
                    if run['status']=='passed' and scratch.exists():shutil.rmtree(scratch)
                if window_expired:
                    assert not run['cleanup_errors'] and run['returned_KFD_contexts']==[]
                    try:
                        run['cpu_checkpoint']=run_cpu(receipt)
                    except BaseException as e:
                        run['checkpoint_error']=repr(e);save(receipt/'STATE.json',run);raise
                    run['status']='passed_after_cpu_checkpoint';save(receipt/'STATE.json',run)
                    # Successful commit owns the new rows; retire only redundant,
                    # attempt-owned inputs/partial links, as upstream KEEP_OLD=0 does.
                    for obsolete in ('row-scratch','previous-checkpoint','partial-checkpoint'):
                        path=receipt/obsolete
                        if path.exists():shutil.rmtree(path)
            checkpoint=OUT/'work/ckpt/job.json'
            state['modules'].append(dict(attempt=number,next_module_idx=json.loads(checkpoint.read_text())['next_module_idx']))
            state['updated']=time.time();save(ROOT/'STATE.json',state);number+=1
    state.update(status='converted_pending_full_quality_performance_validation',finished=time.time());save(ROOT/'STATE.json',state)
except BaseException as e:
    state.update(status='stopped',error=repr(e),finished=time.time());save(ROOT/'STATE.json',state);raise
