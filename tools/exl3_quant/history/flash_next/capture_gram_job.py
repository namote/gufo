"""At a natural module boundary, compare the actual converter capture loop before selecting rollback or retention.

No running converter/code is modified. Keep the existing checkpoint, take one
bounded GPU window, then queue review/resumption in the already authorized chat.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid
from run_stage import OUT, TASK, LOCK, available, save
from owned_cleanup import cleanup_owned
from row_checkpoint import open_rows
from test.experiments.gufo_upstream_port_20261023.window import host, npu, quiet

root = OUT / 'capture-gram'
root.mkdir(exist_ok=True)
owner = (root / 'OWNER.lock').open('a')
fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (root / 'STATE.json').exists(), 'Retain and inspect an existing attempt before rerunning'
state = dict(status='waiting_boundary', started=time.time(), pid=os.getpid())
save(root / 'STATE.json', state)


def stopped(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(4*2**20):
            digest.update(chunk)
    return digest.hexdigest()


signal.signal(signal.SIGTERM, stopped)
try:
    with LOCK.open('a') as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        with (OUT / 'model-job-c03/JOB.lock').open('a') as job_lock:
            until = time.monotonic()+180
            while True:
                try:
                    fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    assert time.monotonic() < until
                    time.sleep(1)
            assert json.loads((OUT / 'WATCH-CONTROL.json').read_text())['enabled']
            cpu=json.loads((root/'CPU.json').read_text())
            assert cpu['status']=='pass' and cpu['gpu_used'] is False
            for path,digest in cpu['identity'].items():
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
            build=json.loads((root/'BUILD.json').read_text())
            assert build['status']=='compiled' and not build['gpu_used']
            assert build['source_sha256']==hashlib.sha256((TASK/'capture_gram.hip').read_bytes()).hexdigest()
            assert build['binary_sha256']==hashlib.sha256((root/'gram.so').read_bytes()).hexdigest()
            prior=json.loads((OUT/'capture-profile-quarter/STATE.json').read_text())
            assert prior['status']=='completed_pending_review' and not prior['cleanup_errors'] and not prior['returned_KFD_contexts']
            assert json.loads((OUT/'capture-profile-quarter/quarter/SUMMARY.json').read_text())['status']=='quarter_projection_screen_complete'
            prior_subset=json.loads((OUT/'capture-subset/STATE.json').read_text())
            assert prior_subset['cleanup_errors']==[] and prior_subset['returned_KFD_contexts']==[]
            previous=OUT/'model-job-c03/module-012'
            assert previous==max((OUT/'model-job-c03').glob('module-[0-9]*'))
            receipt=json.loads((previous/'STATE.json').read_text())
            assert receipt['error']=="KeyboardInterrupt('signal 15')"
            assert receipt['cleanup_errors']==[] and receipt['returned_KFD_contexts']==[]
            stop=json.loads((OUT/'capture-quarter/STOP-FOR-DIAGNOSTIC.json').read_text())
            assert stop['status']=='stopped_verified'
            assert json.loads((OUT/'work/ckpt/job.json').read_text())==stop['checkpoint']
            frozen = json.loads((OUT / 'model-job-c03/FROZEN.json').read_text())
            for path, digest in frozen.items():
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
            ngram_frozen = json.loads((OUT / 'ngram-full/FROZEN.json').read_text())
            for path, digest in ngram_frozen['code'].items():
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
            stat = Path(ngram_frozen['native']).stat()
            assert [stat.st_size, stat.st_mtime_ns, stat.st_ino] == ngram_frozen['native_stat']
            inventory = json.loads((OUT / 'source-inventory.json').read_text())
            for entry in inventory['files']:
                stat = (Path(inventory['source']) / entry['name']).stat()
                assert (stat.st_size, stat.st_mtime_ns) == (entry['bytes'], entry['mtime_ns'])
            for _ in range(30):
                h = host()
                assert h['KFD_contexts'] == [], 'Do not interrupt another GPU owner'
                if quiet(h):
                    break
                time.sleep(2)
            assert quiet(h) and not npu()
            assert available() >= 28*2**30 and shutil.disk_usage(OUT).free >= 150*2**30
            window_started = time.time()
            name, label = 'gufo-capture-gram', uuid.uuid4().hex
            assert subprocess.run(['docker', 'inspect', name], capture_output=True).returncode != 0
            command = receipt['command'].copy()
            command[command.index('--name')+1] = name
            command[command.index('--label')+1] = 'gufo.quant.run='+label
            assert command[-1] == '/task/convert_main.py'
            command[-1] = '/task/capture_gram_probe.py'
            # This diagnostic may read production inputs, never write checkpoints/pack.
            index = command.index('--workdir')
            command[index:index] = [
                '--mount', f'type=bind,src={OUT / "capture-profile-quarter"},dst=/output/capture-profile-quarter,readonly',
                '--mount', f'type=bind,src={OUT / "work"},dst=/output/work,readonly',
                '--mount', f'type=bind,src={OUT / "pack"},dst=/output/pack,readonly',
                '--mount', f'type=bind,src={OUT / "capture-pipeline"},dst=/output/capture-pipeline,readonly',
                '--mount', f'type=bind,src={OUT / "model-job-c03/repair-002-checkpoint"},dst=/output/model-job-c03/repair-002-checkpoint,readonly',
                '--mount', f'type=bind,src={OUT / "capture-quarter"},dst=/output/capture-quarter,readonly',
                '--mount', f'type=bind,src={OUT / "capture-speed"},dst=/output/capture-speed,readonly',
                '--mount', f'type=bind,src={OUT / "capture-regression"},dst=/output/capture-regression,readonly',

            ]
            identity=cpu['identity']
            state.update(status='running', admitted=window_started, command=command, identity=identity,
                         hook_receipt_sha256=hashlib.sha256((root / 'CPU.json').read_bytes()).hexdigest(),
                         previous_module=previous.name, frozen=frozen,
                         checkpoint=json.loads((OUT / 'work/ckpt/job.json').read_text()),
                         minimum_available_bytes=available())
            save(root / 'STATE.json', state)
            with (root / 'run.log').open('x') as log:
                assert time.time()-window_started < 600, 'Diagnostic admission deadline'
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        mem = available()
                        state['minimum_available_bytes'] = min(state['minimum_available_bytes'], mem)
                        assert mem >= 8*2**30 and shutil.disk_usage(OUT).free >= 100*2**30
                        assert time.time()-state['admitted'] < 600, 'Bounded diagnostic deadline'
                        state['updated'] = time.time(); save(root / 'STATE.json', state)
                        time.sleep(2)
                    assert child.returncode == 0, f'Diagnostic exit {child.returncode}'
                    assert json.loads((root / 'SUMMARY.json').read_text())['status'] == 'gram_operator_screen_complete'
                    state['status'] = 'completed_pending_review'
                except BaseException as error:
                    state.update(status='failed', error=repr(error))
                    raise
                finally:
                    state.update(cleanup_owned(name, label, child, host))
                    if state['cleanup_errors'] or state['returned_KFD_contexts']:
                        state['status'] = 'cleanup_failed'
                    save(root / 'STATE.json', state)
                    if state['status'] == 'cleanup_failed' and 'error' not in state:
                        raise RuntimeError('Diagnostic cleanup failed')
except BaseException as error:
    state.update(status='failed', error=repr(error))
    raise
finally:
    # The with-statements have released both leases before requesting review.
    state['finished'] = time.time(); save(root / 'STATE.json', state)
    control = json.loads((OUT / 'WATCH-CONTROL.json').read_text())
    if control['enabled']:
        message = ('[사용자가 승인한 통계Gram WMMA/Lt 연산검사 완료] '
            '/home/m2on/projects/gufo/plan/exl3/F3-CONVERSION-WATCH.md와 '
            '/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104/capture-gram/'
            '{STATE,SUMMARY,RESULTS,PROGRESS,BACKEND,ANALYTIC,BUILD}.json 및run.log/trace를읽어라. '
            'GPUcleanup/독립리뷰후torchFP32BMM실제커널과직접LtFP32/원래FP16Lt/WMMA의수치·성능을판단하라. '
            'FP32입력을낮추지않고원래FP16인입력만WMMA에사용했다. strict exact와별도sampleFP64오차를구분한다. '
            '실패자료/real-input-control재사용, 작은실패검사만수정. 미지원Lt는명시적unsupported이며fallback없다. '
            'packing/masking/H누적제외한연산검사로전체속도/품질을주장하지않는다. '
            '유망후보만실제flush정확성/시간검증으로이어가고,25%보류·본체idx10중지를유지한다. '
            '중복GPU/전체baseline반복/완료checkpoint삭제/실행중코드수정금지.')
        try:
            result = subprocess.run(['/home/m2on/.local/bin/codex', 'queue', '--thread', control['thread'],
                                     '--message', message, '--resume'], capture_output=True, text=True, timeout=45)
            save(root / 'FOLLOWUP.json', dict(exit=result.returncode, stdout=result.stdout, stderr=result.stderr))
        except (OSError, subprocess.TimeoutExpired) as error:
            save(root / 'FOLLOWUP.json', dict(error=repr(error)))
