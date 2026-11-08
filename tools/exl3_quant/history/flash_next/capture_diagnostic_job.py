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

root = OUT / 'capture-profile-quarter'
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
            # Quarter-only repair reads the retained idx2 aggregate read-only.
            # Reuse the completed profile instead of copying/rerunning its64 rows.
            assert available() >= 28*2**30 and shutil.disk_usage(OUT).free >= 150*2**30
            h = host()
            assert quiet(h) and not npu(), 'Fresh GPU admission after fixture preparation'
            name, label = 'gufo-capture-profile-quarter', uuid.uuid4().hex
            assert subprocess.run(['docker', 'inspect', name], capture_output=True).returncode != 0
            command = receipt['command'].copy()
            command[command.index('--name')+1] = name
            command[command.index('--label')+1] = 'gufo.quant.run='+label
            assert command[-1] == '/task/convert_main.py'
            command[-1] = '/task/capture_diagnostic_probe.py'
            # This diagnostic may read production inputs, never write checkpoints/pack.
            index = command.index('--workdir')
            command[index:index] = [
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
                assert time.time()-window_started < 1200, 'Diagnostic admission deadline'
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        mem = available()
                        state['minimum_available_bytes'] = min(state['minimum_available_bytes'], mem)
                        assert mem >= 8*2**30 and shutil.disk_usage(OUT).free >= 100*2**30
                        assert time.time()-state['admitted'] < 1200, 'Bounded diagnostic deadline'
                        state['updated'] = time.time(); save(root / 'STATE.json', state)
                        time.sleep(2)
                    assert child.returncode == 0, f'Diagnostic exit {child.returncode}'
                    assert json.loads((root / 'SUMMARY.json').read_text())['status'] == 'diagnostics_complete_review_required'
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
        message = ('[사용자가 승인한 원인 프로파일 및25% 선별 검사 완료] '
            '/home/m2on/projects/gufo/plan/exl3/F3-CONVERSION-WATCH.md와 '
            '/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104/capture-profile-quarter/'
            '{STATE,SUMMARY}.json, quarter/{PARITY,SUMMARY,RESULTS}.json 및run.log를 읽어라. 프로파일은 형제 디렉터리 capture-profile-quarter-attempt-002/PROFILE.json을 재사용한다. '
            '본체는 사용자의 module012 포기 지시로 idx10에서 멈춰 있다. '
            'GPU 정리 확인 후 독립 리뷰하라. 저장된control 대비 수치와64행 CPU/GPU/flush/실제512행kernel trace를 해석하고, '
            '577행503322토큰으로 새로 양자화한 layer0 대표2projection을 기존full2M packed와 비교하라. '
            '미선택c03 평가는 독립holdout이 아니며, 2projection 결과를 전체품질동등이나 전체변환시간으로 주장하지 않는다. '
            'rotation sign 불일치/worker산술차이도 확인한다. 실패시 원인을 수정하고 영향을받은 작은 검사만 다시 한다. '
            '원인 및25% 적용 여부를 판단하기 전에 본체를 자동 재개하지 말고, 필요한 추가 검증/매핑을 이어서 진행하라. '
            '중복GPU 작업/전체baseline 재실행/완료checkpoint 삭제는 하지 않는다.')
        try:
            result = subprocess.run(['/home/m2on/.local/bin/codex', 'queue', '--thread', control['thread'],
                                     '--message', message, '--resume'], capture_output=True, text=True, timeout=45)
            save(root / 'FOLLOWUP.json', dict(exit=result.returncode, stdout=result.stdout, stderr=result.stderr))
        except (OSError, subprocess.TimeoutExpired) as error:
            save(root / 'FOLLOWUP.json', dict(error=repr(error)))
