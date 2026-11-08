"""At a natural module boundary, diagnose the unmatched capture speed result.

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
from test.experiments.gufo_upstream_port_20261023.window import host, npu, quiet

root = OUT / 'capture-regression'
root.mkdir(exist_ok=True)
owner = (root / 'OWNER.lock').open('a')
fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (root / 'STATE.json').exists(), 'Retain and inspect an existing attempt before rerunning'
state = dict(status='waiting_boundary', started=time.time(), pid=os.getpid())
save(root / 'STATE.json', state)


def stopped(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')
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
            previous = max((OUT / 'model-job-c03').glob('module-[0-9]*'))
            receipt = json.loads((previous / 'STATE.json').read_text())
            assert receipt['status'] in ('passed', 'passed_after_cpu_checkpoint'), receipt
            assert receipt['returned_KFD_contexts'] == [] and not receipt['cleanup_errors']
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
            name, label = 'gufo-capture-regression', uuid.uuid4().hex
            assert subprocess.run(['docker', 'inspect', name], capture_output=True).returncode != 0
            command = receipt['command'].copy()
            command[command.index('--name')+1] = name
            command[command.index('--label')+1] = 'gufo.quant.run='+label
            assert command[-1] == '/task/convert_main.py'
            command[-1] = '/task/capture_regression_probe.py'
            identity = {str(TASK / x): hashlib.sha256((TASK / x).read_bytes()).hexdigest()
                        for x in ('capture_regression_probe.py', 'capture_regression_job.py', 'capture_fast.py')}
            state.update(status='running', admitted=time.time(), command=command, identity=identity,
                         previous_module=previous.name, frozen=frozen,
                         checkpoint=json.loads((OUT / 'work/ckpt/job.json').read_text()),
                         minimum_available_bytes=available())
            save(root / 'STATE.json', state)
            with (root / 'run.log').open('x') as log:
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
                    assert json.loads((root / 'SUMMARY.json').read_text())['status'] == 'matched_capture_regression_pass'
                    state['status'] = 'passed_pending_review'
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
        message = ('[사용자가 요청한 변환 속도 회귀 진단 완료] '
                   '/home/m2on/projects/gufo/plan/exl3/F3-CONVERSION-WATCH.md와 '
                   '/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104/'
                   'capture-regression/{STATE,SUMMARY}.json 및 run.log를 읽어라. '
                   '현재 변환은 자연 체크포인트 경계에서 진단을 위해 멈춘 상태다. '
                   'GPU 정리와 최신 사용자 지시를 확인하고, 비교 결과를 독립 리뷰한 뒤 '
                   '필요한 최소 수정/검증 또는 기존 설정 그대로 체크포인트에서 재개하라. '
                   '원인을 추정으로 단정하지 말고, 중복 변환·기존 baseline 전체 재실행은 하지 않는다.')
        try:
            result = subprocess.run(['/home/m2on/.local/bin/codex', 'queue', '--thread', control['thread'],
                                     '--message', message, '--resume'], capture_output=True, text=True, timeout=45)
            save(root / 'FOLLOWUP.json', dict(exit=result.returncode, stdout=result.stdout, stderr=result.stderr))
        except (OSError, subprocess.TimeoutExpired) as error:
            save(root / 'FOLLOWUP.json', dict(error=repr(error)))
