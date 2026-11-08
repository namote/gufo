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

root = OUT / 'capture-pipeline'
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
            cpu = json.loads((root / 'HOOK-CPU-FIXTURE.json').read_text())
            assert cpu['status'] == 'pass' and cpu['gpu_used'] is False
            assert cpu['script_sha256'] == hashlib.sha256((TASK / 'capture_pipeline_probe.py').read_bytes()).hexdigest()
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
            window_started = time.time()
            # DiskRows creates private hardlinks. A read-only nested mount cannot
            # link into the outer scratch mount (EXDEV). Copy only the 64 selected
            # rows so the original checkpoint stays read-only with no writable alias.
            fixture = root / 'input'
            fixture.mkdir()
            (fixture / 'state.rows').mkdir()
            source = OUT / 'work/ckpt/state.safetensors'
            paths, _ = open_rows(source)
            manifest = json.loads(source.with_suffix('.rows.json').read_text())
            manifest['rows'] = manifest['rows'][:64]
            copies = []
            for i, path in enumerate(paths[:64]):
                assert time.time()-window_started < 1200, 'Diagnostic preparation deadline'
                assert not path.is_symlink()
                target = fixture / 'state.rows' / f'{i}.safetensors'
                shutil.copyfile(path, target)
                digest = digest_file(path)
                assert digest_file(target) == digest
                assert target.stat().st_ino != path.stat().st_ino
                copies.append(dict(source=str(path), target=str(target), sha256=digest,
                                   bytes=target.stat().st_size))
            assert len(copies) == 64
            save(fixture / 'state.rows.json', manifest)
            save(root / 'INPUT-COPY.json', dict(rows=copies, seconds=time.time()-window_started,
                 bytes=sum(x['bytes'] for x in copies), reason='Private row files on the same SSD; protect the original checkpoint with a read-only mount'))
            assert available() >= 28*2**30 and shutil.disk_usage(OUT).free >= 150*2**30
            h = host()
            assert quiet(h) and not npu(), 'Fresh GPU admission after fixture preparation'
            name, label = 'gufo-capture-pipeline', uuid.uuid4().hex
            assert subprocess.run(['docker', 'inspect', name], capture_output=True).returncode != 0
            command = receipt['command'].copy()
            command[command.index('--name')+1] = name
            command[command.index('--label')+1] = 'gufo.quant.run='+label
            assert command[-1] == '/task/convert_main.py'
            command[-1] = '/task/capture_pipeline_probe.py'
            # This diagnostic may read production inputs, never write checkpoints/pack.
            index = command.index('--workdir')
            command[index:index] = [
                '--mount', f'type=bind,src={OUT / "work"},dst=/output/work,readonly',
                '--mount', f'type=bind,src={OUT / "pack"},dst=/output/pack,readonly',
            ]
            identity = {str(TASK / x): hashlib.sha256((TASK / x).read_bytes()).hexdigest()
                        for x in ('capture_pipeline_probe.py', 'capture_pipeline_job.py', 'capture_fast.py', 'capture_progress.py', 'disk_state.py')}
            state.update(status='running', admitted=window_started, command=command, identity=identity,
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
                    assert json.loads((root / 'SUMMARY.json').read_text())['status'] == 'production_pipeline_pass'
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
        message = ('[사용자가 승인한 실제 변환 경로 원복 비교 완료] '
                   '/home/m2on/projects/gufo/plan/exl3/F3-CONVERSION-WATCH.md와 '
                   '/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104/'
                   'capture-pipeline/{STATE,SUMMARY}.json 및 run.log를 읽어라. '
                   '현재 변환은 자연 체크포인트 경계에서 진단을 위해 멈춘 상태다. '
                   'GPU 정리와 최신 사용자 지시를 확인하고, 비교 결과를 독립 리뷰한 뒤 '
                   '실제 연속 capture 루프의 원본/최적화 ABBA wall time과 정확성으로 빠른 쪽을 선택하라. 원복이면 이전 동결을 보존하고 최소 변경/검증/새 동결을 기록한 뒤 같은 체크포인트에서 재개하라. 최적화 유지면 코드 변경 없이 재개하라. 노이즈면 단정하지 말고 현재 설정을 유지하라. '
                   '원인을 추정으로 단정하지 말고, 중복 변환·기존 baseline 전체 재실행은 하지 않는다.')
        try:
            result = subprocess.run(['/home/m2on/.local/bin/codex', 'queue', '--thread', control['thread'],
                                     '--message', message, '--resume'], capture_output=True, text=True, timeout=45)
            save(root / 'FOLLOWUP.json', dict(exit=result.returncode, stdout=result.stdout, stderr=result.stderr))
        except (OSError, subprocess.TimeoutExpired) as error:
            save(root / 'FOLLOWUP.json', dict(error=repr(error)))
