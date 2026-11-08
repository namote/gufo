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

root = OUT / 'capture-subset'
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
            prior=json.loads((OUT/'capture-profile-quarter/STATE.json').read_text())
            assert prior['status']=='completed_pending_review' and not prior['cleanup_errors'] and not prior['returned_KFD_contexts']
            assert json.loads((OUT/'capture-profile-quarter/quarter/SUMMARY.json').read_text())['status']=='quarter_projection_screen_complete'
            holdout=json.loads((OUT/'capture-quarter-holdout/SUMMARY.json').read_text())
            assert holdout['status']=='holdout_operator_screen_complete' and all(not x['screen_flag'] for x in holdout['results'])
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
            # DiskRows creates private hardlinks. A read-only nested mount cannot
            # link into the outer scratch mount (EXDEV). Copy only the 64 selected
            # rows so the original checkpoint stays read-only with no writable alias.
            fixture = root / 'input'
            fixture.mkdir()
            (fixture / 'state.rows').mkdir()
            source = OUT / 'capture-pipeline/input/state.safetensors'
            paths, _ = open_rows(source)
            manifest = json.loads(source.with_suffix('.rows.json').read_text())
            manifest['rows'] = manifest['rows'][:64]
            copies = []
            for i, path in enumerate(paths[:64]):
                assert time.time()-window_started < 600, 'Diagnostic preparation deadline'
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
            name, label = 'gufo-capture-subset', uuid.uuid4().hex
            assert subprocess.run(['docker', 'inspect', name], capture_output=True).returncode != 0
            command = receipt['command'].copy()
            command[command.index('--name')+1] = name
            command[command.index('--label')+1] = 'gufo.quant.run='+label
            assert command[-1] == '/task/convert_main.py'
            command[-1] = '/task/capture_subset_probe.py'
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
                    assert json.loads((root / 'SUMMARY.json').read_text())['status'] == 'subset_mapping_exact'
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
        message = ('[사용자가 승인한25% 통계수집 매핑 실제경로 검사 완료] '
            '/home/m2on/projects/gufo/plan/exl3/F3-CONVERSION-WATCH.md와 '
            '/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104/capture-subset/'
            '{STATE,SUMMARY,PROGRESS}.json 및run.log를 읽어라. '
            'GPU 정리와독립리뷰 후 원본행2309 유지/577행 H선택 방식의정확성을 판단하라. '
            'oracle는모든forward를실행하고H만선택,후보는폐기되는미선택forward만건너뛴다. '
            '첫5normal refs/전체ID/quant_preserves/RNG/H카운터 일치를확인한다. '
            '앞선별도8입력holdout 결과와함께25%적용판단을완료하고,통과시이전동결/idx10보존·최소통합·양자화전행수토큰H카운터검사·unsupported모듈정책·새동결을기록한뒤승인범위에서이어가라. '
            '실패시해당작은검사만고치고완료된baseline과H/packed결과를재사용한다. '
            '전체모델품질동등이나4배가속을주장하지말고,중복GPU/전체baseline재실행/완료checkpoint삭제는하지않는다.')
        try:
            result = subprocess.run(['/home/m2on/.local/bin/codex', 'queue', '--thread', control['thread'],
                                     '--message', message, '--resume'], capture_output=True, text=True, timeout=45)
            save(root / 'FOLLOWUP.json', dict(exit=result.returncode, stdout=result.stdout, stderr=result.stderr))
        except (OSError, subprocess.TimeoutExpired) as error:
            save(root / 'FOLLOWUP.json', dict(error=repr(error)))
