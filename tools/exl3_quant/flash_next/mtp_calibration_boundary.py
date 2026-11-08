"""Reserve the pre-final-mixer boundary without changing/signalling live conversion.

No GPU computation or checkpoint mutation. Start before layer47; wait until its
GPU window is actually running, then join the existing GPU flock queue. The
supervisor's nonblocking next admission exits when this process owns the lock.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from run_stage import OUT, LOCK, save
from disk_boundary import checkpoint as audit_checkpoint
from row_checkpoint import open_rows

R = OUT / 'mtp-calibration-boundary'
CHECKPOINT_IDX = 51


def classify(checkpoint, supervisor, module, log):
    """Pure admission decision; never wait for GPU before final block has begun."""
    idx = checkpoint['next_module_idx']
    if checkpoint.get('bad_rows'):
        raise RuntimeError('Nonfinite main rows require diagnosis')
    if idx > CHECKPOINT_IDX:
        raise RuntimeError('Pre-mixer checkpoint already advanced; no automatic replay')
    if supervisor['status'] != 'converting_main_model':
        raise RuntimeError('Main supervisor not running; inspect before handoff')
    if idx < CHECKPOINT_IDX - 1:
        return 'wait'
    if idx == CHECKPOINT_IDX:
        # A late start races the mixer. Fail visibly instead of pretending a safe
        # boundary was reserved. Normal deployment is armed many hours earlier.
        raise RuntimeError('Reservation armed too late at idx51')
    if module.get('status') == 'running' and (
        ' -- Loading unquantized module: model.language_model.layers.47' in log
    ):
        return 'queue_gpu_lock'
    return 'wait'


def latest():
    supervisor = json.loads((OUT / 'model-job-c03/STATE.json').read_text())
    receipt = OUT / 'model-job-c03' / f"module-{supervisor['module_attempt']:03d}"
    module = json.loads((receipt / 'STATE.json').read_text())
    log = receipt / 'run.log'
    return supervisor, module, log.read_text() if log.exists() else ''


def snapshot(read_checkpoint, read_latest, *, retries=40, pause=time.sleep):
    """Tolerate the upstream checkpoint two-rename gap, never corrupt JSON."""
    for attempt in range(retries):
        try:
            checkpoint = read_checkpoint()
            supervisor, module, log = read_latest()
            if checkpoint == read_checkpoint():
                return checkpoint, supervisor, module, log
        except FileNotFoundError:
            pass
        if attempt + 1 < retries:
            pause(.05)
    raise RuntimeError('Checkpoint snapshot did not stabilize')


def notify(message):
    control = json.loads((OUT / 'WATCH-CONTROL.json').read_text())
    if not control['enabled']:
        return {'status': 'disabled_by_user'}
    result = subprocess.run(
        ['/home/m2on/.local/bin/codex', 'queue', '--thread', control['thread'],
         '--message', message, '--resume'], capture_output=True, text=True, timeout=45)
    return dict(exit=result.returncode, stdout=result.stdout, stderr=result.stderr)


def notify_reserved(message, *, send=notify, persist=None, pause=time.sleep):
    """Notification transport failure must never release the reserved locks."""
    persist = persist or (lambda value: save(R / 'FOLLOWUP.json', value))
    attempts = []
    for attempt in range(3):
        try:
            result = send(message)
        except Exception as error:
            result = {'error': repr(error)}
        attempts.append(result)
        persist({'attempts': attempts, 'delivered': result.get('exit') == 0})
        if result.get('exit') == 0:
            return True
        if attempt < 2:
            pause(5)
    return False  # Retain lease; the hourly check can also deliver the handoff.


def stopped(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


def main():
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGALRM, stopped)
    R.mkdir(exist_ok=True)
    with (R / 'OWNER.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (R / 'BOUNDARY.json').exists(), 'Existing reservation evidence; inspect first'
        assert not (R / 'RELEASE.json').exists(), 'Stale release'
        state = dict(status='waiting_for_layer47', started=time.time(), pid=os.getpid(),
                     expected_checkpoint_idx=CHECKPOINT_IDX,
                     script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        save(R / 'BOUNDARY.json', state)
        try:
            deadline = time.monotonic() + 24 * 3600
            while True:
                assert json.loads((OUT / 'WATCH-CONTROL.json').read_text())['enabled'], 'User stop'
                assert time.monotonic() < deadline, 'Waiting deadline exceeded'
                checkpoint, supervisor, module, log = snapshot(
                    lambda: json.loads((OUT / 'work/ckpt/job.json').read_text()), latest)
                action = classify(checkpoint, supervisor, module, log)
                state.update(updated=time.time(), observed_checkpoint_idx=checkpoint['next_module_idx'])
                save(R / 'BOUNDARY.json', state)
                if action == 'queue_gpu_lock':
                    break
                time.sleep(5)
            with LOCK.open('a') as gpu, (OUT / 'model-job-c03/JOB.lock').open('a') as job:
                state.update(status='waiting_natural_boundary')
                save(R / 'BOUNDARY.json', state)
                signal.alarm(3 * 3600)
                try:
                    fcntl.flock(gpu, fcntl.LOCK_EX)
                finally:
                    signal.alarm(0)
                deadline = time.monotonic() + 180
                while True:
                    try:
                        fcntl.flock(job, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        assert time.monotonic() < deadline, 'Supervisor did not return JOB lock'
                        time.sleep(.2)
                assert json.loads((OUT / 'WATCH-CONTROL.json').read_text())['enabled'], 'User stop'
                checkpoint = json.loads((OUT / 'work/ckpt/job.json').read_text())
                supervisor, module, _ = latest()
                assert checkpoint['next_module_idx'] == CHECKPOINT_IDX, checkpoint
                assert not checkpoint.get('bad_rows'), checkpoint
                assert module['status'] in ('passed', 'passed_after_cpu_checkpoint'), module
                assert module['cleanup_errors'] == [] and module['returned_KFD_contexts'] == [], module
                audited, paths, state_bytes = audit_checkpoint(OUT)
                assert audited == checkpoint and len(paths) == 2309
                _, infos = open_rows(OUT / 'work/ckpt/state.safetensors')
                assert all(dtype == 'F32' and tuple(shape[2:]) == (4, 2560)
                           for dtype, shape in infos), 'Not a full pre-mixer HC stack'
                assert sum(shape[1] for dtype, shape in infos) == 1999496
                protected = {}
                for name in ('job.json', 'state.rows.json', 'original_input_ids.safetensors'):
                    path = OUT / 'work/ckpt' / name
                    stat = path.stat()
                    protected[name] = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
                state.update(status='reserved', acquired=time.time(), checkpoint=checkpoint,
                             protected_stats=protected, state_bytes=state_bytes,
                             rows=len(paths), tokens=1999496)
                save(R / 'BOUNDARY.json', state)
                message = ('[사용자가 승인한 MTP 전체 캘리브레이션 경계 도달] '
                           '/home/m2on/projects/gufo/plan/exl3/F3-CONVERSION-WATCH.md와 '
                           'flash-next-quant-20261104/mtp-calibration-boundary/BOUNDARY.json을 읽어라. '
                           'idx51 pre-mixer 원본 상태에서 GPU/JOB락 예약중이다. '
                           'MTP 혼합비트1546개, 입력FC qmap분리, zero+이전hidden 정렬, '
                           'fc_hidden 4T/H나머지T, 전체2309행1999496토큰 보정 후보를 독립검토하고 '
                           '작은검사 후 별도work에서 MTP를 보정하라. 완료 MTP는 재사용, '
                           '본체idx51 보호/재계산금지. 비전BF16 원본복사와 기존uncalibrated MTP '
                           '자동실행차단·최종팩연결도 검증한 뒤 최소통합/새동결/락반환/본체재개. '
                           '미검증후보 자동적용/중복GPU/실행중코드수정 금지, 최신사용자지시우선.')
                notify_reserved(message)
                deadline = time.monotonic() + 24 * 3600
                while not (R / 'RELEASE.json').exists():
                    assert time.monotonic() < deadline, 'Reservation expired; main remains stopped'
                    assert json.loads((OUT / 'WATCH-CONTROL.json').read_text())['enabled'], 'User stop'
                    time.sleep(2)
                state.update(status='released', finished=time.time())
                save(R / 'BOUNDARY.json', state)
        except BaseException as error:
            state.update(status='failed', error=repr(error), finished=time.time())
            save(R / 'BOUNDARY.json', state)
            try:
                save(R / 'FAILURE-FOLLOWUP.json', notify(
                    '[MTP 캘리브레이션 경계 예약 확인 필요] '
                    + str(R / 'BOUNDARY.json')
                    + ' 예약 실패 기록을 확인하라. 현재 변환 정상 여부와 idx51 상태 보존을 우선 확인하고 '
                    '본체 재계산/중복GPU/실행중코드수정은 하지 않는다.'))
            except Exception as notify_error:
                save(R / 'NOTIFY-ERROR.json', {'error': repr(notify_error)})
            raise


if __name__ == '__main__':
    main()
