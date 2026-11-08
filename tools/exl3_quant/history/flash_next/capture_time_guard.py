"""One explicit, bounded extension of module008, without pausing GPU compute.

The old supervisor is paused ONLY while its Docker CLI child is alive. This
guardian keeps its resource checks. ExecStopPost uses --resume-only so a guardian
crash cannot leave a stopped supervisor or unguarded GPU indefinitely.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_stage import OUT, available, save
import shutil

ROOT = OUT / 'capture-speed'
RECEIPT = ROOT / 'TIME-GUARD.json'
PARENT = 1029432
CHILD = 1031926


def proc(pid):
    path = Path(f'/proc/{pid}')
    try:
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], start=fields[19],
                    command=(path / 'cmdline').read_bytes().decode().replace('\0', ' '))
    except FileNotFoundError:
        return None


def same(pid, original):
    now = proc(pid)
    return now is not None and now['start'] == original['start']


def cleanup_orphan(state):
    """Use immutable container ID plus label; never a possibly reused name."""
    identifier = state['container']
    def owned():
        result = subprocess.run(['docker', 'inspect', identifier], capture_output=True,
                                text=True, timeout=10)
        if result.returncode:
            assert 'no such object' in result.stderr.casefold(), result.stderr
            return None
        item = json.loads(result.stdout)[0]
        assert item['Id'] == identifier and item['Config']['Labels'].get('gufo.quant.run') == state['label']
        return item
    item = owned()
    if item is None or not item['State']['Running']:
        return
    try:
        subprocess.run(['docker', 'stop', '--time', '10', identifier],
                       capture_output=True, timeout=25)
    except subprocess.TimeoutExpired:
        pass
    item = owned()
    if item and item['State']['Running']:
        subprocess.run(['docker', 'kill', identifier], capture_output=True, timeout=15)
        for _ in range(20):
            item = owned()
            if item is None or not item['State']['Running']:
                break
            time.sleep(0.2)
        else:
            raise RuntimeError('Owned orphan container did not stop')


def resume(state):
    if same(PARENT, state['parent']):
        if state.get('pause_requested'):
            os.kill(PARENT, signal.SIGCONT)
    else:
        cleanup_orphan(state)


if '--resume-only' in sys.argv:
    if RECEIPT.exists():
        resume(json.loads(RECEIPT.read_text()))
    raise SystemExit(0)

assert not RECEIPT.exists(), 'Single-use extension; retain prior evidence'
original = json.loads((OUT / 'model-job-c03/module-008/STATE.json').read_text())
parent, child = proc(PARENT), proc(CHILD)
assert parent and child and 'model_job.py' in parent['command']
assert 'docker run' in child['command'] and 'gufo-quant-model' in child['command']
label = original['command'][original['command'].index('--label')+1].split('=', 1)[1]


def inspect():
    result = subprocess.run(['docker', 'inspect', 'gufo-quant-model'],
                            capture_output=True, text=True, timeout=10)
    if result.returncode:
        assert 'no such object' in result.stderr.casefold(), result.stderr
        return None
    item = json.loads(result.stdout)[0]
    assert item['Config']['Labels'].get('gufo.quant.run') == label, 'Never touch another GPU owner'
    return item


item = inspect()
assert item and item['State']['Running']
container_id = item['Id']
state = dict(status='armed', parent=parent, child=child, container=container_id,
             label=label, original_started=original['started'], deadline=original['started']+7200,
             reason='First attention capture still running near one-hour guard; preserve active work',
             pause_requested=False, started=time.time())
save(RECEIPT, state)


def halt(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')
signal.signal(signal.SIGTERM, halt)
try:
    # Let the original watchdog run until close to its time boundary.
    while time.time() < original['started']+3480:
        if not same(PARENT, parent) or not same(CHILD, child) or proc(CHILD)['state'] == 'Z':
            state['status'] = 'unneeded'
            break
        time.sleep(1)
    else:
        assert same(PARENT, parent) and same(CHILD, child)
        if proc(CHILD)['state'] == 'Z':
            state['status'] = 'unneeded'
        else:
            item = inspect()
            assert item and item['Id'] == container_id and item['State']['Running']
            assert proc(PARENT)['state'] != 'T', 'Someone else paused the supervisor'
            assert available() >= 8*2**30 and shutil.disk_usage(OUT).free >= 100*2**30
            # Process-crash-visible intent precedes SIGSTOP. Prefer the loop's
            # sleep so resumption rechecks child.poll() before its time guard.
            state.update(status='pausing', pause_requested=True)
            save(RECEIPT, state)
            for _ in range(40):
                if Path(f'/proc/{PARENT}/wchan').read_text().strip() == 'hrtimer_nanosleep':
                    break
                time.sleep(0.05)
            assert Path(f'/proc/{PARENT}/wchan').read_text().strip() == 'hrtimer_nanosleep'
            os.kill(PARENT, signal.SIGSTOP)
            for _ in range(20):
                if proc(PARENT)['state'] == 'T':
                    break
                time.sleep(0.1)
            assert proc(PARENT)['state'] == 'T'
            state.update(status='watching', paused=time.time(), minimum_available_bytes=available())
            save(RECEIPT, state)
            while True:
                assert same(PARENT, parent), 'Supervisor disappeared'
                current = proc(CHILD)
                if current is None or (current['start'] == child['start'] and current['state'] == 'Z'):
                    state['status'] = 'child_finished'
                    break
                assert current['start'] == child['start'], 'Docker child PID reused'
                assert proc(PARENT)['state'] == 'T', 'Supervisor resumed externally'
                mem = available()
                state['minimum_available_bytes'] = min(state['minimum_available_bytes'], mem)
                assert mem >= 8*2**30 and shutil.disk_usage(OUT).free >= 100*2**30
                assert time.time() < state['deadline'], 'Explicit two-hour module limit'
                state['updated'] = time.time(); save(RECEIPT, state)
                time.sleep(2)
except BaseException as error:
    state.update(status='failed', error=repr(error))
    raise
finally:
    try:
        resume(state)
    except BaseException as error:
        state['recovery_error'] = repr(error)
        raise
    finally:
        state['finished'] = time.time()
        save(RECEIPT, state)
