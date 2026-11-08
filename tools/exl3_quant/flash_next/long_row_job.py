"""Run only the bounded long-row pilot under the existing GPU/host guards."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid

from run_stage import OUT, TASK, LOCK, IMAGE, available, save
from test.experiments.gufo_upstream_port_20261023.window import host, npu, quiet

def stop_signal(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')

signal.signal(signal.SIGTERM, stop_signal)

root = OUT / 'long-row-pilot'
root.mkdir(exist_ok=True)
assert not (root / 'SUMMARY.json').exists()
with LOCK.open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    h = host()
    assert quiet(h) and not npu() and available() >= 28*2**30
    assert shutil.disk_usage(OUT).free >= 150*2**30
    name = 'gufo-quant-long-row'
    label = uuid.uuid4().hex
    assert subprocess.run(['docker', 'inspect', name], capture_output=True).returncode != 0
    command = json.loads((OUT/'ngram-full/STATE.json').read_text())['command'].copy()
    assert IMAGE in command and command[-1] == '/task/ngram_full.py'
    command[command.index('--name')+1] = name
    command[command.index('--label')+1] = 'gufo.quant.run=' + label
    i = command.index('EXL3_QT_KERNEL=opt')
    del command[i-1:i+1]
    i = command.index('--workdir')
    command[i:i] = ['-e', 'EXL3_HIP_F32OUT_VIA_F16=0', '-e', 'EXL3_ENABLE_GROUPING=0']
    command[-1] = '/task/long_row_pilot.py'
    state = dict(status='running', started=time.time(), command=command,
                 minimum_available_bytes=available(), pilot_sha256=hashlib.sha256((TASK/'long_row_pilot.py').read_bytes()).hexdigest())
    save(root/'STATE.json', state)
    with (root/'run.log').open('a') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                state['minimum_available_bytes'] = min(state['minimum_available_bytes'], available())
                assert available() >= 8*2**30, 'Host reserve below8GiB'
                assert time.time()-state['started'] < 1200, 'Long-row pilot deadline'
                state['updated'] = time.time()
                save(root/'STATE.json', state)
                time.sleep(2)
            assert child.returncode == 0, child.returncode
            state['status'] = 'passed'
        except BaseException as e:
            state.update(status='failed', error=repr(e))
            raise
        finally:
            inspected = subprocess.run(['docker','inspect',name], capture_output=True, text=True, timeout=15)
            if inspected.returncode == 0:
                item = json.loads(inspected.stdout)[0]
                assert item['Config']['Labels'].get('gufo.quant.run') == label
                try:
                    subprocess.run(['docker','stop','--time','10',item['Id']], capture_output=True, timeout=25)
                except subprocess.TimeoutExpired:
                    subprocess.run(['docker','kill',item['Id']], capture_output=True, timeout=15, check=True)
                check = subprocess.run(['docker','inspect',item['Id']], capture_output=True, text=True, timeout=15)
                if check.returncode == 0 and json.loads(check.stdout)[0]['State']['Running']:
                    subprocess.run(['docker','kill',item['Id']], capture_output=True, timeout=15, check=True)
            child.wait(timeout=30)
            state['returned_KFD_contexts'] = host()['KFD_contexts']
            if state['returned_KFD_contexts']:
                state['status'] = 'cleanup_failed'
            state['finished'] = time.time()
            save(root/'STATE.json',state)
            assert not state['returned_KFD_contexts']
    assert json.loads((root/'SUMMARY.json').read_text())['status'] == 'long_row_screen_pass'
