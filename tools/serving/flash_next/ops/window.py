"""GPU owner lock and host/NPU quiet checks for the gufo server."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
LOCK = ROOT / 'OWNER.lock'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(cmd, timeout=15):
    return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout).stdout


def host():
    from .desktop import classify
    raw = json.loads(run(['sudo', '-n', '/usr/bin/python3', '-I', str(ROOT / 'root_read.py')]))
    assert raw['coherent_reader_SHA'] == sha(ROOT / 'root_read.py')
    manifest = json.loads((ROOT / 'DESKTOP-BURST-ADMISSION.json').read_text())
    return classify(raw, manifest)


def npu():
    return json.loads(run(['sudo', '-n', '/usr/bin/python3', '-I', str(ROOT / 'npu_audit.py'), '--audit-npu-only']))


def quiet(h):
    return not h['KFD_contexts'] and h['telemetry']['gpu_busy_percent'] <= 4
