#!/usr/bin/env python3
"""Persistent Gufo service. Memory guard and explicit stop control the session; GPU diagnostics only log."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ops.window import LOCK, host, npu, quiet
from ops.journal import KernelDelta
from ops.fence import FenceMonitor

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'server-state.json'
IMAGE = 'ai-tools/gufo-eval:native-exl3-vision-rocm101'
IMAGE_ID = 'sha256:e7a761a40187efb13515a0275362ab09d084a129f4544aa4300ec5cccb8c7468'
MODEL = '/home/m2on/projects/models/llm/flash-next-exl3-gufo-c03'
RUNTIME = '/home/m2on/.local/share/gufo/runtime-hc-20261108'
GUFO_SHA256 = '166953f072272e931328f93aafa9e417f2919a331f0933755584c127fcceca1c'
TEMPERATURE = 0.9
VISION = MODEL + ' (embedded EXL3 vision)'
URL = 'http://127.0.0.1:18740'
BIND_HOST = '0.0.0.0'
CONTEXT = 262144
MAX_REQUEST_BYTES = 64 * 2**20
DISK_EMPTY_CACHE_RESERVE_BYTES = 150 * 2**30
DISK_MIN_FREE_BYTES = 100 * 2**30
ENABLE_YARN = False
RAM_CACHE_BYTES = 2 * 2**30
SESSIONS = 2
# Twelve target and one MTP FP16 KV layers plus thirteen pooled-key planes:
# 7,028,736 bytes/block. Two 256K contexts with one guard each use 2,050 blocks.
KV_POOL_BYTES = 14_408_908_800
DISK_CACHE_BYTES = 50 * 2**30
CACHE_ROOT = ROOT / 'cache-c03'

# Keep an idle admission reserve of 80 GiB; active KV/state uses more RAM.
# The container remains capped at 100 GiB with no additional swap allowance.
ACTIVE_RESERVE_BYTES = 4 * 2**30
MEMORY_WARNING_BYTES = 8 * 2**30
MEMORY_CRITICAL_BYTES = ACTIVE_RESERVE_BYTES
MEMORY_LOW_GRACE_SECONDS = 0


def check_memory(available, now, low_since):
    """User-selected 4GiB host floor; warn at 8GiB in the monitor."""
    if available < MEMORY_CRITICAL_BYTES:
        raise RuntimeError('Host memory reserve below 4 GiB')
    return None


def diagnostic_only(collect, **kwargs):
    """Desktop GPU diagnostics are recorded, never a serving stop trigger."""
    try:
        return collect(**kwargs)
    except InterruptedError:
        raise  # Preserve an explicit stop during preflight.
    except Exception as error:
        return {'monitor_error': repr(error)}


def disk_admitted(free_bytes, cache_allocated):
    # 150GiB before cache growth; the configured 50GiB cache consumes part
    # of that reserve. Count only already allocated cache, never prospective
    # space or unrelated files, and retain a hard 100GiB free floor.
    return (free_bytes >= DISK_MIN_FREE_BYTES and
            free_bytes + min(cache_allocated, DISK_CACHE_BYTES) >= DISK_EMPTY_CACHE_RESERVE_BYTES)


def run(argv, timeout=15):
    return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=timeout)


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def inspect_owned(state):
    result = subprocess.run(['docker', 'inspect', state['container']], capture_output=True, text=True)
    if result.returncode:
        return None
    info = json.loads(result.stdout)[0]
    assert info['Config']['Labels'].get('gufo.run') == state['run_id'], 'Container owner differs'
    return info


def build_command(state):
    identifier = state['run_id']
    command = ['docker', 'run', '--rm', '--name', state['container'], '--label', 'gufo.run=' + identifier,
               '--pull', 'never', '--network', 'host', '--read-only', '--user', '1000:1000',
               '--memory', '100g', '--memory-swap', '100g', '--device', '/dev/kfd', '--device', '/dev/dri',
               '--ipc', 'host', '--tmpfs', '/tmp:rw,size=1g,mode=1777', '-e', 'HOME=/tmp',
               '-e', 'XDG_CACHE_HOME=/cache/identities',
               '--mount', f'type=bind,src={CACHE_ROOT},dst=/cache',
               '-e', 'LD_LIBRARY_PATH=/nix/store/88jjsz00km321bcl9851aiwwdig79b4l-rocm-runtime-10.1/lib',
               '--mount', f'type=bind,src={MODEL},dst=/models/fnx,readonly',
               '--mount', f'type=bind,src={RUNTIME},dst=/candidate,readonly',
               '--mount', 'type=volume,src=gufo-nix,dst=/nix,readonly',
               '--entrypoint', '/candidate/gufo']
    for gid in sorted({Path('/dev/kfd').stat().st_gid, *[p.stat().st_gid for p in Path('/dev/dri').glob('renderD*')]}):
        command += ['--group-add', str(gid)]
    command += [IMAGE_ID, 'serve', 'llm', '--host', BIND_HOST, '--port', '18740',
                '--model', '/models/fnx/', '--served-model-name', 'gufo-flash-next',
                '--speculative', 'mtp', '--mtp-model', '/models/fnx/', '--draft-tokens', '7', '--think', 'on',
                '--reasoning-effort', 'medium',
                *(['--enable-yarn'] if ENABLE_YARN else []), '--context', str(CONTEXT), '--sessions', str(SESSIONS),
                '--kv-cache-type', 'fp16',
                '--kv-pool-bytes', str(KV_POOL_BYTES), '--cache-ram-bytes', str(RAM_CACHE_BYTES),
                '--cache-disk', '/cache/continuations', '--cache-disk-bytes', str(DISK_CACHE_BYTES),
                '--temperature', str(TEMPERATURE), '--top-k', '20', '--top-p', '0.95', '--min-p', '0',
                '--repeat-penalty', '1',
                '--presence-penalty', '0', '--frequency-penalty', '0', '--max-tokens', '-1',
                '--request-timeout-ms', '0', '--stall-timeout-ms', '0',
                '--max-output-bytes', '9223372036854775807',
                '--max-request-bytes', str(MAX_REQUEST_BYTES), '--log-progress']
    return command


def supervise():
    identifier = uuid.uuid4().hex[:12]
    out = ROOT / 'logs' / identifier
    out.mkdir()
    state = {'run_id': identifier, 'container': 'gufo-' + identifier,
             'pid': os.getpid(), 'status': 'preflight', 'image': IMAGE, 'image_id': IMAGE_ID,
             'model': MODEL, 'runtime': RUNTIME, 'elf_sha256': GUFO_SHA256, 'temperature': TEMPERATURE,
             'draft_tokens': 7, 'url': URL, 'bind_host': BIND_HOST, 'logs': str(out), 'seconds': None,
             'supervisor_script': str(Path(__file__).resolve()),
             'request_timeout_ms': 0, 'stall_timeout_ms': 0, 'max_tokens': -1,
             'max_request_bytes': MAX_REQUEST_BYTES,
             'max_output_bytes': 9223372036854775807,
             'context_per_session': CONTEXT, 'sessions': SESSIONS, 'active_reserve_bytes': ACTIVE_RESERVE_BYTES,
             'memory_critical_bytes': MEMORY_CRITICAL_BYTES, 'memory_low_grace_seconds': MEMORY_LOW_GRACE_SECONDS,
             'gpu_diagnostics_policy': 'log_only', 'runtime_resource_stop_guard': 'host_memory',
             'kv_pool_bytes': KV_POOL_BYTES, 'kv_cache_type': 'fp16',
             'enable_yarn': ENABLE_YARN, 'yarn_factor': 2 if ENABLE_YARN else 1, 'ram_cache_bytes': RAM_CACHE_BYTES,
             'thinking': 'on', 'reasoning_effort': 'medium', 'vision_model': VISION,
             'disk_cache_bytes': DISK_CACHE_BYTES, 'disk_cache_directory': str(CACHE_ROOT / 'continuations')}
    def persist():
        save(out / 'RUN.json', state)
        save(STATE, state)
    stopped = False
    launched = False
    def stop_signal(*_):
        nonlocal stopped
        stopped = True
        if not launched:
            raise InterruptedError('Stopped during preflight')
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        persist()
        assert run(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}']).stdout.strip() == IMAGE_ID
        assert Path(MODEL, 'config.json').is_file()
        assert hashlib.file_digest(Path(RUNTIME, 'gufo').open('rb'), 'sha256').hexdigest() == GUFO_SHA256
        assert any(k.startswith('model.visual.') for k in json.loads(
            (Path(MODEL) / 'model.safetensors.index.json').read_text())['weight_map']), 'Embedded EXL3 vision is missing'
        assert not npu(), 'NPU is occupied'
        h = host()
        state['pre'] = {'KFD_contexts': h['KFD_contexts'], 'busy': h['telemetry']['gpu_busy_percent'],
                        'available_bytes': h['available_bytes'], 'disk_free': shutil.disk_usage(ROOT).free}
        # Interactive serving shares the GPU with the desktop. Benchmark quiet
        # utilization is not a serving admission criterion; exclude every KFD
        # compute owner while retaining the exclusive lock and health monitors.
        state['pre']['admission_policy'] = 'no KFD compute owners; desktop graphics allowed; 80 GiB available'
        assert not h['KFD_contexts'] and h['available_bytes'] >= 80 * 2**30, state['pre']
        cache_path = CACHE_ROOT / 'continuations'
        cache_allocated = (int(run(['du', '-s', '-B1', '--', str(cache_path)], timeout=30).stdout.split()[0])
                           if cache_path.exists() else 0)
        state['pre']['disk_cache_allocated_bytes'] = cache_allocated
        state['pre']['disk_cache_credit_bytes'] = min(cache_allocated, DISK_CACHE_BYTES)
        assert disk_admitted(state['pre']['disk_free'], cache_allocated), (
            'Disk reserve requires 100 GiB free and 150 GiB including existing cache (credit capped at 50 GiB)')
        with socket.socket() as sock:
            # Match the HTTP listener: allow our recently closed connections
            # in TIME_WAIT, while still rejecting an active listener.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((BIND_HOST, 18740))
            sock.listen(1)
        CACHE_ROOT.mkdir(mode=0o700, exist_ok=True)
        (CACHE_ROOT / 'identities').mkdir(mode=0o700, exist_ok=True)
        journal = KernelDelta(out, run)
        state['kernel_monitor_start'] = diagnostic_only(journal.start)
        fence = FenceMonitor(out, run, '/sys/kernel/debug/dri/0000:c6:00.0/amdgpu_fence_info', '0000:c6:00.0')
        state['fence_pre'] = diagnostic_only(fence.collect)
        command = build_command(state)
        state.update(command=command, status='loading', started_epoch=time.time(), expires_epoch=None)
        persist()
        proc = None
        try:
            with (out / 'server.log').open('w') as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                launched = True
                low_memory_since = None
                check_at = time.monotonic() + 15
                while not stopped:
                    if (out / 'STOP').exists():
                        stopped = True
                        break
                    if proc.poll() is not None:
                        raise RuntimeError(f'Server exited: {proc.returncode}')
                    if state['status'] == 'loading':
                        try:
                            with urllib.request.urlopen(URL + '/ready', timeout=1) as response:
                                ready = response.status == 200
                        except (OSError, TimeoutError):
                            ready = False
                        if ready:
                            state['status'] = 'ready'
                            state['ready_seconds'] = time.time() - state['started_epoch']
                            with urllib.request.urlopen(URL + '/v1/models', timeout=3) as response:
                                state['models'] = json.load(response)
                            info = inspect_owned(state)
                            assert hashlib.file_digest(
                                Path(f"/proc/{info['State']['Pid']}/exe").open('rb'), 'sha256').hexdigest() == GUFO_SHA256
                            maps = Path(f"/proc/{info['State']['Pid']}/maps").read_text()
                            (out / 'runtime-maps.txt').write_text(maps)
                            assert '/88jjsz00km321bcl9851aiwwdig79b4l-' in maps
                            assert '/al4qwi7jfyrdkyb25xsri8b00b1agj1r-' in maps
                            assert '.gguf' not in maps, 'Unexpected GGUF dependency'
                            assert '/models/fnx/' in maps, 'Native EXL3 weights not mapped'
                            modalities = state['models']['data'][0]['architecture']['input_modalities']
                            assert 'image' in modalities, 'Vision input was not advertised'
                            persist()
                    if time.monotonic() >= check_at:
                        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                             if line.startswith('MemAvailable:'))) * 1024
                        state['host_available_bytes'] = available
                        now = time.monotonic()
                        low_memory_since = check_memory(available, now, low_memory_since)
                        state['memory_warning'] = available < MEMORY_WARNING_BYTES
                        state['low_memory_seconds'] = 0 if low_memory_since is None else now - low_memory_since
                        state['kernel'] = diagnostic_only(journal.collect, raise_error=False)
                        state['fence_latest'] = diagnostic_only(fence.collect)
                        persist()
                        check_at = time.monotonic() + 15
                    time.sleep(1)
            state['stop_reason'] = 'requested'
        except BaseException as error:
            state['error'] = repr(error)
            raise
        finally:
            if proc is not None:
                if inspect_owned(state):
                    run(['docker', 'stop', '--time', '10', state['container']], timeout=25)
                proc.wait(timeout=20)
            time.sleep(3)
            state['kernel_final'] = diagnostic_only(journal.collect, raise_error=False)
            state['fence_final'] = diagnostic_only(fence.collect)
            h = host()
            state['post'] = {'KFD_contexts': h['KFD_contexts'], 'busy': h['telemetry']['gpu_busy_percent']}
            state['health_PASS'] = (quiet(h) and not state['kernel_final'].get('new_GPU_error', False)
                                    and 'monitor_error' not in state['kernel_final']
                                    and 'monitor_error' not in state['fence_final'])
            state['status'] = 'failed' if 'error' in state else 'stopped'
            persist()


def read_state():
    return json.loads(STATE.read_text()) if STATE.exists() else None


def supervisor_alive(state):
    if not state:
        return False
    try:
        command = Path(f"/proc/{state['pid']}/cmdline").read_bytes().split(b'\0')
    except FileNotFoundError:
        return False
    script = Path(state.get('supervisor_script', ROOT / 'server.py'))
    return (script in (ROOT / 'server.py', Path(__file__).resolve())
            and str(script).encode() in command and b'_supervise' in command)


def start():
    state = read_state()
    if state and (supervisor_alive(state) or inspect_owned(state)):
        print('Gufo is already running; ./gufo status')
        if state.get('expires_epoch'):
            print('This is the previous timed session. Use ./gufo stop then ./gufo start to switch.')
        return
    with (ROOT / 'logs/supervisor.log').open('a') as log:
        p = subprocess.Popen([sys.executable, '-B', __file__, '_supervise'],
                             stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, stdin=subprocess.DEVNULL)
    # Wait only for launch acknowledgement, never impose a serving deadline.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if p.poll() is not None:
            raise SystemExit('Gufo startup failed; see logs/supervisor.log')
        state = read_state()
        if state and state.get('pid') == p.pid and state['status'] in ('loading', 'ready'):
            print(f"Starting Gufo without a time limit (PID {p.pid}); ./gufo status")
            return
        time.sleep(0.2)
    print(f'Gufo preflight still running (PID {p.pid}); ./gufo status or ./gufo logs')


def stop():
    state = read_state()
    if not state:
        print('Already stopped')
        return
    alive = supervisor_alive(state)
    info = inspect_owned(state)
    if alive:
        os.kill(state['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 60
        while supervisor_alive(state) and time.monotonic() < deadline:
            time.sleep(0.25)
        if supervisor_alive(state):
            raise SystemExit('Stop requested; cleanup still running. Check ./gufo status and ./gufo logs.')
    elif info:
        # Recover only our labeled container if its supervisor died.
        run(['docker', 'stop', '--time', '10', state['container']], timeout=25)
    else:
        print('Already stopped')
        return
    if inspect_owned(state):
        raise SystemExit('Owned container remains; inspect logs before retrying')
    print('Gufo stopped')


def status():
    state = read_state()
    if not state:
        print('Not started')
        return
    info = inspect_owned(state)
    keys = ['status', 'model', 'temperature', 'draft_tokens', 'url', 'bind_host', 'context_per_session', 'sessions',
            'kv_pool_bytes', 'kv_cache_type', 'disk_cache_bytes', 'thinking',
            'reasoning_effort', 'request_timeout_ms', 'stall_timeout_ms',
            'max_tokens', 'max_output_bytes', 'max_request_bytes', 'expires_epoch', 'logs', 'error']
    report = {k: state.get(k) for k in keys}
    report['supervisor_alive'] = supervisor_alive(state)
    report['container_running'] = bool(info and info['State']['Running'])
    if not report['supervisor_alive'] and not report['container_running']:
        report['status'] = 'stopped' if state['status'] == 'stopped' else 'failed'
    elif report['container_running']:
        try:
            with urllib.request.urlopen(URL + '/ready', timeout=2) as response:
                report['ready'] = response.status == 200
        except (OSError, TimeoutError):
            report['ready'] = False
    if state.get('expires_epoch'):
        report['profile'] = 'previous timed session; stop/start to apply new defaults'
    print(json.dumps(report, indent=2))


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else 'start'
    if len(sys.argv) > 2:
        raise SystemExit('Usage: ./gufo [start|stop|status|logs|config]')
    if action == '_supervise':
        try:
            supervise()
        except BaseException as error:
            state = read_state()
            if state and state.get('pid') == os.getpid():
                state.update(status='failed', error=repr(error))
                save(STATE, state)
                save(Path(state['logs']) / 'RUN.json', state)
            raise
    elif action in ('start', 'stop'):
        # Serialize control commands; shared GPU lock remains held by supervisor.
        with (ROOT / 'gufo-control.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            (start if action == 'start' else stop)()
    elif action == 'status':
        status()
    elif action == 'config':
        print(json.dumps(build_command({'run_id': 'preview', 'container': 'gufo-preview'}), indent=2))
    elif action == 'logs':
        state = read_state()
        log = Path(state['logs']) / 'server.log' if state else ROOT / 'logs/supervisor.log'
        if not log.exists():
            log = ROOT / 'logs/supervisor.log'
        if not log.exists():
            raise SystemExit('No logs yet')
        os.execvp('tail', ['tail', '-n', '60', '-F', str(log)])
    else:
        raise SystemExit('Usage: ./gufo [start|stop|status|logs|config]')


if __name__ == '__main__':
    main()
