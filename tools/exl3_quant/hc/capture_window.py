"""One bounded, serialized HC pilot; restore the owned server in finally.

Explicit --execute is required. Other GPU jobs are never stopped. Only this
run's uniquely named containers may be removed. No model files are written.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from tools.docker.pinned_rocm101 import docker_argv
from .weights import read_weight_payload


REPO = Path(__file__).resolve().parents[3]
LIVE = Path.home() / '.local/share/gufo'
LLM = Path.home() / '.local/bin/llm'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def run(argv, **kwargs):
    return subprocess.run(list(map(str, argv)), check=True, capture_output=True,
                          text=True, timeout=kwargs.pop('timeout', 60), **kwargs)


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('missing MemAvailable')


def load_ops():
    name = 'gufo_hc_installed_ops'
    spec = importlib.util.spec_from_file_location(name, LIVE / 'ops/__init__.py',
                                                submodule_search_locations=[str(LIVE / 'ops')])
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return importlib.import_module(name + '.window'), importlib.import_module(name + '.journal')


def same_config(before, after):
    keys = ('model', 'temperature', 'draft_tokens', 'bind_host', 'context_per_session',
            'sessions', 'kv_pool_bytes', 'kv_cache_type', 'disk_cache_bytes', 'thinking',
            'reasoning_effort', 'request_timeout_ms', 'stall_timeout_ms', 'max_tokens',
            'max_output_bytes', 'max_request_bytes', 'expires_epoch')
    return all(before.get(k) == after.get(k) for k in keys)


def capture(collector, inputs, model, output):
    collector, inputs, model, output = map(lambda p: Path(p).resolve(), (collector, inputs, model, output))
    if output.exists() or not output.is_relative_to(REPO / 'build/exl3-hc'):
        raise ValueError('require new output under build/exl3-hc')
    build = json.loads((collector / 'BUILD.json').read_text())
    binary = collector / 'obj/tests/models/qwen38_flash_next/qwen38_flash_next_rq_logits_probe'
    if sha(binary) != build['binary_sha256'] or sha(collector / 'SOURCE.json') != build['source_manifest_sha256']:
        raise ValueError('collector identity changed')
    proof = json.loads((collector / 'CACHE-PROVENANCE.json').read_text())
    if proof['build_receipt_sha256'] != sha(collector / 'BUILD.json'):
        raise ValueError('missing reviewed B10 cache identity')
    manifest = json.loads((inputs / 'INPUTS.json').read_text())
    if sha(inputs / 'windows.json') != manifest['windows_sha256'] or sha(inputs / 'windows.bin') != manifest['tokens_sha256']:
        raise ValueError('input identity changed')
    if len(manifest['rows']) != 16:
        raise ValueError('pilot window count changed')
    installed = json.loads((LIVE / 'INSTALLED.json').read_text())
    for rel, digest in installed['files'].items():
        if sha(LIVE / rel) != digest:
            raise ValueError('installed server/ops identity changed: ' + rel)
    controller_sha = sha(LIVE / 'gufo-server.py')
    before = json.loads(run([LLM, 'status']).stdout)
    if not before.get('ready') or Path(before['model']).resolve() != model:
        raise ValueError('expected owned ready Gufo service with the target pack')
    if shutil.disk_usage(REPO).free < 150 * 2**30:
        raise ValueError('need 150GiB free before the GPU window')
    output.mkdir(parents=True)
    save(output / 'BEFORE.json', before)
    save(output / 'INSTALLED.json', installed)
    command_before = json.loads(run([LLM, 'config']).stdout)
    save(output / 'COMMAND-BEFORE.json', command_before)
    state = {'status': 'preparing', 'cases': {}, 'cleanup_errors': [],
             'collector': build, 'inputs_sha256': sha(inputs / 'INPUTS.json'),
             'model_index_sha256': sha(model / 'model.safetensors.index.json'),
             'service_restored': False, 'gpu_used': False}
    state['target_hc_identities'] = {p: read_weight_payload(
        model, f'model.language_model.layers.0.attn_hyper_connection.input_mix_weight_{p}.weight', 'F16')[1]
        for p in ('down', 'up')}
    save(output / 'STATE.json', state)
    ops, journal = load_ops()
    kernel = journal.KernelDelta(output, run)
    stopped = False; lock = None; owned = None
    def interrupted(*_):
        raise InterruptedError('HC pilot interrupted')
    old_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        kernel.start()
        stopped = True
        run([LLM, 'stop'], timeout=90)
        lock = ops.LOCK.open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if ops.npu():
            raise RuntimeError('NPU owned by another job')
        # A server exit can leave a short drain interval; never reset hardware.
        for _ in range(10):
            h = ops.host()
            save(output / 'GPU-PRE.json', h)
            if ops.quiet(h):
                break
            time.sleep(2)
        else:
            raise RuntimeError('GPU not quiet after owned server stop')
        if available() < 80 * 2**30:
            raise RuntimeError('not enough host memory after server stop')
        cases = [('control', False, 'prefill', 32, 1),
                 ('capture-control', True, 'prefill', 32, 1),
                 ('decode-control', False, 'decode', 8, 1),
                 ('decode-capture-control', True, 'decode', 8, 1),
                 ('prefill', True, 'prefill', 512, 16),
                 ('decode', True, 'decode', 128, 16)]
        for label, enabled, phase, positions, windows in cases:
            directory = output / label; directory.mkdir()
            name = 'gufo-hc-' + uuid.uuid4().hex
            owned = name
            command = ['GUFO_HC_CAPTURE_DIR=' + ('/build/capture' if enabled else ''),
                       '/collector', '/models/fnx', '--windows', '/inputs/windows.json',
                       '--path', phase, '--out', '/build/logits', '--top-k', '8',
                       '--prefix', '32', '--positions-limit', str(positions),
                       '--windows-limit', str(windows), '--batch', '512',
                       '--label', 'hc-pilot-' + label, '--toolchain', 'Nix-ROCm10.1-B10']
            argv = docker_argv(directory, collector / 'source', command, gpu=True,
                               source_readonly=True, raw_entrypoint='/usr/bin/env',
                               mounts=((str(binary), '/collector', True), (str(model), '/models/fnx', True),
                                       (str(inputs), '/inputs', True)))
            argv[2:2] = ['--name', name]
            save(directory / 'ARGV.json', argv)
            state.update(status='running', active=label, gpu_used=True)
            save(output / 'STATE.json', state)
            start = time.monotonic(); minimum = available()
            with (directory / 'run.log').open('x') as log:
                child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        minimum = min(minimum, available())
                        if minimum < 4 * 2**30 or time.monotonic() - start > 360:
                            raise RuntimeError('owned pilot memory/time bound exceeded')
                        time.sleep(1)
                    if child.returncode:
                        raise RuntimeError(f'{label} failed with rc {child.returncode}; saved run.log')
                finally:
                    cleanup = subprocess.run(['docker', 'rm', '-f', name], capture_output=True, text=True, timeout=30)
                    (directory / 'cleanup.log').write_text(cleanup.stdout + cleanup.stderr)
                    if cleanup.returncode and 'No such container' not in cleanup.stderr:
                        raise RuntimeError('owned pilot container cleanup failed')
                    if child.poll() is None:
                        child.wait(timeout=30)
                    owned = None
            post = ops.host(); save(directory / 'GPU-POST.json', post)
            if post['KFD_contexts']:
                raise RuntimeError('GPU compute owner remains after pilot exit')
            meta = json.loads((directory / 'logits.json').read_text())
            state['cases'][label] = {'status': 'completed', 'seconds': time.monotonic() - start,
                                    'min_available_bytes': minimum, 'logits_sha256': sha(directory / 'logits.bin'),
                                    'full_logits_sha256': meta['full_logits_sha256']}
            save(output / 'STATE.json', state)
            kernel.collect()
            if label in ('capture-control', 'decode-capture-control'):
                baseline = 'control' if label == 'capture-control' else 'decode-control'
                if any(state['cases'][baseline][k] != state['cases'][label][k]
                       for k in ('logits_sha256', 'full_logits_sha256')):
                    raise RuntimeError('capture changed the tiny full-vocabulary control logits')
        state['status'] = 'captured_pending_independent_review'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
    finally:
        if owned:
            try:
                run(['docker', 'rm', '-f', owned], timeout=30)
            except Exception as error:
                state['cleanup_errors'].append(repr(error))
        if lock is not None:
            lock.close()
        if kernel.cursor is not None:
            try:
                state['kernel_final'] = kernel.collect(raise_error=False)
                if state['kernel_final']['new_GPU_error']:
                    state.update(status='failed', error='new kernel error; original journal evidence preserved')
            except Exception as error:
                state['cleanup_errors'].append('kernel evidence: ' + repr(error))
        if stopped:
            try:
                if sha(LIVE / 'gufo-server.py') != controller_sha or any(
                        sha(LIVE / rel) != digest for rel, digest in installed['files'].items()):
                    raise RuntimeError('server controller/ops changed; refusing a different restore recipe')
                run([LLM, 'start'], timeout=120)
                for _ in range(60):
                    after = json.loads(run([LLM, 'status']).stdout)
                    if after.get('ready'):
                        break
                    if after.get('error'):
                        raise RuntimeError('restored server failed: ' + str(after['error']))
                    time.sleep(2)
                save(output / 'AFTER.json', after)
                command_after = json.loads(run([LLM, 'config']).stdout)
                save(output / 'COMMAND-AFTER.json', command_after)
                if not after.get('ready') or not same_config(before, after) or command_before != command_after:
                    raise RuntimeError('server restore readiness/config mismatch')
                state['service_restored'] = True
            except Exception as error:
                state['cleanup_errors'].append('restore: ' + repr(error))
        for s, handler in old_handlers.items():
            signal.signal(s, handler)
        save(output / 'STATE.json', state)
    if state['status'] == 'failed' or state['cleanup_errors'] or not state['service_restored']:
        raise RuntimeError('HC pilot needs diagnosis; see ' + str(output / 'STATE.json'))
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    for name in ('collector', 'inputs', 'model', 'output'):
        parser.add_argument('--' + name, required=True)
    args = vars(parser.parse_args()); args.pop('execute')
    print(json.dumps(capture(**args), indent=2))
