"""One resident Kyojin server: bounded PP and paired AR/MTP, saved Gufo controls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

BASE = Path('/workspace/test/experiments/gufo_upstream_port_20261023')
RUNTIME = BASE / 'kyojin_runtime'
SOURCE = RUNTIME / 'source'
SITE = RUNTIME / 'site'
OUT = Path('/receipt/run')
URL = 'http://127.0.0.1:18731'
MODEL = 'kyojin-local'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def post(path, body, timeout=120):
    request = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    return urllib.request.urlopen(request, timeout=timeout)


def main():
    expected = json.loads((RUNTIME / 'INPUTS.json').read_text())
    for name, digest in expected['files'].items():
        assert sha(RUNTIME / name) == digest, name
    model = Path('/models/kyojin')
    for row in expected['model_files']:
        stat = (model / row['path']).stat()
        assert stat.st_size == row['size'] and stat.st_mtime_ns == row['mtime_ns'], row['path']
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('EXL3_', 'QWSPEC_', 'MPW', 'LD_'))}
    env.update(PYTHONPATH=f'{SOURCE}:{SITE}', PYTHONUNBUFFERED='1',
               PATH=f'{SITE}/bin:/opt/venv/bin:/usr/local/bin:/usr/bin:/bin',
               HOME='/receipt/home', XDG_CACHE_HOME='/receipt/home/.cache',
               TRITON_CACHE_DIR='/receipt/triton-cache',
               EXL3_ROCM_SDK=str(SITE / '_rocm_sdk_devel'),
               EXL3_HIPCC=str(SITE / '_rocm_sdk_devel/bin/hipcc'),
               PYTORCH_ROCM_ARCH='gfx1151', MAX_JOBS='4',
               KYOJIN_REQLOG_DIR='/receipt/requests')
    # Pin the SDK runtime to the same wheel set as Torch, not inherited /opt/rocm7.
    env['LD_PRELOAD'] = str(SITE / '_rocm_sdk_core/lib/libhsa-runtime64.so.1')
    argv = ['/opt/venv/bin/python3', '-u', str(SOURCE / 'tools/qwen/serve.py'),
            '--model', str(model), '--model-id', MODEL, '--host', '127.0.0.1',
            '--port', '18731', '-c', '16384', '--sessions', '1', '--draft-policy', 'mix',
            '--kv-mode', 'quant', '--k-bits', '4', '--v-bits', '4', '--no-vision',
            '--no-thinking', '--no-uncensor', '--slot-save-path', '/receipt/slots']
    report = {'status': 'starting', 'argv': argv, 'inputs': expected, 'runs': [],
              'baseline_repeated': False, 'production_changed': False,
              'limitations': ['Kyojin ~4bpw/K4 PLE versus saved Gufo5.05/K6 PLE (or older IQ4 PLE K-suite).',
                              'Kyojin quant K4V4 differs from Gufo KVarN k4v4.',
                              'Source-pinned ROCm10.1 nightly wheels/compiler differ from Gufo Nix10.1 release.',
                              'Model and ngram copied to internal NVMe with streaming source-byte hash checks; load/JIT excluded from request timing.',
                              'Single pass per measured case; not a full quality or production qualification.']}
    def save():
        (OUT / 'KYOJIN.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    started = time.monotonic()
    log = (OUT / 'server.log').open('w')
    process = subprocess.Popen(argv, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        while True:
            assert process.poll() is None, f'server exit {process.returncode}'
            if time.monotonic() - started > 1100:
                raise TimeoutError('startup including model load and JIT')
            try:
                with urllib.request.urlopen(URL + '/v1/models', timeout=2) as response:
                    models = json.load(response)
                if any(x['id'] == MODEL for x in models['data']):
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(1)
        report['ready_seconds'] = time.monotonic() - started
        maps = Path(f'/proc/{process.pid}/maps').read_text()
        (OUT / 'runtime-maps.txt').write_text(maps)
        paths = sorted({line.split()[-1] for line in maps.splitlines()
                        if any(x in line for x in ('libamdhip64', 'libhsa-runtime64', 'exllamav3_ext'))})
        assert paths and any(str(SOURCE) in p and 'exllamav3_ext' in p for p in paths), paths
        assert any(str(SITE) in p and 'libamdhip64' in p for p in paths), paths
        assert any(str(SITE) in p and 'libhsa-runtime64' in p for p in paths), paths
        assert all(p.startswith(str(SITE) + '/') for p in paths
                   if 'libamdhip64' in p or 'libhsa-runtime64' in p), paths
        assert all(p.startswith(str(SOURCE) + '/') for p in paths if 'exllamav3_ext' in p), paths
        report['runtime_paths'] = paths
        save()

        def request(name, messages, count, spec, measured=True):
            body = {'model': MODEL, 'messages': messages, 'max_tokens': count,
                    'temperature': 0, 'seed': 314159, 'enable_thinking': False,
                    'cache_prompt': False, 'speculative': spec, 'stream': True,
                    'stream_options': {'include_usage': True}, 'return_token_ids': True}
            t0 = time.perf_counter()
            events, content, first, last, final = [], '', None, None, None
            with post('/v1/chat/completions', body) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith('data: {'):
                        continue
                    item = json.loads(line[6:])
                    now = time.perf_counter()
                    events.append({'elapsed_ms': (now - t0) * 1000, 'data': item})
                    assert 'error' not in item, item
                    choice = (item.get('choices') or [{}])[0]
                    delta = choice.get('delta') or {}
                    piece = delta.get('content') or delta.get('reasoning_content')
                    if piece:
                        content += piece
                        first = now if first is None else first
                        last = now
                    if 'usage' in item:
                        final = item
            assert final is not None and final.get('token_ids'), name
            usage, timing = final['usage'], final['timings']
            assert usage['completion_tokens'] > 0 and timing['cache_n'] == 0
            assert timing['speculative'] == spec, timing
            assert timing['prompt_per_second'] > 0
            row = {'name': name, 'measured': measured, 'speculative': spec,
                   'usage': usage, 'timings': timing, 'wall_ms': (time.perf_counter() - t0) * 1000,
                   'client_ttft_ms': None if first is None else (first - t0) * 1000,
                   'client_decode_tps': ((usage['completion_tokens'] - 1) / (last - first)
                                         if first is not None and last > first else None),
                   'text_sha256': hashlib.sha256(content.encode()).hexdigest(),
                   'token_ids': final['token_ids'], 'text': content}
            (OUT / (name + '-events.json')).write_text(json.dumps({'request': body, 'events': events}) + '\n')
            report['runs'].append(row)
            save()
            print(json.dumps({k: row[k] for k in ('name', 'usage', 'timings', 'client_decode_tps')}), flush=True)
            return row

        # Different-length real-text PP, exact actual counts retained, no cache reuse.
        corpus = (SOURCE / 'doc/install.md').read_text() + (SOURCE / 'README.md').read_text()
        corpus *= 8
        for target in (2048, 8192):
            low, high = 1, min(len(corpus), target * 6)
            while low < high:
                mid = (low + high + 1) // 2
                messages = [{'role': 'user', 'content': corpus[:mid]}]
                with post('/tokenize', {'model': MODEL, 'messages': messages, 'enable_thinking': False}) as response:
                    n = json.load(response)['count']
                if n <= target:
                    low = mid
                else:
                    high = mid - 1
            messages = [{'role': 'user', 'content': corpus[:low]}]
            request(f'pp{target}-warm', messages, 1, False, False)
            request(f'pp{target}', messages, 1, False)
        prompts = json.loads((BASE / 'KYOJIN-LOCAL-PROMPTS.json').read_text())
        for spec in (False, True):
            request('warm-' + ('mtp' if spec else 'ar'), [{'role': 'user', 'content': 'Say hello briefly.'}], 24, spec, False)
            for item in prompts:
                request(item['id'] + ('-mtp' if spec else '-ar'),
                        [{'role': 'user', 'content': item['text']}], 256, spec)
        pairs = []
        for item in prompts:
            ar = next(r for r in report['runs'] if r['name'] == item['id'] + '-ar')
            mtp = next(r for r in report['runs'] if r['name'] == item['id'] + '-mtp')
            pairs.append({'id': item['id'], 'token_ids_equal': ar['token_ids'] == mtp['token_ids'],
                          'ar_tokens': len(ar['token_ids']), 'mtp_tokens': len(mtp['token_ids'])})
        report['paired_outputs'] = pairs
        draft_total = sum(r['timings'].get('accepted_draft_tokens', 0) +
                          r['timings'].get('rejected_draft_tokens', 0)
                          for r in report['runs'] if r['measured'] and r['speculative'])
        report['measured_mtp_draft_tokens'] = draft_total
        assert draft_total > 0, 'No actual MTP drafts observed'
        report['status'] = 'COMPLETE' if all(p['token_ids_equal'] for p in pairs) else 'OUTPUT_MISMATCH'
        save()
        assert report['status'] == 'COMPLETE', pairs
    except BaseException as error:
        report.update(status='FAILED_no_fallback', error=repr(error))
        save()
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        report['server_returncode_after_cleanup'] = process.returncode
        save()
        log.close()


if __name__ == '__main__':
    main()
