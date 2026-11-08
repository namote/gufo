"""Bounded same-pack Gufo AR/MTP comparison, reusing saved Kyojin measurements."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

BASE = Path('/workspace/test/experiments/gufo_upstream_port_20261023')
OUT = Path('/receipt/run')
MODEL = Path('/models/kyojin')
ELF = Path('/opt/gufo-native/bin/.gufo-wrapped')
EXPECTED_ELF = 'b4ebe27cb5cbd3173143ddf129b9188db6f13178682921fb3c8dccbaadbc9455'
ROCR = '/nix/store/88jjsz00km321bcl9851aiwwdig79b4l-rocm-runtime-10.1/lib'
CLR = '/nix/store/al4qwi7jfyrdkyb25xsri8b00b1agj1r-clr-10.1/'
URL = 'http://127.0.0.1:18732'


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=120)


def main():
    assert hashlib.sha256(ELF.read_bytes()).hexdigest() == EXPECTED_ELF
    model = json.loads((BASE / 'kyojin_runtime/MODEL.json').read_text())
    for row in model['files']:
        stat = (MODEL / row['path']).stat()
        assert stat.st_size == row['size'] and stat.st_mtime_ns == row['mtime_ns'], row['path']
    prompts = json.loads((BASE / 'KYOJIN-LOCAL-PROMPTS.json').read_text())
    saved = json.loads((BASE / 'KYOJIN-LOCAL-RESULT.json').read_text())
    expected_counts = {r['case']: r['prompt_tokens'] for r in saved['decode']}
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GUFO_', 'LD_'))}
    env['LD_LIBRARY_PATH'] = ROCR
    report = {'status': 'STARTED', 'binary_sha256': EXPECTED_ELF, 'model': model,
              'runs': [], 'servers': [], 'saved_kyojin_repeated': False,
              'limitations': ['Same pack does not mean same effective numerical model: Gufo does not consume Kyojin HC group-int8 sidecar or draft2s head.',
                              'Gufo KVarN k4v4 differs from Kyojin quant K4V4; Nix10.1 release differs from pinned nightly wheels.',
                              'Gufo HTTP exposes completion text, not raw output token IDs. Equality here is text equality only.',
                              'One measured pass per prompt; no confidence interval or whole quality qualification.']}
    def save():
        (OUT / 'SAME-PACK.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    for mode in ('off', 'mtp'):
        args = [str(ELF), 'serve', 'llm', '--host', '127.0.0.1', '--port', '18732',
                '--model', str(MODEL), '--speculative', mode, '--think', 'off',
                '--context', '16384', '--sessions', '1', '--kv-cache-type', 'kvarn-k4v4',
                '--cache-ram-bytes', '0', '--cache-disk-bytes', '0']
        if mode == 'mtp':
            args += ['--mtp-model', str(MODEL)]
        sr = {'mode': mode, 'argv': args}
        report['servers'].append(sr)
        log = (OUT / f'server-{mode}.log').open('w')
        start = time.monotonic()
        process = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                assert process.poll() is None, f'{mode} exited {process.returncode}'
                if time.monotonic() - start > 180:
                    raise TimeoutError('startup')
                try:
                    with urllib.request.urlopen(URL + '/ready', timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
            sr['ready_seconds'] = time.monotonic() - start
            with urllib.request.urlopen(URL + '/v1/models', timeout=3) as response:
                alias = json.load(response)['data'][0]['id']
            maps = Path(f'/proc/{process.pid}/maps').read_text()
            (OUT / f'maps-{mode}.txt').write_text(maps)
            paths = sorted({l.split()[-1] for l in maps.splitlines() if 'libamdhip64' in l or 'libhsa-runtime64' in l})
            assert any(ROCR in p for p in paths) and any(CLR in p for p in paths), paths
            assert all(ROCR in p or CLR in p for p in paths), paths
            assert '.gguf' not in maps
            sr['runtime_paths'] = paths
            save()
            cases = [{'id': 'warm', 'text': 'Say hello briefly.'}] + prompts
            for case in cases:
                measured = case['id'] != 'warm'
                body = {'model': alias, 'messages': [{'role': 'user', 'content': case['text']}],
                        'max_tokens': 256 if measured else 24, 'temperature': 0, 'seed': 314159,
                        'top_k': 1, 'top_p': 1.0, 'min_p': 0.0, 'repeat_penalty': 1.0,
                        'presence_penalty': 0.0, 'frequency_penalty': 0.0,
                        'cache_prompt': False, 'stream': True, 'stream_options': {'include_usage': True},
                        'reasoning_effort': 'none', 'chat_template_kwargs': {'enable_thinking': False}}
                with post('/apply-template', body) as response:
                    rendered = json.load(response)['prompt']
                events, pieces, first, last, final, finish = [], [], None, None, None, None
                start_request = time.perf_counter()
                with post('/v1/chat/completions', body) as response:
                    for raw in response:
                        line = raw.decode().strip()
                        if not line.startswith('data: {'):
                            continue
                        now = time.perf_counter()
                        item = json.loads(line[6:])
                        events.append({'elapsed_ms': (now-start_request)*1000, 'data': item})
                        assert 'error' not in item, item
                        for choice in item.get('choices') or []:
                            delta = choice.get('delta') or {}
                            piece = delta.get('content') or delta.get('reasoning_content')
                            if piece:
                                pieces.append(piece)
                                first = now if first is None else first
                                last = now
                            finish = choice.get('finish_reason') or finish
                        if item.get('usage'):
                            final = item
                name = case['id'] + '-' + mode
                (OUT / (name + '-events.json')).write_text(json.dumps({'request': body, 'rendered_prompt': rendered, 'events': events}) + '\n')
                assert final and finish in ('stop', 'length'), name
                usage = final['usage']; metric = usage['gufo']
                assert usage['cached_tokens'] == 0 and usage['completion_tokens'] > 0, usage
                if mode == 'off':
                    assert usage['draft_tokens'] == 0
                if measured:
                    assert usage['prompt_tokens'] == expected_counts[case['id']], usage
                    assert usage['completion_tokens'] == 256 and finish == 'length', usage
                content = ''.join(pieces)
                row = {'name': name, 'mode': mode, 'measured': measured, 'usage': usage,
                       'timings': final.get('timings'), 'finish_reason': finish, 'text': content,
                       'rendered_prompt': rendered, 'text_sha256': hashlib.sha256(content.encode()).hexdigest(),
                       'client_tps': (usage['completion_tokens']-1)/(last-first) if last and last > first else None,
                       'client_ttft_ms': (first-start_request)*1000 if first else None,
                       'wall_ms': (time.perf_counter()-start_request)*1000}
                report['runs'].append(row); save()
                print(json.dumps({'name': name, 'client_tps': row['client_tps'], 'usage': usage}), flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=10)
            sr['returncode_after_cleanup'] = process.returncode
            log.close(); save()
    report['measured_mtp_drafts'] = sum(r['usage']['draft_tokens'] for r in report['runs'] if r['measured'] and r['mode'] == 'mtp')
    assert report['measured_mtp_drafts'] > 0
    report['pairs'] = []
    for case in prompts:
        a = next(x for x in report['runs'] if x['name'] == case['id'] + '-off')
        b = next(x for x in report['runs'] if x['name'] == case['id'] + '-mtp')
        assert a['rendered_prompt'] == b['rendered_prompt']
        report['pairs'].append({'case': case['id'], 'text_equal': a['text'] == b['text']})
    report['status'] = 'COMPLETE' if all(p['text_equal'] for p in report['pairs']) else 'OUTPUT_MISMATCH'
    save()
    assert report['status'] == 'COMPLETE', report['pairs']


if __name__ == '__main__':
    main()
