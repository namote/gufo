#!/usr/bin/env python3
"""Branch/prefix retention checks against an already owned Flash-Next server.

The caller owns GPU exclusion and server health/lifetime. Keep each request's
response and timings; this is candidate correctness, not a matched speed claim.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import urllib.request
import urllib.error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18740')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--restore', action='store_true', help='Check persisted prefix after server restart')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'requests': []}
    lock = threading.Lock()
    system = 'Preserve the instructions in the final user message.\n' + ''.join(
        f'Cache sharing record {i}: This sentence is background context about a quiet library.\n'
        for i in range(260))

    def save():
        (args.output / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')

    def request(label, code, cold=False, messages=None):
        body = {'model': 'gufo-flash-next', 'temperature': 0, 'seed': 314159,
                'max_tokens': 16, 'cache_prompt': not cold,
                'chat_template_kwargs': {'enable_thinking': False},
                'messages': [{'role': 'system', 'content': system},
                             {'role': 'user', 'content': f'Reply with exactly {code}, and nothing else.'}]}
        if messages is not None:
            body['messages'] = messages
        start = time.monotonic()
        req = urllib.request.Request(args.url + '/v1/chat/completions',
                                     data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode()
            with lock:
                report['requests'].append({'name': label, 'request': body, 'status': error.code, 'error': detail})
                save()
            raise RuntimeError(f'{label}: HTTP {error.code}: {detail}') from error
        item = {'name': label, 'seconds': time.monotonic() - start, 'request': body, 'response': data}
        with lock:
            report['requests'].append(item)
            save()
        assert data['choices'][0]['message']['content'].strip() == code, item
        usage = data['usage']
        assert usage['prompt_tokens'] > 4096, usage
        if cold:
            assert usage['cached_tokens'] == 0, usage
        gufo = usage['gufo']
        assert gufo['component_cache']['kv_layout_version'] == 4, gufo
        print(label, round(item['seconds'], 3), 'cached', usage.get('cached_tokens'),
              'device_payload', gufo['component_cache']['device_payload_bytes'], flush=True)
        return data

    try:
        if args.restore:
            result = request('disk-restart', 'DELTA')
            assert result['usage']['gufo']['cache_disk_hit'], result
            assert result['usage']['cached_tokens'] > 4096, result
        else:
            first = request('parent', 'ALPHA')
            assert first['usage']['gufo']['component_cache']['device_payload_bytes'] > 0, first
            request('branch-learn', 'BETA')
            third = request('branch-restore', 'GAMMA')
            assert third['usage']['cached_tokens'] > 4096, third
            with ThreadPoolExecutor(max_workers=2) as pool:
                branches = list(pool.map(lambda c: request('parallel-' + c, c), ['DELTA', 'EPSILON']))
            for item in branches:
                assert item['usage']['cached_tokens'] > 4096, item
            # Independent uncached controls, once each, validate divergent outputs.
            for code in ['ALPHA', 'GAMMA', 'DELTA', 'EPSILON']:
                request('cold-' + code, code, cold=True)
            if args.image:
                image = base64.b64encode(args.image.read_bytes()).decode()
                messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Name the colors of the left and right halves. Reply exactly RED BLUE if they are red and blue.'},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + image}}]}]
                vision = request('vision-parent', 'RED BLUE', messages=messages)
                messages += [vision['choices'][0]['message'], {'role': 'user', 'content': 'Repeat those colors. Reply exactly RED BLUE.'}]
                cached = request('vision-branch', 'RED BLUE', messages=messages)
                assert cached['usage']['cached_tokens'] > 4096, cached
                request('vision-branch-cold', 'RED BLUE', cold=True, messages=messages)
            retained = request('retained-after-cold', 'DELTA')
            assert retained['usage']['cached_tokens'] > 4096, retained
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
