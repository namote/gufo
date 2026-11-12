#!/usr/bin/env python3
"""Generate the model's own answers to the R5 coding set and score them.

Run on the GPU pass against a running Gufo server (one request at a time):

    python3 tools/qwen38_flash_next/coding_eval/generate.py \\
        --set build/b1/r5-coding --config base

Each request is greedy, thinking off, 600 tokens, with dynamic abliteration
off ({"enabled": false}); --system adds a system message, --extra-body merges
further request fields (for example a steering profile) and --config names
the run. Answers are appended
to <set>/results/<config>.jsonl one problem per line, so an interrupted run
resumes where it stopped. After generation the code blocks are extracted and
the hidden tests run in the Docker sandbox (no network, read-only); the score
is stored on each line. --no-score skips that step (score later with
evaluate.py).
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sandbox  # noqa: E402

DEFAULT_SERVER = 'http://127.0.0.1:18740'
DEFAULT_MODEL = 'gufo-flash-next'


def request_body(problem, model, max_tokens, abliteration, extra=None, system=None):
    messages = [{'role': 'user', 'content': problem['prompt']}]
    if system:
        messages.insert(0, {'role': 'system', 'content': system})
    body = {
        'model': model,
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': 0,
        'chat_template_kwargs': {'enable_thinking': False},
        'abliteration': abliteration,
        # Evaluation prompts are one-off: keep them out of the disk cache.
        'cache_persist': False,
    }
    for key, value in (extra or {}).items():
        body[key] = value
    return body


def post_chat(server, body, timeout):
    request = urllib.request.Request(
        server.rstrip('/') + '/v1/chat/completions', data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        with error:
            detail = error.read().decode(errors='replace')
        raise RuntimeError(f'HTTP {error.code}: {detail[:500]}') from None


def answer_record(problem, config, body, response, elapsed):
    choice = response['choices'][0]
    message = choice.get('message') or {}
    settings = {k: v for k, v in body.items() if k != 'messages'}
    system = [m['content'] for m in body['messages'] if m['role'] == 'system']
    if system:
        settings['system'] = system[0]
    return dict(id=problem['id'], split=problem['split'], lang=problem['lang'],
                category=problem['category'], config=config,
                text=message.get('content') or '',
                reasoning=message.get('reasoning_content') or '',
                finish_reason=choice.get('finish_reason'),
                usage=response.get('usage'), elapsed_s=round(elapsed, 3),
                settings=settings)


def done_ids(path):
    if not Path(path).exists():
        return set()
    return {row['id'] for row in sandbox.read_jsonl(path, torn_tail=True)}


def generate(problems, out, config, server, model, max_tokens, abliteration, extra,
             timeout, post=post_chat, log=print, system=None):
    sandbox.drop_torn_tail(out)
    finished = done_ids(out)
    todo = [p for p in problems if p['id'] not in finished]
    log(f'{config}: {len(todo)} to generate, {len(finished)} already in {out}')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for number, problem in enumerate(todo, 1):
        body = request_body(problem, model, max_tokens, abliteration, extra, system)
        start = time.monotonic()
        response = post(server, body, timeout)
        record = answer_record(problem, config, body, response, time.monotonic() - start)
        with open(out, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        tokens = (record['usage'] or {}).get('completion_tokens', '?')
        log(f'  [{number}/{len(todo)}] {problem["id"]}: {tokens} tokens, '
            f'{record["finish_reason"]}, {record["elapsed_s"]:.1f}s')


def score_file(path, problems, runner, timeout, workers, log=print):
    """Score every line of a results file in place (always rescored)."""
    rows = sandbox.read_jsonl(path)
    by_id = {p['id']: p for p in problems}
    selected = [r for r in rows if r['id'] in by_id]  # other lines are kept as they are
    scores = sandbox.score_answers(runner, problems,
                                   [(r['id'], r['text']) for r in selected],
                                   timeout=timeout, workers=workers)
    for row, result in zip(selected, scores):
        row['score'] = result
    sandbox.write_jsonl(path, rows)
    passed = sum(r['score']['all_pass'] for r in selected)
    log(f'scored {path}: {passed}/{len(selected)} pass all hidden tests')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', type=Path, required=True)
    parser.add_argument('--config', default='base', help='run name (output file stem)')
    parser.add_argument('--out', type=Path, help='default <set>/results/<config>.jsonl')
    parser.add_argument('--splits', nargs='+', choices=sandbox.SPLITS,
                        default=list(sandbox.SPLITS))
    parser.add_argument('--ids', nargs='*', help='only these problem ids')
    parser.add_argument('--server', default=DEFAULT_SERVER)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--max-tokens', type=int, default=600)
    parser.add_argument('--abliteration', default='{"enabled": false}',
                        help='JSON for the request\'s abliteration field')
    parser.add_argument('--extra-body', default='{}',
                        help='JSON object merged into every request body')
    parser.add_argument('--system', help='system message sent before every problem')
    parser.add_argument('--request-timeout', type=float, default=600)
    parser.add_argument('--no-score', action='store_true')
    parser.add_argument('--image', default=sandbox.DEFAULT_IMAGE)
    parser.add_argument('--test-timeout', type=float, default=2.0)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true',
                        help='print the first request body and exit; no server call')
    args = parser.parse_args()
    problems = sandbox.load_problems(args.set / 'problems.jsonl', args.splits)
    if args.ids:
        problems = [p for p in problems if p['id'] in set(args.ids)]
    out = args.out or args.set / 'results' / f'{args.config}.jsonl'
    abliteration = json.loads(args.abliteration)
    extra = json.loads(args.extra_body)
    if args.dry_run:
        print(json.dumps(request_body(problems[0], args.model, args.max_tokens,
                                      abliteration, extra, args.system),
                         ensure_ascii=False, indent=2))
        return 0
    runner = None
    if not args.no_score:  # check the sandbox before spending GPU time
        runner = sandbox.make_runner('docker', image=args.image)
    generate(problems, out, args.config, args.server, args.model, args.max_tokens,
             abliteration, extra, args.request_timeout, system=args.system)
    if runner is not None:
        score_file(out, problems, runner, args.test_timeout, args.jobs)
    return 0


if __name__ == '__main__':
    sys.exit(main())
