#!/usr/bin/env python3
"""Flash-Next serving concurrency matrix (plan/serving/CONCURRENCY-PLAN.md S0).

Runs against the live server on 127.0.0.1:18740 and prints one JSON object per
case. Every prompt carries a fresh salt and cache_prompt=false so no case
reuses another's prefix. Greedy, thinking off.

Usage: concurrency_bench.py OUT.jsonl [--quick]
"""
import json
import random
import sys
import threading
import time
import urllib.request

URL = 'http://127.0.0.1:18740/v1/chat/completions'
TOKENS_PER_LINE = 29.7  # measured: 14,000 lines -> 415,826 prompt tokens


def post(body, timeout=7200):
    request = urllib.request.Request(URL, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.load(response)
    wall = time.time() - started
    usage = reply['usage']
    extra = usage.get('gufo', {})
    return {
        'wall_s': round(wall, 3),
        'prompt_tokens': usage['prompt_tokens'],
        'completion_tokens': usage['completion_tokens'],
        'pp_tps': usage.get('prompt_tokens_per_second'),
        'tg_tps': usage.get('completion_tokens_per_second'),
        'ttft_ms': extra.get('ttft_ms'),
        'prefill_ms': extra.get('prefill_ms'),
        'decode_ms': extra.get('decode_ms'),
        'plan': extra.get('execution_plan'),
        'draft_accepted': usage.get('draft_tokens_accepted'),
        'draft_proposed': usage.get('draft_tokens'),
        'wall_tg_tps': round(usage['completion_tokens'] / max(wall - extra.get('ttft_ms', 0) / 1000, 1e-9), 2),
        'answer': reply['choices'][0]['message']['content'][:80],
    }


def long_prompt(tokens, seed):
    rng = random.Random(seed)
    lines = [f'Entry {i}: item {rng.randint(1, 10**6)} moved to shelf {rng.randint(1, 999)} '
             f'on day {rng.randint(1, 365)}.' for i in range(int(tokens / TOKENS_PER_LINE))]
    lines.insert(len(lines) // 3, f'SECRET NOTE: The code word is KITE-{seed}.')
    return f'[run {seed}]\n' + '\n'.join(lines) + '\n\nWhat is the code word? Answer only the code word.'


def body(content, max_tokens):
    return {'model': 'gufo-flash-next', 'messages': [{'role': 'user', 'content': content}],
            'max_tokens': max_tokens, 'temperature': 0, 'cache_prompt': False,
            'chat_template_kwargs': {'enable_thinking': False}}


def essay(seed):
    topics = ['the history of computing', 'how bridges are designed', 'the water cycle',
              'the invention of printing', 'how vaccines work', 'the life of honeybees']
    return f'[run {seed}] Write a detailed essay of at least 600 words about {topics[seed % len(topics)]}.'


def concurrent(jobs, delays=None):
    results = [None] * len(jobs)
    def run(index):
        if delays:
            time.sleep(delays[index])
        results[index] = post(jobs[index])
    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def main():
    out = open(sys.argv[1], 'a')
    quick = '--quick' in sys.argv
    seed = int(time.time()) % 100000
    def emit(case, results):
        record = {'case': case, 'time': time.strftime('%H:%M:%S'), 'results': results}
        out.write(json.dumps(record, ensure_ascii=False) + '\n')
        out.flush()
        print(json.dumps(record, ensure_ascii=False), flush=True)

    post(body('warm up: say ok', 4))
    depths = [32768] if quick else [2048, 32768, 131072]
    for depth in depths:  # 1: single prefill
        seed += 1
        emit(f'single_pp_{depth}', [post(body(long_prompt(depth, seed), 8))])
    seed += 1
    emit('single_tg_512', [post(body(essay(seed), 512))])  # 1: single decode
    emit('two_decode_512', concurrent([body(essay(seed + 1), 512), body(essay(seed + 2), 512)]))  # 2
    seed += 3
    for depth in ([32768] if quick else [32768, 131072]):  # 3: two prefills
        emit(f'two_pp_{depth}', concurrent([body(long_prompt(depth, seed), 8),
                                             body(long_prompt(depth, seed + 1), 8)]))
        seed += 2
    depth = 32768 if quick else 131072  # 4: prefill + decode
    emit(f'pp_{depth}_plus_decode', concurrent([body(long_prompt(depth, seed), 8), body(essay(seed + 1), 512)],
                                               delays=[0, 15]))


if __name__ == '__main__':
    main()
