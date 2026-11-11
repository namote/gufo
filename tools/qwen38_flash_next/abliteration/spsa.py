#!/usr/bin/env python3
"""Learn per-layer abliteration strengths with SPSA over teacher-forced scores.

The server runs Qwen3.8-Flash-Next with a gufo-abliteration-v2 profile and the
Chat Completions `score_assistant` extension. The direction is fixed (the
request settings `direction_layer`, `direction_kind`, `directions`,
`preserve_norm`); the parameters are the 3 x layers strengths `attention`,
`mlp` and `residual` (each in [0, max]), sent per request so nothing reloads.

Objective (maximized), from teacher-forced log-probabilities (no generation):

    J = mean over pairs of [lp(candid)/n_candid - lp(propaganda)/n_propaganda]
        - lam * mean over ordinary answers of max(0, (lp_OFF(a) - lp_ON(a)) / n_a)

`--pairs` is JSONL {"id", "prompt", "candid", "propaganda"}; `--ordinary` is
JSONL {"id", "prompt", "answer"} (answers the model wrote with abliteration
OFF, so the penalty is the per-token log-likelihood the setting takes away
from the model's own ordinary behavior). OFF scores of ordinary answers are
computed once and cached beside the log.

SPSA (Spall): each step draws a Rademacher vector D, evaluates
theta +/- c_k D (clipped), estimates g = (J+ - J-) / (2 c_k) * D and moves
theta += a_k * g / rms(g), so the step size is set by a_k, not by the
objective's scale. c_k = c / (k + 1)^0.101, a_k = a / (k + 1 + A)^0.602.
Every `--eval-every` steps theta itself is scored and the best evaluated theta
is kept. Every step is appended to `--log` (JSONL); rerunning resumes.
Standard library only.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import urllib.request


def post(base_url, body, timeout=600):
    request = urllib.request.Request(base_url.rstrip('/') + '/v1/chat/completions',
                                     json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def score_body(model, prompt, answer, abliteration):
    body = {'model': model, 'score_assistant': True, 'max_tokens': 0,
            'messages': [{'role': 'user', 'content': prompt},
                         {'role': 'assistant', 'content': answer}]}
    if abliteration is not None:
        body['abliteration'] = abliteration
    return body


def mean_logprob(reply):
    score = reply['score']
    return score['logprob_sum'] / max(score['token_count'], 1), score['token_count']


def load_jsonl(path, keys):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            missing = [k for k in keys if not row.get(k)]
            if missing:
                raise SystemExit(f'{path}: row {row.get("id")} lacks {missing}')
            rows.append(row)
    if not rows:
        raise SystemExit(f'{path}: no rows')
    return rows


class Objective:
    def __init__(self, args, pairs, ordinary, layers):
        self.args, self.pairs, self.ordinary, self.layers = args, pairs, ordinary, layers
        self.pool = concurrent.futures.ThreadPoolExecutor(max(1, args.concurrency))
        self.off = self._off_baseline()

    def _map(self, bodies):
        return list(self.pool.map(lambda b: post(self.args.base_url, b), bodies))

    def _off_baseline(self):
        path = Path(self.args.log + '.off.json')
        key = hashlib.sha256(json.dumps(
            [self.args.model, [(o['id'], o['prompt'], o['answer']) for o in self.ordinary]]
        ).encode()).hexdigest()
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get('key') == key:
                return cached['off']
        replies = self._map([score_body(self.args.model, o['prompt'], o['answer'], None)
                             for o in self.ordinary])
        off = {o['id']: mean_logprob(r)[0] for o, r in zip(self.ordinary, replies)}
        path.write_text(json.dumps(dict(key=key, off=off)) + '\n')
        return off

    def settings(self, theta):
        n = self.layers
        a = self.args
        return dict(enabled=True, profile='default', direction_layer=a.direction_layer,
                    direction_kind=a.direction_kind, directions=a.directions,
                    preserve_norm=a.preserve_norm == 'on',
                    attention=[round(x, 5) for x in theta[:n]],
                    mlp=[round(x, 5) for x in theta[n:2 * n]],
                    residual=[round(x, 5) for x in theta[2 * n:]])

    def __call__(self, theta):
        on = self.settings(theta)
        bodies = []
        for p in self.pairs:
            bodies.append(score_body(self.args.model, p['prompt'], p['candid'], on))
            bodies.append(score_body(self.args.model, p['prompt'], p['propaganda'], on))
        bodies += [score_body(self.args.model, o['prompt'], o['answer'], on)
                   for o in self.ordinary]
        replies = self._map(bodies)
        margins = []
        for i in range(len(self.pairs)):
            candid = mean_logprob(replies[2 * i])[0]
            propaganda = mean_logprob(replies[2 * i + 1])[0]
            margins.append(candid - propaganda)
        drops = []
        for o, r in zip(self.ordinary, replies[2 * len(self.pairs):]):
            drops.append(max(0.0, self.off[o['id']] - mean_logprob(r)[0]))
        margin = sum(margins) / len(margins)
        drop = sum(drops) / len(drops)
        return dict(j=margin - self.args.lam * drop, margin=margin, drop=drop,
                    drop_max=max(drops), margin_min=min(margins))


_SENTENCE_END = re.compile(r'(?:[.!?。！？]|다\.|요\.)(?=\s|$)|\n')


def shorten(text, limit):
    """Cut text to at most limit characters at the last sentence end (or line
    break) inside the limit, else at the limit; 0 keeps the text."""
    if not limit or len(text) <= limit:
        return text
    head = text[:limit]
    ends = [m.end() for m in _SENTENCE_END.finditer(head)]
    cut = ends[-1] if ends and ends[-1] >= limit // 3 else limit
    return head[:cut].rstrip()


def clip(theta, high):
    return [min(max(x, 0.0), high) for x in theta]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--base-url', default='http://127.0.0.1:18740')
    ap.add_argument('--model', default='gufo-flash-next')
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--ordinary', required=True)
    ap.add_argument('--layers', type=int, default=48)
    ap.add_argument('--direction-layer', type=int, required=True)
    ap.add_argument('--direction-kind', default='projected')
    ap.add_argument('--directions', type=int, default=1)
    ap.add_argument('--preserve-norm', choices=('on', 'off'), default='off')
    ap.add_argument('--init', required=True,
                    help='JSON {"attention": [...], "mlp": [...], "residual": [...]} or a number for all')
    ap.add_argument('--max-strength', type=float, default=4.0)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--a', type=float, default=0.15, help='step size scale (strength units)')
    ap.add_argument('--c', type=float, default=0.15, help='perturbation size (strength units)')
    ap.add_argument('--A', type=float, default=10.0)
    ap.add_argument('--lam', type=float, default=4.0, help='ordinary log-likelihood drop penalty')
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--concurrency', type=int, default=2)
    ap.add_argument('--answer-chars', type=int, default=0,
                    help='score only the first N characters of every answer, cut at a sentence end '
                         '(scoring cost grows with answer length); 0 scores whole answers')
    ap.add_argument('--log', required=True)
    ap.add_argument('--output', help='write the best theta as JSON settings')
    args = ap.parse_args(argv)

    n = args.layers
    init = json.loads(Path(args.init).read_text()) if Path(args.init).exists() else json.loads(args.init)
    if isinstance(init, (int, float)):
        theta = [float(init)] * (3 * n)
    else:
        theta = [float(x) for part in ('attention', 'mlp', 'residual') for x in init[part]]
    if len(theta) != 3 * n:
        raise SystemExit('init needs layers values for attention, mlp and residual')
    theta = clip(theta, args.max_strength)
    pairs = load_jsonl(args.pairs, ('id', 'prompt', 'candid', 'propaganda'))
    ordinary = load_jsonl(args.ordinary, ('id', 'prompt', 'answer'))
    for p in pairs:
        p['candid'] = shorten(p['candid'], args.answer_chars)
        p['propaganda'] = shorten(p['propaganda'], args.answer_chars)
    for o in ordinary:
        o['answer'] = shorten(o['answer'], args.answer_chars)
    objective = Objective(args, pairs, ordinary, n)

    log = Path(args.log)
    header = dict(type='header', args={k: v for k, v in vars(args).items() if k not in ('log', 'output', 'steps', 'concurrency', 'base_url')},
                  pairs=[p['id'] for p in pairs], ordinary=[o['id'] for o in ordinary])
    records = []
    if log.exists():
        lines = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        if lines and lines[0] != header:
            raise SystemExit('the log belongs to other settings or data; use a new --log')
        records = lines[1:]
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps(header) + '\n')
    if not records:
        first = dict(step=-1, theta=theta, eval=objective(theta))
        with log.open('a') as f:
            f.write(json.dumps(first) + '\n')
        records = [first]
        print(f'init: J {first["eval"]["j"]:.4f} margin {first["eval"]["margin"]:.4f} '
              f'drop {first["eval"]["drop"]:.4f}', flush=True)
    best = None
    for r in records:
        theta = r['theta']
        if r.get('eval') and (best is None or r['eval']['j'] > best['eval']['j']):
            best = r
    start = sum(1 for r in records if r['step'] >= 0)
    rng = random.Random(args.seed)
    for _ in range(start * 3 * n):  # replay the Rademacher stream for resume
        rng.random()
    for k in range(start, args.steps):
        ck = args.c / (k + 1) ** 0.101
        ak = args.a / (k + 1 + args.A) ** 0.602
        delta = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(3 * n)]
        plus = clip([t + ck * d for t, d in zip(theta, delta)], args.max_strength)
        minus = clip([t - ck * d for t, d in zip(theta, delta)], args.max_strength)
        jp, jm = objective(plus), objective(minus)
        g = [(jp['j'] - jm['j']) / (2 * ck) * d for d in delta]
        rms = math.sqrt(sum(x * x for x in g) / len(g)) or 1.0
        theta = clip([t + ak * x / rms for t, x in zip(theta, g)], args.max_strength)
        record = dict(step=k, plus=jp, minus=jm, theta=theta)
        if (k + 1) % args.eval_every == 0 or k + 1 == args.steps:
            record['eval'] = objective(theta)
            if best is None or record['eval']['j'] > best['eval']['j']:
                best = record
        with log.open('a') as f:
            f.write(json.dumps(record) + '\n')
        e = record.get('eval')
        print(f'step {k + 1}/{args.steps}: J+ {jp["j"]:.4f} J- {jm["j"]:.4f}'
              + (f' | eval J {e["j"]:.4f} margin {e["margin"]:.4f} drop {e["drop"]:.4f}' if e else ''),
              flush=True)
    if best and args.output:
        Path(args.output).write_text(json.dumps(dict(
            step=best['step'], eval=best['eval'], settings=objective.settings(best['theta'])),
            indent=1) + '\n')
        print(f'best step {best["step"] + 1}: {best["eval"]}; wrote {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
