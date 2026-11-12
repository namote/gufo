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
        - mu * mean over facts of max(0, kappa - (lp(correct) - lp(wrong)))

`--pairs` is JSONL {"id", "prompt", "candid", "propaganda"}; `--ordinary` is
JSONL {"id", "prompt", "answer"} (answers the model wrote with abliteration
OFF, so the penalty is the per-token log-likelihood the setting takes away
from the model's own ordinary behavior). OFF scores of ordinary answers are
computed once and cached beside the log. `--facts` (optional) is JSONL
{"id", "prompt", "correct", "wrong"}: short factual answers the model wrote
correctly and the same answer with the fact changed or hedged. The fact term
uses summed (not per-token) log-probabilities, since the two answers differ
in a few tokens, and is a hinge at `--fact-margin` kappa (nats), so it only
guards facts the setting puts at risk instead of rewarding ones already safe.
Facts are scored whole (keep them short); `--answer-tokens` does not apply.

SPSA (Spall): each step draws a Rademacher vector D, evaluates
theta +/- c_k D (clipped), estimates g = (J+ - J-) / (2 c_k) * D and moves
theta += a_k * g / rms(g), so the step size is set by a_k, not by the
objective's scale. c_k = c / (k + 1)^0.101, a_k = a / (k + 1 + A)^0.602.
Every `--eval-every` steps theta itself is scored and the best evaluated theta
is kept. Every step is appended to `--log` (JSONL); rerunning resumes.

Speed: theta+ and theta- are scored in one request burst, so a packing server
(`--sessions` and `--concurrency` near 2 x 72) fills its packs. `--minibatch`
scores a per-step random subset; `--block-size` shares a strength across
consecutive layers (log thetas are then block parameters; `--output` and the
request settings are always per layer, and `--init` may be per layer).
Standard library only.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.request


def post(base_url, body, timeout=600, attempts=4):
    """POST one request; a dropped connection or an unreadable reply is retried
    (scoring is idempotent), an HTTP error status is not."""
    data = json.dumps(body).encode()
    for attempt in range(attempts):
        request = urllib.request.Request(base_url.rstrip('/') + '/v1/chat/completions', data,
                                         {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError:
            raise
        except (OSError, ValueError) as error:  # reset, broken pipe, truncated or empty body
            if attempt + 1 == attempts:
                raise
            print(f'retrying a scoring request after: {error!r}', file=sys.stderr, flush=True)
            time.sleep(0.5 * (attempt + 1))


def score_body(model, prompt, answer, abliteration, answer_tokens=0):
    body = {'model': model, 'score_assistant': True, 'max_tokens': 0,
            'messages': [{'role': 'user', 'content': prompt},
                         {'role': 'assistant', 'content': answer}]}
    if answer_tokens:
        body['score_max_tokens'] = answer_tokens
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
    def __init__(self, args, pairs, ordinary, layers, facts=()):
        self.args, self.pairs, self.ordinary, self.layers = args, pairs, ordinary, layers
        self.facts = list(facts)
        self.pool = concurrent.futures.ThreadPoolExecutor(max(1, args.concurrency))
        self.off = self._off_baseline()

    def _map(self, bodies):
        return list(self.pool.map(lambda b: post(self.args.base_url, b), bodies))

    def _off_baseline(self):
        path = Path(self.args.log + '.off.json')
        key = hashlib.sha256(json.dumps(
            [self.args.model, [(o['id'], o['prompt'], o['answer']) for o in self.ordinary]]
            + ([self.args.answer_tokens] if self.args.answer_tokens else [])
        ).encode()).hexdigest()
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get('key') == key:
                return cached['off']
        replies = self._map([score_body(self.args.model, o['prompt'], o['answer'], None, self.args.answer_tokens)
                             for o in self.ordinary])
        off = {o['id']: mean_logprob(r)[0] for o, r in zip(self.ordinary, replies)}
        path.write_text(json.dumps(dict(key=key, off=off)) + '\n')
        return off

    def settings(self, params):
        n = self.layers
        theta = expand(params, n, self.args.block_size)
        a = self.args
        return dict(enabled=True, profile='default', direction_layer=a.direction_layer,
                    direction_kind=a.direction_kind, directions=a.directions,
                    preserve_norm=a.preserve_norm == 'on',
                    attention=[round(x, 5) for x in theta[:n]],
                    mlp=[round(x, 5) for x in theta[n:2 * n]],
                    residual=[round(x, 5) for x in theta[2 * n:]])

    def __call__(self, theta):
        return self.evaluate([theta])[0]

    def evaluate(self, thetas, pair_idx=None, ord_idx=None, fact_idx=None):
        """Score every theta (layer strengths) on the pairs, ordinary answers and
        facts (all, or the given index subsets) in one request burst, so the
        server can pack the requests of all thetas together."""
        pairs = self.pairs if pair_idx is None else [self.pairs[i] for i in pair_idx]
        ordinary = self.ordinary if ord_idx is None else [self.ordinary[i] for i in ord_idx]
        facts = self.facts if fact_idx is None else [self.facts[i] for i in fact_idx]
        t = self.args.answer_tokens
        bodies = []
        for theta in thetas:
            on = self.settings(theta)
            for p in pairs:
                bodies.append(score_body(self.args.model, p['prompt'], p['candid'], on, t))
                bodies.append(score_body(self.args.model, p['prompt'], p['propaganda'], on, t))
            bodies += [score_body(self.args.model, o['prompt'], o['answer'], on, t) for o in ordinary]
            for f in facts:  # whole answers: the changed fact may lie past --answer-tokens
                bodies.append(score_body(self.args.model, f['prompt'], f['correct'], on))
                bodies.append(score_body(self.args.model, f['prompt'], f['wrong'], on))
        replies = self._map(bodies)
        per = 2 * len(pairs) + len(ordinary) + 2 * len(facts)
        results = []
        for k in range(len(thetas)):
            part = replies[k * per:(k + 1) * per]
            margins = [mean_logprob(part[2 * i])[0] - mean_logprob(part[2 * i + 1])[0]
                       for i in range(len(pairs))]
            nonfact = 2 * len(pairs) + len(ordinary)
            drops = [max(0.0, self.off[o['id']] - mean_logprob(r)[0])
                     for o, r in zip(ordinary, part[2 * len(pairs):nonfact])]
            margin = sum(margins) / len(margins)
            drop = sum(drops) / len(drops)
            result = dict(j=margin - self.args.lam * drop, margin=margin, drop=drop,
                          drop_max=max(drops), margin_min=min(margins))
            if facts:
                gaps = [part[nonfact + 2 * i]['score']['logprob_sum'] - part[nonfact + 2 * i + 1]['score']['logprob_sum']
                        for i in range(len(facts))]
                shortfall = sum(max(0.0, self.args.fact_margin - g) for g in gaps) / len(gaps)
                result.update(j=result['j'] - self.args.mu * shortfall, fact_shortfall=shortfall,
                              fact_gap_min=min(gaps), fact_right=sum(g > 0 for g in gaps))
            results.append(result)
        return results


def expand(params, layers, block):
    """Per-layer strengths [attention, mlp, residual] from block parameters
    (one value per `block` consecutive layers and site kind)."""
    nb = -(-layers // block)
    return [params[kind * nb + layer // block] for kind in range(3) for layer in range(layers)]


def compress(theta, layers, block):
    """Block parameters as the mean of each block's per-layer strengths."""
    nb = -(-layers // block)
    out = []
    for kind in range(3):
        for j in range(nb):
            values = theta[kind * layers + j * block:kind * layers + min((j + 1) * block, layers)]
            out.append(sum(values) / len(values))
    return out


def clip(theta, high):
    return [min(max(x, 0.0), high) for x in theta]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--base-url', default='http://127.0.0.1:18740')
    ap.add_argument('--model', default='gufo-flash-next')
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--ordinary', required=True)
    ap.add_argument('--facts', help='JSONL {"id", "prompt", "correct", "wrong"} fact-recall guard pairs')
    ap.add_argument('--mu', type=float, default=0.2, help='fact shortfall penalty weight')
    ap.add_argument('--fact-margin', type=float, default=2.0,
                    help='summed log-probability lead (nats) a correct fact answer needs over the wrong one')
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
    ap.add_argument('--answer-tokens', type=int, default=0,
                    help='score only the first N answer tokens (server score_max_tokens, 1..1024); '
                         'scoring cost grows with answer length; 0 scores whole answers')
    ap.add_argument('--block-size', type=int, default=1,
                    help='share one strength per site kind across this many consecutive layers '
                         '(fewer parameters, fewer steps); 1 learns every layer')
    ap.add_argument('--minibatch', type=int, default=0,
                    help='score a random subset of this many pairs and this many ordinary answers '
                         'per step (the same subset for theta+ and theta-); evals use all; 0 uses all')
    ap.add_argument('--log', required=True)
    ap.add_argument('--output', help='write the best theta as JSON settings')
    args = ap.parse_args(argv)
    if not 0 <= args.answer_tokens <= 1024:
        ap.error('--answer-tokens must be 0..1024')
    if args.block_size < 1 or args.minibatch < 0:
        ap.error('--block-size must be at least 1 and --minibatch at least 0')

    n = args.layers
    init = json.loads(Path(args.init).read_text()) if Path(args.init).exists() else json.loads(args.init)
    if isinstance(init, dict) and 'settings' in init:  # a previous --output
        init = init['settings']
    if isinstance(init, (int, float)):
        theta = [float(init)] * (3 * n)
    else:
        theta = [float(x) for part in ('attention', 'mlp', 'residual') for x in init[part]]
    if len(theta) != 3 * n:
        raise SystemExit('init needs layers values for attention, mlp and residual')
    theta = clip(compress(theta, n, args.block_size), args.max_strength)
    pairs = load_jsonl(args.pairs, ('id', 'prompt', 'candid', 'propaganda'))
    ordinary = load_jsonl(args.ordinary, ('id', 'prompt', 'answer'))
    facts = load_jsonl(args.facts, ('id', 'prompt', 'correct', 'wrong')) if args.facts else []
    objective = Objective(args, pairs, ordinary, n, facts)

    log = Path(args.log)
    header = dict(type='header', args={k: v for k, v in vars(args).items() if k not in ('log', 'output', 'steps', 'concurrency', 'base_url')
                                 and not (k in ('answer_tokens', 'minibatch') and not v)
                                 and not (k == 'block_size' and v == 1)
                                 and not (k in ('facts', 'mu', 'fact_margin') and not args.facts)},
                  pairs=[p['id'] for p in pairs], ordinary=[o['id'] for o in ordinary])
    if facts:
        header['facts'] = [f['id'] for f in facts]
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
    dim = len(theta)
    for _ in range(start * dim):  # replay the Rademacher stream for resume
        rng.random()
    for k in range(start, args.steps):
        ck = args.c / (k + 1) ** 0.101
        ak = args.a / (k + 1 + args.A) ** 0.602
        delta = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(dim)]
        plus = clip([t + ck * d for t, d in zip(theta, delta)], args.max_strength)
        minus = clip([t - ck * d for t, d in zip(theta, delta)], args.max_strength)
        pair_idx = ord_idx = fact_idx = None
        if args.minibatch:
            pick = random.Random(f'{args.seed}:{k}')  # per step, so resume draws the same subset
            pair_idx = sorted(pick.sample(range(len(pairs)), min(args.minibatch, len(pairs))))
            ord_idx = sorted(pick.sample(range(len(ordinary)), min(args.minibatch, len(ordinary))))
            if facts:  # drawn after the others, so runs without facts keep their subsets
                fact_idx = sorted(pick.sample(range(len(facts)), min(args.minibatch, len(facts))))
        jp, jm = objective.evaluate([plus, minus], pair_idx, ord_idx, fact_idx)
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
              + (f' | eval J {e["j"]:.4f} margin {e["margin"]:.4f} drop {e["drop"]:.4f}' if e else '')
              + (f' facts {e["fact_right"]}/{len(facts)} shortfall {e["fact_shortfall"]:.3f}' if e and facts else ''),
              flush=True)
    if best and args.output:
        Path(args.output).write_text(json.dumps(dict(
            step=best['step'], eval=best['eval'], settings=objective.settings(best['theta'])),
            indent=1) + '\n')
        print(f'best step {best["step"] + 1}: {best["eval"]}; wrote {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
