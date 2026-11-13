#!/usr/bin/env python3
"""Learn per-layer abliteration strengths with SPSA over teacher-forced scores.

The server runs Qwen3.8-Flash-Next with a gufo-abliteration-v2 profile and the
Chat Completions `score_assistant` extension. The direction is fixed (the
request settings `direction_layer`, `direction_kind`, `directions`,
`preserve_norm`; an additive profile, `--additive`, takes none of them); the parameters are the 3 x layers strengths `attention`,
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
`--guard` (optional) is JSONL {"id", "prompt", "answer"} scored like the
ordinary answers, as its own term:
    - nu * mean over guard answers of max(0, (lp_OFF(a) - lp_ON(a)) / n_a - g)
with `--nu` and the tolerance `--guard-margin` g (nats per token); with it,
`--ordinary` may be left out.

Reasoning rows. Any row may also carry "reasoning" (sent as the scored
message's `reasoning_content`, the thinking before the scored text),
"score_reasoning" (true: the scored text continues that reasoning instead of
following it as the answer), "reasoning_effort" and "score_max_tokens" (this
row's cut; it overrides `--answer-tokens`). See docs/SERVER.md, assistant-
message scoring. Rows may carry "chain" (any string, for example the trace
they come from): for each theta, the requests of one chain are sent one after
another in ascending reasoning length, so each can resume from the prompt
checkpoints its predecessor left instead of prefilling the shared prefix again.
`--minibatch-chains` scores all rows of that many random chains per step (and
every row without a chain) instead of `--minibatch`'s per-file subsets.

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
`--learn` names the site kinds SPSA moves (for example `residual`); the others
keep their `--init` strengths and are never perturbed.
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


ROW_KEYS = ('reasoning', 'score_reasoning', 'reasoning_effort', 'score_max_tokens')


def score_body(model, prompt, answer, abliteration, answer_tokens=0, row=None):
    """A scoring request for `answer` after `prompt`; `row`'s reasoning keys
    (ROW_KEYS) frame it inside or after a reasoning trace."""
    row = row or {}
    scored = {'role': 'assistant', 'content': answer}
    if row.get('reasoning'):
        scored['reasoning_content'] = row['reasoning']
    body = {'model': model, 'score_assistant': True, 'max_tokens': 0,
            'messages': [{'role': 'user', 'content': prompt}, scored]}
    answer_tokens = row.get('score_max_tokens') or answer_tokens
    if answer_tokens:
        body['score_max_tokens'] = answer_tokens
    if row.get('score_reasoning'):
        body['score_reasoning'] = True
    if row.get('reasoning_effort'):
        body['reasoning_effort'] = row['reasoning_effort']
    if abliteration is not None:
        body['abliteration'] = abliteration
    return body


def mean_logprob(reply):
    score = reply['score']
    return score['logprob_sum'] / max(score['token_count'], 1), score['token_count']


def row_extras(row):
    """The reasoning keys a row sets (part of its identity in the OFF cache)."""
    return {k: row[k] for k in ROW_KEYS if row.get(k)}


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
    def __init__(self, args, pairs, ordinary, layers, facts=(), guard=()):
        self.args, self.pairs, self.ordinary, self.layers = args, pairs, ordinary, layers
        self.facts = list(facts)
        self.guard = list(guard)
        self.pool = concurrent.futures.ThreadPoolExecutor(max(1, args.concurrency))
        self.off, self.off_guard = self._off_baseline()

    def _map(self, bodies, chains=None):
        """Post every body; bodies sharing a chain key (not None) run in one
        worker, one after another in ascending reasoning length, so each can
        resume from its predecessor's prompt checkpoints. Replies keep the
        order of `bodies`."""
        if chains is None:
            return list(self.pool.map(lambda b: post(self.args.base_url, b), bodies))
        groups, order = {}, []
        for i, chain in enumerate(chains):
            k = ('chain', chain) if chain is not None else ('one', i)
            if k not in groups:
                groups[k] = []
                order.append(k)
            groups[k].append(i)
        reasoning = lambda i: len(bodies[i]['messages'][-1].get('reasoning_content', ''))
        tasks = [sorted(groups[k], key=reasoning) for k in order]  # stable: equal prefixes keep their order
        replies = [None] * len(bodies)

        def run(task):
            for i in task:
                replies[i] = post(self.args.base_url, bodies[i])
        for _ in self.pool.map(run, tasks):
            pass
        return replies

    def _off_baseline(self):
        path = Path(self.args.log + '.off.json')
        rows = [self.ordinary, self.guard]
        extras = any(row_extras(o) for part in rows for o in part)
        ident = lambda o: (o['id'], o['prompt'], o['answer']) + ((row_extras(o),) if extras else ())
        key = hashlib.sha256(json.dumps(
            [self.args.model, [ident(o) for o in self.ordinary]]
            + ([self.args.answer_tokens] if self.args.answer_tokens else [])
            + ([['guard', [ident(o) for o in self.guard]]] if self.guard else [])
        ).encode()).hexdigest()
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get('key') == key:
                return cached['off'], cached.get('guard', {})
        both = self.ordinary + self.guard
        replies = self._map([score_body(self.args.model, o['prompt'], o['answer'], None, self.args.answer_tokens, o)
                             for o in both], [o.get('chain') for o in both])
        off = {o['id']: mean_logprob(r)[0] for o, r in zip(self.ordinary, replies)}
        off_guard = {o['id']: mean_logprob(r)[0] for o, r in zip(self.guard, replies[len(self.ordinary):])}
        cached = dict(key=key, off=off)
        if self.guard:
            cached['guard'] = off_guard
        path.write_text(json.dumps(cached) + '\n')
        return off, off_guard

    def settings(self, params):
        n = self.layers
        theta = expand(params, n, self.args.block_size)
        a = self.args
        settings = dict(enabled=True)
        # An additive profile ("mode": "add") adds each layer's own
        # direction: the server rejects the direction options for it.
        if not a.additive:
            settings.update(direction_layer=a.direction_layer,
                            direction_kind=a.direction_kind,
                            directions=a.directions,
                            preserve_norm=a.preserve_norm == 'on')
        settings.update(attention=[round(x, 5) for x in theta[:n]],
                        mlp=[round(x, 5) for x in theta[n:2 * n]],
                        residual=[round(x, 5) for x in theta[2 * n:]])
        # A server with several profiles needs the name; with one it is the
        # default.
        if a.profile:
            settings['profile'] = a.profile
        return settings

    def __call__(self, theta):
        return self.evaluate([theta])[0]

    def evaluate(self, thetas, pair_idx=None, ord_idx=None, fact_idx=None, guard_idx=None):
        """Score every theta (layer strengths) on the pairs, ordinary answers,
        facts and guard answers (all, or the given index subsets) in one
        request burst, so the server can pack the requests of all thetas
        together."""
        pairs = self.pairs if pair_idx is None else [self.pairs[i] for i in pair_idx]
        ordinary = self.ordinary if ord_idx is None else [self.ordinary[i] for i in ord_idx]
        facts = self.facts if fact_idx is None else [self.facts[i] for i in fact_idx]
        guard = self.guard if guard_idx is None else [self.guard[i] for i in guard_idx]
        t, model = self.args.answer_tokens, self.args.model
        bodies, chains = [], []

        def add(row, answer, on, tokens, k):
            bodies.append(score_body(model, row['prompt'], answer, on, tokens, row))
            chains.append((k, row['chain']) if row.get('chain') is not None else None)
        for k, theta in enumerate(thetas):
            on = self.settings(theta)
            for p in pairs:
                add(p, p['candid'], on, t, k)
                add(p, p['propaganda'], on, t, k)
            for o in ordinary:
                add(o, o['answer'], on, t, k)
            for f in facts:  # whole answers: the changed fact may lie past --answer-tokens
                add(f, f['correct'], on, 0, k)
                add(f, f['wrong'], on, 0, k)
            for g in guard:
                add(g, g['answer'], on, t, k)
        replies = self._map(bodies, chains if any(c is not None for c in chains) else None)
        per = 2 * len(pairs) + len(ordinary) + 2 * len(facts) + len(guard)
        results = []
        for k in range(len(thetas)):
            part = replies[k * per:(k + 1) * per]
            margins = [mean_logprob(part[2 * i])[0] - mean_logprob(part[2 * i + 1])[0]
                       for i in range(len(pairs))]
            nonfact = 2 * len(pairs) + len(ordinary)
            drops = [max(0.0, self.off[o['id']] - mean_logprob(r)[0])
                     for o, r in zip(ordinary, part[2 * len(pairs):nonfact])]
            margin = sum(margins) / len(margins) if margins else 0.0  # a chain minibatch may hold no pair
            drop = sum(drops) / len(drops) if drops else 0.0
            result = dict(j=margin - self.args.lam * drop, margin=margin, drop=drop,
                          drop_max=max(drops, default=0.0), margin_min=min(margins, default=0.0))
            if facts:
                gaps = [part[nonfact + 2 * i]['score']['logprob_sum'] - part[nonfact + 2 * i + 1]['score']['logprob_sum']
                        for i in range(len(facts))]
                shortfall = sum(max(0.0, self.args.fact_margin - g) for g in gaps) / len(gaps)
                result.update(j=result['j'] - self.args.mu * shortfall, fact_shortfall=shortfall,
                              fact_gap_min=min(gaps), fact_right=sum(g > 0 for g in gaps))
            if guard:
                losses = [self.off_guard[g['id']] - mean_logprob(r)[0]
                          for g, r in zip(guard, part[nonfact + 2 * len(facts):])]
                excess = sum(max(0.0, x - self.args.guard_margin) for x in losses) / len(losses)
                result.update(j=result['j'] - self.args.nu * excess, guard_excess=excess,
                              guard_drop_max=max(losses))
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


def clip(theta, high, low=0.0):
    return [min(max(x, low), high) for x in theta]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--base-url', default='http://127.0.0.1:18740')
    ap.add_argument('--model', default='gufo-flash-next')
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--ordinary', help='JSONL {"id", "prompt", "answer"} the setting must not make '
                                       'less likely (required unless --guard is given)')
    ap.add_argument('--facts', help='JSONL {"id", "prompt", "correct", "wrong"} fact-recall guard pairs')
    ap.add_argument('--mu', type=float, default=0.2, help='fact shortfall penalty weight')
    ap.add_argument('--fact-margin', type=float, default=2.0,
                    help='summed log-probability lead (nats) a correct fact answer needs over the wrong one')
    ap.add_argument('--guard', help='JSONL {"id", "prompt", "answer"} answers guarded with their own '
                                    'hinged drop term')
    ap.add_argument('--nu', type=float, default=1.0, help='guard drop penalty weight')
    ap.add_argument('--guard-margin', type=float, default=0.0,
                    help='per-token log-likelihood drop (nats) a guard answer may lose before it costs')
    ap.add_argument('--layers', type=int, default=48)
    ap.add_argument('--direction-layer', type=int,
                    help='the direction layer (suppression profiles; required there)')
    ap.add_argument('--additive', action='store_true',
                    help='the profile is additive ("mode": "add"): send only the '
                         'strengths; use with --min-strength below 0')
    ap.add_argument('--profile', default='',
                    help='loaded profile name (needed when the server loads several)')
    ap.add_argument('--direction-kind', default='projected')
    ap.add_argument('--directions', type=int, default=1)
    ap.add_argument('--preserve-norm', choices=('on', 'off'), default='off')
    ap.add_argument('--init', required=True,
                    help='JSON {"attention": [...], "mlp": [...], "residual": [...]} or a number for all')
    ap.add_argument('--max-strength', type=float, default=4.0)
    ap.add_argument('--min-strength', type=float, default=0.0,
                    help='lowest strength; negative for an additive ("mode": "add") '
                         'profile, whose runtime range is [-64, 64]')
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
    ap.add_argument('--minibatch-chains', type=int, default=0,
                    help='score every row of this many random chains (rows\' "chain") per step, '
                         'plus every row without one; evals use all; not with --minibatch')
    ap.add_argument('--learn', default='attention,mlp,residual',
                    help='comma-separated site kinds to learn; the others stay at --init')
    ap.add_argument('--log', required=True)
    ap.add_argument('--output', help='write the best theta as JSON settings')
    args = ap.parse_args(argv)
    if not 0 <= args.answer_tokens <= 1024:
        ap.error('--answer-tokens must be 0..1024')
    if not args.additive and args.direction_layer is None:
        ap.error('--direction-layer is required for a suppression profile')
    if args.min_strength > args.max_strength:
        ap.error('--min-strength must not exceed --max-strength')
    if args.block_size < 1 or args.minibatch < 0 or args.minibatch_chains < 0:
        ap.error('--block-size must be at least 1 and --minibatch(-chains) at least 0')
    if args.minibatch and args.minibatch_chains:
        ap.error('use --minibatch or --minibatch-chains, not both')
    if not args.ordinary and not args.guard:
        ap.error('--ordinary is required without --guard')
    kinds = ('attention', 'mlp', 'residual')
    learn = [k.strip() for k in args.learn.split(',') if k.strip()]
    if not learn or set(learn) - set(kinds):
        ap.error(f'--learn takes a comma-separated subset of {",".join(kinds)}')
    args.learn = ','.join(k for k in kinds if k in learn)  # canonical, for the log header

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
    theta = clip(compress(theta, n, args.block_size), args.max_strength, args.min_strength)
    pairs = load_jsonl(args.pairs, ('id', 'prompt', 'candid', 'propaganda'))
    ordinary = load_jsonl(args.ordinary, ('id', 'prompt', 'answer')) if args.ordinary else []
    facts = load_jsonl(args.facts, ('id', 'prompt', 'correct', 'wrong')) if args.facts else []
    guard = load_jsonl(args.guard, ('id', 'prompt', 'answer')) if args.guard else []
    objective = Objective(args, pairs, ordinary, n, facts, guard)

    log = Path(args.log)
    header = dict(type='header', args={k: v for k, v in vars(args).items() if k not in ('log', 'output', 'steps', 'concurrency', 'base_url')
                                 and not (k in ('answer_tokens', 'minibatch') and not v)
                                 and not (k == 'block_size' and v == 1)
                                 and not (k == 'min_strength' and v == 0.0)
                                 and not (k == 'additive' and not v)
                                 and not (k == 'direction_layer' and v is None)
                                 and not (k in ('facts', 'mu', 'fact_margin') and not args.facts)
                                 and not (k in ('guard', 'nu', 'guard_margin') and not args.guard)
                                 and not (k == 'minibatch_chains' and not v)
                                 and not (k == 'learn' and v == 'attention,mlp,residual')},
                  pairs=[p['id'] for p in pairs], ordinary=[o['id'] for o in ordinary])
    if facts:
        header['facts'] = [f['id'] for f in facts]
    if guard:
        header['guard'] = [g['id'] for g in guard]
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
    per_kind = dim // 3
    learned = [('attention', 'mlp', 'residual')[i // per_kind] in args.learn.split(',') for i in range(dim)]
    for _ in range(start * dim):  # replay the Rademacher stream for resume
        rng.random()
    for k in range(start, args.steps):
        ck = args.c / (k + 1) ** 0.101
        ak = args.a / (k + 1 + args.A) ** 0.602
        delta = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(dim)]
        delta = [d if keep else 0.0 for d, keep in zip(delta, learned)]  # fixed kinds stay put
        plus = clip([t + ck * d for t, d in zip(theta, delta)], args.max_strength, args.min_strength)
        minus = clip([t - ck * d for t, d in zip(theta, delta)], args.max_strength, args.min_strength)
        pair_idx = ord_idx = fact_idx = guard_idx = None
        if args.minibatch_chains:
            pick = random.Random(f'{args.seed}:{k}')  # per step, so resume draws the same subset
            names = sorted({r['chain'] for part in (pairs, ordinary, facts, guard) for r in part
                            if r.get('chain') is not None})
            chosen = set(pick.sample(names, min(args.minibatch_chains, len(names))))
            within = lambda rows: [i for i, r in enumerate(rows) if r.get('chain') is None or r['chain'] in chosen]
            pair_idx, ord_idx, fact_idx, guard_idx = (within(pairs), within(ordinary),
                                                      within(facts), within(guard))
        elif args.minibatch:
            pick = random.Random(f'{args.seed}:{k}')  # per step, so resume draws the same subset
            pair_idx = sorted(pick.sample(range(len(pairs)), min(args.minibatch, len(pairs))))
            ord_idx = sorted(pick.sample(range(len(ordinary)), min(args.minibatch, len(ordinary))))
            if facts:  # drawn after the others, so runs without facts keep their subsets
                fact_idx = sorted(pick.sample(range(len(facts)), min(args.minibatch, len(facts))))
            if guard:  # drawn last, so runs without a guard keep their subsets
                guard_idx = sorted(pick.sample(range(len(guard)), min(args.minibatch, len(guard))))
        jp, jm = objective.evaluate([plus, minus], pair_idx, ord_idx, fact_idx, guard_idx)
        g = [(jp['j'] - jm['j']) / (2 * ck) * d for d in delta]
        rms = math.sqrt(sum(x * x for x, keep in zip(g, learned) if keep) / sum(learned)) or 1.0
        theta = clip([t + ak * x / rms for t, x in zip(theta, g)], args.max_strength, args.min_strength)
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
              + (f' facts {e["fact_right"]}/{len(facts)} shortfall {e["fact_shortfall"]:.3f}' if e and facts else '')
              + (f' guard excess {e["guard_excess"]:.4f}' if e and guard else ''),
              flush=True)
    if best and args.output:
        Path(args.output).write_text(json.dumps(dict(
            step=best['step'], eval=best['eval'], settings=objective.settings(best['theta'])),
            indent=1) + '\n')
        print(f'best step {best["step"] + 1}: {best["eval"]}; wrote {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
