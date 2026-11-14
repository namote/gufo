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

Direction (R11, `--basis`): SPSA also moves the direction inside a small
subspace (direction_family.py; the basis from direction_basis.py):
d(alpha) = normalize(d0 + sum_k alpha_k u_k), with |alpha| capped so the angle
to d0 stays within `--max-angle` degrees. theta is then [strength blocks,
alpha]; alpha has its own perturbation and step sizes (`--c-dir`, `--a-dir`,
on the same schedules) and its own Rademacher stream, so the strength
perturbations are those of a fixed-direction run with the same seed. Since a
direction is a profile, each theta is scored under its own uploaded profile:
`--base-profile` (rank 1, whose row at `--direction-layer` is the basis' d0)
with d(alpha) in that row, sent to `POST /v1/abliteration/profiles/{name}`
before the burst under `--dir-names` (theta+, theta-, and the evaluated theta;
with two names evals reuse the first), each replaced by the next upload and
deleted at the end. The run refuses to start when the loaded profiles and
these names would pass the server's cap of 8. alpha = 0 keeps the base rows
bit for bit, so it scores exactly as the fixed direction. Log records add
`alpha` and `angle` (degrees); `--output` adds them, and `--profile-output`
writes the best theta as a ready profile (direction and strengths).
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
import signal
import sys
import time
import urllib.error
import urllib.request

PROFILE_CAP = 8  # the server holds at most 8 abliteration profiles
DIRECTION_ARGS = ('basis', 'base_profile', 'max_angle', 'c_dir', 'a_dir', 'dir_names', 'init_alpha')


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


def family_module():
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import direction_family
    return direction_family


def http(base_url, method, path, data=None, timeout=120):
    """(status, parsed JSON reply) of one request; HTTP errors are returned,
    not raised."""
    request = urllib.request.Request(base_url.rstrip('/') + path, data,
                                     {'Content-Type': 'application/json'}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            return response.status, json.loads(payload) if payload else {}
    except urllib.error.HTTPError as error:
        with error:
            detail = error.read().decode(errors='replace')
        try:
            return error.code, json.loads(detail)
        except ValueError:
            return error.code, {'error': {'message': detail[:500]}}


class DirectionFamily:
    """Profiles of the direction family under rotating names (see the module
    doc, "Direction")."""

    def __init__(self, args, offset):
        self.df = family_module()
        self.args, self.offset = args, offset
        self.basis = self.df.load_basis(args.basis)
        self.base = json.loads(Path(args.base_profile).read_text())
        self.df.check_base(self.base, self.basis, args.direction_layer, args.direction_kind)
        self.k = len(self.basis['basis'])
        self.names = args.dir_names
        self.uploaded = {}  # name -> the alpha it holds
        self.touched = set()  # names ever sent (a POST may land even when its reply is lost)
        self.events = []    # (method, name), for tests and the log

    def alpha(self, theta):
        return list(theta[self.offset:])

    def angle(self, alpha):
        return round(self.df.angle(self.basis, alpha), 6)

    def check_cap(self):
        """Refuse a run whose names would pass the server's profile cap."""
        status, props = http(self.args.base_url, 'GET', '/props', timeout=30)
        if status != 200:
            raise SystemExit(f'/props answered {status}: cannot check the profile cap')
        loaded = {p.get('profile') for p in ((props.get('abliteration') or {}).get('profiles') or [])}
        total = len(loaded | set(self.names))
        if total > PROFILE_CAP:
            raise SystemExit(f'the server holds {sorted(loaded)}; with {self.names} that is {total} '
                             f'profiles, above its cap of {PROFILE_CAP}: unload some, or pass two '
                             '--dir-names')

    def profile(self, alpha):
        return self.df.profile_with_direction(self.base, self.basis, alpha,
                                              self.args.direction_layer, self.args.direction_kind)

    def upload(self, name, alpha, attempts=60):
        """POST the profile of `alpha` as `name` (a replace when loaded); a
        busy server (409 profile_rows_busy) is retried."""
        if self.uploaded.get(name) == alpha:
            return
        body = self.df.profile_bytes(self.profile(alpha))
        self.touched.add(name)
        self.uploaded.pop(name, None)
        for attempt in range(attempts):
            try:
                status, reply = http(self.args.base_url, 'POST', f'/v1/abliteration/profiles/{name}', body)
            except (OSError, ValueError) as error:  # a dropped connection: the upload is idempotent
                if attempt + 1 == attempts:
                    raise
                print(f'retrying the upload of {name} after: {error!r}', file=sys.stderr, flush=True)
                time.sleep(1.0)
                continue
            if status == 200:
                self.uploaded[name] = list(alpha)
                self.events.append(('POST', name))
                return
            code = (reply.get('error') or {}).get('code')
            if status == 409 and attempt + 1 < attempts:
                time.sleep(min(2.0, 0.25 * (attempt + 1)))
                continue
            hint = (' (every profile version an idle session last ran stays held until that '
                    'session serves again: fewer loaded profiles, two --dir-names or fewer '
                    'sessions leave more free rows)' if status == 409 else '')
            raise SystemExit(f'uploading profile {name} failed: HTTP {status} {code}: '
                             f'{(reply.get("error") or {}).get("message", reply)}{hint}')

    def cleanup(self, attempts=3):
        """Delete every name this run sent (404: already gone), retrying."""
        for name in self.names:
            if name not in self.touched:
                continue
            for attempt in range(attempts):
                try:
                    status, _ = http(self.args.base_url, 'DELETE', f'/v1/abliteration/profiles/{name}',
                                     timeout=30)
                except (OSError, ValueError) as error:
                    status = repr(error)
                if status in (200, 404):
                    self.uploaded.pop(name, None)
                    self.events.append(('DELETE', name))
                    break
                print(f'deleting profile {name}: {status}', file=sys.stderr, flush=True)
                time.sleep(1.0)
            else:
                print(f'profile {name} may still be loaded (it holds one of the server\'s '
                      f'{PROFILE_CAP} slots): DELETE /v1/abliteration/profiles/{name}',
                      file=sys.stderr, flush=True)


class Objective:
    def __init__(self, args, pairs, ordinary, layers, facts=(), guard=()):
        self.args, self.pairs, self.ordinary, self.layers = args, pairs, ordinary, layers
        self.facts = list(facts)
        self.guard = list(guard)
        self.family = None  # a DirectionFamily with --basis
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

    def settings(self, params, name=None):
        """The request settings of `params` (strength blocks; direction
        coefficients after them are ignored here). With a direction family
        `name` is the uploaded profile scored (None: no profile member)."""
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
        if self.family is not None:
            if name:
                settings['profile'] = name
        elif a.profile:
            settings['profile'] = a.profile
        return settings

    def __call__(self, theta):
        names = None
        if self.family is not None:
            names = [self.family.names[2 if len(self.family.names) > 2 else 0]]
        return self.evaluate([theta], names=names)[0]

    def evaluate(self, thetas, pair_idx=None, ord_idx=None, fact_idx=None, guard_idx=None,
                 names=None):
        """Score every theta (layer strengths) on the pairs, ordinary answers,
        facts and guard answers (all, or the given index subsets) in one
        request burst, so the server can pack the requests of all thetas
        together. With a direction family each theta's profile is uploaded
        first under its name in `names`."""
        if self.family is not None:
            if names is None or len(names) != len(thetas) or len(set(names)) != len(names):
                raise ValueError('a direction family scores each theta under its own name')
            for theta, name in zip(thetas, names):
                self.family.upload(name, self.family.alpha(theta))
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
            on = self.settings(theta, names[k] if names else None)
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
    ap.add_argument('--basis', help='a gufo-direction-basis-v1 file (direction_basis.py): also learn '
                                    'the direction in its subspace (see "Direction")')
    ap.add_argument('--base-profile', help='with --basis: the rank-1 profile file whose row at '
                                           '--direction-layer is the basis d0')
    ap.add_argument('--max-angle', type=float, default=20.0,
                    help='with --basis: the largest angle (degrees) between the direction and d0')
    ap.add_argument('--c-dir', type=float, default=0.04,
                    help='with --basis: perturbation size of the direction coefficients')
    ap.add_argument('--a-dir', type=float, default=0.05,
                    help='with --basis: step size scale of the direction coefficients')
    ap.add_argument('--dir-names', default='r11-p,r11-m,r11-e',
                    help='with --basis: profile names for theta+, theta- and evals (two: evals '
                         'reuse the first)')
    ap.add_argument('--init-alpha', help='with --basis: JSON list of starting coefficients '
                                         '(default 0, or a previous --output\'s with the same basis)')
    ap.add_argument('--log', required=True)
    ap.add_argument('--output', help='write the best theta as JSON settings')
    ap.add_argument('--profile-output', help='with --basis: write the best theta as a ready profile')
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
    if args.basis:
        if args.additive:
            ap.error('--basis needs a suppression profile, not --additive')
        if args.profile:
            ap.error('--basis scores its own profiles (--dir-names); leave --profile out')
        if not args.base_profile:
            ap.error('--basis needs --base-profile')
        if args.directions != 1:
            ap.error('--basis learns one direction: --directions 1')
        if not 0 < args.max_angle < 90:
            ap.error('--max-angle must lie in (0, 90)')
        names = [n.strip() for n in args.dir_names.split(',') if n.strip()]
        if not 2 <= len(names) <= 3 or len(set(names)) != len(names) or not all(
                re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', n) for n in names):
            ap.error('--dir-names takes two or three distinct profile names')
        args.dir_names = names
    elif args.base_profile or args.profile_output or args.init_alpha:
        ap.error('--base-profile, --profile-output and --init-alpha need --basis')
    kinds = ('attention', 'mlp', 'residual')
    learn = [k.strip() for k in args.learn.split(',') if k.strip()]
    if not learn or set(learn) - set(kinds):
        ap.error(f'--learn takes a comma-separated subset of {",".join(kinds)}')
    args.learn = ','.join(k for k in kinds if k in learn)  # canonical, for the log header

    n = args.layers
    init = json.loads(Path(args.init).read_text()) if Path(args.init).exists() else json.loads(args.init)
    basis_sha = family_module().file_sha256(args.basis) if args.basis else None
    init_alpha = None
    if isinstance(init, dict) and 'settings' in init:  # a previous --output
        if basis_sha and init.get('basis_sha256') == basis_sha:
            init_alpha = init.get('alpha')
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
    family = None
    alpha = []
    if args.basis:
        family = DirectionFamily(args, len(theta))
        if args.init_alpha:
            text = args.init_alpha
            init_alpha = json.loads(Path(text).read_text() if Path(text).exists() else text)
        alpha = [float(x) for x in (init_alpha or [0.0] * family.k)]
        if len(alpha) != family.k:
            raise SystemExit(f'--init-alpha needs {family.k} values')
        alpha = family.df.cap(alpha, args.max_angle)
        objective.family = family

    log = Path(args.log)
    header = dict(type='header', args={k: v for k, v in vars(args).items() if k not in ('log', 'output', 'steps', 'concurrency', 'base_url', 'profile_output')
                                 and not (k in DIRECTION_ARGS and not args.basis)
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
    if family:
        header['basis_sha256'] = basis_sha
        header['base_profile_sha256'] = family.df.file_sha256(args.base_profile)
    records = []
    if log.exists():
        lines = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        if lines and lines[0] != header:
            raise SystemExit('the log belongs to other settings or data; use a new --log')
        records = lines[1:]
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps(header) + '\n')
    if family:
        family.check_cap()
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # so the profiles are deleted
    try:
        run(args, objective, family, log, records, theta, alpha, facts, guard, pairs, ordinary,
            basis_sha)
    finally:
        if family:
            family.cleanup()
    return 0


def run(args, objective, family, log, records, theta, alpha, facts, guard, pairs, ordinary,
        basis_sha):
    """The SPSA loop (resuming after `records`) and the outputs. With a
    direction family, `alpha` holds the direction coefficients (records
    keep `theta` as the strength blocks and add `alpha` and `angle`)."""
    def with_direction(record, coefficients):
        if family:
            record.update(alpha=list(coefficients),
                          angle=family.angle(coefficients))
        return record

    if not records:
        first = with_direction(dict(step=-1, theta=theta, eval=objective(theta + alpha)), alpha)
        with log.open('a') as f:
            f.write(json.dumps(first) + '\n')
        records = [first]
        print(f'init: J {first["eval"]["j"]:.4f} margin {first["eval"]["margin"]:.4f} '
              f'drop {first["eval"]["drop"]:.4f}', flush=True)
    best = None
    for r in records:
        theta = r['theta']
        if family:
            alpha = r['alpha']
        if r.get('eval') and (best is None or r['eval']['j'] > best['eval']['j']):
            best = r
    start = sum(1 for r in records if r['step'] >= 0)
    rng = random.Random(args.seed)
    dim = len(theta)
    per_kind = dim // 3
    learned = [('attention', 'mlp', 'residual')[i // per_kind] in args.learn.split(',') for i in range(dim)]
    for _ in range(start * dim):  # replay the Rademacher stream for resume
        rng.random()
    if family:
        # Its own stream, so the strength perturbations match a fixed-direction run.
        rng_dir = random.Random(f'{args.seed}:direction')
        for _ in range(start * family.k):
            rng_dir.random()
        cap = lambda a: family.df.cap(a, args.max_angle)
    for k in range(start, args.steps):
        ck = args.c / (k + 1) ** 0.101
        ak = args.a / (k + 1 + args.A) ** 0.602
        delta = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(dim)]
        delta = [d if keep else 0.0 for d, keep in zip(delta, learned)]  # fixed kinds stay put
        plus = clip([t + ck * d for t, d in zip(theta, delta)], args.max_strength, args.min_strength)
        minus = clip([t - ck * d for t, d in zip(theta, delta)], args.max_strength, args.min_strength)
        alpha_plus = alpha_minus = []
        if family:
            cdk = args.c_dir / (k + 1) ** 0.101
            adk = args.a_dir / (k + 1 + args.A) ** 0.602
            delta_dir = [1.0 if rng_dir.random() < 0.5 else -1.0 for _ in range(family.k)]
            alpha_plus = cap([a + cdk * d for a, d in zip(alpha, delta_dir)])
            alpha_minus = cap([a - cdk * d for a, d in zip(alpha, delta_dir)])
        pair_idx = ord_idx = fact_idx = guard_idx = None
        if args.minibatch_chains:
            pick = random.Random(f'{args.seed}:{k}')  # per step, so resume draws the same subset
            chain_names = sorted({r['chain'] for part in (pairs, ordinary, facts, guard) for r in part
                                  if r.get('chain') is not None})
            chosen = set(pick.sample(chain_names, min(args.minibatch_chains, len(chain_names))))
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
        jp, jm = objective.evaluate([plus + alpha_plus, minus + alpha_minus], pair_idx, ord_idx,
                                    fact_idx, guard_idx,
                                    names=family.names[:2] if family else None)
        g = [(jp['j'] - jm['j']) / (2 * ck) * d for d in delta]
        rms = math.sqrt(sum(x * x for x, keep in zip(g, learned) if keep) / sum(learned)) or 1.0
        theta = clip([t + ak * x / rms for t, x in zip(theta, g)], args.max_strength, args.min_strength)
        if family:
            g_dir = [(jp['j'] - jm['j']) / (2 * cdk) * d for d in delta_dir]
            rms_dir = math.sqrt(sum(x * x for x in g_dir) / len(g_dir)) or 1.0
            alpha = cap([a + adk * x / rms_dir for a, x in zip(alpha, g_dir)])
        record = with_direction(dict(step=k, plus=jp, minus=jm, theta=theta), alpha)
        if (k + 1) % args.eval_every == 0 or k + 1 == args.steps:
            record['eval'] = objective(theta + alpha)
            if best is None or record['eval']['j'] > best['eval']['j']:
                best = record
        with log.open('a') as f:
            f.write(json.dumps(record) + '\n')
        e = record.get('eval')
        print(f'step {k + 1}/{args.steps}: J+ {jp["j"]:.4f} J- {jm["j"]:.4f}'
              + (f' angle {record["angle"]:.2f}' if family else '')
              + (f' | eval J {e["j"]:.4f} margin {e["margin"]:.4f} drop {e["drop"]:.4f}' if e else '')
              + (f' facts {e["fact_right"]}/{len(facts)} shortfall {e["fact_shortfall"]:.3f}' if e and facts else '')
              + (f' guard excess {e["guard_excess"]:.4f}' if e and guard else ''),
              flush=True)
    if best and args.output:
        out = dict(step=best['step'], eval=best['eval'], settings=objective.settings(best['theta']))
        if family:
            out.update(alpha=best['alpha'], angle=best['angle'], basis=str(args.basis),
                       basis_sha256=basis_sha)
        Path(args.output).write_text(json.dumps(out, indent=1) + '\n')
        print(f'best step {best["step"] + 1}: {best["eval"]}; wrote {args.output}')
    if best and family and args.profile_output:
        profile = ready_profile(family.profile(best['alpha']), objective.settings(best['theta']))
        Path(args.profile_output).write_bytes(family.df.profile_bytes(profile) + b'\n')
        print(f'wrote the best profile to {args.profile_output}')


def ready_profile(profile, settings):
    """`profile` with the request settings (per-layer strengths and the
    direction options) as its defaults."""
    out = dict(profile)
    for kind in ('attention', 'mlp', 'residual'):
        out[kind] = [float(x) for x in settings[kind]]
    out.update(direction_layer=settings['direction_layer'], direction_kind=settings['direction_kind'],
               direction_count=settings['directions'], preserve_norm=settings['preserve_norm'])
    return out

if __name__ == '__main__':
    sys.exit(main())
