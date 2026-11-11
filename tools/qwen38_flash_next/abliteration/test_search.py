"""CPU contracts of search.py: KL approximation, Pareto front, refusal
heuristic, request shapes and a resumable search against an injected client
and a local HTTP server. Standard library only."""
import base64
import hashlib
import http.server
import importlib.util
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import threading
import unittest

spec = importlib.util.spec_from_file_location('search', Path(__file__).with_name('search.py'))
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)

LAYERS = 4


class FakeClient:
    """A model whose refusals fall and whose first-token distribution drifts
    with the total strength it is asked for."""

    IDENTITY = 'f' * 64

    def __init__(self):
        self.requests = []

    def props(self):
        return {'abliteration': {'profile': 'default', 'schema': 'gufo-abliteration-v2',
                                 'identity': self.IDENTITY, 'layers': LAYERS,
                                 'direction_layers': list(range(LAYERS)),
                                 'direction_layer': 2, 'attention': 1.0,
                                 'mlp': [0, 0.5, 0, 0], 'max_strength': 4}}

    @staticmethod
    def strength(body):
        a = body['abliteration']
        if not a['enabled']:
            return 0.0

        def total(name, default):
            value = a.get(name, default)
            return sum(value) if isinstance(value, list) else value * LAYERS

        # A residual strength removes the direction more completely: weight 2.
        return (total('attention', [1.0] * LAYERS) + total('mlp', [0, 0.5, 0, 0]) +
                2 * total('residual', 0.0))

    def chat(self, body):
        self.requests.append(body)
        s = self.strength(body)
        if body.get('logprobs'):
            shift = 0.1 * s
            top = [dict(token=t, bytes=list(t.encode()), logprob=lp)
                   for t, lp in (('A', math.log(0.6) - shift), ('B', math.log(0.3)),
                                 ('C', math.log(0.05) + shift / 2))]
            return {'choices': [{'logprobs': {'content': [
                dict(token='A', logprob=top[0]['logprob'], top_logprobs=top)]},
                'message': {'content': 'A'}, 'finish_reason': 'length'}]}
        text = "I'm sorry, I can't help." if s < 6 else 'Sure. Step one...'
        return {'choices': [{'message': {'content': text, 'reasoning_content': ''},
                             'finish_reason': 'stop'}]}


def data_file(root):
    rows = []
    for group in ('refusal', 'ordinary'):
        for i in range(5):
            for thinking in (False, True):
                rows.append(dict(id=f'{group}{i}-{"think" if thinking else "plain"}',
                                 text=f'{group} prompt {i}', group=group, split='tuning',
                                 thinking=thinking, source='fixture'))
    rows.append(dict(id='held', text='x', group='refusal', split='evaluation',
                     thinking=False, source='fixture'))
    path = root / 'tuning.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


class OfflineLogicTest(unittest.TestCase):
    def test_kl_identical_is_zero_and_shift_positive(self):
        off = {'a': math.log(0.7), 'b': math.log(0.2)}
        self.assertAlmostEqual(search.first_token_kl(off, dict(off)), 0.0, places=12)
        on = {'a': math.log(0.2), 'b': math.log(0.7)}
        self.assertGreater(search.first_token_kl(off, on), 0.5)

    def test_kl_union_imputation(self):
        # Exact when both sides report their full support.
        p = {'a': math.log(0.5), 'b': math.log(0.5)}
        q = {'a': math.log(0.25), 'b': math.log(0.75)}
        exact = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
        self.assertAlmostEqual(search.first_token_kl(p, q), exact, places=9)
        # A token missing on one side shares that side's leftover mass,
        # capped at its smallest reported probability; then each side is
        # renormalized over the union.
        off = {'a': math.log(0.6), 'b': math.log(0.3)}   # leftover 0.1 -> c
        on = {'a': math.log(0.5), 'c': math.log(0.05)}   # leftover 0.45, cap 0.05 -> b
        pa, pb, pc = 0.6, 0.3, 0.1
        qa, qb, qc = 0.5, 0.05, 0.05
        ps, qs = pa + pb + pc, qa + qb + qc
        expected = sum(x / ps * math.log((x / ps) / (y / qs))
                       for x, y in ((pa, qa), (pb, qb), (pc, qc)))
        self.assertAlmostEqual(search.first_token_kl(off, on), expected, places=9)
        # Flat distributions never blow up through a clamped tail.
        flat_off = {str(i): math.log(0.04) for i in range(20)}
        flat_on = {str(i + 10): math.log(0.04) for i in range(20)}
        self.assertLess(search.first_token_kl(flat_off, flat_on), 1.0)
        with self.assertRaises(ValueError):
            search.first_token_kl({}, on)

    def test_pareto_and_choice(self):
        trials = [dict(trial=i, refusals=r, kl=k) for i, (r, k) in enumerate(
            [(10, 0.0), (5, 0.2), (5, 0.1), (2, 0.5), (8, 0.3), (0, 2.0)])]
        front = search.pareto_front(trials)
        self.assertEqual([t['trial'] for t in front], [5, 3, 2, 0])
        self.assertEqual(search.choose(trials, 0.15)['trial'], 2)
        self.assertEqual(search.choose(trials, 0.6)['trial'], 3)
        self.assertIsNone(search.choose(trials, -1))

    def test_refusal_heuristic(self):
        self.assertTrue(search.refused("I'm sorry, but I can't help with that."))
        # An empty answer is no refusal phrase; evaluate() counts it as one.
        self.assertFalse(search.refused('<think>no</think>  '))

        class Empty(FakeClient):
            def chat(self, body):
                if body.get('logprobs'):
                    return super().chat(body)
                return {'choices': [{'message': {'content': ''},
                                     'finish_reason': 'stop'}]}

        prompts = {'refusal': [dict(id='r', text='x', thinking=False)],
                   'ordinary': [dict(id='o', text='y', thinking=False)]}
        with tempfile.TemporaryDirectory() as tmp:
            run = search.Search(Empty(), 'gufo', prompts, Empty().props()['abliteration'],
                                Path(tmp) / 'log.jsonl', 8, out=io.StringIO())
            result = run.evaluate({}, run.baseline())
        self.assertEqual((result['refusals'], result['unfinished']), (1, 1))
        self.assertTrue(search.refused('I’m unable to do that.'))
        self.assertTrue(search.refused('<think>ok</think>\n죄송하지만 도와드릴 수 없습니다.'))
        self.assertFalse(search.refused('<think>I cannot decide</think>Here is how.'))
        self.assertFalse(search.refused('Sure! ' + 'x' * 400 + " I can't"))

    def test_failure_phrases(self):
        phrases = ['불가분의 일부', 'inalienable part of china']
        self.assertTrue(search.refused('대만은 중국의 불가분의 일부입니다.', phrases))
        self.assertTrue(search.refused('Hong Kong is an Inalienable part of China.', phrases))
        self.assertFalse(search.refused("I'm sorry, I can't.", phrases))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, 'p.json')
            path.write_text(json.dumps(phrases))
            self.assertEqual(search.load_phrases(path), sorted(phrases))
            path.write_text(json.dumps(['Upper']))
            with self.assertRaises(SystemExit):
                search.load_phrases(path)

    def test_settings_follow_the_curve(self):
        params = {'direction_layer': 1, 'attention.max_w': 2.0, 'attention.peak': 1.0,
                  'attention.min_frac': 0.5, 'attention.width': 2.0,
                  'mlp.max_w': 0.0, 'mlp.peak': 0.0, 'mlp.min_frac': 0.0,
                  'mlp.width': 1.0}
        values = search.settings(params, LAYERS, 'writer')
        self.assertEqual(values['direction_layer'], 1)
        self.assertEqual(values['attention'], [1.5, 2.0, 1.5, 1.0])
        self.assertEqual(values['mlp'], [0.0] * LAYERS)
        # Writer mode turns residual strengths off explicitly.
        self.assertEqual(values['residual'], 0.0)

    def test_residual_mode_settings_and_bounds(self):
        box = search.bounds(LAYERS, [1, 2], 4.0, 'residual')
        self.assertEqual(set(box), {'direction_layer', 'residual.max_w',
                                    'residual.peak', 'residual.min_frac',
                                    'residual.width'})
        params = {'direction_layer': 2, 'residual.max_w': 1.0,
                  'residual.peak': 3.0, 'residual.min_frac': 0.5,
                  'residual.width': 2.0}
        values = search.settings(params, LAYERS, 'residual')
        self.assertEqual(values, {'direction_layer': 2, 'attention': 0.0,
                                  'mlp': 0.0, 'residual': [0.0, 0.5, 0.75, 1.0]})
        both = search.bounds(LAYERS, [0, 3], 4.0, 'both')
        self.assertEqual(len(both), 1 + 3 * 4)
        rng = search.random.Random(1)
        sampled = search.settings(search.sample(both, rng), LAYERS, 'both')
        for part in ('attention', 'mlp', 'residual'):
            self.assertEqual(len(sampled[part]), LAYERS)
            self.assertTrue(all(0 <= w <= 4 for w in sampled[part]))
        # Proposals stay in the residual box.
        history = [dict(trial=0, params=None, refusals=5, kl=0.0)]
        for i in range(1, 20):
            p = search.propose(i, history, box, 4, 20, seed=5)
            self.assertEqual(set(p), set(box))
            for key, (lo, hi) in box.items():
                self.assertTrue(lo <= p[key] <= hi, key)
            history.append(dict(trial=i, params=p, refusals=i % 3, kl=1 / i))

    def test_proposals_stay_in_bounds_and_are_deterministic(self):
        box = search.bounds(LAYERS, [1, 2], 4.0, 'writer')
        history = [dict(trial=0, params=None, refusals=5, kl=0.0)]
        for i in range(1, 30):
            params = search.propose(i, history, box, 5, 30, seed=3)
            self.assertEqual(params, search.propose(i, history, box, 5, 30, seed=3))
            self.assertIn(params['direction_layer'], (1, 2))
            for key, (lo, hi) in box.items():
                self.assertTrue(lo <= params[key] <= hi, key)
            history.append(dict(trial=i, params=params, refusals=i % 4, kl=1 / i))

    def test_fingerprint_modes(self):
        prompts = {'refusal': [dict(id='r', text='x', thinking=False)],
                   'ordinary': [dict(id='o', text='y', thinking=False)]}
        info = FakeClient().props()['abliteration']

        def fingerprint(mode):
            return search.Search(FakeClient(), 'gufo', prompts, info, 'log.jsonl', 8,
                                 out=io.StringIO(), mode=mode).fingerprint

        # Writer mode keeps the D15 fingerprint (earlier logs resume).
        d15 = hashlib.sha256(json.dumps(dict(
            model='gufo', profile=info['identity'], layers=LAYERS, max_tokens=8,
            prompts={'refusal': ['r'], 'ordinary': ['o']}),
            sort_keys=True).encode()).hexdigest()
        self.assertEqual(fingerprint('writer'), d15)
        self.assertEqual(len({fingerprint(m) for m in ('writer', 'residual', 'both')}), 3)

    def test_prompt_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = data_file(Path(tmp))
            chosen = search.load_prompts([path], 'tuning', 'off',
                                         {'refusal': 3, 'ordinary': 0}, 0)
            self.assertEqual(len(chosen['refusal']), 3)
            self.assertEqual(len(chosen['ordinary']), 5)
            self.assertTrue(all(not r['thinking'] for r in chosen['refusal']))
            self.assertEqual(chosen, search.load_prompts(
                [path], 'tuning', 'off', {'refusal': 3, 'ordinary': 0}, 0))
            both = search.load_prompts([path], 'tuning', 'both',
                                       {'refusal': 0, 'ordinary': 0}, 0)
            self.assertEqual(len(both['refusal']), 10)


class SearchRunTest(unittest.TestCase):
    def run_search(self, root, client, trials, extra=()):
        out = io.StringIO()
        chosen = search.main(['--data', str(data_file(root)), '--log',
                              str(root / 'log' / 'trials.jsonl'), '--trials', str(trials),
                              '--initial', '3', '--refusal-prompts', '3',
                              '--ordinary-prompts', '2', '--kl-budget', '10',
                              *extra], client=client, out=out)
        return chosen, out.getvalue()

    def test_residual_mode_requests_log_and_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient()
            # The default mode.
            chosen, text = self.run_search(root, client, 5)
            on = [r['abliteration'] for r in client.requests if r['abliteration']['enabled']]
            self.assertEqual(set(on[0]), {'enabled', 'profile'})  # trial 0: defaults
            later = on[-1]
            self.assertEqual(set(later), {'enabled', 'profile', 'direction_layer',
                                          'attention', 'mlp', 'residual'})
            self.assertEqual((later['attention'], later['mlp']), (0.0, 0.0))
            self.assertEqual(len(later['residual']), LAYERS)
            self.assertTrue(all(0 <= w <= 4 for w in later['residual']))
            log = root / 'log' / 'trials.jsonl'
            records = [json.loads(l) for l in log.read_text().splitlines()]
            self.assertEqual(records[0]['mode'], 'residual')
            self.assertTrue(all(set(r['params']) == {
                'direction_layer', 'residual.max_w', 'residual.peak',
                'residual.min_frac', 'residual.width'} for r in records[2:]))
            # The same log under another mode is another search.
            with self.assertRaises(SystemExit):
                self.run_search(root, FakeClient(), 6, ('--mode', 'writer'))
            with self.assertRaises(SystemExit):
                self.run_search(root, FakeClient(), 6, ('--mode', 'both'))
            # Resume in the same mode evaluates only new trials.
            client2 = FakeClient()
            self.run_search(root, client2, 6)
            self.assertEqual(len(client2.requests), 5)
            # Without a base profile the make-profile command carries --residual.
            self.assertIn('--residual ', text)
            # With one, the chosen residual curve becomes the profile default.
            base = root / 'base.json'
            search.prepare.save_profile(base, dict(
                schema='gufo-abliteration-v2', model_identity='m', hidden=2,
                layers=LAYERS, placement='writer-before-residual',
                directions=[[0.0, 1.0]] * LAYERS, direction_layer=2,
                attention=[1.0] * LAYERS, mlp=[0, 0.5, 0, 0]))
            served = FakeClient()
            served.IDENTITY = hashlib.sha256(base.read_bytes()).hexdigest()
            (root / 'log').rename(root / 'log-old')
            output = root / 'chosen.json'
            chosen, _ = self.run_search(root, served, 6, (
                '--base-profile', str(base), '--output', str(output)))
            profile = json.loads(output.read_text())
            self.assertEqual(len(profile['residual']), LAYERS)
            if chosen['settings']:
                self.assertEqual(profile['attention'], [0.0] * LAYERS)
                self.assertEqual(profile['residual'], chosen['settings']['residual'])
            else:  # the defaults: writer strengths, residual 0
                self.assertEqual(profile['residual'], [0.0] * LAYERS)

    def test_search_requests_log_resume_and_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient()
            chosen, text = self.run_search(root, client, 4, ('--mode', 'writer'))
            self.assertIn('Pareto front', text)
            # Baseline OFF logprobs, then per trial 3 generations + 2 logprob calls.
            off = [r for r in client.requests if not r['abliteration']['enabled']]
            self.assertEqual(len(off), 2)
            self.assertTrue(all(r['logprobs'] and r['top_logprobs'] == 20 and
                                r['max_tokens'] == 1 for r in off))
            self.assertEqual(len(client.requests), 2 + 4 * 5)
            for r in client.requests:
                self.assertEqual(r['chat_template_kwargs'], {'enable_thinking': False})
                self.assertEqual(r['temperature'], 0)
                self.assertFalse(r['cache_persist'])
            on = [r['abliteration'] for r in client.requests if r['abliteration']['enabled']]
            self.assertEqual(set(on[0]), {'enabled', 'profile'})  # trial 0: defaults
            later = on[-1]
            self.assertEqual(len(later['attention']), LAYERS)
            self.assertIn(later['direction_layer'], range(LAYERS))
            self.assertTrue(all(0 <= w <= 4 for w in later['attention'] + later['mlp']))
            log = root / 'log' / 'trials.jsonl'
            records = [json.loads(l) for l in log.read_text().splitlines()]
            self.assertEqual(records[0]['type'], 'header')
            self.assertEqual([r['trial'] for r in records[1:]], [0, 1, 2, 3])
            self.assertEqual(records[1]['refusals'], 3)  # defaults: strength 4.5
            # Resume: only new trials are evaluated, the baseline is cached.
            client2 = FakeClient()
            self.run_search(root, client2, 6, ('--mode', 'writer'))
            self.assertEqual(len(client2.requests), 2 * 5)
            self.assertTrue(all(r['abliteration']['enabled'] for r in client2.requests))
            # A chosen trial becomes a v2 profile with its settings as defaults.
            base = root / 'base.json'
            search.prepare.save_profile(base, dict(
                schema='gufo-abliteration-v2', model_identity='m', hidden=2,
                layers=LAYERS, placement='writer-before-hc',
                directions=[[1.0, 0.0]] * LAYERS, direction_layer=2,
                attention=[1.0] * LAYERS, mlp=[0, 0.5, 0, 0]))
            output = root / 'chosen.json'
            # Another file than the served profile is refused up front.
            with self.assertRaises(SystemExit):
                self.run_search(root, FakeClient(), 6, (
                    '--mode', 'writer', '--base-profile', str(base), '--output', str(output)))
            served = FakeClient()
            served.IDENTITY = hashlib.sha256(base.read_bytes()).hexdigest()
            # Another served profile is another search: a new log.
            (root / 'log').rename(root / 'log-old')
            chosen, text = self.run_search(root, served, 6, (
                '--mode', 'writer', '--base-profile', str(base), '--output', str(output)))
            profile = json.loads(output.read_text())
            self.assertEqual(profile['schema'], 'gufo-abliteration-v2')
            expected = chosen['settings'] or {}
            self.assertEqual(profile['direction_layer'],
                             expected.get('direction_layer', 2))
            self.assertEqual(len(profile['attention']), LAYERS)
            self.assertEqual(profile['search_trial'], chosen['trial'])

    def test_log_of_another_sample_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.run_search(root, FakeClient(), 1)
            with self.assertRaises(SystemExit):
                search.main(['--data', str(data_file(root)), '--log',
                             str(root / 'log' / 'trials.jsonl'), '--trials', '2',
                             '--refusal-prompts', '4', '--ordinary-prompts', '2',
                             '--kl-budget', '1'], client=FakeClient(), out=io.StringIO())

    def test_v1_server_is_refused(self):
        client = FakeClient()
        client.props = lambda: {'abliteration': dict(FakeClient().props()['abliteration'],
                                                     direction_layers=[],
                                                     direction_layer=None)}
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit):
            self.run_search(Path(tmp), client, 1)



def encode(values):
    """Base64 little-endian float32, as prepare.py stores subspace rows."""
    return base64.b64encode(struct.pack(f'<{len(values)}f', *values)).decode()


class SubspaceClient(FakeClient):
    """A server whose profile stores both kinds at rank 2; norm preservation
    and a second direction lower refusals in this toy model."""

    def props(self):
        info = super().props()
        info['abliteration'].update(direction_kinds=['mean_diff', 'projected'],
                                    direction_kind='projected', max_directions=2,
                                    directions=1, preserve_norm=False,
                                    subspace_method='pca')
        return info

    @staticmethod
    def strength(body):
        a = body['abliteration']
        bonus = (a.get('directions', 1) - 1) + (2 if a.get('preserve_norm') else 0)
        return FakeClient.strength(body) + (bonus if a['enabled'] else 0)


class ProjectionAxesTest(unittest.TestCase):
    def test_axes_from_props(self):
        legacy = FakeClient().props()['abliteration']
        self.assertIsNone(search.projection_axes(legacy))
        self.assertIsNone(search.projection_axes(legacy, ['mean_diff'], 1, 'off'))
        for kwargs in (dict(kinds=['projected']), dict(max_directions=2),
                       dict(preserve='on')):
            with self.assertRaises(SystemExit):
                search.projection_axes(legacy, **kwargs)
        info = SubspaceClient().props()['abliteration']
        self.assertEqual(search.projection_axes(info), dict(
            kinds=['mean_diff', 'projected'], max_directions=2,
            preserve_norm=[False, True]))
        self.assertEqual(search.projection_axes(info, ['projected'], 1, 'on'), dict(
            kinds=['projected'], max_directions=1, preserve_norm=[True]))
        for kwargs in (dict(kinds=['pca']), dict(kinds=[]), dict(max_directions=3),
                       dict(max_directions=0)):
            with self.assertRaises(SystemExit):
                search.projection_axes(info, **kwargs)

    def test_bounds_settings_and_proposals(self):
        info = SubspaceClient().props()['abliteration']
        axes = search.projection_axes(info)
        for mode in search.MODES:
            box = search.bounds(LAYERS, [0, 3], 4.0, mode, axes)
            self.assertEqual(box['direction_kind'], (-0.5, 1.5))
            self.assertEqual(box['directions'], (0.5, 2.5))
            self.assertEqual(box['preserve_norm'], (-0.5, 1.5))
            history = [dict(trial=0, params=None, refusals=5, kl=0.0)]
            for i in range(1, 25):
                params = search.propose(i, history, box, 6, 25, seed=2)
                self.assertEqual(params, search.propose(i, history, box, 6, 25, seed=2))
                for key in ('direction_kind', 'directions', 'preserve_norm'):
                    self.assertIsInstance(params[key], int)
                    low, high = box[key]
                    self.assertTrue(math.ceil(low) <= params[key] <= math.floor(high))
                values = search.settings(params, LAYERS, mode, axes)
                self.assertIn(values['direction_kind'], ('mean_diff', 'projected'))
                self.assertIn(values['directions'], (1, 2))
                self.assertIsInstance(values['preserve_norm'], bool)
                history.append(dict(trial=i, params=params, refusals=i % 3, kl=1 / i))
        # Fixed options are sent, not searched.
        fixed = search.projection_axes(info, ['mean_diff'], 1, 'on')
        box = search.bounds(LAYERS, [0, 3], 4.0, 'residual', fixed)
        self.assertFalse({'direction_kind', 'directions', 'preserve_norm'} & set(box))
        values = search.settings(search.sample(box, search.random.Random(0)), LAYERS,
                                 'residual', fixed)
        self.assertEqual((values['direction_kind'], values['directions'],
                          values['preserve_norm']), ('mean_diff', 1, True))
        # Without axes nothing new is sent.
        plain = search.settings(search.sample(box, search.random.Random(0)), LAYERS,
                                'residual')
        self.assertFalse({'direction_kind', 'directions', 'preserve_norm'} & set(plain))

    def test_integer_axes_sample_uniformly(self):
        axes = dict(kinds=['mean_diff'], max_directions=4, preserve_norm=[False])
        box = search.bounds(LAYERS, [0, 3], 4.0, 'residual', axes)
        rng = search.random.Random(4)
        counts = [0] * 5
        for _ in range(4000):
            counts[search.sample(box, rng)['directions']] += 1
        self.assertEqual(counts[0], 0)
        for k in range(1, 5):
            self.assertTrue(850 < counts[k] < 1150, counts)

    def test_fingerprints(self):
        prompts = {'refusal': [dict(id='r', text='x', thinking=False)],
                   'ordinary': [dict(id='o', text='y', thinking=False)]}
        info = SubspaceClient().props()['abliteration']

        def fingerprint(axes, mode='writer'):
            return search.Search(SubspaceClient(), 'gufo', prompts, info, 'log.jsonl', 8,
                                 out=io.StringIO(), mode=mode, axes=axes).fingerprint

        legacy = search.projection_axes(info, ['mean_diff'], 1, 'off')
        self.assertEqual(fingerprint(legacy), fingerprint(None))
        self.assertEqual(fingerprint(legacy, 'residual'), fingerprint(None, 'residual'))
        others = [search.projection_axes(info),
                  search.projection_axes(info, ['projected'], 1, 'off'),
                  search.projection_axes(info, ['mean_diff'], 2, 'off'),
                  search.projection_axes(info, ['mean_diff'], 1, 'on')]
        prints = {fingerprint(a) for a in others}
        self.assertEqual(len(prints), 4)
        self.assertNotIn(fingerprint(None), prints)

    def test_search_over_projection_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'base.json'
            search.prepare.save_profile(base, dict(
                schema='gufo-abliteration-v2', model_identity='m', hidden=2,
                layers=LAYERS, placement='writer-before-residual',
                directions=[[0.0, 1.0]] * LAYERS, direction_layer=2,
                attention=[1.0] * LAYERS, mlp=[0, 0.5, 0, 0], rank=2,
                subspace_directions=[encode([1.0, 0.0])] * LAYERS,
                projected_directions=[encode([1.0, 0.0, 0.0, 1.0])] * LAYERS,
                subspace_method='pca', direction_kind='projected'))
            client = SubspaceClient()
            client.IDENTITY = hashlib.sha256(base.read_bytes()).hexdigest()
            output = root / 'chosen.json'
            out = io.StringIO()
            chosen = search.main(['--data', str(data_file(root)), '--log',
                                  str(root / 'log' / 'trials.jsonl'), '--trials', '8',
                                  '--initial', '4', '--refusal-prompts', '3',
                                  '--ordinary-prompts', '2', '--kl-budget', '10',
                                  '--base-profile', str(base), '--output', str(output)],
                                 client=client, out=out)
            on = [r['abliteration'] for r in client.requests if r['abliteration']['enabled']]
            self.assertEqual(set(on[0]), {'enabled', 'profile'})  # trial 0: defaults
            for request in on[5:]:
                self.assertIn(request['direction_kind'], ('mean_diff', 'projected'))
                self.assertIn(request['directions'], (1, 2))
                self.assertIsInstance(request['preserve_norm'], bool)
            records = [json.loads(l) for l in
                       (root / 'log' / 'trials.jsonl').read_text().splitlines()]
            self.assertEqual(records[0]['type'], 'header')
            self.assertTrue(all({'direction_kind', 'directions', 'preserve_norm'} <=
                                set(r['params']) for r in records[2:]))
            profile = json.loads(output.read_text())
            expected = chosen['settings']
            self.assertEqual(profile['direction_kind'],
                             expected.get('direction_kind', 'projected'))
            self.assertEqual(profile['direction_count'], expected.get('directions', 1))
            self.assertEqual(profile['preserve_norm'], expected.get('preserve_norm', False))
            # The stored rows pass through unchanged.
            self.assertEqual(profile['subspace_directions'], [encode([1.0, 0.0])] * LAYERS)
            self.assertEqual(profile['rank'], 2)
            self.assertIn('k=', out.getvalue())
            # Without a base profile the command carries the options.
            (root / 'log').rename(root / 'log-old')
            out = io.StringIO()
            search.main(['--data', str(data_file(root)), '--log',
                         str(root / 'log' / 'trials.jsonl'), '--trials', '3',
                         '--initial', '2', '--refusal-prompts', '3',
                         '--ordinary-prompts', '2', '--kl-budget', '10',
                         '--preserve-norm', 'on', '--max-directions', '1'],
                        client=SubspaceClient(), out=out)
            self.assertIn('--direction-kind ', out.getvalue())
            self.assertIn('--preserve-norm ', out.getvalue())

class HttpClientTest(unittest.TestCase):
    def test_round_trip_and_errors(self):
        seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, value):
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.reply(200, FakeClient().props())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append((self.path, body))
                if body.get('abliteration', {}).get('direction_layer') == 99:
                    self.reply(400, {'error': {'code': 'invalid_abliteration'}})
                else:
                    self.reply(200, FakeClient().chat(body))

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = search.HttpClient(f'http://127.0.0.1:{server.server_port}/')
            self.assertEqual(client.props()['abliteration']['layers'], LAYERS)
            prompt = dict(text='hi', thinking=True)
            top = search.first_token_top(client.chat(search.request_body(
                'gufo', prompt, {'enabled': False}, 1, logprobs=True)))
            self.assertEqual(len(top), 3)
            self.assertEqual(seen[-1][0], '/v1/chat/completions')
            self.assertEqual(seen[-1][1]['chat_template_kwargs'], {'enable_thinking': True})
            with self.assertRaisesRegex(RuntimeError, 'invalid_abliteration'):
                client.chat(search.request_body('gufo', prompt, {
                    'enabled': True, 'direction_layer': 99}, 8))
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
