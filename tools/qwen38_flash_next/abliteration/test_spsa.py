"""CPU test for spsa.py against a fake score_assistant server (standard library)."""
import http.server
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

spec = importlib.util.spec_from_file_location('spsa', Path(__file__).with_name('spsa.py'))
spsa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spsa)
LAYERS = 4


class Fake(http.server.BaseHTTPRequestHandler):
    """Candid answers gain likelihood with mlp strength at layer 2; ordinary
    answers lose likelihood with any residual strength; the correct fact
    answer loses its lead over the wrong one with attention strength at
    layer 0, which nothing else rewards or penalizes."""

    bodies = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        Fake.bodies.append(body.get('abliteration'))
        answer = body['messages'][-1]['content']
        ab = body.get('abliteration')
        mlp = ab['mlp'][2] if ab else 0.0
        residual = sum(ab['residual']) if ab else 0.0
        if answer.startswith('candid'):
            value = -2.0 + 0.5 * min(mlp, 2.0)
        elif answer.startswith('prop') or answer.startswith('wrong'):
            value = -1.0
        elif answer.startswith('right'):
            value = -1.0 + 2.0 * (1.0 - (ab['attention'][0] if ab else 0.0))
        else:
            value = -1.0 - 0.2 * residual
        reply = json.dumps({'score': {'token_count': 1, 'logprob_sum': value}}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


class FakeTrace(http.server.BaseHTTPRequestHandler):
    """Reasoning rows: a re-check opener ("Wait") loses likelihood with total
    residual strength, its forward alternative does not, and the guard answer
    loses likelihood once total residual strength passes 2. Records every
    request and whether two requests of one (settings, prompt) chain were in
    flight together."""

    lock = threading.Lock()
    bodies, active, overlap = [], {}, []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        ab = body.get('abliteration')
        chain = (json.dumps(ab, sort_keys=True), body['messages'][0]['content'])
        with FakeTrace.lock:
            FakeTrace.bodies.append(body)
            FakeTrace.active[chain] = FakeTrace.active.get(chain, 0) + 1
            if FakeTrace.active[chain] > 1:
                FakeTrace.overlap.append(chain)
        time.sleep(0.002)
        residual = sum(ab['residual']) if ab else 0.0
        answer = body['messages'][-1]['content']
        if answer.startswith('Wait'):
            value = -1.0 - 0.1 * residual
        elif answer.startswith('Guarded'):
            value = -0.5 - 0.3 * max(0.0, residual - 2.0)
        else:
            value = -1.0
        reply = json.dumps({'score': {'token_count': 1, 'logprob_sum': value}}).encode()
        with FakeTrace.lock:
            FakeTrace.active[chain] -= 1
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


class SpsaTest(unittest.TestCase):
    def test_reasoning_rows_chains_and_guard(self):
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), FakeTrace)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f'http://127.0.0.1:{server.server_address[1]}'
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pairs, guard = [], []
            for t in range(3):
                for b in (3, 1, 2):  # out of order: the chain sorts by reasoning length
                    pairs.append({'id': f't{t}b{b}', 'prompt': f'problem {t}', 'reasoning': 'x\n\n' * b,
                                  'score_reasoning': True, 'reasoning_effort': 'xhigh',
                                  'score_max_tokens': 3, 'chain': f't{t}',
                                  'candid': 'So go on.', 'propaganda': 'Wait, check.'})
                guard.append({'id': f't{t}g', 'prompt': f'problem {t}', 'reasoning': 'x\n\n' * 9,
                              'reasoning_effort': 'xhigh', 'chain': f't{t}', 'answer': 'Guarded answer'})
            (d / 'pairs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pairs))
            (d / 'guard.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in guard))
            base = ['--base-url', url, '--pairs', str(d / 'pairs.jsonl'), '--guard', str(d / 'guard.jsonl'),
                    '--nu', '4', '--layers', str(LAYERS), '--direction-layer', '1',
                    '--log', str(d / 'log.jsonl'), '--concurrency', '6', '--eval-every', '5',
                    '--a', '0.3', '--c', '0.2', '--seed', '3', '--answer-tokens', '64',
                    '--learn', 'mlp,residual,mlp', '--init', '{"attention": [0.5, 0, 0, 0], "mlp": [0, 0, 0, 0], '
                    '"residual": [0, 0, 0, 0]}']
            with self.assertRaises(SystemExit):  # an unknown site kind
                spsa.main(base + ['--steps', '1', '--learn', 'resid'])
            with self.assertRaises(SystemExit):  # one minibatch kind at a time
                spsa.main(base + ['--steps', '1', '--minibatch', '2', '--minibatch-chains', '1'])
            with self.assertRaises(SystemExit):  # --ordinary is required without --guard
                spsa.main(base[:4] + base[6:] + ['--steps', '1'])
            FakeTrace.bodies.clear()
            FakeTrace.overlap.clear()
            self.assertEqual(spsa.main(base + ['--steps', '30', '--minibatch-chains', '2']), 0)
            self.assertEqual(FakeTrace.overlap, [])  # a chain's requests never overlap
            on = [b for b in FakeTrace.bodies if b.get('abliteration')]
            opener = next(b for b in on if b.get('score_reasoning'))
            self.assertEqual(opener['messages'][1]['reasoning_content'][:3], 'x\n\n')
            self.assertEqual((opener['reasoning_effort'], opener['score_max_tokens']), ('xhigh', 3))
            closed = next(b for b in on if b['messages'][1]['content'] == 'Guarded answer')
            self.assertNotIn('score_reasoning', closed)
            self.assertEqual(closed['score_max_tokens'], 64)  # no row cut: --answer-tokens
            # Within a chain and a setting, requests run in ascending reasoning length.
            seen = {}
            for b in on:
                k = (json.dumps(b['abliteration'], sort_keys=True), b['messages'][0]['content'])
                seen.setdefault(k, []).append(len(b['messages'][1].get('reasoning_content', '')))
            self.assertTrue(all(v == sorted(v) for v in seen.values()))
            lines = [json.loads(l) for l in (d / 'log.jsonl').read_text().splitlines()]
            header = lines[0]
            self.assertEqual(header['guard'], ['t0g', 't1g', 't2g'])
            self.assertEqual(header['args']['minibatch_chains'], 2)
            self.assertEqual(header['args']['learn'], 'mlp,residual')
            # Attention is not learned: never perturbed, kept at its init.
            self.assertTrue(all(b['abliteration']['attention'] == [0.5, 0, 0, 0] for b in on))
            self.assertTrue(any(max(b['abliteration']['mlp']) > 0 for b in on))
            self.assertIsNone(header['args']['ordinary'])
            # A chain minibatch scores every row of the chosen chains: 2 chains x (3 pairs x 2 + guard 1) x 2 thetas.
            step_bodies = 2 * (3 * 2 + 1) * 2
            self.assertGreaterEqual(len(on), step_bodies)
            last = lines[-1]
            self.assertIn('guard_excess', last['eval'])
            self.assertEqual(last['theta'][:LAYERS], [0.5, 0.0, 0.0, 0.0])
            total = sum(last['theta'][2 * LAYERS:])
            self.assertGreater(total, 1.0)   # suppression grows ...
            self.assertLess(total, 3.5)      # ... until the guard holds it near 2
            off = json.loads((d / 'log.jsonl.off.json').read_text())
            self.assertEqual(off['off'], {})
            self.assertAlmostEqual(off['guard']['t0g'], -0.5)
            # Resume keeps the header (and the OFF cache).
            self.assertEqual(spsa.main(base + ['--steps', '32', '--minibatch-chains', '2']), 0)
        server.shutdown()


    def test_signed_strengths_for_additive_profiles(self):
        self.assertEqual(spsa.clip([-5.0, 1.0, 70.0], 4.0), [0.0, 1.0, 4.0])
        self.assertEqual(spsa.clip([-70.0, -5.0, 70.0], 64.0, -64.0),
                         [-64.0, -5.0, 64.0])

    def test_additive_requests_carry_only_strengths(self):
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Fake)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f'http://127.0.0.1:{server.server_address[1]}'
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'pairs.jsonl').write_text(json.dumps(
                {'id': 'p', 'prompt': 'q', 'candid': 'candid a', 'propaganda': 'prop a'}) + '\n')
            (d / 'ord.jsonl').write_text(json.dumps({'id': 'o', 'prompt': 'q', 'answer': 'plain'}) + '\n')
            base = ['--base-url', url, '--pairs', str(d / 'pairs.jsonl'), '--ordinary', str(d / 'ord.jsonl'),
                    '--layers', str(LAYERS), '--init', '-1', '--log', str(d / 'log.jsonl'),
                    '--concurrency', '1', '--eval-every', '5', '--seed', '3']
            with self.assertRaises(SystemExit):   # a suppression run needs its layer
                spsa.main(base + ['--steps', '1'])
            Fake.bodies.clear()
            self.assertEqual(spsa.main(base + ['--additive', '--profile', 'steer',
                                               '--min-strength', '-64', '--max-strength', '64',
                                               '--steps', '2']), 0)
            sent = [b for b in Fake.bodies if b]
            self.assertTrue(sent)
            for body in sent:
                self.assertEqual(set(body), {'enabled', 'profile', 'attention', 'mlp', 'residual'})
            self.assertTrue(any(min(b['attention']) < 0 for b in sent))  # signed strengths kept
            header = json.loads((d / 'log.jsonl').read_text().splitlines()[0])
            self.assertTrue(header['args']['additive'])
        server.shutdown()

    def test_blocks_round_trip(self):
        theta = [float(i) for i in range(3 * 5)]
        params = spsa.compress(theta, 5, 2)
        self.assertEqual(params, [0.5, 2.5, 4.0, 5.5, 7.5, 9.0, 10.5, 12.5, 14.0])
        self.assertEqual(spsa.expand(params, 5, 2)[:5], [0.5, 0.5, 2.5, 2.5, 4.0])
        self.assertEqual(spsa.expand(spsa.compress(theta, 5, 1), 5, 1), theta)

    def test_blocks_and_minibatch_learn(self):
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Fake)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f'http://127.0.0.1:{server.server_address[1]}'
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'pairs.jsonl').write_text(''.join(json.dumps(
                {'id': f'p{i}', 'prompt': f'q{i}', 'candid': 'candid a', 'propaganda': 'prop a'}) + '\n'
                for i in range(4)))
            (d / 'ord.jsonl').write_text(''.join(json.dumps(
                {'id': f'o{i}', 'prompt': f'q{i}', 'answer': 'plain'}) + '\n' for i in range(4)))
            base = ['--base-url', url, '--pairs', str(d / 'pairs.jsonl'), '--ordinary', str(d / 'ord.jsonl'),
                    '--layers', str(LAYERS), '--direction-layer', '1', '--init', '0.5',
                    '--log', str(d / 'log.jsonl'), '--concurrency', '4', '--eval-every', '5',
                    '--a', '0.3', '--c', '0.2', '--seed', '3', '--block-size', '2', '--minibatch', '2']
            self.assertEqual(spsa.main(base + ['--steps', '20', '--output', str(d / 'best.json')]), 0)
            records = [json.loads(l) for l in (d / 'log.jsonl').read_text().splitlines()[1:]]
            self.assertEqual(len(records[-1]['theta']), 3 * LAYERS // 2)
            best = json.loads((d / 'best.json').read_text())['settings']
            self.assertEqual(len(best['mlp']), LAYERS)
            self.assertEqual(best['mlp'][2], best['mlp'][3])     # one block
            self.assertGreater(best['mlp'][2], 0.5)               # the block holding layer 2 grows
            self.assertLess(sum(best['residual']), 0.5 * LAYERS)
            # The resumed run reproduces the same subsets and stream.
            self.assertEqual(spsa.main(base + ['--steps', '22']), 0)
            records = [json.loads(l) for l in (d / 'log.jsonl').read_text().splitlines()[1:]]
            self.assertEqual([r['step'] for r in records], list(range(-1, 22)))
        server.shutdown()

    def test_facts_guard(self):
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Fake)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f'http://127.0.0.1:{server.server_address[1]}'
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'pairs.jsonl').write_text(json.dumps(
                {'id': 'p', 'prompt': 'q', 'candid': 'candid a', 'propaganda': 'prop a'}) + '\n')
            (d / 'ord.jsonl').write_text(json.dumps({'id': 'o', 'prompt': 'q', 'answer': 'plain'}) + '\n')
            (d / 'facts.jsonl').write_text(''.join(json.dumps(
                {'id': f'f{i}', 'prompt': f'fq{i}', 'correct': 'right a', 'wrong': 'wrong a'}) + '\n'
                for i in range(3)))
            run = lambda log, steps: spsa.main([
                '--base-url', url, '--pairs', str(d / 'pairs.jsonl'), '--ordinary', str(d / 'ord.jsonl'),
                '--facts', str(d / 'facts.jsonl'), '--mu', '1', '--layers', str(LAYERS),
                '--direction-layer', '1', '--init', '0.5', '--log', str(d / log),
                '--concurrency', '4', '--eval-every', '5', '--a', '0.3', '--c', '0.2', '--seed', '3',
                '--minibatch', '2', '--steps', str(steps)])
            self.assertEqual(run('log.jsonl', 10), 0)
            self.assertEqual(run('log.jsonl', 20), 0)   # resumed
            lines = [json.loads(l) for l in (d / 'log.jsonl').read_text().splitlines()]
            self.assertEqual(lines[0]['facts'], ['f0', 'f1', 'f2'])
            self.assertEqual([r['step'] for r in lines[1:]], list(range(-1, 20)))
            self.assertEqual(run('straight.jsonl', 20), 0)
            straight = (d / 'straight.jsonl').read_text().splitlines()
            self.assertEqual(json.loads(straight[-1])['theta'], lines[-1]['theta'])
            first, last = lines[1], lines[-1]
            # attention 0.5 at layer 0: gap 1 nat, shortfall 1 against the 2-nat margin.
            self.assertAlmostEqual(first['eval']['fact_shortfall'], 1.0)
            self.assertEqual(first['eval']['fact_right'], 3)
            self.assertAlmostEqual(first['eval']['j'], first['eval']['margin'] - 4 * first['eval']['drop'] - 1.0)
            self.assertLess(last['theta'][0], 0.25)              # attention at layer 0 falls
            self.assertLess(last['eval']['fact_shortfall'], 0.5)
        server.shutdown()

    def test_learns_and_resumes(self):
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Fake)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f'http://127.0.0.1:{server.server_address[1]}'
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'pairs.jsonl').write_text(json.dumps(
                {'id': 'p', 'prompt': 'q', 'candid': 'candid a', 'propaganda': 'prop a'}) + '\n')
            (d / 'ord.jsonl').write_text(json.dumps({'id': 'o', 'prompt': 'q', 'answer': 'plain'}) + '\n')
            base = ['--base-url', url, '--pairs', str(d / 'pairs.jsonl'), '--ordinary', str(d / 'ord.jsonl'),
                    '--layers', str(LAYERS), '--direction-layer', '1', '--init', '0.5',
                    '--log', str(d / 'log.jsonl'), '--concurrency', '1', '--eval-every', '5',
                    '--a', '0.3', '--c', '0.2', '--seed', '3']
            self.assertEqual(spsa.main(base + ['--steps', '20']), 0)
            header = json.loads((d / 'log.jsonl').read_text().splitlines()[0])
            self.assertNotIn('facts', header)   # logs of runs without --facts keep their old header
            self.assertFalse({'facts', 'mu', 'fact_margin'} & set(header['args']))
            records = [json.loads(l) for l in (d / 'log.jsonl').read_text().splitlines()[1:]]
            self.assertEqual(records[0]['step'], -1)
            last = records[-1]['theta']
            self.assertGreater(last[LAYERS + 2], 0.5)            # mlp at layer 2 grows
            self.assertLess(sum(last[2 * LAYERS:]), 0.5 * LAYERS)  # residual shrinks
            self.assertTrue(all(0.0 <= x <= 4.0 for x in last))
            # Resume continues the same run.
            self.assertEqual(spsa.main(base + ['--steps', '25', '--output', str(d / 'best.json')]), 0)
            records = [json.loads(l) for l in (d / 'log.jsonl').read_text().splitlines()[1:]]
            self.assertEqual([r['step'] for r in records], list(range(-1, 25)))
            best = json.loads((d / 'best.json').read_text())
            self.assertEqual(len(best['settings']['mlp']), LAYERS)
            # Another data set is refused for this log.
            with self.assertRaises(SystemExit):
                spsa.main([*base[:5], str(d / 'pairs.jsonl'), '--ordinary', str(d / 'ord.jsonl'),
                           '--layers', str(LAYERS), '--direction-layer', '2', '--init', '0.5',
                           '--log', str(d / 'log.jsonl'), '--steps', '26'])
        server.shutdown()


if __name__ == '__main__':
    unittest.main()
