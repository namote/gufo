"""CPU test for spsa.py against a fake score_assistant server (standard library)."""
import http.server
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest

spec = importlib.util.spec_from_file_location('spsa', Path(__file__).with_name('spsa.py'))
spsa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spsa)
LAYERS = 4


class Fake(http.server.BaseHTTPRequestHandler):
    """Candid answers gain likelihood with mlp strength at layer 2; ordinary
    answers lose likelihood with any residual strength."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        answer = body['messages'][-1]['content']
        ab = body.get('abliteration')
        mlp = ab['mlp'][2] if ab else 0.0
        residual = sum(ab['residual']) if ab else 0.0
        if answer.startswith('candid'):
            value = -2.0 + 0.5 * min(mlp, 2.0)
        elif answer.startswith('prop'):
            value = -1.0
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


class SpsaTest(unittest.TestCase):
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
