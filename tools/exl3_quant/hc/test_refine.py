"""Independent small CPU oracles; no model or GPU execution."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file, load_file

from . import quant as q
from . import refine as r
from .test_cpu import bf16_fixture
from .weights import load_weight


class RefinementTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(82)
        x = rng.normal(size=(160, 64)); x[:, 32:] += x[:, :32] * 2
        w = rng.normal(size=(3, 64)).astype(np.float32)
        h = x.T @ x / len(x)
        b, _, damping = q.gptq(w, h)
        return w, h, b, damping, x

    def test_code_sweep_matches_brute_force_full_objective(self):
        w, h, b, damping, _ = self.fixture()
        hd = h + damping * np.eye(64)
        c = b[..., 2:].copy().view(np.int8).reshape(w.shape).astype(np.int16)
        scales = b[..., :2].copy().view('<f2').reshape(3, 2).astype(float)
        # Literal full-matrix objective for each candidate; no G update formula.
        def loss(row, codes):
            e = w[row].astype(float) - codes * np.repeat(scales[row], 32)
            return float(e @ hd @ e)
        for j in range(64):
            for row in range(3):
                before = c[row].copy(); best = loss(row, before); chosen = before
                for step in (-1, 1):
                    trial = before.copy(); trial[j] += step
                    if -127 <= trial[j] <= 127 and loss(row, trial) < best:
                        best, chosen = loss(row, trial), trial
                c[row] = chosen
        got, _ = r.refine(w, b, h, damping, sweeps=1)
        np.testing.assert_array_equal(got, q.pack(c.astype(np.int8), scales.astype(np.float16)))

    def test_scale_step_matches_exhaustive_positive_f16_grid(self):
        codes = np.arange(96, 128, dtype=np.int8).reshape(1, 32)
        w = (codes.astype(float) * 1.0007).astype(np.float32)
        initial = q.pack(codes, np.ones((1, 1), dtype=np.float16))
        got, info = r.refine(w, initial, np.eye(32), .01, sweeps=1, scales=True)
        grid = np.arange(1, 0x7c00, dtype=np.uint16).view(np.float16).astype(float)
        errors = np.sum((w - grid[:, None] * codes) ** 2, axis=1)
        expected = np.float16(grid[np.argmin(errors)])
        np.testing.assert_array_equal(got[..., :2], q.pack(codes, np.array([[expected]], dtype=np.float16))[..., :2])
        self.assertEqual(info['trace'][0]['code_moves'], 0)
        self.assertEqual(info['trace'][0]['scale_moves'], 1)

    def test_repeat_batch_invariance_and_actual_output_objective(self):
        w, h, b, damping, x = self.fixture()
        for scales in (False, True):
            got, info = r.refine(w, b, h, damping, scales=scales, row_batch=1)
            again, _ = r.refine(w, b, h, damping, scales=scales, row_batch=3)
            np.testing.assert_array_equal(got, again)
            self.assertTrue(all(t['after'] <= t['before'] + 1e-14 for t in info['trace']))
            err = q.unpack(got) - w
            metric = q.measure(w, got, h, x)['calibration_weighted_sse']
            self.assertAlmostEqual(metric, float(np.sum((x @ err.T)**2)) / len(x), places=11)
            self.assertFalse(np.array_equal(b[..., 2:], got[..., 2:]))
            if not scales:
                np.testing.assert_array_equal(b[..., :2], got[..., :2])

    def test_boundary_scales_zero_and_invalid_packs(self):
        w = np.zeros((1, 32), dtype=np.float32)
        b = q.pack(np.zeros((1, 32), dtype=np.int8), np.zeros((1, 1), dtype=np.float16))
        got, _ = r.refine(w, b, np.eye(32), .01, scales=True)
        np.testing.assert_array_equal(got, b)
        choices = r.scale_choices(np.array([-1., 1e-100, 1e100]))
        self.assertEqual(choices[0, 0], float(np.nextafter(np.float16(0), np.float16(1))))
        self.assertEqual(choices[-1, 0], 65504.)
        for broken in (np.full((32, 32), np.nan), -np.eye(32)):
            with self.assertRaises((ValueError, np.linalg.LinAlgError)):
                r.refine(w, b, broken, .01)
        invalid = b.copy(); invalid[0, 0, 2] = 128
        with self.assertRaises(ValueError): r.refine(w, invalid, np.eye(32), .01)
        with self.assertRaises(ValueError): r.refine(w + 1, b, np.eye(32), .01)

    def test_saved_baseline_reuse_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp); source = root / 'source'; target = root / 'target'; target.mkdir()
            key = 'fixture.input_mix_weight_up.weight'
            bits = (np.random.default_rng(3).normal(size=(2, 32)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            bf16_fixture(source, key, bits)
            w, src = load_weight(source, key, 'BF16')
            save_file({key: w.astype(np.float16)}, str(target / 'weights.safetensors'))
            (target / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {key: 'weights.safetensors'}}))
            _, dst = load_weight(target, key, 'F16')
            stats = root / 'stats.npz'; np.savez(stats, H=np.eye(32), X_holdout=np.eye(32))
            receipt = {'schema': 'gufo-hc-input-gram-v2', 'tensor': key, 'source_identity': src,
                       'target_identity': dst, 'stats_sha256': q.sha(stats), 'input_space': 'unfolded_projection_input',
                       'H_normalization': 'sum_xxt_div_rows', 'synthetic': False,
                       'calibration_rows': 64, 'holdout_rows': 32, 'calibration_document_ids': ['train'],
                       'holdout_document_ids': ['holdout'], 'capture_source_sha256': 'a'*64,
                       'capture_runtime_sha256': 'b'*64, 'upstream_weight_state_sha256': 'c'*64}
            path = root / 'receipt.json'; path.write_text(json.dumps(receipt))
            baseline = root / 'baseline'; q.generate(source, target, key, stats, path, baseline, .01)
            before = {p.name: q.sha(p) for p in baseline.iterdir()}
            out = root / 'refined'; result = r.compare(source, target, baseline, stats, out)
            self.assertEqual(before, {p.name: q.sha(p) for p in baseline.iterdir()})
            self.assertEqual(len(result['metrics']), 5)
            self.assertEqual(result['selection'], 'none')
            self.assertFalse(result['holdout_used_for_optimization'])
            for name in ('gptq_codes', 'gptq_codes_scales'):
                self.assertEqual(load_file(str(out / (name + '.safetensors')))[key].shape, (2, 1, 34))
            # Change holdout only, issue a consistent receipt in a synthetic
            # copied baseline: candidate bytes must remain independent of it.
            changed = root / 'changed-stats.npz'; np.savez(changed, H=np.eye(32), X_holdout=np.eye(32)*100)
            meta = json.loads((baseline / 'RESULT.json').read_text()); meta['receipt']['stats_sha256'] = q.sha(changed)
            (baseline / 'RESULT.json').write_text(json.dumps(meta))
            out2 = root / 'other-holdout'; r.compare(source, target, baseline, changed, out2)
            for name in ('gptq_codes', 'gptq_codes_scales'):
                np.testing.assert_array_equal(load_file(str(out / (name + '.safetensors')))[key],
                                              load_file(str(out2 / (name + '.safetensors')))[key])
            with (baseline / 'candidate.safetensors').open('ab') as f: f.write(b'changed')
            with self.assertRaises(ValueError): r.compare(source, target, baseline, changed, root / 'reject')
            self.assertFalse((root / 'reject').exists())


if __name__ == '__main__':
    unittest.main()
