"""Synthetic CPU tests only. These results are not HC model calibration."""
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from safetensors.numpy import load_file, save_file

from . import quant as q
from .inventory import inventory
from .weights import load_weight


def bf16_fixture(model, key, bits):
    """Write raw BF16 test bits independently of the reader and torch."""
    model.mkdir()
    payload = bits.astype('<u2').tobytes()
    header = json.dumps({key: {'dtype': 'BF16', 'shape': list(bits.shape),
                              'data_offsets': [0, len(payload)]}}).encode()
    header += b' ' * (-len(header) % 8)
    (model / 'one.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + payload)
    (model / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {key: 'one.safetensors'}}))


class QuantTests(unittest.TestCase):
    def test_bf16_reader_preserves_bits_without_f16_hop(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as d:
            key = 'hc.input_mix_weight_down.weight'
            # +1, -1, 99840 (>F16 max), smallest BF16 subnormal, negative zero.
            bits = np.array(([0x3f80, 0xbf80, 0x47c3, 0x0001, 0x8000] + [0] * 27), dtype=np.uint16).reshape(1, 32)
            model = Path(d) / 'source'; bf16_fixture(model, key, bits)
            w, identity = load_weight(model, key, 'BF16')
            np.testing.assert_array_equal(w.view(np.uint32), bits.astype(np.uint32) << 16)
            self.assertEqual(w[0, 2], 99840)
            self.assertGreater(w[0, 3], 0)
            self.assertEqual(identity['weight_sha256'], q.tensor_sha(bits.astype('<u2')))
            with self.assertRaises(ValueError):
                load_weight(model, key, 'F16')

    def test_bf16_source_above_f16_range_can_fit_q8_scale(self):
        w = np.zeros((1, 32), dtype=np.float32); w[0, 0] = 99840
        candidate, _, _ = q.gptq(w, np.eye(32))
        self.assertTrue(np.isfinite(q.unpack(candidate)).all())
        self.assertGreater(q.unpack(candidate)[0, 0], 65504)
        for bad in (np.finfo(np.float32).max, np.float32(2**-133)):
            w[0, 0] = bad
            with self.assertRaises(ValueError):
                q.gptq(w, np.eye(32))

    def test_rounding_below_and_at_tie(self):
        a = np.array([.5, -.5, 1.5, -1.5, np.nextafter(np.float32(.5), np.float32(0))])
        np.testing.assert_array_equal(q.round_away(a), [1, -1, 2, -2, 0])
        below = np.nextafter(np.float64(.5), np.float64(0))
        above = np.nextafter(np.float64(.5), np.float64(np.inf))
        np.testing.assert_array_equal(q.round_away([below, -below, .5, -.5, above, -above]),
                                      [0, 0, 1, -1, 1, -1])

    def test_rtn_binary_layout_independent_scalar_reference(self):
        rng = np.random.default_rng(43)
        w = rng.normal(size=(3, 64)).astype(np.float16)
        w[0, :32] = 0
        codes, scales = q.rtn(w)
        packed = q.pack(codes, scales)
        reference = bytearray()
        for block in w.reshape(-1, 32):
            d = np.float32(float(max(abs(float(v)) for v in block)) / 127)
            inv = np.float32(1.0 / float(d)) if d else np.float32(0)
            reference.extend(struct.pack('<e', float(d)))
            for v in block:
                value = float(np.float32(np.float32(v) * inv))
                integer = int(abs(value) + .5) * (-1 if value < 0 else 1)
                reference.extend(struct.pack('b', integer))
        self.assertEqual(packed.tobytes(), bytes(reference))
        self.assertEqual(packed.nbytes, w.size // 32 * 34)

    def test_diagonal_h_is_nearest_stored_scale(self):
        w = np.random.default_rng(4).normal(size=(5, 64)).astype(np.float16)
        candidate, control, _ = q.gptq(w, np.diag(np.arange(1, 65)))
        _, scales = q.rtn(w)
        expected = np.clip(q.round_away(w.astype(float) / np.repeat(scales.astype(float), 32, axis=1)), -127, 127)
        np.testing.assert_array_equal(candidate[..., 2:].view(np.int8).reshape(w.shape), expected)
        np.testing.assert_array_equal(candidate[..., :2], control[..., :2])

    def test_correlated_h_feedback_and_determinism(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(256, 64))
        x[:, 1::2] = x[:, ::2] * .95 + x[:, 1::2] * .05
        h = x.T @ x / len(x)
        w = rng.normal(size=(8, 64)).astype(np.float16)
        candidate, control, _ = q.gptq(w, h)
        again, _, _ = q.gptq(w, h)
        np.testing.assert_array_equal(candidate, again)
        c = q.measure(w, candidate, h, x)
        b = q.measure(w, control, h, x)
        self.assertLess(c['calibration_weighted_sse'], b['calibration_weighted_sse'])
        self.assertAlmostEqual(c['calibration_weighted_sse'], c['holdout_output_sse'] / len(x), places=10)

    def test_invalid_inputs_fail_without_fallback(self):
        w = np.ones((2, 32), dtype=np.float16)
        for h in (np.zeros((32, 32)), np.full((32, 32), np.nan), -np.eye(32)):
            with self.assertRaises(ValueError):
                q.gptq(w, h)
        h = np.eye(32); h[0, 1] = 1
        with self.assertRaises(ValueError):
            q.gptq(w, h)
        h = np.eye(32); h[0, 1] = h[1, 0] = 10
        with self.assertRaises(np.linalg.LinAlgError):
            q.gptq(w, h)
        with self.assertRaises(ValueError):
            q.rtn(np.full((2, 32), np.nextafter(np.float16(0), np.float16(1)), dtype=np.float16))

    def test_file_workflow_and_provenance_rejections(self):
        # Deliberately synthetic fixture impersonates the receipt schema only
        # inside a temporary directory; never exported as measured model H.
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as d:
            root = Path(d); model = root / 'model'; model.mkdir()
            key = 'fixture.hc.input_mix_weight_down.weight'
            original = np.random.default_rng(2).normal(size=(3, 32)).astype(np.float32)
            bits = (original.view(np.uint32) >> 16).astype(np.uint16)
            source = root / 'source'; bf16_fixture(source, key, bits)
            source_w, source_identity = load_weight(source, key, 'BF16')
            w = source_w.astype(np.float16)
            w[0, 0] += np.float16(.125)  # prove the reference/candidate comes from BF16, not target pack
            save_file({key: w}, str(model / 'one.safetensors'))
            index = model / 'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map': {key: 'one.safetensors'}}))
            stats = root / 'stats.npz'
            np.savez(stats, H=np.eye(32), X_holdout=np.eye(32))
            _, target_identity = load_weight(model, key, 'F16')
            meta = {'schema': 'gufo-hc-input-gram-v2', 'tensor': key,
                    'source_identity': source_identity, 'target_identity': target_identity,
                    'stats_sha256': q.sha(stats), 'input_space': 'unfolded_projection_input',
                    'H_normalization': 'sum_xxt_div_rows', 'synthetic': False,
                    'calibration_document_ids': ['fixture_train'], 'holdout_document_ids': ['fixture_test'],
                    'calibration_rows': 64, 'holdout_rows': 32,
                    'capture_source_sha256': 'a'*64, 'capture_runtime_sha256': 'b'*64,
                    'upstream_weight_state_sha256': 'c'*64}
            receipt = root / 'receipt.json'; receipt.write_text(json.dumps(meta))
            result = q.generate(source, model, key, stats, receipt, root / 'out', .01)
            self.assertTrue(result['scale_bytes_equal'])
            self.assertEqual(result['weight_loader_sha256'], q.sha(Path(q.__file__).with_name('weights.py')))
            expected, _, _ = q.gptq(source_w, np.eye(32))
            np.testing.assert_array_equal(load_file(str(root / 'out/candidate.safetensors'))[key], expected)
            self.assertGreater(result['bf16_vs_pack_f16_max_abs'], .1)
            self.assertEqual(load_file(str(root / 'out/candidate.safetensors'))[key].shape, (3, 1, 34))
            self.assertEqual(inventory(source)['tensor_count'], 1)
            self.assertEqual(inventory(model, 'F16')['tensor_count'], 1)
            self.assertEqual(result['reference'], 'original_BF16_exactly_expanded_to_FP32')
            self.assertTrue((root / 'out/legacy-f16-rtn-control.safetensors').exists())
            with self.assertRaises(ValueError):
                q.generate(source, model, key, stats, receipt, root / 'out', .01)
            with self.assertRaises(ValueError):
                q.generate(source, model, key, stats, receipt, model / 'out', .01)
            with self.assertRaises(ValueError):
                q.generate(source, model, key, stats, receipt, source / 'out', .01)
            with self.assertRaises(ValueError):
                q.generate(model, model, key, stats, receipt, root / 'reject_f16_source', .01)
            for mutation in ({'stats_sha256': 'd'*64}, {'holdout_document_ids': ['fixture_train']},
                             {'source_identity': {}}, {'target_identity': {}}, {'synthetic': True}, {'holdout_rows': 31}):
                receipt.write_text(json.dumps(meta | mutation))
                with self.assertRaises(ValueError):
                    q.generate(source, model, key, stats, receipt, root / 'rejected', .01)
                self.assertFalse((root / 'rejected').exists())


if __name__ == '__main__':
    unittest.main()
