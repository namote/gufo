"""Small independent checks for held-out metric interpretation."""
import unittest

import numpy as np

from .score_saved import connected_errors, errors, silu_quarter
from .quant import pack


class ScoreTests(unittest.TestCase):
    def test_projection_relative_rmse(self):
        w = np.ones((1, 32))
        x = np.eye(32)
        packed = pack(np.full((1, 32), 2, dtype=np.int8), np.ones((1, 1), dtype=np.float16))
        metric = errors(x, w, packed)
        self.assertEqual(metric['output_sse'], 32)
        self.assertEqual(metric['reference_sse'], 32)
        self.assertEqual(metric['relative_rmse'], 1)
        with self.assertRaises(ValueError):
            errors(x, np.zeros_like(w), packed)

    def test_connected_scale_and_extreme_silu(self):
        np.testing.assert_allclose(silu_quarter(np.array([-1000., 0., 1000.])), [0., 0., 250.], atol=1e-100)
        x = np.array([[1., -1.]])
        down = np.eye(2)
        reference = (x / 4) / (1 + np.exp(-x / 4))
        exact = connected_errors(x, down, down, reference)
        self.assertLess(exact['output_sse'], 1e-30)
        doubled = connected_errors(x, down, 2 * down, reference)
        self.assertAlmostEqual(doubled['relative_rmse'], 1)


if __name__ == '__main__':
    unittest.main()
