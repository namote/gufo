#!/usr/bin/env python3
"""CPU tests of the W8 record builder: the candidate top-k path must give byte-identical records
to logits_io.records_from_logits (the Python mirror of the gufo collector), including tie rows,
and the run must load and pair in metrics.py. The torch path runs when torch is importable (the
EXL3 image on CPU); otherwise it is skipped."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import logits_io  # noqa: E402
import metrics  # noqa: E402
import records  # noqa: E402

K = 16


def synthetic(seed: int, n: int = 40, vocab: int = 600) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = (rng.standard_normal((n, vocab)) * 3).astype(np.float32)
    # Tie rows: a tie group across the k boundary, a group longer than the candidate pad, and
    # a row of identical values.
    logits[1, 50:60] = logits[1].max() + 1
    logits[1, 300:310] = logits[1].max()
    logits[2, :] = np.round(logits[2]) / 4
    logits[3, 100:100 + K + records.CANDIDATE_PAD + 5] = 20.0
    logits[4, :] = 1.5
    targets = rng.integers(0, vocab, n)
    return logits, targets


class RecordsTest(unittest.TestCase):
    def test_numpy_matches_collector_mirror(self):
        for seed in range(3):
            logits, targets = synthetic(seed)
            n = len(logits)
            win, pos = np.full(n, 2), np.arange(n)
            ref = logits_io.records_from_logits(logits, targets, win, pos, K)
            got = records.records_numpy(logits, targets, win, pos, K)
            self.assertEqual(ref.tobytes(), got.tobytes())

    def test_reference_ids_and_metrics_pairing(self):
        logits, targets = synthetic(7)
        noisy = logits + np.random.default_rng(1).standard_normal(logits.shape).astype(np.float32) * 0.05
        n = len(logits)
        win, pos = np.zeros(n, dtype=int), np.arange(n)
        with tempfile.TemporaryDirectory() as tmp:
            meta = dict(path="prefill", windows_tokens_sha256="x", windows=[dict(index=0, category="c")])
            ref = logits_io.records_from_logits(logits, targets, win, pos, K)
            rmeta = logits_io.write(f"{tmp}/ref", ref, **meta)
            cand = records.records_numpy(noisy, targets, win, pos, K, reference_ids=ref["ids"])
            mirror = logits_io.records_from_logits(noisy, targets, win, pos, K, reference_ids=ref["ids"])
            self.assertEqual(cand.tobytes(), mirror.tobytes())
            logits_io.write(f"{tmp}/cand", cand, reference=dict(bin_sha256=rmeta["bin_sha256"]), **meta)
            report = metrics.score(logits_io.load(f"{tmp}/ref"), logits_io.load(f"{tmp}/cand"), 200, 8, 1)
            self.assertGreater(report["kl"]["mean"], 0)

    def test_torch_path(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        logits, targets = synthetic(11)
        n = len(logits)
        win, pos = np.full(n, 5), np.arange(n)
        ref_ids = logits_io.records_from_logits(logits[::-1].copy(), targets, win, pos, K)["ids"]
        for dtype in (torch.float32, torch.float16):
            x = torch.as_tensor(logits).to(dtype)
            values = x.float().numpy()
            base, refs = records.records_torch(x, targets, win, pos, K, [None, ref_ids])
            own = logits_io.records_from_logits(values, targets, win, pos, K)
            other = logits_io.records_from_logits(values, targets, win, pos, K, reference_ids=ref_ids)
            # logsumexp is float64 either way but summed in a different order: last-ulp equal only.
            np.testing.assert_allclose(base["logsumexp"], own["logsumexp"], rtol=1e-14, atol=0)
            same = base.copy()
            same["logsumexp"] = own["logsumexp"]
            self.assertEqual(same.tobytes(), own.tobytes())
            np.testing.assert_array_equal(refs[0], own["ref_logits"])
            np.testing.assert_array_equal(refs[1], other["ref_logits"])


if __name__ == "__main__":
    unittest.main()
