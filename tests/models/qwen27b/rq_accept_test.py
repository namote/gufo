#!/usr/bin/env python3
"""RQ DFlash2 text acceptance analysis (tools/qwen27b/rq_quality/accept.py) on synthetic reports.

Run with NumPy (eval image): python3 -I tests/models/qwen27b/rq_accept_test.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/qwen27b/rq_quality"))

import accept  # noqa: E402


def report(cases: dict[str, tuple[int, int, int, float]], repetitions: int = 3,
           jitter: float = 0.0, seed: int = 0) -> dict:
    """A serving_bench corpus report: case -> (drafted, accepted, completion tokens, decode ms)."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        for case, (drafted, accepted, tokens, ms) in cases.items():
            samples.append({"case_id": case, "draft_tokens": drafted, "draft_accepted_tokens": accepted,
                            "completion_tokens": tokens, "decode_ms": ms * (1 + jitter * rng.normal()),
                            "completion_sha256": f"{case}-{accepted}"})
    return {"results": {"c1": {"samples": samples}}}


def synthetic(n: int, mean: float, rng: np.random.Generator, tok_s: float) -> dict:
    cases = {}
    for i in range(n):
        drafted = int(rng.integers(150, 400))
        rate = float(np.clip(rng.normal(mean, 0.15), 0.05, 0.95))
        cases[f"p{i:02d}"] = (drafted, int(round(rate * drafted)), 256, 256 / tok_s * 1000)
    return cases


class PromptTest(unittest.TestCase):
    def test_aggregates_and_ratio(self):
        r = report({"a": (100, 50, 200, 1000.0), "b": (300, 270, 200, 500.0)}, repetitions=2)
        cases = accept.prompts(r)
        self.assertEqual(cases["a"]["repetitions"], 2)
        self.assertEqual(cases["a"]["completion_tokens"], 400)
        s = accept.summary(r, resamples=200)
        self.assertAlmostEqual(s["acceptance"]["value"], 320 / 400)
        self.assertAlmostEqual(s["accepted_per_token"]["value"], 320 / 400)
        self.assertAlmostEqual(s["decode_tok_s"]["value"], 800 / 3000 * 1000)

    def test_rejects_nondeterministic_repetitions(self):
        r = report({"a": (100, 50, 200, 1000.0)}, repetitions=2)
        r["results"]["c1"]["samples"][1]["draft_accepted_tokens"] = 51
        with self.assertRaisesRegex(ValueError, "disagree"):
            accept.prompts(r)

    def test_rejects_missing_case_id_and_decode_time(self):
        r = report({"a": (100, 50, 200, 1000.0)}, repetitions=1)
        r["results"]["c1"]["samples"][0]["decode_ms"] = None
        with self.assertRaisesRegex(ValueError, "decode_ms"):
            accept.prompts(r)
        r["results"]["c1"]["samples"][0]["case_id"] = None
        with self.assertRaisesRegex(ValueError, "case_id"):
            accept.prompts(r)


class BootstrapTest(unittest.TestCase):
    def test_interval_covers_true_acceptance(self):
        hits = 0
        for t in range(100):
            rng = np.random.default_rng(t)
            r = report(synthetic(64, 0.6, rng, 30.0), repetitions=1)
            s = accept.summary(r, resamples=400, seed=t)
            lo, hi = s["acceptance"]["ci"]
            hits += lo <= 0.6 <= hi
        self.assertGreaterEqual(hits, 85)
        width = hi - lo
        self.assertLess(width, 0.1)  # 64 prompts, per-prompt sd 0.15

    def test_compare_resolves_a_real_drop_and_not_noise(self):
        rng = np.random.default_rng(3)
        base = synthetic(64, 0.6, rng, 30.0)
        same = {k: v for k, v in base.items()}
        drop = {k: (d, max(0, a - int(0.05 * d)), t, ms * 1.04) for k, (d, a, t, ms) in base.items()}
        c = accept.compare(report(base, jitter=0.01, seed=1), report(drop, jitter=0.01, seed=2), resamples=500)
        self.assertTrue(c["acceptance_difference"]["resolved"])
        self.assertAlmostEqual(c["acceptance_difference"]["value"], 0.05, delta=0.01)
        self.assertTrue(c["decode_tok_s_ratio"]["resolved"])
        self.assertGreater(c["decode_tok_s_ratio"]["value"], 1.0)
        c = accept.compare(report(base, jitter=0.01, seed=1), report(same, jitter=0.01, seed=4), resamples=500)
        self.assertEqual(c["acceptance_difference"]["value"], 0.0)
        self.assertFalse(c["acceptance_difference"]["resolved"])
        self.assertEqual(c["identical_completions"], 64)

    def test_speedup_over_ar(self):
        rng = np.random.default_rng(5)
        spec = synthetic(32, 0.6, rng, 30.0)
        ar = {k: (0, 0, t, t / 12.0 * 1000) for k, (d, a, t, ms) in spec.items()}
        s = accept.summary(report(spec), report(ar), resamples=300)
        self.assertAlmostEqual(s["speedup_over_ar"]["value"], 30.0 / 12.0, places=6)
        self.assertEqual(s["prompts_with_drafts"], 32)

    def test_compare_rejects_different_prompts(self):
        with self.assertRaisesRegex(ValueError, "differ in prompts"):
            accept.compare(report({"a": (1, 1, 1, 1.0)}), report({"b": (1, 1, 1, 1.0)}), resamples=100)


class SuiteTest(unittest.TestCase):
    def test_prompt_file(self):
        doc = json.loads((ROOT / "tools/qwen27b/rq_quality/accept_prompts.json").read_text())
        self.assertEqual(doc["schema"], "gufo.speculative-corpus.v1")
        ids = [p["id"] for p in doc["prompts"]]
        self.assertEqual(len(ids), 64)
        self.assertEqual(len(set(ids)), 64)
        self.assertEqual(sum(p["source"] == "W2-authored" for p in doc["prompts"]), 38)
        self.assertTrue(all(p["text"] and p["category"] for p in doc["prompts"]))
        self.assertEqual(set(doc["manifest"]["overlap"]["against"]), {"r0_a01", "r3_c01", "r3_c02"})

    def test_overlap(self):
        # A shared run of at least width + step characters always contains one whole fragment.
        shared = "the quick brown fox jumps over the lazy dog while the cat sleeps"
        doc = {"prompts": [{"id": "x", "text": "a" * 10 + shared + "b" * 30},
                           {"id": "y", "text": "short"}]}
        result = accept.check_overlap(doc, {"t": "zz " + shared + " zz", "u": "nothing"})
        self.assertIn("x", result["against"]["t"]["hits"])
        self.assertEqual(result["against"]["u"]["hits"], {})

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.json", Path(d) / "b.json"
            a.write_text(json.dumps(report({"p": (10, 5, 20, 100.0), "q": (10, 8, 20, 90.0)})))
            b.write_text(json.dumps(report({"p": (10, 4, 20, 100.0), "q": (10, 8, 20, 95.0)})))
            out = Path(d) / "c.json"
            self.assertEqual(accept.main(["compare", str(a), str(b), "--bootstrap", "200", "--out", str(out)]), 0)
            self.assertAlmostEqual(json.loads(out.read_text())["acceptance_difference"]["value"], 0.05)


if __name__ == "__main__":
    unittest.main()
