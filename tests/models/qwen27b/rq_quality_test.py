#!/usr/bin/env python3
"""RQ R0 quality harness on synthetic logits: metrics, bootstrap and corpus.

Run with NumPy (eval image): python3 -I tests/models/qwen27b/rq_quality_test.py
GUFO_QWEN_CHAT_TEMPLATE=<chat_template.jinja> also checks the chat framing
against the model's own template.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/qwen27b/rq_quality"))
sys.path.insert(0, str(ROOT / "tools"))

import corpus  # noqa: E402
import logits_io  # noqa: E402
import metrics  # noqa: E402

VOCAB = 300


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    top = x.max(axis=1, keepdims=True)
    return x - top - np.log(np.exp(x - top).sum(axis=1, keepdims=True))


def synthetic(rng: np.random.Generator, windows: int = 4, length: int = 64,
              noise: float = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference and candidate logits on the same (window, position) grid."""
    n = windows * length
    ref = rng.normal(0, 3, size=(n, VOCAB)).astype(np.float32)
    cand = (ref + rng.normal(0, noise, size=ref.shape)).astype(np.float32)
    window = np.repeat(np.arange(windows), length).astype(np.int32)
    position = np.tile(np.arange(length), windows).astype(np.int32)
    # Targets drawn from the reference distribution, as text the model fits.
    p = np.exp(log_softmax(ref))
    target = np.array([rng.choice(VOCAB, p=row / row.sum()) for row in p], dtype=np.int32)
    return ref, cand, window, position, target


class Runs:
    """Writes a reference run and candidates collected against it."""

    def __init__(self, directory: Path, ref, window, position, target, k: int,
                 tokens_sha="t" * 64):
        self.directory, self.window, self.position, self.target, self.k = (
            directory, window, position, target, k)
        self.tokens_sha = tokens_sha
        records = logits_io.records_from_logits(ref, target, window, position, k)
        self.ref_meta = self.write("ref", records, None)
        self.ref_ids = records["ids"]
        self.ref = logits_io.load(directory / "ref")

    def write(self, name, records, reference, **extra):
        windows = [{"index": int(w), "category": "even" if w % 2 == 0 else "odd"}
                   for w in np.unique(records["window"])]
        return logits_io.write(self.directory / name, records, label=name,
                               windows_tokens_sha256=self.tokens_sha,
                               reference=reference, windows=windows, **extra)

    def candidate(self, name, logits, select=None):
        select = np.arange(len(self.window)) if select is None else select
        records = logits_io.records_from_logits(
            logits[select], self.target[select], self.window[select], self.position[select],
            self.k, reference_ids=self.ref_ids[select])
        self.write(name, records, {"bin_sha256": self.ref_meta["bin_sha256"]})
        return logits_io.load(self.directory / name)


class RecordTest(unittest.TestCase):
    def test_layout_matches_the_collector(self):
        for k in (1, 8, 64):
            self.assertEqual(logits_io.record_dtype(k).itemsize, 24 + 12 * k)

    def test_records_mirror_the_collector(self):
        logits = np.array([[1.0, 3.0, 3.0, -2.0, 0.5]], dtype=np.float32)
        r = logits_io.records_from_logits(logits, np.array([3]), np.array([0]), np.array([0]), 3)
        # Descending, ties to the lower id.
        self.assertEqual(r["ids"][0].tolist(), [1, 2, 0])
        self.assertEqual(r["logits"][0].tolist(), [3.0, 3.0, 1.0])
        self.assertAlmostEqual(r["logsumexp"][0],
                               float(np.log(np.exp(logits.astype(np.float64)).sum())), places=12)
        self.assertEqual(r["target_logit"][0], -2.0)

    def test_round_trip_validates_hash_and_sorts(self):
        rng = np.random.default_rng(1)
        ref, _, window, position, target = synthetic(rng, windows=2, length=8)
        records = logits_io.records_from_logits(ref, target, window, position, 4)[::-1]
        with tempfile.TemporaryDirectory() as d:
            logits_io.write(Path(d) / "run", records)
            run = logits_io.load(Path(d) / "run")
            self.assertTrue(np.all(np.diff(run.keys()) > 0))
            payload = bytearray((Path(d) / "run.bin").read_bytes())
            payload[30] ^= 1
            (Path(d) / "run.bin").write_bytes(bytes(payload))
            with self.assertRaisesRegex(ValueError, "sha256"):
                logits_io.load(Path(d) / "run")
            logits_io.write(Path(d) / "dup", np.concatenate([records[:2], records[:1]]))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                logits_io.load(Path(d) / "dup")


class MetricTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.rng = np.random.default_rng(20261029)
        self.ref, self.cand, self.window, self.position, self.target = synthetic(self.rng)

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_k_kl_equals_the_exact_kl(self):
        runs = Runs(self.dir, self.ref, self.window, self.position, self.target, VOCAB)
        paired = metrics.Paired(runs.ref, runs.candidate("c", self.cand))
        logp, logq = log_softmax(self.ref), log_softmax(self.cand)
        exact = (np.exp(logp) * (logp - logq)).sum(axis=1)
        np.testing.assert_allclose(paired.kl, exact, rtol=1e-5, atol=1e-9)
        np.testing.assert_allclose(paired.coverage, 1.0, atol=1e-9)

    def test_truncated_kl_is_the_coarsened_kl(self):
        k = 16
        runs = Runs(self.dir, self.ref, self.window, self.position, self.target, k)
        paired = metrics.Paired(runs.ref, runs.candidate("c", self.cand))
        logp, logq = log_softmax(self.ref), log_softmax(self.cand)
        p, q = np.exp(logp), np.exp(logq)
        exact = (p * (logp - logq)).sum(axis=1)
        top = np.argsort(-self.ref, axis=1, kind="stable")[:, :k]
        ps, qs = np.take_along_axis(p, top, 1), np.take_along_axis(q, top, 1)
        pr, qr = 1 - ps.sum(1), 1 - qs.sum(1)
        coarse = (ps * np.log(ps / qs)).sum(1) + pr * np.log(pr / qr)
        np.testing.assert_allclose(paired.kl, coarse, rtol=1e-4, atol=1e-7)
        self.assertTrue(np.all(paired.kl <= exact + 1e-7))
        self.assertTrue(np.all(paired.kl >= 0))
        np.testing.assert_allclose(paired.coverage, ps.sum(1), rtol=1e-6)

    def test_nll_and_top1(self):
        runs = Runs(self.dir, self.ref, self.window, self.position, self.target, 8)
        paired = metrics.Paired(runs.ref, runs.candidate("c", self.cand))
        rows = np.arange(len(self.target))
        expected = (-log_softmax(self.cand)[rows, self.target]
                    + log_softmax(self.ref)[rows, self.target])
        np.testing.assert_allclose(paired.values["delta_nll"], expected, atol=1e-5)
        top1 = (self.cand.argmax(1) == self.ref.argmax(1)).astype(float)
        np.testing.assert_array_equal(paired.top1, top1)

    def test_reference_against_itself_is_exact(self):
        runs = Runs(self.dir, self.ref, self.window, self.position, self.target, 8)
        report = metrics.score(runs.ref, runs.ref, 200, 16, 1)
        self.assertEqual(report["kl"]["mean"], 0.0)
        self.assertEqual(report["delta_nll"]["mean"], 0.0)
        self.assertEqual(report["top1"]["mean"], 1.0)
        self.assertEqual(report["kl"]["token_ci"], [0.0, 0.0])

    def test_noisier_candidate_scores_worse_and_compare_resolves_it(self):
        runs = Runs(self.dir, self.ref, self.window, self.position, self.target, 32)
        near = runs.candidate("near", self.cand)
        far_logits = (self.ref + self.rng.normal(0, 1.0, size=self.ref.shape)).astype(np.float32)
        far = runs.candidate("far", far_logits)
        a = metrics.score(runs.ref, near, 400, 16, 1)
        b = metrics.score(runs.ref, far, 400, 16, 1)
        self.assertLess(a["kl"]["mean"], b["kl"]["mean"])
        self.assertGreater(a["top1"]["mean"], b["top1"]["mean"])
        diff = metrics.compare(runs.ref, near, far, 400, 16, 1)
        self.assertTrue(diff["kl"]["token_resolved"])
        self.assertLess(diff["kl"]["token_ci"][1], 0)
        self.assertEqual(set(a["categories"]), {"even", "odd"})
        self.assertEqual(sum(c["n"] for c in a["categories"].values()), a["n"])

    def test_rejects_a_candidate_of_another_reference(self):
        runs = Runs(self.dir, self.ref, self.window, self.position, self.target, 8)
        records = logits_io.records_from_logits(self.cand, self.target, self.window,
                                                self.position, 8)
        runs.write("stray", records, {"bin_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "not collected against"):
            metrics.Paired(runs.ref, logits_io.load(self.dir / "stray"))
        runs.write("other", records, {"bin_sha256": runs.ref_meta["bin_sha256"]})
        meta = json.loads((self.dir / "other.json").read_text())
        meta["windows_tokens_sha256"] = "u" * 64
        (self.dir / "other.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "different corpus windows"):
            metrics.Paired(runs.ref, logits_io.load(self.dir / "other"))

    def test_subset_of_positions_is_scored_on_that_subset(self):
        runs = Runs(self.dir, self.ref, self.window, self.position, self.target, 8)
        select = np.flatnonzero(self.position < 10)
        paired = metrics.Paired(runs.ref, runs.candidate("part", self.cand, select))
        self.assertEqual(len(paired.kl), len(select))
        full = metrics.Paired(runs.ref, runs.candidate("full", self.cand))
        np.testing.assert_allclose(paired.kl, full.kl[select])


class BootstrapTest(unittest.TestCase):
    def test_token_interval_covers_the_mean_about_95_percent(self):
        rng = np.random.default_rng(7)
        hits, trials, n = 0, 200, 400
        window = np.zeros(n, dtype=np.int64)
        position = np.arange(n, dtype=np.int64)
        for t in range(trials):
            v = rng.exponential(1.0, size=n)  # skewed like per-token KL
            ci = metrics.bootstrap_mean({"x": v}, None, 400, np.random.default_rng(t))["x"]
            hits += ci[0] <= 1.0 <= ci[1]
        self.assertGreaterEqual(hits / trials, 0.88)
        self.assertLessEqual(hits / trials, 0.99)
        del window, position

    def test_block_intervals_widen_under_window_effects(self):
        rng = np.random.default_rng(11)
        windows, length = 8, 512
        window = np.repeat(np.arange(windows), length).astype(np.int64)
        position = np.tile(np.arange(length), windows).astype(np.int64)
        # A per-window offset plus slow AR(1) drift: positions are not i.i.d.
        values = np.repeat(rng.normal(0, 0.5, windows), length)
        drift = np.zeros(len(values))
        for i in range(1, len(values)):
            drift[i] = 0.98 * drift[i - 1] + rng.normal(0, 0.2)
        values = values + drift + rng.normal(0, 1, len(values))
        report = metrics.summarize({"x": values}, window, position, 1000, 64, 3)["x"]
        width = {k: report[k][1] - report[k][0] for k in ("token_ci", "block_ci", "window_ci")}
        self.assertLess(width["token_ci"], width["block_ci"])
        self.assertLess(width["block_ci"], width["window_ci"])
        for k in width:
            self.assertLessEqual(report[k][0], report["mean"])
            self.assertGreaterEqual(report[k][1], report["mean"])

    def test_units(self):
        window = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        position = np.array([0, 3, 4, 0, 5], dtype=np.int64)
        self.assertEqual(metrics.units(window, position, None).tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(metrics.units(window, position, 4).tolist(), [0, 0, 1, 2, 3])

    def test_cli_writes_a_report(self):
        rng = np.random.default_rng(5)
        ref, cand, window, position, target = synthetic(rng, windows=2, length=32)
        with tempfile.TemporaryDirectory() as d:
            runs = Runs(Path(d), ref, window, position, target, 8)
            runs.candidate("c", cand)
            out = Path(d) / "report.json"
            self.assertEqual(metrics.main(["score", "--reference", str(Path(d) / "ref"),
                                           "--bootstrap", "200", "--out", str(out),
                                           str(Path(d) / "c")]), 0)
            report = json.loads(out.read_text())["reports"][0]
            self.assertEqual(report["n"], 64)
            self.assertIn("window_ci", report["kl"])


class CorpusTest(unittest.TestCase):
    def test_prose_strip(self):
        text = ("# T\n\nkeep this\n\n```cpp\nint x;\n```\n| a | b |\n|---|---|\n"
                "<!-- hidden -->\n\n\n\nand this\n")
        self.assertEqual(corpus.strip_prose(text), "# T\n\nkeep this\n\nand this\n")

    def test_sections(self):
        title, parts = corpus.sections("# Doc\n\nintro\n\n## A\n\nalpha\n\n### B\n\n## C\n\ngamma\n")
        self.assertEqual(title, "Doc")
        self.assertEqual(parts, [("A", "alpha"), ("C", "gamma")])

    def test_language(self):
        self.assertEqual(corpus.language("한국어 " * 30), "ko")
        self.assertEqual(corpus.language("English only"), "en")

    def test_chat_framing(self):
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"}]
        expected = ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n\n</think>\n\nA<|im_end|>\n")
        self.assertEqual(corpus.render_chat(messages), expected)

    @unittest.skipUnless(os.environ.get("GUFO_QWEN_CHAT_TEMPLATE"),
                         "GUFO_QWEN_CHAT_TEMPLATE not set")
    def test_chat_framing_matches_the_model_template(self):
        import jinja2

        def fail(message):
            raise ValueError(message)

        env = jinja2.Environment()
        env.globals["raise_exception"] = fail
        template = env.from_string(Path(os.environ["GUFO_QWEN_CHAT_TEMPLATE"]).read_text())
        messages = [{"role": "system", "content": corpus.SYSTEM["ko"]},
                    {"role": "user", "content": "첫 질문"},
                    {"role": "assistant", "content": "첫 답\n\n둘째 문단"},
                    {"role": "user", "content": "Second question?"},
                    {"role": "assistant", "content": "Second answer."}]
        rendered = template.render(messages=messages, enable_thinking=False,
                                   add_generation_prompt=False)
        self.assertEqual(corpus.render_chat(messages), rendered)


if __name__ == "__main__":
    unittest.main()
