#!/usr/bin/env python3
"""Paired quality metrics of qwen27b_rq_logits runs against a reference run.

Per scored position (window w, position p, predicting corpus token p + 1):

- paired NLL difference: d = NLL_cand - NLL_ref, NLL = logsumexp - logit[target]
  (natural log, nats per token; both from the full vocabulary).
- truncated KL(ref || cand): S = the reference's top-k ids. With
  p_i = softmax_ref(i), q_i = softmax_cand(i) for i in S (both normalized over
  the full vocabulary) and the rest bucket p_rest = 1 - sum_S p,
  q_rest = 1 - sum_S q, KL_k = sum_S p_i log(p_i / q_i) + p_rest log(p_rest / q_rest).
  It is the exact KL of the two distributions coarsened to k + 1 outcomes,
  so 0 <= KL_k <= the full-vocabulary KL; the gap is bounded by the
  reference mass outside S, reported as `coverage`.
- top-1 agreement: the candidate's full-vocabulary argmax equals the
  reference's (ties to the lower id in both).

Each mean has three percentile-bootstrap 95% intervals: token level
(positions resampled i.i.d.), block level (contiguous `--block-tokens`
positions of one window resampled as units) and window level (whole windows
resampled). Block and window intervals use the ratio estimator
sum(values) / count over the resampled units.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logits_io  # noqa: E402

METRICS = ("delta_nll", "kl", "top1")
DEFAULT_BOOTSTRAP = 2000
DEFAULT_SEED = 20261029
DEFAULT_BLOCK_TOKENS = 256


class Paired:
    """Per-position metric values of one candidate against the reference."""

    def __init__(self, reference: logits_io.Run, candidate: logits_io.Run):
        if reference.top_k != candidate.top_k:
            raise ValueError("reference and candidate top-k differ")
        if (reference.meta.get("windows_tokens_sha256")
                != candidate.meta.get("windows_tokens_sha256")):
            raise ValueError("reference and candidate use different corpus windows")
        is_self = reference.meta["bin_sha256"] == candidate.meta["bin_sha256"]
        ref_meta = candidate.meta.get("reference")
        if not is_self and (ref_meta is None
                            or ref_meta.get("bin_sha256") != reference.meta["bin_sha256"]):
            raise ValueError(
                f"{candidate.prefix} was not collected against {reference.prefix} "
                "(its ref_logits belong to another reference)")
        rkeys, ckeys = reference.keys(), candidate.keys()
        index = np.searchsorted(rkeys, ckeys)
        if np.any(index >= len(rkeys)) or np.any(rkeys[np.minimum(index, len(rkeys) - 1)] != ckeys):
            raise ValueError("candidate scores positions the reference does not have")
        r = reference.records[index]
        c = candidate.records
        if np.any(r["target"] != c["target"]):
            raise ValueError("reference and candidate targets differ")
        self.window = c["window"].astype(np.int64)
        self.position = c["position"].astype(np.int64)
        r_lse, c_lse = r["logsumexp"], c["logsumexp"]
        self.nll_ref = r_lse - r["target_logit"].astype(np.float64)
        self.nll_cand = c_lse - c["target_logit"].astype(np.float64)
        logp = r["logits"].astype(np.float64) - r_lse[:, None]
        logq = c["ref_logits"].astype(np.float64) - c_lse[:, None]
        self.kl, self.coverage = truncated_kl(logp, logq)
        self.top1 = (c["ids"][:, 0] == r["ids"][:, 0]).astype(np.float64)
        self.values = {"delta_nll": self.nll_cand - self.nll_ref, "kl": self.kl,
                       "top1": self.top1}


def truncated_kl(logp: np.ndarray, logq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """KL over the k reference ids plus one rest bucket; also the ref top-k mass."""
    p = np.exp(logp)
    q = np.exp(logq)
    covered = p.sum(axis=1)
    p_rest = np.clip(1.0 - covered, 0.0, 1.0)
    q_rest = np.clip(1.0 - q.sum(axis=1), 1e-300, 1.0)
    head = (p * (logp - logq)).sum(axis=1)
    rest = np.where(p_rest > 0, p_rest * (np.log(np.maximum(p_rest, 1e-300)) - np.log(q_rest)), 0.0)
    # Coarsening keeps KL >= 0; tiny negative sums are float rounding.
    return np.maximum(head + rest, 0.0), covered


def units(window: np.ndarray, position: np.ndarray, block_tokens: int | None) -> np.ndarray:
    """Dense unit index per position: a window, or a block of one window."""
    key = window << 32 if block_tokens is None else (window << 32) | (position // block_tokens)
    _, inverse = np.unique(key, return_inverse=True)
    return inverse


def bootstrap_mean(values: dict[str, np.ndarray], unit: np.ndarray | None,
                   resamples: int, rng: np.random.Generator) -> dict[str, list[float]]:
    """Percentile 95% intervals of each mean; `unit` None resamples positions."""
    names = list(values)
    stacked = np.stack([values[n] for n in names])  # [metrics, n]
    n = stacked.shape[1]
    estimates = np.empty((resamples, len(names)))
    if unit is None:
        chunk = max(1, (1 << 22) // max(n, 1))
        for start in range(0, resamples, chunk):
            count = min(chunk, resamples - start)
            draw = rng.integers(0, n, size=(count, n))
            estimates[start:start + count] = stacked[:, draw].mean(axis=2).T
    else:
        units_count = int(unit.max()) + 1
        sums = np.stack([np.bincount(unit, weights=row, minlength=units_count) for row in stacked])
        counts = np.bincount(unit, minlength=units_count).astype(np.float64)
        draw = rng.integers(0, units_count, size=(resamples, units_count))
        estimates[:] = (sums[:, draw].sum(axis=2) / counts[draw].sum(axis=1)).T
    low, high = np.percentile(estimates, [2.5, 97.5], axis=0)
    return {name: [float(low[i]), float(high[i])] for i, name in enumerate(names)}


def summarize(values: dict[str, np.ndarray], window: np.ndarray, position: np.ndarray,
              resamples: int, block_tokens: int, seed: int) -> dict:
    out = {"n": int(len(window)), "windows": int(len(np.unique(window)))}
    intervals = {
        "token_ci": bootstrap_mean(values, None, resamples, np.random.default_rng(seed)),
        "block_ci": bootstrap_mean(values, units(window, position, block_tokens), resamples,
                                   np.random.default_rng(seed + 1)),
        "window_ci": bootstrap_mean(values, units(window, position, None), resamples,
                                    np.random.default_rng(seed + 2)),
    }
    for name, v in values.items():
        out[name] = {"mean": float(v.mean()),
                     **{kind: ci[name] for kind, ci in intervals.items()}}
    return out


def categories_of(run: logits_io.Run) -> dict[int, str]:
    return {int(w["index"]): w.get("category", "") for w in run.meta.get("windows", [])}


def score(reference: logits_io.Run, candidate: logits_io.Run, resamples: int,
          block_tokens: int, seed: int) -> dict:
    paired = Paired(reference, candidate)
    report = {
        "label": candidate.meta.get("label", ""),
        "model": candidate.meta.get("model", ""),
        "path": candidate.meta.get("path", ""),
        "reference_label": reference.meta.get("label", ""),
        "reference_path": reference.meta.get("path", ""),
        "reference_bin_sha256": reference.meta["bin_sha256"],
        "candidate_bin_sha256": candidate.meta["bin_sha256"],
        "top_k": candidate.top_k, "bootstrap": resamples, "block_tokens": block_tokens,
        "seed": seed,
        "nll_ref": float(paired.nll_ref.mean()), "nll_cand": float(paired.nll_cand.mean()),
        "coverage": {"mean": float(paired.coverage.mean()),
                     "p01": float(np.percentile(paired.coverage, 1)),
                     "min": float(paired.coverage.min())},
        **summarize(paired.values, paired.window, paired.position, resamples, block_tokens, seed),
    }
    names = categories_of(candidate)
    by_category = {}
    labels = np.array([names.get(int(w), "") for w in paired.window])
    for category in sorted(set(labels)):
        mask = labels == category
        values = {k: v[mask] for k, v in paired.values.items()}
        entry = summarize(values, paired.window[mask], paired.position[mask], resamples,
                          block_tokens, seed)
        entry["nll_ref"] = float(paired.nll_ref[mask].mean())
        by_category[category] = entry
    report["categories"] = by_category
    return report


def compare(reference: logits_io.Run, first: logits_io.Run, second: logits_io.Run,
            resamples: int, block_tokens: int, seed: int) -> dict:
    """Paired difference first - second of each metric on the same positions."""
    a, b = Paired(reference, first), Paired(reference, second)
    if not (np.array_equal(a.window, b.window) and np.array_equal(a.position, b.position)):
        raise ValueError("the two candidates score different positions")
    values = {name: a.values[name] - b.values[name] for name in METRICS}
    report = {"first": first.meta.get("label", ""), "second": second.meta.get("label", ""),
              **summarize(values, a.window, a.position, resamples, block_tokens, seed)}
    for name in METRICS:
        entry = report[name]
        for kind in ("token_ci", "block_ci", "window_ci"):
            low, high = entry[kind]
            entry[kind.replace("_ci", "_resolved")] = bool(low > 0 or high < 0)
    return report


def format_score(report: dict) -> str:
    lines = [f"{report['label'] or report['model']} [{report['path']}] vs "
             f"{report['reference_label']} [{report['reference_path']}]: n={report['n']} "
             f"windows={report['windows']} top-k mass mean={report['coverage']['mean']:.4f} "
             f"min={report['coverage']['min']:.4f}"]
    for name in METRICS:
        m = report[name]
        lines.append(f"  {name:9s} {m['mean']:+.6f}  token {fmt(m['token_ci'])}  "
                     f"block {fmt(m['block_ci'])}  window {fmt(m['window_ci'])}")
    return "\n".join(lines)


def fmt(ci: list[float]) -> str:
    return f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("score", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--reference", required=True, help="reference run prefix")
        p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
        p.add_argument("--block-tokens", type=int, default=DEFAULT_BLOCK_TOKENS)
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)
        p.add_argument("--out", type=Path, help="write the JSON report")
    sub.choices["score"].add_argument("candidates", nargs="+", help="candidate run prefixes")
    sub.choices["compare"].add_argument("first")
    sub.choices["compare"].add_argument("second")
    args = parser.parse_args(argv)
    if args.bootstrap < 100 or args.block_tokens < 1:
        parser.error("--bootstrap must be >= 100 and --block-tokens >= 1")
    reference = logits_io.load(args.reference)
    if args.command == "score":
        reports = [score(reference, logits_io.load(c), args.bootstrap, args.block_tokens,
                         args.seed) for c in args.candidates]
        for report in reports:
            print(format_score(report))
        result = {"schema": "gufo.rq-quality-score.v1", "reports": reports}
    else:
        result = compare(reference, logits_io.load(args.first), logits_io.load(args.second),
                         args.bootstrap, args.block_tokens, args.seed)
        result["schema"] = "gufo.rq-quality-compare.v1"
        for name in METRICS:
            m = result[name]
            print(f"{result['first']} - {result['second']} {name}: {m['mean']:+.6f} "
                  f"token {fmt(m['token_ci'])} block {fmt(m['block_ci'])} "
                  f"window {fmt(m['window_ci'])}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
