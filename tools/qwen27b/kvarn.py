#!/usr/bin/env python3
"""KVarN K0 study helpers (plan/gufo-upstream-port/KVARN.md).

check BIN       Compares the C++ fake quantization (BIN is the
                qwen27b_kvarn_fake_quant executable) with this independent
                NumPy implementation of the huawei-csl/KVarN store path
                (Apache-2.0) on deterministic FP16 rows with outlier channels.
                Exits 77 without NumPy.
summarize FILE  Prints one row per configuration from qwen27b_kvarn_k0 JSON
                lines: KL distribution, top-1 agreement, free-run divergence
                and the K/V reconstruction error. No step is averaged away.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

STD_MIN, STD_MAX, LOG_MIN, LOG_MAX = 1e-3, 1e3, -0.3, 10.0


def parse(name):
    """(method, key_bits, value_bits, group) of a configuration name."""
    if name == "fp16":
        return "fp16", 16, 16, 128
    method, rest = name.split("_", 1)
    bits, group = rest.split("_g")
    key, value = bits[1:].split("v")
    return method, int(key), int(value), int(group)


def hadamard(np, n):
    h = np.ones((1, 1), np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(np.float32(n))).astype(np.float32)


def balance(np, tiles, iterations):
    """Batched log-domain balancing of [N, R, C] tiles (sinkhorn.py)."""
    def imbalance(x):
        columns, rows = x.std(axis=1, ddof=1), x.std(axis=2, ddof=1)
        return (columns.max(-1) / np.maximum(columns.min(-1), 1e-8) +
                rows.max(-1) / np.maximum(rows.min(-1), 1e-8))

    n, r, c = tiles.shape
    log_column = np.zeros((n, 1, c), np.float32)
    log_row = np.zeros((n, r, 1), np.float32)
    best = imbalance(tiles)
    column_best = np.ones_like(log_column)
    row_best = np.ones_like(log_row)
    current = tiles
    for _ in range(iterations):
        s = np.clip(current.std(axis=1, ddof=1, keepdims=True), STD_MIN, STD_MAX)
        log_column = np.clip(log_column + np.log(s), LOG_MIN, LOG_MAX)
        current = tiles / np.exp(log_column) / np.exp(log_row)
        s = np.clip(current.std(axis=2, ddof=1, keepdims=True), STD_MIN, STD_MAX)
        log_row = np.clip(log_row + np.log(s), LOG_MIN, LOG_MAX)
        current = tiles / np.exp(log_column) / np.exp(log_row)
        value = imbalance(current)
        better = value <= best
        column_best = np.where(better[:, None, None], np.exp(log_column), column_best)
        row_best = np.where(better[:, None, None], np.exp(log_row), row_best)
        best = np.where(better, value, best)
    return tiles / column_best / row_best, column_best, row_best


def quantize_tiles(np, tiles, bits, balanced, iterations):
    """Dequantized per-row asymmetric RTN of [N, R, C] tiles (kvarn_store.py)."""
    if balanced:
        tiles, column, row = balance(np, tiles, iterations)
    else:
        column = np.ones((tiles.shape[0], 1, tiles.shape[2]), np.float32)
        row = np.ones((tiles.shape[0], tiles.shape[1], 1), np.float32)
    qmax = np.float32((1 << bits) - 1)
    low, high = tiles.min(axis=2, keepdims=True), tiles.max(axis=2, keepdims=True)
    scale = np.maximum((high - low) / qmax, np.float32(1e-10))
    q = np.clip(np.rint((tiles - low) / scale), 0, qmax)
    half = lambda x: x.astype(np.float16).astype(np.float32)
    return (q * half(row * scale) + half(row * low)) * half(column)


def fake_quantize(np, rows, key, name, first, end, iterations=8):
    """rows: FP16 [token, head, dim]; quantizes groups [first, end)."""
    method, key_bits, value_bits, group = parse(name)
    if method == "fp16" or end <= first:
        return rows.copy()
    _, heads, dim = rows.shape
    rotate = method != "rtn"
    h = hadamard(np, dim)
    half = lambda x: x.astype(np.float16).astype(np.float32)
    x = rows[first * group:end * group].astype(np.float32)
    x = x.reshape(end - first, group, heads, dim).transpose(0, 2, 1, 3)
    if rotate:
        x = half(x @ h)
    tiles = x.reshape(-1, group, dim)
    if key:
        tiles = tiles.transpose(0, 2, 1)
    out = quantize_tiles(np, np.ascontiguousarray(tiles), key_bits if key else value_bits,
                         method == "kvarn", iterations)
    if key:
        out = out.transpose(0, 2, 1)
    out = half(out).reshape(end - first, heads, group, dim)
    if rotate:
        out = out @ h
    result = rows.copy()
    result[first * group:end * group] = (
        out.astype(np.float16).transpose(0, 2, 1, 3).reshape(-1, heads, dim))
    return result


def check(binary):
    try:
        import numpy as np
    except ImportError:
        print("SKIP: NumPy is unavailable")
        return 77
    rng = np.random.default_rng(20261026)
    heads, dim, tokens = 2, 256, 3 * 128 + 37
    channel = np.exp(rng.normal(0, 1, (heads, dim))).astype(np.float32)
    channel[:, rng.choice(dim, 6, replace=False)] *= 12
    data = {
        "key": (rng.normal(0, 1, (tokens, heads, dim)) * channel +
                rng.normal(0, 2, (heads, dim))).astype(np.float16),
        "value": (rng.normal(0, 0.5, (tokens, heads, dim)) *
                  np.exp(rng.normal(0, 0.3, (tokens, 1, 1)))).astype(np.float16),
    }
    failures = 0
    names = ["fp16"] + [f"{m}_k{k}v{v}_g{g}" for m in ("rtn", "hrtn", "kvarn")
                        for k, v in ((4, 2), (4, 4), (8, 4), (2, 2)) for g in (128, 64)]
    with tempfile.TemporaryDirectory() as directory:
        for kind, rows in data.items():
            source = Path(directory) / f"{kind}.f16"
            rows.tofile(source)
            for name in names:
                group = parse(name)[3]
                first, end = 1, tokens // group
                target = Path(directory) / f"{kind}-{name}.f16"
                subprocess.run([binary, name, kind, str(heads), str(dim), str(first),
                                str(end), str(source), str(target)],
                               check=True, capture_output=True)
                ours = np.fromfile(target, np.float16).reshape(rows.shape)
                reference = fake_quantize(np, rows, kind == "key", name, first, end)
                quantized = slice(first * group, end * group)
                kept = np.r_[0:first * group, end * group:tokens]
                ref32 = reference[quantized].astype(np.float32)
                original = rows[quantized].astype(np.float32)
                quant_error = np.linalg.norm(ref32 - original)
                difference = np.linalg.norm(ours[quantized].astype(np.float32) - ref32)
                exact = float(np.mean(ours[quantized] == reference[quantized]))
                untouched = bool(np.array_equal(ours[kept], rows[kept]))
                relative = difference / quant_error if quant_error else difference
                # Balancing accumulates in a different order, so a stored FP16
                # scale can land one ulp apart and move a whole tile slightly.
                ok = untouched and (difference == 0 if name == "fp16" else relative < 0.05)
                failures += not ok
                print(f"{'PASS' if ok else 'FAIL'} {kind} {name}: exact={exact:.4f} "
                      f"implementation/quantization error={relative:.4f} "
                      f"relative error={quant_error / np.linalg.norm(original):.4f}")
    return 1 if failures else 0


def summarize(paths):
    rows = [json.loads(line) for path in paths for line in Path(path).read_text().splitlines()
            if line.strip()]
    print("|config|prompt|chunk|K err|V err|KL median|KL p95|KL max|top-1|first diff|"
          "free identical|free diverges at|")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        kl = r["kl"]
        print(f"|{r['config']}|{r['prompt_tokens']}|{r['prefill_chunk']}|"
              f"{r['key_relative_error']:.4f}|{r['value_relative_error']:.4f}|"
              f"{kl.get('median', 0):.3g}|{kl.get('p95', 0):.3g}|{kl.get('max', 0):.3g}|"
              f"{r['top1_agreement']:.4f}|{r['first_top1_disagreement']}/{r['steps']}|"
              f"{r['free_identical']}|{r['free_divergence']}/{r['free_tokens']}|")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check").add_argument("binary")
    sub.add_parser("summarize").add_argument("files", nargs="+")
    args = parser.parse_args()
    return check(args.binary) if args.command == "check" else summarize(args.files)


if __name__ == "__main__":
    sys.exit(main())
