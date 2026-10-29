#!/usr/bin/env python3
"""Check a rotated GGUF against its source on real tensors (CPU, NumPy).

For sampled tensors (every rotated group; first, middle and last layer of
each) this verifies:

- the prism.hadamard sign vector for the tensor's input width equals the
  converter's derivation: all +1 for a `--signs none` file
  (gufo.rotation.source_signs = "none"), otherwise the seeded draw (--seed,
  or the seed recorded in source_signs);
- the stored weights are bit-identical to BF16-RNE(W·D·H) recomputed in FP32
  (rows sampled with --rows; 0 = all rows);
- W'·(H·D·x) ≈ W·x for random x and for x with a few large outlier channels,
  reported as the relative error distribution of the outputs.

Example:
  check_rotated.py --groups qwen27b --seed 1 --source BF16-00001-of-00002.gguf \\
      --rotated rot-BF16.gguf --json result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from rotquant.gguf_io import as_float32, open_gguf  # noqa: E402
from rotquant.hadamard import f32_to_bf16_bits, rotate_last_axis, signs  # noqa: E402
from rotquant.plan import SIGNS_KEY, check_rotation, file_signs, load_groups, sign_key  # noqa: E402
from rotquant.prism import read_prism  # noqa: E402

BF16_HALF_ULP = 2.0 ** -8  # BF16 keeps 8 significant bits


def pick_samples(model, groups) -> list:
    """(group, instance, tensor name): first, middle and last layer per group,
    cycling through the group's member tensors."""
    folded = set(read_prism(model).weight_names)
    by_group = {}
    for tensor in model.tensors:
        group, instance = groups.instance(tensor.name)
        if group is not None and tensor.name in folded:
            by_group.setdefault(group, {}).setdefault(instance, []).append(tensor.name)
    samples = []
    for group, instances in by_group.items():
        keys = sorted(instances, key=lambda k: int(k.split(":")[1]) if ":" in k else 0)
        chosen = sorted({0, len(keys) // 2, len(keys) - 1})
        for position, index in enumerate(chosen):
            members = instances[keys[index]]
            samples.append((group, keys[index], members[(index + position) % len(members)]))
    return samples


def probe_inputs(k: int, rng, columns: int) -> dict:
    random = rng.standard_normal((columns, k)).astype(np.float32)
    outlier = rng.standard_normal((columns, k)).astype(np.float32)
    for row in outlier:
        channels = rng.choice(k, size=8, replace=False)
        row[channels] *= 100.0
    return {"random": random, "outlier": outlier}


def rel_errors(reference: np.ndarray, got: np.ndarray) -> dict:
    """Per-column relative L2 error and per-element error over the column RMS."""
    diff = got.astype(np.float64) - reference
    norm = np.linalg.norm(reference, axis=0)
    rms = norm / np.sqrt(reference.shape[0])
    per_element = np.abs(diff) / rms
    return {"rel_l2": (np.linalg.norm(diff, axis=0) / norm).tolist(),
            "elem_p50": float(np.percentile(per_element, 50)),
            "elem_p99": float(np.percentile(per_element, 99)),
            "elem_max": float(per_element.max())}


def expected_signs(seed, k: int) -> np.ndarray:
    """None means D = I (--signs none); otherwise the seeded draw."""
    return np.ones(k, dtype=np.float32) if seed is None else signs(seed, sign_key(k), k)


def file_seed(rot, seed_arg):
    """Seed to verify against, or None for a sign-free file."""
    label = rot.kv.get(SIGNS_KEY)
    if label == "none":
        if seed_arg is not None:
            raise ValueError(f"the file has {SIGNS_KEY} = none; --seed does not apply")
        return None
    if seed_arg is not None:
        return seed_arg
    if isinstance(label, str) and label.startswith("random seed="):
        return int(label.split("=", 1)[1])
    raise ValueError(f"the file has no {SIGNS_KEY}; pass --seed")


def check_tensor(src, rot, sign_table, seed, group, instance, name, rng, columns, row_sample):
    started = time.time()
    tensor, rotated = src.by_name[name], rot.by_name[name]
    k = tensor.ne[0]
    sign = sign_table[k]
    derived = expected_signs(seed, k)
    w = as_float32(src, tensor)
    w_rot = as_float32(rot, rotated)
    n = w.shape[0]
    rows = np.arange(n) if row_sample == 0 or row_sample >= n else np.sort(rng.choice(n, row_sample, replace=False))
    recomputed = f32_to_bf16_bits(rotate_last_axis(w[rows], sign))
    stored = np.frombuffer(rot.tensor_bytes(rotated), dtype="<u2").reshape(n, k)[rows]
    mismatched = int(np.count_nonzero(recomputed != stored))
    result = {"group": group, "instance": instance, "tensor": name, "shape": [n, k],
              "signs_match_seed": bool(np.array_equal(sign, derived)),
              "bit_identical_rows": int(rows.size), "bit_mismatches": mismatched}
    for label, x in probe_inputs(k, rng, columns).items():
        reference = w.astype(np.float64) @ x.T.astype(np.float64) if n * k <= 300_000_000 \
            else (w @ x.T).astype(np.float64)
        x_rot = rotate_last_axis(x, sign)
        got = w_rot @ x_rot.T
        result[label] = rel_errors(reference, got)
    # What plain BF16 rounding of the rotated FP32 weights costs (the bound
    # the stored file should meet): same probe, FP32 rotated weights.
    exact = rotate_last_axis(w[rows], sign)
    rounding = np.abs(w_rot[rows] - exact) / np.maximum(np.abs(exact), np.finfo(np.float32).tiny)
    result["weight_rel_rounding_p99"] = float(np.percentile(rounding[exact != 0], 99))
    result["weight_rel_rounding_max"] = float(rounding[exact != 0].max())
    result["seconds"] = round(time.time() - started, 1)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--seed", type=int, help="sign seed (default: the seed recorded in the file)")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rotated", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4, help="probe vectors per kind")
    parser.add_argument("--rows", type=int, default=2048, help="rows for the bit-identity check (0 = all)")
    parser.add_argument("--tensors", help="comma-separated tensor names instead of the default sample")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    groups = load_groups(args.groups)
    rng = np.random.default_rng(20261029)
    with open_gguf(args.source) as src, open_gguf(args.rotated) as rot:
        rotated_groups = check_rotation(rot, groups)
        if not rotated_groups:
            parser.error(f"{args.rotated} has no prism.hadamard folding")
        bound = groups.for_model(rot.kv)
        sign_table = file_signs(rot)
        try:
            seed = file_seed(rot, args.seed)
        except ValueError as error:
            parser.error(str(error))
        folded = set(read_prism(rot).weight_names)
        results = []
        if args.tensors:
            samples = [(*bound.instance(name), name) for name in args.tensors.split(",")]
            unrotated = [name for group, _, name in samples if group is None or name not in folded]
            if unrotated:
                parser.error(f"not rotated in this file: {', '.join(unrotated)}")
        else:
            samples = pick_samples(rot, bound)
        for group, instance, name in samples:
            result = check_tensor(src, rot, sign_table, seed, group, instance, name, rng,
                                  args.columns, args.rows)
            results.append(result)
            print(f"{name:<32} {instance:<15} bits {result['bit_mismatches']}/{result['bit_identical_rows']} rows "
                  f"signs={'ok' if result['signs_match_seed'] else 'DIFF'} "
                  f"random rel_l2 max {max(result['random']['rel_l2']):.2e} "
                  f"outlier rel_l2 max {max(result['outlier']['rel_l2']):.2e} "
                  f"elem p99 {result['random']['elem_p99']:.2e} ({result['seconds']} s)", flush=True)
    summary = {"rotated_groups": list(rotated_groups), "half_ulp": BF16_HALF_ULP, "tensors": results}
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    bad = [r["tensor"] for r in results if r["bit_mismatches"] or not r["signs_match_seed"]]
    worst = max(max(r[k]["rel_l2"]) for r in results for k in ("random", "outlier"))
    print(f"checked {len(results)} tensors in {len(rotated_groups)} groups; worst rel_l2 {worst:.3e}; "
          f"bit/sign failures: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
