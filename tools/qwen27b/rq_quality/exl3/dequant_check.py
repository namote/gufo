#!/usr/bin/env python3
"""Check a fake-quant BF16 GGUF (llama-quantize --allow-requantize ... BF16) against gguf-py.

For every quantized tensor type of the source GGUF, a few tensors and rows are dequantized with
gguf-py `quants.dequantize` (pass the reference revision's gguf-py directory with --gguf-py),
rounded to BF16 (round to nearest even, as ggml_fp32_to_bf16), and compared bit for bit with the
same rows of the BF16 file. Tensors that are F32/BF16 in the source must be byte-identical or
an exact BF16 rounding. Also checks that both files hold the same tensor names and shapes.
Exit 0 only when every sampled row matches.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def bf16_bits(x):
    import numpy as np
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    rounded = ((u + (0x7FFF + ((u >> 16) & 1))) >> 16).astype(np.uint16)
    return np.where(nan, ((u >> 16) | 64).astype(np.uint16), rounded)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("bf16")
    ap.add_argument("--gguf-py", required=True)
    ap.add_argument("--tensors-per-type", type=int, default=3)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.gguf_py)
    import numpy as np
    from gguf import GGUFReader, GGMLQuantizationType
    from gguf.quants import dequantize

    src, dst = GGUFReader(a.source), GGUFReader(a.bf16)
    dmap = {t.name: t for t in dst.tensors}
    smap = {t.name: t for t in src.tensors}
    report = dict(source=a.source, bf16=a.bf16, missing=sorted(set(smap) ^ set(dmap)), types={}, failures=[])
    by_type = defaultdict(list)
    for t in src.tensors:
        by_type[t.tensor_type.name].append(t)
    for tname, tensors in sorted(by_type.items()):
        picks = tensors if len(tensors) <= a.tensors_per_type else \
            [tensors[i * (len(tensors) - 1) // (a.tensors_per_type - 1)] for i in range(a.tensors_per_type)]
        checked = 0
        out_types = set()
        for t in picks:
            d = dmap.get(t.name)
            if d is None:
                continue
            out_types.add(d.tensor_type.name)
            if [int(x) for x in t.shape] != [int(x) for x in d.shape]:
                report["failures"].append(dict(tensor=t.name, error="shape"))
                continue
            row_len = int(t.shape[0])
            n_rows = int(np.prod([int(x) for x in t.shape[1:]])) if len(t.shape) > 1 else 1
            rows = sorted({0, n_rows // 2, max(n_rows - a.rows, 0)})
            raw = np.asarray(t.data)
            per_row = raw.size // n_rows if n_rows else raw.size
            flat = raw.reshape(n_rows, per_row) if len(t.shape) > 1 else raw.reshape(1, -1)
            want_rows = []
            for r in rows:
                block = flat[r:r + a.rows]
                vals = dequantize(block, t.tensor_type).reshape(-1, row_len)
                want_rows.append((r, vals))
            dd = np.asarray(d.data)
            dflat = dd.reshape(n_rows, -1) if len(t.shape) > 1 else dd.reshape(1, -1)
            for r, vals in want_rows:
                got = dflat[r:r + len(vals)]
                if d.tensor_type == GGMLQuantizationType.BF16:
                    got_bits = got.view(np.uint16).reshape(len(vals), row_len)
                    ok = np.array_equal(got_bits, bf16_bits(vals))
                elif d.tensor_type == GGMLQuantizationType.F32:
                    ok = np.array_equal(got.reshape(len(vals), row_len).view(np.uint32),
                                        vals.astype(np.float32).view(np.uint32))
                else:
                    ok = np.array_equal(got, flat[r:r + len(vals)])  # copied unchanged
                checked += len(vals)
                if not ok:
                    report["failures"].append(dict(tensor=t.name, type=tname, out=d.tensor_type.name, row=r))
        report["types"][tname] = dict(tensors=len(tensors), sampled=[t.name for t in picks], rows_checked=checked,
                                      out_types=sorted(out_types))
        print(f"{tname:8s} -> {','.join(sorted(out_types)):5s} tensors {len(tensors):4d} rows checked {checked}", flush=True)
    report["PASS"] = not report["failures"] and not report["missing"]
    print(json.dumps(dict(PASS=report["PASS"], failures=report["failures"][:10], missing=report["missing"][:10])))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
