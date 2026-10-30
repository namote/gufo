#!/usr/bin/env python3
"""Compare converter qtensors (or a finished pack) with a reference EXL3 pack by safetensors headers.

  python3 -I compare_qtensors.py <ours: qtensors dir or pack dir> <reference pack dir> --out <json>

For every layer that has at least one tensor in `ours`, every tensor of that layer in either side
must exist in the other with the same shape and dtype (values differ: the calibration differs).
Layers are `<prefix>.layers.N` keys; other keys (embed, norm, lm_head, mtp.*) are compared when
present in `ours`. Exit 0 only when all match. Standard library only; reads headers, not data.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

LAYER = re.compile(r"^(.*\.layers\.\d+)\.")


def headers(directory: Path) -> dict:
    out = {}
    for f in sorted(directory.glob("*.safetensors")):
        with f.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            out[k] = {"dtype": v["dtype"], "shape": v["shape"], "bytes": v["data_offsets"][1] - v["data_offsets"][0], "file": f.name}
    return out


def group(key: str) -> str:
    m = LAYER.match(key)
    return m.group(1) if m else key.rsplit(".", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ours", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ours, ref = headers(a.ours), headers(a.reference)
    groups = sorted({group(k) for k in ours})
    rep = {"ours": str(a.ours), "reference": str(a.reference), "groups": {}, "mismatch": [], "missing_in_ours": [],
           "missing_in_reference": []}
    for g in groups:
        ok = [k for k in ours if group(k) == g]
        rk = [k for k in ref if group(k) == g]
        rep["groups"][g] = {"tensors": len(ok), "reference_tensors": len(rk),
                            "bytes": sum(ours[k]["bytes"] for k in ok), "reference_bytes": sum(ref[k]["bytes"] for k in rk)}
        rep["missing_in_ours"] += sorted(set(rk) - set(ok))
        rep["missing_in_reference"] += sorted(set(ok) - set(rk))
        for k in sorted(set(ok) & set(rk)):
            if (ours[k]["dtype"], ours[k]["shape"]) != (ref[k]["dtype"], ref[k]["shape"]):
                rep["mismatch"].append({"key": k, "ours": [ours[k]["dtype"], ours[k]["shape"]], "reference": [ref[k]["dtype"], ref[k]["shape"]]})
    rep["trellis_K"] = {k: ours[k]["shape"][-1] // 16 for k in ours if k.endswith(".trellis")}
    rep["PASS"] = bool(groups) and not (rep["mismatch"] or rep["missing_in_ours"] or rep["missing_in_reference"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"PASS": rep["PASS"], "groups": rep["groups"], "mismatch": rep["mismatch"][:10],
                      "missing_in_ours": rep["missing_in_ours"][:10], "missing_in_reference": rep["missing_in_reference"][:10]}, indent=1))
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
