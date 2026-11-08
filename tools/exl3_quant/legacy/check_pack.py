#!/usr/bin/env python3
"""Post-run checks of an XQ EXL3 pack against the turboderp reference pack (XQ, 2026-10-30).

  python3 -I check_pack.py <ours: pack dir or qtensors dir> <reference pack dir> \
      [--summary run.summary.json ...] [--out report.json] [--partial]

Checks (all must pass):
  headers     every tensor of the reference exists in ours with the same dtype and shape and vice
              versa (with --partial only for the layer groups present in ours, e.g. a qtensors dir).
  out_scales  for every EXL3 linear (a `.svh` tensor): output channel scales are present when the
              svh magnitudes spread (max|svh| / min|svh| > 1.01); without scales svh holds only
              signs (|svh| within fp16 rounding of 1). Ours must match the reference tensor by
              tensor (turboderp 5.00 = "always": every tensor has scales).
  config      quantization_config (pack only): quant_method exl3, bits 5.0, head_bits 6, mtp_bits 4,
              out_scales always, codebook mul1, calibration 250 x 2048.
  sqnr        from the converter lines kept in run_convert.py summaries: lm_head is quantized with
              sqnr >= 10 dB (upstream PR #458 symptom), and no module has sqnr < 10 dB.
Standard library only; reads headers and the small svh tensors.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

LAYER = re.compile(r"^(.*\.layers\.\d+)\.")
QLINE = re.compile(r"Quantized: (\S+)\s.*?sqnr: ([-0-9.]+)")
EXPECT = {"quant_method": "exl3", "bits": 5.0, "head_bits": 6, "mtp_bits": 4, "out_scales": "always",
          "codebook": "mul1", "calibration": {"rows": 250, "cols": 2048}}


def headers(directory: Path) -> dict:
    out = {}
    for f in sorted(directory.glob("*.safetensors")):
        with f.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            out[k] = {"dtype": v["dtype"], "shape": v["shape"], "file": f, "base": 8 + n, "offsets": v["data_offsets"]}
    return out


def f16(entry: dict) -> tuple[float, ...]:
    a, b = entry["offsets"]
    with entry["file"].open("rb") as fh:
        fh.seek(entry["base"] + a)
        data = fh.read(b - a)
    return struct.unpack(f"<{len(data) // 2}e", data)


def has_scales(entry: dict) -> tuple[bool, float]:
    if entry["dtype"] != "F16":
        raise ValueError(f"svh dtype {entry['dtype']}")
    mags = [abs(x) for x in f16(entry)]
    lo, hi = min(mags), max(mags)
    ratio = hi / lo if lo > 0 else float("inf")
    return ratio > 1.01, ratio


def group(key: str) -> str:
    m = LAYER.match(key)
    return m.group(1) if m else key.rsplit(".", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ours", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--summary", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--partial", action="store_true", help="compare only the groups present in ours")
    a = ap.parse_args()
    ours, ref = headers(a.ours), headers(a.reference)
    rep: dict = {"ours": str(a.ours), "reference": str(a.reference), "checks": {}}

    # headers
    groups = {group(k) for k in ours}
    ref_keys = {k for k in ref if not a.partial or group(k) in groups}
    mism = [k for k in sorted(set(ours) & ref_keys) if (ours[k]["dtype"], ours[k]["shape"]) != (ref[k]["dtype"], ref[k]["shape"])]
    missing = sorted(ref_keys - set(ours))
    extra = sorted(set(ours) - set(ref))
    rep["checks"]["headers"] = {"PASS": bool(ours) and not (mism or missing or extra), "tensors": len(ours),
                                "reference_tensors": len(ref_keys), "mismatch": mism[:50], "missing_in_ours": missing[:50],
                                "missing_in_reference": extra[:50], "groups": len(groups)}

    # out_scales
    rows, bad = [], []
    for k in sorted(k for k in ours if k.endswith(".svh")):
        o, ro = has_scales(ours[k])
        r, rr = has_scales(ref[k]) if k in ref else (None, None)
        rows.append({"key": k[:-4], "ours": o, "ours_ratio": round(ro, 4), "reference": r,
                     "reference_ratio": None if rr is None else round(rr, 4)})
        if r is None or o != r:
            bad.append(k[:-4])
    rep["checks"]["out_scales"] = {"PASS": bool(rows) and not bad, "linears": len(rows), "with_scales": sum(r["ours"] for r in rows),
                                   "reference_with_scales": sum(bool(r["reference"]) for r in rows), "mismatch": bad, "per_tensor": rows}

    # config
    qc_path = a.ours / "config.json"
    if qc_path.exists():
        qc = json.loads(qc_path.read_text()).get("quantization_config", {})
        diff = {k: qc.get(k) for k, v in EXPECT.items() if qc.get(k) != v}
        rep["checks"]["config"] = {"PASS": not diff, "quantization_config": qc, "differs": diff}
    elif not a.partial:
        rep["checks"]["config"] = {"PASS": False, "error": "no config.json"}

    # sqnr
    if a.summary:
        mods = {}
        for s in a.summary:
            for _, line in json.loads(s.read_text())["lines"]:
                m = QLINE.search(line)
                if m and "proxy_err" not in line:
                    mods[m.group(1)] = float(m.group(2))
        quantized = {k: v for k, v in mods.items() if v != 0.0}  # embed/norm print 0
        low = {k: v for k, v in quantized.items() if v < 10.0}
        head = mods.get("lm_head")
        need_head = not a.partial
        rep["checks"]["sqnr"] = {"PASS": not low and (head is not None and head >= 10.0 or not need_head),
                                 "lm_head": head, "modules": len(quantized), "below_10_dB": low,
                                 "min": min(quantized.items(), key=lambda t: t[1]) if quantized else None}
    elif not a.partial:
        rep["checks"]["sqnr"] = {"PASS": False, "error": "no --summary given"}

    rep["PASS"] = all(c["PASS"] for c in rep["checks"].values())
    text = json.dumps(rep, indent=1, default=str) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    short = {k: {kk: vv for kk, vv in v.items() if kk not in ("per_tensor", "quantization_config")} for k, v in rep["checks"].items()}
    print(json.dumps({"PASS": rep["PASS"], **short}, indent=1, default=str))
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
