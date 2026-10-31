#!/usr/bin/env python3
"""Header and config checks of the downloaded turboderp Flash-Next EXL3 5.05bpw_h6_ng6 pack (XQ, 2026-10-31).

  python3 -I check_fn_pack.py <pack dir> --out <report.json>

Checks (all must pass; headers only, no tensor data is read):
  config      config.json quantization_config equals the values recorded in plan/exl3/F2-PLAN.md section 2.
  storage     every tensor listed in quantization_config.json tensor_storage exists in the safetensors
              files (shards plus ngram_embedding.safetensors) with the recorded shape, dtype and byte size,
              and every stored tensor is listed, except the parameters of non-leaf modules that upstream
              does not list (quant_config.py lists leaf modules only: GDN A_log, dt_bias, conv1d and the
              PLE conv1d) and the MTP, vision and patch-file tensors, which are checked by `side` instead.
  side        trellis K (= last dim / 16): MTP experts and indexer 5 (1,537 tensors), MTP attention and
              shared expert 7 (-hq, 7), MTP fc_embedding/fc_hidden 6 (2); vision 6 (164).
  bits        EXL3 bits per category match F2-PLAN.md: experts 5, GDN/attention/shared expert 7,
              indexer 5, lm_head 6; router, in_proj_a/b, hyper-connections, PLE projections not EXL3;
              73,728 expert tensors; n-gram trellis int16 [320001536, 61].
  index       model.safetensors.index.json maps every shard tensor to the file that holds it.

--survey (for packs without recorded expectations, e.g. other bitrates or other quantizers): config, bits
and side only report what the pack holds (bits per category, MTP/vision K, n-gram trellis shape) and do
not fail; storage and index stay strict. --extra FILE adds the headers of a safetensors file kept outside
the pack (e.g. an n-gram table shared with another pack) to the storage check. --ignore-file NAME reports
the tensors of an engine-specific sidecar file (e.g. Kyojin's hc_gs_sidecar.safetensors) instead of
requiring them in tensor_storage.
Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import struct
import sys
from pathlib import Path

EXPECT_QC = {"quant_method": "exl3", "version": "1.4.4", "bits": 5.05, "head_bits": 6,
             "calibration": {"rows": 250, "cols": 2048}, "out_scales": "always", "codebook": "mul1",
             "vision_bits": 6, "mtp_bits": 5}
DTYPE = {"torch.float16": "F16", "torch.bfloat16": "BF16", "torch.int16": "I16", "torch.int32": "I32",
         "torch.int64": "I64", "torch.float32": "F32", "torch.uint8": "U8"}
BITS = [  # (pattern on the module key, expected bits or None = not EXL3)
    (r"\.mlp\.experts\.\d+\.(gate|up|down)_proj$", 5),
    (r"\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$", 7),
    (r"\.self_attn\.(q|k|v|o)_proj$", 7),
    (r"\.mlp\.shared_expert\.(gate|up|down)_proj$", 7),
    (r"\.self_attn\.indexer\.index_qk_proj$", 5),
    (r"^lm_head$", 6),
    (r"\.mlp\.gate$", None),
    (r"\.linear_attn\.in_proj_[ab]$", None),
    (r"hyper_connection", None),
    (r"\.ple\.(key|value)_proj$", None),
]


def headers(directory: Path, extra: list[Path] = ()) -> dict:
    out = {}
    for f in sorted(directory.glob("*.safetensors")) + list(extra):
        with f.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            out[k] = {"dtype": v["dtype"], "shape": v["shape"], "bytes": v["data_offsets"][1] - v["data_offsets"][0], "file": f.name}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--survey", action="store_true")
    ap.add_argument("--extra", type=Path, action="append", default=[])
    ap.add_argument("--ignore-file", action="append", default=[],
                    help="engine-specific sidecar file whose tensors are reported, not required in tensor_storage")
    a = ap.parse_args()
    rep: dict = {"pack": str(a.pack), "checks": {}}

    qc = json.loads((a.pack / "config.json").read_text()).get("quantization_config", {})
    rep["checks"]["config"] = {"PASS": a.survey or qc == EXPECT_QC, "quantization_config": qc,
                               "differs": {k: qc.get(k) for k in set(EXPECT_QC) | set(qc) if qc.get(k) != EXPECT_QC.get(k)}}

    hdr = headers(a.pack, a.extra)
    q = json.loads((a.pack / "quantization_config.json").read_text())
    storage = q["tensor_storage"]
    listed, bad, missing = set(), [], []
    for mod, v in storage.items():
        for name, t in v.get("stored_tensors", {}).items():
            listed.add(name)
            h = hdr.get(name)
            if h is None:
                missing.append(name)
            elif (h["shape"], h["dtype"], h["bytes"]) != (t["shape"], DTYPE.get(t["dtype"], t["dtype"]), t["n_bytes"]):
                bad.append({"tensor": name, "header": [h["shape"], h["dtype"], h["bytes"]], "listed": [t["shape"], t["dtype"], t["n_bytes"]]})
    unlisted = sorted(k for k in hdr if k not in listed)
    side = [k for k in unlisted if k.startswith("mtp.") or k.startswith("model.visual.") or hdr[k]["file"].startswith("mtp_")]
    # Parameters upstream may leave out of tensor_storage: non-leaf GDN/PLE parameters, and the PLE norms
    # (listed by the 5.05 pack, not by the 4.05 pack of the same repository)
    nonleaf = re.compile(r"\.(linear_attn\.(A_log|dt_bias|conv1d\.weight)|ple\.(conv1d|norm_conv|norm_key|norm_query)\.weight)$")
    side += [k for k in unlisted if hdr[k]["file"] in a.ignore_file and k not in side]
    unlisted_nonleaf = [k for k in unlisted if k not in side and nonleaf.search(k)]
    unlisted_main = [k for k in unlisted if k not in side and k not in unlisted_nonleaf]
    rep["checks"]["storage"] = {"PASS": not (bad or missing or unlisted_main), "listed": len(listed), "header_tensors": len(hdr),
                                "mismatch": bad[:20], "missing": missing[:20], "unlisted_main": unlisted_main[:20],
                                "unlisted_nonleaf": len(unlisted_nonleaf), "unlisted_mtp_vision_patch": len(side)}
    ks = collections.defaultdict(collections.Counter)
    for k in side:
        if k.endswith(".trellis"):
            ks["mtp" if k.startswith("mtp.") else "vision"][hdr[k]["shape"][-1] // 16] += 1
    rep["checks"]["side"] = {"PASS": a.survey or (dict(ks["mtp"]) == {5: 1537, 7: 7, 6: 2} and dict(ks["vision"]) == {6: 164}),
                             "trellis_K": {g: {str(k): n for k, n in c.items()} for g, c in ks.items()}}

    cats = collections.defaultdict(collections.Counter)
    wrong = []
    experts = 0
    for mod, v in storage.items():
        bits = v.get("bits_per_weight") if v.get("quant_format") == "exl3" else None
        for pat, want in BITS:
            if re.search(pat, mod):
                cats[pat][bits] += 1
                if bits != want:
                    wrong.append({"module": mod, "bits": bits, "want": want})
                if want == 5 and "experts" in pat:
                    experts += 1
                break
    ng = {k: v for k, v in hdr.items() if k.endswith("ngram_embedding.trellis")}
    ng_ok = len(ng) == 1 and next(iter(ng.values()))["shape"] == [320001536, 61] and next(iter(ng.values()))["dtype"] == "I16"
    if a.survey:
        cats = collections.defaultdict(collections.Counter)
        for mod, v in storage.items():
            bits = v.get("bits_per_weight") if v.get("quant_format") == "exl3" else None
            for pat, _ in BITS:
                if re.search(pat, mod):
                    cats[pat][bits] += 1
                    break
    rep["checks"]["bits"] = {"PASS": a.survey or (not wrong and experts == 73728 and ng_ok), "expert_tensors": experts,
                             "categories": {p: {str(k): n for k, n in c.items()} for p, c in cats.items()},
                             "wrong": wrong[:20], "ngram_trellis": ng}

    idx = json.loads((a.pack / "model.safetensors.index.json").read_text())["weight_map"]
    idx_bad = [k for k, f in idx.items() if hdr.get(k, {}).get("file") != f]
    rep["checks"]["index"] = {"PASS": not idx_bad, "entries": len(idx), "bad": idx_bad[:20]}

    rep["PASS"] = all(c["PASS"] for c in rep["checks"].values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"PASS": rep["PASS"], **{k: {kk: vv for kk, vv in v.items() if kk in ("PASS", "differs", "listed", "header_tensors",
          "mismatch", "missing", "unlisted_main", "unlisted_nonleaf", "unlisted_mtp_vision_patch", "trellis_K", "expert_tensors", "categories", "wrong", "entries", "bad")}
          for k, v in rep["checks"].items()}}, indent=1))
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
