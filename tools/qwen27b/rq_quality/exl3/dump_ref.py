#!/usr/bin/env python3
"""Upstream ExLlamaV3 reference values for gufo's EXL3 CPU reference (X0, plan/exl3/archive/XA-BRIEF.md).

Runs in the W8 runtime image (ai-tools/exl3-eval:1.6.5-rocm100) on the GPU and writes JSON that
tests/core/quant/exl3_upstream_test.cpp compares against src/core/quant/exl3/.

- `synth`: every integer K 1..8 with each codebook (3INST, mcg, mul1) and the half-integer rates 1.5,
  2.5, 3.5 (mul1). Trellis words, suh, svh and x come from splitmix64 streams that the C++ test
  regenerates (no binary fixture). Recorded: sha256 of the upstream `ext.reconstruct` FP16 inner
  matrix W' (bits), its first 64 values, the upstream FP16 forward `LinearEXL3.forward` (the GEMV
  path) and an FP64 evaluation of ((x * suh) H) W' H * svh with upstream's own Hadamard matrix.
- `pack`: for named tensors of real checkpoints, the sha256 of the full upstream inner matrix and its
  first 64 values.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

MASK64 = (1 << 64) - 1


def splitmix64(seed: int):
    state = seed & MASK64
    while True:
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        yield z ^ (z >> 31)


def stream_u16(seed: int, count: int) -> np.ndarray:
    gen = splitmix64(seed)
    return np.array([next(gen) & 0xFFFF for _ in range(count)], dtype=np.uint16)


def scale_bits(seed: int, count: int, base: int) -> np.ndarray:
    """FP16 bit patterns sign | base exponent | random mantissa (exact values on both sides)."""
    r = stream_u16(seed, count).astype(np.uint32)
    return (((r >> 10) & 1) << 15 | base | (r & 0x3FF)).astype(np.uint16)


CODEBOOKS = {"3inst": (False, False), "mcg": (True, False), "mul1": (False, True)}
MARKER = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}


def cases():
    for cb in ("3inst", "mcg", "mul1"):
        for k_bits in range(1, 9):
            yield cb, float(k_bits)
    for k in (1.5, 2.5, 3.5):
        yield "mul1", k


def bits_of(t) -> np.ndarray:
    import torch
    return t.contiguous().cpu().view(torch.int16).numpy().view(np.uint16)


def synth(args, torch, ext, LinearEXL3, had128):
    out = []
    k, n = args.k, args.n
    for index, (cb, K) in enumerate(cases()):
        mcg, mul1 = CODEBOOKS[cb]
        words = int(16 * K)
        seed = 0x5EED0000 + index
        trellis_u16 = stream_u16(seed, (k // 16) * (n // 16) * words)
        suh_u16 = scale_bits(seed + 0x100, k, 0x3800)   # +-[0.5, 1)
        svh_u16 = scale_bits(seed + 0x200, n, 0x2000)   # +-[2^-7, 2^-6)
        x_u16 = scale_bits(seed + 0x300, k, 0x3C00)     # +-[1, 2)
        dev = "cuda"
        trellis = torch.from_numpy(trellis_u16.view(np.int16).copy()).view(k // 16, n // 16, words).to(dev)
        suh = torch.from_numpy(suh_u16.view(np.float16).copy()).to(dev)
        svh = torch.from_numpy(svh_u16.view(np.float16).copy()).to(dev)
        x = torch.from_numpy(x_u16.view(np.float16).copy()).to(dev).view(1, k)
        inner = torch.empty((k, n), dtype=torch.half, device=dev)
        ext.reconstruct(inner, trellis, K, mcg, mul1)
        torch.cuda.synchronize()
        inner_bits = bits_of(inner)
        markers = {}
        if cb in MARKER:
            markers[cb] = torch.tensor(MARKER[cb], dtype=torch.uint32).view(torch.int).to(dev)
        lin = LinearEXL3(None, k, n, suh=suh, svh=svh, trellis=trellis, mcg=markers.get("mcg"),
                         mul1=markers.get("mul1"), key=f"synth.{cb}.{K}")
        y_rt = lin.forward(x, {}).float().cpu().numpy().reshape(-1)
        # FP64 reference with upstream's Hadamard matrix (scaled 1/sqrt(128))
        H = had128.double().cpu().numpy() / np.sqrt(128.0)
        xd = x.double().cpu().numpy().reshape(-1) * suh.double().cpu().numpy()
        xh = (xd.reshape(-1, 128) @ H).reshape(-1)
        t = xh @ inner.double().cpu().numpy()
        y64 = (t.reshape(-1, 128) @ H).reshape(-1) * svh.double().cpu().numpy()
        out.append(dict(codebook=cb, K=K, k=k, n=n, seed=seed,
                        inner_sha256=hashlib.sha256(inner_bits.tobytes()).hexdigest(),
                        inner_head=[int(v) for v in inner_bits.reshape(-1)[:64]],
                        y_runtime_f16=[float(v) for v in y_rt], y_fp64=[float(v) for v in y64]))
        print(f"synth {cb} K={K}: inner {out[-1]['inner_sha256'][:16]}", flush=True)
    return out


def pack(args, torch, ext):
    from safetensors import safe_open
    out = []
    for spec in args.tensor:
        pack_dir, _, module = spec.partition(":")
        index = json.loads((Path(pack_dir) / "model.safetensors.index.json").read_text())
        shard = index["weight_map"][module + ".trellis"]
        with safe_open(str(Path(pack_dir) / shard), framework="pt", device="cpu") as f:
            trellis = f.get_tensor(module + ".trellis").cuda()
            mul1 = (module + ".mul1") in f.keys()
            mcg = (module + ".mcg") in f.keys()
        kb, nb, words = trellis.shape
        K = words / 16
        k, n = kb * 16, nb * 16
        inner = torch.empty((k, n), dtype=torch.half, device="cuda")
        ext.reconstruct(inner, trellis, K, mcg, mul1)
        torch.cuda.synchronize()
        bits = bits_of(inner)  # row-major [k][n]
        digest = hashlib.sha256(bits.tobytes()).hexdigest()
        head = [int(v) for v in bits.reshape(-1)[:64]]
        del inner, bits
        out.append(dict(pack=pack_dir, module=module, K=K, k=k, n=n, codebook="mul1" if mul1 else ("mcg" if mcg else "3inst"),
                        inner_sha256=digest, inner_head=head))
        print(f"pack {module}: K={K} {k}x{n} {digest[:16]}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("synth", "pack"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--tensor", action="append", default=[], help="PACK_DIR:module (pack mode)")
    a = ap.parse_args(argv)
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.quant.exl3 import LinearEXL3
    from exllamav3.util.hadamard import get_hadamard
    with torch.inference_mode():
        report = dict(schema="gufo.exl3-upstream-ref.v1", exllamav3=importlib.metadata.version("exllamav3"),
                      torch=torch.__version__, device=torch.cuda.get_device_properties(0).gcnArchName,
                      prng="splitmix64; trellis word = low 16 bits; suh/svh/x bits = sign(bit 10) | base | mantissa(low 10)")
        if a.mode == "synth":
            report["cases"] = synth(a, torch, ext, LinearEXL3, get_hadamard(128))
        else:
            report["tensors"] = pack(a, torch, ext)
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
