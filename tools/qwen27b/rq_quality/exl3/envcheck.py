#!/usr/bin/env python3
"""W8 smoke: record the ExLlamaV3 runtime and the GPU kernels it actually launches.

Writes a JSON with the torch / HIP / exllamav3 versions, the device gfx target, the extension
module path, the model's storage bits, and the kernel names (with call counts and device time)
of one prefill chunk and a few single-token decode steps taken with torch.profiler. The kernel
list shows whether the RDNA EXL3 kernels (exl3_* GEMV / GEMM, trellis decode) or some other
path ran.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def kernel_table(prof) -> list[dict]:
    rows = defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        if getattr(e, "device_type", None) is None or "CUDA" not in str(e.device_type):
            continue
        rows[e.name][0] += 1
        rows[e.name][1] += getattr(e, "device_time_total", 0.0) or getattr(e, "cuda_time_total", 0.0)
    return [dict(name=k, calls=v[0], device_us=v[1]) for k, v in sorted(rows.items(), key=lambda kv: -kv[1][1])]


def main(argv=None) -> int:
    import torch
    with torch.inference_mode():  # ExLlamaV3 cache / recurrent-state tensors are inference tensors
        return run(argv)


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--windows", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prefill-tokens", type=int, default=512)
    ap.add_argument("--decode-steps", type=int, default=8)
    a = ap.parse_args(argv)

    import torch
    import exllamav3
    from exllamav3 import Cache, Config, Model
    from exllamav3.cache import CacheLayer_fp16
    from exllamav3.ext import exllamav3_ext
    from torch.profiler import ProfilerActivity, profile

    index = json.loads(a.windows.read_text())
    tokens = np.fromfile(a.windows.parent / index["tokens_file"], dtype="<i4").astype(np.int64)
    report = dict(torch=torch.__version__, hip=torch.version.hip, exllamav3=importlib.metadata.version("exllamav3"),
                  exllamav3_path=exllamav3.__file__, ext_path=getattr(exllamav3_ext, "__file__", "?"),
                  device=torch.cuda.get_device_properties(0).gcnArchName,
                  device_name=torch.cuda.get_device_name(0), model_dir=a.model_dir)
    t0 = time.perf_counter()
    config = Config.from_directory(a.model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096, layer_type=CacheLayer_fp16, max_batch_size=1)
    model.load(progressbar=False, max_chunk_size=4096, max_batch_size=1)
    report["load_s"] = time.perf_counter() - t0
    report["vocab_size"] = int(config.vocab_size)
    try:
        report["storage_info"] = [float(x) for x in model.get_storage_info()]
    except Exception as exc:  # informational only
        report["storage_info"] = repr(exc)
    state = cache.get_new_state()
    base = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, 4096), "recurrent_states": [state]}
    n = a.prefill_tokens
    with torch.inference_mode():
        model.prefill(torch.as_tensor(tokens[None, :n]), dict(base, past_len=0))  # warm-up / Triton JIT
        cache.release_state(state)
        state = cache.get_new_state()
        base["recurrent_states"] = [state]
        torch.cuda.synchronize()
        try:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                model.prefill(torch.as_tensor(tokens[None, :n]), dict(base, past_len=0))
                torch.cuda.synchronize()
            report["prefill_kernels"] = kernel_table(prof)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for i in range(a.decode_steps):
                    logits = model.forward(torch.as_tensor(tokens[None, n + i:n + i + 1]),
                                           dict(base, past_len=n + i, recurrent_states=[state]))
                torch.cuda.synchronize()
            report["decode_kernels"] = kernel_table(prof)
            report["decode_logits_shape"] = list(logits.shape)
            report["decode_logits_dtype"] = str(logits.dtype)
        except Exception as exc:  # keep the versions and load facts even if profiling fails
            report["profile_error"] = repr(exc)
            report.setdefault("prefill_kernels", []); report.setdefault("decode_kernels", [])
        report["torch_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    cache.release_state(state)
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    for key in ("prefill_kernels", "decode_kernels"):
        print(key)
        for row in report[key][:25]:
            print(f"  {row['device_us']:12.0f} us {row['calls']:6d}  {row['name'][:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
