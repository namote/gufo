#!/usr/bin/env python3
"""Collect an independent EXL3 vision-only oracle from a pinned exllamav3.

Run inside the existing exllamav3 environment, after the Gufo vision test has
written resized.rgb and shape.txt. This loads only the vision component. It
calls upstream modules, including their EXL3 decode, rather than sharing Gufo's
loader, arithmetic, or projection implementation. Comparison is diagnostic
unless an explicit, preselected relative-RMS limit is supplied.

Example:
  python vision_exl3_reference.py --model /models/fnx \
      --input-dir /receipt/gufo --output-dir /receipt/upstream \
      --compare-dir /receipt/gufo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compare-dir", type=Path)
    parser.add_argument("--relative-rms-limit", type=float)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.relative_rms_limit is not None and args.relative_rms_limit <= 0:
        parser.error("relative-RMS limit must be positive")

    import numpy as np
    from PIL import Image
    import torch
    import exllamav3
    from exllamav3 import Config, Model
    from exllamav3.architecture.mm_processing.qwen2 import (
        qwen2_position_embedding_grid_2d,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    width, height = map(int, (args.input_dir / "shape.txt").read_text().split())
    rgb = (args.input_dir / "resized.rgb").read_bytes()
    if len(rgb) != width * height * 3:
        raise ValueError("prepared RGB size does not match shape.txt")
    image = Image.frombytes("RGB", (width, height), rgb)
    source = Path(exllamav3.__file__).resolve().parent.parent
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    source_diff = subprocess.run(
        ["git", "-C", str(source), "diff", "HEAD", "--", "exllamav3"],
        capture_output=True,
        check=False,
    ).stdout
    config = Config.from_directory(str(args.model))
    model = Model.from_config(config, component="vision")
    patches, prepared_size, grid = model.preprocess(image)
    if tuple(prepared_size) != (width, height):
        raise ValueError(
            f"upstream resized prepared RGB: {(width, height)} -> {prepared_size}; "
            "use a shared valid image grid before comparing encoders"
        )
    if patches.dtype != torch.float16:
        raise ValueError(f"unexpected upstream patch dtype: {patches.dtype}")
    patches.cpu().numpy().tofile(args.output_dir / "patches.f16")
    model.load(device=args.device, max_chunk_size=patches.shape[0])
    params = {
        "causal": False,
        "grid_thw": torch.tensor([grid], dtype=torch.int),
        "inv_freq": qwen2_position_embedding_grid_2d(
            grid, config.vision.head_dim, config.vision.spatial_merge_size,
            config.vision.rope_theta,
        ),
    }
    records = []
    fail = False

    def collect(stage, value, seconds):
        nonlocal fail
        values = value.detach().float().cpu().contiguous().numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite upstream output at {stage}")
        values.tofile(args.output_dir / f"{stage}.f32")
        record = {"stage": stage, "shape": list(values.shape), "seconds": seconds}
        if args.compare_dir is not None:
            candidate_path = args.compare_dir / f"{stage}.f32"
            candidate = np.fromfile(candidate_path, dtype=np.float32)
            if candidate.size != values.size or not np.isfinite(candidate).all():
                raise ValueError(f"invalid candidate output at {stage}")
            x, y = values.reshape(-1).astype(np.float64), candidate.astype(np.float64)
            difference = y - x
            relative = float(np.linalg.norm(difference) / max(np.linalg.norm(x), 1e-30))
            record.update({
                "relative_rms": relative,
                "max_absolute": float(np.max(np.abs(difference))),
                "cosine": float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30)),
                "candidate_file": str(candidate_path),
            })
            if args.relative_rms_limit is not None:
                record["pass"] = relative <= args.relative_rms_limit
                fail |= not record["pass"]
        records.append(record)
        print(json.dumps(record), flush=True)

    with torch.inference_mode():
        x = patches.unsqueeze(0).to(args.device)
        for index, module in enumerate(model.modules):
            torch.cuda.synchronize()
            start = time.perf_counter()
            x = module.forward(x.to(module.device), params)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if index == 1:
                collect("patch", x, seconds)
            elif hasattr(module, "layer_idx") and module.layer_idx is not None:
                collect(f"layer{module.layer_idx}", x, seconds)
            elif index == len(model.modules) - 1:
                collect("embedding", x, seconds)
    metadata = {
        "model": str(args.model),
        "source": str(source),
        "source_revision": revision or None,
        "source_diff_sha256": hashlib.sha256(source_diff).hexdigest(),
        "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": args.device,
        "vision_only": True,
        "grid": list(grid),
        "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
        "environment": {k: v for k, v in sorted(os.environ.items()) if k.startswith("EXL3_")},
        "relative_rms_limit": args.relative_rms_limit,
        "qualified": None if args.relative_rms_limit is None else not fail,
        "stages": records,
    }
    (args.output_dir / "REFERENCE.json").write_text(json.dumps(metadata, indent=2) + "\n")
    model.unload()
    return int(fail)


if __name__ == "__main__":
    raise SystemExit(main())
