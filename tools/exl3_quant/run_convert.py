#!/usr/bin/env python3
"""Run the ExLlamaV3 converter inside a window and record time and memory (XQ, 2026-10-30).

  python3 -I run_convert.py --summary <json> [--sha FILE ...] -- <convert_model arguments>

Runs `<this python> -I -m exllamav3.conversion.convert_model <arguments>` (the installed wheel,
never a source tree next to a convert.py), passing its output through, and samples every
`--interval` seconds: the container cgroup memory.current (peak also from memory.peak), the sum
of RSS over the converter's process tree, and the GPU VRAM and GTT use from amdgpu sysfs (the
whole device, not only this process). The summary JSON holds the argv, environment knobs,
versions, sha256 of the `--sha` files, wall time, exit code, the peaks, and the per-module lines
the converter prints (Quantized / rfn / Estimated remaining time), each with its elapsed second.
Standard library only.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

KEEP = re.compile(r"^ -- (Loading unquantized module|Quantized:|Layer:|Module:|Estimated remaining|Saving checkpoint|"
                  r"Resuming|Final bitrate|Calibration|Capturing|Preparing|Creating|Deciding)|rfn|sqnr|!!|Error|Traceback")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def tree_rss(root: int) -> int:
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for d in glob.glob("/proc/[0-9]*"):
        try:
            pid = int(d.rsplit("/", 1)[1])
            fields = Path(d, "stat").read_text().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(pid)
            for line in Path(d, "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss[pid] = int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            continue
    total, todo = 0, [root]
    while todo:
        p = todo.pop()
        total += rss.get(p, 0)
        todo.extend(children.get(p, []))
    return total


def gpu_mem() -> dict:
    out = {}
    for kind in ("vram", "gtt"):
        vals = [read_int(p) for p in glob.glob(f"/sys/class/drm/card*/device/mem_info_{kind}_used")]
        vals = [v for v in vals if v is not None]
        out[kind] = sum(vals) if vals else None
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--sha", action="append", default=[])
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("convert_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = a.convert_args[1:] if a.convert_args[:1] == ["--"] else a.convert_args
    cmd = [sys.executable, "-I", "-m", "exllamav3.conversion.convert_model", *args]

    s: dict = {"argv": cmd, "env": {k: os.environ.get(k) for k in (
        "MALLOC_MMAP_THRESHOLD_", "TRITON_CACHE_DIR", "HIP_VISIBLE_DEVICES", "ROCBLAS_USE_HIPBLASLT",
        "TORCH_BLAS_PREFER_HIPBLASLT", "PYTORCH_HIP_ALLOC_CONF")}}
    try:
        ver = subprocess.run([sys.executable, "-I", "-c",
                              "import torch, exllamav3; print(exllamav3.__version__); print(torch.__version__); "
                              "print(torch.version.hip); print(torch.cuda.get_device_properties(0).gcnArchName)"],
                             capture_output=True, text=True, timeout=300).stdout.split("\n")
        s["versions"] = dict(zip(("exllamav3", "torch", "hip", "gcn_arch"), ver))
    except Exception as exc:  # recorded, not fatal
        s["versions"] = {"error": repr(exc)}
    s["sha256"] = {p: sha256(p) for p in a.sha}

    peaks = {"cgroup_current": 0, "tree_rss": 0, "vram": 0, "gtt": 0}
    stop = threading.Event()
    s["gpu_before"] = gpu_mem()
    s["cgroup_before"] = read_int("/sys/fs/cgroup/memory.current")

    t0 = time.monotonic()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)

    def sample():
        while not stop.is_set():
            cur = {"cgroup_current": read_int("/sys/fs/cgroup/memory.current") or 0,
                   "tree_rss": tree_rss(p.pid), **{k: v or 0 for k, v in gpu_mem().items()}}
            for k, v in cur.items():
                peaks[k] = max(peaks[k], v)
            stop.wait(a.interval)
    th = threading.Thread(target=sample, daemon=True)
    th.start()

    lines = []
    assert p.stdout is not None
    for raw in p.stdout:
        sys.stdout.write(raw)
        sys.stdout.flush()
        for part in ANSI.sub("", raw).replace("\r", "\n").split("\n"):
            if part.strip() and KEEP.search(part):
                lines.append([round(time.monotonic() - t0, 1), part.rstrip()])
    rc = p.wait()
    stop.set()
    th.join()
    s.update(returncode=rc, wall_s=round(time.monotonic() - t0, 1), peaks=peaks,
             cgroup_memory_peak=read_int("/sys/fs/cgroup/memory.peak"), gpu_after=gpu_mem(), lines=lines)
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps({"returncode": rc, "wall_s": s["wall_s"], "peaks": peaks,
                      "cgroup_memory_peak": s["cgroup_memory_peak"]}))
    return rc


if __name__ == "__main__":
    sys.exit(main())
