#!/usr/bin/env python3
"""Collect RQ quality records from an ExLlamaV3 model (W8, plan/rotated-quant/W8-BRIEF.md).

Feeds the corpus a01 token ids (windows.json / windows.bin, no re-tokenization) through ExLlamaV3
and writes logits_io runs that metrics.py scores against the gufo BF16 reference:

- `prefill`: one full-sequence forward per window from a fresh state (no cache), positions
  0 .. length - 2. gufo's prefill and verify paths have no ExLlamaV3 counterpart; this is the
  prefill analogue.
- `decode`: the first `--prefix` tokens of each window are prefilled into an fp16 cache, then
  one token per forward; positions prefix .. prefix + `--positions-limit` - 1, as the gufo
  decode reference.

Position p holds the logits after token p, predicting token p + 1. Each `--reference NAME=PREFIX`
(a logits_io run of the same path) gives one output: the first is written to `--out`, the others
to `<out>.vs-<NAME>`; they differ only in `ref_logits`. A reference prefix of `-` writes a
self-referenced run (the ExLlamaV3 BF16 control that later runs pair against).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import logits_io  # noqa: E402
import records  # noqa: E402

PAGE = 256


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def load_corpus(index_path: Path) -> dict:
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    tokens_path = index_path.parent / index["tokens_file"]
    raw = tokens_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != index["tokens_sha256"]:
        raise SystemExit(f"{tokens_path}: sha256 differs from {index_path}")
    tokens = np.frombuffer(raw, dtype="<i4")
    windows = []
    for i, w in enumerate(index["windows"]):
        windows.append(dict(index=i, id=w["id"], category=w.get("category", ""),
                            tokens=tokens[w["offset"]:w["offset"] + w["length"]].astype(np.int64)))
    return dict(path=str(index_path), index_sha256=hashlib.sha256(index_bytes).hexdigest(),
                tokens_sha256=index["tokens_sha256"], windows=windows)


class Reference:
    def __init__(self, name: str, prefix: str):
        # A trailing "?" on the name marks a partial reference (e.g. the ExLlamaV3 BF16 decode run
        # over fewer windows): windows it lacks are left out of that output only.
        self.partial = name.endswith("?")
        self.name = name.rstrip("?")
        self.run = None if prefix == "-" else logits_io.load(prefix)
        if self.run is not None:
            self.keys = self.run.keys()

    def ids(self, window: int, positions: np.ndarray) -> np.ndarray | None:
        if self.run is None:
            return None
        keys = (np.int64(window) << 32) | positions.astype(np.int64)
        index = np.searchsorted(self.keys, keys)
        if np.any(index >= len(self.keys)) or np.any(self.keys[np.minimum(index, len(self.keys) - 1)] != keys):
            if self.partial:
                return "absent"
            raise SystemExit(f"reference {self.name} lacks window {window} positions")
        return self.run.records["ids"][index]


def main(argv=None) -> int:
    import torch
    # ExLlamaV3 allocates its cache and recurrent-state tensors as inference tensors; creating or
    # clearing a recurrent state outside InferenceMode raises (w8s-a02), so the whole run is inside it.
    with torch.inference_mode():
        return run(argv)


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--windows", required=True, type=Path)
    ap.add_argument("--path", required=True, choices=("prefill", "decode"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--reference", action="append", required=True, help="NAME=PREFIX (or NAME=-)")
    ap.add_argument("--top-k", type=int, default=256)
    ap.add_argument("--prefix", type=int, default=2048)
    ap.add_argument("--positions-limit", type=int, default=512)
    ap.add_argument("--window-limit", type=int, default=0, help="only the first N windows (0: all)")
    ap.add_argument("--window-ids", default="", help="comma-separated window indices (default: all)")
    ap.add_argument("--row-chunk", type=int, default=512, help="logits rows per record batch")
    ap.add_argument("--toolchain", default="")
    a = ap.parse_args(argv)

    import torch
    import exllamav3
    from exllamav3 import Cache, Config, Model
    from exllamav3.cache import CacheLayer_fp16

    corpus = load_corpus(a.windows)
    windows = corpus["windows"][:a.window_limit or None]
    if a.window_ids:
        keep = {int(x) for x in a.window_ids.split(",")}
        windows = [w for w in windows if w["index"] in keep]
    refs = []
    for spec in a.reference:
        name, _, prefix = spec.partition("=")
        refs.append(Reference(name, prefix))
    for r in refs:
        if r.run is not None:
            if r.run.top_k != a.top_k or r.run.meta.get("path") != a.path:
                raise SystemExit(f"reference {r.name}: top-k/path differ")
            if r.run.meta.get("windows_tokens_sha256") != corpus["tokens_sha256"]:
                raise SystemExit(f"reference {r.name}: corpus differs")

    t0 = time.perf_counter()
    config = Config.from_directory(a.model_dir)
    model = Model.from_config(config)
    cache = None
    cap = 0
    if a.path == "decode":
        # Model.load runs its autosplit dummy forward at max_chunk_size (4096) against the attached
        # cache, so the cache must hold at least that many tokens.
        cap = max(-(-(a.prefix + a.positions_limit + 1) // PAGE) * PAGE, 4096)
        cache = Cache(model, max_num_tokens=cap, layer_type=CacheLayer_fp16, max_batch_size=1)
    model.load(progressbar=False, max_chunk_size=4096, max_batch_size=1)
    load_ms = (time.perf_counter() - t0) * 1e3
    vocab = int(config.vocab_size)
    print(f"loaded {a.model_dir} in {load_ms / 1e3:.1f} s, vocab {vocab}", flush=True)

    parts = [[] for _ in refs]
    meta_windows = []
    dtype_seen = set()
    for w in windows:
        tok = w["tokens"]
        tw = time.perf_counter()
        if a.path == "prefill":
            positions = np.arange(len(tok) - 1)
            # The forward returns logits for every token; the last one predicts past the window.
            logits = model.forward(torch.as_tensor(tok[None]), {"attn_mode": "flash_attn_nc"})[0][:len(positions)]
            torch.cuda.synchronize()
            forward_ms = (time.perf_counter() - tw) * 1e3
            rows = [(logits[s:s + a.row_chunk], positions[s:s + a.row_chunk])
                    for s in range(0, len(positions), a.row_chunk)]
        else:
            end = min(len(tok) - 1, a.prefix + a.positions_limit)
            positions = np.arange(a.prefix, end)
            state = cache.get_new_state()
            outs = []
            with torch.inference_mode():
                base = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, cap),
                        "recurrent_states": [state]}
                model.prefill(torch.as_tensor(tok[None, :a.prefix]), dict(base, past_len=0))
                for p in positions:
                    params = dict(base, past_len=int(p), recurrent_states=[state])
                    outs.append(model.forward(torch.as_tensor(tok[None, p:p + 1]), params)[0, -1:])
                    state = params["recurrent_states"][0]
            cache.release_state(state)
            torch.cuda.synchronize()
            forward_ms = (time.perf_counter() - tw) * 1e3
            logits = torch.cat(outs)
            rows = [(logits[s:s + a.row_chunk], positions[s:s + a.row_chunk])
                    for s in range(0, len(positions), a.row_chunk)]
        dtype_seen.add(str(logits.dtype))
        if logits.shape[-1] < vocab:
            raise SystemExit(f"logits width {logits.shape[-1]} < vocab {vocab}")
        for chunk, pos in rows:
            chunk = chunk[:, :vocab]
            ref_ids = [r.ids(w["index"], pos) for r in refs]
            present = [not (isinstance(x, str) and x == "absent") for x in ref_ids]
            ref_ids = [x if ok else None for x, ok in zip(ref_ids, present)]
            base_rec, ref_logits = records.records_torch(
                chunk, tok[pos + 1], np.full(len(pos), w["index"]), pos, a.top_k, ref_ids)
            for i, rl in enumerate(ref_logits):
                if not present[i]:
                    continue
                rec = base_rec.copy()
                rec["ref_logits"] = rl
                parts[i].append(rec)
        del logits, rows
        meta_windows.append(dict(index=w["index"], id=w["id"], category=w["category"], tokens=int(len(tok)),
                                 first_position=int(positions[0]), scored=int(len(positions)),
                                 forward_ms=forward_ms, wall_ms=(time.perf_counter() - tw) * 1e3))
        print(f"window {w['index']} {w['id']}: {len(positions)} positions, forward {forward_ms / 1e3:.1f} s", flush=True)

    model_files = sorted(Path(a.model_dir).glob("*.safetensors"))
    common = dict(
        label=a.label, engine="exllamav3 " + importlib.metadata.version("exllamav3"),
        toolchain=a.toolchain, torch=torch.__version__, hip_version=torch.version.hip,
        device=torch.cuda.get_device_properties(0).gcnArchName if torch.version.hip else torch.cuda.get_device_name(0),
        logits_dtype=",".join(sorted(dtype_seen)), model=a.model_dir,
        model_file_bytes=sum(p.stat().st_size for p in model_files), path=a.path, top_k=a.top_k,
        prefix=a.prefix if a.path == "decode" else 0, context=cap, vocab=vocab,
        windows_file=corpus["path"], windows_index_sha256=corpus["index_sha256"],
        windows_tokens_sha256=corpus["tokens_sha256"], load_ms=load_ms,
        total_ms=(time.perf_counter() - t0) * 1e3, torch_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        windows=meta_windows)
    for i, r in enumerate(refs):
        out = a.out if i == 0 else f"{a.out}.vs-{r.name}"
        reference = None if r.run is None else dict(meta=r.run.prefix + ".json", bin_sha256=r.run.meta["bin_sha256"],
                                                    name=r.name)
        if not parts[i]:
            print(f"no records for reference {r.name}; {out} not written", flush=True)
            continue
        recs = np.concatenate(parts[i])
        kept = set(int(x) for x in np.unique(recs["window"]))
        logits_io.write(out, recs, reference=reference,
                        **dict(common, windows=[m for m in common["windows"] if m["index"] in kept]))
        print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
