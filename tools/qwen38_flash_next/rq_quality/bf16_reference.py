#!/usr/bin/env python3
"""CPU, disk-streamed HF BF16 oracle for the Flash-Next R0 a01 windows.

No checkpoint conversion, quantization, remote code, GPU, or full model load.
Uses the HF text model unchanged except its huge PLE embedding lookup: that
lookup gathers identical BF16 rows directly from the 128 checkpoint shards.
Each decoder layer's parameters are mmap'ed before forward and unmapped after;
only pages actually used (including routed experts) enter the process RSS.
The head runs in small row chunks and emits the existing rq-logits.v1 records.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import mmap
from pathlib import Path
import resource
import struct
import sys
import time

SHARED = Path(__file__).resolve().parents[2] / "qwen27b" / "rq_quality"
# parents[2] is tools, so reuse the single record implementation.
sys.path.insert(0, str(SHARED))
DEFAULT_MODEL = Path("/run/media/m2on/T5 EVO/AI/model/originals-flash-next-hf-20261030/"
                     "Qwen-Qwen3.8-Flash-Next")
MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
EXPECTED_INDEX_SHA256 = "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de"
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class Checkpoint:
    """Read only small safetensors headers until a forward needs a tensor."""

    def __init__(self, directory: Path):
        self.directory = directory
        raw = (directory / "model.safetensors.index.json").read_bytes()
        self.index_sha256 = digest(raw)
        self.weight_map = json.loads(raw)["weight_map"]
        self.headers = {}

    def tensor(self, name: str) -> tuple[Path, int, dict]:
        path = self.directory / self.weight_map[name]
        if path not in self.headers:
            with path.open("rb") as f:
                count = f.read(8)
                if len(count) != 8:
                    raise ValueError(f"short safetensors file: {path}")
                size = struct.unpack("<Q", count)[0]
                if not 0 < size < 100_000_000:
                    raise ValueError(f"invalid safetensors header: {path}")
                header = json.loads(f.read(size))
            self.headers[path] = (8 + size, header)
        start, header = self.headers[path]
        entry = header[name]
        a, b = entry["data_offsets"]
        expected = math.prod(entry["shape"]) * DTYPE_BYTES[entry["dtype"]]
        if a < 0 or b - a != expected or start + b > path.stat().st_size:
            raise ValueError(f"invalid tensor range: {name}")
        return path, start + a, entry

    def embedding_parts(self, name: str) -> list[str]:
        # Runtime HF embedding is logical concatenation along dimension zero.
        base = name.removesuffix(".weight") + ".shard_"
        parts = [key for key in self.weight_map if key.startswith(base)
                 and key.endswith(".weight")]
        parts.sort(key=lambda key: int(key[len(base):-len(".weight")]))
        if parts != [base + str(i) + ".weight" for i in range(len(parts))]:
            raise ValueError(f"non-contiguous embedding shards: {name}")
        return parts or [name]

    def map(self, name: str, torch):
        path, offset, entry = self.tensor(name)
        start = offset // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
        size = math.prod(entry["shape"]) * DTYPE_BYTES[entry["dtype"]]
        # ACCESS_COPY is writable for torch.frombuffer, but never changes disk.
        with path.open("rb") as f:
            region = mmap.mmap(f.fileno(), offset - start + size,
                               access=mmap.ACCESS_COPY, offset=start)
        dtype = {"BF16": torch.bfloat16, "F16": torch.float16,
                 "F32": torch.float32, "I64": torch.int64}[entry["dtype"]]
        tensor = torch.frombuffer(region, dtype=dtype, count=math.prod(entry["shape"]),
                                  offset=offset - start).reshape(entry["shape"])
        return region, tensor


def source_key(runtime_name: str) -> str:
    if runtime_name.startswith("model."):
        return "model.language_model." + runtime_name[len("model."):]
    return runtime_name


def environment() -> dict:
    packages = {}
    for name in ("torch", "transformers", "accelerate", "numpy", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig  # noqa: F401
        packages["qwen4_exp_supported"] = bool(
            packages["torch"] and "dummy" not in Qwen4ExpForCausalLM.__module__)
    except (ImportError, RuntimeError) as exc:
        packages["qwen4_exp_supported"] = False
        packages["error"] = str(exc)
    return packages


def build_model(checkpoint: Checkpoint):
    import torch
    from accelerate import init_empty_weights
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    raw = (checkpoint.directory / "config.json").read_bytes()
    config_dict = json.loads(raw)
    if config_dict.get("model_type") != "qwen4_exp":
        raise ValueError("expected qwen4_exp checkpoint")
    config = Qwen4ExpTextConfig.from_dict(config_dict["text_config"])
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    config.dtype = torch.bfloat16
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        # Keep generated hash/RoPE buffers on CPU; parameters alone are meta.
        with init_empty_weights(include_buffers=False):
            model = Qwen4ExpForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.eval()

    class DiskEmbedding(torch.nn.Module):
        def __init__(self, name, original):
            super().__init__()
            self.num_embeddings = original.num_embeddings
            self.embedding_dim = original.embedding_dim
            self.parts = checkpoint.embedding_parts(name)
            self.ends = []
            rows = 0
            for key in self.parts:
                _, _, e = checkpoint.tensor(key)
                if e["dtype"] != "BF16" or len(e["shape"]) != 2 or e["shape"][1] != self.embedding_dim:
                    raise ValueError(f"invalid embedding shard: {key}")
                rows += e["shape"][0]
                self.ends.append(rows)
            if rows != self.num_embeddings:
                raise ValueError(f"embedding rows {rows} != HF {self.num_embeddings}")

        def forward(self, ids):
            flat = ids.reshape(-1)
            if flat.numel() and (flat.min() < 0 or flat.max() >= self.num_embeddings):
                raise ValueError("PLE row outside embedding")
            out = torch.empty((flat.numel(), self.embedding_dim), dtype=torch.bfloat16)
            start = 0
            for key, end in zip(self.parts, self.ends):
                slots = ((flat >= start) & (flat < end)).nonzero().flatten()
                if slots.numel():
                    region, weight = checkpoint.map(key, torch)
                    try:
                        out[slots] = weight.index_select(0, flat[slots] - start)
                    finally:
                        del weight
                        region.close()
                start = end
            return out.reshape(*ids.shape, self.embedding_dim)

    # Replace only PLE lookup, preserving HF hashing, EOS handling and history.
    ple_parts = []
    for name, module in list(model.named_modules()):
        if name.endswith(".ngram_embedding"):
            replacement = DiskEmbedding(source_key(name + ".weight"), module)
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, replacement)
            ple_parts.extend(replacement.parts)
    if len(ple_parts) != config.split_ngram_parts * len(config.ple_layer_ids):
        raise ValueError("unexpected HF PLE layout; use the pinned transformers version")

    # The checkpoint carries the PLE hash multipliers, offsets and prime
    # sizes. Load those exact integers rather than relying on config defaults.
    for name, buffer in list(model.named_buffers()):
        key = source_key(name)
        parent, _, leaf = name.rpartition(".")
        owner = model.get_submodule(parent)
        if key not in checkpoint.weight_map and leaf not in owner._non_persistent_buffers_set:
            raise ValueError(f"missing persistent checkpoint buffer: {name}")
        if key in checkpoint.weight_map:
            region, value = checkpoint.map(key, torch)
            try:
                if value.shape != buffer.shape or value.dtype != buffer.dtype:
                    raise ValueError(f"HF/checkpoint buffer mismatch: {name}")
                owner._buffers[leaf] = value.clone()
            finally:
                del value
                region.close()

    # Whole-layer hooks also cover weights accessed directly (e.g. conv1d.weight),
    # which ordinary leaf hooks miss. Expert banks are mmap views, never cloned.
    units = [("model.embed_tokens", model.model.embed_tokens),
             ("model.hyper_connection_mixer", model.model.hyper_connection_mixer),
             ("lm_head", model.lm_head)]
    units += [(f"model.layers.{i}", layer) for i, layer in enumerate(model.model.layers)]
    covered = set()
    for base, unit in units:
        entries = []
        for relative, param in unit.named_parameters():
            name = base + "." + relative
            key = source_key(name)
            _, _, entry = checkpoint.tensor(key)  # fail before any expensive forward
            if tuple(entry["shape"]) != tuple(param.shape) or entry["dtype"] not in ("BF16", "F32"):
                raise ValueError(f"HF/checkpoint shape or dtype mismatch: {name}")
            parent, _, leaf = relative.rpartition(".")
            owner = unit.get_submodule(parent) if parent else unit
            entries.append((owner, leaf, tuple(param.shape), key))
            covered.add(name)

        def pre(module, args, entries=entries):
            module._rq_maps = []
            for owner, leaf, shape, key in entries:
                region, value = checkpoint.map(key, torch)
                module._rq_maps.append(region)
                owner._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)

        def post(module, args, result, entries=entries):
            for owner, leaf, shape, key in entries:
                owner._parameters[leaf] = torch.nn.Parameter(
                    torch.empty(shape, device="meta", dtype=torch.bfloat16), requires_grad=False)
            for region in getattr(module, "_rq_maps", []):
                region.close()
            module._rq_maps = []

        unit.register_forward_pre_hook(pre)
        unit.register_forward_hook(post, always_call=True)
    if covered != {name for name, _ in model.named_parameters()}:
        raise ValueError("unmapped HF parameters remain")
    return model, digest(raw)


def load_corpus(path: Path):
    import numpy as np
    raw = path.read_bytes()
    index = json.loads(raw)
    if index.get("schema") != "gufo.rq-corpus-windows.v1":
        raise ValueError("unexpected corpus schema")
    tokens_raw = (path.parent / index["tokens_file"]).read_bytes()
    if digest(tokens_raw) != index["tokens_sha256"]:
        raise ValueError("corpus token hash mismatch")
    tokens = np.frombuffer(tokens_raw, dtype="<i4")
    if np.any(tokens < 0) or np.any(tokens >= 248320):
        raise ValueError("corpus token outside vocabulary")
    for w in index["windows"]:
        if w["offset"] < 0 or not 2 <= w["length"] <= 4096 or w["offset"] + w["length"] > len(tokens):
            raise ValueError("corpus window outside token file")
    return index, tokens, digest(raw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", nargs="?", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--check-environment", action="store_true")
    ap.add_argument("--inspect-checkpoint", action="store_true", help="headers only; no torch needed")
    ap.add_argument("--windows", type=Path)
    ap.add_argument("--path", choices=("prefill", "decode"))
    ap.add_argument("--out")
    ap.add_argument("--top-k", type=int, default=256)
    ap.add_argument("--prefix", type=int, default=2048)
    ap.add_argument("--positions-limit", type=int, help="default: all prefill rows / 512 decode rows")
    ap.add_argument("--windows-limit", type=int, default=8)
    ap.add_argument("--row-chunk", type=int, default=16)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--label", default="flash-next-hf-bf16")
    ap.add_argument("--toolchain", default="")
    a = ap.parse_args(argv)
    if a.check_environment:
        info = environment()
        print(json.dumps(info, indent=2))
        return 0 if info["qwen4_exp_supported"] and info["accelerate"] and info["numpy"] else 1
    checkpoint = Checkpoint(a.model_dir)
    if a.inspect_checkpoint:
        totals = {"trunk": 0, "ple": 0, "other": 0}
        largest = 0
        for name in checkpoint.weight_map:
            _, _, entry = checkpoint.tensor(name)
            size = math.prod(entry["shape"]) * DTYPE_BYTES[entry["dtype"]]
            kind = "ple" if ".ngram_embedding." in name else (
                "trunk" if name.startswith("model.language_model.") or name == "lm_head.weight" else "other")
            totals[kind] += size
            largest = max(largest, size)
        print(json.dumps(dict(bytes=totals, largest_tensor_bytes=largest,
                              index_sha256=checkpoint.index_sha256), indent=2))
        return 0
    if not a.windows or not a.path or not a.out:
        ap.error("--windows, --path and --out are required for collection")
    if checkpoint.index_sha256 != EXPECTED_INDEX_SHA256:
        ap.error("checkpoint index differs from pinned Flash-Next de4b8e4d snapshot")
    tokenizer_sha = digest((a.model_dir / "tokenizer.json").read_bytes())
    manifest = json.loads((SHARED / "corpus_manifest.json").read_text())
    if tokenizer_sha != manifest["tokenizer"]["sha256"]:
        ap.error("Flash-Next tokenizer differs from the R0 a01 tokenizer")
    if not (1 <= a.top_k <= 1024 and a.prefix >= 0
            and (a.positions_limit is None or a.positions_limit > 0)
            and a.windows_limit > 0 and 1 <= a.row_chunk <= 128 and a.threads > 0):
        ap.error("invalid collection bounds")

    import numpy as np
    import torch
    import logits_io

    torch.set_num_threads(a.threads)
    info = environment()
    if not info["qwen4_exp_supported"]:
        raise SystemExit("Install transformers==5.17.0 with native qwen4_exp support; no trust_remote_code")
    corpus, tokens, index_sha = load_corpus(a.windows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # Never leave an older success receipt next to a failed/incomplete new bin.
    Path(a.out + ".json").unlink(missing_ok=True)
    t0 = time.perf_counter()
    model, config_sha = build_model(checkpoint)
    load_ms = (time.perf_counter() - t0) * 1000
    parts, windows_meta, cache_dtypes = [], [], set()
    head_checks = 0
    with torch.inference_mode():
        for wi, w in enumerate(corpus["windows"][:a.windows_limit]):
            seq = tokens[w["offset"]:w["offset"] + w["length"]].astype(np.int64)
            first = 0 if a.path == "prefill" else a.prefix
            limit = a.positions_limit if a.positions_limit is not None else (
                len(seq) - 1 if a.path == "prefill" else 512)
            end = first + min(limit, len(seq) - 1 - first)
            if first >= end:
                raise ValueError("prefix leaves no scored positions")
            tw = time.perf_counter()
            cache = None

            def forward(start, stop, scored):
                nonlocal cache, head_checks
                ids = torch.from_numpy(seq[start:stop].copy())[None]
                # Full-sequence prefill, as R0; logits are projected separately
                # so [4095,248320] never has to remain in RAM at once.
                result = model.model(input_ids=ids, past_key_values=cache,
                                     use_cache=a.path == "decode", return_dict=True)
                cache = result.past_key_values
                if cache is not None:
                    for layer in cache.layers:
                        for attr in ("keys", "values"):
                            value = getattr(layer, attr, None)
                            if value is not None:
                                cache_dtypes.add(str(value.dtype))
                if scored:
                    hidden = result.last_hidden_state[0]
                    for r in range(0, stop - start, a.row_chunk):
                        h = hidden[r:r + a.row_chunk]
                        logits = model.lm_head(h).float().numpy()
                        replay = model.lm_head(h).float().numpy()
                        if logits.tobytes() != replay.tobytes():
                            raise ValueError("HF head replay differs")
                        head_checks += 1
                        if not np.isfinite(logits).all():
                            raise ValueError("non-finite HF logits")
                        positions = np.arange(start + r, start + r + len(h))
                        parts.append(logits_io.records_from_logits(
                            logits, seq[positions + 1], np.full(len(h), wi), positions, a.top_k))

            if a.path == "prefill":
                forward(0, end, True)
            else:
                if first:
                    forward(0, first, False)
                for p in range(first, end):
                    forward(p, p + 1, True)
            windows_meta.append(dict(index=wi, id=w["id"], category=w["category"],
                                     tokens=len(seq), first_position=first, scored=end - first,
                                     wall_ms=(time.perf_counter() - tw) * 1000))
            del cache
            print(json.dumps(windows_meta[-1]), flush=True)
    records = np.concatenate(parts)
    meta = logits_io.write(a.out, records, model=str(a.model_dir), model_revision=MODEL_REVISION,
                    revision_note="expected provenance; pin/verify source files externally",
                    config_sha256=config_sha, checkpoint_index_sha256=checkpoint.index_sha256,
                    tokenizer_sha256=tokenizer_sha,
                    label=a.label, toolchain=a.toolchain, environment=info, path=a.path,
                    vocab_size=248320, prefix=0 if a.path == "prefill" else a.prefix,
                    windows_file=str(a.windows), windows_index_sha256=index_sha,
                    windows_tokens_sha256=corpus["tokens_sha256"], windows=windows_meta,
                    kv_cache="HF native", cache_dtypes=sorted(cache_dtypes),
                    weights_dtype="bfloat16 (checkpoint F32 tensors preserved)",
                    head_replay_checks=head_checks, load_ms=load_ms,
                    total_ms=(time.perf_counter() - t0) * 1000,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    # Include the record serialization peak, too (Linux ru_maxrss is KiB).
    meta["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    Path(a.out + ".json").write_text(json.dumps(meta, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
