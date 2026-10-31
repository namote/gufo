#!/usr/bin/env python3
"""Copy Flash-Next GGUF metadata for direct EXL3 loading; read headers only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def headers_digest(paths):
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError(f"{path}: missing safetensors header")
            length, = struct.unpack("<Q", prefix)
            if not 0 < length <= 64 << 20:
                raise ValueError(f"{path}: invalid header size")
            header = source.read(length)
        if len(header) != length:
            raise ValueError(f"{path}: truncated header")
        entries = json.loads(header)
        payload = path.stat().st_size - 8 - length
        for name, entry in entries.items():
            if name != "__metadata__" and not 0 <= entry["data_offsets"][0] <= entry["data_offsets"][1] <= payload:
                raise ValueError(f"{path}: truncated tensor {name}")
        digest.update(path.name.encode() + b"\0" + str(path.stat().st_size).encode() + b"\0")
        digest.update(prefix + header)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("template", type=Path, help="retained IQ4_NL GGUF trunk/PLE source")
    parser.add_argument("output", type=Path)
    parser.add_argument("--mtp-mixer", type=Path, required=True)
    args = parser.parse_args()
    from rotquant.gguf_io import encode_value, open_gguf, write_gguf
    directory = args.checkpoint.resolve()
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    paths = [directory / name for name in sorted(set(index["weight_map"].values()))]
    # The same metadata supports --model and the existing --mtp-model option.
    with open_gguf(args.template) as template:
        if template.kv.get("general.architecture") != "qwen4exp":
            raise ValueError("template architecture must be qwen4exp")
        table = template.by_name.get("per_layer_token_embd.weight")
        if table is None or table.type_name != "IQ4_NL":
            raise ValueError("template must retain the IQ4_NL disk PLE table")
        ple_digest = hashlib.sha256()
        for shard in template.shards:
            ple_digest.update(shard.path.name.encode() + b"\0" + str(len(shard.data)).encode() + b"\0")
            ple_digest.update(shard.data[:shard.data_offset])
        kv = [(key, raw) for key, raw in template.kv_raw
              if not key.startswith(("split.", "gufo.exl3."))]
    base = args.output.resolve().parent
    additions = {
        "gufo.exl3.checkpoint": os.path.relpath(directory, base),
        "gufo.exl3.ple_source": os.path.relpath(args.template.resolve(), base),
        "gufo.exl3.mtp_mixer": os.path.relpath(args.mtp_mixer.resolve(), base),
        "gufo.exl3.ple_headers_sha256": ple_digest.hexdigest(),
        "gufo.exl3.headers_sha256": headers_digest(paths),
        "gufo.exl3.mtp_headers_sha256": headers_digest(paths + [args.mtp_mixer.resolve()]),
    }
    for filename in ("config", "tokenizer"):
        additions[f"gufo.exl3.{filename}_sha256"] = hashlib.sha256(
            (directory / f"{filename}.json").read_bytes()).hexdigest()
    # Keep metadata with an MTP layer (trunk files may omit nextn).
    from rotquant.gguf_io import U32
    with open_gguf(args.template) as template:
        layers = int(template.kv["qwen4exp.block_count"]) - int(template.kv.get("qwen4exp.nextn_predict_layers", 0))
    kv = [(key, raw) for key, raw in kv if key not in
          ("qwen4exp.block_count", "qwen4exp.nextn_predict_layers")]
    kv += [("qwen4exp.block_count", encode_value(U32(layers + 1))),
           ("qwen4exp.nextn_predict_layers", encode_value(U32(1)))]
    # Compression ratios must include the full-attention predictor block.
    from rotquant.gguf_io import I32Array
    with open_gguf(args.template) as template:
        ratios = list(template.kv["qwen4exp.attention.compress_ratios"])
    kv = [(key, raw) for key, raw in kv if key != "qwen4exp.attention.compress_ratios"]
    kv.append(("qwen4exp.attention.compress_ratios", encode_value(I32Array(ratios[:layers] + [4]))))
    kv += [(key, encode_value(value)) for key, value in additions.items()]
    write_gguf(args.output, kv, [])
    print(args.output)


if __name__ == "__main__":
    main()
