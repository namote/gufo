#!/usr/bin/env python3
"""Copy Flash-Next metadata; read headers and optional MTP mixer operands only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_header(path):
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
    if not isinstance(entries, dict):
        raise ValueError(f"{path}: invalid header object")
    payload = path.stat().st_size - 8 - length
    for name, entry in entries.items():
        if name == "__metadata__":
            continue
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= payload):
            raise ValueError(f"{path}: truncated tensor {name} or invalid offsets")
    return prefix, header, entries


def headers_digest(paths):
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        prefix, header, _ = read_header(path)
        digest.update(path.name.encode() + b"\0" + str(path.stat().st_size).encode() + b"\0")
        digest.update(prefix + header)
    return digest.hexdigest()


MTP_MIXER_NAMES = {
    "mtp.hyper_connection_mixer." + suffix for suffix in (
        "hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight")
}


def mtp_sources(paths, patch=None):
    """Select in-shard mixer operands and validate optional patch overlap."""
    def entries(path):
        _, header, tensors = read_header(path)
        return 8 + len(header), tensors

    config = json.loads((paths[0].parent / "config.json").read_text())
    text = config.get("text_config", config)
    width = int(text["hidden_size"]) * int(text["hc_count"])
    rank = int(text["hc_lowrank"])
    shapes = dict(zip(sorted(MTP_MIXER_NAMES), ([width], [rank, width], [width, rank])))

    def validate(name, entry):
        shape = shapes[name]
        element = {"F16": 2, "BF16": 2, "F32": 4}.get(entry.get("dtype"))
        elements = 1
        for dimension in shape:
            elements *= dimension
        begin, end = entry["data_offsets"]
        if (element is None or entry.get("shape") != shape
                or end - begin != elements * element):
            raise ValueError(f"invalid MTP mixer geometry/dtype/payload: {name}")

    main = {}
    for path in paths:
        start, header = entries(path)
        for name in MTP_MIXER_NAMES & header.keys():
            validate(name, header[name])
            main[name] = (path, start, header[name])
    extra = {}
    if patch is not None:
        patch = patch.resolve()
        if patch in paths:
            raise ValueError("MTP patch is already a checkpoint shard")
        start, header = entries(patch)
        if set(header) - MTP_MIXER_NAMES - {"__metadata__"}:
            raise ValueError("unexpected tensor in MTP mixer patch")
        for name in MTP_MIXER_NAMES & header.keys():
            validate(name, header[name])
            extra[name] = (patch, start, header[name])
    if MTP_MIXER_NAMES - (main.keys() | extra.keys()):
        raise ValueError("MTP mixer is missing in-shard and in the optional patch")
    # Read only these small operands when checking duplicate source bytes;
    # projection payloads and all unrelated extra files remain untouched.
    for name in main.keys() & extra.keys():
        p, base, a = main[name]
        q, other_base, b = extra[name]
        if a["dtype"] != b["dtype"] or a["shape"] != b["shape"]:
            raise ValueError(f"in-shard MTP mixer differs from patch: {name}")
        begin, end = a["data_offsets"]
        other_begin, other_end = b["data_offsets"]
        if end - begin != other_end - other_begin:
            raise ValueError(f"in-shard MTP mixer differs from patch: {name}")
        with p.open("rb") as lhs, q.open("rb") as rhs:
            lhs.seek(base + begin)
            rhs.seek(other_base + other_begin)
            remaining = end - begin
            while remaining:
                size = min(remaining, 1 << 20)
                left, right = lhs.read(size), rhs.read(size)
                if len(left) != size or left != right:
                    raise ValueError(f"in-shard MTP mixer differs from patch: {name}")
                remaining -= size
    return list(paths) + ([patch] if patch is not None else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("template", type=Path, help="retained IQ4_NL GGUF trunk/PLE source")
    parser.add_argument("output", type=Path)
    parser.add_argument("--mtp-mixer", type=Path, help="optional supplemental MTP mixer")
    args = parser.parse_args()
    from rotquant.gguf_io import encode_value, open_gguf, write_gguf
    directory = args.checkpoint.resolve()
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    paths = [directory / name for name in sorted(set(index["weight_map"].values()))]
    mtp_paths = mtp_sources(paths, args.mtp_mixer)
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
        "gufo.exl3.ple_headers_sha256": ple_digest.hexdigest(),
        "gufo.exl3.headers_sha256": headers_digest(paths),
        "gufo.exl3.mtp_headers_sha256": headers_digest(mtp_paths),
    }
    if args.mtp_mixer is not None:
        additions["gufo.exl3.mtp_mixer"] = os.path.relpath(args.mtp_mixer.resolve(), base)
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
