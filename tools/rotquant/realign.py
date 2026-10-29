#!/usr/bin/env python3
"""Rewrite a GGUF with 256-byte data alignment; tensor bytes are unchanged.

Metadata keeps its original bytes (`general.alignment` is replaced and a
split input loses its `split.*` keys, becoming one file), tensors keep their
order, and the output is read back and compared tensor by tensor with the
input before the command succeeds.

Example:
  realign.py --input Qwen3.8-27B-UD-Q4_K_M.gguf --output Qwen3.8-27B-UD-Q4_K_M-align256.gguf
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rotquant.gguf_io import (SPLIT_KEYS, WRITE_ALIGNMENT, OutTensor, open_gguf,  # noqa: E402
                              write_gguf)

CHUNK_BYTES = 64 << 20


def _copy(model, tensor):
    def write(out):
        raw = model.tensor_bytes(tensor)
        for start in range(0, len(raw), CHUNK_BYTES):
            out.write(raw[start:start + CHUNK_BYTES])
    return write


def realign(source: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    with open_gguf(source) as model:
        tensors = [OutTensor(t.name, t.ne, t.type_name, _copy(model, t)) for t in model.tensors]
        kv_raw = [(key, raw) for key, raw in model.kv_raw if key not in SPLIT_KEYS]
        write_gguf(output, kv_raw, tensors)
        with open_gguf(output) as out:
            if [t.name for t in out.tensors] != [t.name for t in model.tensors]:
                raise ValueError(f"{output}: tensor order changed")
            src_kv = [(k, r) for k, r in model.kv_raw if k not in SPLIT_KEYS and k != "general.alignment"]
            out_kv = [(k, r) for k, r in out.kv_raw if k != "general.alignment"]
            if src_kv != out_kv:
                raise ValueError(f"{output}: metadata differs from the input")
            base = out.shards[0].data_offset
            for src, got in zip(model.tensors, out.tensors):
                if (got.type_name, tuple(got.ne)) != (src.type_name, tuple(src.ne)):
                    raise ValueError(f"{got.name}: type or shape changed")
                if (base + got.offset) % WRITE_ALIGNMENT:
                    raise ValueError(f"{got.name}: offset {base + got.offset} is not aligned")
                if out.tensor_bytes(got) != model.tensor_bytes(src):
                    raise ValueError(f"{got.name}: bytes differ from the input")
            return {"tensors": len(out.tensors), "alignment": out.alignment, "data_offset": base,
                    "input_alignment": model.alignment}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="GGUF (any shard of a split set)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = realign(args.input, args.output)
    except ValueError as error:
        print(f"realign: {error}", file=sys.stderr)
        return 2
    print(f"wrote {args.output}: {summary['tensors']} tensors byte-identical to the input, "
          f"alignment {summary['input_alignment']} -> {summary['alignment']}, data at {summary['data_offset']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
