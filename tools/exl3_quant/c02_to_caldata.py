#!/usr/bin/env python3
"""Turn the RQ c02 calibration corpus into an ExLlamaV3 `convert.py --cal_data` file (XQ, 2026-10-30).

ExLlamaV3 v1.6.5 (`exllamav3/conversion/calibration_data.py` get_file_calibration) reads a
safetensors file with an `input_ids` tensor (rows, cols) and uses its first `cal_rows` rows,
cropped to `cal_cols`. This tool writes rows of `--cols` tokens (default 2048 = 4 c02 chunks of
512), `--rows` of them (default 250, the converter's default and the turboderp packs' size).

Rule (plan/exl3/XQ-RESULTS.md section 2, approved by the supervisor 2026-10-30):
  1. c02 chunks are put back in stream order: segment = (stream, base c01 or extension), and
     inside a segment by the stream-local index (`<stream>-N`, `<stream>-xN`). Chats that span
     chunks (`documents_span_chunks`) become contiguous again.
  2. Consecutive groups of cols/512 chunks of one segment form one row; a segment's tail of
     fewer chunks is dropped.
  3. `--rows` rows are allocated to segments in proportion to their row counts (largest
     remainder; ties by segment order).
  4. Inside a segment the rows are drawn with random.Random(f"{seed}:{stream}:{segment}"), and
     the final row order is shuffled with random.Random(f"{seed}:order"). Seed default 20261029.
  5. `input_ids` is int64, no `lengths` tensor (every row is full).

Outputs: <out>.safetensors and <out>.json (inputs and their sha256, rule, allocation, chunk ids
of every row, output sha256). Standard library only.

  python3 -I tools/exl3_quant/c02_to_caldata.py \
      --c02 ~/projects/ai.tools/storage/evaluations/rq-r3-calib-20261029/c02 \
      --out "/run/media/m2on/T5 EVO/AI/model/xq-exl3-20261030/cal/c02-250x2048"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
from pathlib import Path

SCHEMA = "gufo.exl3-caldata.v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def segments(chunks: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Chunks by (stream, 'base' | 'ext'), each list in stream-local index order."""
    out: dict[tuple[str, str], list[tuple[int, dict]]] = {}
    for c in chunks:
        stream, index = c["id"].rsplit("-", 1)
        if stream != c["stream"]:
            raise ValueError(f"chunk id {c['id']} does not name its stream {c['stream']}")
        ext = index.startswith("x")
        n = int(index[1:] if ext else index)
        out.setdefault((stream, "ext" if ext else "base"), []).append((n, c))
    result = {}
    for key in sorted(out, key=lambda k: (k[0], k[1] != "base")):
        items = sorted(out[key], key=lambda t: t[0])
        if [n for n, _ in items] != list(range(len(items))):
            raise ValueError(f"segment {key} is not numbered 0..n-1")
        result[key] = [c for _, c in items]
    return result


def allocate(counts: dict, total: int) -> dict:
    available = sum(counts.values())
    if available < total:
        raise ValueError(f"only {available} rows available, {total} requested")
    share = {k: total * v / available for k, v in counts.items()}
    alloc = {k: math.floor(s) for k, s in share.items()}
    order = list(counts)
    for k in sorted(order, key=lambda k: (-(share[k] - alloc[k]), order.index(k)))[:total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def build(c02: Path, rows: int, cols: int, seed: int) -> tuple[list[list[int]], dict]:
    manifest_bytes = (c02 / "MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    index_bytes = (c02 / "chunks.json").read_bytes()
    index = json.loads(index_bytes)
    payload = (c02 / "chunks.bin").read_bytes()
    for name, data in (("chunks.bin", payload), ("chunks.json", index_bytes)):
        if sha256_bytes(data) != manifest["outputs"][name]:
            raise ValueError(f"{c02 / name}: sha256 differs from MANIFEST.json")
    if sha256_bytes(payload) != index["tokens_sha256"]:
        raise ValueError("chunks.bin sha256 differs from chunks.json")
    chunk = index["chunk_tokens"]
    if cols % chunk:
        raise ValueError(f"cols {cols} is not a multiple of the chunk size {chunk}")
    per_row = cols // chunk
    tokens = struct.unpack(f"<{len(payload) // 4}i", payload)

    segs = segments(index["chunks"])
    seg_rows = {k: [v[i:i + per_row] for i in range(0, len(v) - per_row + 1, per_row)] for k, v in segs.items()}
    alloc = allocate({k: len(v) for k, v in seg_rows.items()}, rows)

    picked = []
    for (stream, seg), cand in seg_rows.items():
        rng = random.Random(f"{seed}:{stream}:{seg}")
        for i in sorted(rng.sample(range(len(cand)), alloc[(stream, seg)])):
            picked.append(((stream, seg), i, cand[i]))
    random.Random(f"{seed}:order").shuffle(picked)

    data, row_info = [], []
    for (stream, seg), i, group in picked:
        ids = []
        for c in group:
            if c["length"] != chunk:
                raise ValueError(f"chunk {c['id']} has length {c['length']}")
            ids.extend(tokens[c["offset"]:c["offset"] + chunk])
        assert len(ids) == cols
        data.append(ids)
        row_info.append({"segment": f"{stream}/{seg}", "segment_row": i, "chunks": [c["id"] for c in group]})

    meta = {
        "schema": SCHEMA,
        "rule": "plan/exl3/XQ-RESULTS.md section 2: stream order, %d chunks per row, proportional "
                "largest-remainder allocation, Random(f'{seed}:{stream}:{segment}') per segment, "
                "Random(f'{seed}:order') final order, int64, no lengths" % per_row,
        "seed": seed,
        "rows": rows,
        "cols": cols,
        "chunk_tokens": chunk,
        "source": {
            "dir": str(c02),
            "corpus_version": manifest["corpus_version"],
            "MANIFEST.json": sha256_bytes(manifest_bytes),
            "chunks.json": sha256_bytes(index_bytes),
            "chunks.bin": sha256_bytes(payload),
            "tokenizer_sha256": manifest["tokenizer"]["sha256"],
        },
        "segment_rows_available": {f"{s}/{g}": len(v) for (s, g), v in seg_rows.items()},
        "allocation": {f"{s}/{g}": n for (s, g), n in alloc.items()},
        "max_token_id": max(max(r) for r in data),
        "min_token_id": min(min(r) for r in data),
        "row_list": row_info,
    }
    return data, meta


def write_safetensors(path: Path, rows: list[list[int]]) -> bytes:
    n_rows, n_cols = len(rows), len(rows[0])
    body = b"".join(struct.pack(f"<{n_cols}q", *r) for r in rows)
    header = {"input_ids": {"dtype": "I64", "shape": [n_rows, n_cols], "data_offsets": [0, len(body)]},
              "__metadata__": {"schema": SCHEMA}}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    blob = struct.pack("<Q", len(hb)) + hb + body
    path.write_bytes(blob)
    return blob


def read_safetensors_ids(blob: bytes) -> list[list[int]]:
    n = struct.unpack("<Q", blob[:8])[0]
    h = json.loads(blob[8:8 + n])["input_ids"]
    rows, cols = h["shape"]
    a, b = h["data_offsets"]
    flat = struct.unpack(f"<{rows * cols}q", blob[8 + n + a:8 + n + b])
    return [list(flat[i * cols:(i + 1) * cols]) for i in range(rows)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--c02", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output path without extension")
    ap.add_argument("--rows", type=int, default=250)
    ap.add_argument("--cols", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20261029)
    a = ap.parse_args(argv)
    rows, meta = build(a.c02, a.rows, a.cols, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    st = a.out.with_suffix(".safetensors")
    blob = write_safetensors(st, rows)
    if read_safetensors_ids(st.read_bytes()) != rows:
        raise AssertionError("written file does not read back")
    meta["output"] = {"file": st.name, "bytes": len(blob), "sha256": sha256_bytes(blob)}
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("allocation", "output", "max_token_id")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
