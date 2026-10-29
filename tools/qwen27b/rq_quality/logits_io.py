"""Read and write qwen27b_rq_logits runs (tests/models/qwen27b/rq_logits.cpp).

A run is `<prefix>.json` (metadata, schema gufo.rq-logits.v1) plus
`<prefix>.bin`: one packed little-endian record per scored position,

    int32 window, int32 position, int32 target, float32 target_logit,
    float64 logsumexp, int32 ids[k], float32 logits[k], float32 ref_logits[k]

`position` p holds the logits after window token p, predicting token p + 1
(`target`). `ids`/`logits` are the run's own top-k, descending, ties to the
lower id; `logsumexp` covers the full vocabulary. `ref_logits[i]` is this
run's logit at the reference run's `ids[i]` (its own `ids[i]` without a
reference), which makes the truncated KL computable without full logits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "gufo.rq-logits.v1"


def record_dtype(k: int) -> np.dtype:
    return np.dtype([
        ("window", "<i4"), ("position", "<i4"), ("target", "<i4"),
        ("target_logit", "<f4"), ("logsumexp", "<f8"),
        ("ids", "<i4", (k,)), ("logits", "<f4", (k,)),
        ("ref_logits", "<f4", (k,)),
    ])


class Run:
    """One collector run: metadata and records sorted by (window, position)."""

    def __init__(self, meta: dict, records: np.ndarray, prefix: str):
        self.meta = meta
        self.records = records
        self.prefix = prefix

    @property
    def top_k(self) -> int:
        return int(self.meta["top_k"])

    def keys(self) -> np.ndarray:
        return (self.records["window"].astype(np.int64) << 32) | self.records["position"]


def load(prefix: str | Path) -> Run:
    prefix = str(prefix)
    meta = json.loads(Path(prefix + ".json").read_text())
    if meta.get("schema") != SCHEMA:
        raise ValueError(f"{prefix}.json: schema {meta.get('schema')!r}, expected {SCHEMA}")
    payload = Path(prefix + ".bin").read_bytes()
    if hashlib.sha256(payload).hexdigest() != meta["bin_sha256"]:
        raise ValueError(f"{prefix}.bin: sha256 differs from its metadata")
    dtype = record_dtype(int(meta["top_k"]))
    if dtype.itemsize != int(meta["record_bytes"]) or len(payload) % dtype.itemsize:
        raise ValueError(f"{prefix}.bin: record size mismatch")
    records = np.frombuffer(payload, dtype=dtype)
    if len(records) != int(meta["records"]):
        raise ValueError(f"{prefix}.bin: {len(records)} records, metadata says {meta['records']}")
    order = np.lexsort((records["position"], records["window"]))
    records = records[order]
    keys = (records["window"].astype(np.int64) << 32) | records["position"]
    if len(keys) > 1 and np.any(keys[1:] == keys[:-1]):
        raise ValueError(f"{prefix}.bin: duplicate (window, position) records")
    return Run(meta, records, prefix)


def write(prefix: str | Path, records: np.ndarray, **meta) -> dict:
    """Write a run with the collector's layout (synthetic tests and tools)."""
    prefix = str(prefix)
    k = records.dtype["ids"].shape[0]
    payload = records.astype(record_dtype(k)).tobytes()
    Path(prefix + ".bin").write_bytes(payload)
    full = {
        "schema": SCHEMA, "top_k": k, "record_bytes": record_dtype(k).itemsize,
        "records": len(records), "bin_sha256": hashlib.sha256(payload).hexdigest(),
        "reference": None,
    }
    full.update(meta)
    Path(prefix + ".json").write_text(json.dumps(full, indent=1) + "\n")
    return full


def records_from_logits(logits: np.ndarray, targets: np.ndarray, windows: np.ndarray,
                        positions: np.ndarray, k: int,
                        reference_ids: np.ndarray | None = None) -> np.ndarray:
    """Build records from full logits [n, vocab] exactly as the collector does.

    This is the Python mirror of rq_logits.cpp Writer::Add, used by the
    synthetic tests; the collector itself never stores full logits.
    """
    logits = np.asarray(logits, dtype=np.float32)
    n, vocab = logits.shape
    if not 1 <= k <= vocab:
        raise ValueError("k outside the vocabulary")
    out = np.zeros(n, dtype=record_dtype(k))
    top = logits.max(axis=1).astype(np.float64)
    lse = top + np.log(np.exp(logits.astype(np.float64) - top[:, None]).sum(axis=1))
    # Descending logit, ties to the lower id: a stable sort of -logit.
    ids = np.argsort(-logits, axis=1, kind="stable")[:, :k].astype(np.int32)
    out["window"] = windows
    out["position"] = positions
    out["target"] = targets
    out["target_logit"] = logits[np.arange(n), targets]
    out["logsumexp"] = lse
    out["ids"] = ids
    out["logits"] = np.take_along_axis(logits, ids, axis=1)
    ref_ids = ids if reference_ids is None else reference_ids
    out["ref_logits"] = np.take_along_axis(logits, ref_ids, axis=1)
    return out


def same_records(old: Run, new: Run) -> dict:
    """Whether every record of `new` is byte-identical to the record of `old` at the same
    (window, position). Used to show a rebuilt collector reproduces an earlier reference run."""
    if old.top_k != new.top_k:
        raise ValueError("top-k differs")
    if old.meta.get("windows_tokens_sha256") != new.meta.get("windows_tokens_sha256"):
        raise ValueError("corpus windows differ")
    okeys, nkeys = old.keys(), new.keys()
    index = np.searchsorted(okeys, nkeys)
    present = (index < len(okeys)) & (okeys[np.minimum(index, len(okeys) - 1)] == nkeys)
    if not present.all():
        raise ValueError(f"{int((~present).sum())} records of the new run are not in the old run")
    a = old.records[index].tobytes()
    b = new.records.tobytes()
    size = new.records.dtype.itemsize
    differ = [i for i in range(len(new.records)) if a[i * size:(i + 1) * size] != b[i * size:(i + 1) * size]]
    return {"records": len(new.records), "identical": len(new.records) - len(differ),
            "first_differences": [{"window": int(new.records[i]["window"]),
                                   "position": int(new.records[i]["position"])} for i in differ[:10]]}


# ---------------------------------------------------------------- llama.cpp --kl-divergence-base
# Mirrors upstream llama.cpp 8ab29fae tools/perplexity/perplexity.cpp: the file perplexity writes
# with --kl-divergence-base and kl_divergence() reads back.
#   "_logits_" | uint32 n_ctx | int32 n_vocab | int32 n_chunk | int32 tokens[n_chunk * n_ctx]
#   then per chunk, rows for positions first .. n_ctx - 2 (first = n_ctx / 2), position p predicting
#   token p + 1, each row nv = 2 * ((n_vocab + 1) / 2) + 4 uint16:
#   float32 scale, float32 min_log_prob, uint16 q[n_vocab] (log p = scale * q + min_log_prob,
#   0 for logits at or below max - 16), zero padding to nv.
# llama.cpp replaces the first token of each chunk with BOS when the vocabulary adds one; Qwen3.8
# does not (tokenizer add_bos_token false), so tokens are written unchanged.
LLAMA_KLD_MAGIC = b"_logits_"


def llama_nearest_int(x: np.ndarray) -> np.ndarray:
    """perplexity.cpp nearest_int: float32 round half to even (valid below 2**22)."""
    return np.rint(np.asarray(x, dtype=np.float32)).astype(np.int64)


def llama_row_width(n_vocab: int) -> int:
    return 2 * ((n_vocab + 1) // 2) + 4


def llama_encode_row(logits: np.ndarray) -> np.ndarray:
    """One base row, as perplexity.cpp log_softmax(n_vocab, logits, uint16 *log_prob, tok)."""
    x = np.asarray(logits, dtype=np.float32)
    n_vocab = len(x)
    max_logit = np.float32(x.max())
    min_logit = np.float32(max(np.float32(x.min()), np.float32(max_logit - np.float32(16))))
    sum_exp = float(np.exp(x - max_logit, dtype=np.float32).astype(np.float64).sum())
    log_sum_exp = np.float32(np.log(sum_exp))
    min_log_prob = np.float32(np.float32(min_logit - max_logit) - log_sum_exp)
    scale = np.float32(np.float32(max_logit - min_logit) / np.float32(65535))
    row = np.zeros(llama_row_width(n_vocab), dtype=np.uint16)
    row[:4] = np.frombuffer(np.array([scale, min_log_prob], dtype="<f4").tobytes(), dtype="<u2")
    if scale:
        inv = np.float32(np.float32(1) / scale)
        q = llama_nearest_int(np.float32(inv) * (x - min_logit))
        row[4:4 + n_vocab] = np.where(x > min_logit, q, 0).astype(np.uint16)
    return row


def write_llama_kld_base(path, n_ctx: int, tokens: np.ndarray, logits: np.ndarray) -> int:
    """tokens [n_chunk, n_ctx] int32; logits [n_chunk, n_ctx - 1 - n_ctx // 2, n_vocab] for
    positions n_ctx // 2 .. n_ctx - 2. Returns the file size in bytes."""
    tokens = np.asarray(tokens, dtype="<i4")
    n_chunk, n_vocab = tokens.shape[0], logits.shape[2]
    rows = n_ctx - 1 - n_ctx // 2
    if tokens.shape != (n_chunk, n_ctx) or logits.shape[:2] != (n_chunk, rows):
        raise ValueError("tokens must be [n_chunk, n_ctx] and logits [n_chunk, n_ctx - 1 - n_ctx/2, n_vocab]")
    with open(path, "wb") as out:
        out.write(LLAMA_KLD_MAGIC)
        out.write(np.array([n_ctx], dtype="<u4").tobytes())
        out.write(np.array([n_vocab, n_chunk], dtype="<i4").tobytes())
        out.write(tokens.tobytes())
        for c in range(n_chunk):
            for r in range(rows):
                out.write(llama_encode_row(logits[c, r]).astype("<u2").tobytes())
    return Path(path).stat().st_size


def llama_kld_base_bytes(n_ctx: int, n_vocab: int, n_chunk: int) -> int:
    return 20 + 4 * n_ctx * n_chunk + 2 * llama_row_width(n_vocab) * (n_ctx - 1 - n_ctx // 2) * n_chunk


def read_llama_kld_base(path) -> dict:
    """Header, tokens and the decoded base log-probabilities [n_chunk, rows, n_vocab] (float32)."""
    raw = Path(path).read_bytes()
    if raw[:8] != LLAMA_KLD_MAGIC:
        raise ValueError("not a llama.cpp --kl-divergence-base file")
    n_ctx = int(np.frombuffer(raw, "<u4", 1, 8)[0])
    n_vocab, n_chunk = (int(v) for v in np.frombuffer(raw, "<i4", 2, 12))
    offset = 20
    tokens = np.frombuffer(raw, "<i4", n_chunk * n_ctx, offset).reshape(n_chunk, n_ctx)
    offset += 4 * n_chunk * n_ctx
    nv, rows = llama_row_width(n_vocab), n_ctx - 1 - n_ctx // 2
    data = np.frombuffer(raw, "<u2", n_chunk * rows * nv, offset).reshape(n_chunk, rows, nv)
    if offset + data.nbytes != len(raw):
        raise ValueError("file size does not match its header")
    head = np.frombuffer(data[:, :, :4].copy().tobytes(), "<f4").reshape(n_chunk, rows, 2)
    logp = head[:, :, :1] * data[:, :, 4:4 + n_vocab].astype(np.float32) + head[:, :, 1:2]
    return {"n_ctx": n_ctx, "n_vocab": n_vocab, "n_chunk": n_chunk, "tokens": tokens, "log_probs": logp}


def llama_kld(base_log_probs: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Per-row KL(base || candidate) as kl_divergence() sums it: base terms with log p > -16 only."""
    x = np.asarray(logits, dtype=np.float32)
    top = x.max(axis=-1, keepdims=True)
    logq = x - top - np.log(np.exp((x - top).astype(np.float64)).sum(axis=-1, keepdims=True))
    keep = base_log_probs > -16
    p = np.where(keep, np.exp(base_log_probs.astype(np.float64)), 0.0)
    return (p * (base_log_probs - logq)).sum(axis=-1)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4 or sys.argv[1] != "same":
        raise SystemExit("usage: logits_io.py same OLD_PREFIX NEW_PREFIX")
    result = same_records(load(sys.argv[2]), load(sys.argv[3]))
    print(json.dumps(result))
    raise SystemExit(0 if result["identical"] == result["records"] else 1)
