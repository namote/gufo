"""Build qwen27b_rq_logits records (logits_io.v1) from another engine's full logits.

The records must match what tests/models/qwen27b/rq_logits.cpp writes for the same logits, so
metrics.py can pair an ExLlamaV3 run with the gufo BF16 reference: top-k descending with ties to
the lower id, float64 logsumexp over the full vocabulary, and the run's logits at the reference
run's top-k ids (`ref_logits`). `select_top_k` turns an over-wide, arbitrarily tie-ordered
top-(k + pad) candidate set (torch.topk) into that exact order, falling back to a full stable sort
for a row whose tie group may reach past the candidates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logits_io  # noqa: E402

CANDIDATE_PAD = 64


def select_top_k(cand_ids: np.ndarray, cand_vals: np.ndarray, k: int,
                 full_row=None) -> tuple[np.ndarray, np.ndarray]:
    """cand_ids/cand_vals [n, k + pad] hold each row's k + pad largest logits in any order.
    Returns ids/logits [n, k] in descending logit, ties to the lower id. `full_row(i)` returns
    row i's full logits; it is called only when the k-th value also ends the candidate set (its
    tie group may continue outside the candidates)."""
    n, width = cand_ids.shape
    if width <= k:
        raise ValueError("the candidate set must be wider than k")
    ids = np.empty((n, k), dtype=np.int32)
    vals = np.empty((n, k), dtype=np.float32)
    for i in range(n):
        order = np.lexsort((cand_ids[i], -cand_vals[i].astype(np.float64)))
        v = cand_vals[i][order]
        if v[k - 1] == v[-1]:
            if full_row is None:
                raise ValueError(f"row {i}: tie group reaches past the candidates")
            row = np.asarray(full_row(i), dtype=np.float32)
            top = np.argsort(-row, kind="stable")[:k]
            ids[i], vals[i] = top, row[top]
            continue
        ids[i] = cand_ids[i][order[:k]]
        vals[i] = v[:k]
    return ids, vals


def assemble(windows: np.ndarray, positions: np.ndarray, targets: np.ndarray,
             target_logits: np.ndarray, logsumexp: np.ndarray, ids: np.ndarray,
             logits: np.ndarray, ref_logits: np.ndarray) -> np.ndarray:
    k = ids.shape[1]
    out = np.zeros(len(windows), dtype=logits_io.record_dtype(k))
    out["window"] = windows
    out["position"] = positions
    out["target"] = targets
    out["target_logit"] = target_logits
    out["logsumexp"] = logsumexp
    out["ids"] = ids
    out["logits"] = logits
    out["ref_logits"] = ref_logits
    return out


def records_numpy(logits: np.ndarray, targets, windows, positions, k: int,
                  reference_ids: np.ndarray | None = None) -> np.ndarray:
    """The candidate path on full numpy logits (CPU tests): argpartition gives an unordered
    top-(k + pad) set like torch.topk, then select_top_k orders it."""
    logits = np.asarray(logits, dtype=np.float32)
    n, vocab = logits.shape
    width = min(vocab, k + CANDIDATE_PAD)
    cand = np.argpartition(-logits, width - 1, axis=1)[:, :width]
    ids, vals = select_top_k(cand, np.take_along_axis(logits, cand, axis=1), k,
                             full_row=lambda i: logits[i])
    top = logits.max(axis=1).astype(np.float64)
    lse = top + np.log(np.exp(logits.astype(np.float64) - top[:, None]).sum(axis=1))
    ref = ids if reference_ids is None else reference_ids
    return assemble(np.asarray(windows), np.asarray(positions), np.asarray(targets),
                    logits[np.arange(n), targets], lse, ids, vals,
                    np.take_along_axis(logits, ref, axis=1))


def records_torch(logits, targets, windows, positions, k: int, reference_ids: list) -> tuple:
    """GPU path: logits is a torch tensor [n, vocab] (any float dtype, upcast to float32 first,
    the values the records keep). reference_ids is a list of [n, k] int arrays (None entries use
    the run's own ids). Returns (base records with ref_logits = own ids, [ref_logits per
    reference])."""
    import torch
    x = logits.float()
    n = x.shape[0]
    width = min(x.shape[1], k + CANDIDATE_PAD)
    cand_vals, cand_ids = torch.topk(x, width, dim=1, largest=True, sorted=False)
    lse = torch.logsumexp(x.double(), dim=1).cpu().numpy()
    t = torch.as_tensor(np.asarray(targets, dtype=np.int64), device=x.device)
    target_logits = x.gather(1, t[:, None])[:, 0].cpu().numpy()
    ids, vals = select_top_k(cand_ids.cpu().numpy().astype(np.int32), cand_vals.cpu().numpy(), k,
                             full_row=lambda i: x[i].cpu().numpy())
    refs = []
    for ref in reference_ids:
        r = ids if ref is None else np.asarray(ref, dtype=np.int64)
        g = torch.as_tensor(r.astype(np.int64), device=x.device)
        refs.append(x.gather(1, g).cpu().numpy())
    base = assemble(np.asarray(windows), np.asarray(positions), np.asarray(targets),
                    target_logits, lse, ids, vals, vals)
    return base, refs
