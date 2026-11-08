"""Opt-in experiment: batch only the additions in the pinned capture flush.

The producer, chunk size (64), FP32 bmm, finite-row masking and row order stay
unchanged. No import-time installation; the caller must explicitly install the
returned function after qualifying it. Aliased destinations use the original
ordered additions, including repeated Hessian dictionaries.
"""
import time
import torch


def _independent_destinations(tensors):
    """Conservative alias guard: distinct contiguous storage, not just views."""
    seen = set()
    for tensor in tensors:
        if not tensor.is_contiguous() or tensor.numel() == 0:
            return False
        identity = (tensor.device, tensor.untyped_storage().data_ptr())
        if identity in seen:
            return False
        seen.add(identity)
    return True


def make_flush_capture_H(original):
    """Construct a drop-in candidate while retaining upstream profile counters."""
    upstream = original.__globals__
    stats = dict(foreach_chunks=0, ordered_chunks=0)

    def flush_capture_H(params):
        profiling = upstream.get('_CAP_PROFILE', False)
        started = time.perf_counter() if profiling else None
        pending = params.pop('_cap_pending', None)
        if not pending:
            return
        shape_groups = {}
        order = []
        for hd, x in pending:
            key = (x.shape, x.dtype, x.device)
            if key not in shape_groups:
                shape_groups[key] = []
                order.append(key)
            shape_groups[key].append((hd, x))
        for key in order:
            entries = shape_groups[key]
            for start in range(0, len(entries), 64):
                chunk = entries[start:start + 64]
                xs = torch.stack([entry[1] for entry in chunk])
                xf = xs.to(torch.float)
                finite = torch.isfinite(xf).all(dim=2)
                xf = xf.masked_fill_(~finite.unsqueeze(2), 0.0)
                products = torch.bmm(xf.transpose(1, 2), xf)
                finite_counts = finite.sum(dim=1)
                hessians, counts = [], []
                for hd, _ in chunk:
                    hessians.append(hd['H'])
                    count = hd.get('count_dev')
                    if count is None:
                        count = torch.zeros((), dtype=torch.long, device=hd['H'].device)
                        hd['count_dev'] = count
                    counts.append(count)
                # Different storage is deliberately stricter than disjoint slices.
                # A repeated H or count must preserve the original per-entry order.
                eligible = (
                    all(h.dtype == torch.float and h.device == products.device
                        and h.shape == products.shape[1:] for h in hessians)
                    and all(c.dtype == torch.long and c.device == products.device
                            and c.shape == torch.Size([]) for c in counts)
                    and _independent_destinations(hessians + counts)
                )
                if eligible:
                    torch._foreach_add_(hessians, list(products.unbind(0)))
                    torch._foreach_add_(counts, list(finite_counts.unbind(0)))
                    stats['foreach_chunks'] += 1
                else:
                    for index, (hessian, count) in enumerate(zip(hessians, counts)):
                        hessian.add_(products[index])
                        count.add_(finite_counts[index])
                    stats['ordered_chunks'] += 1
                xs = xf = products = finite_counts = None
        if started is not None:
            upstream['FLUSH_STAT']['wall'] += time.perf_counter() - started
            upstream['FLUSH_STAT']['calls'] += 1

    flush_capture_H.stats = stats
    return flush_capture_H
