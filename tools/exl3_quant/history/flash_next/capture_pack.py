"""Uninstalled experiment: write a capture chunk directly into an FP32 stack.

Only the two stack/cast statements change. The pinned function supplies grouping,
chunk64, finite masking, BMM, ordered H/count updates and profiling unchanged.
CPU equality is not GPU qualification or evidence of a speed improvement.
"""
import hashlib
import inspect

PINNED_FLUSH_SHA256 = '3d167e36fefeb8a9e3bcd81adaea3dd235af448c3d6b3c3a2a8ea24d2dd33761'
OLD = '''            xs = torch.stack([e[1] for e in chunk])
            xf = xs.to(torch.float)
'''
NEW = '''            xf = torch.empty((len(chunk), *chunk[0][1].shape),
                             dtype=torch.float, device=chunk[0][1].device)
            torch.stack([e[1] for e in chunk], out=xf)
'''


def make_flush_capture_H(original):
    """Verify the exact pinned body, then change only stack materialization.

This follows the converter's inference-only contract. torch.stack(out=...) does
not support autograd; callers must not use this candidate for training.
"""
    source = inspect.getsource(original)
    assert hashlib.sha256(source.encode()).hexdigest() == PINNED_FLUSH_SHA256, \
        'Unreviewed capture flush source'
    assert source.count(OLD) == 1
    namespace = {}
    # Use original globals so profile flags/counters retain their live behavior.
    exec(compile(source.replace(OLD, NEW), '<capture-pack-candidate>', 'exec'),
         original.__globals__, namespace)
    candidate = namespace['flush_capture_H']
    candidate.original = original
    candidate.source_sha256 = PINNED_FLUSH_SHA256
    return candidate


def install(cm):
    """Explicit diagnostic installation; return an idempotent restore function."""
    original = cm._flush_capture_H
    candidate = make_flush_capture_H(original)
    cm._flush_capture_H = candidate
    restored = False

    def restore():
        nonlocal restored
        if restored:
            return
        assert cm._flush_capture_H is candidate, 'Capture flush changed after installation'
        cm._flush_capture_H = original
        restored = True

    return restore
