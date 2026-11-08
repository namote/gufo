"""Uninstalled candidate: change only same-shape flush chunk64 to chunk128.

Keep all calibration tokens, original FP32 arithmetic, group and H-add order.
Larger BMM batch may select different kernels; strict GPU parity is required.
"""
import hashlib
import inspect

PINNED_FLUSH_SHA256 = '3d167e36fefeb8a9e3bcd81adaea3dd235af448c3d6b3c3a2a8ea24d2dd33761'
OLD = 'for s in range(0, len(entries), 64):'
NEW = 'for s in range(0, len(entries), 128):'
OLD_SLICE = 'chunk = entries[s:s + 64]'
NEW_SLICE = 'chunk = entries[s:s + 128]'


def make_flush_capture_H(original):
    """Pin original source and change only chunk width, never sample count."""
    source = inspect.getsource(original)
    assert hashlib.sha256(source.encode()).hexdigest() == PINNED_FLUSH_SHA256, \
        'Unreviewed capture flush source'
    assert source.count(OLD) == source.count(OLD_SLICE) == 1
    namespace = {}
    # Use original globals so profile flags/counters retain their live behavior.
    exec(compile(source.replace(OLD, NEW).replace(OLD_SLICE, NEW_SLICE), '<capture-chunk128-candidate>', 'exec'),
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
