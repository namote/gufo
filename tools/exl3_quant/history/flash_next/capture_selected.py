"""Diagnostic-only prefix capture; stop before a selected terminal expert down matmul.

The original block runs unchanged through attention and the selected expert gate/up operations. Keep
only specified Linear.capture_H calls; abort after capturing the terminal down input.
This is NOT a faster full converter and never supplies a block output.
"""
from contextlib import contextmanager
from exllamav3.modules import Linear

class PrefixCaptured(BaseException):
    pass

@contextmanager
def selected_capture(selected, terminal_key, inputs=None):
    assert terminal_key in selected
    original = Linear.capture_H
    seen = []
    def capture(linear, x, params):
        if linear.key not in selected:
            return
        assert linear.key not in seen
        seen.append(linear.key)
        if inputs is not None:
            inputs[linear.key] = x.detach().clone()
        original(linear, x, params)
        if linear.key == terminal_key:
            assert set(seen) == set(selected), 'A requested projection occurs after the stop point'
            raise PrefixCaptured()
    Linear.capture_H = capture
    try:
        yield
        assert set(seen) == set(selected)
    finally:
        Linear.capture_H = original
