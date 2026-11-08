"""Diagnostic input-only observer; never accumulates H or returns a block output."""
from contextlib import contextmanager
from exllamav3.modules import Linear

class InputsCaptured(BaseException):
    pass

@contextmanager
def capture_inputs(keys, terminal, inputs):
    assert terminal in keys and not inputs
    original=Linear.capture_H
    def observe(linear,x,params):
        if linear.key not in keys:return
        assert linear.key not in inputs
        inputs[linear.key]=x.detach().clone()
        if linear.key==terminal:
            assert set(inputs)==set(keys)
            raise InputsCaptured()
    Linear.capture_H=observe
    try:
        yield
        assert set(inputs)==set(keys)
    finally:Linear.capture_H=original
