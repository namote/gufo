"""Replay retained packed modules with the pinned converter's state-advance code."""
import hashlib,inspect
from capture_fast import _MAIN_SHA
START='        # Collect output tensors\n'
END='        # Reload module from memory\n'
def replay_source(original):
    source=inspect.getsource(inspect.unwrap(original))
    assert hashlib.sha256(source.encode()).hexdigest()==_MAIN_SHA
    assert source.count(START)==source.count(END)==1
    a=source.index(START);b=source.index(END)
    replacement='''        assert 2 <= idx <= 5, 'Replay is limited to healthy packed prefix'
        assert module.num_slices == 1
        q_tensors = _gufo_load_packed(module)
        capture_H = None
        ref_states = {}
        linears = []  # No new quality score: no unquantized reference is recomputed.
        final_bpw = None  # Replay does not compute a new bitrate/quality score.
'''
    return source[:a]+replacement+source[b:]
def make_replay(original,load_packed):
    source=replay_source(original)
    namespace=dict(inspect.unwrap(original).__globals__,_gufo_load_packed=load_packed)
    exec(compile(source,'<gufo-packed-prefix-replay>','exec'),namespace)
    return namespace['main']
