"""Pinned Lt flush, varying only same-shape matrix batch width."""
import hashlib
import inspect
from capture_lt import FLUSH_SHA

def make_flush(original, gram, chunk):
    assert type(chunk) is int and chunk in (64, 128)
    source = inspect.getsource(original)
    assert hashlib.sha256(source.encode()).hexdigest() == FLUSH_SHA
    replacements = {
        'P = torch.bmm(xf.transpose(1, 2), xf)': 'P = _gufo_gram(xf, xs, finite)',
        'for s in range(0, len(entries), 64):': f'for s in range(0, len(entries), {chunk}):',
        'chunk = entries[s:s + 64]': f'chunk = entries[s:s + {chunk}]',
    }
    for old, new in replacements.items():
        assert source.count(old) == 1
        source = source.replace(old, new)
    namespace = dict(original.__globals__, _gufo_gram=gram)
    exec(compile(source, '<capture-lt-chunk>', 'exec'), namespace)
    return namespace['flush_capture_H']
