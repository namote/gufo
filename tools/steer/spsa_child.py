"""Run spsa.py's main() with the gufo API key (environment variable
STEER_GUFO_API_KEY, set by training.py) sent as a bearer token. spsa.py
itself is unchanged; only its request helper is wrapped. On Linux the child
gets SIGTERM when the tool (STEER_PARENT_PID) dies, so no orphaned spsa.py
keeps scoring against gufo."""
import os
from pathlib import Path
import signal
import sys
import urllib.request

PARENT = os.environ.pop('STEER_PARENT_PID', None)
if PARENT:
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass
    if os.getppid() != int(PARENT):  # the parent died before prctl
        sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'qwen38_flash_next' / 'abliteration'))
import spsa  # noqa: E402

KEY = os.environ.pop('STEER_GUFO_API_KEY', None)
if KEY:
    _Request = urllib.request.Request

    class _AuthRequest(_Request):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_header('Authorization', f'Bearer {KEY}')

    urllib.request.Request = _AuthRequest

if __name__ == '__main__':
    sys.exit(spsa.main(sys.argv[1:]))
