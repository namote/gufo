"""GPU owner lock and quiet-host checks for the conversion drivers.

The c03 run imported these from ai.tools
(test.experiments.gufo_upstream_port_20261023.window). The same host(), npu()
and quiet() now live in the gufo server's ops package (paths.OPS), whose
OWNER.lock is the shared GPU owner lock. It is imported lazily, so CPU tests
need no ops tree.
"""
import sys

from paths import OPS

LOCK = OPS / 'OWNER.lock'


def _window():
    if str(OPS.parent) not in sys.path:
        sys.path.insert(0, str(OPS.parent))
    from ops import window
    return window


def host():
    return _window().host()


def npu():
    return _window().npu()


def quiet(h):
    return _window().quiet(h)
