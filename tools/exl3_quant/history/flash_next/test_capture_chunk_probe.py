"""CPU-only tests of AST-extracted observer and event-instrumented arithmetic."""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import torch
import capture_chunk

probe = Path(__file__).with_name('capture_chunk_probe.py')
source = probe.read_text()
tree = ast.parse(source)
source_path = Path(sys.argv[1])
output = Path(sys.argv[2])
checks = []
node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'CaptureTimer')
classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CaptureFinished']
forbidden = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'forbidden')
trace = []
collected = {}
store = object()
rng = torch.tensor([3, 7], dtype=torch.uint8)
class Base:
    text = ' -- Capturing: fixture'
    count = 64
    def __enter__(self):
        trace.append('enter')
        return self
    def __exit__(self, *args):
        trace.append(('exit', args[0]))
        return False
fake_torch = SimpleNamespace(equal=torch.equal, is_inference_mode_enabled=lambda: True,
    is_grad_enabled=lambda: False, random=SimpleNamespace(get_rng_state=lambda: rng),
    cuda=SimpleNamespace(get_rng_state=lambda dev: rng, synchronize=lambda: trace.append('sync')))
ns = dict(base_bar=Base, sys=sys, torch=fake_torch, time=time, collected=collected,
          indices=list(range(64)), target=9, stores=[store], cpu_tree=lambda x: x,
          case_cpu_rng=rng, case_gpu_rng=rng)
exec(compile(ast.Module(body=classes+[forbidden,node], type_ignores=[]), str(probe), 'exec'), ns)
exec('''def fixture_main(fail=False):
    idx=9
    bad_rows=set()
    state=SimpleNamespace(store=stores[0])
    capture_H={str(i): i for i in range(516)}
    ref_states=list(range(5))
    quant_preserves=[]
    original_input_ids=[]
    with CaptureTimer():
        if fail:
            raise ValueError('capture failed')
    forbidden()
''', dict(ns, SimpleNamespace=SimpleNamespace), ns)
# The function owns its execution globals; install its precise caller identity.
main = ns['fixture_main']; main.__globals__['main_body'] = main
ns['main_body'] = main
try:
    main()
except ns['CaptureFinished']:
    pass
else:
    raise AssertionError('Capture did not stop')
assert collected['numeric']['capture_H']['515'] == 515
checks.append('actual_frame_snapshot_stops_before_quant')
assert trace.count('sync') == 2
checks.append('only_two_sync_boundaries')
assert trace[1] == 'enter' and trace[-2] == ('exit', None)
checks.append('base_bar_delegation')
collected.clear(); trace.clear()
try:
    main(True)
except ValueError:
    pass
else:
    raise AssertionError('Capture error swallowed')
assert not collected and trace.count('sync') == 1
checks.append('capture_error_propagates_without_success_snapshot')
try:
    ns['forbidden']()
except AssertionError:
    pass
else:
    raise AssertionError('Write guard did not stop')
checks.append('quant_write_guard_fails_closed')

result=dict(status='pass',gpu_used=False,checks=checks,script_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
