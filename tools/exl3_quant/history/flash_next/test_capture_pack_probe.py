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
import capture_pack

probe = Path(__file__).with_name('capture_pack_probe.py')
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

# Original arithmetic executes on CPU. Only Event is mocked, no GPU APIs run.
linear_source = source_path.read_text()
flush_node = next(n for n in ast.parse(linear_source).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'flush_capture_H')
original_ns = dict(torch=torch, _CAP_PROFILE=False, _time=time, FLUSH_STAT={})
exec(compile(ast.Module(body=[flush_node], type_ignores=[]), str(source_path), 'exec'), original_ns)
original = original_ns['flush_capture_H']
assert hashlib.sha256(inspect.getsource(original).encode()).hexdigest() == capture_pack.PINNED_FLUSH_SHA256
profile_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'instrument_original_flush')
profile_ns = dict(torch=torch, inspect=inspect, hashlib=hashlib, original_flush=original,
                  capture_pack=capture_pack, lengths=[3217, 7793, 4655, 3671, 512])
exec(compile(ast.Module(body=[profile_node], type_ignores=[]), str(probe), 'exec'), profile_ns)
events = []
class Event:
    def __init__(self, enable_timing):
        assert enable_timing
        self.index=len(events); events.append(self)
    def record(self):
        pass
    def elapsed_time(self, end):
        assert end.index > self.index
        return float(end.index-self.index)
old_event = torch.cuda.Event
try:
    torch.cuda.Event = Event
    flush, report = profile_ns['instrument_original_flush']()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    for count in (65, 1, 0, 0, 1):
        params=dict(_cap_pending=[(dict(H=torch.zeros(5, 5)),
                    torch.arange(15, dtype=torch.float16).reshape(3, 5)/8) for _ in range(count)])
        if count:
            params['_cap_pending'][0][1][0,0] = float('inf')
        a,b=copy.deepcopy(params),copy.deepcopy(params)
        aa,bb=a['_cap_pending'],b['_cap_pending']
        original(a); flush(b)
        for (ha,xa),(hb,xb) in zip(aa,bb):
            assert torch.equal(ha['H'].view(torch.uint8), hb['H'].view(torch.uint8))
            assert torch.equal(ha['count_dev'], hb['count_dev'])
            assert torch.equal(xa.view(torch.uint8),xb.view(torch.uint8))
    checks.append('instrumentation_preserves_original_CPU_arithmetic')
    assert len(events)==16
    for _ in range(59):
        flush({})
    assert len(events)==16
    checks.append('only_selected_three_flushes_instrumented')
    r=report()
    assert r['rows'][0]['chunks']==2 and r['rows'][1]['chunks']==1
    assert r['rows'][2]['row']==4 and r['rows'][2]['chunks']==1
    assert set(r['totals'].values()) == {4.0}
    checks.append('three_stage_event_intervals_reported')
finally:
    torch.cuda.Event=old_event
result=dict(status='pass', gpu_used=False, checks=checks,
            scope='Actual observer/profile functions extracted by AST; CPU arithmetic, fake CUDA events/frame fixture. No runtime/model/GPU qualification.',
            torch_version=torch.__version__, hip_version=torch.version.hip,
            script_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
            test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
output.parent.mkdir(parents=True,exist_ok=True)
assert not output.exists()
output.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
