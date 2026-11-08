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
import capture_lt

probe = Path(__file__).with_name('capture_lt_chunk_probe.py')
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
    cuda=SimpleNamespace(get_rng_state=lambda dev: rng, synchronize=lambda: trace.append('sync')),save=lambda *args:None)
ns = dict(base_bar=Base, sys=sys, torch=fake_torch, time=time, collected=collected,
          indices=list(range(64)), target=9, stores=[store], cpu_tree=lambda x: x,
          case_cpu_rng=rng, case_gpu_rng=rng,case_dir=Path('/unused'),observer=False,immediate={})
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

diff_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='h_difference')
diff_ns={'torch':torch}
exec(compile(ast.Module(body=[diff_node],type_ignores=[]),str(probe),'exec'),diff_ns)
a=torch.tensor([[1.,2.],[3.,4.]])
base_stats={'scopes':64}
same=diff_ns['h_difference'](a,a.clone());different=diff_ns['h_difference'](a,a+1)
assert same['exact'] and same['relative_l2']==0
assert not different['exact'] and different['max_abs']==1
assert abs(different['relative_l2']-2/(30**.5))<1e-12
json.dumps({'skip_stats':base_stats,'H_differences':different},allow_nan=False)
assert base_stats=={'scopes':64}
checks.append('H_difference_numeric_JSON_and_skip_stats_preserved')
# User-approved numeric evaluation is separate from immutable data contracts.
numeric_nodes=[n for n in tree.body if getattr(n,'name',None) in ('compare_numeric','numeric_difference','exact','h_difference')]
num_ns={'torch':torch}
exec(compile(ast.Module(body=numeric_nodes,type_ignores=[]),str(probe),'exec'),num_ns)
base=dict(capture_H={'x':{'H':torch.eye(2),'count_dev':torch.tensor(4)}},ref_states={0:torch.ones(2,2)},original_input_ids=[torch.tensor([1,2])],cpu_rng=rng)
candidate=copy.deepcopy(base);candidate['capture_H']['x']['H'][0,0]+=.01;candidate['ref_states'][0][0,0]+=.01
report=num_ns['compare_numeric'](base,candidate)
assert not report['H_exact'] and not report['refs_exact'] and report['max_H_relative_l2']>0
assert torch.equal(base['capture_H']['x']['H'],torch.eye(2))
checks.append('numeric_differences_reported_not_automatic_quality_failure')
for key,mutate in [
 ('count',lambda c:c['capture_H']['x']['count_dev'].add_(1)),
 ('input',lambda c:c['original_input_ids'][0].add_(1)),
 ('nonfinite',lambda c:c['ref_states'][0].fill_(float('nan'))),
 ('shape',lambda c:c['capture_H']['x'].update(H=torch.ones(3,3))),
 ('dtype',lambda c:c['ref_states'].update({0:torch.ones(2,2,dtype=torch.float64)})),
]:
 bad=copy.deepcopy(base);mutate(bad)
 try:num_ns['compare_numeric'](base,bad)
 except AssertionError:pass
 else:raise AssertionError(('integrity failure accepted',key))
 checks.append('reject_'+key)
# In-run reference mutation remains a hard failure.
try:num_ns['exact'](base['ref_states'],candidate['ref_states'],'reference_lifetime')
except AssertionError:pass
else:raise AssertionError('reference mutation not rejected')
checks.append('reference_lifetime_remains_exact')
result=dict(status='pass',gpu_used=False,checks=checks,script_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
