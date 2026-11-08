"""CPU-only, pinned-source oracle checks for the uninstalled direct FP32 pack candidate."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from capture_lt_chunk import make_flush
from types import SimpleNamespace


source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
source = source_path.read_text()
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name == 'flush_capture_H')
namespace = dict(torch=torch, _CAP_PROFILE=False, _time=time,
                 FLUSH_STAT=dict(wall=0.0, calls=0))
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), 'exec'), namespace)
baseline = namespace['flush_capture_H']
dispatches=[]
def fake_gram(xf,xs,finite):
    assert xf.dtype==torch.float32 and torch.isfinite(xf).all()
    assert torch.equal(xf.masked_select(~finite.unsqueeze(2).expand_as(xf)),torch.zeros_like(xf.masked_select(~finite.unsqueeze(2).expand_as(xf))))
    dispatches.append((tuple(xf.shape),xs.dtype))
    return torch.bmm(xf.transpose(1,2),xf)
candidate = make_flush(baseline,fake_gram,128)
torch.set_grad_enabled(False)
torch.set_num_threads(2)
torch.manual_seed(317)
checks = []


def byte_equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.uint8).reshape(-1),
                            right.contiguous().view(torch.uint8).reshape(-1)))


def fixture(count, rows=7, dim=5, dtype=torch.float32):
    entries = []
    for i in range(count):
        # Dyadic values also support an independent FP64 sum-of-outer-products oracle.
        x = torch.randint(-8, 9, (rows, dim)).to(dtype) / 8
        hd = dict(H=torch.eye(dim, dtype=torch.float32) / 4, count=0,
                  num_total=x.numel(), inf_nan=torch.zeros(2, dtype=torch.long),
                  first_key=f'linear.{i}')
        if i % 2:
            hd['count_dev'] = torch.tensor(11, dtype=torch.long)
        entries.append((hd, x))
    return dict(_cap_pending=entries, untouched='preserve')


def evaluate(name, params, independent=True):
    left = copy.deepcopy(params)
    right = copy.deepcopy(params)
    left_entries = left['_cap_pending'][:]
    right_entries = right['_cap_pending'][:]
    expected = {}
    if independent:
        for hd, x in right_entries:
            key = id(hd)
            if key not in expected:
                expected[key] = [hd['H'].double().clone(),
                                 int(hd.get('count_dev', torch.tensor(0)))]
            good = x[torch.isfinite(x).all(dim=1)].double()
            for row in good:
                expected[key][0] += row[:, None] * row[None, :]
            expected[key][1] += len(good)
    input_copies = [x.clone() for _, x in right_entries]
    metadata = [(hd['num_total'], hd['count'], hd['first_key'], hd['inf_nan'].clone())
                for hd, _ in right_entries]
    baseline(left)
    candidate(right)
    assert '_cap_pending' not in left and '_cap_pending' not in right
    assert left['untouched'] == right['untouched'] == 'preserve'
    for (lh, lx), (rh, rx), original, meta in zip(
            left_entries, right_entries, input_copies, metadata):
        assert byte_equal(lh['H'], rh['H']), name
        assert byte_equal(lh['count_dev'].reshape(1), rh['count_dev'].reshape(1)), name
        assert byte_equal(lx, rx) and byte_equal(rx, original), name
        assert (rh['num_total'], rh['count'], rh['first_key']) == meta[:3]
        assert byte_equal(rh['inf_nan'], meta[3])
        if independent:
            target, count = expected[id(rh)]
            assert torch.equal(rh['H'].double(), target), name
            assert int(rh['count_dev']) == count, name
    checks.append(dict(name=name, exact=True))
    return right


for entries in (1, 64, 65, 128, 129):
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        evaluate(f'entries{entries}-{dtype}', fixture(entries, dtype=dtype))

ragged = fixture(3, rows=1, dim=3)
ragged['_cap_pending'] += fixture(65, rows=9, dim=7, dtype=torch.float16)['_cap_pending']
ragged['_cap_pending'] += fixture(2, rows=33, dim=1)['_cap_pending']
evaluate('mixed_shapes_dtypes_awkward_tails', ragged)

nonfinite = fixture(65)
for index, value in enumerate((float('nan'), float('inf'), -float('inf'))):
    nonfinite['_cap_pending'][index][1][index, 1] = value
nonfinite['_cap_pending'][3][1][:] = float('nan')
evaluate('nonfinite_rows_and_all_nonfinite_entry', nonfinite)

duplicate = fixture(65)
duplicate['_cap_pending'][1] = (duplicate['_cap_pending'][0][0], duplicate['_cap_pending'][1][1])
duplicate['_cap_pending'][64] = (duplicate['_cap_pending'][0][0], duplicate['_cap_pending'][64][1])
evaluate('duplicate_hd_within_and_across_chunks', duplicate)

for alias in ('H', 'count_dev'):
    params = fixture(3)
    first, second = [entry[0] for entry in params['_cap_pending'][:2]]
    if alias == 'count_dev':
        first[alias] = torch.tensor(3, dtype=torch.long)
    second[alias] = first[alias]
    evaluate(f'distinct_hd_shared_{alias}', params, independent=False)

views = fixture(2)
storage = torch.zeros(26, dtype=torch.float32)
views['_cap_pending'][0][0]['H'] = storage[:25].view(5, 5)
views['_cap_pending'][1][0]['H'] = storage[1:26].view(5, 5)
evaluate('overlapping_H_views', views, independent=False)

noncontiguous = fixture(2)
for hd, _ in noncontiguous['_cap_pending']:
    hd['H'] = torch.arange(25, dtype=torch.float32).view(5, 5).T
evaluate('noncontiguous_H', noncontiguous)

for dtype in (torch.float16, torch.bfloat16, torch.float32):
    params = fixture(65, rows=9, dim=7, dtype=dtype)
    for i, (hd, x) in enumerate(params['_cap_pending']):
        backing = torch.empty((9, 14), dtype=dtype)
        backing[:, ::2].copy_(x)
        x = backing[:, ::2]
        x[0, 0] = -0.0
        x[0, 1] = 0.0
        if i == 0:
            x[1, 0] = float('nan')
            x[2, 0] = float('inf')
            x[3, 0] = -float('inf')
        assert not x.is_contiguous()
        params['_cap_pending'][i] = (hd, x)
    evaluate(f'noncontiguous_signedzero_nonfinite_{dtype}', params)

for params in ({}, {'_cap_pending': []}):
    assert candidate(params) is None and '_cap_pending' not in params
checks.append(dict(name='missing_empty_pending', exact=True))
assert dispatches
assert any(shape[0]==128 for shape,dtype in dispatches)
result = dict(status='pass', gpu_used=False, production_installed=False,
              torch_version=torch.__version__, hip_version=torch.version.hip, checks=checks,
              baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
              baseline_function_sha256=hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest(),
              candidate_sha256=hashlib.sha256(Path(__file__).with_name('capture_lt_chunk.py').read_bytes()).hexdigest(),
              test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
