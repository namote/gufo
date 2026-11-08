"""One retained matched baseline, two capture candidates, two real block types."""
import copy
import json
import os
from pathlib import Path
import time
import torch
from exllamav3 import Config, Model
from exllamav3.conversion import convert_model as cm
from exllamav3.loader.safetensors_alt import safe_open
from exllamav3.modules.linear import flush_capture_H
from exllamav3.modules.quant import fp16, exl3
from disk_state import _read_row
from capture_fast import install, _discarded_capture, _active, STATS
from capture_foreach import make_flush_capture_H
from contextlib import nullcontext

torch.set_grad_enabled(False)
torch.manual_seed(0)
assert os.environ.get('EXL3_HIP_F32OUT_VIA_F16') == '0'
assert not fp16._f32_via_f16 and not exl3._f32_via_f16
out = Path('/output/capture-speed')
assert not (out / 'SUMMARY.json').exists(), 'Reuse qualified evidence'
config = Config.from_directory('/source')
model = Model.from_config(config)
foreach = make_flush_capture_H(flush_capture_H)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def exact(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype, path
        av = a.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        bv = b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        assert torch.equal(av, bv), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            exact(a[k], b[k], path + '.' + str(k))
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, path + '.' + str(i))
    else:
        assert a == b, (path, a, b)


with safe_open('/task/c03-input-ids.safetensors') as reader:
    # These are the same full ragged rows used in the existing 8K pilot.
    print('calibration_keys', list(reader.keys())[:3], flush=True)
    token_ids = {i: _read_row(reader, f'input_ids.{i}') for i in (4, 364)}

results = []
started = time.monotonic()
for module_index in (2, 6):
    module = model.modules[module_index]
    inputs = {}
    if module_index == 2:
        source = '/output/model-job-c03/repair-002-checkpoint/checkpoint-idx2/state.safetensors'
        with safe_open(source) as reader:
            inputs = {i: _read_row(reader, f'tensor.{i}') for i in (4, 364)}
    else:
        for i in (4, 364):
            with safe_open(str(out / f'attention-input-{i}.safetensors')) as reader:
                inputs[i] = _read_row(reader, 'tensor')
    defer = module.can_defer_load()
    if defer:
        config.stc.begin_deferred_load()
    try:
        module.load(torch.device('cuda:0'))
    finally:
        if defer:
            config.stc.end_deferred_load()
    config.stc.close()
    samples = [(4, 33), (4, 512), (364, 8192)]
    baseline_h = out / f'baseline-{module_index}-H.pt'
    baseline_y = out / f'baseline-{module_index}-normal.pt'
    assert not baseline_h.exists(), 'Saved baseline exists; do not overwrite/rerun'
    restore = install(cm, automatic=False)
    base_normal = []
    base_preserves = []
    baseline_small_capture = None
    for mode in ('baseline', 'skip_down', 'skip_down_foreach'):
        H, timings = {}, []
        for row_index, length in samples:
            x = inputs[row_index][:, :length].contiguous().cuda()
            original = x.clone()
            params = dict(input_ids=token_ids[row_index][:, :length], attn_mode='flash_attn_nc',
                          quant_preserve={}, capture=H, activate_all_experts=True)
            preserved = cpu_tree(params['quant_preserve'])
            scope = _discarded_capture(module, params) if mode != 'baseline' else nullcontext()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with scope:
                y = module.forward(module.prepare_for_device(x, params), params)
            (foreach if mode == 'skip_down_foreach' else flush_capture_H)(params)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            assert _active.get() is None
            assert torch.isfinite(y).all()
            exact(original, x, 'input')
            exact(preserved, params['quant_preserve'], 'preserve')
            assert 'hc_pending' not in params
            if mode == 'baseline' and length == 33:
                baseline_small_capture = y.cpu()
            del y
            normal = dict(input_ids=token_ids[row_index][:, :length],
                          attn_mode='flash_attn_nc', quant_preserve={})
            y = module.forward(module.prepare_for_device(x, normal), normal)
            assert torch.isfinite(y).all()
            exact(original, x, 'normal-input')
            if mode == 'baseline':
                base_normal.append(y.cpu())
                base_preserves.append(cpu_tree(normal['quant_preserve']))
            else:
                j = len(timings)
                exact(base_normal[j], y, 'normal-output')
                exact(base_preserves[j], normal['quant_preserve'], 'normal-preserve')
            timings.append(dict(row=row_index, length=length, seconds=elapsed))
            print(module_index, mode, length, elapsed, flush=True)
            del x, original, y
        for hd in H.values():
            assert torch.isfinite(hd['H']).all() and int(hd['inf_nan'].sum()) == 0
        if mode == 'baseline':
            torch.save(cpu_tree(H), baseline_h)
            torch.save(dict(outputs=base_normal, preserves=base_preserves), baseline_y)
        else:
            reference = torch.load(baseline_h, map_location='cpu', weights_only=True)
            exact(reference, H, 'all-H-keys-counters')
            del reference
        results.append(dict(module=module.key, index=module_index, mode=mode,
                            hessians=len(H), timings=timings, exact=True))
        del H
        (out / 'PROGRESS.json').write_text(json.dumps(results, indent=2))
    # A consumer of all-expert output outside the private scope must get the real output.
    x = inputs[4][:, :33].contiguous().cuda()
    params = dict(input_ids=token_ids[4][:, :33], attn_mode='flash_attn_nc',
                  quant_preserve={}, capture={}, activate_all_experts=True)
    y = module.forward(module.prepare_for_device(x, params), params)
    exact(baseline_small_capture, y, 'unscoped-all-expert-output')
    flush_capture_H(params)
    try:
        with _discarded_capture(module, params):
            raise ValueError('scope-reset-test')
    except ValueError:
        pass
    assert _active.get() is None
    for bad in ({'export_state_layers': [module.layer_idx]}, {'hc_pending': None}, {'q_mlp_slice': 0}):
        try:
            with _discarded_capture(module, {**params, **bad}):
                raise RuntimeError('Unsupported contract was accepted')
        except AssertionError:
            pass
    restore()
    del x, y, params, base_normal, base_preserves, baseline_small_capture, inputs
    module.unload()
    torch.cuda.empty_cache()
assert STATS['skipped_down_calls'] > 0 and foreach.stats['foreach_chunks'] > 0
summary = dict(status='capture_exact_pass', results=results, skip_stats=STATS,
               foreach_stats=foreach.stats, elapsed_seconds=time.monotonic()-started,
               peak_gpu_bytes=torch.cuda.max_memory_allocated(),
               calibration_sha256=json.loads(Path('/task/c03-input-ids.json').read_text())['sha256'],
               fp32_output_shortcut=False,
               scope='Affected capture only; baseline once retained; first pass of each shape may include allocation/warmup; speed provisional until full conversion')
(out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
