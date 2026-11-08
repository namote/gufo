"""New matched diagnostic controls, not a repeat of the earlier mismatched probe.

Run only at an owned natural GPU boundary. Original functions vs skip-down, under
real inference_mode, with identical saved accumulated H and calibration inputs.
CPU state read, H2D, forward, and H flush are separately synchronized/timed.
Initial H upload, validation, normal forwards, and artifact writes are NOT timed.
"""
import copy
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from exllamav3 import Config, Model
from exllamav3.conversion import convert_model as cm
from exllamav3.loader.safetensors_alt import safe_open
from exllamav3.modules import Linear
from exllamav3.modules.linear import flush_capture_H
from exllamav3.modules.quant import fp16, exl3
import capture_fast
from disk_state import _read_row


OUT = Path('/output/capture-regression')
TASK = Path('/task')
OLD = Path('/output/capture-speed')
STATE_SOURCE = Path('/output/model-job-c03/repair-002-checkpoint/checkpoint-idx2/state.safetensors')
OUT.mkdir(exist_ok=True)  # The boundary runner creates STATE/OWNER/run.log first.
assert not any((OUT/name).exists() for name in ('PROVENANCE.json', 'PROGRESS.json', 'SUMMARY.json'))
assert not list(OUT.glob('control-*.pt')), 'Never overwrite a prior matched control'
assert os.environ.get('EXL3_HIP_F32OUT_VIA_F16') == '0'
assert not fp16._f32_via_f16 and not exl3._f32_via_f16
assert os.environ.get('EXL3_QUANT_PROFILE', '0') != '1'
assert json.loads((OLD/'SUMMARY.json').read_text())['status'] == 'capture_exact_pass'
old_run = json.loads((OLD/'STATE.json').read_text())
assert old_run['status'] == 'passed' and old_run['cleanup_errors'] == []
assert old_run['returned_KFD_contexts'] == []
fast_sha = hashlib.sha256((TASK/'capture_fast.py').read_bytes()).hexdigest()
assert old_run['identity'][str(Path('/home/m2on/projects/ai.tools/test/experiments/flash_next_quant_20261104/capture_fast.py'))] == fast_sha
manifest = json.loads((TASK/'c03-input-ids.json').read_text())
assert manifest['sha256'] == json.loads((OLD/'SUMMARY.json').read_text())['calibration_sha256']


def tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, dict):
        return {key: tree(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tree(item, device) for item in value)
    return copy.deepcopy(value)


def exact(left, right, label):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype, label
        a = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        b = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        assert torch.equal(a, b), label
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), label
        for key in left:
            exact(left[key], right[key], label+'.'+str(key))
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), label
        for i, (a, b) in enumerate(zip(left, right)):
            exact(a, b, label+'.'+str(i))
    else:
        assert left == right, (label, left, right)


def stamp(path):
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns)


def write(name, value):
    temporary = OUT/(name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(OUT/name)


saved_h = torch.load(OLD/'baseline-2-H.pt', map_location='cpu', weights_only=True)
old_normal = torch.load(OLD/'baseline-2-normal.pt', map_location='cpu', weights_only=True)
assert len(saved_h) == 516
assert all(not hd['finalized'] and torch.isfinite(hd['H']).all() for hd in saved_h.values())
assert all(int(hd['count_dev']) == 8737 for hd in saved_h.values())
rows = {512: 4, 8192: 364, 3217: 0}
with safe_open(str(TASK/'c03-input-ids.safetensors')) as reader:
    ids = {length: _read_row(reader, f'input_ids.{index}') for length, index in rows.items()}
assert all(tuple(ids[length].shape) == (1, length) for length in rows)
original_cpu_rng = torch.random.get_rng_state()
original_gpu_rng = torch.cuda.get_rng_state(0)
started = time.monotonic()
results, warmups = [], []
references, normal_references = {}, {}
input_stats = {str(path): stamp(path) for path in
               (STATE_SOURCE, TASK/'c03-input-ids.safetensors', OLD/'baseline-2-H.pt', OLD/'baseline-2-normal.pt')}
with torch.inference_mode():
    config = Config.from_directory('/source')
    model = Model.from_config(config)
    module = model.modules[2]
    assert module.key == 'model.language_model.layers.0'
    deferred = module.can_defer_load()
    if deferred:
        config.stc.begin_deferred_load()
    try:
        module.load(torch.device('cuda:0'))
    finally:
        if deferred:
            config.stc.end_deferred_load()
config.stc.close()
weight_dtypes = Counter()
weight_examples = []
for linear in module:
    if isinstance(linear, Linear):
        weight = linear.inner.get_weight_tensor()
        weight_dtypes[str(weight.dtype)] += 1
        if len(weight_examples) < 7:
            weight_examples.append(dict(key=linear.key, dtype=str(weight.dtype), device=str(weight.device),
                                        shape=list(weight.shape), stride=list(weight.stride())))
write('PROVENANCE.json', dict(reason='Earlier timing lacked inference_mode, warmup balance, and accumulated-H controls',
    prior_control_reused_for_initial_H_and_normal_goldens=True, new_matched_control=True,
    calibration_sha256=manifest['sha256'], source_inputs=input_stats, torch_version=torch.__version__,
    torch_file=torch.__file__, hip_version=torch.version.hip, capture_fast_sha256=fast_sha,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    module=module.key, weight_dtypes=dict(weight_dtypes), weight_examples=weight_examples,
    H_initial_tokens=8737, H_initial_keys=len(saved_h), original_H_flush=True,
    quant_profile=False, old_probe_identity=old_run['identity']))


@torch.inference_mode()
def run_case(length, mode, ordinal, warmup=False):
    assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
    assert capture_fast._active.get() is None
    assert time.monotonic()-started < 1050, 'Leave cleanup margin inside the 1200-second runner'
    # Reset complete GPU state/counters from the same immutable CPU artifact.
    # This setup is outside measurement, matching an already resident capture H.
    H = tree(saved_h, 'cuda:0')
    torch.random.set_rng_state(original_cpu_rng)
    torch.cuda.set_rng_state(original_gpu_rng, 0)
    restore = capture_fast.install(cm, automatic=False) if mode == 'skip_down' else None
    stats_before = capture_fast.STATS.copy()
    try:
        torch.cuda.synchronize()
        read_start = time.perf_counter()
        with safe_open(str(STATE_SOURCE)) as reader:
            cpu_x = _read_row(reader, f'tensor.{rows[length]}')
        read_seconds = time.perf_counter()-read_start
        assert tuple(cpu_x.shape) == (1, length, 4, 2560)
        assert cpu_x.dtype == torch.float32
        params = dict(input_ids=ids[length], attn_mode='flash_attn_nc', quant_preserve={},
                      capture=H, activate_all_experts=True)
        torch.cuda.synchronize()
        transfer_start = time.perf_counter()
        x = module.prepare_for_device(cpu_x, params)
        torch.cuda.synchronize()
        transfer_seconds = time.perf_counter()-transfer_start
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        with capture_fast._discarded_capture(module, params) if restore else nullcontext():
            y = module.forward(x, params)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter()-forward_start
        flush_start = time.perf_counter()
        flush_capture_H(params)
        torch.cuda.synchronize()
        flush_seconds = time.perf_counter()-flush_start
        # All validation/transfers/artifact writes below are outside measured scope.
        assert torch.isfinite(y).all()
        del y
        exact(cpu_x, x, 'capture-input')
        assert params['quant_preserve'] == {} and 'hc_pending' not in params
        assert '_cap_pending' not in params
        for key, hd in H.items():
            assert torch.isfinite(hd['H']).all(), key
            assert int(hd['inf_nan'].sum()) == 0, key
            assert int(hd['count_dev']) == 8737+length, key
        h_cpu = tree(H, 'cpu')
        normal = dict(input_ids=ids[length], attn_mode='flash_attn_nc', quant_preserve={})
        normal_y = module.forward(x, normal)
        assert torch.isfinite(normal_y).all()
        exact(cpu_x, x, 'normal-input')
        assert normal['quant_preserve'] == {} and 'hc_pending' not in normal
        if length in (512, 8192):
            index = 1 if length == 512 else 2
            exact(old_normal['outputs'][index], normal_y, 'reused-normal-golden')
            exact(old_normal['preserves'][index], normal['quant_preserve'], 'reused-normal-preserve')
        if not warmup:
            if length not in references:
                assert mode == 'original'
                references[length] = h_cpu
                normal_references[length] = normal_y.cpu()
                torch.save(h_cpu, OUT/f'control-{length}-H.pt')
                torch.save(normal_references[length], OUT/f'control-{length}-normal.pt')
            else:
                exact(references[length], h_cpu, 'all-H-and-counters')
                exact(normal_references[length], normal_y, 'paired-normal-output')
        exact(original_cpu_rng, torch.random.get_rng_state(), 'CPU-RNG')
        exact(original_gpu_rng, torch.cuda.get_rng_state(0), 'GPU-RNG')
        delta = {key: capture_fast.STATS[key]-value for key, value in stats_before.items()}
        assert delta['skipped_down_calls'] == (512 if restore else 0)
        record = dict(length=length, row=rows[length], mode=mode, ordinal=ordinal, warmup=warmup,
            inference_mode=True, grad_enabled=False, input_dtype=str(x.dtype), input_stride=list(x.stride()),
            input_shape=list(x.shape), hessians=len(H), initial_hessian_tokens=8737,
            cpu_read_seconds=read_seconds, h2d_seconds=transfer_seconds,
            forward_seconds=forward_seconds, flush_seconds=flush_seconds,
            resident_capture_seconds=forward_seconds+flush_seconds,
            measured_pipeline_seconds=read_seconds+transfer_seconds+forward_seconds+flush_seconds,
            validation_excluded=True, exact=True, skip_stats=delta)
        (warmups if warmup else results).append(record)
        write('PROGRESS.json', dict(results=results, warmups=warmups))
        print(json.dumps(record), flush=True)
    finally:
        if restore is not None:
            restore()
        assert capture_fast._active.get() is None


try:
    # 512 ABBA; 8192 AB and, if within budget, BA; actual ragged3217 AB.
    for length in (512, 8192, 3217):
        run_case(length, 'original', 'warmup', warmup=True)
        modes = ['original', 'skip_down', 'skip_down', 'original'] if length == 512 else ['original', 'skip_down']
        for ordinal, mode in enumerate(modes):
            run_case(length, mode, ordinal)
        if length == 8192 and time.monotonic()-started < 700:
            for ordinal, mode in enumerate(('skip_down', 'original'), 2):
                run_case(length, mode, ordinal)
    assert all(stamp(Path(path)) == expected for path, expected in input_stats.items()), 'Input artifact changed'
    write('SUMMARY.json', dict(status='matched_capture_regression_pass', results=results, warmups=warmups,
        elapsed_seconds=time.monotonic()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        comparisons='New controls under inference_mode and identical accumulated H; earlier timing not directly comparable',
        limitations='Stage-boundary synchronizations intentionally equal in both modes; pipeline excludes model load, H initialization, validation, quantization and state advance. CPU reads reopen the immutable idx2 aggregate file, unlike current per-row production checkpoints, and are warm-cache observations; overall conversion speed remains separate.'))
finally:
    torch.random.set_rng_state(original_cpu_rng)
    torch.cuda.set_rng_state(original_gpu_rng, 0)
    module.unload()
