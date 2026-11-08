"""Bounded ABBA of the real pinned cm.main capture loop at the latest checkpoint.

Only prepare_state and the ProgressBar observer are substituted. Model loading,
main code, row-cache reads, first five normal references, and H flush are real.
The observer exits before quantization; write/quant hooks and read-only mounts
independently protect the conversion artifacts. No per-row GPU synchronization.
"""
import copy
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import torch
from exllamav3.conversion import convert_model as cm
from exllamav3.loader.safetensors_alt import safe_open
from exllamav3.modules.gated_delta_net import GatedDeltaNet
from exllamav3.modules import Linear
from exllamav3.modules.quant import fp16, exl3
from disk_state import DiskRows, _read_row
from row_checkpoint import open_rows
import capture_fast
import capture_progress


OUT = Path('/output/capture-pipeline')
WORK = Path('/output/work')
CHECKPOINT = WORK/'ckpt'
PACK = Path('/output/pack')
OUT.mkdir(exist_ok=True)
assert not any((OUT/n).exists() for n in ('PROVENANCE.json', 'PROGRESS.json', 'SUMMARY.json', 'control.pt'))
assert os.environ.get('EXL3_HIP_F32OUT_VIA_F16') == '0'
assert not fp16._f32_via_f16 and not exl3._f32_via_f16
assert os.environ.get('EXL3_QUANT_PROFILE', '0') != '1'
assert os.environ.get('EXL3_PROF_ROWS', '0') == '0'
mounts = [line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines()]
for protected in (WORK, PACK):
    assert any(parts[4] == str(protected) and 'ro' in parts[5].split(',') for parts in mounts), protected
main_body = inspect.unwrap(cm.main)
main_sha = hashlib.sha256(inspect.getsource(main_body).encode()).hexdigest()
assert main_sha == capture_fast._MAIN_SHA
original_main, original_get_model, original_progress = cm.main, cm.get_base_model, cm.ProgressBar
original_prepare_state = cm.prepare_state
original_flush = cm._flush_capture_H
base_args = json.loads((WORK/'args.json').read_text())
job = json.loads((CHECKPOINT/'job.json').read_text())
assert job['bad_rows'] == []
target = job['next_module_idx']
assert target > 1
cal = json.loads(Path('/task/c03-input-ids.json').read_text())
assert base_args['cal_rows'] == cal['rows']
paths, infos = open_rows(CHECKPOINT/'state.safetensors')
assert len(paths) == cal['rows'] and not (CHECKPOINT/'state.safetensors').exists()
indices = list(range(64))
lengths = [cal['sequences'][i]['tokens'] for i in indices]
assert sum(lengths) == 50056 and Counter(lengths)[512] == 60
assert all(infos[i] == ('F32', (1, lengths[i], 4, 2560)) for i in indices)
fixture_source = OUT/'input/state.safetensors'
fixture_paths, fixture_infos = open_rows(fixture_source)
assert len(fixture_paths) == len(indices) and fixture_infos == infos[:len(indices)]


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(4*2**20):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(value):
    assert value.device.type == 'cpu'
    raw = value.contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def exact(a, b, name='capture'):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype, name
        assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                           b.contiguous().reshape(-1).view(torch.uint8)), name
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), name
        for key in a:
            exact(a[key], b[key], name+'.'+str(key))
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), name
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, name+'.'+str(i))
    else:
        assert a == b, (name, a, b)


def stat_record(path):
    s = path.stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino)


def write(name, data):
    temporary = OUT/(name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(OUT/name)


input_hashes = []
fixture_hashes = {}
for i in indices:
    source_stat, fixture_stat = paths[i].stat(), fixture_paths[i].stat()
    assert source_stat.st_dev == fixture_stat.st_dev, 'Inputs must use the same internal filesystem'
    assert source_stat.st_ino != fixture_stat.st_ino, 'Private fixture must not alias the protected inode'
    copied_digest = digest_file(fixture_paths[i])
    assert digest_file(paths[i]) == copied_digest, 'Private fixture differs from checkpoint bytes'
    fixture_hashes[str(fixture_paths[i])] = copied_digest
    with safe_open(str(paths[i])) as reader:
        input_hashes.append(tensor_digest(_read_row(reader, 'tensor')))
with safe_open(str(CHECKPOINT/'original_input_ids.safetensors')) as reader:
    token_ids = [_read_row(reader, f'tensor.{i}') for i in indices]
with safe_open('/task/c03-input-ids.safetensors') as reader:
    for i, value in zip(indices, token_ids):
        current = _read_row(reader, f'input_ids.{i}')
        assert value.shape == current.shape and torch.equal(value.long(), current.long())
protected_metadata = {str(p): digest_file(p) for p in (
    WORK/'args.json', CHECKPOINT/'job.json', CHECKPOINT/'state.rows.json',
    CHECKPOINT/'original_input_ids.safetensors', Path('/task/b1-recipe.json'))}
protected_stats = {str(p): stat_record(p) for p in paths}
protected_stats[str(PACK/'ngram_embedding.safetensors')] = stat_record(PACK/'ngram_embedding.safetensors')
old_cpu_rng = torch.random.get_rng_state()
old_gpu_rng = torch.cuda.get_rng_state(0)
started = time.monotonic()
records = []
control = None
model_key = None


class CaptureFinished(BaseException):
    pass


class ReadOnlySubset:
    def __init__(self, store):
        self.store = store

    def __len__(self):
        return len(indices)

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < len(indices):
            raise IndexError(index)
        return self.store[indices[index]]

    def __setitem__(self, index, value):
        raise AssertionError('Diagnostic must stop before state advancement')


def forbidden(*args, **kwargs):
    raise AssertionError('Quantization/checkpoint/compile/write is forbidden in capture diagnostic')


guard_names = ('save_tensor', 'save_dict', 'quantize_linears_single', 'quantize_linears_parallel', 'compile_model')
original_guards = {name: getattr(cm, name) for name in guard_names}
original_linear_convert = Linear.convert_exl3
for name in guard_names:
    setattr(cm, name, forbidden)
Linear.convert_exl3 = forbidden


def run_case(mode, ordinal, warmup=False):
    global control, model_key
    assert time.monotonic()-started < 1050, 'Leave cleanup margin inside 1200-second runner'
    assert cm.main is original_main and cm.get_base_model is original_get_model
    assert cm._flush_capture_H is original_flush
    case_dir = OUT/f'{ordinal}-{mode}'
    case_dir.mkdir()
    (case_dir/'work').mkdir()
    scratch = case_dir/'scratch'
    scratch.mkdir()
    args = copy.deepcopy(base_args)
    # prepare_ngram_table_for_conversion sees equal source/destination paths and
    # only reads/adds the existing K8 table to the model collection. Mount is ro.
    args.update(work_dir=str(case_dir/'work'), out_dir=str(PACK),
                ngram_file=str(PACK/'ngram_embedding.safetensors'),
                cal_rows=len(indices), cal_cols=max(lengths), max_module=target)
    args.pop('clear_files', None)
    stores, live_models = [], []
    collected = {}

    def prepare_subset(in_args, job_state, config, model, tokenizer):
        nonlocal collected
        global model_key
        assert job_state == job and in_args is args
        assert model.modules[target].attn.__class__ is GatedDeltaNet, 'Expected latest GDN block'
        assert model.calibration_all_experts and model.modules[target].num_slices == 1
        if model_key is None:
            model_key = model.modules[target].key
        assert model.modules[target].key == model_key
        live_models.append((model, config))
        store = DiskRows(fixture_source, scratch)
        stores.append(store)
        return ReadOnlySubset(store), [value.clone() for value in token_ids]

    cm.prepare_state = prepare_subset
    restore_fast = restore_host = None
    stats_before = capture_fast.STATS.copy()
    if mode == 'skip_down':
        restore_fast = capture_fast.install(cm, automatic=True)
        restore_host = capture_progress.install(cm)
    base_bar = cm.ProgressBar

    class CaptureTimer(base_bar):
        def __enter__(self):
            assert self.text.startswith(' -- Capturing: '), 'Unexpected pre-capture progress stage'
            assert self.count == len(indices)
            assert sys._getframe(1).f_code is main_body.__code__, 'Must observe actual main frame'
            assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
            torch.cuda.synchronize()
            self.started = time.perf_counter()
            return super().__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is not None:
                return super().__exit__(exc_type, exc_value, traceback)
            assert sys._getframe(1).f_code is main_body.__code__
            super().__exit__(None, None, None)  # Include original/host-progress teardown.
            torch.cuda.synchronize()
            collected['seconds'] = time.perf_counter()-self.started
            frame = sys._getframe(1)
            local = frame.f_locals
            assert local['idx'] == target and local['bad_rows'] == set()
            assert local['state'].store is stores[0]
            assert len(local['capture_H']) == 516 and len(local['ref_states']) == 5
            # Copies and exact checks occur only AFTER the complete-loop timer.
            collected['numeric'] = cpu_tree({name: local[name] for name in
                ('capture_H', 'ref_states', 'quant_preserves', 'original_input_ids')})
            collected['numeric']['cpu_rng'] = torch.random.get_rng_state()
            collected['numeric']['gpu_rng'] = torch.cuda.get_rng_state(0)
            del local, frame
            raise CaptureFinished()

    cm.ProgressBar = CaptureTimer
    torch.random.set_rng_state(old_cpu_rng)
    torch.cuda.set_rng_state(old_gpu_rng, 0)
    try:
        try:
            cm.main(args, copy.deepcopy(job))
        except CaptureFinished:
            pass
        else:
            raise AssertionError('Main returned without terminating at capture boundary')
        assert collected and capture_fast._active.get() is None
        for hd in collected['numeric']['capture_H'].values():
            assert torch.isfinite(hd['H']).all() and int(hd['inf_nan'].sum()) == 0
            assert int(hd['count_dev']) == sum(lengths)
        # Do not retain all CPU input rows during the measured one-row-cache loop.
        # Check the last resident row before eviction, then re-read every selected
        # input through DiskRows outside timing and compare its original bytes.
        store = stores[0]
        assert store.index == indices[-1] and tensor_digest(store.tensor) == input_hashes[-1]
        assert not store.dirty and store.changed == set()
        observed = [tensor_digest(store[i]) for i in indices]
        assert observed == input_hashes
        collected['numeric']['input_hashes'] = observed
        delta = {key: capture_fast.STATS[key]-value for key, value in stats_before.items()}
        assert delta['scopes'] == (len(indices) if mode == 'skip_down' else 0)
        assert delta['skipped_down_calls'] == (512*len(indices) if mode == 'skip_down' else 0)
        if not warmup:
            if control is None:
                assert mode == 'original'
                control = collected['numeric']
                torch.save(control, OUT/'control.pt')
            else:
                exact(control, collected['numeric'])
        record = dict(mode=mode, ordinal=ordinal, warmup=warmup, seconds=collected['seconds'],
                      rows=len(indices), tokens=sum(lengths), module=model_key,
                      hessians=516, normal_references=5, exact=None if warmup else True,
                      paired_exact=not warmup, automatic_scope=mode == 'skip_down',
                      skip_stats=delta, inference_mode=True, per_row_sync=False)
        records.append(record)
        write('PROGRESS.json', records)
        print('PIPELINE_CASE', json.dumps(record), flush=True)
    finally:
        cm.ProgressBar = base_bar
        if restore_host:
            restore_host()
        if restore_fast:
            restore_fast()
        cm.prepare_state = original_prepare_state
        for store in stores:
            store.close()
        with torch.inference_mode():
            for model, config in live_models:
                model.modules[target].unload()
                config.stc.close()
        assert cm.ProgressBar is original_progress


write('PROVENANCE.json', dict(target=target, checkpoint=job, selected_rows=indices,
    tokens=sum(lengths), length_counts=dict(Counter(lengths)), calibration_sha256=cal['sha256'],
    protected_metadata=protected_metadata, protected_stats=protected_stats, private_fixture_hashes=fixture_hashes,
    main_sha256=main_sha, script_sha256=digest_file(Path(__file__)),
    torch_version=torch.__version__, torch_file=torch.__file__, hip_version=torch.version.hip,
    main_and_get_base_model_unchanged=True, state_reader='Actual DiskRows constructor and one-row cache over 64 private byte-identical copies of the latest checkpoint rows; same filesystem/format, independent inodes; warm cache; fixture copy excluded from timed loop',
    ngram='Existing read-only /output/pack path; no copy/hardlink or model output writes',
    timing='Single capture ProgressBar enter/exit span including base behavior and first five normal references; only two GPU sync boundaries; validation and model loading excluded',
    scope='New actual-path diagnostic control; previous baseline artifacts remain untouched'))
try:
    run_case('original', 'warm-A', warmup=True)
    run_case('skip_down', 'warm-B', warmup=True)
    for ordinal, mode in enumerate(('original', 'skip_down', 'skip_down', 'original')):
        run_case(mode, ordinal)
    a = [r['seconds'] for r in records if not r['warmup'] and r['mode'] == 'original']
    b = [r['seconds'] for r in records if not r['warmup'] and r['mode'] == 'skip_down']
    if max(b) <= .95*min(a) and min(a)-max(b) >= .010:
        decision = 'keep'
    elif max(a) <= .95*min(b) and min(b)-max(a) >= .010:
        decision = 'rollback'
    else:
        decision = 'inconclusive_keep_current'
    assert all(digest_file(Path(p)) == h for p, h in protected_metadata.items())
    assert all(stat_record(Path(p)) == record for p, record in protected_stats.items())
    assert all(digest_file(Path(p)) == h for p, h in fixture_hashes.items()), 'Private inputs changed'
    write('SUMMARY.json', dict(status='production_pipeline_pass', results=records,
        original_seconds=a, skip_down_seconds=b, decision=decision,
        selection_rule='Both candidate times must beat both controls by at least 5% AND 10ms; symmetric rule for rollback; overlap/noise means inconclusive, retain current setting',
        elapsed_seconds=time.monotonic()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        scope='64-row complete real capture loop, not a full-conversion speed claim; both modes warmed, input page cache reused, quantization and state advance excluded'))
finally:
    cm.prepare_state = original_prepare_state
    for name, function in original_guards.items():
        setattr(cm, name, function)
    Linear.convert_exl3 = original_linear_convert
    torch.random.set_rng_state(old_cpu_rng)
    torch.cuda.set_rng_state(old_gpu_rng, 0)
