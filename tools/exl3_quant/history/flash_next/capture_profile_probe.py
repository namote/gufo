"""Matched ABBA for direct FP32 packing in the unchanged real capture loop.

Reuse the saved layer6/idx9 fixture and exact control; never interpret the live
checkpoint as those historical inputs. No quantization, serving or writes to
production artifacts. The last separate diagnostic case instruments only its
original flushes on rows 0, 1 and 4 with CUDA events; timed ABBA has no added events.
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
import capture_pack


OUT = Path('/output/capture-profile-quarter')
PREVIOUS = Path('/output/capture-pipeline')
WORK = Path('/output/work')
CHECKPOINT = WORK/'ckpt'
PACK = Path('/output/pack')
OUT.mkdir(exist_ok=True)
assert not any((OUT/n).exists() for n in ('PROVENANCE.json', 'PROGRESS.json', 'SUMMARY.json'))
assert os.environ.get('EXL3_HIP_F32OUT_VIA_F16') == '0'
assert not fp16._f32_via_f16 and not exl3._f32_via_f16
assert os.environ.get('EXL3_QUANT_PROFILE', '0') != '1'
assert os.environ.get('EXL3_PROF_ROWS', '0') == '0'
mounts = [line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines()]
for protected in (WORK, PACK, PREVIOUS):
    assert any(parts[4] == str(protected) and 'ro' in parts[5].split(',') for parts in mounts), protected
main_body = inspect.unwrap(cm.main)
main_sha = hashlib.sha256(inspect.getsource(main_body).encode()).hexdigest()
assert main_sha == capture_fast._MAIN_SHA
original_main, original_get_model, original_progress = cm.main, cm.get_base_model, cm.ProgressBar
original_prepare_state = cm.prepare_state
original_flush = cm._flush_capture_H
base_args = json.loads((WORK/'args.json').read_text())
previous_provenance = json.loads((PREVIOUS/'PROVENANCE.json').read_text())
previous_state = json.loads((PREVIOUS/'STATE.json').read_text())
previous_summary = json.loads((PREVIOUS/'SUMMARY.json').read_text())
assert previous_state['status'] == 'passed_pending_review'
assert previous_state['cleanup_errors'] == [] and previous_state['returned_KFD_contexts'] == []
assert previous_summary['status'] == 'production_pipeline_pass' and previous_summary['decision'] == 'keep'
assert all(r['exact'] for r in previous_summary['results'] if not r['warmup'])
job = previous_provenance['checkpoint']
assert job == dict(next_module_idx=9, q_strategy=None, bad_rows=[])
target = previous_provenance['target']
assert target == 9 and previous_provenance['main_sha256'] == main_sha
cal = json.loads(Path('/task/c03-input-ids.json').read_text())
assert base_args['cal_rows'] == cal['rows'] and cal['sha256'] == previous_provenance['calibration_sha256']
indices = list(range(64))
assert previous_provenance['selected_rows'] == indices
lengths = [cal['sequences'][i]['tokens'] for i in indices]
assert [lengths[i] for i in (0, 1, 4)] == [3217, 7793, 512]
assert sum(lengths) == 50056 and Counter(lengths)[512] == 60
fixture_source = OUT/'input/state.safetensors'
fixture_paths, fixture_infos = open_rows(fixture_source)
previous_paths, previous_infos = open_rows(PREVIOUS/'input/state.safetensors')
assert len(fixture_paths) == len(previous_paths) == len(indices)
assert fixture_infos == previous_infos
assert all(fixture_infos[i] == ('F32', (1, lengths[i], 4, 2560)) for i in indices)


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


def numeric_difference(a, b):
    """Retain strict byte parity and bounded error detail, never redefine a pass."""
    mismatches=[]
    def walk(x,y,path):
        if isinstance(x,torch.Tensor):
            if not isinstance(y,torch.Tensor) or x.shape!=y.shape or x.dtype!=y.dtype:
                mismatches.append(dict(path=path,reason='tensor metadata'));return
            equal=torch.equal(x.contiguous().reshape(-1).view(torch.uint8),y.contiguous().reshape(-1).view(torch.uint8))
            if equal:return
            entry=dict(path=path,shape=list(x.shape),dtype=str(x.dtype))
            if len(mismatches)<8 and x.is_floating_point():
                xx=x.float(); yy=y.float();diff=yy-xx
                entry.update(changed_values=int(torch.count_nonzero(x!=y)),
                             max_abs=float(diff.abs().max()),
                             relative_l2=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(xx).clamp_min(1e-30)),
                             finite=bool(torch.isfinite(yy).all()))
            mismatches.append(entry)
        elif isinstance(x,dict):
            if not isinstance(y,dict) or x.keys()!=y.keys():
                mismatches.append(dict(path=path,reason='dict keys'));return
            for k in x:walk(x[k],y[k],path+'.'+str(k))
        elif isinstance(x,(list,tuple)):
            if type(x)!=type(y) or len(x)!=len(y):
                mismatches.append(dict(path=path,reason='sequence'));return
            for i,(xx,yy) in enumerate(zip(x,y)):walk(xx,yy,path+'.'+str(i))
        elif x!=y:mismatches.append(dict(path=path,reason='value',expected=str(x),actual=str(y)))
    threads=torch.get_num_threads();torch.set_num_threads(2)
    try:walk(a,b,'capture')
    finally:torch.set_num_threads(threads)
    return dict(exact=not mismatches,mismatches=mismatches)


def stat_record(path):
    s = path.stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino)


def write(name, data):
    temporary = OUT/(name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(OUT/name)


cpu_receipt_path = Path('/output/capture-optimization-cpu-active-runtime/PACK-CPU.json')
cpu_receipt = json.loads(cpu_receipt_path.read_text())
assert cpu_receipt['status'] == 'pass' and not cpu_receipt['gpu_used']
assert len(cpu_receipt['checks']) == 23 and all(x['exact'] for x in cpu_receipt['checks'])
assert cpu_receipt['torch_version'] == torch.__version__ and cpu_receipt['hip_version'] == torch.version.hip
assert digest_file(Path('/task/capture_pack.py')) == cpu_receipt['candidate_sha256']
assert digest_file(Path('/task/test_capture_pack.py')) == cpu_receipt['test_sha256']
assert digest_file(Path(inspect.getsourcefile(original_flush))) == cpu_receipt['baseline_source_sha256']
capture_pack.make_flush_capture_H(original_flush)  # Source SHA gate before GPU work.
copy_receipt = json.loads((PREVIOUS/'INPUT-COPY.json').read_text())
assert len(copy_receipt['rows']) == 64
control = torch.load(PREVIOUS/'control.pt', map_location='cpu', weights_only=True)
assert len(control['capture_H']) == 516 and len(control['ref_states']) == 5
input_hashes = []
fixture_hashes = {}
for i in indices:
    src, dst = previous_paths[i], fixture_paths[i]
    assert src.stat().st_dev == dst.stat().st_dev and src.stat().st_ino != dst.stat().st_ino
    expected = copy_receipt['rows'][i]['sha256']
    assert Path(copy_receipt['rows'][i]['target']).name == f'{i}.safetensors'
    assert digest_file(src) == digest_file(dst) == expected
    assert previous_provenance['private_fixture_hashes'][str(src)] == expected
    fixture_hashes[str(src)] = fixture_hashes[str(dst)] = expected
    with safe_open(str(dst)) as reader:
        input_hashes.append(tensor_digest(_read_row(reader, 'tensor')))
assert input_hashes == control['input_hashes']
token_ids = control['original_input_ids']
assert len(token_ids) == len(indices)
with safe_open('/task/c03-input-ids.safetensors') as reader:
    for i, value in zip(indices, token_ids):
        current = _read_row(reader, f'input_ids.{i}')
        assert value.shape == current.shape and torch.equal(value.long(), current.long())
# Only stable original identities are compared to the previous receipt. The live
# checkpoint may have advanced and must never be mistaken for layer6 input state.
for path in (WORK/'args.json', Path('/task/b1-recipe.json')):
    assert digest_file(path) == previous_provenance['protected_metadata'][str(path)]
protected_metadata = {str(p): digest_file(p) for p in (
    WORK/'args.json', CHECKPOINT/'job.json', CHECKPOINT/'state.rows.json',
    CHECKPOINT/'original_input_ids.safetensors', Path('/task/b1-recipe.json'),
    PREVIOUS/'PROVENANCE.json', PREVIOUS/'STATE.json', PREVIOUS/'SUMMARY.json',
    PREVIOUS/'INPUT-COPY.json', PREVIOUS/'control.pt', PREVIOUS/'input/state.rows.json',
    OUT/'input/state.rows.json', cpu_receipt_path)}
current_paths, _ = open_rows(CHECKPOINT/'state.safetensors')
protected_stats = {str(p): stat_record(p) for p in current_paths}
protected_stats[str(PACK/'ngram_embedding.safetensors')] = stat_record(PACK/'ngram_embedding.safetensors')
old_cpu_rng = torch.random.get_rng_state()
old_gpu_rng = torch.cuda.get_rng_state(0)
# Capture consumes no RNG. Saved end-state is the reusable matched start state.
case_cpu_rng, case_gpu_rng = control['cpu_rng'], control['gpu_rng']
started = time.monotonic()
records = []
model_key = None


def instrument_original_flush(enable_trace=True):
    """All64 flushes plus read/transfer/forward; no per-row synchronization."""
    source = inspect.getsource(original_flush)
    assert hashlib.sha256(source.encode()).hexdigest() == capture_pack.PINNED_FLUSH_SHA256
    events, rows, reads = [], [], []
    active = None
    trace = None
    trace_done = False
    original_read = DiskRows.__getitem__
    from exllamav3.modules.transformer import TransformerBlock
    original_prepare = TransformerBlock.prepare_for_device

    def event():
        e = torch.cuda.Event(enable_timing=True); e.record(); return e

    def read(store, index):
        nonlocal trace
        if enable_trace and index == 4 and not trace_done and trace is None:
            trace = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA])
            trace.start()
        start = time.perf_counter(); hit = store.index == index
        value = original_read(store, index)
        reads.append(dict(index=index, cache_hit=hit, host_seconds=time.perf_counter()-start))
        return value

    def prepare(module, x, params):
        nonlocal active
        if 'capture' not in params:
            return original_prepare(module, x, params)
        active = dict(row=len(rows), tokens=x.shape[1], start=event(),
                      prepare_host_start=time.perf_counter())
        value = original_prepare(module, x, params)
        active.update(prepared=event(), prepare_host_end=time.perf_counter())
        rows.append(active)
        return value

    def event_begin(chunk):
        events.append(dict(row=active['row'], entries=len(chunk),
                           shape=list(chunk[0][1].shape), dtype=str(chunk[0][1].dtype), events=[]))
        event_mark()

    def event_mark():
        events[-1]['events'].append(event())

    for old, new in (
        ('            xs = torch.stack([e[1] for e in chunk])',
         '            _event_begin(chunk)\n            xs = torch.stack([e[1] for e in chunk])'),
        ('            P = torch.bmm(xf.transpose(1, 2), xf)',
         '            _event_mark()\n            P = torch.bmm(xf.transpose(1, 2), xf)'),
        ('            fs = finite.sum(dim = 1)',
         '            _event_mark()\n            fs = finite.sum(dim = 1)'),
        ('            xs = xf = P = fs = None',
         '            _event_mark()\n            xs = xf = P = fs = None')):
        assert source.count(old) == 1
        source = source.replace(old, new)
    ns = dict(original_flush.__globals__, _event_begin=event_begin, _event_mark=event_mark)
    local = {}; exec(compile(source, '<capture-profile>', 'exec'), ns, local)
    instrumented = local['flush_capture_H']
    DiskRows.__getitem__ = read
    TransformerBlock.prepare_for_device = prepare

    def flush(params):
        nonlocal trace_done
        assert active is not None and active['row'] == len(rows)-1
        active.update(forward_end=event(), forward_host_end=time.perf_counter())
        instrumented(params)
        active.update(flush_end=event(), flush_host_end=time.perf_counter())
        if enable_trace and active['row'] == 4:
            assert trace is not None
            trace.stop(); trace_done = True
        active['trace_instrumented'] = enable_trace and active['row'] == 4

    restored = False
    def restore():
        nonlocal restored, trace_done
        if restored:
            return
        restored = True
        DiskRows.__getitem__ = original_read
        TransformerBlock.prepare_for_device = original_prepare
        if trace is not None and not trace_done:
            trace.stop(); trace_done = True
    flush.restore = restore

    def report():
        assert len(rows) == 64 and (trace_done or not enable_trace)
        stages = ('pack_cast_finite_ms', 'bmm_ms', 'count_sum_H_count_add_ms')
        result = []
        for row in rows:
            selected = [e for e in events if e['row'] == row['row']]
            sums = {key: sum(e['events'][i].elapsed_time(e['events'][i+1]) for e in selected)
                    for i, key in enumerate(stages)}
            result.append(dict(row=row['row'], tokens=row['tokens'], chunks=len(selected),
                trace_instrumented=row['trace_instrumented'],
                prepare_gpu_interval_ms=row['start'].elapsed_time(row['prepared']),
                forward_gpu_interval_ms=row['prepared'].elapsed_time(row['forward_end']),
                flush_gpu_interval_ms=row['forward_end'].elapsed_time(row['flush_end']),
                prepare_host_seconds=row['prepare_host_end']-row['prepare_host_start'],
                forward_host_seconds=row['forward_host_end']-row['prepare_host_end'],
                flush_host_seconds=row['flush_host_end']-row['forward_host_end'], **sums))
        operators = []
        if trace is not None:
            trace.export_chrome_trace(str(OUT/'row4-trace.json'))
            operators = [dict(name=e.key, calls=e.count, self_cpu_us=e.self_cpu_time_total,
                              self_device_us=e.self_device_time_total) for e in trace.key_averages()]
        return dict(rows=result, reads=reads, operators=operators,
            caveat='64 historical layer6 rows, already cached; row4 has torch-profiler overhead. CUDA intervals can include host submission gaps. Host/device spans overlap; do not sum them or claim whole-model regression explained.')
    return flush, report


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


def run_case(mode, ordinal, warmup=False, profile=False, enable_trace=False):
    global model_key
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
        assert model.modules[target].attn.__class__ is GatedDeltaNet, 'Expected historical idx9 GDN block'
        assert model.calibration_all_experts and model.modules[target].num_slices == 1
        if model_key is None:
            model_key = model.modules[target].key
        assert model.modules[target].key == model_key == 'model.language_model.layers.6'
        live_models.append((model, config))
        store = DiskRows(fixture_source, scratch)
        stores.append(store)
        return ReadOnlySubset(store), [value.clone() for value in token_ids]

    cm.prepare_state = prepare_subset
    restore_fast = restore_host = restore_pack = None
    profile_report = None
    profile_flush = None
    stats_before = capture_fast.STATS.copy()
    restore_fast = capture_fast.install(cm, automatic=True)
    restore_host = capture_progress.install(cm)
    if mode == 'pack':
        restore_pack = capture_pack.install(cm)
    elif profile:
        assert mode == 'original'
        profile_flush, profile_report = instrument_original_flush(enable_trace=enable_trace)
        cm._flush_capture_H = profile_flush
    base_bar = cm.ProgressBar

    class CaptureTimer(base_bar):
        def __enter__(self):
            assert self.text.startswith(' -- Capturing: '), 'Unexpected pre-capture progress stage'
            assert self.count == len(indices)
            assert sys._getframe(1).f_code is main_body.__code__, 'Must observe actual main frame'
            assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
            assert torch.equal(torch.random.get_rng_state(), case_cpu_rng), 'Pre-capture CPU RNG changed'
            assert torch.equal(torch.cuda.get_rng_state(0), case_gpu_rng), 'Pre-capture GPU RNG changed'
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
    torch.random.set_rng_state(case_cpu_rng)
    torch.cuda.set_rng_state(case_gpu_rng, 0)
    try:
        try:
            cm.main(args, copy.deepcopy(job))
        except CaptureFinished:
            pass
        else:
            raise AssertionError('Main returned without terminating at capture boundary')
        if profile_flush:
            profile_flush.restore()
        assert collected and capture_fast._active.get() is None
        profile_data = profile_report() if profile_report else None
        if profile_data is not None:
            write(f'{ordinal}-PROFILE-PROVISIONAL.json', profile_data)
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
        assert delta['scopes'] == len(indices)
        assert delta['skipped_down_calls'] == 512*len(indices)
        comparison = numeric_difference(control, collected['numeric'])
        if not comparison['exact']:
            torch.save(collected['numeric'], OUT/f'{ordinal}-numeric-mismatch.pt')
        write(f'{ordinal}-COMPARISON.json', comparison)
        record = dict(mode=mode, ordinal=ordinal, warmup=warmup, profile=profile,
                      seconds=collected['seconds'], rows=len(indices), tokens=sum(lengths),
                      module=model_key, hessians=516, normal_references=5, exact=comparison['exact'],
                      saved_control_exact=comparison['exact'], comparison=comparison, automatic_scope=True, skip_stats=delta,
                      inference_mode=True, explicit_per_row_sync=False, profiler_row4_may_synchronize=enable_trace)
        if profile_report:
            record['stage_profile'] = profile_data
        records.append(record)
        write('PROGRESS.json', records)
        print('PACK_CASE', json.dumps(record), flush=True)
    except BaseException as error:
        if 'numeric' in collected:
            torch.save(collected['numeric'], OUT/f'{ordinal}-failed-numeric.pt')
        write(f'{ordinal}-FAILURE.json', dict(error=repr(error), numeric_saved='numeric' in collected))
        raise
    finally:
        if restore_pack:
            restore_pack()
        elif profile:
            profile_flush.restore()
            cm._flush_capture_H = original_flush
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
    pack_sha256=digest_file(Path('/task/capture_pack.py')), cpu_receipt_sha256=digest_file(cpu_receipt_path),
    historical_target=9, current_checkpoint=json.loads((CHECKPOINT/'job.json').read_text()),
    saved_control_sha256=digest_file(PREVIOUS/'control.pt'),
    torch_version=torch.__version__, torch_file=torch.__file__, hip_version=torch.version.hip,
    main_and_get_base_model_unchanged=True, state_reader='Actual DiskRows constructor and one-row cache over 64 private byte-identical copies of the retained capture-pipeline idx9/layer6 fixture rows; same filesystem/format, independent inodes; warm cache; fixture copy excluded from timed loop',
    ngram='Existing read-only /output/pack path; no copy/hardlink or model output writes',
    timing='Profiled capture span includes initial five normal references, which are outside capture stage totals. Explicit sync at loop boundaries; row4 torch profiler may synchronize and add overhead. Validation/model loading excluded; not uninstrumented speed evidence',
    scope='Matched controls for a changed flush only, saved numerical control reused without recomputation; previous artifacts untouched'))
try:
    # Only the failed instrumentation is isolated; saved numeric control is reused.
    run_case('original', 'events-only', profile=True)
    if records[-1]['exact']:
        run_case('original', 'events-and-trace', profile=True, enable_trace=True)
    assert all(digest_file(Path(p)) == h for p, h in protected_metadata.items())
    assert all(stat_record(Path(p)) == record for p, record in protected_stats.items())
    assert all(digest_file(Path(p)) == h for p, h in fixture_hashes.items()), 'Private inputs changed'
    write('SUMMARY.json', dict(status='capture_profile_pass' if all(r['exact'] for r in records) else 'profile_numeric_mismatch', results=records,
        elapsed_seconds=time.monotonic()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        scope='Profile only, saved numeric control reused; no baseline rerun or speedup claim'))

finally:
    cm.prepare_state = original_prepare_state
    for name, function in original_guards.items():
        setattr(cm, name, function)
    Linear.convert_exl3 = original_linear_convert
    torch.random.set_rng_state(old_cpu_rng)
    torch.cuda.set_rng_state(old_gpu_rng, 0)
