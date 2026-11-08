"""Focused row0 normal-forward reproducibility around original and Lt flush.

Reuse the saved layer6/idx9 fixture and exact control; never interpret the live
checkpoint as those historical inputs. No quantization, serving or writes to
production artifacts. All timed cases are unprofiled. No calibration subsampling or quantization.
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
import capture_lt


OUT = Path('/output/capture-lt-prefix')
FAILED = Path('/output/capture-lt-pipeline')
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
fixture_source = FAILED/'input/state.safetensors'
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


def h_difference(a,b):
    h_delta=b.double()-a.double()
    return dict(exact=torch.equal(a.view(torch.uint8),b.view(torch.uint8)),
                max_abs=float(h_delta.abs().max()),
                relative_l2=float(torch.linalg.vector_norm(h_delta)/torch.linalg.vector_norm(a.double()).clamp_min(1e-30)))


def stat_record(path):
    s = path.stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino)


def write(name, data):
    temporary = OUT/(name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(OUT/name)


cpu_receipt_path=FAILED/'ARITHMETIC-CPU.json'
cpu_receipt=json.loads(cpu_receipt_path.read_text())
assert cpu_receipt['status']=='pass' and not cpu_receipt['gpu_used']
assert all(x['exact'] for x in cpu_receipt['checks'])
assert cpu_receipt['candidate_sha256']==digest_file(Path('/task/capture_lt.py'))
assert cpu_receipt['baseline_source_sha256']==digest_file(Path(inspect.getsourcefile(original_flush)))
cache_receipt=json.loads((FAILED/'CACHE-CPU.json').read_text())
assert cache_receipt['status']=='pass' and cache_receipt['helper_sha256']==digest_file(Path('/task/capture_lt.py'))
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
    FAILED/'input/state.rows.json', cpu_receipt_path)}
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


# Only row0 is executed; retain the historical64-row setup so load/call semantics match.
case_dir=OUT/'row0'
case_dir.mkdir();(case_dir/'work').mkdir();(case_dir/'scratch').mkdir()
args=copy.deepcopy(base_args)
args.update(work_dir=str(case_dir/'work'),out_dir=str(PACK),
            ngram_file=str(PACK/'ngram_embedding.safetensors'),
            cal_rows=len(indices),cal_cols=max(lengths),max_module=target)
args.pop('clear_files',None)
stores=[];live_models=[];outputs={};report={};flush_calls=0

def prepare_subset(in_args,job_state,config,model,tokenizer):
    assert job_state==job and in_args is args
    assert model.modules[target].key=='model.language_model.layers.6'
    assert model.modules[target].attn.__class__ is GatedDeltaNet
    assert model.calibration_all_experts and model.modules[target].num_slices==1
    live_models.append((model,config))
    store=DiskRows(fixture_source,case_dir/'scratch');stores.append(store)
    return ReadOnlySubset(store),[v.clone() for v in token_ids]

def compare(a,b):
    r=h_difference(a,b)
    r.update(finite=bool(torch.isfinite(b).all()),changed_values=int(torch.count_nonzero(a!=b)))
    return r

native=capture_lt.LtGram('/output/capture-gram/gram.so','f32')
lt_flush=capture_lt.make_flush(original_flush,native)

restore_fast=capture_fast.install(cm,automatic=True)
restore_host=capture_progress.install(cm)
base_bar=cm.ProgressBar
class StopAfterRow0(base_bar):
    def __enter__(self):
        assert self.text.startswith(' -- Capturing: ') and self.count==64
        assert sys._getframe(1).f_code is main_body.__code__
        exact(case_cpu_rng,torch.random.get_rng_state(),'entry_cpu_rng')
        exact(case_gpu_rng,torch.cuda.get_rng_state(0),'entry_gpu_rng')
        torch.cuda.synchronize()
        return super().__enter__()
    def update(self,index):
        assert 0<=index<=5
        if index:
            frame=sys._getframe(1)
            assert frame.f_code is main_body.__code__
            loc=frame.f_locals
            assert set(loc['ref_states'])==set(range(index))
            i=index-1
            outputs[i]=cpu_tree(loc['ref_states'][i])
            torch.save(outputs[i],case_dir/(str(i)+'-immediate.pt'))
            report.setdefault('immediate',{})[i]=compare(control['ref_states'][i],outputs[i])
            report.setdefault('cache_by_row',{})[i]=native.cache.stats.copy()
            write('PROGRESS.json',report)
            if index==5:
                retained=cpu_tree(loc['ref_states'])
                torch.save(retained,case_dir/'retained-refs.pt')
                report['retained_refs']={i:compare(outputs[i],v) for i,v in retained.items()}
                report['plan_cache']=native.cache.stats.copy()
                write('COMPARISONS.json',report)
                assert all(v['exact'] for v in report['retained_refs'].values()),'CPU reference mutated later'
                assert report['plan_cache']['evictions']>0,'Eviction coverage missing'
                assert report['plan_cache']['peak_plans']<=8
                assert report['plan_cache']['max_workspace_bytes']==256*2**20
                assert len(loc['capture_H'])==516
                assert all(int(hd['count_dev'])==sum(lengths[:5]) for hd in loc['capture_H'].values())
                report['H_count']=516;report['H_tokens']=sum(lengths[:5])
                report['plan_cache']=native.cache.stats.copy()
                report['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
                exact(case_cpu_rng,torch.random.get_rng_state(),'cpu_rng')
                exact(case_gpu_rng,torch.cuda.get_rng_state(0),'gpu_rng')
                assert all(tensor_digest(loc['state'][i])==input_hashes[i] for i in range(5))
                exact(control['original_input_ids'],loc['original_input_ids'],'IDs')
                exact(control['quant_preserves'],loc['quant_preserves'],'preserves')
                failed=torch.load(FAILED/'warm-f32-f32/failed-numeric.pt',map_location='cpu',weights_only=True)
                report['failed_candidate']={i:compare(failed['ref_states'][i],outputs[i]) for i in outputs}
                report['source_preserve_ID_RNG_unchanged']=True
                write('COMPARISONS.json',report)
                raise CaptureFinished()
        return super().update(index)

cm.prepare_state=prepare_subset;cm.ProgressBar=StopAfterRow0;cm._flush_capture_H=lt_flush
try:
    torch.random.set_rng_state(case_cpu_rng);torch.cuda.set_rng_state(case_gpu_rng,0)
    try:cm.main(args,copy.deepcopy(job))
    except CaptureFinished:pass
    else:raise AssertionError('Row0 boundary not reached')
    assert capture_fast.STATS['scopes']==5
    assert all(digest_file(Path(p))==v for p,v in protected_metadata.items())
    assert all(stat_record(Path(p))==v for p,v in protected_stats.items())
    assert all(digest_file(Path(p))==v for p,v in fixture_hashes.items())
    write('SUMMARY.json',dict(status='cold_lt_prefix_complete',**report,
        seconds=time.monotonic()-started,production_changed=False,
        scope='Cold Lt-first first5 actual cm.main rows; observe each CPU reference immediately and at end. Captures ragged-shape cache eviction. No added normal forward, not a timing/full-model qualification.'))
finally:
    native.close();cm._flush_capture_H=original_flush
    cm.prepare_state=original_prepare_state;cm.ProgressBar=base_bar
    restore_host();restore_fast()
    for name,function in original_guards.items():setattr(cm,name,function)
    Linear.convert_exl3=original_linear_convert
    for store in stores:store.close()
    with torch.inference_mode():
        for model,config in live_models:model.modules[target].unload();config.stc.close()
    torch.random.set_rng_state(old_cpu_rng);torch.cuda.set_rng_state(old_gpu_rng,0)
