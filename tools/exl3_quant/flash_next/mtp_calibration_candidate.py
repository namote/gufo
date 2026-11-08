"""OFFLINE candidate: calibrated MTP stages from protected trunk idx51.

Not integrated or GPU-qualified. Invoke only inside the parent's bounded,
exclusive GPU/JOB lease. This file never starts/stops services or main jobs.
It does not compile/promote final packs or modify trunk tensors/checkpoints.
Stage input uses explicit original FP32 Gram; stage decoder uses the already
qualified Lt/static capture helpers. No uncalibrated fallback is accepted.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

from mtp_calibration_contract import configure_mtp, expected_count
from mtp_calibration_gufo import SEMANTICS, install_input_adapter, install_decoder_gram


def write_json(path, value):
    pending = path.with_suffix('.pending')
    with pending.open('x') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(pending, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def rows_identity(path):
    """Bind immutable row files by validated header + inode/stat, not a 76GiB rehash."""
    from row_checkpoint import open_rows
    paths, infos = open_rows(path)
    records = []
    for filename, info in zip(paths, infos, strict=True):
        stat = filename.stat()
        records.append([stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, info])
    return {'manifest_sha256': digest(Path(path).with_suffix('.rows.json')),
            'rows_stat_sha256': hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()}


def candidate_identity():
    return {name: digest(Path(__file__).with_name(name)) for name in
            ('mtp_calibration_contract.py', 'mtp_calibration_candidate.py', 'mtp_calibration_cache.py', 'mtp_calibration_gufo.py', 'mtp_borrowed_rows.py', 'mtp_capture_identity.py')}


def capture_provenance(cm):
    """Original MTP weights, pinned runtime math and selected GPU arithmetic."""
    import torch
    source_inventory=Path('/output/source-inventory.json')
    inventory=json.loads(source_inventory.read_text())
    for entry in inventory['files']:
        stat=(Path('/source')/entry['name']).stat()
        assert (stat.st_size,stat.st_mtime_ns)==(entry['bytes'],entry['mtime_ns'])
    frozen=json.loads(Path('/output/model-job-c03/FROZEN.json').read_text())
    runtime_root=Path(cm.__file__).resolve().parents[1]
    runtime={}
    for path,expected in frozen.items():
        marker='/source/exllamav3/'
        if marker in path and path.endswith('.py'):
            relative=path.split(marker,1)[1]
            assert digest(runtime_root/relative)==expected
            runtime[relative]=expected
    assert runtime
    from exllamav3.ext import exllamav3_ext
    native_path=Path(exllamav3_ext.__file__).resolve()
    assert native_path.is_file()
    gpu=torch.cuda.get_device_properties(0)
    return {'source_inventory_sha256':digest(source_inventory),'source_revision':inventory['revision'],
            'runtime_source_sha256':hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest(),
            'extension_sha256':digest(native_path),'target_semantics':SEMANTICS,
            'torch':torch.__version__,'hip':torch.version.hip,'gpu':gpu.name,
            'torch_math_flags':{'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                'fp16_reduced_precision':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
            'arch':getattr(gpu,'gcnArchName',None),
            'flags':{key:value for key,value in sorted(os.environ.items()) if key.startswith(('EXL3_','HIPBLAS','ROCBLAS','HSA_','TORCH_','PYTORCH_'))}}


def preserve_scratch_until_commit(stores):
    """Return successful cleanup; failures survive close and Python finalization."""
    cleanup = stores.scratch.cleanup
    assert stores.scratch._finalizer.detach() is not None
    stores.scratch.cleanup = lambda: None
    return cleanup


def swap_captured_h_cpu(value, device, resumed):
    if resumed:
        assert value['H_swap_device']==device and value['device']==device
    else:
        assert value['H'].device==device
        value['H_swap_device']=value['H'].device
    value['H']=value['H'].cpu()


@contextlib.contextmanager
def checked_quantization(linear_class, token_count, hc_mult):
    import torch
    original = linear_class.convert_exl3
    lock = threading.Lock()

    def convert(self, H_data, quant_args, *args, **kwargs):
        if H_data['H'].is_meta:
            raise RuntimeError('MTP calibration cannot use meta/uncalibrated Hessians')
        with lock:
            count = int(H_data.get('count_dev', H_data['count']))
            if count != expected_count(self.key, token_count, hc_mult):
                raise RuntimeError(f'Incomplete MTP calibration: {self.key}: {count}')
            if not H_data.get('_gufo_mtp_audited'):
                assert torch.isfinite(H_data['H']).all()
                assert int(H_data['inf_nan'].sum()) == 0
                H_data['_gufo_mtp_audited'] = True
        result = original(self, H_data, quant_args, *args, **kwargs)
        assert not H_data.get('q_fallback') and not quant_args.get('q_fallback')
        proxy = result[0] if isinstance(result, tuple) else result
        assert math.isfinite(float(proxy)) and float(proxy) >= 0
        assert math.isfinite(float(quant_args['g_scale']))
        for key in ('suh', 'svh', 'bias'):
            value = getattr(self.inner, key)
            assert value is None or torch.isfinite(value).all()
        return result

    linear_class.convert_exl3 = convert
    try:
        yield
    finally:
        assert linear_class.convert_exl3 is convert
        linear_class.convert_exl3 = original


def run_module(cm, module, config, states, ids, args, strategy, index,
               token_count, hc_mult, *, advance_all, resume_packed=False, resume_capture=False):
    """Reuse pinned converter math/workers; only MTP data plumbing differs."""
    import torch
    from exllamav3.modules import Linear
    device = torch.device('cuda:0')
    assert not (resume_packed and resume_capture)
    started = time.time()
    from safetensors.torch import load_file
    work = Path(args['work_dir'])
    filename = f'qtensors/{module.key}.safetensors'
    refs_file = f'{module.key}.references.safetensors'
    packed_path = work/f'{module.key}.PACKED.json'
    binding = {'module': module.key, 'index': index, 'rows': len(states),
               'tokens': token_count, 'strategy': strategy,
               'inputs_sha256': digest(work/'INPUTS.json'),
               'state_identity': args['mtp_state_identity'],
               'code_sha256': candidate_identity(),
               'qualification_sha256': args.get('mtp_qualification_sha256')}
    if resume_packed:
        saved = json.loads(packed_path.read_text())
        assert all(saved[k] == v for k, v in binding.items())
        assert saved['packed_sha256'] == digest(work/filename)
        assert saved['refs_sha256'] == digest(work/refs_file)
        tensors = load_file(str(work/filename))
        refs = {int(k): v for k, v in load_file(str(work/refs_file)).items()}
        assert set(refs) == set(range(min(5, len(states))))
        counters = saved['H']
    else:
        assert not packed_path.exists() and not (work/filename).exists() and not (work/refs_file).exists()
        module.load(device)
        config.stc.close()
        linears = [m for m in module if isinstance(m, Linear) and m.qmap and m.device is not None]
        from mtp_calibration_cache import save_capture, load_capture
        import inspect
        if index==0:
            backend = {'flush_source_sha256':hashlib.sha256(inspect.getsource(cm._flush_capture_H).encode()).hexdigest(),
                       'choice':'original_fp32_bmm'}
        else:
            assert index==1
            from mtp_capture_identity import dynamic_flush_identity
            backend = {**dynamic_flush_identity(cm._flush_capture_H),
                       'choice':'qualified_lt_static_decoder'}
        if index==1:
            backend['dependencies']={name:digest(Path('/task')/name) for name in
                ('capture-static-contract.json','native/capture-gram.so',
                 'capture_static_runtime.py','capture_static_moe.py','capture_lean_gram.py','capture_fast.py',
                 'capture_lt.py','capture_owner.py','capture_repeat_counts.py')}
            backend['discarded_output_scope']='explicit_index1_all_expert_capture_only'
            backend['derived_gram_contract_sha256']=digest(args['mtp_gram_contract'])
            fc=args['mtp_input_fc_identity']
            assert fc['bits']=={key:strategy[key] for key in ('mtp.fc_hidden','mtp.fc_embedding')}, 'Decoder H requires actual input FC bits'
        capture_binding={'module':module.key,'index':index,'rows':len(states),'tokens':token_count,
                         'state_identity':args['mtp_state_identity'],
                         'context':args['mtp_capture_context'], 'math_code':candidate_identity(),
                         'backend':backend,'input_fc_identity':args.get('mtp_input_fc_identity') if index==1 else None}
        cache_path=Path(args.get('mtp_capture_cache_path') or work/f'{module.key}.capture')
        if resume_capture:
            capture,refs,_=load_capture(cache_path,capture_binding,strategy,args.get('mtp_capture_remap'))
            assert set(refs)==set(range(min(5,len(states))))
        else:
            assert not cache_path.exists(), 'Existing capture requires explicit resume/recovery; do not recapture'
            capture, refs = {}, {}
            if index==1:
                import capture_fast, capture_static_moe
                capture_stats_before={'fast':dict(capture_fast.STATS),'static':dict(capture_static_moe.STATS)}
            with cm.ProgressBar(f' -- MTP Capturing: {module.key}', len(states)) as progress:
                for i in range(len(states)):
                    progress.update(i)
                    params = {'attn_mode': 'flash_attn_nc', 'capture': capture,
                              'activate_all_experts': True, 'input_ids': ids[i], 'quant_preserve': {}}
                    x = module.prepare_for_device(states[i], params)
                    if index==1:
                        from capture_fast import _discarded_capture
                        with _discarded_capture(module,params):
                            y = module.forward(x, params)
                    else:
                        y = module.forward(x, params)
                    assert 'hc_pending' not in params, 'Deferred HC output is not a complete calibration state'
                    cm._flush_capture_H(params)
                    if i < 5:
                        params = {'attn_mode': 'flash_attn_nc', 'input_ids': ids[i], 'quant_preserve': {}}
                        x = module.prepare_for_device(states[i], params)
                        y = module.forward(x, params)
                        assert 'hc_pending' not in params
                        assert torch.isfinite(y).all()
                        refs[i] = y.cpu()
                    del x, y, params
            if index==1:
                scope_delta={kind:{key:value-capture_stats_before[kind][key] for key,value in stats.items()}
                             for kind,stats in [('fast',capture_fast.STATS),('static',capture_static_moe.STATS)]}
                assert scope_delta['fast']['scopes']==len(states)
                assert scope_delta['static']['scopes']==scope_delta['static']['static_routes']==len(states)
                write_json(work/'DECODER-CAPTURE-SCOPE.json',scope_delta)
        expected_maps = {m.qmap for m in linears}
        assert set(capture) == expected_maps, (set(capture) - expected_maps, expected_maps - set(capture))
        counters = {}
        for key, value in capture.items():
            count = int(value.get('count_dev', value['count']))
            assert count == expected_count(value['first_key'], token_count, hc_mult)
            assert int(value['inf_nan'].sum()) == 0
            counters[key] = {'count': count, 'first_key': value['first_key'],
                             'num_total': int(value['num_total'])}
            swap_captured_h_cpu(value,device,resume_capture)
        assert sum(int(x.numel()) for x in ids) == token_count
        if not resume_capture:
            save_capture(cache_path,capture,refs,capture_binding,strategy)
        cm.build_rwh_blends(linears, capture)
        for linear in linears:
            linear.inner.swap_cpu()
        torch.cuda.synchronize()
        with checked_quantization(Linear, token_count, hc_mult):
            cm.quantize_linears_parallel(args, linears, config, strategy, index,
                                        [0], None, capture, states)
        tensors = {}
        for m in module:
            tensors.update(m.get_tensors())
        module.unload()
        # The caller owns an isolated work tree. Never overwrite a completed stage.
        assert not (Path(args['work_dir'])/filename).exists()
        cm.save_tensor(tensors, filename, args)
        cm.save_tensor({str(i): value for i, value in refs.items()}, refs_file, args)
        packed_receipt = {**binding, 'H': counters,
                          'packed_sha256': digest(work/filename),
                          'refs_sha256': digest(work/refs_file)}
        write_json(packed_path, packed_receipt)
        del capture
    config.stc.set_new_tensors(tensors)
    module.load(device, source=tensors)
    config.stc.set_new_tensors(None)
    del tensors
    errors = []
    with cm.ProgressBar(f' -- MTP Forward: {module.key}', len(states) if advance_all else len(refs)) as progress:
        for i in range(len(states) if advance_all else len(refs)):
            progress.update(i)
            params = {'attn_mode': 'flash_attn_nc', 'input_ids': ids[i], 'quant_preserve': {}}
            x = module.prepare_for_device(states[i], params)
            y = module.forward(x, params)
            assert 'hc_pending' not in params
            assert torch.isfinite(y).all()
            if i in refs:
                rfn, cos, sq = cm.get_state_error(y, refs[i].to(y.device))
                assert all(math.isfinite(float(v)) for v in (rfn, cos, sq))
                errors.append({'row': i, 'rfn': float(rfn), 'cos': float(cos), 'sqnr': float(sq)})
            if advance_all:
                states[i] = y.cpu()
            del x, y, params
    module.unload()
    return {'module': module.key, 'seconds': time.time()-started,
            'rows': len(states), 'tokens': token_count, 'H': counters,
            'normal_reference_errors': errors, 'advanced_all_rows': advance_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['input', 'decoder'])
    resume_group=ap.add_mutually_exclusive_group()
    resume_group.add_argument('--resume-packed', action='store_true', help='Reuse committed packed stage; never rerun H/quantization')
    resume_group.add_argument('--resume-capture', action='store_true', help='Reuse committed pre-quant H/references; never rerun capture')
    ap.add_argument('--source-checkpoint', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--recipe', type=Path, required=True)
    ap.add_argument('--qualification', type=Path, required=True)
    ap.add_argument('--gram-contract', type=Path, required=True)
    ap.add_argument('--calibration', type=Path, default=Path('/task/c03-input-ids.json'))
    a = ap.parse_args()
    # Parent-runner contract. No safety/ownership inference from a lock pathname.
    assert os.environ.get('GUFO_MTP_LEASE_VERIFIED') == '1', 'Requires bounded parent GPU/JOB lease'
    assert os.environ.get('EXL3_HIP_F32OUT_VIA_F16') == '0'
    assert os.environ.get('EXL3_ENABLE_GROUPING') == '0'
    assert os.environ.get('EXL3_RWH') == '0'
    import torch
    from safetensors.torch import load_file
    from exllamav3 import Config, Model
    from exllamav3.conversion import convert_model as cm
    from exllamav3.modules import Linear
    from exllamav3.modules.linear import flush_capture_H
    from exllamav3.modules.attn import Attention
    from disk_state_large import DiskRows
    from mtp_borrowed_rows import BorrowedLargeRows
    from capture_progress import install as progress_install
    from capture_fast import install as fast_install
    from capture_owner import install as owner_install
    from capture_static_runtime import install_gram, install_moe
    source = a.source_checkpoint.resolve(strict=True)
    work = a.work.resolve()
    assert source != work and source not in work.parents and work not in source.parents
    job = json.loads((source/'job.json').read_text())
    assert job == {'next_module_idx': 51, 'q_strategy': None, 'bad_rows': []}
    provenance = {name: digest(source/name) for name in ('job.json','state.rows.json','original_input_ids.safetensors')}
    calibration = json.loads(a.calibration.read_text())
    assert calibration['rows'] == 2309 and calibration['total_tokens'] == 1999496
    qualification = json.loads(a.qualification.read_text())
    assert qualification['status'] == 'qualified_for_full_mtp_calibration'
    assert qualification['target_semantics']==SEMANTICS and qualification['gufo_input_fc_gpu_validated'] is True
    assert qualification['gram_contract_sha256']==digest(a.gram_contract)
    assert qualification['calibration_sha256'] == calibration['sha256']
    assert qualification['recipe_sha256'] == digest(a.recipe)
    for filename in ('mtp_calibration_contract.py', 'mtp_calibration_candidate.py', 'mtp_calibration_cache.py', 'mtp_calibration_gufo.py', 'mtp_borrowed_rows.py', 'mtp_capture_identity.py'):
        assert qualification['code_sha256'][filename] == digest(Path(__file__).with_name(filename))
    effective_tokens=calibration['total_tokens']-calibration['rows']
    assert effective_tokens==1997187
    recipe_document = json.loads(a.recipe.read_text()); strategy = recipe_document['tensors']
    assert len(strategy) == 1546
    receipt = {'source': str(source), 'source_identity': provenance, 'recipe_sha256': digest(a.recipe),
               'calibration_sha256': calibration['sha256'], 'alignment': SEMANTICS,
               'total_tokens': effective_tokens, 'source_total_tokens':calibration['total_tokens'], 'rows': calibration['rows'],
               'code_sha256': candidate_identity(), 'qualification_sha256': digest(a.qualification),
               'input_fc_bits':{key:strategy[key] for key in ('mtp.fc_hidden','mtp.fc_embedding')}}
    if a.stage == 'input' and not (a.resume_packed or a.resume_capture):
        work.mkdir()
        (work/'qtensors').mkdir(); (work/'scratch').mkdir(); (work/'input-checkpoint').mkdir()
        write_json(work/'INPUTS.json', receipt)
        state_path = source/'state.safetensors'
    else:
        assert json.loads((work/'INPUTS.json').read_text()) == receipt
        if a.stage == 'input':
            assert not (work/'INPUT-COMPLETE.json').exists()
            assert not (work/'input-checkpoint/state.rows.json').exists(), 'Recover completed checkpoint explicitly'
            state_path = source/'state.safetensors'
        else:
            assert json.loads((work/'INPUT-COMPLETE.json').read_text())['status'] == 'input_calibrated'
            assert not (work/'DECODER-COMPLETE.json').exists()
            state_path = work/'input-checkpoint/state.safetensors'
            assert json.loads((work/'INPUT-COMPLETE.json').read_text())['derived_state_identity'] == rows_identity(state_path)
    args = json.loads((source.parent/'args.json').read_text())
    args.update(work_dir=str(work), image_dump=False, verbose=False, mtp_state_identity=rows_identity(state_path),
                mtp_qualification_sha256=digest(a.qualification),
                mtp_capture_context={k:receipt[k] for k in ('source_identity','calibration_sha256','alignment','total_tokens','rows')})
    args['mtp_capture_context']['weights_runtime_arithmetic']=capture_provenance(cm)
    args['mtp_gram_contract']=str(a.gram_contract)
    if a.stage=='decoder':
        completed_input=json.loads((work/'INPUT-COMPLETE.json').read_text())
        assert completed_input['input_fc_pack_sha256']==digest(work/'qtensors/mtp.safetensors')
        args['mtp_input_fc_identity']={k:completed_input[k] for k in ('input_fc_pack_sha256','derived_state_identity')}
        args['mtp_input_fc_identity']['bits']=completed_input['input_fc_bits']
    ids = cm.load_tensor('ckpt/original_input_ids.safetensors', {'work_dir': str(source.parent)})
    assert len(ids) == calibration['rows'] and sum(x.numel() for x in ids) == calibration['total_tokens']
    ids_path = a.calibration.with_suffix('.safetensors')
    assert digest(ids_path) == calibration['sha256']
    original_ids = load_file(str(ids_path))
    assert set(original_ids) == {f'input_ids.{i}' for i in range(len(ids))}
    for i, row in enumerate(ids):
        assert row.dtype == torch.int64 and row.shape == (1, calibration['sequences'][i]['tokens'])
        assert torch.equal(row, original_ids[f'input_ids.{i}'].to(torch.int64))
    del original_ids
    source_lengths=[row.shape[1] for row in ids]
    ids=[row[:,1:].contiguous() for row in ids]
    assert sum(row.numel() for row in ids)==effective_tokens
    config = Config.from_directory('/source'); trunk = Model.from_config(config)
    mtp = Model.from_config(config, component='mtp')
    assert config.hc_mult == 4 and config.hidden_size == 2560
    assert len(mtp.modules) == 3 and mtp.modules[1].key == 'mtp.layers.0'
    configure_mtp(mtp, strategy, Linear)
    mtp.attach_to(trunk)
    clear_adapter=install_input_adapter(mtp.input_layer)
    stores = (BorrowedLargeRows if a.stage == 'input' else DiskRows)(state_path, work/'scratch')
    assert len(stores) == calibration['rows']
    assert all(dtype == 'F32' and shape == (1,source_lengths[i] if a.stage=='input' else source_lengths[i]-1,4,2560)
               for i, (dtype, shape) in enumerate(stores.infos)), 'Unexpected HC state row layout'
    # Inherited close joins I/O then cleans its TemporaryDirectory even when
    # joining fails. Disable both explicit and GC cleanup until successful commit.
    successful_cleanup = preserve_scratch_until_commit(stores)
    restore_progress = progress_install(cm)
    failure = None
    try:
        with torch.inference_mode():
            if a.stage == 'input':
                assert cm._flush_capture_H is flush_capture_H, 'Input FC selects original FP32 Gram explicitly'
                trunk.modules[0].load(torch.device('cuda:0'))
                result = run_module(cm, mtp.modules[0], config, stores, ids, args, strategy, 0,
                                    effective_tokens, config.hc_mult, advance_all=True, resume_packed=a.resume_packed, resume_capture=a.resume_capture)
                trunk.modules[0].unload()
                stores.save_checkpoint(work/'input-checkpoint/state.safetensors')
                result.update(status='input_calibrated', gram_backend='original_fp32_bmm',
                              derived_state_identity=rows_identity(work/'input-checkpoint/state.safetensors'),
                              input_fc_bits=receipt['input_fc_bits'],
                              input_fc_pack_sha256=digest(work/'qtensors/mtp.safetensors'))
                write_json(work/'INPUT-COMPLETE.json', result)
            else:
                fast_install(cm)
                with contextlib.ExitStack() as stack:
                    stack.callback(owner_install(Attention))
                    stack.callback(install_decoder_gram(cm,a.gram_contract,'/task/native/capture-gram.so',calibration))
                    stack.callback(install_moe('/task/capture-static-contract.json'))
                    result = run_module(cm, mtp.modules[1], config, stores, ids, args, strategy, 1,
                                        effective_tokens, config.hc_mult, advance_all=False, resume_packed=a.resume_packed, resume_capture=a.resume_capture)
                terminal = mtp.modules[2]
                terminal.load(torch.device('cpu'))
                tensors = {}
                for m in terminal:
                    tensors.update(m.get_tensors())
                assert tensors and all(not isinstance(m, Linear) or not m.qmap for m in terminal)
                filename = 'qtensors/mtp.hyper_connection_mixer.safetensors'
                assert not (work/filename).exists()
                cm.save_tensor(tensors, filename, args)
                terminal.unload()
                result.update(status='mtp_calibrated_pending_full_validation', gram_backend='qualified_lt_static_decoder')
                write_json(work/'DECODER-COMPLETE.json', result)
    except BaseException as error:
        failure = {'time': time.time(), 'stage': a.stage, 'error': repr(error),
                   'scratch': stores.scratch.name,
                   'resume_policy': 'explicit --resume-packed reuses committed pack/H work; retained partial rows require separate audited CPU recovery'}
        # Preserve complete/partial files for diagnosis, while close still joins
        # I/O workers and closes descriptors. Never delete a failed stage's rows.
        raise
    finally:
        restore_progress()
        clear_adapter()
        try:
            stores.close()
        except BaseException as close_error:
            if failure is None:
                failure = {'time':time.time(),'stage':a.stage,'error':repr(close_error),
                           'scratch':stores.scratch.name,'resume_policy':'explicit CPU recovery required'}
                raise
            failure['close_error'] = repr(close_error)
        finally:
            if failure is not None:
                failure['changed_rows'] = sorted(stores.changed)
                failure['row_infos'] = stores.infos
                write_json(work/f'FAILURE-{time.time_ns()}.json', failure)
            assert provenance == {name: digest(source/name) for name in provenance}, 'Source checkpoint metadata changed'
            if failure is None:
                successful_cleanup()


if __name__ == '__main__':
    main()
