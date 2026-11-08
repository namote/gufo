"""Bounded real-input probe of the same MTP candidate, never auto-qualified.

Parent supplies exclusive GPU/JOB lease and <=1200s/resource guards. Reads two
whole c03 rows (512 +8192 tokens) from protected idx51; never replays trunk.
This is a plumbing/numerical screen, not full calibration quality equivalence.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import time

from mtp_calibration_candidate import run_module, write_json, digest, capture_provenance
from mtp_calibration_contract import configure_mtp
from mtp_calibration_gufo import SEMANTICS, install_input_adapter, prepare_gufo_prompt, install_decoder_gram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-checkpoint', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--recipe', type=Path, required=True)
    ap.add_argument('--gram-contract', type=Path, required=True)
    a = ap.parse_args()
    assert os.environ.get('GUFO_MTP_LEASE_VERIFIED') == '1'
    assert os.environ.get('EXL3_HIP_F32OUT_VIA_F16') == '0'
    assert os.environ.get('EXL3_ENABLE_GROUPING') == '0' and os.environ.get('EXL3_RWH') == '0'
    import torch
    from safetensors.torch import load_file
    from exllamav3 import Config, Model
    from exllamav3.conversion import convert_model as cm
    from exllamav3.modules import Linear
    from exllamav3.modules.attn import Attention
    from exllamav3.modules.linear import flush_capture_H
    from mtp_borrowed_rows import BorrowedRows as DiskRows
    from capture_progress import install as progress_install
    from capture_fast import install as fast_install
    from capture_owner import install as owner_install
    from capture_static_runtime import install_gram, install_moe
    source = a.source_checkpoint.resolve(strict=True); work = a.work.resolve()
    assert source != work and source not in work.parents and work not in source.parents
    assert json.loads((source/'job.json').read_text()) == {'next_module_idx':51,'q_strategy':None,'bad_rows':[]}
    provenance = {name:digest(source/name) for name in ('job.json','state.rows.json','original_input_ids.safetensors')}
    calibration = json.loads(Path('/task/c03-input-ids.json').read_text())
    indices = [next(i for i,s in enumerate(calibration['sequences']) if s['tokens']==length) for length in (512,8192)]
    work.mkdir(); (work/'scratch').mkdir(); (work/'qtensors').mkdir()
    inputs = {'source_identity':provenance,'row_indices':indices,'tokens':8702,'source_tokens':8704,'alignment':SEMANTICS,
              'calibration_sha256':calibration['sha256'],'recipe_sha256':digest(a.recipe),
              'code_sha256':{f:digest(Path(__file__).with_name(f)) for f in
                            ('mtp_calibration_contract.py','mtp_calibration_candidate.py','mtp_calibration_probe.py','mtp_calibration_cache.py','mtp_calibration_gufo.py','mtp_borrowed_rows.py')}}
    write_json(work/'INPUTS.json',inputs)
    all_ids = cm.load_tensor('ckpt/original_input_ids.safetensors',{'work_dir':str(source.parent)})
    original_ids = load_file('/task/c03-input-ids.safetensors')
    assert digest('/task/c03-input-ids.safetensors') == calibration['sha256']
    ids = [all_ids[i] for i in indices]
    for index, row in zip(indices, ids):
        assert torch.equal(row, original_ids[f'input_ids.{index}'].to(torch.int64))
    original = DiskRows(source/'state.safetensors',work/'scratch')
    states = [original[i].clone() for i in indices]
    source_values = [s.clone() for s in states]
    ids=[row[:,1:].contiguous() for row in ids]
    for state,row in zip(states,ids):
        expected=state.index_select(1,torch.arange(row.shape[1])).contiguous()
        got=prepare_gufo_prompt(state,row)
        assert got.dtype==torch.float32 and torch.equal(got.view(torch.uint8),expected.view(torch.uint8))
    args = json.loads((source.parent/'args.json').read_text());args.update(work_dir=str(work),image_dump=False,verbose=False)
    args['mtp_gram_contract']=str(a.gram_contract)
    args['mtp_state_identity'] = {k:inputs[k] for k in ('source_identity','row_indices')}
    args['mtp_capture_context'] = {k:inputs[k] for k in ('source_identity','row_indices','tokens','calibration_sha256')}
    args['mtp_capture_context']['weights_runtime_arithmetic']=capture_provenance(cm)
    strategy = json.loads(a.recipe.read_text())['tensors']
    config = Config.from_directory('/source');trunk = Model.from_config(config);mtp = Model.from_config(config,component='mtp')
    configure_mtp(mtp,strategy,Linear);mtp.attach_to(trunk);clear_adapter=install_input_adapter(mtp.input_layer)
    started=time.time();restore_progress=progress_install(cm)
    result={'status':'running','rows':2,'tokens':8702,'source_tokens':8704,'alignment':SEMANTICS,'row_indices':indices}
    write_json(work/'STATE.json',result)
    try:
        with torch.inference_mode():
            assert cm._flush_capture_H is flush_capture_H
            trunk.modules[0].load(torch.device('cuda:0'))
            input_result=run_module(cm,mtp.modules[0],config,states,ids,args,strategy,0,8702,4,advance_all=True)
            trunk.modules[0].unload()
            args['mtp_state_identity'] = {'probe_input_pack_sha256':digest(work/'qtensors/mtp.safetensors')}
            args['mtp_input_fc_identity'] = {'bits':{key:strategy[key] for key in ('mtp.fc_hidden','mtp.fc_embedding')},
                                             'input_fc_pack_sha256':digest(work/'qtensors/mtp.safetensors')}
            fast_install(cm)
            with contextlib.ExitStack() as stack:
                stack.callback(owner_install(Attention))
                stack.callback(install_decoder_gram(cm,a.gram_contract,'/task/native/capture-gram.so',calibration))
                stack.callback(install_moe('/task/capture-static-contract.json'))
                decoder_result=run_module(cm,mtp.modules[1],config,states,ids,args,strategy,1,8702,4,advance_all=False)
            terminal=mtp.modules[2];terminal.load(torch.device('cpu'))
            terminal_tensors={}
            for module in terminal:
                terminal_tensors.update(module.get_tensors())
            assert terminal_tensors and all(k.startswith('mtp.hyper_connection_mixer.') for k in terminal_tensors)
            assert all(not isinstance(m,Linear) or not m.qmap for m in terminal)
            cm.save_tensor(terminal_tensors,'qtensors/mtp.hyper_connection_mixer.safetensors',args)
            terminal.unload()
            for index, before in zip(indices,source_values):
                after=original[index]
                assert torch.equal(before.view(torch.uint8),after.view(torch.uint8))
            assert provenance == {name:digest(source/name) for name in provenance}
            result.update(status='completed_pending_review',input=input_result,decoder=decoder_result,
                          seconds=time.time()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                          source_unchanged=True,alignment_exact=True,
                          qualification_scope='two_real_rows_only_not_full_model_quality',
                          terminal_tensor_keys=sorted(terminal_tensors))
            write_json(work/'SUMMARY.json',result)
    except BaseException as error:
        result.update(status='failed',error=repr(error),seconds=time.time()-started)
        write_json(work/'FAILURE.json',result)
        raise
    finally:
        restore_progress();clear_adapter();original.close();write_json(work/'STATE.json',result)


if __name__=='__main__':
    main()
