"""CPU-only prepare pending final assembly evidence after actual main completion.

Does not launch work, approve the run, mutate checkpoints or hash large payloads.
Parent must review/change pending statuses and refresh dependent contract pins.
"""
import json,time
from pathlib import Path
from run_stage import OUT,TASK,RUNTIME,save
from mtp_final_compile_job import sha,stat_id,required_pins,required_disk,RECEIPTS
from mtp_final_compile_candidate import (tensor_map,validate_mtp,validate_causal_mtp_receipts,
                                        validate_calibration_identity)
from mtp_calibration_candidate import rows_identity


def main():
    main_work=OUT/'work';mtp=OUT/'mtp-calibration-work/full/work'
    assert json.loads((main_work/'ckpt/job.json').read_text())=={'next_module_idx':53,'q_strategy':None,'bad_rows':[]}
    marker=OUT/'MAIN-CONVERSION.json';args=OUT/'MAIN-COMPILE-ARGS.json'
    mc=json.loads(marker.read_text());assert mc['status']=='main_complete_side_models_skipped'
    assert mc['skip_side_models'] is True and mc['vision_bits']==16
    full_path=OUT/'mtp-full-decoder-review/REVIEW.json';full=json.loads(full_path.read_text())
    assert full['status']=='reviewed_full_mtp_artifacts_for_final_assembly' and not full['findings']
    assert full['source_unchanged'] and not full['cleanup_errors'] and not full['returned_KFD_contexts']
    protected=OUT/'mtp-protected-idx51/ckpt';preservation=OUT/'mtp-protected-idx51/PRESERVATION.json'
    p=json.loads(preservation.read_text());assert p['checkpoint']==str(protected)
    protected_identity=rows_identity(protected/'state.safetensors')
    assert protected_identity==p['rows_identity']==full['source_rows_identity']
    assert json.loads((protected/'job.json').read_text())=={'next_module_idx':51,'q_strategy':None,'bad_rows':[]}
    recipe_path=OUT/'MTP-MIXED-BITS-PROPOSAL.json';recipe=json.loads(recipe_path.read_text())['tensors']
    assert len(recipe)==1546
    receipt_hashes={name:sha(mtp/name) for name in RECEIPTS}
    assert receipt_hashes==full['receipt_sha256']
    receipts={name:json.loads((mtp/name).read_text()) for name in RECEIPTS}
    validate_causal_mtp_receipts(receipts,receipt_hashes,full['pack_sha256'],recipe,protected_identity)
    cal=json.loads((TASK/'c03-input-ids.json').read_text())
    validate_calibration_identity(receipts['INPUTS.json'],sha(recipe_path),mc,cal)
    mtp_files=[mtp/'qtensors'/n for n in ('mtp.safetensors','mtp.layers.0.safetensors','mtp.hyper_connection_mixer.safetensors')]
    mtp_map=tensor_map(mtp_files);validate_mtp(mtp_map,recipe)
    main_files=sorted((main_work/'qtensors').glob('*.safetensors'));assert main_files
    main_map=tensor_map(main_files);assert not any(k.startswith(('mtp.','model.visual.')) for k in main_map)
    assert 'lm_head.trellis' in main_map
    assert all('model.language_model.hyper_connection_mixer.'+k in main_map for k in ('hc_norm.weight','input_mix_weight_down.weight','input_mix_weight_up.weight'))
    ngram=OUT/'pack/ngram_embedding.safetensors';assert ngram.is_file()
    def container(p):return '/output/'+str(p.relative_to(OUT))
    artifacts={container(p):stat_id(p) for p in main_files+mtp_files}
    artifacts['/mtp-run/ngram_embedding.safetensors']=stat_id(ngram)
    runtime={n:sha(RUNTIME/'source/exllamav3/conversion'/(n+'.py')) for n in ('convert_model','compile')}
    base=OUT/'mtp-final-assembly-runner';contracts=base/'contracts';contracts.mkdir(exist_ok=True)
    assembly_path=base/'assembly-final001.json';contract_path=contracts/'final001.json'
    assert not assembly_path.exists() and not contract_path.exists(),'Preserve existing drafts'
    r=dict(status='pending_independent_final_mtp_vision_compile_review',prepared_by='/root/quant_stall_review',time=time.time(),
       candidate_sha256=sha(TASK/'mtp_final_compile_candidate.py'),recipe_sha256=sha(recipe_path),
       main_receipt_sha256=sha(marker),main_args_sha256=sha(args),mtp_complete_sha256=receipt_hashes['DECODER-COMPLETE.json'],
       protected_idx51={'checkpoint':container(protected),'rows_identity':protected_identity},
       mtp_files_sha256=full['pack_sha256'],mtp_receipts_sha256=receipt_hashes,
       mtp_cache_payload_stats={n:stat_id(mtp/n/'numeric.safetensors') for n in ('mtp.capture','mtp.layers.0.capture')},
       artifact_stats=artifacts,runtime_sha256=runtime,
       evidence_sha256={str(p):sha(p) for p in [full_path,preservation,OUT/'mtp-protected-idx51/REVIEW.json',base/'REVIEW.json',base/'SUPPLEMENTAL-REVIEW.json']},
       pack_digest_basis='Reuse independently audited full MTP pack SHA; current headers/stat pinned; GPU candidate rechecks actual SHA before compile',
       CPU_checks=['Actual main idx53 marker and args','Protected independent idx51 actual row headers/inode identity',
                   'All seven exact receipt SHAs and causal chain','Mixed1546 actual packheaders','Mainqtensors exclude MTP/vision',
                   'Calibration source/recipe identity','Live sourceartifact/H metadata, fixed runtime'],
       disk_budget=required_disk(main_files,mtp_files),limitations=['Assembly not yet run','No final inference quality/vision behavior/actual-token throughput validation'])
    save(assembly_path,r)
    c=dict(status='pending_independent_final_assembly_review',purpose='assembly_not_benchmark',review=str(contracts/'final001.REVIEW.json'),
       cpu=str(base/'CPU.json'),assembly_review=container(assembly_path),main_work='/output/work',
       mtp_work='/output/mtp-calibration-work/full/work',recipe=container(recipe_path),ngram_source=container(ngram),
       receipt_dir=str(OUT/'mtp-final-assembly/runs/final001'),data_dir=str(OUT/'mtp-final-assembly/data/final001'),seconds=1200)
    paths=required_pins(c)|set(r['evidence_sha256'])|{str(marker),str(args),str(protected/'job.json'),str(protected/'state.rows.json'),str(TASK/'prepare_final_assembly_contract.py')}
    paths|={str(mtp/n) for n in RECEIPTS}
    paths|={str(RUNTIME/'source/exllamav3/conversion'/(n+'.py')) for n in ('convert_model','compile')}
    c['identity']={str(p):sha(p) for p in sorted(paths)}
    # Freeze itself is pinned; reuse its independently frozen source digests without
    # a repeated large-code payload scan in this metadata-only preparation step.
    frozen=json.loads((OUT/'model-job-c03/FROZEN.json').read_text())
    for path,digest in frozen.items():
        if path in c['identity']:assert c['identity'][path]==digest,path
        c['identity'][path]=digest
    save(contract_path,c)
    save(base/'final001-PREPARATION.json',dict(status='pending_parent_review_no_gpu',assembly_review_sha256=sha(assembly_path),
       contract_sha256=sha(contract_path),identity_pins=len(c['identity']),disk_budget=r['disk_budget'],
       next='Parent independently approves assembly status, refreshes assembly identity in contract, approves contract status, then writes contract-hash per-run review; no automatic launch'))
    print(json.dumps({'assembly_review':str(assembly_path),'assembly_sha256':sha(assembly_path),'contract':str(contract_path),'contract_sha256':sha(contract_path),'pins':len(c['identity'])}))


if __name__=='__main__':main()
