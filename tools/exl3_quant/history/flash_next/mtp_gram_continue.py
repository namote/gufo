"""Eight unexecuted Gram edge checks; immutable earlier support/control reused.

The earlier 1e-5 failure stays failed. This explicitly approved follow-up applies
3e-5 as a separate numerical-screen flag, never a full calibration qualification.
"""
import hashlib
import json
from pathlib import Path
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schedule(shapes):
    patterns=sorted({(b,d,h) for b,_,d,h in shapes})
    return [[b,r,d,h] for r in (min(s[1] for s in shapes),max(s[1] for s in shapes)) for b,d,h in patterns]


def validate_prior(prior, continuation, pending, calibration, old):
    from mtp_calibration_gufo import SEMANTICS,derived_gram_shapes
    assert continuation['status']=='prepared_for_independent_bounded_followup_review'
    assert continuation['prior_root']==str(prior)
    assert continuation['original_limit']==1e-5 and continuation['followup_limit']==3e-5
    assert continuation['new_seed_base']==81303
    for filename,expected in continuation['prior_sha256'].items():
        assert Path(filename).name==filename and sha(prior/filename)==expected
    assert set(continuation['prior_sha256'])=={'STATE.json','FAILURE.json','NUMERIC.json','SUPPORT.json','SAVED-CPU-DIAGNOSIS.json','failed-numeric.pt'}
    failure=json.loads((prior/'FAILURE.json').read_text())
    assert failure['status']=='failed' and failure['support_done']==1510 and failure['numeric_done']==12
    assert failure['code_sha256']=='da144e9991d5db0d5c533669757aa46334cfdbd11321da892b45e9bb98ec68ae'
    assert failure['pending_sha256']==continuation['pending_sha256']
    assert pending['semantics']==SEMANTICS and pending['allowed_shapes']==derived_gram_shapes(calibration,old)
    support=json.loads((prior/'SUPPORT.json').read_text())
    assert len(support)==1510 and [r['shape'] for r in support]==pending['allowed_shapes']
    assert all(r['supported'] and r['code']==0 and r['algorithms']>r['selected'] for r in support)
    numeric=json.loads((prior/'NUMERIC.json').read_text());all_shapes=schedule(pending['allowed_shapes'])
    assert len(all_shapes)==20 and len(numeric)==12 and [r['shape'] for r in numeric]==all_shapes[:12]
    assert numeric[-1]['shape']==[1,8191,640,True]
    assert all(r['repeat_exact'] for r in numeric)
    assert all(r['control']['relative_l2']<=1e-5 and r['sample_fp64']['relative_l2']<=1e-5 for r in numeric[:11])
    assert numeric[-1]['control']['relative_l2']>1e-5 and numeric[-1]['sample_fp64']['relative_l2']>1e-5
    assert all(r['control']['relative_l2']<=3e-5 and r['sample_fp64']['relative_l2']<=3e-5 for r in numeric)
    assert continuation['remaining_shapes']==all_shapes[12:]
    return numeric,all_shapes[12:]


def run(a):
    from mtp_qualify_gram import write
    from mtp_calibration_gufo import SEMANTICS
    continuation=json.loads(a.continuation_contract.read_text())
    pending=json.loads(a.pending.read_text());cal=json.loads(Path('/task/c03-input-ids.json').read_text())
    old=json.loads(Path('/task/capture-lt-contract.json').read_text())
    assert sha(a.pending)==continuation['pending_sha256']
    assert sha(a.library)==pending['native_sha256']==continuation['native_sha256']
    prior_numeric,remaining=validate_prior(a.continue_from,continuation,pending,cal,old)
    out=a.output;assert out.resolve()!=a.continue_from.resolve();out.mkdir()
    import torch
    from capture_lean_gram import PreparedLtGram
    started=time.monotonic();native=PreparedLtGram(a.library);numeric=[]
    result={'status':'running','semantics':SEMANTICS,'original_strict_limit':1e-5,
            'original_strict_status':'failed_preserved','followup_limit':3e-5,
            'support_reused':1510,'numeric_reused':12,'new_numeric_expected':8,
            'prior_sha256':continuation['prior_sha256'],'continuation_sha256':sha(a.continuation_contract),
            'code_sha256':{'mtp_qualify_gram.py':sha(Path(__file__).with_name('mtp_qualify_gram.py')),'mtp_gram_continue.py':sha(__file__)},
            'native_sha256':sha(a.library),'no_whole_model_quality_or_speed_claim':True}
    write(out/'STATE.json',result)
    def diff(ref,y):
        assert torch.isfinite(ref).all() and torch.isfinite(y).all()
        delta=y.double()-ref.double()
        return {'relative_l2':float(delta.norm()/ref.double().norm().clamp_min(1e-30)),
                'max_abs':float(delta.abs().max()),'reference_rms':float(ref.double().square().mean().sqrt()),
                'candidate_rms':float(y.double().square().mean().sqrt()),'exact':torch.equal(ref.view(torch.uint8),y.view(torch.uint8))}
    try:
        with torch.inference_mode():
            for ordinal,shape in enumerate(remaining,12):
                assert time.monotonic()-started<a.max_seconds
                b,r,d,half=shape;seed=continuation['new_seed_base']+ordinal
                generator=torch.Generator(device='cuda').manual_seed(seed)
                x=(torch.randn((b,r,d),generator=generator,device='cuda')*.25).to(torch.float16 if half else torch.float32)
                reference=torch.bmm(x.float().transpose(1,2),x.float())
                y=native(x);torch.cuda.synchronize()
                # Preserve new operands before any failing finite/repeat gate.
                try:
                    control=diff(reference,y)
                    pos=torch.linspace(0,d-1,8,device=x.device).long();teacher=x[:min(b,2),:,pos].double();teacher=teacher.transpose(1,2)@teacher
                    fp64=diff(teacher,y[:min(b,2)][:,pos][:,:,pos].double())
                    repeat=native(x);torch.cuda.synchronize();exact=torch.equal(y.view(torch.uint8),repeat.view(torch.uint8))
                except BaseException:
                    torch.save({'x':x.cpu(),'control':reference.cpu(),'candidate':y.cpu()},out/f'failed-{ordinal}.pt')
                    raise
                strict=control['relative_l2']<=1e-5 and fp64['relative_l2']<=1e-5
                followup=control['relative_l2']<=3e-5 and fp64['relative_l2']<=3e-5 and exact
                entry={'ordinal':ordinal,'shape':shape,'seed':seed,'control':control,'sample_fp64':fp64,
                       'repeat_exact':exact,'original_strict_pass':strict,'followup_screen_pass':followup}
                numeric.append(entry);write(out/'NEW-NUMERIC.json',numeric)
                if not followup:
                    torch.save({'x':x.cpu(),'control':reference.cpu(),'candidate':y.cpu()},out/f'failed-{ordinal}.pt')
                assert exact,'Repeat mismatch; preserve failed input'
                del x,reference,y,repeat,teacher
        result.update(status='completed_pending_independent_review',new_numeric_done=len(numeric),
                      followup_screen_all_pass=all(r['followup_screen_pass'] for r in numeric),
                      new_strict_failure_shapes=[r['shape'] for r in numeric if not r['original_strict_pass']],
                      original_failed_shape=prior_numeric[-1]['shape'],seconds=time.monotonic()-started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated())
        write(out/'SUMMARY.json',result)
    except BaseException as e:
        result.update(status='failed',error=repr(e),new_numeric_done=len(numeric));write(out/'FAILURE.json',result);raise
    finally:
        native.close();write(out/'STATE.json',result)
