"""Bounded shifted-row Lt support/numerical screen. Never self-qualifies a contract.

Run under parent's exclusive GPU/JOB lease. Reuses existing capture-gram.so;
creates/destroys all required plans serially, runs only edge lengths per pattern.
"""
import argparse,ctypes as C,hashlib,json,os,time
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pending',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-seconds',type=int,default=900)
    ap.add_argument('--library',type=Path,default=Path('/task/native/capture-gram.so'))
    ap.add_argument('--continue-from',type=Path)
    ap.add_argument('--continuation-contract',type=Path)
    a=ap.parse_args();assert os.environ.get('GUFO_MTP_LEASE_VERIFIED')=='1'
    if a.continue_from is not None:
        assert a.continuation_contract is not None
        from mtp_gram_continue import run
        return run(a)
    assert a.continuation_contract is None
    import torch
    from capture_lean_gram import PreparedLtGram
    from mtp_calibration_gufo import SEMANTICS,derived_gram_shapes
    started=time.monotonic();out=a.output;out.mkdir()
    contract=json.loads(a.pending.read_text());cal=json.loads(Path('/task/c03-input-ids.json').read_text())
    old=json.loads(Path('/task/capture-lt-contract.json').read_text())
    assert contract['status']=='pending_operator_discovery_and_numeric_qualification'
    assert contract['semantics']==SEMANTICS and contract['allowed_shapes']==derived_gram_shapes(cal,old)
    assert contract['native_sha256']==sha(a.library)
    native=PreparedLtGram(a.library);lib=native.lib;support=[];numeric=[]
    def guard():assert time.monotonic()-started<a.max_seconds,'Bounded Gram deadline exceeded'
    def diff(ref,value):
        assert torch.isfinite(ref).all() and torch.isfinite(value).all()
        d=value.double()-ref.double()
        return {'relative_l2':float(d.norm()/ref.double().norm().clamp_min(1e-30)),
                'max_abs':float(d.abs().max()),'reference_rms':float(ref.double().square().mean().sqrt()),
                'candidate_rms':float(value.double().square().mean().sqrt()),'exact':torch.equal(ref.view(torch.uint8),value.view(torch.uint8))}
    result={'status':'running','native_sha256':sha(a.library),'pending_sha256':sha(a.pending),'semantics':SEMANTICS,
            'code_sha256':sha(__file__),'calibration_sha256':cal['sha256']}
    write(out/'STATE.json',result)
    try:
        # Descriptor/heuristic discovery only; never allocate all plans/workspaces together.
        for shape in contract['allowed_shapes']:
            guard();b,r,d,half=shape;code=C.c_int();found=C.c_int();p=None
            try:
                p=lib.gram_lt_create(b,r,d,int(half),C.byref(code),C.byref(found))
                selected=1 if half else 3
                entry={'shape':shape,'code':code.value,'algorithms':found.value,'selected':selected,
                       'supported':bool(p) and code.value==0 and found.value>selected}
                support.append(entry)
                assert entry['supported'],entry
            finally:
                if p:lib.gram_lt_destroy(p)
            if len(support)%100==0:write(out/'PROGRESS.json',{'support_done':len(support),'numeric_done':len(numeric)})
        write(out/'SUPPORT.json',support)
        patterns=sorted({(b,d,h) for b,_,d,h in contract['allowed_shapes']})
        edges=(min(s[1] for s in contract['allowed_shapes']),max(s[1] for s in contract['allowed_shapes']))
        generator=torch.Generator(device='cuda').manual_seed(71303)
        with torch.inference_mode():
            for rows in edges:
                for b,d,half in patterns:
                    guard();x=(torch.randn((b,rows,d),generator=generator,device='cuda')*.25).to(torch.float16 if half else torch.float32)
                    reference=torch.bmm(x.float().transpose(1,2),x.float())
                    t=time.monotonic();y=native(x);torch.cuda.synchronize();seconds=time.monotonic()-t
                    comparison=diff(reference,y)
                    pos=torch.linspace(0,d-1,8,device=x.device).long();teacher=x[:min(b,2),:,pos].double()
                    teacher=teacher.transpose(1,2)@teacher
                    teacher_error=diff(teacher,y[:min(b,2)][:,pos][:,:,pos].double())
                    repeat=native(x);torch.cuda.synchronize()
                    assert torch.equal(y.view(torch.uint8),repeat.view(torch.uint8))
                    entry={'shape':[b,rows,d,half],'control':comparison,'sample_fp64':teacher_error,
                           'repeat_exact':True,'single_call_seconds':seconds}
                    numeric.append(entry);write(out/'NUMERIC.json',numeric)
                    passed=comparison['relative_l2']<=1e-5 and teacher_error['relative_l2']<=1e-5
                    if not passed:torch.save({'x':x.cpu(),'control':reference.cpu(),'candidate':y.cpu()},out/'failed-numeric.pt')
                    assert passed,entry
                    del x,reference,y,repeat,teacher
        result.update(status='completed_pending_independent_review',support_count=len(support),numeric_count=len(numeric),
                      scope='All plan discovery; only20 synthetic edge-shape numeric checks. No whole model or capture speed claim.',
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started)
        write(out/'SUMMARY.json',result)
    except BaseException as e:
        result.update(status='failed',error=repr(e),support_done=len(support),numeric_done=len(numeric));write(out/'FAILURE.json',result)
        raise
    finally:
        native.close();write(out/'STATE.json',result)

if __name__=='__main__':main()
