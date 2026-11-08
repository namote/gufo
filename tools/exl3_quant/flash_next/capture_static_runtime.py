"""Qualified capture traffic removal, retaining native Gram shape/precision guards."""
import hashlib,json
from pathlib import Path
import torch
from capture_lean_gram import PreparedLtGram,make_flush
from capture_static_moe import installed,STATS

class QualifiedPrepared:
    def __init__(self,contract,library):
        assert contract['status']=='qualified_for_idx10_resume'
        assert hashlib.sha256(Path(library).read_bytes()).hexdigest()==contract['native_sha256']
        self.allowed={tuple(s) for s in contract['allowed_shapes']}
        self.native=PreparedLtGram(library)
        self.stats=dict(half_calls=0,fp32_calls=0);self.observed=set()
    def __call__(self,x):
        assert x.dtype in (torch.float16,torch.float32)
        b,r,d=x.shape;half=x.dtype==torch.float16;sig=(b,r,d,half)
        assert sig in self.allowed,('Unqualified Gram shape/dtype',sig,str(x.dtype))
        self.stats['half_calls' if half else 'fp32_calls']+=1
        if not self.observed:
            print('GUFO_LT_ACTIVE',json.dumps(dict(shape=sig,prepared_dtype=str(x.dtype),accumulation='float32',half_algorithm=1,fp32_algorithm=3,lean_preparation=True)),flush=True)
        self.observed.add(sig)
        return self.native(x)
    def close(self):
        self.native.close()
        print('GUFO_LT_STATS',json.dumps(dict(**self.stats,unique_shapes=len(self.observed),cache=self.native.cache.stats)),flush=True)

def install_gram(cm,contract_path,library,calibration_sha256):
    contract=json.loads(Path(contract_path).read_text())
    assert contract['calibration_sha256']==calibration_sha256
    gram=QualifiedPrepared(contract,library);original=cm._flush_capture_H
    replacement=make_flush(original,gram);cm._flush_capture_H=replacement
    def restore():
        assert cm._flush_capture_H is replacement
        cm._flush_capture_H=original;gram.close()
    return restore

def install_moe(contract_path):
    ctx=installed(json.loads(Path(contract_path).read_text()));ctx.__enter__()
    print('GUFO_STATIC_CAPTURE_ACTIVE',flush=True)
    def restore():
        try:ctx.__exit__(None,None,None)
        finally:print('GUFO_STATIC_CAPTURE_STATS',json.dumps(STATS),flush=True)
    return restore
