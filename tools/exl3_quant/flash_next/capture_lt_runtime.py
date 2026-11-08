"""Qualified conversion Gram dispatch; original F16 input, F32 accumulation.

No backend/precision retry. Keep unsupported execution visible and bounded.
"""
import hashlib,json
from pathlib import Path
import torch
from capture_lt import LtGram,make_flush

class QualifiedGram:
    def __init__(self,contract,library):
        assert contract['status']=='qualified_for_idx10_resume'
        assert hashlib.sha256(Path(library).read_bytes()).hexdigest()==contract['native_sha256']
        self.allowed={tuple(s) for s in contract['allowed_shapes']}
        self.native=LtGram(library,'original_f16')
        self.stats=dict(half_calls=0,fp32_calls=0);self.observed=set()
    def __call__(self,xf,xs,finite):
        assert xs.dtype in (torch.float16,torch.float32,torch.bfloat16)
        b,r,d=xs.shape;half=xs.dtype==torch.float16;sig=(b,r,d,half)
        assert sig in self.allowed,('Unqualified Gram shape/dtype',sig,str(xs.dtype))
        self.stats['half_calls' if half else 'fp32_calls']+=1
        if not self.observed:
            print('GUFO_LT_ACTIVE',json.dumps(dict(shape=sig,original_dtype=str(xs.dtype),accumulation='float32',half_algorithm=1,fp32_algorithm=3)),flush=True)
        self.observed.add(sig)
        return self.native(xf,xs,finite)
    def close(self):
        self.native.close()
        print('GUFO_LT_STATS',json.dumps(dict(**self.stats,unique_shapes=len(self.observed),cache=self.native.cache.stats)),flush=True)

def install(cm,contract_path,library,calibration_sha256):
    contract=json.loads(Path(contract_path).read_text())
    assert contract['calibration_sha256']==calibration_sha256
    gram=QualifiedGram(contract,library);original=cm._flush_capture_H
    replacement=make_flush(original,gram);cm._flush_capture_H=replacement
    def restore():
        assert cm._flush_capture_H is replacement
        cm._flush_capture_H=original;gram.close()
    return restore
