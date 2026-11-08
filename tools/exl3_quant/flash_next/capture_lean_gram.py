"""Candidate: prepare native Gram operands once; same Lt and H addition order."""
import ctypes as C
import hashlib,inspect
import torch
from capture_lt import LtGram,FLUSH_SHA


def prepare_stack(xs):
    """xs is the private torch.stack allocation owned by this flush chunk."""
    assert xs.dtype in (torch.float16,torch.bfloat16,torch.float32)
    xf=xs if xs.dtype==torch.float16 else xs.to(torch.float32)
    finite=torch.isfinite(xf).all(dim=2)
    xf.masked_fill_(~finite.unsqueeze(2),0.0)
    return xf,finite


def make_flush(original,gram):
    source=inspect.getsource(original)
    assert hashlib.sha256(source.encode()).hexdigest()==FLUSH_SHA
    old='''xf = xs.to(torch.float)
            finite = torch.isfinite(xf).all(dim = 2)
            xf = xf.masked_fill_(~finite.unsqueeze(2), 0.0)
            P = torch.bmm(xf.transpose(1, 2), xf)'''
    assert source.count(old)==1
    source=source.replace(old,'xf, finite = _prepare_stack(xs)\n            P = _prepared_gram(xf)')
    namespace=dict(original.__globals__,_prepare_stack=prepare_stack,_prepared_gram=gram)
    exec(compile(source,'<lean-Gram-flush>','exec'),namespace)
    return namespace['flush_capture_H']


class PreparedLtGram(LtGram):
    """Already masked FP16/FP32 input, original alg1/3 and plan bounds retained."""
    def __init__(self,path):super().__init__(path,'original_f16')
    def __call__(self,x):
        assert x.dtype in (torch.float16,torch.float32) and x.is_cuda and x.is_contiguous()
        half=x.dtype==torch.float16
        b,r,d=x.shape;stream=torch.cuda.current_stream().cuda_stream
        assert torch.cuda.current_device()==x.device.index
        if self.stream is None:self.stream=stream;self.device=x.device.index
        assert self.stream==stream and self.device==x.device.index
        key=(b,r,d,half,x.device.index)
        def create():
            code=C.c_int();found=C.c_int()
            p=self.lib.gram_lt_create(b,r,d,int(half),C.byref(code),C.byref(found))
            assert p and code.value==0,('Unsupported selected Lt shape',key,code.value)
            algo=1 if half else 3
            if found.value<=algo:
                self.lib.gram_lt_destroy(p)
                raise RuntimeError(('Selected Lt algorithm unavailable',key,algo,found.value))
            return p,algo
        p,algo=self.cache.get(key,create)
        y=torch.empty((b,d,d),dtype=torch.float32,device=x.device)
        code=self.lib.gram_lt_run(p,algo,x.data_ptr(),y.data_ptr(),stream)
        assert code==0,('Selected Lt execution failed',key,algo,code)
        return y
