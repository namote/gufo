"""Uninstalled Lt flush candidate; preserve original grouping/masking/H additions."""
import ctypes as C
import hashlib
import inspect
from collections import OrderedDict
import torch

FLUSH_SHA='3d167e36fefeb8a9e3bcd81adaea3dd235af448c3d6b3c3a2a8ea24d2dd33761'

def make_flush(original,gram):
    source=inspect.getsource(original)
    assert hashlib.sha256(source.encode()).hexdigest()==FLUSH_SHA
    old='P = torch.bmm(xf.transpose(1, 2), xf)'
    assert source.count(old)==1
    namespace=dict(original.__globals__,_gufo_gram=gram)
    exec(compile(source.replace(old,'P = _gufo_gram(xf, xs, finite)'),'<capture-lt-flush>','exec'),namespace)
    return namespace['flush_capture_H']

class BoundedPlans:
    """Serial LRU; finish queued users before freeing any native workspace."""
    def __init__(self,limit,synchronize,destroy):
        assert type(limit) is int and limit>0
        self.limit=limit;self.synchronize=synchronize;self.destroy=destroy
        self.plans=OrderedDict();self.closed=False
        self.stats=dict(hits=0,misses=0,evictions=0,peak_plans=0,limit=limit,
                        max_workspace_bytes=limit*32*1024*1024)
    def get(self,key,create):
        assert not self.closed
        if key in self.plans:
            self.stats['hits']+=1;self.plans.move_to_end(key);return self.plans[key]
        self.stats['misses']+=1
        if len(self.plans)==self.limit:
            self.synchronize()
            _,old=self.plans.popitem(last=False);self.destroy(old)
            self.stats['evictions']+=1
        value=create()
        self.plans[key]=value
        self.stats['peak_plans']=max(self.stats['peak_plans'],len(self.plans))
        return value
    def close(self):
        if self.closed:return
        if self.plans:
            self.synchronize()
            for value in self.plans.values():self.destroy(value)
            self.plans.clear()
        self.closed=True

class LtGram:
    """Explicit selected algorithm, no backend fallback. Use one serial stream."""
    def __init__(self,path,mode):
        assert mode in ('f32','original_f16')
        self.mode=mode;self.stream=None;self.device=None
        self.lib=C.CDLL(str(path))
        self.lib.gram_lt_create.argtypes=[C.c_int,C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_int)]
        self.lib.gram_lt_create.restype=C.c_void_p
        self.lib.gram_lt_run.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p]
        self.lib.gram_lt_run.restype=C.c_int
        self.lib.gram_lt_destroy.argtypes=[C.c_void_p];self.lib.gram_lt_destroy.restype=None
        self.cache=BoundedPlans(8,lambda:torch.cuda.current_stream().synchronize(),
                               lambda value:self.lib.gram_lt_destroy(value[0]))
    def __call__(self,xf,xs,finite):
        assert xf.dtype==torch.float32 and xf.is_contiguous() and xf.is_cuda
        # Only already-half inputs take half Lt. Original FP32 inputs stay FP32.
        half=self.mode=='original_f16' and xs.dtype==torch.float16
        x=xs.masked_fill(~finite.unsqueeze(2),0) if half else xf
        b,r,d=x.shape;stream=torch.cuda.current_stream().cuda_stream
        assert torch.cuda.current_device()==x.device.index
        if self.stream is None:self.stream=stream;self.device=x.device.index
        assert self.stream==stream, 'Diagnostic plan/temporary buffers are single-stream'
        assert self.device==x.device.index
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
    def close(self):
        if not self.cache.closed and self.stream is not None:
            assert self.device==torch.cuda.current_device()
            assert self.stream==torch.cuda.current_stream().cuda_stream
        self.cache.close()
