"""Strict bounded operator ABI, no dtype reduction or fallback."""
import ctypes as C
import torch

class Unsupported(RuntimeError):pass

class Native:
    def __init__(self,path):
        self.lib=C.CDLL(str(path));self.plans=[]
        self.lib.forward_create.argtypes=[C.c_int]*5+[C.POINTER(C.c_int)]*2
        self.lib.forward_create.restype=C.c_void_p
        self.lib.forward_run.argtypes=[C.c_void_p,C.c_int]+[C.c_void_p]*4
        self.lib.forward_run.restype=C.c_int
        self.lib.forward_destroy.argtypes=[C.c_void_p];self.lib.forward_destroy.restype=None
        self.lib.forward_wmma_run.argtypes=[C.c_void_p]*3+[C.c_int]*5+[C.c_void_p]
        self.lib.forward_wmma_run.restype=C.c_int
    @staticmethod
    def shape(x,w,y):
        assert x.ndim==2 and w.ndim==3 and y.ndim==3
        b,k,n=w.shape;m=x.shape[0]
        assert x.shape[1]==k and y.shape==(b,m,n)
        assert x.dtype==w.dtype==torch.float16 and y.dtype in (torch.float16,torch.float32)
        assert x.is_cuda and x.device==w.device==y.device
        assert x.is_contiguous() and w.is_contiguous() and y.is_contiguous()
        return b,m,k,n,int(y.dtype==torch.float16)
    def create(self,x,w,y):
        shape=self.shape(x,w,y)
        assert len(self.plans)<8,'Workspace limit8x32MiB exceeded'
        status=C.c_int();found=C.c_int()
        p=self.lib.forward_create(*shape,C.byref(status),C.byref(found))
        if not p:
            if status.value in (-2,7):
                raise Unsupported(f'Lt unsupported {shape}: status={status.value}')
            raise RuntimeError(f'Lt creation failed {shape}: status={status.value}')
        assert status.value==0 and 0<found.value<=8
        self.plans.append((p,shape,found.value));return self.plans[-1]
    def run(self,plan,algorithm,x,w,y):
        p,shape,found=plan;assert plan in self.plans
        assert self.shape(x,w,y)==shape and 0<=algorithm<found
        assert self.lib.forward_run(p,algorithm,x.data_ptr(),w.data_ptr(),y.data_ptr(),torch.cuda.current_stream(x.device).cuda_stream)==0
    def wmma(self,x,w,y):
        shape=self.shape(x,w,y)
        assert self.lib.forward_wmma_run(x.data_ptr(),w.data_ptr(),y.data_ptr(),*shape,torch.cuda.current_stream(x.device).cuda_stream)==0
    def close(self):
        if self.plans:torch.cuda.synchronize()
        for p,_,_ in self.plans:self.lib.forward_destroy(p)
        self.plans.clear()
