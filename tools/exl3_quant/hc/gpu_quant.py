"""Offline GPU FP64 HC GPTQ and small-up refinement; never loads the model."""
import ctypes
import time
from pathlib import Path
import numpy as np
import torch
from . import quant as q


def feedback_library(path):
    lib=ctypes.CDLL(str(Path(path).resolve()))
    fn=lib.gufo_hc_feedback
    fn.argtypes=[ctypes.c_void_p]*5+[ctypes.c_uint,ctypes.c_uint,ctypes.c_void_p]
    fn.restype=ctypes.c_int
    return lib


def gptq(w, h, library, damping=None):
    start=time.monotonic()
    if w.dtype != np.float32 or w.ndim != 2 or w.shape[1]>10240:
        raise ValueError('expected bounded direct-BF16 FP32 weights')
    _, scales=q.rtn(w)
    n=w.shape[1]
    if h.device.type!='cuda' or h.dtype!=torch.float64 or tuple(h.shape)!=(n,n):
        raise ValueError('expected GPU full FP64 H')
    if not bool(torch.isfinite(h).all()) or not bool(torch.allclose(h,h.T,rtol=1e-10,atol=1e-12)):
        raise ValueError('invalid H')
    if bool((h.diag()<0).any()) or h.diag().mean().item()<=0:raise ValueError('invalid H energy')
    if h.device.index!=torch.cuda.current_device():raise ValueError('H must use the current GPU')
    damping=float(.01*h.diag().mean().item()) if damping is None else float(damping)
    if not np.isfinite(damping) or damping<=0: raise ValueError('invalid damping')
    hd=(h+h.T)*.5;hd.diagonal().add_(damping)
    l=torch.linalg.cholesky(hd)
    li=torch.linalg.solve_triangular(l,torch.eye(n,dtype=h.dtype,device=h.device),upper=False)
    u=torch.linalg.cholesky(li.T@li).T.contiguous()
    del hd,l,li
    if not bool(torch.isfinite(u).all()) or not bool((u.diag()>0).all()): raise ValueError('invalid factor')
    weights=torch.as_tensor(w.astype(np.float64),device=h.device)
    ds=torch.as_tensor(scales.astype(np.float64),device=h.device)
    codes=torch.empty(w.shape,device=h.device,dtype=torch.int8)
    failed=torch.zeros(1,device=h.device,dtype=torch.int32)
    status=library.gufo_hc_feedback(weights.data_ptr(),u.data_ptr(),ds.data_ptr(),codes.data_ptr(),
                                    failed.data_ptr(),len(w),n,torch.cuda.current_stream(h.device).cuda_stream)
    if status: raise RuntimeError('native feedback launch failed: '+str(status))
    torch.cuda.synchronize()
    if failed.item(): raise ValueError('native feedback produced nonfinite intermediate')
    packed=q.pack(codes.cpu().numpy(),scales)
    return packed, {'damping':damping,'seconds':time.monotonic()-start,
                    'method':'FP64_GPU_inverse_Cholesky_sequential_feedback',
                    'precision':'source_BF16_exact_FP32_to_FP64; H_and_feedback_FP64'}


def refine_up(w, packed, h, damping, scales=True, sweeps=6):
    if w.ndim!=2 or not len(w) or w.shape[1]!=320 or not np.isfinite(w).all():
        raise ValueError('bounded refinement only covers finite 320-input up projections')
    if (h.device.type!='cuda' or h.dtype!=torch.float64 or tuple(h.shape)!=(320,320)
            or not bool(torch.isfinite(h).all()) or not bool(torch.allclose(h,h.T,rtol=1e-10,atol=1e-12))):
        raise ValueError('invalid full FP64 H')
    if h.device.index!=torch.cuda.current_device():raise ValueError('H must use the current GPU')
    if not np.isfinite(damping) or damping<=0 or type(sweeps) is not int or not 1<=sweeps<=16:
        raise ValueError('invalid refinement budget/damping')
    if packed.dtype!=np.uint8 or packed.shape!=(len(w),10,34):raise ValueError('invalid pack')
    cc=np.ascontiguousarray(packed[...,2:]).view(np.int8).reshape(w.shape)
    dd=np.ascontiguousarray(packed[...,:2]).view('<f2').reshape(len(w),10)
    if not np.array_equal(q.pack(cc,dd),packed) or np.any((dd==0)&np.any(w.reshape(len(w),10,32)!=0,axis=2)):
        raise ValueError('invalid packed values/scales')
    start=time.monotonic();dev=h.device
    w=torch.as_tensor(w.astype(np.float64),device=dev)
    c=torch.as_tensor(np.ascontiguousarray(packed[...,2:]).view(np.int8).reshape(w.shape).astype(np.float64),device=dev)
    d=torch.as_tensor(np.ascontiguousarray(packed[...,:2]).view('<f2').reshape(len(w),-1).astype(np.float64),device=dev)
    hd=(h+h.T)*.5;hd.diagonal().add_(damping)
    torch.linalg.cholesky(hd)
    error=w-c*torch.repeat_interleave(d,32,dim=1)
    grad=error@hd;previous=(error*grad).sum(dim=1)
    trace=[]
    for sweep in range(sweeps):
        for j in range(320):
            scale=d[:,j//32]
            best=torch.zeros_like(scale);step=torch.zeros_like(scale)
            for direction in (-1,1):
                delta=direction*scale
                gain=-2*delta*grad[:,j]+delta*delta*hd[j,j]
                take=(c[:,j]+direction>=-127)&(c[:,j]+direction<=127)&(gain<best)
                best=torch.where(take,gain,best);step=torch.where(take,direction,step)
            delta=step*scale
            c[:,j]+=step;error[:,j]-=delta;grad-=delta[:,None]*hd[j,:]
        if scales:
            grad=error@hd
            for group in range(10):
                span=slice(group*32,(group+1)*32);v=c[:,span]
                den=((v@hd[span,span])*v).sum(dim=1)
                num=(v*grad[:,span]).sum(dim=1);old=d[:,group].clone()
                opt=old+torch.where(den>0,num/den,0.)
                if not bool(torch.isfinite(opt).all()): raise ValueError('invalid scale optimum')
                center=opt.clamp(2**-24,65504).half()
                choices=(center,torch.nextafter(center,torch.zeros_like(center)),
                         torch.nextafter(center,torch.full_like(center,float('inf'))))
                best=torch.zeros_like(old);selected=old.clone()
                for ch in choices:
                    candidate=ch.double();valid=torch.isfinite(candidate)&(candidate>0)&(old>0)&(den>0)
                    delta=torch.where(valid,candidate-old,0.)
                    gain=-2*delta*num+delta*delta*den;take=valid&(gain<best)
                    best=torch.where(take,gain,best);selected=torch.where(take,candidate,selected)
                delta_w=(selected-old)[:,None]*v
                error[:,span]-=delta_w;grad-=delta_w@hd[span,:];d[:,group]=selected
        error=w-c*torch.repeat_interleave(d,32,dim=1);grad=error@hd
        now=(error*grad).sum(dim=1)
        if not bool(torch.isfinite(now).all()) or bool((now>previous+1e-10*previous.abs().clamp_min(1e-20)).any()):
            raise ValueError('GPU refinement increased row objective')
        trace.append({'sweep':sweep+1,'before':previous.sum().item(),'after':now.sum().item()})
        previous=now
    result=q.pack(c.cpu().numpy().astype(np.int8),d.cpu().numpy().astype(np.float16))
    return result,{'seconds':time.monotonic()-start,'sweeps':sweeps,'scales':scales,'trace':trace,
                   'holdout_used_for_optimization':False}


def measure(w, packed, h, x):
    wt=torch.as_tensor(w.astype(np.float64),device=h.device)
    error=torch.as_tensor(q.unpack(packed),device=h.device)-wt
    delta=x@error.T;reference=x@wt.T
    result={'calibration_weighted_sse':((error@h)*error).sum().item(),
            'holdout_output_sse':delta.square().sum().item(),
            'holdout_reference_sse':reference.square().sum().item(),
            'holdout_abs_p99':torch.quantile(delta.abs().flatten(),.99).item(),
            'holdout_abs_max':delta.abs().max().item()}
    if not all(np.isfinite(v) for v in result.values()):raise ValueError('nonfinite metric')
    return result
