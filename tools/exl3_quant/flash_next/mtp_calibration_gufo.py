"""Offline Gufo-target MTP input semantics; GPU numerical qualification pending.

References: Gufo executor.cpp MtpStep / DenseBatch and exl3_weights.cpp Exl3MatVec.
Torch reductions/GEMM are semantic candidates, not native-kernel byte equality.
No original-basis reconstructed weight shortcut is used for packed FCs.
"""
import types

SEMANTICS = 'gufo_mtp_f32_full_hc_shifted_v1'
HADAMARD_SCALE = 0.0883883461356163


def rms_norm_f32(x, raw_gamma, eps):
    import torch
    assert x.dtype == torch.float32 and raw_gamma.numel() == x.shape[-1]
    gamma = raw_gamma.float().reshape(x.shape[-1]) + 1.0
    return (x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)) * gamma


def hadamard128_f32(x):
    import torch
    assert x.dtype == torch.float32 and x.shape[-1] % 128 == 0
    shape=x.shape
    y=x.contiguous().reshape(-1,128)
    for length in (1,2,4,8,16,32,64):
        grouped=y.reshape(-1,128//(2*length),2,length)
        a=grouped[:,:,0].clone();b=grouped[:,:,1].clone()
        y=torch.stack((a+b,a-b),dim=2).reshape(-1,128)
    return (y * HADAMARD_SCALE).reshape(shape)


def gufo_input_rotation(x, suh):
    import torch
    assert x.dtype==torch.float32 and suh.shape==(x.shape[-1],)
    return hadamard128_f32(x * suh.float()).to(torch.float16)


def packed_fc_f32(x, inner_weight, suh, svh, row_chunk=512):
    """Rotated-basis weight [K,N], FP16 after input rotation, F32 inner/output."""
    import torch
    assert x.dtype==torch.float32 and inner_weight.dtype==torch.float16
    assert inner_weight.ndim==2 and inner_weight.shape[0]==x.shape[-1]
    assert svh.shape==(inner_weight.shape[1],) and row_chunk>0
    flat=x.reshape(-1,x.shape[-1]);w=inner_weight.float()
    result=torch.empty((flat.shape[0],w.shape[1]),device=x.device,dtype=torch.float32)
    for start in range(0,flat.shape[0],row_chunk):
        xr=gufo_input_rotation(flat[start:start+row_chunk],suh)
        # Casts after input FP16 rounding preserve its exact values. F32 GEMM
        # reduction differs from native GEMV and requires a bounded comparison.
        accum=xr.float() @ w
        result[start:start+row_chunk]=hadamard128_f32(accum) * svh.float()
    return result.reshape(*x.shape[:-1],inner_weight.shape[1])


def raw_fc_f32(x, weight, row_chunk=512):
    import torch
    assert x.dtype==torch.float32 and weight.ndim==2 and x.shape[-1]==weight.shape[0]
    flat=x.reshape(-1,x.shape[-1]);w=weight.float()
    result=torch.empty((flat.shape[0],w.shape[1]),device=x.device,dtype=torch.float32)
    for start in range(0,flat.shape[0],row_chunk):
        result[start:start+row_chunk]=flat[start:start+row_chunk] @ w
    return result.reshape(*x.shape[:-1],w.shape[1])


def prepare_gufo_prompt(state, shifted_ids, hc_mult=4, hidden_size=2560):
    import torch
    assert state.dtype==torch.float32 and shifted_ids.dtype==torch.int64
    assert shifted_ids.ndim==2 and shifted_ids.shape[0]==1 and shifted_ids.shape[1]>0
    assert state.shape==(1,shifted_ids.shape[1]+1,hc_mult,hidden_size)
    assert torch.isfinite(state).all()
    return state[:,:-1].contiguous()  # no FP16 tap; no fabricated zero-prefix row


def install_input_adapter(module):
    """Patch only this candidate instance; keep original Linear capture/quant state."""
    import torch
    from exllamav3.modules.quant.fp16 import LinearFP16
    from exllamav3.modules.quant.exl3 import LinearEXL3
    caches=[]
    for linear in (module.fc_hidden,module.fc_embedding):
        cache={'inner':None,'weight':None};caches.append(cache)
        def projection(self,x,params,out_dtype=None,_cache=cache):
            assert x.dtype==torch.float32 and x.shape[-1]==self.in_features
            assert not self.lora_a_tensors and self.pre_scale==self.post_scale==1 and self.softcap==0
            assert self.in_features==self.in_features_unpadded and self.out_features==self.out_features_unpadded
            assert self.inner.bias is None
            if 'capture' in params:self.capture_H(x,params)
            inner=self.inner
            if _cache['inner'] is not inner:
                _cache['inner']=inner
                if isinstance(inner,LinearFP16):
                    # Upstream raw loader uses FP16 weights; widening preserves
                    # those loaded values. BF16->FP16 source rounding is audited
                    # separately, not silently described as BF16 exact.
                    _cache['weight']=inner.weight.float()
                elif isinstance(inner,LinearEXL3):
                    _cache['weight']=inner.get_inner_weight_tensor()
                else:raise TypeError(f'Unsupported MTP FC storage: {type(inner)}')
            if isinstance(inner,LinearFP16):return raw_fc_f32(x,_cache['weight'])
            return packed_fc_f32(x,_cache['weight'],inner.suh,inner.svh)
        linear.forward=types.MethodType(projection,linear)

    def prepare(self,state,params):
        assert 'target_hidden' not in params and 'hc_pending' not in params
        ids=params['input_ids']
        params['target_hidden']=prepare_gufo_prompt(state,ids,self.hc_mult,self.hidden_size).to(self.device)
        return ids.to(self.device)

    def forward(self,x,params,out_dtype=None):
        assert out_dtype in (None,torch.float32)
        state=params['target_hidden']
        assert state.shape==(*x.shape,self.hc_mult,self.hidden_size) and state.dtype==torch.float32
        normalized=rms_norm_f32(state.flatten(-2),self.norm_hidden_w_raw,self.rms_eps).reshape_as(state)
        h=self.fc_hidden.forward(normalized.contiguous(),params)
        embedding=self.attached_model().modules[0].forward(x,params,out_dtype=torch.float32)
        embedding=rms_norm_f32(embedding.to(self.device),self.pre_fc_norm_embedding.weight,self.rms_eps)
        e=self.fc_embedding.forward(embedding.contiguous(),params)
        return (h+e.unsqueeze(-2)).contiguous()
    module.prepare_for_device=types.MethodType(prepare,module)
    module.forward=types.MethodType(forward,module)
    def clear():
        for cache in caches:cache.clear()
    return clear


def derived_gram_shapes(calibration, old_contract):
    lengths={s['tokens']-1 for s in calibration['sequences']}
    assert min(lengths)>=33
    patterns={(b,d,half) for b,_,d,half in old_contract['allowed_shapes']}
    return sorted([b,length,d,half] for length in lengths for b,d,half in patterns)


def install_decoder_gram(cm, path, library, calibration):
    """No old T-shape contract is valid for the new T-1 workload implicitly."""
    import hashlib,json
    from pathlib import Path
    from capture_lean_gram import PreparedLtGram,make_flush
    contract=json.loads(Path(path).read_text())
    assert contract['status']=='qualified_for_gufo_mtp_shifted_rows'
    assert contract['semantics']==SEMANTICS and contract['calibration_sha256']==calibration['sha256']
    old=json.loads(Path('/task/capture-lt-contract.json').read_text())
    assert contract['allowed_shapes']==derived_gram_shapes(calibration,old)
    assert hashlib.sha256(Path(library).read_bytes()).hexdigest()==contract['native_sha256']==old['native_sha256']
    allowed={tuple(shape) for shape in contract['allowed_shapes']}
    native=PreparedLtGram(library)
    def gram(x):
        import torch
        assert x.dtype in (torch.float16,torch.float32)
        assert (*x.shape,x.dtype==torch.float16) in allowed
        return native(x)
    original=cm._flush_capture_H
    cm._flush_capture_H=make_flush(original,gram)
    def restore():
        cm._flush_capture_H=original
        native.close()
    return restore
