"""Static all-expert routing and repeated nonfinite-count candidate.

Only pinned std all-expert routing: one owning gather per call; each expert still
owns its independent activated up output queued for down H. Empty routed-down
results never reach a consumer. Shared expert and normal forward stay original.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib,inspect,textwrap
from weakref import WeakSet
import torch
import capture_fast
from capture_repeat_counts import RepeatedCounts

_current=ContextVar('lean_discarded_moe',default=None)
STATS=dict(scopes=0,gathers=0,gather_reuses=0,empty_down=0,dead_scatter_skips=0,static_routes=0,count_kernels=0,count_reuses=0,count_commits=0)

class Gather:
    def __init__(self,expected):self.source=None;self.value=None;self.calls=0;self.expected=expected
    def get(self,y,top_x,count):
        assert count==y.shape[0] and top_x.numel()==count
        if self.value is None:
            self.source=y;self.value=y.index_select(0,top_x);STATS['gathers']+=1
        else:
            assert y is self.source
            STATS['gather_reuses']+=1
        self.calls+=1
        assert self.calls<=self.expected
        return self.value
    def drained(self):assert self.calls==self.expected


def guards(mlp,params):
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.block_sparse_mlp_routing import routing_std
    from exllamav3.modules.quant.fp16 import LinearFP16
    import exllamav3.modules.block_sparse_mlp as mod
    assert type(mlp) is BlockSparseMLP and mlp.routing_fn is routing_std
    assert mlp.routing_gate is not None
    assert mlp.router_pre_norm is None and mlp.tier is None
    assert not any((mlp.support_dec_moe,mlp.support_hip_grouped,mlp.support_hip_prefill,mlp.support_hip_wmma))
    assert mlp.mixed_k is None and mlp.batch_recon in (None,False)
    import exllamav3.modules.block_sparse_mlp as routing_module
    assert routing_module._ROUTE_TRACE is None
    assert params.get('activate_all_experts') is True and 'capture' in params
    assert not any(k in params for k in ('dec_routed','_cap_rw','mlp_residual_out'))
    assert not mod._RWH and mlp.bc is None and not mlp._row_chunk_ok()
    assert not mlp.support_quant_paths and mlp.fused_mode_buffers is None
    assert mlp.num_local_experts==mlp.num_experts==mlp.routing_cfg.num_experts
    assert mlp.routing_first in (None,0) and not mlp.pruned
    assert mlp.cpu_split_first is None and not mlp.cpu_offload and mlp.routing_device is None
    assert mlp.tp_mode is None and not mlp.tp_reduce and mlp.num_slices==1
    assert mlp.latent_in is None and mlp.latent_out is None and not mlp.alt_residual_channel
    assert mlp.routed_pre_norm is None and mlp.routed_post_norm is None
    assert len(mlp.ups)==len(mlp.gates)==len(mlp.downs)==mlp.num_experts and mlp.gated
    for outer in (*mlp.ups,*mlp.gates,*mlp.downs):
        assert type(outer.inner) is LinearFP16 and outer.inner.bias is None and outer.inner._pinned_store is None
        assert not outer.is_sliced
        # trim flag is harmless only when equal widths (asserted below) bypass its branch.
        assert not outer.lora_a_tensors and not outer.lora_b_tensors
        assert outer.pre_scale==outer.post_scale==1 and outer.softcap==0
        assert outer.in_features==outer.in_features_unpadded and outer.out_features==outer.out_features_unpadded
    for outer in (*mlp.ups,*mlp.gates):assert outer.inner.out_dtype==torch.float16
    for outer in mlp.downs:assert outer.inner.out_dtype==torch.float32


def gather(y,top_x,count):
    active=_current.get();assert active is not None
    return active[2].get(y,top_x,count)


def skip_scatter(params,mlp):
    active=_current.get();assert active is not None and active[0] is mlp and active[1] is params
    STATS['dead_scatter_skips']+=1


@contextmanager
def installed(contract):
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.quant.fp16 import LinearFP16
    from exllamav3.modules.linear import Linear
    from exllamav3.modules.block_sparse_mlp_routing import routing_std,_routing_std_torch
    from exllamav3.util.tensor import buffered_interleaved_arange
    old_moe,old_inner,old_capture=BlockSparseMLP.forward,LinearFP16.forward,Linear.capture_H
    assert getattr(old_inner,'_gufo_capture_fast',False)
    for name,fn in [('moe',old_moe),('linear',Linear.forward),('capture',old_capture),('routing',routing_std),('routing_fallback',_routing_std_torch),('tokens',buffered_interleaved_arange)]:
        assert hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()==contract[name],name
    source=textwrap.dedent(inspect.getsource(old_moe))
    old='current_state = y.index_select(0, top_x)';assert source.count(old)==1
    source=source.replace(old,'current_state = _lean_gather(y, top_x, count)')
    old='current_state.mul_(w)\n                final_hidden_states.index_add_(0, top_x, current_state)';assert source.count(old)==1
    source=source.replace(old,'_lean_skip_scatter(params, self)')
    old='selected_experts, routing_weights = self.routing_fn(bsz, self.routing_cfg, z, params)'
    assert source.count(old)==1
    source=source.replace(old,'selected_experts, routing_weights = None, None')
    start=source.index('            num_tokens, top_k = selected_experts.shape',source.index('# One spare row:'))
    end=source.index('            # Deterministic accumulation',start)
    source=source[:start]+'            num_tokens, top_k = bsz, self.num_experts\n            E = self.num_local_experts\n            expert_count_list = [bsz] * E + [0]\n            moeg = False\n            fused_cap = self.fused_rows\n            self._fused_cap = fused_cap\n            self.batch_recon = False\n            recon = None\n            groups = []\n            min_rows = 0\n            fused_total = 0\n            _static_tokens = torch.arange(bsz, device=y.device)\n\n'+source[end:]
    old='top_x = token_sorted[start:end]\n                w = weight_sorted[start:end].unsqueeze(1)'
    assert source.count(old)==1
    source=source.replace(old,'top_x = _static_tokens\n                w = None')
    namespace=dict(old_moe.__globals__,_lean_gather=gather,_lean_skip_scatter=skip_scatter)
    exec(compile(source,'<lean-discarded-moe>','exec'),namespace);lean=namespace['forward']
    validated=WeakSet()
    def moe_forward(self,x,params,out_dtype=None):
        scope=capture_fast._active.get()
        if scope is None:return old_moe(self,x,params,out_dtype)
        assert params is scope[0] and id(self.downs[0].inner) in scope[1]
        assert x.numel()//x.shape[-1]>=33, 'Single-token routing is not qualified'
        assert _current.get() is None
        if self not in validated:guards(self,params);validated.add(self)
        assert not any(k in params for k in ('dec_routed','_cap_rw','mlp_residual_out'))
        state=Gather(self.num_experts);counts=RepeatedCounts()
        eligible=frozenset(id(outer) for outer in (*self.ups,*self.gates))
        token=_current.set((self,params,state,counts,eligible));STATS['scopes']+=1;STATS['static_routes']+=1
        try:
            value=lean(self,x,params,out_dtype);state.drained()
            assert counts.calls==2*self.num_experts and counts.kernels==1
            counts.commit();STATS['count_kernels']+=counts.kernels;STATS['count_reuses']+=counts.calls-counts.kernels;STATS['count_commits']+=len(counts.entries)
            return value
        finally:_current.reset(token)
    def inner_forward(self,x,params,out_dtype=None):
        scope=capture_fast._active.get();active=_current.get()
        if active is not None and scope is not None and id(self) in scope[1]:
            assert params is active[1] is scope[0]
            assert (out_dtype or self.out_dtype)==torch.float32
            STATS['empty_down']+=1
            return x.new_empty((0,),dtype=torch.float32)
        return old_inner(self,x,params,out_dtype)
    def count_shared(self,x,params):
        active=_current.get()
        target=params['capture'][self.qmap]['inf_nan']
        if active is not None and params is active[1] and id(self) in active[4]:
            assert x is active[2].value, 'Only immutable owning gather is cached'
            active[3].add(x,target,old_capture.__globals__['ext'].count_inf_nan)
        else:old_capture.__globals__['ext'].count_inf_nan(x,target)
    cap_source=textwrap.dedent(inspect.getsource(old_capture))
    old='ext.count_inf_nan(x, params["capture"][self.qmap]["inf_nan"])'
    assert cap_source.count(old)==1
    cap_source=cap_source.replace(old,'_static_count(self,x,params)')
    cap_ns=dict(old_capture.__globals__,_static_count=count_shared)
    exec(compile(cap_source,'<static-capture-H>','exec'),cap_ns)
    Linear.capture_H=cap_ns['capture_H']
    BlockSparseMLP.forward=moe_forward;LinearFP16.forward=inner_forward
    try:yield
    finally:
        assert _current.get() is None
        BlockSparseMLP.forward=old_moe;LinearFP16.forward=old_inner;Linear.capture_H=old_capture
