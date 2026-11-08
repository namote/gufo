"""Elide dead routed-down GEMMs ONLY in the pinned converter's discarded capture.

Linear.forward still captures the original down input, with original padding,
Hessian arithmetic and counters. Normal routed references/state never enter this
scope. Upstream source, quantization and random number streams remain unchanged.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import inspect
import sys

import torch

_active = ContextVar('discarded_terminal_capture', default=None)
_MAIN_SHA = '53723d1e4a92b163c6dac8657010e76a4ad5b2fb05286551309ccebd3476dec8'
STATS = dict(scopes=0, skipped_down_calls=0, skipped_down_rows=0)


def _targets(block, params):
    from exllamav3.modules.transformer import TransformerBlock
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.hyperconnections import GatedResidual
    from exllamav3.modules.quant.fp16 import LinearFP16
    assert type(block) is TransformerBlock
    mlp = block.mlp
    assert type(mlp) is BlockSparseMLP, 'Requires a terminal routed MoE'
    assert type(block.mlp_hc) is GatedResidual, 'Only audited gated residual tail'
    assert block.mlp_post_norm is None and block.ablit is None
    assert mlp.latent_out is None and not mlp.tp_reduce
    assert mlp.cpu_split_first is None and not mlp.cpu_offload and not mlp.pruned
    assert mlp.tp_mode is None and mlp.routing_device is None
    assert mlp.routed_post_norm is None and mlp.shared_experts_post_norm is None
    assert params.get('activate_all_experts') is True and isinstance(params.get('capture'), dict)
    assert 'quant_preserve' in params, 'Capture must protect the original residual input'
    assert not any(k in params for k in ('export_states', 'export_state_layers',
                                       'hc_pending', 'q_mlp_slice', 'backend'))
    assert mlp.num_slices == 1
    downs = mlp.downs
    assert len(downs) == mlp.num_experts and len({id(d) for d in downs}) == len(downs)
    inners = set()
    for down in downs:
        assert down.qmap and not down.lora_a_tensors
        inner = down.inner
        assert type(inner) is LinearFP16 and inner.bias is None
        assert inner.out_dtype is torch.float32 and inner._pinned_store is None
        assert inner.in_features == down.in_features and inner.out_features == down.out_features
        inners.add(id(inner))
    assert len(inners) == len(downs)
    return frozenset(inners)


@contextmanager
def _discarded_capture(block, params):
    """Private scope used by the pinned caller and its affected-block probe."""
    assert _active.get() is None, 'Nested capture scopes are unsupported'
    targets = _targets(block, params)
    token = _active.set((params, targets))
    STATS['scopes'] += 1
    try:
        yield
    finally:
        _active.reset(token)


def install(cm, *, automatic=True):
    """Wrap only our process; refuse an unaudited conversion call-site change."""
    from exllamav3.modules.transformer import TransformerBlock
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.quant.fp16 import LinearFP16
    main_body = inspect.unwrap(cm.main)
    assert hashlib.sha256(inspect.getsource(main_body).encode()).hexdigest() == _MAIN_SHA
    assert not getattr(LinearFP16.forward, '_gufo_capture_fast', False)
    original_inner = LinearFP16.forward
    original_block = TransformerBlock.forward
    caller_code = main_body.__code__

    def inner_forward(self, x, params, out_dtype=None):
        scope = _active.get()
        if scope is not None and params is scope[0] and id(self) in scope[1]:
            STATS['skipped_down_calls'] += 1
            STATS['skipped_down_rows'] += x.numel() // x.shape[-1]
            return torch.zeros((*x.shape[:-1], self.out_features), device=x.device,
                               dtype=out_dtype or self.out_dtype or torch.half)
        return original_inner(self, x, params, out_dtype)

    def block_forward(self, x, params, out_dtype=None):
        # In this exact cm.main, all-expert output is discarded. Reference rows
        # are recomputed without capture/all-experts before finite/error checks.
        is_discarded = (automatic and sys._getframe(1).f_code is caller_code
                        and 'capture' in params and params.get('activate_all_experts') is True
                        and type(self.mlp) is BlockSparseMLP)
        if is_discarded:
            with _discarded_capture(self, params):
                return original_block(self, x, params, out_dtype)
        return original_block(self, x, params, out_dtype)

    inner_forward._gufo_capture_fast = True
    LinearFP16.forward = inner_forward
    TransformerBlock.forward = block_forward

    def restore():
        assert _active.get() is None
        LinearFP16.forward = original_inner
        TransformerBlock.forward = original_block
    return restore
