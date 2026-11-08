"""Experimental discarded-capture up/gate Lt pair; never installed by production.

Outer Linear.forward remains intact, including capture_H. Each expert gets fresh
output storage: down.capture_H retains its activated up slice until row flush.
Normal references never enter capture_fast's private discarded scope.
"""
from contextlib import contextmanager
import hashlib
import inspect
import torch
import capture_fast
from capture_forward_ops import Native
from capture_lt import BoundedPlans


class LtPairs:
    def __init__(self, library):
        self.native = Native(library)
        self.stream = None
        self.device = None
        self.cache = BoundedPlans(8, self.synchronize, self.destroy)

    def validate_context(self):
        if self.stream is not None:
            stream = torch.cuda.current_stream(self.device)
            assert stream.cuda_stream == self.stream.cuda_stream

    def synchronize(self):
        if self.stream is not None:
            self.stream.synchronize()

    def destroy(self, plan):
        self.native.lib.forward_destroy(plan[0])
        self.native.plans.remove(plan)

    def __call__(self, x, weights):
        assert x.is_cuda and x.dtype == torch.float16 and x.ndim == 2
        stream = torch.cuda.current_stream(x.device)
        if self.stream is None:
            self.stream, self.device = stream, x.device
        assert self.device == x.device and self.stream.cuda_stream == stream.cuda_stream
        # Do NOT cache/reuse y: prior down H entries still own the activation view.
        y = torch.empty((2, x.shape[0], weights.shape[-1]), device=x.device, dtype=torch.float16)
        key = self.native.shape(x, weights, y)
        plan = self.cache.get(key, lambda: self.native.create(x, weights, y))
        self.native.run(plan, 0, x, weights, y)
        return y

    def close(self):
        self.cache.close()
        assert not self.native.plans


class PairState:
    """Strict adjacent up/gate protocol; injected executor permits CPU tests."""
    def __init__(self, pairs, execute, validate_context=lambda: None):
        self.roles = {}
        self.pending = None
        self.execute = execute
        self.validate_context = validate_context
        self.stats = dict(up_calls=0, gate_calls=0)
        for up, gate, weights in pairs:
            assert id(up) not in self.roles and id(gate) not in self.roles and up is not gate
            self.roles[id(up)] = ('up', id(gate), weights)
            self.roles[id(gate)] = ('gate', id(up), weights)

    def call(self, inner, x, params, out_dtype):
        self.validate_context()
        role, other, weights = self.roles[id(inner)]
        assert (out_dtype or inner.out_dtype or torch.float16) == torch.float16
        assert x.dtype == torch.float16 and x.is_contiguous() and x.shape[-1] == weights.shape[1]
        if role == 'up':
            assert self.pending is None, 'Missing gate consumption before next up'
            flat = x.view(-1, x.shape[-1])
            y = self.execute(flat, weights)
            assert y.shape == (2, flat.shape[0], weights.shape[-1]) and y.dtype == torch.float16
            # Exact Python tensor identity relies on pinned, padding-free outer
            # Linear and adjacent up/gate caller. No _version on inference tensors.
            self.pending = (other, x, params, y[1].view(*x.shape[:-1], weights.shape[-1]))
            self.stats['up_calls'] += 1
            return y[0].view(*x.shape[:-1], weights.shape[-1])
        assert self.pending is not None, 'Gate called without its up'
        expected, original_x, original_params, gate = self.pending
        assert id(inner) == expected and x is original_x and params is original_params
        self.pending = None
        self.stats['gate_calls'] += 1
        return gate

    def drained(self):
        assert self.pending is None, 'Capture ended before matching gate'
        assert self.stats['up_calls'] == self.stats['gate_calls']


def prepare(block, native, contract, max_weight_bytes=4*2**30):
    from exllamav3.modules.linear import Linear
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.quant.fp16 import LinearFP16
    for name, fn in [('linear', Linear.forward), ('moe', BlockSparseMLP.forward)]:
        assert hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() == contract[name]
    mlp = block.mlp
    assert type(mlp) is BlockSparseMLP and mlp.gated and mlp.interm_dtype == torch.float16
    assert mlp.bc is None and not mlp.support_quant_paths and not mlp.pruned
    assert len(mlp.ups) == len(mlp.gates) == mlp.num_experts
    pairs, total = [], 0
    # Check all eligibility before allocating additional weights.
    for up, gate in zip(mlp.ups, mlp.gates):
        for outer in (up, gate):
            assert type(outer) is Linear and type(outer.inner) is LinearFP16
            assert outer.qmap and not outer.lora_a_tensors and not outer.lora_b_tensors
            assert not outer.is_sliced and not outer.trim_padded_out
            assert outer.in_features == outer.in_features_unpadded
            assert outer.out_features == outer.out_features_unpadded
            assert outer.pre_scale == outer.post_scale == 1.0 and outer.softcap == 0.0
            inner = outer.inner
            assert inner.bias is None and inner._pinned_store is None and inner.out_dtype == torch.float16
            assert inner.in_features == outer.in_features and inner.out_features == outer.out_features
            assert inner.weight.dtype == torch.float16 and inner.weight.is_cuda and inner.weight.is_contiguous()
            assert inner.weight.shape == (inner.in_features, inner.out_features)
        assert up.inner.weight.shape == gate.inner.weight.shape
        assert up.inner.weight.device == gate.inner.weight.device
        total += (up.inner.weight.numel() + gate.inner.weight.numel()) * 2
    assert total <= max_weight_bytes, ('Weight pack budget exceeded', total)
    for up, gate in zip(mlp.ups, mlp.gates):
        pairs.append((up.inner, gate.inner, torch.stack((up.inner.weight, gate.inner.weight))))
    return PairState(pairs, native, native.validate_context), total


@contextmanager
def installed(block, state):
    """Process-local hook, only after capture_fast.install; restore on all exits."""
    from exllamav3.modules.quant.fp16 import LinearFP16
    original = LinearFP16.forward
    assert getattr(original, '_gufo_capture_fast', False)
    assert capture_fast._active.get() is None
    def forward(inner, x, params, out_dtype=None):
        scope = capture_fast._active.get()
        if scope is not None and id(inner) in state.roles:
            assert params is scope[0], 'Capture params identity changed'
            # The existing scope validates terminal-block eligibility and downs.
            assert id(block.mlp.downs[0].inner) in scope[1]
            return state.call(inner, x, params, out_dtype)
        assert state.pending is None, 'Unexpected call between paired expert projections'
        return original(inner, x, params, out_dtype)
    LinearFP16.forward = forward
    try:
        yield state
        state.drained()
    finally:
        LinearFP16.forward = original
        # On failure release retained tensors; no fallback or retry.
        state.pending = None


def cleanup_all(actions, primary=None):
    """Run every cleanup even when one fails; retain the original exception."""
    errors = []
    for name, action in actions:
        try:
            action()
        except BaseException as error:
            errors.append((name, error))
    if errors:
        message = '; '.join(name + ': ' + repr(error) for name, error in errors)
        if primary is not None:
            primary.add_note('Cleanup failures: ' + message)
        else:
            raise RuntimeError('Cleanup failures: ' + message) from errors[0][1]
    return [(name, repr(error)) for name, error in errors]
