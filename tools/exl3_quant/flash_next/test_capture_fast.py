"""CPU scope contract, using explicit fake modules and a source-inspection mock.

First independently verify the real pinned cm.main source hash. Then mock only
inspect.getsource for a tiny fake main caller, so its direct-call frame can test
the real install/context/inner wrappers without importing GPU runtime modules.
This is not a numerical model test; the real affected-block probe supplies that.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import torch
import capture_fast as candidate


source_path, output_path = map(Path, sys.argv[1:3])
source = source_path.read_text()
main_node = next(n for n in ast.parse(source).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'main')
source_start = min([main_node.lineno] + [d.lineno for d in main_node.decorator_list])
pinned_main = ''.join(source.splitlines(keepends=True)[source_start-1:main_node.end_lineno])
actual_sha = hashlib.sha256(pinned_main.encode()).hexdigest()
assert actual_sha == candidate._MAIN_SHA, 'Actual pinned cm.main changed'


class LinearFP16:
    def __init__(self):
        self.in_features, self.out_features = 5, 3
        self.out_dtype = torch.float32
        self.bias = self._pinned_store = None
        self.real_calls = 0

    def forward(self, x, params, out_dtype=None):
        self.real_calls += 1
        return torch.full((*x.shape[:-1], self.out_features), 7.0,
                          device=x.device, dtype=out_dtype or self.out_dtype or torch.half)


class BlockSparseMLP:
    def __init__(self, inner):
        self.latent_out = self.cpu_split_first = self.tp_mode = self.routing_device = None
        self.routed_post_norm = self.shared_experts_post_norm = None
        self.tp_reduce = self.cpu_offload = False
        self.pruned = set()
        self.num_slices = self.num_experts = 1
        self.downs = [SimpleNamespace(inner=inner, qmap='block.mlp.0.down',
                                     lora_a_tensors=[], in_features=5, out_features=3)]


class GatedResidual:
    pass


class TransformerBlock:
    def __init__(self):
        self.target, self.decoy = LinearFP16(), LinearFP16()
        self.mlp = BlockSparseMLP(self.target)
        self.mlp_hc = GatedResidual()
        self.mlp_post_norm = self.ablit = None
        self.requested_dtype = None
        self.raise_after_inner = False

    def forward(self, x, params, out_dtype=None):
        target = self.target.forward(x, params, self.requested_dtype)
        decoy = self.decoy.forward(x, params, self.requested_dtype)
        copied_params = self.target.forward(x, dict(params), self.requested_dtype)
        if self.raise_after_inner:
            raise ValueError('intentional block failure')
        return target, decoy, copied_params


@torch.inference_mode()
def fake_main(block, x, params):
    return block.forward(x, params)


def other_caller(block, x, params):
    return block.forward(x, params)


fake_cm = SimpleNamespace(main=fake_main)
fake_main_body = inspect.unwrap(fake_main)
assert fake_main_body is not fake_main and fake_main_body.__code__ is not fake_main.__code__
real_getsource = inspect.getsource


def audited_source_mock(obj):
    if obj is fake_main_body:
        return pinned_main
    return real_getsource(obj)


imports = {}
for name, attr, cls in (
    ('exllamav3.modules.transformer', 'TransformerBlock', TransformerBlock),
    ('exllamav3.modules.block_sparse_mlp', 'BlockSparseMLP', BlockSparseMLP),
    ('exllamav3.modules.hyperconnections', 'GatedResidual', GatedResidual),
    ('exllamav3.modules.quant.fp16', 'LinearFP16', LinearFP16),
):
    module = ModuleType(name)
    setattr(module, attr, cls)
    imports[name] = module

checks = ['actual_pinned_main_source_hash_including_decorator',
          'actual_torch_inference_mode_decorator_has_distinct_wrapper_code']
x = torch.arange(35, dtype=torch.float32).reshape(1, 7, 5)
original_x = x.clone()
params = dict(capture={}, activate_all_experts=True, quant_preserve={})
original_inner, original_block = LinearFP16.forward, TransformerBlock.forward


def all_real(outputs):
    assert all(torch.all(v == 7).item() for v in outputs)


def rejected(fn):
    try:
        fn()
    except AssertionError:
        return
    raise AssertionError('Unsupported contract accepted')


with patch.dict(sys.modules, imports):
    # A source mismatch must fail before changing either class method.
    with patch.object(candidate.inspect, 'getsource', return_value=pinned_main+'# changed\n'):
        rejected(lambda: candidate.install(fake_cm))
    assert LinearFP16.forward is original_inner and TransformerBlock.forward is original_block
    checks.append('source_change_rejected_before_patch')

    with patch.object(candidate.inspect, 'getsource', side_effect=audited_source_mock):
        restore = candidate.install(fake_cm)
    try:
        block = TransformerBlock()
        for dtype in (None, torch.float16, torch.bfloat16, torch.float64):
            block.requested_dtype = dtype
            result = fake_main(block, x, params)
            target, decoy, copied = result
            assert target.shape == (1, 7, 3) and target.device == x.device
            assert target.dtype == (dtype or torch.float32)
            assert torch.count_nonzero(target).item() == 0
            assert torch.all(decoy == 7).item() and torch.all(copied == 7).item()
            assert candidate._active.get() is None
        checks.append('direct_caller_only_target_identity_same_params_shape_device_dtype')
        assert torch.equal(x, original_x) and params == dict(capture={}, activate_all_experts=True, quant_preserve={})
        checks.append('input_and_params_unmodified')

        all_real(other_caller(block, x, params))
        all_real(block.forward(x, params))
        checks.append('foreign_and_direct_external_callers_keep_real_output')
        for ordinary in (dict(quant_preserve={}),
                         dict(capture={}, activate_all_experts=False, quant_preserve={}),
                         dict(activate_all_experts=True, quant_preserve={})):
            all_real(fake_main(block, x, ordinary))
        checks.append('normal_reference_and_advance_keep_real_output')

        block.raise_after_inner = True
        try:
            fake_main(block, x, params)
        except ValueError as error:
            assert str(error) == 'intentional block failure'
        else:
            raise AssertionError('Missing fixture failure')
        assert candidate._active.get() is None
        block.raise_after_inner = False
        all_real(other_caller(block, x, params))
        checks.append('exception_resets_scope_and_next_foreign_call_is_normal')

        for key in ('export_states', 'export_state_layers', 'hc_pending', 'q_mlp_slice', 'backend'):
            rejected(lambda key=key: fake_main(block, x, {**params, key: None}))
            assert candidate._active.get() is None
        checks.append('export_pending_slice_backend_rejected')
        for key, bad in (('latent_out', object()), ('tp_reduce', True),
                         ('cpu_split_first', 0), ('cpu_offload', True), ('pruned', {0}),
                         ('tp_mode', object()), ('routing_device', 'cpu'),
                         ('routed_post_norm', object()), ('shared_experts_post_norm', object()),
                         ('num_slices', 2)):
            previous = getattr(block.mlp, key)
            setattr(block.mlp, key, bad)
            try:
                rejected(lambda: fake_main(block, x, params))
            finally:
                setattr(block.mlp, key, previous)
            assert candidate._active.get() is None
        checks.append('unsupported_moe_structures_rejected')
        block.mlp.downs[0].lora_a_tensors = [object()]
        rejected(lambda: fake_main(block, x, params))
        block.mlp.downs[0].lora_a_tensors = []
        for key, bad in (('bias', torch.zeros(3)), ('_pinned_store', object()),
                         ('out_dtype', torch.float16)):
            previous = getattr(block.target, key)
            setattr(block.target, key, bad)
            try:
                rejected(lambda: fake_main(block, x, params))
            finally:
                setattr(block.target, key, previous)
        checks.append('lora_bias_pinned_storage_wrong_default_dtype_rejected')
        with candidate._discarded_capture(block, params):
            rejected(lambda: fake_main(block, x, params))
        assert candidate._active.get() is None
        checks.append('nested_scope_rejected_and_reset')
        with patch.object(candidate.inspect, 'getsource', side_effect=audited_source_mock):
            rejected(lambda: candidate.install(fake_cm))
        checks.append('duplicate_install_rejected')
    finally:
        restore()
    assert LinearFP16.forward is original_inner and TransformerBlock.forward is original_block
    all_real(fake_main(block, x, params))
    checks.append('restore_original_class_methods')

    with patch.object(candidate.inspect, 'getsource', side_effect=audited_source_mock):
        restore = candidate.install(fake_cm, automatic=False)
    try:
        all_real(fake_main(block, x, params))
        with candidate._discarded_capture(block, params):
            assert torch.count_nonzero(other_caller(block, x, params)[0]).item() == 0
        all_real(other_caller(block, x, params))
        checks.append('probe_only_mode_requires_explicit_scope_and_resets')
    finally:
        restore()

result = dict(status='pass', gpu_used=False, checks=checks, stats=candidate.STATS,
              scope='CPU fake-module control flow only; real numerical qualification is separate',
              mock='Only inspect.getsource(inspect.unwrap(fake_main)) returns separately verified decorator-inclusive pinned main source; fake_main uses actual torch.inference_mode(); runtime modules are CPU fixtures; _targets and wrappers are real candidate code',
              actual_main_sha256=actual_sha,
              candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
              test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
