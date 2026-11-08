"""MTP calibration input/count contracts, independent of a GPU runner."""
from __future__ import annotations

import types

# The current adapter mirrors pinned Kyojin semantics. Gufo uses a different
# hidden normalization/tap precision; do not run/qualify this as Gufo calibration.
TARGET_GUFO_SEMANTICS_QUALIFIED = False


def shifted_prompt_state(state, input_ids, *, hc_mult=4, hidden_size=2560):
    """Match GatedResidual export + GeneratorJob prompt MTP alignment.

    Each calibration row is a fresh sequence: position zero has no previous
    trunk state. IDs keep their original positions and every token is retained.
    Conversion checkpoints contain already-applied HC residuals, not a deferred
    hc_pending operation. Callers must forbid pending HC state.
    """
    import torch
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError('Expected one independent token sequence')
    if state.shape != (*input_ids.shape, hc_mult, hidden_size):
        raise ValueError('Expected pre-mixer HC stack, never collapsed hidden state')
    if state.dtype != torch.float32 or input_ids.dtype != torch.int64:
        raise ValueError('Unexpected checkpoint/ID precision')
    if input_ids.shape[1] < 1 or not torch.isfinite(state).all():
        raise ValueError('Empty or nonfinite source state')
    tap = state.flatten(-2).half()
    if not torch.isfinite(tap).all():
        raise ValueError('Nonfinite runtime FP16 MTP tap')
    shifted = torch.empty_like(tap)
    shifted[:, :1].zero_()
    shifted[:, 1:].copy_(tap[:, :-1])
    return shifted


def expected_count(key, token_count, hc_mult=4):
    return token_count * hc_mult if key == 'mtp.fc_hidden' else token_count


def configure_mtp(model, recipe, linear_class):
    """Explicit key allocation; separate input Hessians with different samples."""
    targets = {m.key: m for module in model.modules for m in module
               if isinstance(m, linear_class) and m.qmap}
    if set(targets) != set(recipe):
        raise ValueError('MTP recipe must cover exactly all quantized projections')
    if any(type(v) is not int or not 1 <= v <= 8 for v in recipe.values()):
        raise ValueError('Unsupported MTP bit allocation')
    if set(('mtp.fc_hidden', 'mtp.fc_embedding')) - targets.keys():
        raise ValueError('Missing MTP input projections')
    targets['mtp.fc_hidden'].qmap = 'block.mtp.fc_hidden'
    targets['mtp.fc_embedding'].qmap = 'block.mtp.fc_embedding'
    model.calibration_all_experts = True
    return targets


def attach_prompt_prepare(module, *, hc_mult=4, hidden_size=2560):
    original = module.prepare_for_device

    def prepare(self, state, params):
        if 'hc_pending' in params or 'target_hidden' in params:
            raise ValueError('Unexpected reused or deferred MTP state')
        ids = params['input_ids']
        params['target_hidden'] = shifted_prompt_state(
            state, ids, hc_mult=hc_mult, hidden_size=hidden_size).to(self.device)
        return ids.to(self.device)

    module.prepare_for_device = types.MethodType(prepare, module)
    return original
