"""Bounded c03 long-row forward/capture/packed-operator check; no full conversion."""
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import types

import torch
from exllamav3 import Config, Model, Tokenizer
from exllamav3.modules import Linear
from exllamav3.modules.linear import flush_capture_H
from exllamav3.modules.quant import fp16, exl3
from exllamav3.conversion.convert_model import make_quant_args
from exllamav3.conversion.ngram import prepare_ngram_table_for_conversion
from ragged_calibration import load_calibration

torch.set_grad_enabled(False)
torch.manual_seed(0)
assert not fp16._f32_via_f16 and not exl3._f32_via_f16
out = Path('/output/long-row-pilot')
out.mkdir(exist_ok=True)
assert not (out / 'SUMMARY.json').exists(), 'Reuse an existing pass instead of repeating'
manifest = json.loads(Path('/task/c03-input-ids.json').read_text())
config = Config.from_directory('/source')
model = Model.from_config(config)
prepare_ngram_table_for_conversion(dict(out_dir='/output/pack',
    ngram_file='/output/pack/ngram_embedding.safetensors'), config, model)
tokenizer = Tokenizer.from_config(config)
all_rows = load_calibration(dict(cal_data='/task/c03-input-ids.safetensors',
    in_dir='/source', cal_rows=manifest['rows'], cal_cols=manifest['max_columns']), tokenizer)
long_index = next(i for i, r in enumerate(all_rows) if r.shape[1] == 8192)
short_index = next(i for i, r in enumerate(all_rows) if r.shape[1] == 512)
ids = [all_rows[long_index], all_rows[short_index]]
state = [r.clone() for r in ids]
preserves = [{}, {}]
recipe = json.loads(Path('/task/b1-recipe.json').read_text())['tensors']
results, stages = [], []
start = time.monotonic()
for index, module in enumerate(model.modules[:7]):
    capture = index in (2, 6)  # First GDN and first full-attention block.
    suffixes = ('linear_attn.in_proj_qkv', 'mlp.experts.0.down_proj',
                'self_attn.q_proj', 'self_attn.k_proj')
    selected = {m.key: m for m in module if isinstance(m, Linear)
                and any(m.key.endswith('.' + s) for s in suffixes)} if capture else {}
    inputs = {key: [] for key in selected}
    for key, linear in selected.items():
        original = linear.capture_H
        def hook(self, x, params, _original=original, _key=key):
            inputs[_key].append(x.detach().reshape(-1, x.shape[-1]).cpu().clone())
            return _original(x, params)
        linear.capture_H = types.MethodType(hook, linear)
    defer = module.can_defer_load()
    if defer:
        config.stc.begin_deferred_load()
    try:
        module.load(torch.device('cpu') if module.caps.get('prefer_cpu') else torch.device('cuda:0'))
    finally:
        if defer:
            config.stc.end_deferred_load()
    config.stc.close()
    H = {}
    for i, x in enumerate(state):
        params = dict(input_ids=ids[i], attn_mode='flash_attn_nc', quant_preserve=preserves[i])
        if capture:
            params.update(capture=H, activate_all_experts=True)
        y = module.forward(module.prepare_for_device(x, params), params)
        flush_capture_H(params)
        assert torch.isfinite(y).all(), module.key
        if capture:
            # All-expert output is a capture probe, not the real routed state.
            del y
            normal = dict(input_ids=ids[i], attn_mode='flash_attn_nc', quant_preserve=preserves[i])
            y = module.forward(module.prepare_for_device(x, normal), normal)
            assert torch.isfinite(y).all(), module.key
        state[i] = y.cpu()
        preserves[i] = params['quant_preserve']
        del y
    torch.cuda.synchronize()
    for hd in H.values():
        assert torch.isfinite(hd['H']).all()
        assert int(hd['inf_nan'].sum().item()) == 0
    for key, linear in selected.items():
        original = linear.inner
        w = original.get_weight_tensor().float().clone()
        x = torch.cat(inputs[key]).to(w.device).float()
        expected = x @ w
        qa = make_quant_args(dict(apply_out_scales=True, codebook='mul1', work_dir=str(out)),
                             index, recipe[key], [0])
        try:
            proxy, wq = linear.convert_exl3(copy.deepcopy(H[linear.qmap]), qa, return_weight_q=True)
            packed = linear.inner.forward(x.to(inputs[key][0].dtype).unsqueeze(0).contiguous(), {})
            packed = packed.reshape_as(expected).float()
            relative = float(torch.linalg.vector_norm(packed - expected) / torch.linalg.vector_norm(expected))
            assert torch.isfinite(wq).all() and torch.isfinite(packed).all()
            for name in ('suh', 'svh', 'bias'):
                value = getattr(linear.inner, name)
                assert value is None or torch.isfinite(value).all()
            assert relative < 0.05 and math.isfinite(float(proxy)) and float(proxy) >= 0 and not qa.get('q_fallback'), (key, relative)
            results.append(dict(key=key, K=recipe[key], packed_relative_error=relative, proxy=float(proxy)))
        finally:
            linear.inner = original
        del w, x, expected, packed, wq
    stages.append(dict(index=index, key=module.key, capture=capture, hessians=len(H),
                       peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
    module.unload()
    del H, inputs
    torch.cuda.empty_cache()
    (out / 'PROGRESS.json').write_text(json.dumps(dict(stages=stages, operators=results), indent=2))
    print(module.key, 'passed', flush=True)
assert any('.linear_attn.' in r['key'] for r in results)
assert any('.self_attn.' in r['key'] for r in results)
summary = dict(status='long_row_screen_pass', calibration_sha256=manifest['sha256'],
    pilot_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_indices=[long_index, short_index], lengths=[8192, 512], stages=stages, operators=results,
    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic()-start,
    scope='First GDN/full-attention, mixed lengths, completed K8 ngram/PLE; full-model memory and quality remain monitored/unqualified')
(out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
