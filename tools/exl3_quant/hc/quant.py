"""CPU reference for fixed-scale, full-H GPTQ-style HC Q8_0 candidates.

No torch import, device discovery, model execution or runtime installation.
The candidate is numerical (not RTN-exact); native kernel parity is still required.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from .weights import load_weight


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def round_away(a):
    # Adding .5 first can round a value just BELOW .5 up to 1.0, even in
    # float64. Compare the fractional part without that addition.
    a = np.asarray(a, dtype=np.float64)
    magnitude = np.abs(a)
    lower = np.floor(magnitude)
    return np.copysign(lower + (magnitude - lower >= 0.5), a)


def rtn(w):
    """Group32 RTN; BF16 source arrives as exact FP32, legacy control as F16.

    F16 inputs reproduce Gufo QuantizeHalfQ8_0 arithmetic. FP32 inputs do NOT
    assert equivalence to that current native F16 conversion path.
    """
    if w.dtype not in (np.dtype('float16'), np.dtype('float32')) or w.ndim != 2 or w.shape[1] % 32:
        raise ValueError('expected F16/F32 [out,in] with in divisible by 32')
    if not np.isfinite(w).all() or not w.size:
        raise ValueError('empty or nonfinite weights')
    g = w.astype(np.float32).reshape(w.shape[0], -1, 32)
    d32 = np.max(np.abs(g), axis=2) / np.float32(127)
    with np.errstate(over='ignore', under='ignore'):
        d16 = d32.astype(np.float16)
    # A nonzero group whose stored scale vanishes cannot be calibrated with
    # this fixed-scale contract. Keep the failure explicit, never change scale.
    if not np.isfinite(d16).all() or np.any((np.max(np.abs(g), axis=2) != 0) & (d16 == 0)):
        raise ValueError('source group scale not representable in Q8_0 F16; unsupported')
    inv = np.divide(np.float32(1), d32, out=np.zeros_like(d32), where=d32 != 0)
    codes = round_away(g * inv[..., None]).astype(np.int8)
    return codes.reshape(w.shape), d16


def pack(codes, scales):
    if codes.dtype != np.int8 or codes.ndim != 2 or codes.shape[1] % 32:
        raise ValueError('invalid int8 code matrix')
    if scales.dtype != np.float16 or scales.shape != (codes.shape[0], codes.shape[1] // 32):
        raise ValueError('invalid scale layout')
    if not np.isfinite(scales).all() or np.any(scales < 0) or np.any(codes == -128):
        raise ValueError('invalid code/scale values')
    blocks = np.zeros((*scales.shape, 34), dtype=np.uint8)
    blocks[..., :2] = np.ascontiguousarray(scales.astype('<f2')).view(np.uint8).reshape(*scales.shape, 2)
    blocks[..., 2:] = codes.view(np.uint8).reshape(*scales.shape, 32)
    return blocks


def unpack(blocks):
    if blocks.dtype != np.uint8 or blocks.ndim != 3 or blocks.shape[-1] != 34:
        raise ValueError('invalid Q8_0 blocks')
    scales = np.ascontiguousarray(blocks[..., :2]).view('<f2').reshape(blocks.shape[:2])
    codes = np.ascontiguousarray(blocks[..., 2:]).view(np.int8)
    return (codes.astype(np.float64) * scales[..., None]).reshape(blocks.shape[0], -1)


def gptq(w, h, damp_fraction=0.01):
    """Sequential inverse-Cholesky error feedback; no activation-order sort.

    H is sum(X.T@X)/N of UNFOLDED projection inputs, not a diagonal or
    groupwise substitute. Scales remain the RTN stored F16 scales.
    This slow CPU reference is not a qualified full-model GPU converter.
    """
    base, scales = rtn(w)
    k = w.shape[1]
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (k, k) or not np.isfinite(h).all():
        raise ValueError('invalid H')
    if not np.allclose(h, h.T, rtol=1e-10, atol=1e-12):
        raise ValueError('H is not symmetric')
    if not np.isfinite(damp_fraction) or not 0 < damp_fraction <= 1:
        raise ValueError('invalid damping')
    h = (h + h.T) * 0.5
    diag = np.diag(h)
    if np.any(diag < 0) or diag.mean() <= 0:
        raise ValueError('H has no positive input energy')
    damping = float(damp_fraction * diag.mean())
    hd = h.copy()
    hd.flat[::k + 1] += damping
    # An invalid/non-SPD matrix must fail instead of substituting identity H.
    l = np.linalg.cholesky(hd)
    li = np.linalg.solve(l, np.eye(k))
    u = np.linalg.cholesky(li.T @ li).T
    cur = w.astype(np.float64)
    q = np.zeros_like(base)
    s = np.repeat(scales.astype(np.float64), 32, axis=1)
    for col in range(k):
        d = s[:, col]
        code = round_away(np.divide(cur[:, col], d, out=np.zeros(w.shape[0]), where=d != 0))
        code = np.clip(code, -127, 127).astype(np.int8)
        code[d == 0] = 0
        q[:, col] = code
        err = (cur[:, col] - code.astype(np.float64) * d) / u[col, col]
        cur[:, col:] -= err[:, None] * u[col, col:][None, :]
    if not np.isfinite(cur).all():
        raise ValueError('nonfinite feedback')
    return pack(q, scales), pack(base, scales), damping


def measure(w, blocks, h, x):
    err = unpack(blocks) - w.astype(np.float64)
    diff = x @ err.T
    ref = x @ w.astype(np.float64).T
    return {
        'calibration_weighted_sse': float(np.sum((err @ h) * err)),
        'holdout_output_sse': float(np.sum(diff * diff)),
        'holdout_reference_sse': float(np.sum(ref * ref)),
        'holdout_abs_p99': float(np.quantile(np.abs(diff), .99)),
        'holdout_abs_max': float(np.max(np.abs(diff))),
    }


def generate(source_model, target_model, tensor, stats, receipt, output, damp_fraction):
    """One projection per directory. Never writes into the model directory."""
    source_model, target_model, stats, receipt, output = map(Path, (source_model, target_model, stats, receipt, output))
    if any(output.resolve().is_relative_to(m.resolve()) for m in (source_model, target_model)) or output.exists():
        raise ValueError('output must be new and outside both models')
    if not tensor.endswith(('.input_mix_weight_down.weight', '.input_mix_weight_up.weight')):
        raise ValueError('only HC down/up weights supported')
    w, source_identity = load_weight(source_model, tensor, 'BF16')
    legacy_w, target_identity = load_weight(target_model, tensor, 'F16')
    if w.shape != legacy_w.shape:
        raise ValueError('source/target HC shape mismatch')
    meta = json.loads(receipt.read_text())
    expected = {
        'schema': 'gufo-hc-input-gram-v2', 'tensor': tensor,
        'source_identity': source_identity, 'target_identity': target_identity,
        'stats_sha256': sha(stats), 'input_space': 'unfolded_projection_input',
        'H_normalization': 'sum_xxt_div_rows', 'synthetic': False,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f'receipt mismatch: {key}')
    train = meta.get('calibration_document_ids', [])
    held = meta.get('holdout_document_ids', [])
    if not train or not held or set(train) & set(held):
        raise ValueError('require disjoint nonempty document IDs')
    for key in ('capture_source_sha256', 'capture_runtime_sha256', 'upstream_weight_state_sha256'):
        v = meta.get(key, '')
        if not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError(f'missing capture identity: {key}')
    if type(meta.get('calibration_rows')) is not int or meta['calibration_rows'] <= 0:
        raise ValueError('invalid calibration rows')
    with np.load(stats, allow_pickle=False) as f:
        h = np.array(f['H'], dtype=np.float64)
        x = np.array(f['X_holdout'], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != w.shape[1] or not x.shape[0] or not np.isfinite(x).all():
        raise ValueError('invalid holdout inputs')
    if meta.get('holdout_rows') != x.shape[0]:
        raise ValueError('holdout row count mismatch')
    candidate, control, damping = gptq(w, h, damp_fraction)
    legacy_control = pack(*rtn(legacy_w))
    result = {
        'status': 'candidate_pending_native_and_model_validation',
        'tensor': tensor, 'shape': list(w.shape), 'receipt': meta,
        'source_identity': source_identity, 'target_identity': target_identity,
        'reference': 'original_BF16_exactly_expanded_to_FP32',
        'method': 'fixed_stored_F16_scale_full_H_sequential_GPTQ_style',
        'damp_fraction': damp_fraction, 'damping': damping,
        'generator_sha256': sha(__file__), 'numpy': np.__version__,
        'weight_loader_sha256': sha(Path(__file__).with_name('weights.py')),
        'candidate': measure(w, candidate, h, x), 'bf16_rtn': measure(w, control, h, x),
        'legacy_f16_rtn': measure(w, legacy_control, h, x),
        'bf16_vs_pack_f16_max_abs': float(np.max(np.abs(w.astype(np.float64) - legacy_w))),
        'scale_bytes_equal': bool(np.array_equal(candidate[..., :2], control[..., :2])),
        'native_kernel_parity': 'not_run', 'runtime_loader': 'not_implemented',
        'activation_Q8_error_corrected': False,
    }
    # Refuse to call a failed finite calculation a usable candidate.
    if not all(np.isfinite(v) for section in ('candidate', 'bf16_rtn', 'legacy_f16_rtn') for v in result[section].values()):
        raise ValueError('nonfinite output metric')
    output.mkdir(parents=True)
    metadata = {'schema': 'gufo-hc-q8-sidecar-experimental-v2', 'tensor': tensor,
                'source_dtype': 'BF16', 'source_identity': json.dumps(source_identity, sort_keys=True),
                'target_identity': json.dumps(target_identity, sort_keys=True),
                'layout': 'uint8[out,in/32,34]: little_endian_f16_scale,int8_codes',
                'status': 'NOT_RUNTIME_QUALIFIED'}
    save_file({tensor: candidate}, str(output / 'candidate.safetensors'), metadata=metadata | {'role': 'bf16_gptq_candidate'})
    save_file({tensor: control}, str(output / 'bf16-rtn-control.safetensors'), metadata=metadata | {'role': 'bf16_rtn_control'})
    save_file({tensor: legacy_control}, str(output / 'legacy-f16-rtn-control.safetensors'),
              metadata=metadata | {'role': 'target_pack_f16_rtn_control', 'source_dtype': 'F16'})
    result['candidate_sha256'] = sha(output / 'candidate.safetensors')
    result['bf16_control_sha256'] = sha(output / 'bf16-rtn-control.safetensors')
    result['legacy_control_sha256'] = sha(output / 'legacy-f16-rtn-control.safetensors')
    (output / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-model', 'target-model', 'tensor', 'stats', 'receipt', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--damp-fraction', type=float, default=.01)
    args = vars(p.parse_args(argv))
    print(json.dumps(generate(**args), indent=2))


if __name__ == '__main__':
    main()
