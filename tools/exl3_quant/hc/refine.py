"""CPU Q8_0 coordinate refinement; reuses a saved GPTQ candidate and its H.

Independent implementation of output-error coordinate descent. No GPU use,
activation capture, model deployment, or held-out-data-driven optimization.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from safetensors.numpy import load_file, save_file

from . import quant as q
from .weights import load_weight


def scale_choices(optimum):
    """Nearest representable positive F16 optimum and its two neighbors.

    The fixed-code objective is quadratic in the scale. Its minimum on the
    positive F16 grid lies at a bracketing value (including boundary values).
    Zero is excluded for nonzero groups; the current scale is always retained.
    """
    tiny = float(np.nextafter(np.float16(0), np.float16(1)))
    top = float(np.finfo(np.float16).max)
    if not np.isfinite(optimum).all():
        raise ValueError('nonfinite scale optimum')
    center = np.clip(optimum, tiny, top).astype(np.float16)
    with np.errstate(over='ignore'):
        choices = np.stack((center, np.nextafter(center, np.float16(0)),
                            np.nextafter(center, np.float16(np.inf))), axis=1)
    return choices.astype(np.float64)


def refine(w, blocks, h, damping, *, sweeps=6, scales=False, row_batch=32):
    """Minimize sum(e @ (H + damping I) * e) with original BF16 reference.

    Codes stay in [-127,127], groups stay 32, scales stay positive stored F16.
    Each sweep visits code columns then (optionally) group scales. Scale moves
    use the constrained least-squares optimum snapped to actual F16 bytes.
    This is a bounded CPU reference, not a full-size throughput implementation.
    """
    start = time.monotonic()
    w = np.asarray(w, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    if w.ndim != 2 or not w.size or w.shape[1] % 32 or not np.isfinite(w).all():
        raise ValueError('invalid weights')
    n = w.shape[1]
    if h.shape != (n, n) or not np.isfinite(h).all() or not np.allclose(h, h.T, rtol=1e-10, atol=1e-12):
        raise ValueError('invalid H')
    if not np.isfinite(damping) or damping <= 0 or type(sweeps) is not int or not 1 <= sweeps <= 16:
        raise ValueError('invalid damping or sweep budget')
    if type(row_batch) is not int or not 1 <= row_batch <= 256:
        raise ValueError('invalid row budget')
    if blocks.dtype != np.uint8 or blocks.shape != (w.shape[0], n // 32, 34):
        raise ValueError('invalid packed shape')
    codes = np.ascontiguousarray(blocks[..., 2:]).view(np.int8).reshape(w.shape).copy()
    ds = np.ascontiguousarray(blocks[..., :2]).view('<f2').reshape(w.shape[0], -1).copy()
    if not np.array_equal(q.pack(codes, ds), blocks):
        raise ValueError('invalid packed values')
    if np.any((ds == 0) & np.any(w.reshape(w.shape[0], -1, 32) != 0, axis=2)):
        raise ValueError('zero stored scale on nonzero source group')
    hd = (h + h.T) * .5
    hd.flat[::n + 1] += damping
    np.linalg.cholesky(hd)  # Reject invalid objective; no identity/RTN fallback.
    trace = []
    for r0 in range(0, len(w), row_batch):
        sl = slice(r0, min(r0 + row_batch, len(w)))
        c = codes[sl].astype(np.int16)
        d = ds[sl].astype(np.float64)
        error = w[sl] - c * np.repeat(d, 32, axis=1)
        grad = error @ hd
        initial = float(np.sum(error * grad))
        previous = initial
        for sweep in range(sweeps):
            code_moves = scale_moves = 0
            for j in range(n):
                scale = d[:, j // 32]
                best = np.zeros(len(c)); step = np.zeros(len(c), dtype=np.int16)
                for direction in (-1, 1):
                    delta = direction * scale
                    gain = -2 * delta * grad[:, j] + delta * delta * hd[j, j]
                    valid = (c[:, j] + direction >= -127) & (c[:, j] + direction <= 127)
                    take = valid & (gain < best)
                    best[take] = gain[take]; step[take] = direction
                delta = step * scale
                c[:, j] += step
                error[:, j] -= delta
                grad -= delta[:, None] * hd[j, :]
                code_moves += int(np.count_nonzero(step))
            if scales:
                # Refresh after a long sequence of incremental updates.
                grad = error @ hd
                for group in range(n // 32):
                    span = slice(group * 32, (group + 1) * 32)
                    v = c[:, span].astype(np.float64)
                    den = np.sum((v @ hd[span, span]) * v, axis=1)
                    num = np.sum(v * grad[:, span], axis=1)
                    old = d[:, group].copy()
                    opt = old + np.divide(num, den, out=np.zeros_like(num), where=den > 0)
                    best = np.zeros(len(c)); selected = old.copy()
                    for candidate in scale_choices(opt).T:
                        valid = np.isfinite(candidate) & (candidate > 0) & (old > 0) & (den > 0)
                        delta = np.where(valid, candidate - old, 0)
                        gain = -2 * delta * num + delta * delta * den
                        take = valid & (gain < best)
                        best[take] = gain[take]; selected[take] = candidate[take]
                    delta_w = (selected - old)[:, None] * v
                    error[:, span] -= delta_w
                    grad -= delta_w @ hd[span, :]
                    d[:, group] = selected
                    scale_moves += int(np.count_nonzero(selected != old))
            # Check the actual packed-value objective, not the incremental G.
            error = w[sl] - c * np.repeat(d, 32, axis=1)
            grad = error @ hd
            now = float(np.sum(error * grad))
            tolerance = 1e-10 * max(abs(previous), 1e-20)
            if not np.isfinite(now) or now > previous + tolerance:
                raise ValueError('refinement increased stored-value objective')
            trace.append({'row_start': r0, 'rows': len(c), 'sweep': sweep + 1,
                          'before': previous, 'after': now,
                          'code_moves': code_moves, 'scale_moves': scale_moves})
            previous = now
            if code_moves == 0 and scale_moves == 0:
                break
        codes[sl] = c.astype(np.int8); ds[sl] = d.astype(np.float16)
    packed = q.pack(codes, ds)
    return packed, {'method': 'full_H_Q8_coordinate_codes_and_F16_scales' if scales else 'full_H_Q8_coordinate_codes',
                    'objective': 'H_plus_existing_GPTQ_damping_I', 'damping': damping,
                    'sweep_limit': sweeps, 'row_batch': row_batch, 'trace': trace,
                    'seconds': time.monotonic() - start}


def compare(source_model, target_model, baseline, stats, output, sweeps=6):
    """No GPTQ/baseline rerun: read the existing three packed artifacts."""
    source_model, target_model, baseline, stats, output = map(Path, (source_model, target_model, baseline, stats, output))
    if output.exists() or any(output.resolve().is_relative_to(p.resolve()) for p in (source_model, target_model, baseline)):
        raise ValueError('output must be new and outside models/baseline')
    result = json.loads((baseline / 'RESULT.json').read_text())
    key = result['tensor']
    w, identity = load_weight(source_model, key, 'BF16')
    _, target = load_weight(target_model, key, 'F16')
    if identity != result['source_identity'] or target != result['target_identity']:
        raise ValueError('source/target identity mismatch')
    receipt = result['receipt']
    expected = {'schema': 'gufo-hc-input-gram-v2', 'tensor': key,
                'source_identity': identity, 'target_identity': target,
                'input_space': 'unfolded_projection_input',
                'H_normalization': 'sum_xxt_div_rows', 'synthetic': False}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('incompatible saved capture receipt')
    if result.get('method') != 'fixed_stored_F16_scale_full_H_sequential_GPTQ_style':
        raise ValueError('unsupported baseline method')
    if not key.endswith(('.input_mix_weight_down.weight', '.input_mix_weight_up.weight')):
        raise ValueError('only HC down/up supported')
    if type(receipt.get('calibration_rows')) is not int or receipt['calibration_rows'] <= 0:
        raise ValueError('invalid calibration rows')
    for k in ('capture_source_sha256', 'capture_runtime_sha256', 'upstream_weight_state_sha256'):
        v = receipt.get(k, '')
        if not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError('missing capture identity')
    if receipt['stats_sha256'] != q.sha(stats) or receipt.get('synthetic') is not False:
        raise ValueError('stats identity mismatch')
    train, held = receipt['calibration_document_ids'], receipt['holdout_document_ids']
    if not train or not held or set(train) & set(held):
        raise ValueError('invalid document split')
    with np.load(stats, allow_pickle=False) as f:
        h = np.asarray(f['H'], dtype=np.float64)
        x = np.asarray(f['X_holdout'], dtype=np.float64)
    if x.ndim != 2 or x.shape != (receipt['holdout_rows'], w.shape[1]) or not x.size or not np.isfinite(x).all():
        raise ValueError('invalid holdout')
    controls = {}
    for name, filename, digest in (
        ('gptq', 'candidate.safetensors', 'candidate_sha256'),
        ('bf16_rtn', 'bf16-rtn-control.safetensors', 'bf16_control_sha256'),
        ('legacy_f16_rtn', 'legacy-f16-rtn-control.safetensors', 'legacy_control_sha256')):
        path = baseline / filename
        if q.sha(path) != result[digest]:
            raise ValueError('changed saved baseline: ' + filename)
        tensors = load_file(str(path))
        if set(tensors) != {key}:
            raise ValueError('unexpected baseline tensor set')
        controls[name] = tensors[key]
    artifacts, reports = {}, {}
    for name, change_scales in (('gptq_codes', False), ('gptq_codes_scales', True)):
        artifacts[name], reports[name] = refine(w, controls['gptq'], h, result['damping'],
                                               sweeps=sweeps, scales=change_scales)
    all_packs = controls | artifacts
    metrics = {name: q.measure(w, packed, h, x) for name, packed in all_packs.items()}
    for name, packed in all_packs.items():
        error = q.unpack(packed) - w.astype(np.float64)
        metrics[name]['regularized_weighted_sse'] = metrics[name]['calibration_weighted_sse'] + result['damping'] * float(np.sum(error * error))
    if not all(np.isfinite(v) for m in metrics.values() for v in m.values()):
        raise ValueError('nonfinite metric')
    output.mkdir(parents=True)
    for name, packed in artifacts.items():
        save_file({key: packed}, str(output / (name + '.safetensors')),
                  metadata={'status': 'NOT_RUNTIME_QUALIFIED', 'role': name,
                            'layout': 'uint8[out,in/32,34]: little_endian_f16_scale,int8_codes'})
    summary = {'status': 'comparison_pending_native_connected_HC_and_model_validation',
               'reference': 'original_BF16_exactly_expanded_to_FP32', 'tensor': key,
               'source_identity': identity, 'target_identity': target, 'receipt': receipt,
               'baseline_result_sha256': q.sha(baseline / 'RESULT.json'),
               'baseline_files': {name: result[name] for name in ('candidate_sha256', 'bf16_control_sha256', 'legacy_control_sha256')},
               'refiner_sha256': q.sha(__file__), 'quant_sha256': q.sha(q.__file__),
               'weight_loader_sha256': q.sha(Path(__file__).with_name('weights.py')),
               'numpy': np.__version__, 'metrics': metrics, 'runs': reports,
               'artifacts': {name: q.sha(output / (name + '.safetensors')) for name in artifacts},
               'selection': 'none', 'holdout_used_for_optimization': False,
               'activation_Q8_error_corrected': False, 'gpu_used': False}
    (output / 'RESULT.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-model', 'target-model', 'baseline', 'stats', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--sweeps', type=int, default=6)
    print(json.dumps(compare(**vars(parser.parse_args(argv))), indent=2))


if __name__ == '__main__':
    main()
