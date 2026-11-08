"""Score saved HC candidates per document without capturing or fitting again."""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from .prepare_stats import read_windows
from .quant import sha, unpack
from .weights import load_weight


def errors(x, w, packed):
    reference = x @ w.T
    delta = x @ (unpack(packed) - w).T
    sse = float(np.sum(delta * delta))
    energy = float(np.sum(reference * reference))
    if energy <= 0:
        raise ValueError('zero reference energy')
    result = {'output_sse': sse, 'reference_sse': energy,
              'relative_rmse': float(np.sqrt(sse / energy)),
              'abs_p99': float(np.quantile(np.abs(delta), .99)),
              'abs_max': float(np.max(np.abs(delta)))}
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('nonfinite metric')
    return result


def silu_quarter(x):
    # Stable FP64 ideal arithmetic; not Gufo's GPU activation/Q8 rounding.
    x = x * .25
    z = np.exp(-np.abs(x))
    sigmoid = np.where(x >= 0, 1 / (1 + z), z / (1 + z))
    return x * sigmoid


def connected_errors(x, down, up, reference):
    candidate = silu_quarter(x @ down.T) @ up.T
    delta = candidate - reference
    sse = float(np.sum(delta * delta))
    energy = float(np.sum(reference * reference))
    if energy <= 0 or not np.isfinite(delta).all():
        raise ValueError('invalid connected reference or candidate')
    result = {'output_sse': sse, 'reference_sse': energy,
              'relative_rmse': float(np.sqrt(sse / energy)),
              'abs_p99': float(np.quantile(np.abs(delta), .99)),
              'abs_max': float(np.max(np.abs(delta)))}
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('nonfinite connected metric')
    return result


def score(capture, inputs, source_model, target_model, results, output):
    capture, inputs, source_model, target_model, results, output = map(
        Path, (capture, inputs, source_model, target_model, results, output))
    if output.exists() or any(output.resolve().is_relative_to(p.resolve())
                              for p in (capture, inputs, source_model, target_model)):
        raise ValueError('require a new report outside source inputs and models')
    state = json.loads((capture / 'STATE.json').read_text())
    if (state['status'] != 'captured_pending_independent_review'
            or not state['service_restored'] or state['cleanup_errors']
            or state.get('kernel_final', {}).get('new_GPU_error', True)):
        raise ValueError('unclean capture')
    if sha(inputs / 'INPUTS.json') != state['inputs_sha256']:
        raise ValueError('changed input manifest')
    manifest = json.loads((inputs / 'INPUTS.json').read_text())
    held = [d['id'] for d in manifest['rows'] if d['split'] == 'holdout']
    report = {'scope': 'two independent layer0 attention HC projections; not connected HC or model quality',
              'fit_phase': 'decode', 'selection': 'none', 'gpu_used': False,
              'holdout_used_for_optimization': False, 'projections': {}}
    references, saved_packs = {}, {}
    for projection in ('up', 'down'):
        baseline = results / ('gptq-' + projection)
        refined = results / ('refined-' + projection)
        base = json.loads((baseline / 'RESULT.json').read_text())
        comp = json.loads((refined / 'RESULT.json').read_text())
        if comp['baseline_result_sha256'] != sha(baseline / 'RESULT.json'):
            raise ValueError('changed baseline result')
        key = base['tensor']
        w, identity = load_weight(source_model, key, 'BF16')
        _, target = load_weight(target_model, key, 'F16')
        if identity != comp['source_identity'] or target != comp['target_identity']:
            raise ValueError('changed source or target identity')
        if target != state['target_hc_identities'][projection]:
            raise ValueError('target no longer matches capture')
        receipt = base['receipt']
        if (receipt['phase'] != 'decode' or receipt['holdout_document_ids'] != held
                or receipt['capture_runtime_sha256'] != state['collector']['binary_sha256']):
            raise ValueError('capture/statistics mismatch')
        w = w.astype(np.float64)
        paths = {
            'legacy_f16_rtn': (baseline / 'legacy-f16-rtn-control.safetensors', base['legacy_control_sha256']),
            'bf16_rtn': (baseline / 'bf16-rtn-control.safetensors', base['bf16_control_sha256']),
            'gptq': (baseline / 'candidate.safetensors', base['candidate_sha256']),
            **{n: (refined / (n + '.safetensors'), digest) for n, digest in comp['artifacts'].items()},
        }
        packs = {}
        for name, (path, digest) in paths.items():
            if sha(path) != digest:
                raise ValueError('changed artifact: ' + str(path))
            tensors = load_file(str(path))
            if set(tensors) != {key} or unpack(tensors[key]).shape != w.shape:
                raise ValueError('unexpected artifact layout')
            packs[name] = tensors[key]
        references[projection] = w
        saved_packs[projection] = packs
        item = {'tensor': key, 'phases': {}, 'fit_metrics': comp['metrics'],
                'refinement_seconds': {n: v['seconds'] for n, v in comp['runs'].items()},
                'source_result_sha256': sha(refined / 'RESULT.json')}
        for phase in ('decode', 'prefill'):
            values, records = read_windows(capture, manifest, phase, projection)
            if phase == 'decode' and records != receipt['input_capture_records']:
                raise ValueError('changed fitted inputs')
            documents = []
            for doc in held:
                metrics = {name: errors(values[doc], w, pack) for name, pack in packs.items()}
                control = metrics['legacy_f16_rtn']['output_sse']
                if control <= 0:
                    raise ValueError('zero control error; ratio undefined')
                for metric in metrics.values():
                    metric['sse_ratio_to_legacy_rtn'] = metric['output_sse'] / control
                documents.append({'id': doc, 'rows': len(values[doc]), 'metrics': metrics})
            summary = {}
            for name in packs:
                ratios = [d['metrics'][name]['sse_ratio_to_legacy_rtn'] for d in documents]
                summary[name] = {'document_mean_sse_ratio_to_legacy_rtn': float(np.mean(ratios)),
                                 'document_max_sse_ratio_to_legacy_rtn': float(np.max(ratios)),
                                 'pooled_sse_ratio_to_legacy_rtn': sum(d['metrics'][name]['output_sse'] for d in documents)
                                 / sum(d['metrics']['legacy_f16_rtn']['output_sse'] for d in documents)}
            item['phases'][phase] = {'documents': documents, 'summary': summary,
                                     'input_records': records}
        report['projections'][projection] = item
    connected = {'scope': 'fixed B10 normalized inputs, BF16 down -> SiLU(down / 4) -> BF16 up reference; '
                         'ideal FP64 weight-only calculation, no activation Q8 or native GPU rounding',
                 'phases': {}}
    decoded = {p: {n: unpack(b) for n, b in packs.items()} for p, packs in saved_packs.items()}
    for phase in ('decode', 'prefill'):
        values, _ = read_windows(capture, manifest, phase, 'down')
        documents = []
        for doc in held:
            reference = silu_quarter(values[doc] @ references['down'].T) @ references['up'].T
            metrics = {n: connected_errors(values[doc], decoded['down'][n], decoded['up'][n],
                                           reference)
                       for n in decoded['down']}
            control = metrics['legacy_f16_rtn']['output_sse']
            if control <= 0:
                raise ValueError('zero connected control error')
            for metric in metrics.values():
                metric['sse_ratio_to_legacy_rtn'] = metric['output_sse'] / control
            documents.append({'id': doc, 'rows': len(values[doc]), 'metrics': metrics})
        connected['phases'][phase] = {'documents': documents, 'summary': {n: {
            'document_mean_sse_ratio_to_legacy_rtn': float(np.mean([
                d['metrics'][n]['sse_ratio_to_legacy_rtn'] for d in documents])),
            'document_max_sse_ratio_to_legacy_rtn': float(np.max([
                d['metrics'][n]['sse_ratio_to_legacy_rtn'] for d in documents])),
            'pooled_sse_ratio_to_legacy_rtn': sum(d['metrics'][n]['output_sse'] for d in documents)
            / sum(d['metrics']['legacy_f16_rtn']['output_sse'] for d in documents),
        } for n in decoded['down']}}
    report['ideal_connected_gate'] = connected
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    return {p: v['phases']['decode']['summary'] for p, v in report['projections'].items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'inputs', 'source-model', 'target-model', 'results', 'output'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(score(**vars(parser.parse_args())), indent=2))
