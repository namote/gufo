"""Summarize the one saved local Kyojin run, including failed output equality."""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(receipt, baseline):
    data = json.loads((receipt / 'receipt/run/KYOJIN.json').read_text())
    health = json.loads((receipt / 'WINDOW.json').read_text())
    saved = json.loads(baseline.read_text())
    rows = {r['name']: r for r in data['runs']}
    assert len(rows) == len(data['runs']) == 12
    measured = [r for r in data['runs'] if r['measured']]
    assert all(r['timings']['cache_n'] == 0 for r in measured)
    result = {'schema': 1, 'status': 'MEASURED_OUTPUT_EQUIVALENCE_FAILED',
              'health_PASS': health['health_PASS'], 'receipt': str(receipt),
              'source_revision': data['inputs']['source_revision'],
              'ready_seconds': data['ready_seconds'], 'runtime_paths': data['runtime_paths'],
              'prefill': {}, 'decode': [], 'limitations': data['limitations'],
              'baseline_source': str(baseline), 'baseline_repeated': False,
              'baseline_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(),
              'actual_measured_draft_tokens': data['measured_mtp_draft_tokens']}
    for size in (2048, 8192):
        row = rows[f'pp{size}']
        assert row['usage']['prompt_tokens'] == size
        result['prefill'][str(size)] = {'tokens_per_second': row['timings']['prompt_per_second'],
                                       'prompt_ms': row['timings']['prompt_ms'],
                                       'warmup_prompt_ms': rows[f'pp{size}-warm']['timings']['prompt_ms'],
                                       'scope': 'real text, AR prefill, server phase timer; distinct from Gufo fixed-pattern PP'}
    for case in ('k_prose_0', 'k_chat_0', 'k_code_0'):
        ar, mtp = rows[case + '-ar'], rows[case + '-mtp']
        assert len(ar['token_ids']) == len(mtp['token_ids']) == 256
        a = json.loads((receipt / f'receipt/run/{case}-ar-events.json').read_text())['request']
        b = json.loads((receipt / f'receipt/run/{case}-mtp-events.json').read_text())['request']
        assert {k: v for k, v in a.items() if k != 'speculative'} == {k: v for k, v in b.items() if k != 'speculative'}
        first = next((i for i, (x, y) in enumerate(zip(ar['token_ids'], mtp['token_ids'])) if x != y), None)
        old = next(x for x in saved['requests'] if x['case_id'] == case)
        assert ar['timings']['speculative'] is False and mtp['timings']['speculative'] is True
        result['decode'].append({'case': case, 'prompt_tokens': ar['usage']['prompt_tokens'],
                                 'completion_tokens': 256, 'ar_client_tps': ar['client_decode_tps'],
                                 'mtp_client_tps': mtp['client_decode_tps'],
                                 'ar_server_tps': ar['timings']['predicted_per_second'],
                                 'mtp_server_tps': mtp['timings']['predicted_per_second'],
                                 'accepted': mtp['timings'].get('accepted_draft_tokens', 0),
                                 'rejected': mtp['timings'].get('rejected_draft_tokens', 0),
                                 'token_ids_equal': ar['token_ids'] == mtp['token_ids'],
                                 'text_equal': ar['text'] == mtp['text'],
                                 'first_divergence_zero_based': first,
                                 'saved_gufo_mtp_client_tps': old['client_decode_tokens_per_second'],
                                 'relative_to_saved_gufo_percent': 100 * (mtp['client_decode_tps'] / old['client_decode_tokens_per_second'] - 1),
                                 'comparison_scope': 'historical Gufo5.05/IQ4 PLE, KVarN, different runtime/capacity; not current native parity'})
    assert result['health_PASS'] and result['actual_measured_draft_tokens'] > 0
    assert all(not r['token_ids_equal'] and not r['text_equal'] for r in result['decode'])
    result['interpretation'] = ('All requests returned tokens and timing, but AR/MTP equality failed on all3prompts. '
                                'Cause is not isolated; no quality-loss magnitude, deterministic-regression, '
                                'pure engine speedup or production acceptance claim. Retain original Gufo selection.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    parser.add_argument('baseline', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.receipt, args.baseline), indent=2))
