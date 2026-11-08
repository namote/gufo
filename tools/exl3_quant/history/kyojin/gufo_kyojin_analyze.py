"""Compare saved same-pack serving rates with explicit configuration limitations."""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(base, experiment):
    current = base / 'gufo-kyojin-pack-neutral-nix101'
    previous = base / 'kyojin-local-ssd-rocm101'
    report = json.loads((current / 'receipt/run/SAME-PACK.json').read_text())
    kyojin = json.loads((previous / 'receipt/run/KYOJIN.json').read_text())
    window = json.loads((current / 'WINDOW.json').read_text())
    assert report['status'] == 'COMPLETE' and window['health_PASS']
    assert len(report['runs']) == 8 and all(r['text_equal'] for r in report['pairs'])
    result = {'schema': 1, 'status': 'MEASURED_SAME_PACK_TEXT_EQUIVALENCE_PASS',
              'receipt': str(current), 'saved_kyojin_receipt': str(previous),
              'model_path': report['model']['root'], 'binary_sha256': report['binary_sha256'],
              'baseline_repeated': False, 'health_PASS': True,
              'prior_invalid_comparison_receipt': str(base / 'gufo-kyojin-pack-nix101'),
              'prior_stop_reason': 'Omitted penalties inherited Gufo presence1.5; stopped before MTP requests, corrected Gufo only rerun.',
              'limitations': report['limitations'], 'rows': []}
    for pair in report['pairs']:
        case = pair['case']
        a = next(r for r in report['runs'] if r['name'] == case + '-off')
        b = next(r for r in report['runs'] if r['name'] == case + '-mtp')
        ka = next(r for r in kyojin['runs'] if r['name'] == case + '-ar')
        kb = next(r for r in kyojin['runs'] if r['name'] == case + '-mtp')
        bodies = [json.loads((current / f'receipt/run/{case}-{mode}-events.json').read_text())['request'] for mode in ('off','mtp')]
        assert bodies[0] == bodies[1]
        for body in bodies:
            assert body['temperature'] == 0 and body['repeat_penalty'] == 1
            assert body['presence_penalty'] == body['frequency_penalty'] == 0
        assert a['usage']['prompt_tokens'] == b['usage']['prompt_tokens'] == ka['usage']['prompt_tokens'] == kb['usage']['prompt_tokens']
        assert all(r['usage']['completion_tokens'] == 256 for r in (a,b,ka,kb))
        row = {'case': case, 'prompt_tokens': a['usage']['prompt_tokens'], 'completion_tokens': 256,
               'gufo_ar_client_tps': a['client_tps'], 'kyojin_ar_client_tps': ka['client_decode_tps'],
               'gufo_mtp_client_tps': b['client_tps'], 'kyojin_mtp_client_tps': kb['client_decode_tps'],
               'ar_relative_percent': 100*(a['client_tps']/ka['client_decode_tps']-1),
               'mtp_relative_percent': 100*(b['client_tps']/kb['client_decode_tps']-1),
               'gufo_drafted': b['usage']['draft_tokens'], 'gufo_accepted': b['usage']['draft_tokens_accepted'],
               'gufo_cycles': b['usage']['gufo']['draft_rounds'],
               'gufo_tokens_per_cycle': 256/b['usage']['gufo']['draft_rounds'],
               'kyojin_drafted': kb['timings']['accepted_draft_tokens']+kb['timings']['rejected_draft_tokens'],
               'kyojin_accepted': kb['timings']['accepted_draft_tokens'],
               'gufo_paired_text_equal': a['text'] == b['text'],
               'kyojin_paired_text_equal': ka['text'] == kb['text'],
               'cross_engine_ar_text_equal': a['text'] == ka['text'],
               'cross_engine_mtp_text_equal': b['text'] == kb['text']}
        result['rows'].append(row)
    result['interpretation'] = ('Same-pack observed MTP rates are close (prose+0.08%,chat-5.06%,code-0.89%); no new kernel speedup, pure-engine isolation, statistical parity or cross-engine quality parity claim. Effective HC/head/KV/runtime and generated text differ. Retain native5.05 selection.')
    sources = [current/'WINDOW.json', current/'receipt/run/SAME-PACK.json', previous/'receipt/run/KYOJIN.json',
               base/'fn-native-prose-profile-nix101/PROFILE-ANALYSIS.json', experiment/'gufo_kyojin_pack.py',
               experiment/'native_prose_profile.py', experiment/'native_prose_analyze.py']
    result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('receipts',type=Path);parser.add_argument('experiment',type=Path);args=parser.parse_args()
    print(json.dumps(analyze(args.receipts,args.experiment),indent=2))
