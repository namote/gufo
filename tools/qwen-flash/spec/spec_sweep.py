#!/usr/bin/env python3
"""W9 M1 policy sweep over a spec_sim trace: TierRule grid and P2 settings chosen on non-K prompts, reported on the
held-out K prompts (prefix k_), plus per-category rates. Usage (eval image, repository at /src):
  python3 -I tools/qwen-flash/spec/spec_sweep.py TRACE.jsonl > sweep.json"""
import itertools, json, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import spec_sim as s

header, cases = s.load_trace(__import__('pathlib').Path(sys.argv[1]))
stops = set(header['stop_tokens'])
heldout_ids = {c['case'] for c in cases if c['case'].startswith('k_')}
train = [c for c in cases if c['case'] not in heldout_ids]
test = [c for c in cases if c['case'] in heldout_ids]


def rate(cs, factory):
    tok = ms = 0.0
    for c in cs:
        r = s.simulate(c, stops, factory(), header['max_tokens'])
        tok += r['tokens']; ms += r['ms']
    return 1000.0 * tok / ms


def by_cat(cs, factory):
    out = {}
    for c in cs:
        r = s.simulate(c, stops, factory(), header['max_tokens'])
        a = out.setdefault(c['category'], [0.0, 0.0]); a[0] += r['tokens']; a[1] += r['ms']
    return {k: 1000.0 * v[0] / v[1] for k, v in out.items()}


base_train, base_test = rate(train, s.Policy), rate(test, s.Policy)
results = []
for thf, thv, thfd, thvd, dsplit in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7], [0.1, 0.2, 0.3, 0.4],
                                                      [0.6, 0.8, 0.95], [0.4, 0.6, 0.85], [3, 4, 7]):
    if thv > thf or thvd > thfd:
        continue
    f = lambda t=(thf, thv, thfd, thvd, dsplit): s.TierPolicy(*t)
    results.append((rate(train, f) / base_train, (thf, thv, thfd, thvd, dsplit)))
results.sort(reverse=True)
best = results[0][1]
fb = lambda: s.TierPolicy(*best)
fk = lambda: s.TierPolicy()
p2 = []
for decay, prior in itertools.product([0.9, 0.95, 0.98, 0.995], [0.5, 2.0, 8.0]):
    f = lambda d=decay, p=prior: s.CostConfidencePolicy(d, p)
    p2.append((rate(train, f) / base_train, (decay, prior)))
p2.sort(reverse=True)
fp2 = lambda: s.CostConfidencePolicy(*p2[0][1])
out = {
    'train_prompts': len(train), 'test_prompts': len(test),
    'P0_tok_s': {'train': base_train, 'test': base_test},
    'tier_top5_train': results[:5],
    'tier_best': {'params': best, 'train': results[0][0], 'test': rate(test, fb) / base_test},
    'tier_kyojin_default': {'train': rate(train, fk) / base_train, 'test': rate(test, fk) / base_test},
    'p2_top3_train': p2[:3],
    'p2_best': {'params': p2[0][1], 'test': rate(test, fp2) / base_test},
    'oracle': {'train': rate(train, s.OraclePolicy) / base_train, 'test': rate(test, s.OraclePolicy) / base_test},
    'by_category_P0': by_cat(cases, s.Policy),
    'by_category_tier_best': by_cat(cases, fb),
    'by_category_oracle': by_cat(cases, s.OraclePolicy),
}
print(json.dumps(out, indent=1))
