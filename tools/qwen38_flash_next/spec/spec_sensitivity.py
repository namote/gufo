#!/usr/bin/env python3
"""W9 M1 sensitivity of the TierRule gain to modeled costs: a computed-then-dropped draft (1-6 ms) and an extra cost per
draft for producing its probability (0 or 0.3 ms). Usage: python3 -I tools/qwen38_flash_next/spec/spec_sensitivity.py TRACE.jsonl"""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import spec_sim as s
h, cases = s.load_trace(pathlib.Path(sys.argv[1])); st = set(h['stop_tokens'])
def rate(f, drop, extra=0.0):
    tok = ms = 0.0
    for c in cases:
        r = s.simulate(c, st, f(), h['max_tokens'], drop); tok += r['tokens']; ms += r['ms'] + extra * r['drafted']
    return 1000 * tok / ms
best = lambda: s.TierPolicy(0.7, 0.4, 0.8, 0.4, 3)
out = {}
for drop in (1.0, 2.0, 4.0, 6.0):
    for extra in (0.0, 0.3):  # extra ms per draft for producing p (candidate transfer)
        base = rate(s.Policy, drop)
        out[f'drop{drop}_extra{extra}'] = {'P0': base, 'tier_best_vs_P0': rate(best, drop, extra) / base,
                                           'tier_kyojin_vs_P0': rate(s.TierPolicy, drop, extra) / base}
print(json.dumps(out, indent=1))
