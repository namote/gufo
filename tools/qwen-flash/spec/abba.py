#!/usr/bin/env python3
"""Per-request/phase timing verdict for W9 ABBA windows (CLAUDE.md: 5 % and 10 ms, never averaged across requests).

Inputs are serving benchmark reports (tools/gufo/serving_bench.py --suite, C=1) of the same suite: every MAIN report and
every candidate report of an alternated sequence (for example MAIN a1, a2 and I1 b1, b2). For each prompt and phase
(server prefill_ms, decode_ms) the samples of all repetitions of an arm are pooled. With d = median(candidate) -
median(MAIN) and the threshold t = max(10 ms, 5 % of the MAIN median):
  pass          |d| <= t
  slower/faster |d| > t and every candidate sample is beyond every MAIN sample in that direction by more than t
  inconclusive  |d| > t but the sample ranges are not separated by t (noise; stays visible, never averaged away)
It also checks that each prompt's completion sha256 and draft counts agree within each arm and that completions agree
across arms (greedy output identity). Decode tok/s per arm and the paired ratio are reported for information only.

Usage: abba.py --main A1.json A2.json --candidate B1.json B2.json [--out OUT.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

PHASES = ("prefill_ms", "decode_ms")


def samples(paths: list) -> dict:
    out: dict = {}
    for path in paths:
        for s in json.loads(Path(path).read_text())["results"]["c1"]["samples"]:
            out.setdefault(s["case_id"], []).append(s)
    return out


def verdict(main: list, cand: list) -> dict:
    mm, cm = statistics.median(main), statistics.median(cand)
    d, t = cm - mm, max(10.0, 0.05 * mm)
    if abs(d) <= t:
        state = "pass"
    elif d > 0 and min(cand) - max(main) > t:
        state = "slower"
    elif d < 0 and min(main) - max(cand) > t:
        state = "faster"
    else:
        state = "inconclusive"
    return {"state": state, "main_median": mm, "candidate_median": cm, "delta_ms": d, "threshold_ms": t,
            "main": main, "candidate": cand}


def consistent(rows: list) -> bool:
    keys = {(r["completion_sha256"], r["draft_tokens"], r["draft_accepted_tokens"]) for r in rows}
    return len(keys) == 1


def run(main_paths: list, cand_paths: list) -> dict:
    main, cand = samples(main_paths), samples(cand_paths)
    if set(main) != set(cand):
        raise ValueError(f"arms differ in prompts: {sorted(set(main) ^ set(cand))[:5]}")
    cases, counts = {}, {"pass": 0, "slower": 0, "faster": 0, "inconclusive": 0}
    identity = {"same_completion": 0, "arm_inconsistent": []}
    tok_s = {"main": [0, 0.0], "candidate": [0, 0.0]}
    for case in sorted(main):
        m, c = main[case], cand[case]
        for arm, rows in (("main", m), ("candidate", c)):
            if not consistent(rows):
                identity["arm_inconsistent"].append(f"{arm}:{case}")
            tok_s[arm][0] += sum(r["completion_tokens"] for r in rows)
            tok_s[arm][1] += sum(r["decode_ms"] for r in rows)
        identity["same_completion"] += m[0]["completion_sha256"] == c[0]["completion_sha256"]
        row = {"category": m[0].get("category")}
        for phase in PHASES:
            v = verdict([r[phase] for r in m], [r[phase] for r in c])
            row[phase] = v
            counts[v["state"]] += 1
        cases[case] = row
    rate = {arm: 1000.0 * n / ms for arm, (n, ms) in tok_s.items()}
    return {"prompts": len(cases), "phase_verdicts": counts, "identity": identity,
            "decode_tok_s": rate, "decode_tok_s_ratio": rate["candidate"] / rate["main"],
            "flags": {k: {p: v[p]["state"] for p in PHASES if v[p]["state"] != "pass"}
                      for k, v in cases.items() if any(v[p]["state"] != "pass" for p in PHASES)},
            "cases": cases}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", type=Path, nargs="+", required=True)
    ap.add_argument("--candidate", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    result = run(a.main, a.candidate)
    text = json.dumps(result, indent=1) + "\n"
    if a.out:
        a.out.write_text(text)
    summary = {k: result[k] for k in ("prompts", "phase_verdicts", "decode_tok_s", "decode_tok_s_ratio", "flags")}
    summary["identity"] = {"same_completion": result["identity"]["same_completion"],
                           "arm_inconsistent": result["identity"]["arm_inconsistent"]}
    sys.stdout.write(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
