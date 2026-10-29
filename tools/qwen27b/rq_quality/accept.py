#!/usr/bin/env python3
"""DFlash2 text acceptance from gufo serving benchmark artifacts (tools/gufo/serving_bench.py --suite).

Reads the C=1 samples of a corpus report (`results.c1.samples`: case_id, draft_tokens,
draft_accepted_tokens, completion_tokens, decode_ms) and aggregates them per prompt (case).
Greedy decoding with prompt caching off is deterministic, so repetitions of a case must agree on
draft counts; `prompts()` checks that and keeps the first repetition's counts, while decode time
and tokens are summed over every repetition for the effective rate.

Statistics, all with a prompt-level percentile bootstrap (2000 resamples, prompts resampled as
units, ratio estimators):
  acceptance            sum(accepted) / sum(drafted)
  accepted_per_token    sum(accepted) / sum(completion tokens)
  decode_tok_s          sum(completion tokens) / sum(decode ms) * 1000
`summary` reports one run, optionally its speedup over an AR run of the same prompts (paired);
`compare` reports the paired difference (acceptance, accepted_per_token) and ratio (decode_tok_s)
of two runs on the same prompts, e.g. rotated against plain Q4 (R2 (3), E5).
`check-overlap` records which prompts share a 40-character fragment with given text files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

DEFAULT_BOOTSTRAP = 2000
DEFAULT_SEED = 20261029


def prompts(report: dict, level: str = "c1") -> dict[str, dict]:
    """Per-case counts of one corpus report."""
    samples = report["results"][level]["samples"]
    out: dict[str, dict] = {}
    for s in samples:
        case = s.get("case_id")
        if case is None:
            raise ValueError("sample without case_id: not a --suite report")
        entry = out.setdefault(case, {"drafted": s["draft_tokens"], "accepted": s["draft_accepted_tokens"],
                                      "first_completion_tokens": s["completion_tokens"],
                                      "completion_tokens": 0, "decode_ms": 0.0, "repetitions": 0,
                                      "completion_sha256": s.get("completion_sha256")})
        if (s["draft_tokens"], s["draft_accepted_tokens"]) != (entry["drafted"], entry["accepted"]):
            raise ValueError(f"case {case}: repetitions disagree on draft counts "
                             "(greedy without prompt cache must be deterministic)")
        if s.get("decode_ms") is None:
            raise ValueError(f"case {case}: sample has no server decode_ms")
        entry["completion_tokens"] += s["completion_tokens"]
        entry["decode_ms"] += s["decode_ms"]
        entry["repetitions"] += 1
    return out


def arrays(cases: dict[str, dict], order: list[str]) -> dict[str, np.ndarray]:
    get = lambda k: np.array([cases[c][k] for c in order], dtype=np.float64)
    return {"drafted": get("drafted"), "accepted": get("accepted"),
            "first_tokens": get("first_completion_tokens"),
            "tokens": get("completion_tokens"), "decode_ms": get("decode_ms")}


def ratios(a: dict[str, np.ndarray], idx: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Ratio statistics over the prompts `idx` (rows of resamples, or all prompts)."""
    pick = (lambda v: v) if idx is None else (lambda v: v[idx])
    s = lambda k: pick(a[k]).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"acceptance": s("accepted") / s("drafted"),
                "accepted_per_token": s("accepted") / s("first_tokens"),
                "decode_tok_s": s("tokens") / s("decode_ms") * 1000.0}


def interval(values: np.ndarray) -> list[float]:
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def draws(n: int, resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def common_order(a: dict, b: dict) -> list[str]:
    if set(a) != set(b):
        raise ValueError(f"runs differ in prompts: {sorted(set(a) ^ set(b))[:5]}")
    return sorted(a)


def summary(run: dict, ar: dict | None = None, resamples: int = DEFAULT_BOOTSTRAP,
            seed: int = DEFAULT_SEED) -> dict:
    cases = prompts(run)
    order = sorted(cases)
    a = arrays(cases, order)
    idx = draws(len(order), resamples, seed)
    point, boot = ratios(a), ratios(a, idx)
    out = {"prompts": len(order), "repetitions": sorted({cases[c]["repetitions"] for c in order}),
           "prompts_with_drafts": int((a["drafted"] > 0).sum()), "bootstrap": resamples, "seed": seed}
    for k in point:
        out[k] = {"value": float(point[k]), "ci": interval(boot[k])}
    if ar is not None:
        ar_cases = prompts(ar)
        common_order(cases, ar_cases)
        b = arrays(ar_cases, order)
        speed = ratios(a)["decode_tok_s"] / ratios(b)["decode_tok_s"]
        boot_speed = ratios(a, idx)["decode_tok_s"] / ratios(b, idx)["decode_tok_s"]
        out["ar_decode_tok_s"] = {"value": float(ratios(b)["decode_tok_s"]),
                                  "ci": interval(ratios(b, idx)["decode_tok_s"])}
        out["speedup_over_ar"] = {"value": float(speed), "ci": interval(boot_speed)}
    return out


def compare(first: dict, second: dict, resamples: int = DEFAULT_BOOTSTRAP,
            seed: int = DEFAULT_SEED) -> dict:
    """Paired first - second (acceptance, accepted_per_token) and first / second (decode_tok_s)."""
    ca, cb = prompts(first), prompts(second)
    order = common_order(ca, cb)
    a, b = arrays(ca, order), arrays(cb, order)
    idx = draws(len(order), resamples, seed)
    pa, pb, ba, bb = ratios(a), ratios(b), ratios(a, idx), ratios(b, idx)
    out = {"prompts": len(order), "bootstrap": resamples, "seed": seed,
           "identical_completions": int(sum(ca[c]["completion_sha256"] == cb[c]["completion_sha256"]
                                            for c in order))}
    for k in ("acceptance", "accepted_per_token"):
        if a["drafted"].sum() == 0 or b["drafted"].sum() == 0:
            out[k + "_difference"] = None  # an AR run has no drafts
            continue
        ci = interval(ba[k] - bb[k])
        out[k + "_difference"] = {"value": float(pa[k] - pb[k]), "ci": ci,
                                  "resolved": bool(ci[0] > 0 or ci[1] < 0)}
    ci = interval(ba["decode_tok_s"] / bb["decode_tok_s"])
    out["decode_tok_s_ratio"] = {"value": float(pa["decode_tok_s"] / pb["decode_tok_s"]), "ci": ci,
                                 "resolved": bool(ci[0] > 1 or ci[1] < 1)}
    return out


def check_overlap(prompt_doc: dict, texts: dict[str, str], width: int = 40, step: int = 20) -> dict:
    """Prompt ids with a `width`-character fragment (every `step`) found in each text."""
    result = {"method": f"{width}-character fragments every {step} characters, exact substring", "against": {}}
    for name, text in texts.items():
        hits = {}
        for p in prompt_doc["prompts"]:
            t = p["text"]
            frags = [t[i:i + width] for i in range(0, max(1, len(t) - width + 1), step)] if len(t) >= width else [t]
            n = sum(f in text for f in frags)
            if n:
                hits[p["id"]] = n
        result["against"][name] = {"sha256": hashlib.sha256(text.encode()).hexdigest(), "hits": hits}
    return result


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summary")
    s.add_argument("run", type=Path)
    s.add_argument("--ar", type=Path, help="AR report of the same prompts (speedup)")
    c = sub.add_parser("compare")
    c.add_argument("first", type=Path)
    c.add_argument("second", type=Path)
    for p in (s, c):
        p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)
        p.add_argument("--out", type=Path)
    o = sub.add_parser("check-overlap")
    o.add_argument("prompts", type=Path)
    o.add_argument("texts", nargs="+", help="name=path of a text file (decoded corpus)")
    o.add_argument("--write", action="store_true", help="store the result as the prompt file's manifest.overlap")
    a = ap.parse_args(argv)
    if a.command == "check-overlap":
        doc = load(a.prompts)
        texts = {}
        for item in a.texts:
            name, _, path = item.partition("=")
            raw = Path(path).read_text(errors="replace")
            # A .jsonl corpus (R0 windows_text.jsonl) is matched on its decoded "text" fields.
            texts[name] = ("\n".join(json.loads(line)["text"] for line in raw.splitlines() if line.strip())
                           if path.endswith(".jsonl") else raw)
        result = check_overlap(doc, texts)
        print(json.dumps({k: v["hits"] for k, v in result["against"].items()}, ensure_ascii=False))
        if a.write:
            doc.setdefault("manifest", {})["overlap"] = result
            a.prompts.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        return 0
    if a.bootstrap < 100:
        ap.error("--bootstrap must be >= 100")
    if a.command == "summary":
        result = summary(load(a.run), load(a.ar) if a.ar else None, a.bootstrap, a.seed)
    else:
        result = compare(load(a.first), load(a.second), a.bootstrap, a.seed)
    text = json.dumps(result, indent=1)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
