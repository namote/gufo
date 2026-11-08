#!/usr/bin/env python3
"""W9 speculative-decoding statistics (plan/archive/rotated-quant/SPEC-DECODE.md 4.1).

Inputs are the artifacts of the ai.tools rq_serving driver: a serving benchmark report (bench-<i>.json,
tools/gufo/serving_bench.py --suite, C=1) and, when the plan entry sets record_responses, the proxy record
responses-<i>.jsonl of the same bench. Per prompt the bench report gives draft/accepted counts, completion tokens,
decode time and the completion sha256 (tools/qwen27b/rq_quality/accept.py `prompts`, reused unchanged). The record
adds the server timings the bench report drops: draft_rounds (verification cycles), lookup_n and lookup_n_accepted.
A record is joined to its case by the request's user text; every record of a case (warmup and repetitions) must agree
on these counts, since greedy decoding without a prompt cache is deterministic.

Statistics per category and overall, each with a prompt-level percentile bootstrap (accept.py resampling, ratio
estimators):
  acceptance              sum(accepted) / sum(drafted)
  accepted_per_token      sum(accepted) / sum(completion tokens)
  tokens_per_cycle        (sum(accepted) + sum(rounds)) / sum(rounds)        [needs a record]
  drafts_per_cycle        sum(drafted) / sum(rounds)                         [needs a record]
  ar_token_share          1 - (sum(accepted) + sum(rounds)) / sum(tokens)    [tokens from non-speculative steps]
  lookup_share            sum(lookup_n) / sum(drafted)                       [needs a record]
  lookup_acceptance       sum(lookup_n_accepted) / sum(lookup_n)             [needs a record]
  decode_tok_s            sum(tokens) / sum(decode ms) * 1000
With an AR report of the same prompts: speedup_over_ar (paired) and the number of byte-identical completions.

Usage:
  spec_stats.py summary BENCH.json [--responses R.jsonl] [--ar AR_BENCH.json] [--out OUT.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qwen27b" / "rq_quality"))
import accept  # noqa: E402  (tools/qwen27b/rq_quality/accept.py, unchanged)

RECORD_KEYS = ("draft_rounds", "draft_n", "draft_n_accepted", "lookup_n", "lookup_n_accepted", "prompt_n")


def stream_timings(response: str) -> dict | None:
    """The last `timings` object of a server-sent-event stream (or of a plain JSON body)."""
    found = None
    for line in response.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and isinstance(event.get("timings"), dict):
            found = event["timings"]
    return found


def request_text(body: str | None) -> str | None:
    if not body:
        return None
    try:
        messages = json.loads(body).get("messages") or []
    except (json.JSONDecodeError, AttributeError):
        return None
    texts = [m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    return texts[-1] if texts and isinstance(texts[-1], str) else None


def record_counts(path: Path, texts: dict[str, str]) -> dict[str, dict]:
    """Per-case server counts from a proxy record; `texts` maps prompt text to case id."""
    out: dict[str, dict] = {}
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("method") != "POST" or not str(rec.get("path", "")).endswith("/chat/completions"):
            continue
        case = texts.get(request_text(rec.get("request")))
        if case is None:
            continue  # not a suite prompt (for example a model probe)
        records.append((case, rec, json.loads(rec["request"]).get("max_tokens")))
    # serving_bench warms up with a shorter budget (16 tokens); keep the measured requests' budget only.
    budgets = [b for _, _, b in records]
    budget = max(set(budgets), key=budgets.count) if budgets else None
    for case, rec, b in records:
        if b != budget:
            out.setdefault("_skipped_other_budget", {"records": 0})["records"] += 1
            continue
        timings = stream_timings(rec.get("response", ""))
        if timings is None:
            raise ValueError(f"case {case}: response without timings (status {rec.get('status')})")
        counts = {k: int(timings.get(k, 0)) for k in RECORD_KEYS}
        entry = out.setdefault(case, {**counts, "records": 0})
        if any(entry[k] != counts[k] for k in RECORD_KEYS if k != "prompt_n"):
            raise ValueError(f"case {case}: records disagree on draft counts {entry} vs {counts}")
        entry["records"] += 1
    return out


def load_suite_texts(paths: list[Path]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in paths:
        for p in json.loads(path.read_text(encoding="utf-8"))["prompts"]:
            texts[p["text"]] = p["id"]
    return texts


def stats(cases: dict, counts: dict | None, order: list[str], idx: np.ndarray) -> dict:
    a = accept.arrays(cases, order)
    point, boot = accept.ratios(a), accept.ratios(a, idx)
    out = {k: {"value": float(point[k]), "ci": accept.interval(boot[k])} for k in point}
    if counts is not None:
        get = lambda k: np.array([counts[c][k] for c in order], dtype=np.float64)
        rounds, lookup, lookup_acc = get("draft_rounds"), get("lookup_n"), get("lookup_n_accepted")

        def ratio(fn):
            with np.errstate(divide="ignore", invalid="ignore"):
                value = fn(lambda v: v.sum())
                sample = fn(lambda v: v[idx].sum(axis=-1))
            return {"value": float(value), "ci": accept.interval(np.asarray(sample, dtype=np.float64))}

        acc, tok = a["accepted"], a["first_tokens"]
        out["tokens_per_cycle"] = ratio(lambda s: (s(acc) + s(rounds)) / s(rounds))
        out["drafts_per_cycle"] = ratio(lambda s: s(a["drafted"]) / s(rounds))
        out["ar_token_share"] = ratio(lambda s: 1.0 - (s(acc) + s(rounds)) / s(tok))
        out["lookup_share"] = ratio(lambda s: s(lookup) / s(a["drafted"]))
        out["lookup_acceptance"] = ratio(lambda s: s(lookup_acc) / s(lookup))
        out["cycles"] = int(rounds.sum())
    return out


def summary(report: dict, responses: Path | None, suites: list[Path], ar: dict | None,
            resamples: int = accept.DEFAULT_BOOTSTRAP, seed: int = accept.DEFAULT_SEED) -> dict:
    cases = accept.prompts(report)
    categories = {s["case_id"]: s.get("category") for s in report["results"]["c1"]["samples"]}
    counts = None
    if responses is not None:
        counts = record_counts(responses, load_suite_texts(suites))
        missing = sorted(set(cases) - set(counts))
        if missing:
            raise ValueError(f"record lacks cases {missing[:5]}")
        for c in cases:
            if (counts[c]["draft_n"], counts[c]["draft_n_accepted"]) != (cases[c]["drafted"], cases[c]["accepted"]):
                raise ValueError(f"case {c}: record and bench report disagree on draft counts")
    ar_cases = accept.prompts(ar) if ar is not None else None
    groups = {"all": sorted(cases)}
    for c in sorted(cases):
        groups.setdefault(str(categories.get(c)), []).append(c)
    out = {"prompts": len(cases), "bootstrap": resamples, "seed": seed, "groups": {}}
    if counts is not None:
        out["records_per_case"] = sorted({counts[c]["records"] for c in cases})
        out["skipped_records_other_budget"] = counts.get("_skipped_other_budget", {"records": 0})["records"]
    for name, order in groups.items():
        idx = accept.draws(len(order), resamples, seed)
        g = {"prompts": len(order), **stats(cases, counts, order, idx)}
        if ar_cases is not None:
            accept.common_order({c: cases[c] for c in order}, {c: ar_cases[c] for c in order if c in ar_cases})
            a, b = accept.arrays(cases, order), accept.arrays(ar_cases, order)
            speed = accept.ratios(a)["decode_tok_s"] / accept.ratios(b)["decode_tok_s"]
            boot = accept.ratios(a, idx)["decode_tok_s"] / accept.ratios(b, idx)["decode_tok_s"]
            g["ar_decode_tok_s"] = {"value": float(accept.ratios(b)["decode_tok_s"]),
                                    "ci": accept.interval(accept.ratios(b, idx)["decode_tok_s"])}
            g["speedup_over_ar"] = {"value": float(speed), "ci": accept.interval(boot)}
            g["identical_completions_vs_ar"] = int(sum(cases[c]["completion_sha256"] ==
                                                       ar_cases[c]["completion_sha256"] for c in order))
        out["groups"][name] = g
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summary")
    s.add_argument("bench", type=Path)
    s.add_argument("--responses", type=Path)
    s.add_argument("--suite", type=Path, action="append", default=[],
                   help="suite JSON(s) used by the bench, needed with --responses to join records to cases")
    s.add_argument("--ar", type=Path)
    s.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.responses is not None and not a.suite:
        ap.error("--responses needs --suite")
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    result = summary(load(a.bench), a.responses, a.suite, load(a.ar) if a.ar else None)
    text = json.dumps(result, indent=1) + "\n"
    if a.out:
        a.out.write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
