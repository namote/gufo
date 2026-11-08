#!/usr/bin/env python3
"""Offline replay of Flash-Next greedy C1 speculative decoding over an MTP chain trace (plan W9 M1).

Input: a `qwen38_flash_next_gpu_probe --mtp-trace` file (JSON lines). Its first line gives the stop tokens; then
per prompt one "case" line (committed greedy completion, prompt token count, whether it stopped) followed by one
line per decoded position t with the anchor (completion[t]) and the greedy MTP chain a cycle anchored there would
draft: depth j proposes completion[t + 1 + j] with top-candidate probability p and logit gap.

A cycle's outcome is fully determined by that chain and the committed tokens, so any policy can be replayed. The
P0 policy is a line-by-line port of the production C1 greedy path at gufo c94572b3:
  - serving loop (src/cli/serve/text_generation_scheduler.cpp StepMultiTokenDecode, inference_backend.cpp
    QwenFlashNextTextRunner::DecodeStep): DecodeStep(remaining) per step, the last token (remaining 1) through
    the generic AR path;
  - Session::PrepareDecode / FinishDecode (src/models/qwen38_flash_next/engine.cpp) including prompt lookup;
  - MtpLengthController (mtp_policy.hpp) and MtpCycleCosts (mtp_costs.hpp) in float32;
  - speculative::PromptLookup (src/core/speculative/prompt_lookup.hpp).
It must reproduce the server's per-prompt cycles, drafted, accepted and lookup counts exactly (`validate`).

Other policies change only how many drafts a cycle proposes (they keep the target verification, so greedy output
is unchanged): P1 Kyojin TierRule (tools/qwen/spec_policy.py at Yamz-Labs/kyojin 2132bd24, MIT), P2 a cost-aware
confidence rule calibrated on committed history, P3 the hindsight-optimal length (upper bound). Time is estimated
from the C1 cycle-cost table; this is a model, not a measurement.

Usage:
  spec_sim.py validate TRACE.jsonl --responses R.jsonl --suite SUITE.json [--responses ... --suite ...]
  spec_sim.py compare TRACE.jsonl [--out OUT.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_stats  # noqa: E402

f32 = np.float32
MAX_DRAFTS = 7          # kMaxMtpDraftTokens
MAX_SPECULATIVE = 8     # executor rows: max_draft_tokens + 1

# mtp_costs.hpp kMtpCycleMilliseconds, C1 rows at contexts 0 / 4096 / 32768 (widths 1..8).
C1_COSTS = [
    [38.5, 47.3, 56.1, 65.7, 74.1, 82.2, 89.7, 96.8],
    [39.7, 49.3, 59.8, 69.7, 79.1, 89.1, 98.6, 108.8],
    [42.5, 51.9, 61.5, 71.1, 82.5, 92.5, 103.6, 112.6],
]


def cycle_costs(context: int) -> list:
    """MtpCycleCosts(context, concurrency 1, sampled_subset false) in float32."""
    interval = 0 if context <= 4096 else 1
    if interval == 0:
        fraction = f32(f32(context) / f32(4096.0))
    else:
        fraction = f32(f32(min(context, 262144) - 4096) / f32(28672.0))
    low_row = [f32(x) for x in C1_COSTS[interval]]
    high_row = [f32(x) for x in C1_COSTS[interval + 1]]
    costs = [f32(0)] * 8
    for i in range(8):
        low, high = low_row[i], high_row[i]
        costs[i] = f32(low + f32(fraction * max(f32(high - low), f32(0.0))))
        if fraction > f32(1.0) and i != 0:
            marginal = f32(high - high_row[i - 1])
            costs[i] = max(costs[i], f32(costs[i - 1] + marginal))
    return costs


class LengthController:
    """mtp_policy.hpp MtpLengthController (limit 7, concurrency 1)."""

    def __init__(self, limit: int = MAX_DRAFTS):
        self.limit = max(1, min(limit, MAX_DRAFTS))
        self.reset()

    def reset(self):
        self.s = [f32(1.5)] * MAX_DRAFTS
        self.f = [f32(0.5)] * MAX_DRAFTS
        self.retry = 0
        self.probe_depth = 0
        self.explored = 0
        self.probe_delay = 0
        self.failed = 0

    def probability(self, depth: int):
        if depth < self.explored and not (self.failed & (1 << depth)):
            return f32(1.0)
        return f32(self.s[depth] / f32(self.s[depth] + self.f[depth]))

    def best_length(self, cap: int, context: int) -> int:
        costs = cycle_costs(context)
        survival, expected, best_score, best = f32(1.0), f32(1.0), f32(1.0), 0
        for length in range(1, cap + 1):
            survival = f32(survival * self.probability(length - 1))
            expected = f32(expected + survival)
            cost = f32(costs[length] / costs[0])
            score = f32(expected / cost)
            if score > best_score:
                best_score, best = score, length
        return best

    def choose(self, budget: int, context: int) -> int:
        cap = min(budget, self.limit)
        if cap == 0 or self.retry != 0:
            return 0
        return min(cap, max(1, self.best_length(cap, context), self.probe_depth))

    def observe(self, accepted: int, drafted: int, context: int):
        if drafted == 0:
            return
        drafted = min(drafted, self.limit)
        accepted = min(accepted, drafted)
        observed = accepted + (1 if accepted < drafted else 0)
        for d in range(observed):
            self.s[d] = f32(f32(f32(0.75) * self.s[d]) + f32(1.0 if d < accepted else 0.0))
            self.f[d] = f32(f32(f32(0.75) * self.f[d]) + f32(1.0 if d == accepted else 0.0))
        self.probe_depth = 0
        if accepted == drafted and drafted < self.limit:
            if drafted >= self.explored or self.probe_delay == 0:
                self.probe_depth = min(2 * drafted, self.limit)
                self.probe_delay = 16
            else:
                self.probe_delay -= 1
        self.explored = max(self.explored, observed)
        if accepted < drafted:
            self.failed |= 1 << accepted
        if self.best_length(self.limit, context) == 0:
            self.retry = 16

    def observe_ar(self):
        if self.retry:
            self.retry -= 1


class PromptLookup:
    """prompt_lookup.hpp PromptLookup (greedy: no source boundary at the request start)."""
    KEY, MIN_MATCH, MAX_MATCH, MAX_CANDIDATES, RESUME_MIN, RESUME_WINDOW, RESUME_BACK = 3, 12, 64, 16, 5, 512, 8

    def __init__(self):
        self.clear()

    def clear(self):
        self.head: dict = {}
        self.prev: list = []
        self.last = None
        self.resume = None
        self.request_start = None

    def begin_request(self):
        self.resume = None
        self.request_start = None

    def step(self, position: int) -> bool:
        if self.request_start is not None:
            return True
        self.request_start = position
        return False

    def observe(self, start: int, proposed: int, accepted: int):
        self.resume = None if 2 * accepted < proposed else start + accepted

    def extend(self, committed: list):
        if len(committed) < len(self.prev) or (self.prev and committed[len(self.prev) - 1] != self.last):
            self.head, self.prev, self.resume = {}, [], None
        for end in range(len(self.prev), len(committed)):
            previous = None
            if end >= self.KEY:
                key = tuple(committed[end - self.KEY:end])
                previous = self.head.get(key)
                self.head[key] = end
            self.prev.append(previous)
        if self.prev:
            self.last = committed[len(self.prev) - 1]

    def find(self, committed: list, tail: list, minimum: int):
        context = len(committed) + len(tail)
        available = context if self.request_start is None else context - min(context, self.request_start)
        max_match = min(self.MAX_MATCH, available)
        if max_match < self.KEY:
            return None
        at = lambda i: committed[i] if i < len(committed) else tail[i - len(committed)]
        key = tuple(at(context - self.KEY + i) for i in range(self.KEY))
        best = (0, 0)  # (start, length)
        end = self.head.get(key)
        n = 0
        while n < self.MAX_CANDIDATES and end is not None:
            length = self.KEY
            while length < max_match and end > length and committed[end - length - 1] == at(context - length - 1):
                length += 1
            resumes = (self.resume is not None and end + self.RESUME_BACK >= self.resume
                       and end <= self.resume + self.RESUME_WINDOW)
            allowed = length >= minimum or (resumes and length >= self.RESUME_MIN)
            if end < len(self.prev) and length > best[1] and allowed:
                best = (end, length)
            end = self.prev[end]
            n += 1
        return best if best[1] else None


def load_trace(path: Path):
    header, cases, current = None, [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if "stop_tokens" in rec:
            header = rec
        elif "case" in rec:
            current = {**rec, "chains": []}
            cases.append(current)
        else:
            assert rec["c"] == current["index"] and rec["t"] == len(current["chains"]), rec
            assert rec["a"] == current["completion"][rec["t"]], rec
            current["chains"].append({"small": rec["s"], "drafts": [d[0] for d in rec["d"]],
                                      "p": [d[1] for d in rec["d"]], "gap": [d[2] for d in rec["d"]]})
    if header is None:
        raise ValueError("trace has no stop-token header")
    return header, cases


class Policy:
    """Default draft rule: the production controller chooses the width before drafting."""
    name = "P0"
    computes_dropped_draft = True  # a rule that drops a computed draft still paid its MTP step

    def __init__(self):
        self.ctl = LengthController()

    def width(self, budget: int, context: int) -> int:
        return self.ctl.choose(budget, context)

    def keep_drafting(self, depth: int, chain: dict) -> tuple:
        """After drafting depth `depth` (0-based): (keep this draft, draft another)."""
        return True, True

    def observe(self, accepted: int, drafted: int, context: int, chain: dict, mtp_drafts: int):
        self.ctl.observe(accepted, drafted, context)

    def observe_ar(self):
        self.ctl.observe_ar()


class TierPolicy(Policy):
    """Kyojin TierRule: reach = product of draft probabilities; depths 1-3 continue at 0.6 and keep at 0.3,
    deeper ones need 0.95 / 0.85; depth 1 is always kept. Maximum 7 drafts; no AR backoff."""
    name = "P1-tier"

    def __init__(self, thf=0.6, thv=0.3, thfd=0.95, thvd=0.85, dsplit=3):
        super().__init__()
        self.t = (thf, thv, thfd, thvd, dsplit)

    def width(self, budget, context):
        return min(budget, MAX_DRAFTS)

    def keep_drafting(self, depth, chain):
        thf, thv, thfd, thvd, dsplit = self.t
        reach = math.prod(chain["p"][:depth + 1])
        k = depth + 1
        if depth > 0 and reach < (thv if k <= dsplit else thvd):
            return False, False
        return True, not (reach < (thf if k < dsplit else thfd))

    def observe(self, *args):
        pass

    def observe_ar(self):
        pass


class TierBackoffPolicy(TierPolicy):
    """I1a (f1cf33ac, W9 I1V and the first ABBA): TierRule chains with Kyojin's thresholds, while the production length
    controller still observes every cycle and keeps its AR backoff (a zero Choose means an AR step); its chosen width
    is not used."""
    name = "I1a-tier-backoff"

    def width(self, budget, context):
        chosen = self.ctl.choose(budget, context)
        return min(budget, self.ctl.limit) if chosen else 0

    def observe(self, accepted, drafted, context, chain, mtp_drafts):
        self.ctl.observe(accepted, drafted, context)

    def observe_ar(self):
        self.ctl.observe_ar()


class ConfidencePolicy(TierBackoffPolicy):
    """I1 (engine.cpp PrepareDecode, C1 greedy, after the first ABBA): like I1a, but a request's cycles keep the
    production controller's width and full chain until it has observed acceptance (explored depth 0), and deep drafts
    continue at 0.9 and are kept at 0.4 (shallow 0.6 / 0.3 / 3 stay Kyojin's)."""
    name = "I1-confidence"

    def __init__(self):
        super().__init__(0.6, 0.3, 0.9, 0.4, 3)
        self.first = False

    def width(self, budget, context):
        chosen = self.ctl.choose(budget, context)
        self.first = self.ctl.explored == 0
        if not chosen:
            return 0
        return chosen if self.first else min(budget, self.ctl.limit)

    def keep_drafting(self, depth, chain):
        return (True, True) if self.first else super().keep_drafting(depth, chain)


class CostConfidencePolicy(Policy):
    """P2: draft one token at a time; keep it while the expected tokens per cost improve. The acceptance of a
    draft with probability p is estimated from committed history: decayed accepted/tested counts per p bin (prior
    from the raw p). Depends only on committed history, never on timings."""
    name = "P2-cost-confidence"
    BINS = 10

    def __init__(self, decay=0.98, prior=2.0):
        super().__init__()
        self.decay, self.prior = decay, prior
        self.acc = [0.0] * self.BINS
        self.tested = [0.0] * self.BINS
        self.context = 0
        self.expected = 1.0

    def bin(self, p):
        return min(self.BINS - 1, max(0, int(p * self.BINS)))

    def estimate(self, p):
        b = self.bin(p)
        return (self.acc[b] + self.prior * p) / (self.tested[b] + self.prior)

    def width(self, budget, context):
        self.context = context
        self.reach, self.expected = 1.0, 1.0
        return min(budget, MAX_DRAFTS)

    def keep_drafting(self, depth, chain):
        costs = cycle_costs(self.context)
        reach = self.reach * self.estimate(chain["p"][depth])
        gain = (self.expected + reach) / float(costs[depth + 1])
        if depth > 0 and gain <= self.expected / float(costs[depth]):
            return False, False
        self.reach, self.expected = reach, self.expected + reach
        return True, True

    def observe(self, accepted, drafted, context, chain, mtp_drafts):
        for b in range(self.BINS):
            self.acc[b] *= self.decay
            self.tested[b] *= self.decay
        for d in range(min(mtp_drafts, accepted + 1)):
            b = self.bin(chain["p"][d])
            self.tested[b] += 1.0
            self.acc[b] += 1.0 if d < accepted else 0.0

    def observe_ar(self):
        pass


class OraclePolicy(Policy):
    """P3: hindsight width: exactly the drafts that will be accepted (plus none wasted)."""
    name = "P3-oracle"
    computes_dropped_draft = False

    def __init__(self):
        super().__init__()
        self.limit = 0

    def width(self, budget, context):
        return min(budget, MAX_DRAFTS)

    def keep_drafting(self, depth, chain):
        ok = chain["drafts"][depth] == chain["truth"][depth] if depth < len(chain["truth"]) else False
        return ok, ok

    def observe(self, *args):
        pass

    def observe_ar(self):
        pass


def simulate(case: dict, stops: set, policy: Policy, max_tokens: int, dropped_draft_ms: float = 2.0) -> dict:
    """One request: returns the server counters and the modeled decode time."""
    completion, chains = case["completion"], case["chains"]
    n, prompt = len(completion), case["prompt_tokens"]
    lookup = PromptLookup()
    lookup.begin_request()
    tokens = list(case["prompt"])
    assert len(tokens) == prompt
    pos, remaining = 0, max_tokens
    out = {"cycles": 0, "drafted": 0, "accepted": 0, "lookup": 0, "lookup_accepted": 0, "ms": 0.0,
           "tokens": 0, "ar_steps": 0}
    while remaining > 0 and pos <= n:
        context = prompt + pos
        costs = cycle_costs(context)
        if pos == n:  # the anchor is the stop token (or the budget ended)
            break
        if remaining == 1:  # generic AR path: SelectNext + Advance, no policy update
            out["ms"] += float(costs[0])
            out["tokens"] += 1
            out["ar_steps"] += 1
            pos += 1
            remaining -= 1
            continue
        cap = min(remaining, MAX_SPECULATIVE)
        width = 1 + policy.width(cap - 1, context) if cap > 1 else cap
        anchor = completion[pos]
        if width < 2:
            policy.observe_ar()
            out["ms"] += float(costs[0])
            out["tokens"] += 1
            out["ar_steps"] += 1
            pos += 1
            remaining -= 1
            continue
        committed = tokens + completion[:pos]
        chain = [anchor]
        info = dict(chains[pos])
        info["truth"] = completion[pos + 1:pos + 1 + MAX_DRAFTS]
        lookup_cap = cap if lookup.step(context) else 0
        if lookup_cap:
            lookup.extend(committed)
        copied, copy_start = 0, 0

        def append_lookup(minimum=1):
            nonlocal copied, copy_start
            if len(chain) >= lookup_cap or chain[-1] in stops:
                return False
            m = lookup.find(committed, chain, PromptLookup.MIN_MATCH)
            if m is None:
                return False
            available = min(lookup_cap - len(chain), len(lookup.prev) - m[0])
            if available < minimum:
                return False
            count = 0
            while count < available:
                token = committed[m[0] + count]
                chain.append(token)
                count += 1
                if token in stops:
                    break
            copied, copy_start = count, m[0]
            return count != 0

        if not append_lookup(3):
            depth = 0
            while len(chain) < width:
                keep, more = policy.keep_drafting(depth, info)
                if not keep:
                    out["ms"] += dropped_draft_ms if policy.computes_dropped_draft else 0.0
                    break
                chain.append(info["drafts"][depth])
                depth += 1
                if chain[-1] in stops:
                    break
                if append_lookup():
                    break
                if not more:
                    break
        k = len(chain)
        keep, stop = 1, False
        while keep < k:
            target_index = pos + keep
            pred = completion[target_index] if target_index < n else None
            if pred is None:  # the target predicts the stop token here (keep <= remaining)
                stop = case["stop"]
                break
            if pred != chain[keep]:
                break
            keep += 1
        out["cycles"] += 1
        out["drafted"] += k - 1
        out["accepted"] += keep - 1
        mtp_drafts = k - 1 - copied
        mtp_accepted = min(keep - 1, mtp_drafts)
        out["lookup"] += copied
        out["lookup_accepted"] += keep - 1 - mtp_accepted
        if copied:
            lookup.observe(copy_start, copied, keep - 1 - mtp_accepted)
        mtp_all = mtp_accepted == mtp_drafts
        observed = keep - 1 if mtp_all else mtp_accepted
        policy.observe(observed, observed if (stop or mtp_all) else mtp_drafts, context, info, mtp_drafts)
        out["ms"] += float(costs[k - 1])
        out["tokens"] += keep
        pos += keep
        remaining -= keep
        if stop:
            break
    return out


def run(cases, header, policy_factory) -> dict:
    stops = set(header["stop_tokens"])
    return {c["case"]: simulate(c, stops, policy_factory(), header["max_tokens"]) for c in cases}


POLICIES = {"P0": Policy, "I1a": TierBackoffPolicy, "I1": ConfidencePolicy}


def validate(trace: Path, pairs: list, policy: str = "P0") -> dict:
    header, cases = load_trace(trace)
    sim = run(cases, header, POLICIES[policy])
    report = {"cases": 0, "exact": 0, "completion_tokens_match": 0, "prompt_tokens_match": 0, "mismatches": []}
    for responses, suite in pairs:
        counts = spec_stats.record_counts(responses, spec_stats.load_suite_texts([suite]))
        counts.pop("_skipped_other_budget", None)
        for c in cases:
            if c["case"] not in counts:
                continue
            want, got = counts[c["case"]], sim[c["case"]]
            report["cases"] += 1
            report["prompt_tokens_match"] += want["prompt_n"] == c["prompt_tokens"]
            same = (want["draft_rounds"], want["draft_n"], want["draft_n_accepted"], want["lookup_n"],
                    want["lookup_n_accepted"]) == (got["cycles"], got["drafted"], got["accepted"], got["lookup"],
                                                   got["lookup_accepted"])
            report["exact"] += same
            if not same:
                report["mismatches"].append({"case": c["case"], "server": want, "sim": got,
                                             "trace_prompt_tokens": c["prompt_tokens"]})
    return report


def compare(trace: Path) -> dict:
    header, cases = load_trace(trace)
    policies = [Policy, TierPolicy, TierBackoffPolicy, ConfidencePolicy, CostConfidencePolicy, OraclePolicy]
    out = {"note": "modeled time from the C1 cycle-cost table (mtp_costs.hpp); a dropped draft adds 2 ms",
           "policies": {}}
    for factory in policies:
        sims = run(cases, header, factory)
        groups: dict = {"all": list(sims)}
        for c in cases:
            groups.setdefault(c["category"], []).append(c["case"])
        res = {}
        for g, ids in groups.items():
            tot = {k: sum(sims[i][k] for i in ids) for k in ("cycles", "drafted", "accepted", "tokens", "ms")}
            res[g] = {**tot, "tok_s": 1000.0 * tot["tokens"] / tot["ms"] if tot["ms"] else None,
                      "drafts_per_cycle": tot["drafted"] / tot["cycles"] if tot["cycles"] else None,
                      "acceptance": tot["accepted"] / tot["drafted"] if tot["drafted"] else None}
        out["policies"][factory.name] = res
    base = out["policies"]["P0"]
    for name, res in out["policies"].items():
        for g in res:
            res[g]["vs_P0"] = res[g]["tok_s"] / base[g]["tok_s"] if base[g]["tok_s"] else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate")
    v.add_argument("trace", type=Path)
    v.add_argument("--responses", type=Path, action="append", required=True)
    v.add_argument("--suite", type=Path, action="append", required=True)
    v.add_argument("--out", type=Path)
    v.add_argument("--policy", choices=sorted(POLICIES), default="P0",
                   help="policy the server ran (P0: c94572b3 controller, I1a: f1cf33ac, I1: current confidence rule)")
    c = sub.add_parser("compare")
    c.add_argument("trace", type=Path)
    c.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.command == "validate":
        if len(a.responses) != len(a.suite):
            ap.error("give one --suite per --responses")
        result = validate(a.trace, list(zip(a.responses, a.suite)), a.policy)
    else:
        result = compare(a.trace)
    text = json.dumps(result, indent=1) + "\n"
    if a.out:
        a.out.write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
