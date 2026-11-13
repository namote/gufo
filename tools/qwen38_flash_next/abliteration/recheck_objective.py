#!/usr/bin/env python3
"""Label re-check thoughts in thinking traces and write SPSA objective rows.

Input: think_eval.py records ({"id", "answer", "correct", "finish_reason",
"reasoning", "content"}) and the problem set they ran on ({"id", "problem"}).
A trace is split into thoughts at blank lines (as build_think_dirs.py does);
a thought "opens with a re-check" when RECHECK (the same expression as
build_think_dirs.py) matches its start.

Running answer candidate (the heuristic). Each thought may state a candidate:
  strong      the last \\boxed{N} value, or the last "answer is N" /
              "answer: N" / "answer N" / "final answer ... N" number; when
              an operator follows N ("answer is 657 + 64 = 721"), the last
              "= N" result of that sentence instead, or nothing;
  conclusion  the last "= N" inside a clause that opens with so, thus,
              therefore, hence, this/that/which gives, we get or giving
              (up to the end of its sentence or line);
  result      the last "= N" anywhere whose number is not followed by an
              operator (the result of a computation, not a term in it).
A forward thought states its strong candidate, else its conclusion, else its
result. A re-check thought states only a strong or conclusion candidate:
verification arithmetic ("Let me verify: 45 - 9 = 36") is not an answer.
The running candidate after thought i is the latest strong candidate in
thoughts 0..i; while none has appeared yet, the latest other one. Values are
integers (AIME and the MATH subset have integer answers); a candidate that is
not an integer is ignored, and a trace whose gold answer is not an integer
is skipped (counted as non_integer_gold).

Failure modes, by construction:
  - Intermediate results: before any strong candidate, a running candidate is
    often an intermediate quantity. Only changes to or from the gold value
    count, so this mostly costs recall, but a sub-result that equals the gold
    answer by chance can produce a false useful/harmful.
  - Conclusions about sub-quantities: "So solutions exist mod 17^2 = 289"
    inside a re-check reads as a candidate, so a false harmful remains
    possible; the report keeps examples to judge the rate.
  - Strong precedence: once a strong candidate exists, a recomputation stated
    without one does not move the candidate until the model restates the
    answer, so a re-check that silently corrects a stated answer reads as no
    change.
  - Episode boundary: the episode is the re-check thought and the thoughts
    after it up to the next forward thought (one whose opener does not match
    RECHECK). A recomputation that continues in forward-looking thoughts after
    "Wait, let me recompute." falls outside the episode, so it reads as no
    change (neutral).
  - Phrasing outside the patterns ("it is 204") is missed.

Labels for each re-check thought j (episode j..e-1; before = running candidate
after thought j-1; after = running candidate after thought e-1; gold = the
problem's answer):
  useful     before != gold and after == gold (including no candidate before),
             and gold was not the running candidate at any earlier thought (a
             return to an answer already found is not a rescue; this also
             drops the rare right-wrong-right recovery);
  harmful    before == gold and after is a different candidate that the run
             also ended with (its final boxed answer, finished runs only): a
             change the run keeps. Without that confirmation a "change" is
             nearly always a conclusion about a sub-quantity inside the check
             ("so 1080 - 1070 = 10" while the answer stays n = 8);
  redundant  before == after == gold, and at least --redundant-after earlier
             episodes already confirmed the gold candidate (a re-check loop on
             an answer that is already right). Confirmations count per
             episode: every re-check thought of the first confirming episode
             is a first confirmation, not just its opener;
  neutral    everything else, with a reason in the report: no candidate, no
             change, wrong to wrong, the first confirmations of a right
             answer, an unconfirmed harmful change, or a return to an earlier
             gold candidate.

Objective rows (spsa.py JSONL; every row is scored teacher-forced inside the
reasoning channel, so each carries "reasoning" = the trace before the scored
thought, "score_reasoning": true, "reasoning_effort" and "chain"):
  pairs     (--pairs; suppress) harmful and redundant re-checks with a forward
            alternative: "propaganda" is the re-check thought's opener,
            "candid" is the forward thought that ends its episode (thought e),
            both scored on their first --opener-tokens tokens at the boundary
            before thought j. spsa.py maximizes lp(candid) - lp(propaganda).
            Re-checks whose episode runs to the end of the trace have no
            forward alternative and are counted, not written.
  keep      (--ordinary) the opener of useful re-checks; spsa.py penalizes a
            per-token log-likelihood drop against OFF. With
            --keep-first-confirmations, also the opener of the first
            confirmations of a right answer (neutral by label): a first check
            of a fresh answer is the re-check worth keeping, the repeats
            after it are the redundant ones suppressed.
  guard     (--guard) the final answer content after the full reasoning
            (closed, no "score_reasoning"), only for traces the OFF run got
            right and finished; spsa.py penalizes a per-token drop against OFF
            beyond --guard-margin.
Neutral re-checks are excluded and only counted.

Rows of one trace share "chain" (the trace key): spsa.py sends a chain's
requests one after another in ascending prefix length, so each request can
resume from the previous one's prompt checkpoints instead of prefilling the
trace again. --max-boundaries caps the boundaries per trace (a seeded sample).

Standard library only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys

# Identical to build_think_dirs.py (the re-check direction's labels).
RECHECK = re.compile(r'^\W*(wait|hmm|alternatively|actually|but wait|hold on|hmm+|double[- ]check|'
                     r'let me (?:double[- ]check|re-?check|verify|check|reconsider|re-?examine|re-?read|'
                     r're-?compute|re-?count|re-?verify|re-?think|make sure))\b', re.I)
SUFFIX = '\n\nPut the final answer in \\boxed{}.'  # think_eval.py's user message suffix
NUMBER = r'(-?\d{1,3}(?:(?:,|\\,|\{,\})\d{3})+|-?\d+)(?:\.0+)?(?![\d.]*\d)'
ANSWER = re.compile(r'\banswers?\b\s*(?:is|=|:|would be|should be|will be|of)?\s*(?:\*\*|\$|\\\(|\\boxed\{|\{|\s)*'
                    + NUMBER, re.I)
FINAL = re.compile(r'\bfinal answer\b[^\n\d]{0,40}?' + NUMBER, re.I)
# A term continues after N: an operator, a digit, or a LaTeX command other than a
# delimiter close, spacing, percent or text ("\\cdot" is one, "\\)" and "\\text{ min}" are not).
TERM = r'\s*(?:[-+*/^×·(]|\d|\\(?![)\]]|text|mathrm|,|;|!|%|quad|qquad|\\))'
RESULT = re.compile(r'=\s*(?:\*\*|\$|\\\(|\s)*' + NUMBER + r'(?![A-Za-z])(?!' + TERM + ')')
OPERATOR = re.compile(r'[A-Za-z]|' + TERM)  # what follows a term ("3x", "657 + 64"), not a result
CONCLUSION = re.compile(r'\b(?:so|thus|therefore|hence|(?:this|that|which) gives|we get|giving)\b', re.I)


def split_thoughts(reasoning):
    """Thoughts as (start, end) character offsets; each but the last ends
    after its blank line, so reasoning[:start] is the exact prefix before a
    thought. Whitespace-only pieces join the thought before them."""
    spans, start = [], 0
    while start < len(reasoning):
        cut = reasoning.find('\n\n', start)
        end = len(reasoning) if cut < 0 else cut + 2
        while end < len(reasoning) and reasoning[end] == '\n':
            end += 1  # a longer newline run stays with the thought it ends
        if spans and not reasoning[start:end].strip():
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
        start = end
    return spans


def boxed_values(text):
    out, i = [], text.find('\\boxed{')
    while i >= 0:
        j, depth = i + 7, 1
        while j < len(text) and depth:
            depth += {'{': 1, '}': -1}.get(text[j], 0)
            j += 1
        out.append((i, text[i + 7:j - 1] if not depth else text[i + 7:j]))
        i = text.find('\\boxed{', j)
    return out


def as_int(text):
    """The integer a candidate string states, or None."""
    if text is None:
        return None
    t = re.sub(r'\\[,!; ]|\$|\s|\\text\{[^}]*\}|\{,\}', '', str(text)).replace(',', '')
    t = re.sub(r'^[a-zA-Z]\w*=', '', t)
    m = re.fullmatch(r'(-?\d+)(?:\.0+)?', t)
    return int(m.group(1)) if m else None


def _last(pairs):
    return max(pairs)[1] if pairs else None


def thought_candidates(text):
    """(strong, conclusion, result) integer candidates a thought states, each
    the last of its kind in the thought; None if absent."""
    strong = [(pos, as_int(value)) for pos, value in boxed_values(text)]
    for pattern in (ANSWER, FINAL):
        for m in pattern.finditer(text):
            if not OPERATOR.match(text, m.end()):
                strong.append((m.start(1), as_int(m.group(1))))
                continue
            # "answer is 657 + 64 = 721": the last result in the sentence, if any.
            sentence = text[m.end():]
            stop = re.search(r'\.(?:\s|$)|\n', sentence)
            sentence = sentence[:stop.start() + 1] if stop else sentence
            results = [w for w in RESULT.finditer(sentence)]
            if results:
                strong.append((m.end() + results[-1].start(1), as_int(results[-1].group(1))))
    conclusion = []
    for m in CONCLUSION.finditer(text):
        clause = text[m.end():]
        stop = re.search(r'\.(?:\s|$)|\n', clause)
        clause = clause[:stop.start() + 1] if stop else clause
        conclusion += [(m.end() + w.start(1), as_int(w.group(1))) for w in RESULT.finditer(clause)]
    result = [(w.start(1), as_int(w.group(1))) for w in RESULT.finditer(text)]
    return tuple(_last([(p, v) for p, v in kind if v is not None])
                 for kind in (strong, conclusion, result))


def running_candidates(reasoning, spans, rechecks):
    """The running candidate after each thought (see the module doc)."""
    out, strong, other = [], None, None
    for (start, end), recheck in zip(spans, rechecks):
        s, conclusion, result = thought_candidates(reasoning[start:end])
        if s is not None:
            strong = s
        stated = conclusion if recheck or conclusion is not None else result
        if stated is not None:
            other = stated
        out.append(strong if strong is not None else other)
    return out


def label_trace(reasoning, gold, final=None, redundant_after=1):
    """One record per re-check thought: index j, episode end e (the forward
    thought after it, or len(spans)), before/after candidates, label and, for
    a neutral one, why. `final` is the run's final answer (None when the run
    did not finish with one)."""
    spans = split_thoughts(reasoning)
    rechecks = [bool(RECHECK.match(reasoning[s:e])) for s, e in spans]
    running = running_candidates(reasoning, spans, rechecks)
    gold, final = as_int(gold), as_int(final)
    if gold is None:
        raise ValueError('the gold answer is not an integer')
    confirmed, out, opening = [], [], 0  # episode starts that confirmed the gold answer
    for j in range(1, len(spans)):
        if not rechecks[j]:
            continue
        if not rechecks[j - 1]:
            opening = j  # this thought opens an episode; later re-checks in it share it
        e = j
        while e < len(spans) and rechecks[e]:
            e += 1
        before, after = running[j - 1], running[e - 1]
        reason = None
        if after == gold and before != gold:
            if gold in running[:j]:
                label, reason = 'neutral', 'returns to an earlier gold candidate'
            else:
                label = 'useful'
        elif before == gold and after is not None and after != gold:
            if after == final:
                label = 'harmful'
            else:
                label, reason = 'neutral', 'unconfirmed harmful'
        elif before == gold and after == gold:
            if opening not in confirmed:
                confirmed.append(opening)
            if confirmed.index(opening) >= redundant_after:
                label = 'redundant'
            else:
                label, reason = 'neutral', 'first confirmation'
        else:
            label = 'neutral'
            reason = ('no candidate' if before is None and after is None
                      else 'no change' if before == after else 'wrong to wrong')
        out.append(dict(j=j, e=e, before=before, after=after, label=label, reason=reason,
                        has_candidate=before is not None or after is not None))
    return spans, out


def opener(text, chars=240):
    """The start of a thought, cut at a space after `chars` characters so the
    first tokens tokenize as in the whole thought."""
    if len(text) <= chars:
        return text
    cut = text.find(' ', chars)
    return text if cut < 0 else text[:cut]


def key(*parts):
    return hashlib.sha256(':'.join(parts).encode()).hexdigest()[:16]


def cost(rows, rate):
    """Per-theta scoring cost of `rows` in estimated prompt tokens: every
    request prefilling its whole prefix, and each chain prefilling its longest
    prefix once (checkpoint reuse along the chain), plus the scored rows."""
    tokens = lambda x: round(len(x['reasoning']) / rate[x['chain']])
    requests = sum(2 if 'candid' in x else 1 for x in rows)
    without = sum((2 if 'candid' in x else 1) * tokens(x) for x in rows)
    longest = {}
    for x in rows:
        longest[x['chain']] = max(longest.get(x['chain'], 0), tokens(x))
    scored = sum((2 if 'candid' in x else 1) * x['score_max_tokens'] for x in rows)
    return dict(requests=requests, chains=len(longest), prefill_without_reuse=without,
                prefill_with_reuse=sum(longest.values()), max_prefix=max(longest.values(), default=0),
                scored_tokens=scored)


def build(sets, opener_tokens=3, guard_tokens=1024, max_boundaries=8, redundant_after=1, seed=0,
          examples=3, keep_first=False):
    """Rows and a report from [(name, problems, records, effort)]."""
    pairs, keep, guard = [], [], []
    report = {}
    for name, problems, records, effort in sets:
        counts = dict(traces=0, rechecks=0, useful=0, harmful=0, redundant=0, neutral=0,
                      with_candidate=0, no_alternative=0, capped=0, pairs=0, keep=0, guard=0,
                      non_integer_gold=0)
        shown = {k: [] for k in ('useful', 'harmful', 'redundant', 'neutral')}
        reasons = {}
        rate = {}  # characters per generated token, per chain (for the cost estimate)
        for r in records:
            reasoning = r.get('reasoning') or ''
            if not reasoning or r['id'] not in problems:
                continue
            p = problems[r['id']]
            if str(p.get('level')) == '2025' or 'aime2025' in str(r['id']).lower():
                raise SystemExit(f'{name}: {r["id"]} is AIME 2025, the held-out set')
            if as_int(r['answer']) is None:
                counts['non_integer_gold'] += 1
                continue
            counts['traces'] += 1
            prompt = p['problem'] + SUFFIX
            chain = f'{name}:{r["id"]}'
            generated = (r.get('usage') or {}).get('completion_tokens')
            rate[chain] = (len(reasoning) + len(r.get('content') or '')) / generated if generated else 2.0
            final = r.get('got') if r.get('finish_reason') == 'stop' else None
            spans, labels = label_trace(reasoning, r['answer'], final, redundant_after)
            counts['rechecks'] += len(labels)
            for lab in labels:
                counts[lab['label']] += 1
                if lab['reason']:
                    reasons[lab['reason']] = reasons.get(lab['reason'], 0) + 1
                counts['with_candidate'] += lab['has_candidate']
                if len(shown[lab['label']]) < examples:
                    s, t = spans[lab['j']]
                    shown[lab['label']].append(dict(
                        trace=chain, thought=lab['j'], before=lab['before'], after=lab['after'],
                        reason=lab['reason'],
                        gold=r['answer'], text=reasoning[s:t][:160]))
            base = dict(prompt=prompt, score_reasoning=True, reasoning_effort=effort,
                        score_max_tokens=opener_tokens, chain=chain)
            chosen = []
            for lab in labels:
                if lab['label'] in ('harmful', 'redundant'):
                    if lab['e'] >= len(spans):
                        counts['no_alternative'] += 1
                        continue
                    chosen.append(('pair', lab))
                elif lab['label'] == 'useful' or (keep_first and lab['reason'] == 'first confirmation'):
                    chosen.append(('keep', lab))
            if max_boundaries and len(chosen) > max_boundaries:
                rng = random.Random(f'{seed}:{chain}')
                keepers = [c for c in chosen if c[0] == 'keep']  # rare: keep them all
                rest = [c for c in chosen if c[0] != 'keep']
                room = max(0, max_boundaries - len(keepers))
                picked = rng.sample(rest, min(room, len(rest)))
                counts['capped'] += len(chosen) - len(keepers) - len(picked)
                chosen = sorted(keepers + picked, key=lambda c: c[1]['j'])
            for kind, lab in chosen:
                j = lab['j']
                start = spans[j][0]
                row = dict(base, reasoning=reasoning[:start])
                text = opener(reasoning[start:spans[j][1]])
                if kind == 'pair':
                    s, t = spans[lab['e']]
                    pairs.append(dict(row, id=key(chain, str(j), 'p'), label=lab['label'],
                                      propaganda=text, candid=opener(reasoning[s:t])))
                else:
                    keep.append(dict(row, id=key(chain, str(j), 'k'), answer=text,
                                     label=lab['label'] if lab['label'] == 'useful' else 'first-confirmation'))
            counts['pairs'] += sum(1 for kind, _ in chosen if kind == 'pair')
            counts['keep'] += sum(1 for kind, _ in chosen if kind == 'keep')
            content = r.get('content') or ''
            if r.get('correct') and r.get('finish_reason') == 'stop' and content.strip():
                guard.append(dict(id=key(chain, 'g'), prompt=prompt, answer=content,
                                  reasoning=reasoning, reasoning_effort=effort,
                                  score_max_tokens=guard_tokens, chain=chain))
                counts['guard'] += 1
        counts['candidate_rate'] = round(counts['with_candidate'] / counts['rechecks'], 3) if counts['rechecks'] else None
        report[name] = dict(counts=counts, neutral_reasons=reasons, examples=shown,
                            cost=cost([x for x in pairs + keep + guard if x['chain'] in rate], rate))
    return pairs, keep, guard, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--set', action='append', required=True, metavar='NAME:PROBLEMS:RECORDS:EFFORT',
                    help='a trace set: its name, the problem JSONL, the think_eval.py records '
                         'and the reasoning_effort they ran at (low, medium or xhigh)')
    ap.add_argument('--out', required=True, help='output directory (pairs, keep, guard, report)')
    ap.add_argument('--opener-tokens', type=int, default=3)
    ap.add_argument('--guard-tokens', type=int, default=1024)
    ap.add_argument('--max-boundaries', type=int, default=8, help='per trace; 0 keeps all')
    ap.add_argument('--redundant-after', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--keep-first-confirmations', action='store_true',
                    help='also keep the openers of first confirmations of a right answer')
    args = ap.parse_args(argv)
    if not 1 <= args.opener_tokens <= 1024 or not 1 <= args.guard_tokens <= 1024:
        ap.error('--opener-tokens and --guard-tokens must be 1..1024')
    sets = []
    for spec in args.set:
        name, problems, records, effort = spec.split(':')
        if effort not in ('low', 'medium', 'xhigh'):
            ap.error(f'{spec}: effort must be low, medium or xhigh')
        P = {p['id']: p for p in map(json.loads, Path(problems).read_text().splitlines()) if p}
        R = [json.loads(l) for l in Path(records).read_text().splitlines() if l.strip()]
        sets.append((name, P, R, effort))
    pairs, keep, guard, report = build(sets, args.opener_tokens, args.guard_tokens,
                                       args.max_boundaries, args.redundant_after, args.seed,
                                       keep_first=args.keep_first_confirmations)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fname, rows in (('pairs.jsonl', pairs), ('keep.jsonl', keep), ('guard.jsonl', guard)):
        (out / fname).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (out / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    for name, rep in report.items():
        print(name, json.dumps(rep['counts']))
    print(f'wrote {len(pairs)} pairs, {len(keep)} keep, {len(guard)} guard rows to {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
