"""Fit a dynamic-abliteration detector (`trigger`) for one runtime profile.

CPU only (numpy); never starts a server or a GPU job.

The detector is one unit direction at one layer, read exactly where the
native collector (collect.cpp, readout
`last-prompt-token-post-layer-hc-mean-fp32-v1`) reads its activations: the
post-layer residual of the last prompt token, averaged over the HC streams.
The server computes the same score <h, d> at that layer for the last prompt
row of a prefill and for every decoded row, and switches the profile ON for
the rest of the request when the score exceeds `threshold`.

    d = normalize(mean(target group) - mean(ordinary)),  score = <h, d>

Validation is split by topic, never by sentence: the prompts of one topic
all fall into one fold, so a held-out score never comes from a direction
that saw a paraphrase of the same question. What a topic is must be stated:
`--topics FILE` (JSONL `{"id": ..., "topic": ...}`) maps records to topics,
for example the same question in several languages and templates;
`--topics none` uses no map. Records the map does not name (all of them with
`none`) take a record's own `topic` member, else the record ID without its
`-plain`/`-think` suffix: only the two thinking modes of one text share a
topic, while language and template variants of a question are separate
topics and may straddle folds.

Grouped k-fold: each fold's direction is fitted on the other folds' topics
of the collection (target group against its `ordinary` group) and scores the
fold's held-out rows. Extra negatives (`--negatives DIR[:GROUP]`, for example
a harmless-question collection) never enter a direction; they are dealt into
the folds by topic too, so each is scored once, out of fold.

The threshold is the smallest value whose out-of-fold false-positive rate
is at most `--max-false-positive` on each negative set separately: the
held-out ordinary group and every extra collection. The nested estimate picks each fold's threshold from
the other folds' negatives only and evaluates it on the fold, so its recall
and false-positive rate are not tuned on the data that reports them. The
deployed direction is fitted on every topic.

`--threshold X` replaces the fitted threshold with an explicit one (for
example chosen from server measurements); the report keeps the fitted value
as `fitted_threshold` and marks `threshold_source` "override".

`--with NAME` (repeatable) names companion profiles the server turns ON
together with this one when it triggers (`trigger.with`); they need no
trigger of their own.

The output is the input profile with a `trigger` member; the input file is
never modified, and an existing output is refused.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

import numpy as np

READOUTS = {
    'last-prompt-token-post-layer-hc-mean-fp32-v1': 'writer-before-hc',
    'last-prompt-token-post-layer-fp32-v1': 'writer-before-residual',
}
TRIGGER_SCHEMA = 'gufo-abliteration-trigger-v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def topic_of(record, topics):
    identity = record['id']
    if identity in topics:
        return topics[identity]
    if record.get('topic'):
        return str(record['topic'])
    for suffix in ('-plain', '-think'):
        if identity.endswith(suffix):
            return identity[:-len(suffix)]
    return identity


def load_topics(path):
    topics = {}
    if path is None or path == 'none':
        return topics
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            topics[row['id']] = str(row['topic'])
    return topics


def load_collection(directory, layer, groups, topics=None, prefix=''):
    """(rows [n][hidden] at `layer`, groups, topic keys, thinking, manifest)."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('split') != 'collection':
        raise ValueError(f'{directory}: fit only a collection split')
    if manifest.get('readout') not in READOUTS:
        raise ValueError(f'{directory}: unknown collector readout')
    hidden, layers = manifest['hidden'], manifest['layers']
    if not 0 <= layer < layers:
        raise ValueError(f'layer must be in [0, {layers})')
    rows, names, keys, thinking = [], [], [], []
    for record in manifest['records']:
        if record['group'] not in groups:
            continue
        receipt = json.loads((directory / (record['id'] + '.json')).read_text())
        raw = (directory / (record['id'] + '.f32')).read_bytes()
        if sha(raw) != receipt['activation_sha256']:
            raise ValueError(f'{record["id"]}: activation integrity mismatch')
        values = np.frombuffer(raw, dtype='<f4')
        if values.size != layers * hidden:
            raise ValueError(f'{record["id"]}: activation size mismatch')
        row = values.reshape(layers, hidden)[layer].astype(np.float64)
        if not np.isfinite(row).all():
            raise ValueError(f'{record["id"]}: nonfinite activation')
        rows.append(row)
        names.append(record['group'])
        keys.append(prefix + topic_of(record, topics or {}))
        thinking.append(bool(record.get('thinking')))
    if not rows:
        raise ValueError(f'{directory}: no records of groups {sorted(groups)}')
    return (np.stack(rows), np.array(names), np.array(keys),
            np.array(thinking), manifest)


def auc(positive, negative):
    """Mann-Whitney AUC with ties counted half."""
    positive, negative = np.asarray(positive), np.asarray(negative)
    if not len(positive) or not len(negative):
        return float('nan')
    scores = np.concatenate([positive, negative])
    order = scores.argsort(kind='mergesort')
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average the ranks of ties.
    sorted_scores = scores[order]
    start = 0
    for end in range(1, len(scores) + 1):
        if end == len(scores) or sorted_scores[end] != sorted_scores[start]:
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
    count = len(positive)
    return float((ranks[:count].sum() - count * (count + 1) / 2) /
                 (count * len(negative)))


def threshold_for(negative_sets, max_false_positive):
    """The smallest t with mean(n > t) <= max_false_positive for every set."""
    result = -np.inf
    for negatives in negative_sets:
        negatives = np.sort(np.asarray(negatives))[::-1]
        if not len(negatives):
            continue
        allowed = int(np.floor(max_false_positive * len(negatives) + 1e-9))
        result = max(result, float(negatives[min(allowed, len(negatives) - 1)]))
    return float(result)


def direction(rows, names, target):
    d = rows[names == target].mean(0) - rows[names == 'ordinary'].mean(0)
    norm = np.linalg.norm(d)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('target and ordinary means coincide')
    return d / norm


def assign_folds(keys, positive_keys, folds, seed):
    """Topic -> fold, positive and negative topics dealt separately."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(keys.tolist()))
    pos = [k for k in unique if k in positive_keys]
    neg = [k for k in unique if k not in positive_keys]
    fold = {}
    offset = 0
    for group in (pos, neg):
        order = rng.permutation(len(group))
        for i, index in enumerate(order):
            fold[group[index]] = (i + offset) % folds
        offset += len(group)
    return np.array([fold[k] for k in keys.tolist()])


def rate(values, threshold):
    values = np.asarray(values)
    return float(np.mean(values > threshold)) if len(values) else float('nan')


def fit(collection, target, layer, negatives=(), topics_path=None, folds=6,
        max_false_positive=0.01, seed=0, threshold_override=None):
    topics = load_topics(topics_path)
    rows, names, keys, thinking, manifest = load_collection(
        collection, layer, {target, 'ordinary'}, topics)
    if target == 'ordinary':
        raise ValueError('the target group must differ from ordinary')
    for group in (target, 'ordinary'):
        if np.count_nonzero(names == group) < 2:
            raise ValueError(f'{group} needs at least two records')
    target_topics = set(keys[names == target].tolist())
    if target_topics & set(keys[names == 'ordinary'].tolist()):
        raise ValueError('a topic holds both target and ordinary records')
    if len(target_topics) < folds or len(set(keys.tolist()) - target_topics) < folds:
        raise ValueError('every fold needs target and ordinary topics')
    fold = assign_folds(keys, target_topics, folds, seed)
    extra = []
    for index, spec in enumerate(negatives):
        path, _, group = spec.partition(':')
        x, _, k, _, m = load_collection(path, layer, {group or 'ordinary'},
                                     prefix=f'neg{index}:')
        if m['model_identity'] != manifest['model_identity'] or \
                m['readout'] != manifest['readout']:
            raise ValueError(f'{path}: another model or readout')
        extra.append((spec, x, assign_folds(k, set(), folds, seed + 1 + index)))
    # Out-of-fold scores.
    oof = np.empty(len(rows))
    extra_oof = [np.empty(len(x)) for _, x, _ in extra]
    directions = []
    for f in range(folds):
        train = fold != f
        d = direction(rows[train], names[train], target)
        directions.append(d)
        oof[~train] = rows[~train] @ d
        for (_, x, xf), out in zip(extra, extra_oof):
            out[xf == f] = x[xf == f] @ d
    positive = oof[names == target]
    ordinary = oof[names == 'ordinary']
    harmless = np.concatenate(extra_oof) if extra else np.empty(0)
    pooled_negative = np.concatenate([ordinary, harmless])
    # Each negative set (held-out ordinary, every extra collection) stays
    # within the false-positive budget on its own.
    threshold = threshold_for([ordinary] + extra_oof, max_false_positive)
    # An explicit threshold (for example chosen from server measurements)
    # replaces the fitted one; recall and false positives below are reported
    # at it, the nested estimate stays the fitted rule's.
    fitted_threshold = threshold
    if threshold_override is not None and not math.isfinite(threshold_override):
        raise ValueError('the threshold override must be finite')
    if threshold_override is not None:
        threshold = float(threshold_override)
    # Nested: each fold's threshold from the other folds' negatives.
    nested_tp = nested_fp = 0
    nested_pos = nested_neg = 0
    for f in range(folds):
        other = [ordinary[fold[names == 'ordinary'] != f]]
        held = [ordinary[fold[names == 'ordinary'] == f]]
        for (_, _, xf), out in zip(extra, extra_oof):
            other.append(out[xf != f])
            held.append(out[xf == f])
        t = threshold_for(other, max_false_positive)
        p = positive[fold[names == target] == f]
        n = np.concatenate(held)
        nested_tp += int(np.sum(p > t))
        nested_pos += len(p)
        nested_fp += int(np.sum(n > t))
        nested_neg += len(n)
    final = direction(rows, names, target)
    report = dict(
        layer=layer,
        target_group=target,
        folds=folds,
        seed=seed,
        max_false_positive=max_false_positive,
        counts=dict(target=int(np.sum(names == target)),
                    ordinary=int(np.sum(names == 'ordinary')),
                    extra_negatives=int(len(harmless)),
                    target_topics=len(target_topics),
                    ordinary_topics=len(set(keys[names == 'ordinary'].tolist()))),
        auc_ordinary=auc(positive, ordinary),
        auc_extra_negatives=auc(positive, harmless) if extra else None,
        auc_all_negatives=auc(positive, pooled_negative),
        threshold=threshold,
        threshold_source='override' if threshold_override is not None else 'fit',
        fitted_threshold=fitted_threshold,
        recall=rate(positive, threshold),
        false_positive_ordinary=rate(ordinary, threshold),
        false_positive_extra=rate(harmless, threshold) if extra else None,
        false_positive_all=rate(pooled_negative, threshold),
        nested_recall=nested_tp / nested_pos,
        nested_false_positive=nested_fp / nested_neg,
        fold_direction_cosine_min=float(min(d @ final for d in directions)),
        # The deployed direction on the extra negatives, which no direction
        # was fitted on: an in-sample-free false-positive check.
        final_false_positive_extra=(
            rate(np.concatenate([x @ final for _, x, _ in extra]), threshold)
            if extra else None),
        score_quantiles=dict(
            target=np.quantile(positive, [0.05, 0.5, 0.95]).tolist(),
            ordinary=np.quantile(ordinary, [0.05, 0.5, 0.95]).tolist(),
            extra=(np.quantile(harmless, [0.05, 0.5, 0.95]).tolist()
                   if extra else None)),
    )
    think = thinking[names == target]
    report['recall_by_thinking'] = dict(
        plain=rate(positive[~think], threshold),
        think=rate(positive[think], threshold))
    sources = dict(collection=sha((Path(collection) / 'manifest.json').read_bytes()),
                   topics=(sha(Path(topics_path).read_bytes())
                           if topics_path not in (None, 'none') else 'none'),
                   negatives=[dict(spec=spec, manifest_sha256=sha(
                       (Path(spec.partition(':')[0]) / 'manifest.json').read_bytes()))
                       for spec, _, _ in extra])
    report['sources'] = sources
    return final.astype(np.float32), report, manifest


def encode(values):
    return base64.b64encode(np.asarray(values, dtype='<f4').tobytes()).decode()


def write_profile(source, output, final, report, manifest, with_profiles=()):
    source, output = Path(source), Path(output)
    if output.exists():
        raise SystemExit('use a new output path; existing profiles are preserved')
    if output.resolve() == source.resolve():
        raise SystemExit('the input profile is never modified')
    profile = json.loads(source.read_text())
    if profile.get('model_identity') != manifest['model_identity'] or \
            profile.get('hidden') != manifest['hidden'] or \
            profile.get('layers') != manifest['layers'] or \
            profile.get('placement') != READOUTS[manifest['readout']]:
        raise SystemExit('profile and collection belong to different models')
    profile['trigger'] = dict(
        schema=TRIGGER_SCHEMA,
        layer=report['layer'],
        readout=manifest['readout'],
        direction=encode(final),
        threshold=float('%.9g' % report['threshold']),
        fit=report,

    )
    if with_profiles:
        # Companions the server turns ON with this profile when it triggers.
        names = list(with_profiles)
        if len(set(names)) != len(names) or output.stem in names or \
                any(not n for n in names):
            raise SystemExit('--with names distinct other profiles')
        profile['trigger']['with'] = names
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(profile, ensure_ascii=False,
                                    separators=(',', ':')) + '\n')
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--collection', required=True,
                        help='collector output with the target and ordinary groups')
    parser.add_argument('--target-group', default='refusal')
    parser.add_argument('--layer', type=int, required=True,
                        help='detector layer (the collector index)')
    parser.add_argument('--negatives', action='append', default=[],
                        help='DIR[:GROUP] extra negatives (default group ordinary); '
                             'repeatable, never used for the direction')
    parser.add_argument('--topics', required=True,
                        help='JSONL {"id", "topic"} topic map, or "none" (thinking '
                             'modes of one text only)')
    parser.add_argument('--folds', type=int, default=6)
    parser.add_argument('--max-false-positive', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--with', dest='with_profiles', action='append', default=[],
                        help='companion profile (file name without .json) the server '
                             'turns ON with this one when it triggers; repeatable')
    parser.add_argument('--threshold', type=float,
                        help='explicit threshold replacing the fitted one (recorded '
                             'as threshold_source "override" beside fitted_threshold)')
    parser.add_argument('--profile', help='runtime profile to copy the trigger into')
    parser.add_argument('--output', help='new profile path (requires --profile)')
    args = parser.parse_args()
    if (args.profile is None) != (args.output is None):
        parser.error('--profile and --output go together')
    if not 2 <= args.folds:
        parser.error('--folds must be at least 2')
    if not 0 <= args.max_false_positive < 1:
        parser.error('--max-false-positive must be in [0, 1)')
    if args.threshold is not None and not math.isfinite(args.threshold):
        parser.error('--threshold must be finite')
    final, report, manifest = fit(
        args.collection, args.target_group, args.layer, args.negatives,
        args.topics, args.folds, args.max_false_positive, args.seed,
        args.threshold)
    print(json.dumps(report, indent=2))
    if args.profile:
        write_profile(args.profile, args.output, final, report, manifest,
                      args.with_profiles)
        print(args.output)


if __name__ == '__main__':
    main()
