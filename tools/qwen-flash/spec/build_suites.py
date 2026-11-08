"""Builds the W9 speculative-decoding prompt suites (plan/archive/rotated-quant/SPEC-DECODE.md 4.2).

suite_k.json: the Kyojin comparison prompts, copied verbatim from Yamz-Labs/kyojin (MIT; LICENSE "Copyright (c) 2025 Turboderp", Kyojin additions
MIT per doc/credits.md) at 2132bd248e3bfe07258a408669608f3d2cfdda79:
  - tools/bench.py DECODE_PROMPTS (prose/chat/code, three each; Kyojin's published card prompts)
  - tools/qwen/spec_prompts.py EXTRA (multi/copy, the first four each, as tools/qwen/spec_bench.py uses them)
  tools/qwen/spec_bench.py also reads chat/prose/code from prompts10.py, which the repository does not contain.

suite_d.json: long-context prompts (category d8k / d32k, four each). Context text is this repository's own Markdown
documentation read with `git show <revision>:<path>` so the bytes are pinned; each prompt takes a distinct slice.
Two prompts per depth end with Kyojin's context-curve instruction (tools/qwen/serve_bench_ctx.py); two ask about
the context itself. The token count is approximate (server usage reports the exact prompt_tokens).

Usage: python3 -I tools/qwen-flash/spec/build_suites.py [--revision REV] [--out DIR]
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

KYOJIN = 'Yamz-Labs/kyojin@2132bd248e3bfe07258a408669608f3d2cfdda79'

DECODE_PROMPTS = {
    "prose": [
        "Write a 300-word story about a lighthouse.",
        "Write a 300-word story about a lighthouse. Use a different keeper and a different coast.",
        "Tell a 300-word story set in a lighthouse during a winter storm, with a different ending.",
    ],
    "chat": [
        "Explain how a hash map works, with its time complexity, in about 300 words.",
        "Explain how a hash map works and where it degrades, in about 300 words.",
        "Explain hash maps to a backend engineer: buckets, collisions, resizing. About 300 words.",
    ],
    "code": [
        "Write a Python function that parses an ISO-8601 duration string into seconds, with tests.",
        "Write a Python function that parses an ISO-8601 duration into seconds, handling negative "
        "durations, with unit tests.",
        "Write a Python function parsing ISO-8601 durations (P[n]Y[n]M[n]DT[n]H[n]M[n]S) into a "
        "timedelta, with tests for every unit.",
    ],
}
CODE_SNIP = '''def moving_average(values, window):
    if window <= 0:
        raise ValueError("window must be positive")
    result = []
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        if i >= window - 1:
            result.append(total / window)
    return result

def median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2
'''
QUOTE = ("The committee noted that the proposed regulation would require all operators of automated "
         "vehicles to maintain a continuous audit log of sensor inputs, control decisions and manual overrides, "
         "retained for no fewer than twenty-four months and made available to the supervisory authority within "
         "five working days of a written request. Operators who fail to comply may be subject to a fine of up to "
         "two percent of annual turnover.")
EXTRA = {
    "multi": [
        "Explique-moi en detail comment fonctionne une pompe a chaleur et pourquoi elle est plus efficace qu'un "
        "radiateur electrique.",
        "Escribe un cuento corto sobre un pescador que encuentra una lampara magica en la playa.",
        "Erklaere mir ausfuehrlich den Unterschied zwischen einer GmbH und einer AG in Deutschland.",
        "请详细介绍一下如何准备一次长途徒步旅行，包括装备、路线和安全注意事项。",
    ],
    "copy": [
        "Here is a Python module:\n\n```python\n" + CODE_SNIP + "```\n\nRewrite the whole module with type hints "
        "and docstrings, keeping the logic identical.",
        "Here is a Python module:\n\n```python\n" + CODE_SNIP + "```\n\nRename `values` to `data` everywhere and "
        "print the full updated module.",
        "Quote the following passage word for word, then summarize it in two sentences.\n\n" + QUOTE,
        "Here is a passage:\n\n" + QUOTE + "\n\nExtract every obligation as a bullet list, reusing the exact wording "
        "of the passage.",
    ],
}

# Kyojin tools/qwen/serve_bench_ctx.py instruction, and two questions about the context itself.
STORY = "\n\nWrite a long, detailed continuation of the story about a lighthouse keeper (at least 300 words)."
ASK = [
    "\n\nExplain in detail the main design decisions described in the documentation above, quoting the relevant "
    "option names and numbers.",
    "\n\nWrite a detailed step-by-step guide, based only on the documentation above, for a new developer who must "
    "build, run and benchmark this project.",
]
DEPTHS = {"d8k": 8192, "d32k": 32768}
# Characters per token assumed for the slice length. Markdown with code measured nowhere here yet: 3.6 keeps a
# 32K slice below the 49152-token window even at 2.5 characters per token.
CHARS_PER_TOKEN = 3.6


def kyojin_suite():
    prompts = []
    for category, texts in list(DECODE_PROMPTS.items()) + list(EXTRA.items()):
        source = 'tools/bench.py DECODE_PROMPTS' if category in DECODE_PROMPTS else 'tools/qwen/spec_prompts.py EXTRA'
        for i, text in enumerate(texts):
            prompts.append({"id": f"k_{category}_{i}", "category": category, "text": text,
                            "source": f"{KYOJIN} {source}[{category!r}][{i}]"})
    return {"schema": "gufo.speculative-corpus.v1", "version": "w9-kyojin-1",
            "note": f"W9 Kyojin comparison suite, verbatim from {KYOJIN} (MIT); built by "
                    "tools/qwen-flash/spec/build_suites.py.",
            "prompts": prompts}


def git(*args):
    return subprocess.run(['git', *args], check=True, capture_output=True).stdout


def docs_text(revision):
    paths = sorted(p for p in git('ls-tree', '-r', '--name-only', revision, '--', 'docs').decode().splitlines()
                   if p.endswith('.md'))
    parts = [git('show', f'{revision}:{p}').decode('utf-8') for p in paths]
    return '\n\n'.join(parts), paths


def context_suite(revision):
    text, paths = docs_text(revision)
    prompts, offset = [], 0
    for category, depth in DEPTHS.items():
        chars = int(depth * CHARS_PER_TOKEN)
        for i, tail in enumerate([STORY, ASK[0], STORY, ASK[1]]):
            if offset + chars > len(text):
                offset = 0  # wrap: earlier slices are of other lengths, so prompts still differ
            body = text[offset:offset + chars]
            prompts.append({"id": f"{category}_{i}", "category": category, "text": body + tail,
                            "source": f"git show {revision}:docs/**/*.md chars [{offset}, {offset + chars})"})
            offset += chars
    return {"schema": "gufo.speculative-corpus.v1", "version": "w9-context-1",
            "note": f"W9 long-context suite: {len(paths)} Markdown files under docs/ at {revision} "
                    f"({len(text)} characters, sha256 {hashlib.sha256(text.encode()).hexdigest()}), sliced at "
                    f"{CHARS_PER_TOKEN} characters per target token; instructions from {KYOJIN} "
                    "tools/qwen/serve_bench_ctx.py (STORY) and W9 (ASK). Built by "
                    "tools/qwen-flash/spec/build_suites.py.",
            "prompts": prompts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--revision', default='c94572b33a9c86f315e7860bf22b121f36b47bbd')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()
    for name, doc in (('suite_k.json', kyojin_suite()), ('suite_d.json', context_suite(a.revision))):
        path = a.out / name
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(name, len(doc['prompts']), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
