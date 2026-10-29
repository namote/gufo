#!/usr/bin/env python3
"""Build the RQ R0 evaluation corpus: fixed token windows for teacher forcing.

Four categories with the same token budget: English prose, Korean prose,
code, and chat (rendered with the Qwen chat framing). Every source is a file
of this repository (MIT, see LICENSE) read at one pinned git revision, so the
corpus is reproducible from history alone. The repository was started on
2026-08-11 and the chosen files were added from 2026-09-16 on, after the
Qwen3.8 27B release, so the text cannot be in the model's training data.
The technical-documentation domain is the cost of that choice; it is
recorded in the manifest. R3 calibration (W4) must not use the files listed
in the manifest.

Windows: each category fills `--windows-per-category` windows of
`--window-tokens` tokens. Documents are tokenized one at a time, capped at
`--doc-tokens`, and packed in order; the document that overflows a window is
truncated there and its rest is dropped, so every window starts at a document
boundary. Windows do not overlap.

Outputs in --out: windows.bin (little-endian int32 token ids), windows.json
(index read by tests/models/qwen27b/rq_logits.cpp), windows_text.jsonl
(decoded text per window, for review) and MANIFEST.json (sources, licenses,
token counts, sha256 of every output).

Run in the eval image:
  python3 -I tools/qwen27b/rq_quality/corpus.py build --repo /src \
      --revision <rev> --tokenizer /models/<dir>/tokenizer.json --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from gufo.control_tokens import kImEnd, kImStart  # noqa: E402

CORPUS_VERSION = "rq-r0-corpus 1"
LICENSE = "MIT (gufo repository LICENSE, Copyright (c) 2026 gufo contributors)"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

# Category -> ordered source files. Prose drops fenced code and tables;
# code is verbatim; chat turns document sections into question/answer turns.
SOURCES = {
    "en_prose": [
        "docs/cache-redesign/cards/07-retention-policy.md",
        "docs/cache-redesign/cards/09-idle-spill.md",
        "docs/cache-redesign/cards/11-publication-and-crash-safety.md",
        "docs/cache-redesign/cards/12-streamed-transfers.md",
        "docs/models/qwen3.8-27b/CONTINUATION-ADAPTER.md",
        "docs/cache-redesign/cards/13-tiered-lookup-and-eviction.md",
        "docs/cache-redesign/cards/10-disk-format-and-index.md",
    ],
    "ko_prose": [
        "plan/gufo-upstream-port/KVARN.md",
        "plan/gufo-upstream-port/SKINNY.md",
        "plan/v11-gufo/V11-GUFO-EVAL-20261020.md",
        "plan/gufo-upstream-port/R10-RESEARCH.md",
        "plan/v11-gufo/V11-GUFO-SEAT-DISK-PLAN-20261020.md",
        "plan/gufo-upstream-port/T-RESULTS.md",
        "plan/gufo-upstream-port/TESTS.md",
    ],
    "code": [
        "src/models/qwen/continuation_layout.cpp",
        "src/models/qwen/hip/kvarn_layout.hpp",
        "tools/qwen27b/drafts.py",
        "src/models/qwen/hip/kernels/attention_kvarn.hip",
        "tools/qwen27b/kvarn.py",
        "src/models/qwen/dflash_policy.hpp",
        "tools/rotquant/recipe.py",
        "src/core/crypto/sha256.cpp",
    ],
    "chat": [
        "docs/cache-redesign/cards/03-resource-ledger.md",
        "plan/gufo-upstream-port/P-RESULTS.md",
        "docs/cache-redesign/cards/05-prefix-index-and-lookup.md",
        "plan/gufo-upstream-port/U4F-RESULTS.md",
        "docs/cache-redesign/cards/06-slot-leases-and-mutation.md",
        "plan/v11-gufo/V11-GUFO-CACHE-DESIGN-20261020.md",
        "docs/cache-redesign/cards/08-committed-backing-and-streams.md",
        "plan/gufo-upstream-port/U5B-RESULTS.md",
    ],
}
CATEGORIES = tuple(SOURCES)

SYSTEM = {
    "en": "You are a helpful assistant for the gufo inference engine project. "
          "Answer from the project documents.",
    "ko": "당신은 gufo 추론 엔진 프로젝트를 돕는 어시스턴트입니다. 프로젝트 문서를 근거로 답하세요.",
}
QUESTIONS = {
    "en": ('Can you walk me through "{h}" in {t}?',
           'What does {t} say about "{h}"?',
           'Summarize the "{h}" section of {t} for me.'),
    "ko": ('{t}에서 "{h}" 부분을 설명해 줘.',
           '{t}의 "{h}" 내용을 정리해 줄래?',
           '"{h}"에 대해 {t}에는 뭐라고 되어 있나요?'),
}
HANGUL = re.compile(r"[가-힣]")


# ---------------------------------------------------------------- text

def strip_prose(text: str) -> str:
    """Markdown prose without fenced blocks, tables or HTML comments."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    out, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced or line.lstrip().startswith("|"):
            continue
        out.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def language(text: str) -> str:
    return "ko" if len(HANGUL.findall(text)) > 50 else "en"


def sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """(document title, [(heading, body)]) split at ## and ### headings."""
    title, current, body, out = "", None, [], []
    for line in text.splitlines():
        m = re.match(r"^(#{1,3}) (.+)$", line)
        if m and len(m.group(1)) == 1 and not title:
            title = m.group(2).strip()
            continue
        if m and len(m.group(1)) > 1:
            if current is not None:
                out.append((current, "\n".join(body).strip()))
            current, body = m.group(2).strip(), []
            continue
        if current is not None:
            body.append(line)
    if current is not None:
        out.append((current, "\n".join(body).strip()))
    return title, [(h, b) for h, b in out if b]


def render_chat(messages: list[dict]) -> str:
    """Qwen3.8 chat framing of a finished conversation, thinking disabled.

    Matches the model's chat_template.jinja with enable_thinking=False:
    every assistant turn carries an empty think block (preserve_thinking is
    the template default). Checked against the template in the tests.
    """
    out = []
    for m in messages:
        content = m["content"].strip()
        if m["role"] == "assistant":
            content = f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n" + content
        out.append(f"{kImStart}{m['role']}\n{content}{kImEnd}\n")
    return "".join(out)


# ---------------------------------------------------------------- build

class Tokenizer:
    def __init__(self, path: Path):
        from tokenizers import Tokenizer as HfTokenizer
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.tokenizer = HfTokenizer.from_file(str(path))
        added = json.loads(path.read_text())["added_tokens"]
        self.added = {t["id"]: t["content"] for t in added}

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def check_added(self, ids: list[int], allowed: set[str], where: str) -> None:
        bad = sorted({self.added[i] for i in ids if i in self.added} - allowed)
        if bad:
            raise ValueError(f"{where}: unexpected control tokens {bad}")


class Repo:
    def __init__(self, path: Path, revision: str):
        self.path = path
        self.revision = self.git("rev-parse", "--verify", revision + "^{commit}").strip()

    def git(self, *args: str) -> str:
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(self.path), *args],
                              check=True, capture_output=True, text=True).stdout

    def read(self, path: str) -> tuple[str, str, str]:
        """(text, blob sha, first-add date) of `path` at the pinned revision."""
        blob = self.git("rev-parse", f"{self.revision}:{path}").strip()
        text = self.git("cat-file", "blob", blob)
        added = self.git("log", "--diff-filter=A", "--format=%ad", "--date=short",
                         self.revision, "--", path).split()
        return text, blob, added[-1] if added else "unknown"


def chat_documents(text: str, doc_cap: int, answer_cap: int, tok: Tokenizer,
                   rotation: int) -> tuple[str, int]:
    """One conversation from a document's sections, within `doc_cap` tokens."""
    title, parts = sections(text)
    lang = language(text)
    messages = [{"role": "system", "content": SYSTEM[lang]}]
    turns = 0
    for heading, body in parts:
        paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
        answer = ""
        for p in paragraphs:
            candidate = (answer + "\n\n" + p).strip()
            if len(tok.encode(candidate)) > answer_cap:
                break
            answer = candidate
        if not answer:
            continue
        question = QUESTIONS[lang][(rotation + turns) % 3].format(h=heading, t=title or "the document")
        trial = messages + [{"role": "user", "content": question},
                            {"role": "assistant", "content": answer}]
        if turns and len(tok.encode(render_chat(trial))) > doc_cap:
            break
        messages, turns = trial, turns + 1
    if turns == 0:
        raise ValueError("document has no section that fits a chat turn")
    return render_chat(messages), turns


def build(repo: Repo, tok: Tokenizer, window_tokens: int, per_category: int,
          doc_tokens: int, answer_tokens: int) -> tuple[list[dict], list[int], list[dict]]:
    windows, tokens, sources = [], [], []
    chat_tokens = {kImStart, kImEnd, THINK_OPEN, THINK_CLOSE}
    for category in CATEGORIES:
        queue = list(SOURCES[category])
        for index in range(per_category):
            window, parts = [], []
            while len(window) < window_tokens:
                if not queue:
                    raise ValueError(f"{category}: sources run out in window {index}")
                path = queue.pop(0)
                text, blob, added = repo.read(path)
                entry = {"category": category, "path": path, "blob": blob,
                         "first_added": added, "license": LICENSE}
                if category == "chat":
                    rendered, turns = chat_documents(text, doc_tokens, answer_tokens, tok,
                                                     len(sources))
                    ids = tok.encode(rendered)
                    tok.check_added(ids, chat_tokens, path)
                    entry.update(transform="chat sections", turns=turns,
                                 language=language(text))
                else:
                    body = strip_prose(text) if category.endswith("prose") else text
                    ids = tok.encode(body.rstrip("\n") + "\n\n")
                    tok.check_added(ids, set(), path)
                    entry["transform"] = ("markdown prose without fences and tables"
                                          if category.endswith("prose") else "verbatim")
                    entry["language"] = language(body)
                total = len(ids)
                ids = ids[:min(doc_tokens, window_tokens - len(window))]
                entry.update(tokens_available=total, tokens_used=len(ids),
                             window=f"{category}-{index}", window_offset=len(window))
                window.extend(ids)
                parts.append(path)
                sources.append(entry)
            windows.append({"id": f"{category}-{index}", "category": category,
                            "offset": len(tokens), "length": len(window), "sources": parts})
            tokens.extend(window)
    return windows, tokens, sources


def write_outputs(out: Path, repo: Repo, tok: Tokenizer, windows: list[dict],
                  tokens: list[int], sources: list[dict], args) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    payload = b"".join(int(t).to_bytes(4, "little", signed=True) for t in tokens)
    (out / "windows.bin").write_bytes(payload)
    index = {"schema": "gufo.rq-corpus-windows.v1", "corpus_version": CORPUS_VERSION,
             "tokens_file": "windows.bin",
             "tokens_sha256": hashlib.sha256(payload).hexdigest(),
             "window_tokens": args.window_tokens, "overlap_tokens": 0,
             "windows": [{k: w[k] for k in ("id", "category", "offset", "length")}
                         for w in windows]}
    index_text = json.dumps(index, indent=1) + "\n"
    (out / "windows.json").write_text(index_text)
    lines = [json.dumps({"id": w["id"], "category": w["category"],
                         "text": tok.decode(tokens[w["offset"]:w["offset"] + w["length"]])},
                        ensure_ascii=False) for w in windows]
    text_jsonl = "\n".join(lines) + "\n"
    (out / "windows_text.jsonl").write_text(text_jsonl)
    per_category = {c: sum(w["length"] for w in windows if w["category"] == c)
                    for c in CATEGORIES}
    manifest = {
        "schema": "gufo.rq-corpus-manifest.v1", "corpus_version": CORPUS_VERSION,
        "repository_revision": repo.revision,
        "tokenizer": {"file": tok.path.name, "sha256": tok.sha256},
        "window_tokens": args.window_tokens, "overlap_tokens": 0,
        "windows_per_category": args.windows_per_category,
        "doc_tokens_cap": args.doc_tokens, "answer_tokens_cap": args.answer_tokens,
        "total_tokens": len(tokens), "tokens_per_category": per_category,
        "scored_positions": sum(w["length"] - 1 for w in windows),
        "outputs": {"windows.bin": index["tokens_sha256"],
                    "windows.json": hashlib.sha256(index_text.encode()).hexdigest(),
                    "windows_text.jsonl": hashlib.sha256(text_jsonl.encode()).hexdigest()},
        "novelty": "Every source was added to this repository on or after 2026-09-16 "
                   "(first_added per source); the repository starts on 2026-08-11, after "
                   "the Qwen3.8 27B training data. Domain: engineering documents and code.",
        "calibration_exclusion": "R3 imatrix calibration must not use any listed path.",
        "windows": windows, "sources": sources,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--repo", type=Path, default=ROOT)
    b.add_argument("--revision", required=True)
    b.add_argument("--tokenizer", type=Path, required=True, help="HF tokenizer.json")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--window-tokens", type=int, default=4096)
    b.add_argument("--windows-per-category", type=int, default=2)
    b.add_argument("--doc-tokens", type=int, default=1536)
    b.add_argument("--answer-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    if args.window_tokens < 2 or args.windows_per_category < 1 or args.doc_tokens < 1:
        parser.error("window, per-category and document sizes must be positive")
    repo = Repo(args.repo, args.revision)
    tok = Tokenizer(args.tokenizer)
    windows, tokens, sources = build(repo, tok, args.window_tokens, args.windows_per_category,
                                     args.doc_tokens, args.answer_tokens)
    manifest = write_outputs(args.out, repo, tok, windows, tokens, sources, args)
    print(json.dumps({k: manifest[k] for k in ("repository_revision", "total_tokens",
                                                "tokens_per_category", "outputs")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
