#!/usr/bin/env python3
"""Build imatrix calibration corpora (RQ R3): fixed token chunks.

Model-independent within the shared Qwen vocabulary: Qwen3.8 27B and
Qwen3.8 Flash-Next have the same tokenizer (ids up to 248076 and every
merge), so one chunks.bin serves both. `build` makes c01; `extend` makes
c02, which keeps c01 as its exact prefix and adds wider domains so the
rarely routed experts of MoE models get enough tokens.

Streams (sources pinned by Hugging Face revision and sha256 in SOURCES):
  v6        bartowski's Qwen3.8-27B calibration v6 text (prose, code and
            tool-use chats already in the model's chat framing), whole file
            without its trailing <|endoftext|> padding.
  ko_wiki   Korean Wikipedia (wikimedia/wikipedia 20231101.ko, one shard).
  ko_web    Korean web text (HuggingFaceFW/fineweb-2 kor_Hang test shard).
  ko_chat   Korean function-calling chats (iknow-lab/hermes-function-calling-
            v1-ko), converted to messages and tools and rendered with the
            model's chat_template.jinja.

The v6 stream sets the size; the Korean streams together take
`--korean-share` of all chunks, `--chat-fraction` of them chat and the rest
prose (wiki and web alike). The tool chats are mostly English template text
and JSON, so the manifest also records the share of Korean tokens. Korean documents are drawn in a seeded order, capped at
`--doc-tokens`, end with <|endoftext|> and are packed so every chunk starts
at a document boundary; the document that overflows a chunk is truncated
there and its rest is dropped. Partial last chunks are padded with
<|endoftext|>. Chunks are interleaved in a seeded order.

None of these sources is a file of this repository, so the corpus shares no
text with the R0 evaluation corpus a01 (repository files only). The build
also checks the content: no `--overlap-tokens`-gram of the chunks may occur
in the `--exclude-windows` token files (a01 windows.bin), or it fails.
Calibration text need not be new to the model; only the evaluation corpus
must be.

Outputs in --out: chunks.bin (little-endian int32 token ids), chunks.json
(index for the gufo imatrix collector), calibration.txt (decoded chunks in
order, for llama-imatrix -c <chunk tokens>; the manifest records whether it
re-tokenizes to exactly chunks.bin) and MANIFEST.json (sources, licenses,
used documents, rejections, sha256 of every output).

Run in the eval image:
  python3 -I tools/rotquant/calibration.py build \
      --sources <dir with SOURCES files> --model-dir /models/<dir> \
      --exclude-windows <a01>/windows.bin --out <dir>
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gufo.control_tokens import kEndOfText, kImEnd, kImStart  # noqa: E402

CORPUS_VERSION = "rq-r3-calibration 1"
EXTEND_VERSION = "rq-r3-calibration 2 (c01 + MoE extension)"
STREAMS = ("v6", "ko_wiki", "ko_web", "ko_chat")

# Stream -> pinned source file, relative to --sources.
SOURCES = {
    "v6": {
        "file": "bartowski-v6/Qwen3.8-27B-calibration-v6.txt",
        "sha256": "b3e9d2371c8931cafa5b6184adcee5cb59adb7f4103e28d0f99eddac1cced4e6",
        "origin": "hf model bartowski/Qwen3.8-27B-GGUF Qwen3.8-27B-calibration-v6.txt",
        "revision": "87410e7b9a5c21e6ca70c0c9cc5237de909fe621",
        "license": "not stated (credits Ed Addario, Kalomaze, Dampf; chats from "
                   "interstellarninja/hermes_reasoning_tool_use); do not redistribute",
    },
    "ko_wiki": {
        "file": "wiki-ko/train-00002-of-00003.parquet",
        "sha256": "5e16f59e25871042fcd2bb1ca8540085e02a4a88177df95447aefe9420e2be64",
        "origin": "hf dataset wikimedia/wikipedia 20231101.ko/train-00002-of-00003.parquet",
        "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "license": "CC BY-SA 3.0 / GFDL",
    },
    "ko_web": {
        "file": "fineweb2-kor/000_00000.parquet",
        "sha256": "65097577414fce64b8aa722d1e223bd80961ad16f8c9f1a1941ce7d47af548a2",
        "origin": "hf dataset HuggingFaceFW/fineweb-2 data/kor_Hang/test/000_00000.parquet",
        "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9",
        "license": "ODC-By 1.0 (Common Crawl terms of use)",
    },
    "ko_chat": {
        "file": "hermes-fc-ko/train-00000-of-00001.parquet",
        "sha256": "0c9d77c4ed94f5131cd9d4b101962010d1bbff34f44764f4a1cbbfce4c512daa",
        "origin": "hf dataset iknow-lab/hermes-function-calling-v1-ko "
                  "data/train-00000-of-00001.parquet (ko_conversations)",
        "revision": "5cd13f24a016d5713814eb89a4205c5869e5d3d7",
        "license": "Apache-2.0",
    },
}

TOOL_SYSTEM = "당신은 함수를 호출할 수 있는 AI 어시스턴트입니다. 필요하면 제공된 함수를 사용해 사용자를 도우세요."
CHAT_TOKENS = {kImStart, kImEnd, kEndOfText, "<think>", "</think>", "<tool_call>",
               "</tool_call>", "<tool_response>", "</tool_response>"}
HANGUL = re.compile(r"[가-힣]")
# Trailing reference sections of Korean Wikipedia articles.
WIKI_TAIL = re.compile(r"^\s*(각주|외부 링크|같이 보기|참고 문헌|참고 자료|출처|내용주)\s*$")
TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
TOOLS = re.compile(r"<tools>(.*?)</tools>", re.S)


# ---------------------------------------------------------------- text

def hangul_ratio(text: str) -> float:
    letters = sum(1 for c in text if not c.isspace())
    return len(HANGUL.findall(text)) / letters if letters else 0.0


def clean_wiki(text: str) -> str:
    """Article text up to its reference sections, blank-line runs collapsed."""
    out = []
    for line in text.splitlines():
        if WIKI_TAIL.match(line):
            break
        out.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def korean_prose(text: str, min_hangul: int) -> bool:
    return len(HANGUL.findall(text)) >= min_hangul and hangul_ratio(text) >= 0.5


def parse_literal(text: str):
    """JSON, else a Python literal (the translated rows mix both)."""
    text = text.strip()
    if text.startswith("\\n"):
        text = text[2:]
    if text.endswith("\\n"):
        text = text[:-2]
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        return ast.literal_eval(text)


def hermes_messages(conversation: list[dict]) -> tuple[list[dict], list[dict] | None]:
    """Messages and tools for the chat template from one Hermes row.

    Raises ValueError for rows whose translated tool markup does not parse.
    """
    messages, tools, names = [], None, set()
    for turn in conversation:
        role, content = turn["role"], turn["content"] or ""
        if role == "system":
            # The prose often names the empty tag pair before the real block.
            found = [b for b in TOOLS.findall(content) if b.strip()]
            if found:
                try:
                    tools = parse_literal(found[-1])
                except (ValueError, SyntaxError) as e:
                    raise ValueError("tools do not parse") from e
                if isinstance(tools, dict):
                    tools = [tools]
                if not isinstance(tools, list) or not tools:
                    raise ValueError("tools are not a list")
                for tool in tools:
                    fn = tool.get("function", tool) if isinstance(tool, dict) else None
                    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                        raise ValueError("tool without a name")
                    names.add(fn["name"])
                content = TOOL_SYSTEM
            messages.append({"role": "system", "content": content.strip()})
        elif role == "user":
            messages.append({"role": "user", "content": content.strip()})
        elif role == "assistant":
            calls = []
            for raw in TOOL_CALL.findall(content):
                try:
                    call = parse_literal(raw)
                except (ValueError, SyntaxError) as e:
                    raise ValueError("tool call does not parse") from e
                if (not isinstance(call, dict) or call.get("name") not in names
                        or not isinstance(call.get("arguments"), dict)):
                    raise ValueError("tool call without a known name and arguments")
                calls.append({"type": "function", "function": {
                    "name": call["name"], "arguments": call["arguments"]}})
            message = {"role": "assistant", "content": TOOL_CALL.sub("", content).strip()}
            if calls:
                message["tool_calls"] = calls
            elif "<tool_call>" in content or "</tool_call>" in content:
                raise ValueError("unbalanced tool call tags")
            messages.append(message)
        elif role == "tool":
            body = re.sub(r"^\s*<tool_response>|</tool_response>\s*$", "", content.strip())
            messages.append({"role": "tool", "content": body.strip()})
        else:
            raise ValueError(f"unknown role {role}")
    roles = [m["role"] for m in messages]
    if "assistant" not in roles or roles[-1] != "assistant":
        raise ValueError("conversation does not end with an assistant turn")
    if any(m["role"] == "tool" for m in messages) and tools is None:
        raise ValueError("tool turns without tools")
    return messages, tools


class Tokenizer:
    """HF tokenizer.json, as tools/qwen27b/rq_quality/corpus.py reads it."""

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


# ---------------------------------------------------------------- packing

def plan(v6_chunks: int, share: float, chat_fraction: float) -> dict[str, int]:
    """Chunks per stream: Korean takes `share` of the total."""
    korean = round(v6_chunks * share / (1.0 - share))
    chat = round(korean * chat_fraction)
    wiki = (korean - chat) // 2
    return {"v6": v6_chunks, "ko_wiki": wiki, "ko_web": korean - chat - wiki,
            "ko_chat": chat}


def split_stream(ids: list[int], chunk: int, pad: int) -> list[list[int]]:
    """Fixed chunks of one continuous stream; the last one padded."""
    out = [ids[i:i + chunk] for i in range(0, len(ids), chunk)]
    if out and len(out[-1]) < chunk:
        out[-1] = out[-1] + [pad] * (chunk - len(out[-1]))
    return out


def pack(documents, count: int, chunk: int, doc_cap: int, pad: int,
         span: bool = False, sep: list[int] | None = None):
    """`count` chunks of documents (iterable of (id, token ids)).

    Each document is capped at `doc_cap` tokens and ends with `sep` (default
    `pad`). Without `span` every chunk starts at a document boundary and the
    rest of an overflowing document is dropped; with `span` the document
    continues in the next chunks. Returns (chunks, used entries).
    """
    sep = [pad] if sep is None else sep
    chunks, used, current = [], [], []
    for doc_id, ids in documents:
        if len(chunks) == count:
            break
        ids = ids[:doc_cap - len(sep)] + sep
        entry = {"id": doc_id, "chunk": len(chunks), "chunk_offset": len(current),
                 "tokens_available": len(ids), "tokens_used": 0}
        used.append(entry)
        while ids and len(chunks) < count:
            take = ids[:chunk - len(current)]
            current.extend(take)
            entry["tokens_used"] += len(take)
            ids = ids[len(take):]
            if len(current) == chunk:
                chunks.append(current)
                current = []
            if not span:
                break
    if len(chunks) < count:
        raise ValueError(f"documents run out after {len(chunks)} of {count} chunks")
    return chunks, used


# ---------------------------------------------------------------- build

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sources(directory: Path, sources: dict = SOURCES) -> dict[str, Path]:
    paths = {}
    for stream, source in sources.items():
        path = directory / source["file"]
        actual = sha256_file(path)
        if actual != source["sha256"]:
            raise ValueError(f"{path}: sha256 {actual}, expected {source['sha256']}")
        paths[stream] = path
    return paths


def shuffled(rows: list, seed: int, stream: str) -> list:
    order = list(range(len(rows)))
    random.Random(f"{seed}:{stream}").shuffle(order)
    return [rows[i] for i in order]


class Renderer:
    """The model's own chat template, through transformers."""

    def __init__(self, model_dir: Path, template: Path | None = None):
        import transformers
        self.version = transformers.__version__
        self.template = template or model_dir / "chat_template.jinja"
        self.sha256 = sha256_file(self.template)
        self.text = self.template.read_text()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir))

    def __call__(self, messages: list[dict], tools: list[dict] | None) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False,
            chat_template=self.text)


def prose_documents(rows, stream: str, tok: Tokenizer, rejected: dict, min_hangul: int):
    for row in rows:
        text = clean_wiki(row["text"]) if stream == "ko_wiki" else row["text"].strip()
        if stream == "ko_web" and row["language_score"] < 0.95:
            rejected["language score"] = rejected.get("language score", 0) + 1
            continue
        if not korean_prose(text, min_hangul):
            rejected["too little Korean"] = rejected.get("too little Korean", 0) + 1
            continue
        ids = tok.encode(text + "\n")
        tok.check_added(ids, set(), f"{stream} {row['id']}")
        yield row["id"], ids


def chat_documents(rows, tok: Tokenizer, render: Renderer, rejected: dict):
    for row in rows:
        try:
            messages, tools = hermes_messages(row["ko_conversations"])
        except ValueError as e:
            rejected[str(e)] = rejected.get(str(e), 0) + 1
            continue
        text = render(messages, tools)
        ids = tok.encode(text)
        try:
            tok.check_added(ids, CHAT_TOKENS, f"ko_chat {row['id']}")
        except ValueError:
            rejected["control tokens in the text"] = rejected.get(
                "control tokens in the text", 0) + 1
            continue
        yield row["id"], ids


def build(paths: dict[str, Path], tok: Tokenizer, render: Renderer, args) -> dict:
    import pyarrow.parquet as pq
    pad = tok.tokenizer.token_to_id(kEndOfText)
    v6_text = paths["v6"].read_text(encoding="utf-8")
    v6_body = v6_text.rstrip()
    stripped = 0
    while v6_body.endswith(kEndOfText):
        v6_body, stripped = v6_body[:-len(kEndOfText)].rstrip(), stripped + 1
    v6_ids = tok.encode(v6_body + "\n")
    streams = {"v6": split_stream(v6_ids, args.chunk_tokens, pad)}
    counts = plan(len(streams["v6"]), args.korean_share, args.chat_fraction)
    stats = {"v6": {"tokens": len(v6_ids), "trailing_endoftext_stripped": stripped,
                    "control_tokens": {t: v6_body.count(t) for t in sorted(CHAT_TOKENS)}}}
    for stream in ("ko_wiki", "ko_web", "ko_chat"):
        columns = ["id", "ko_conversations"] if stream == "ko_chat" else (
            ["id", "text"] + (["language_score"] if stream == "ko_web" else []))
        rows = shuffled(pq.read_table(paths[stream], columns=columns).to_pylist(),
                        args.seed, stream)
        rejected: dict[str, int] = {}
        docs = (chat_documents(rows, tok, render, rejected) if stream == "ko_chat" else
                prose_documents(rows, stream, tok, rejected, args.min_hangul))
        streams[stream], used = pack(docs, counts[stream], args.chunk_tokens,
                                     args.doc_tokens, pad)
        stats[stream] = {"rows": len(rows), "documents_used": len(used),
                         "documents_rejected": dict(sorted(rejected.items())),
                         "documents": used}
    order = [(s, i) for s in STREAMS for i in range(len(streams[s]))]
    random.Random(f"{args.seed}:order").shuffle(order)
    return {"streams": streams, "counts": counts, "stats": stats, "order": order}


def read_tokens(path: Path) -> list[int]:
    data = path.read_bytes()
    return [int.from_bytes(data[i:i + 4], "little", signed=True)
            for i in range(0, len(data), 4)]


def overlap(chunks: list[list[int]], excluded: list[int], n: int) -> int:
    """Chunk n-grams that also occur in `excluded` (as one token stream)."""
    grams = {tuple(excluded[i:i + n]) for i in range(len(excluded) - n + 1)}
    return sum(tuple(c[i:i + n]) in grams for c in chunks for i in range(len(c) - n + 1))


def write_outputs(out: Path, tok: Tokenizer, render: Renderer, built: dict, args) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    tokens, chunks = [], []
    for stream, index in built["order"]:
        chunks.append({"id": f"{stream}-{index}", "stream": stream,
                       "offset": len(tokens), "length": args.chunk_tokens})
        tokens.extend(built["streams"][stream][index])
    payload = b"".join(int(t).to_bytes(4, "little", signed=True) for t in tokens)
    (out / "chunks.bin").write_bytes(payload)
    index = {"schema": "gufo.rq-calibration-chunks.v1", "corpus_version": CORPUS_VERSION,
             "tokens_file": "chunks.bin",
             "tokens_sha256": hashlib.sha256(payload).hexdigest(),
             "chunk_tokens": args.chunk_tokens, "chunks": chunks}
    index_text = json.dumps(index, indent=1) + "\n"
    (out / "chunks.json").write_text(index_text)
    text = tok.decode(tokens)
    (out / "calibration.txt").write_text(text, encoding="utf-8")
    # A token counts as Korean when it decodes to Hangul or to a partial
    # UTF-8 sequence (multi-token syllables).
    korean_ids = {}
    for i in set(tokens):
        piece = tok.decode([i])
        korean_ids[i] = bool(HANGUL.search(piece)) or "\ufffd" in piece
    for stream, stat in built["stats"].items():
        ids = [t for c in built["streams"][stream] for t in c]
        stat["hangul_ratio"] = hangul_ratio(tok.decode(ids))
        stat["korean_token_share"] = sum(korean_ids[t] for t in ids) / len(ids)
    counts = built["counts"]
    total = sum(counts.values())
    korean = total - counts["v6"]
    manifest = {
        "schema": "gufo.rq-calibration-manifest.v1", "corpus_version": CORPUS_VERSION,
        "tokenizer": {"file": tok.path.name, "sha256": tok.sha256},
        "chat_template": {"file": render.template.name, "sha256": render.sha256,
                          "renderer": f"transformers {render.version} apply_chat_template"},
        "seed": args.seed, "chunk_tokens": args.chunk_tokens,
        "doc_tokens_cap": args.doc_tokens, "min_hangul": args.min_hangul,
        "korean_share_requested": args.korean_share,
        "korean_share": korean / total,
        "chat_fraction": args.chat_fraction,
        "korean_token_share": sum(korean_ids[t] for t in tokens) / len(tokens),
        "chunks": total, "total_tokens": len(tokens), "chunks_per_stream": counts,
        "text_roundtrip_exact": tok.encode(text) == tokens,
        "outputs": {"chunks.bin": index["tokens_sha256"],
                    "chunks.json": hashlib.sha256(index_text.encode()).hexdigest(),
                    "calibration.txt": hashlib.sha256(text.encode("utf-8")).hexdigest()},
        "evaluation_exclusion": {
            "rule": "No source is a file of this repository, so no text is shared with "
                    "the R0 evaluation corpus a01 (repository files only).",
            "content_check": built["exclusion"]},
        "sources": SOURCES, "streams": built["stats"],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return manifest


# ---------------------------------------------------------------- extend (c02)
#
# c02 = an earlier corpus (c01) as an exact prefix, plus chunks from wider
# domains for MoE models, whose rarely routed experts need more tokens. The
# Korean streams reuse the c01 shards and skip every document c01 used.

EXTEND_SOURCES = {
    "ko_wiki": SOURCES["ko_wiki"],
    "ko_web": SOURCES["ko_web"],
    "ko_chat": SOURCES["ko_chat"],
    "en_tool": {
        "file": "hermes-reasoning-tool-use/train-00000-of-00001.parquet",
        "sha256": "b3730a6ebcd0cf8c9ca679093db2811ce8a66406fa3244d55a7c40fe539915b1",
        "origin": "hf dataset interstellarninja/hermes_reasoning_tool_use "
                  "data/train-00000-of-00001.parquet",
        "revision": "c27e454925db6295b160d20ad8a756aa75bfdb77",
        "license": "Apache-2.0",
    },
    "en_chat": {
        "file": "ultrachat-200k/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
        "sha256": "c18fd6e77395577652bdefbc5a87044ea799e989451ecfca5c6cf977ae5c6f70",
        "origin": "hf dataset HuggingFaceH4/ultrachat_200k "
                  "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
        "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
        "license": "MIT",
    },
    "code": {
        "file": "eaddario/code_small.parquet",
        "sha256": "8dc7f124120c2ae369261a77ef5be655783803feb907ea66493c7066f8aeb8a6",
        "origin": "hf dataset eaddario/imatrix-calibration code_small.parquet (Open-Critic-GPT, "
                  "opc-sft-stage2, Magicoder-Evol-Instruct-110K, McEval-Instruct)",
        "revision": "e87ed55dcba9d9c3a3e41539f3e728e981b1daa4",
        "license": "MIT (dataset); upstream sets carry their own terms",
    },
    "math": {
        "file": "eaddario/math_small.parquet",
        "sha256": "edad29f9673837cf7ec43e8161a0b3a08125b99ff9cc5c1434cf32b85e6f510d",
        "origin": "hf dataset eaddario/imatrix-calibration math_small.parquet "
                  "(nvidia/OpenMathInstruct-2)",
        "revision": "e87ed55dcba9d9c3a3e41539f3e728e981b1daa4",
        "license": "MIT (dataset); OpenMathInstruct-2 is CC BY 4.0",
    },
    "multi": {
        "file": "eaddario/text_all_small.parquet",
        "sha256": "e56a0cb67c9d7eff69138479f987473a606f7cbb3bac41ff7785287efd5adcba",
        "origin": "hf dataset eaddario/imatrix-calibration text_all_small.parquet "
                  "(fineweb, fineweb-2; 18 languages)",
        "revision": "e87ed55dcba9d9c3a3e41539f3e728e981b1daa4",
        "license": "MIT (dataset); fineweb sources are ODC-By 1.0",
    },
}
# Stream -> share of the added chunks. Korean prose and chat keep the Korean
# token share near c01's (about 20 %); en_tool and ko_chat keep the share of
# chunks with tool calls near c01's (about 36 %).
EXTEND_MIX = {"ko_wiki": 0.12, "ko_web": 0.10, "ko_chat": 0.08, "en_tool": 0.45,
              "en_chat": 0.09, "code": 0.09, "math": 0.04, "multi": 0.03}
CHAT_STREAMS = ("ko_chat", "en_tool", "en_chat")
LINE_STREAMS = ("code", "math", "multi")
THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)


def mix_counts(total: int, mix: dict[str, float]) -> dict[str, int]:
    """Largest-remainder split of `total` chunks by `mix`, in mix order."""
    raw = {k: total * v / sum(mix.values()) for k, v in mix.items()}
    counts = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[:total - sum(counts.values())]:
        counts[k] += 1
    return counts


def reasoning_tool_messages(row: dict) -> tuple[list[dict], list[dict] | None]:
    """Messages and tools from a hermes_reasoning_tool_use row.

    Like bartowski's v6 rendering: the Hermes system prompt (which restates
    the tools and a JSON call format) is replaced by the template's tool
    section, <think> becomes reasoning_content, and a trailing tool turn
    without an answer is dropped.
    """
    try:
        tools = json.loads(row["tools"]) if row["tools"] else None
    except ValueError as e:
        raise ValueError("tools do not parse") from e
    if tools is not None:
        if not isinstance(tools, list) or not all(
                isinstance(t, dict) and isinstance(t.get("name"), str) for t in tools):
            raise ValueError("tools are not named objects")
        tools = [{"type": "function", "function": t} for t in tools]
    names = {t["function"]["name"] for t in tools or []}
    messages = []
    for turn in row["conversations"]:
        role, content = turn["from"], turn["value"] or ""
        if role == "system":
            continue
        if role == "human":
            messages.append({"role": "user", "content": content.strip()})
        elif role == "gpt":
            message = {"role": "assistant"}
            think = THINK.match(content)
            if think:
                message["reasoning_content"] = think.group(1).strip()
                content = content[think.end():]
            calls = []
            for raw in TOOL_CALL.findall(content):
                try:
                    call = parse_literal(raw)
                except (ValueError, SyntaxError) as e:
                    raise ValueError("tool call does not parse") from e
                if (not isinstance(call, dict) or call.get("name") not in names
                        or not isinstance(call.get("arguments"), dict)):
                    raise ValueError("tool call without a known name and arguments")
                calls.append({"type": "function", "function": {
                    "name": call["name"], "arguments": call["arguments"]}})
            message["content"] = TOOL_CALL.sub("", content).strip()
            if "<tool_call>" in message["content"] or "<think>" in message["content"]:
                raise ValueError("stray markup in the answer")
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
        elif role == "tool":
            body = re.sub(r"^\s*<tool_response>|</tool_response>\s*$", "", content.strip())
            messages.append({"role": "tool", "content": body.strip()})
        else:
            raise ValueError(f"unknown role {role}")
    while messages and messages[-1]["role"] == "tool":
        messages.pop()
    if not messages or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
        raise ValueError("conversation is not user ... assistant")
    return messages, tools


def ultrachat_messages(row: dict) -> tuple[list[dict], None]:
    messages = [{"role": m["role"], "content": m["content"].strip()} for m in row["messages"]]
    if (not messages or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant"
            or any(m["role"] not in ("user", "assistant") for m in messages)):
        raise ValueError("conversation is not user ... assistant")
    return messages, None


def source_lines(text: str, stream: str) -> list[str]:
    """One sample per line; the code file escapes newlines inside samples."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if stream == "code":
        lines = [line.replace("\\n", "\n").replace("\\t", "\t") for line in lines]
    return lines


def count(rejected: dict, reason: str) -> None:
    rejected[reason] = rejected.get(reason, 0) + 1


def extend_chats(rows, stream: str, tok: Tokenizer, renders: list[Renderer],
                 rejected: dict, skip: set, v6_text: str):
    convert = {"ko_chat": lambda r: hermes_messages(r["ko_conversations"]),
               "en_tool": reasoning_tool_messages, "en_chat": ultrachat_messages}[stream]
    for row in rows:
        doc_id = row.get("id") or row.get("prompt_id")
        if doc_id in skip:
            count(rejected, "used by the base corpus")
            continue
        try:
            messages, tools = convert(row)
        except ValueError as e:
            count(rejected, str(e))
            continue
        if stream == "en_tool" and messages[0]["content"][:160] in v6_text:
            count(rejected, "first user turn is in v6")
            continue
        try:
            texts = [render(messages, tools) for render in renders]
        except Exception as e:  # noqa: BLE001 - template errors reject the row
            count(rejected, f"template error {type(e).__name__}")
            continue
        if any(t != texts[0] for t in texts[1:]):
            count(rejected, "templates render differently")
            continue
        ids = tok.encode(texts[0])
        try:
            tok.check_added(ids, CHAT_TOKENS, f"{stream} {doc_id}")
        except ValueError:
            count(rejected, "control tokens in the text")
            continue
        yield doc_id, ids


def extend_lines(lines: list[str], stream: str, tok: Tokenizer, rejected: dict):
    for index, line in lines:
        ids = tok.encode(line)
        try:
            tok.check_added(ids, set(), f"{stream} line {index}")
        except ValueError:
            count(rejected, "control tokens in the text")
            continue
        yield f"line {index}", ids


def extend(paths: dict[str, Path], base: dict, tok: Tokenizer, renders: list[Renderer],
           v6_text: str, args) -> dict:
    import pyarrow.parquet as pq
    pad = tok.tokenizer.token_to_id(kEndOfText)
    counts = mix_counts(args.add_chunks, EXTEND_MIX)
    streams, stats = {}, {}
    for stream in EXTEND_MIX:
        rejected: dict[str, int] = {}
        skip = {d["id"] for d in base["manifest"]["streams"].get(stream, {}).get("documents", [])}
        if stream in LINE_STREAMS:
            text = pq.read_table(paths[stream], columns=["content"]).column("content")[0].as_py()
            lines = shuffled(list(enumerate(source_lines(text, stream))), args.seed, stream)
            rows = len(lines)
            docs = extend_lines(lines, stream, tok, rejected)
            span, cap, sep = False, args.doc_tokens, tok.encode("\n")
        else:
            columns = {"ko_wiki": ["id", "text"], "ko_web": ["id", "text", "language_score"],
                       "ko_chat": ["id", "ko_conversations"], "en_tool": ["conversations", "tools"],
                       "en_chat": ["prompt_id", "messages"]}[stream]
            table = pq.read_table(paths[stream], columns=columns).to_pylist()
            if stream == "en_tool":
                for i, row in enumerate(table):
                    row["id"] = f"row {i}"
            table = shuffled(table, args.seed, f"extend:{stream}")
            rows = len(table)
            if stream in CHAT_STREAMS:
                docs = extend_chats(table, stream, tok, renders, rejected, skip, v6_text)
                span, cap, sep = True, args.chat_doc_tokens, None
            else:
                table = [r for r in table if r["id"] not in skip]
                rejected["used by the base corpus"] = rows - len(table)
                docs = prose_documents(table, stream, tok, rejected, args.min_hangul)
                span, cap, sep = False, args.doc_tokens, None
        streams[stream], used = pack(docs, counts[stream], args.chunk_tokens, cap, pad,
                                     span=span, sep=sep)
        stats[stream] = {"rows": rows, "documents_used": len(used),
                         "documents_rejected": dict(sorted(rejected.items())),
                         "doc_tokens_cap": cap, "documents_span_chunks": span,
                         "documents": used}
    order = [(s, i) for s in EXTEND_MIX for i in range(len(streams[s]))]
    random.Random(f"{args.seed}:extend-order").shuffle(order)
    return {"streams": streams, "counts": counts, "stats": stats, "order": order}


def load_base(directory: Path) -> dict:
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    index_text = (directory / "chunks.json").read_text()
    payload = (directory / "chunks.bin").read_bytes()
    for name, data in (("chunks.bin", payload), ("chunks.json", index_text.encode())):
        if hashlib.sha256(data).hexdigest() != manifest["outputs"][name]:
            raise ValueError(f"{directory / name}: sha256 differs from its manifest")
    return {"manifest": manifest, "index": json.loads(index_text), "payload": payload,
            "manifest_sha256": sha256_file(directory / "MANIFEST.json")}


def token_stats(tok: Tokenizer, tokens: list[int]) -> tuple[dict[int, bool], int]:
    korean = {}
    for i in set(tokens):
        piece = tok.decode([i])
        korean[i] = bool(HANGUL.search(piece)) or "�" in piece
    return korean, tok.tokenizer.token_to_id("<tool_call>")


def write_extended(out: Path, tok: Tokenizer, renders: list[Renderer], base: dict,
                   built: dict, args) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    chunk = args.chunk_tokens
    base_tokens = read_tokens_bytes(base["payload"])
    chunks = [dict(c) for c in base["index"]["chunks"]]
    tokens = list(base_tokens)
    for stream, index in built["order"]:
        chunks.append({"id": f"{stream}-x{index}", "stream": stream,
                       "offset": len(tokens), "length": chunk})
        tokens.extend(built["streams"][stream][index])
    payload = b"".join(int(t).to_bytes(4, "little", signed=True) for t in tokens)
    if payload[:len(base["payload"])] != base["payload"]:
        raise AssertionError("base corpus is not an exact prefix")
    (out / "chunks.bin").write_bytes(payload)
    index = {"schema": "gufo.rq-calibration-chunks.v1", "corpus_version": EXTEND_VERSION,
             "tokens_file": "chunks.bin", "tokens_sha256": hashlib.sha256(payload).hexdigest(),
             "chunk_tokens": chunk, "chunks": chunks}
    index_text = json.dumps(index, indent=1) + "\n"
    (out / "chunks.json").write_text(index_text)
    text = tok.decode(tokens)
    (out / "calibration.txt").write_text(text, encoding="utf-8")
    korean, tool_call = token_stats(tok, tokens)
    by_stream: dict[str, list[list[int]]] = {}
    for c in chunks:
        by_stream.setdefault(c["stream"], []).append(tokens[c["offset"]:c["offset"] + chunk])
    streams = {}
    for stream, rows in by_stream.items():
        flat = [t for r in rows for t in r]
        stat = dict(built["stats"].get(stream, {}))
        stat.update(chunks=len(rows), korean_token_share=sum(korean[t] for t in flat) / len(flat),
                    tool_call_chunks=sum(tool_call in r for r in rows))
        streams[stream] = stat
    added = len(chunks) - len(base["index"]["chunks"])
    base_m = base["manifest"]
    manifest = {
        "schema": "gufo.rq-calibration-manifest.v1", "corpus_version": EXTEND_VERSION,
        "tokenizer": {"file": tok.path.name, "sha256": tok.sha256},
        "chat_templates": [{"file": str(r.template.name), "sha256": r.sha256,
                            "renderer": f"transformers {r.version} apply_chat_template"}
                           for r in renders],
        "chat_template_rule": "Every added chat renders to the same text with each listed "
                              "template, or it is rejected (count under documents_rejected).",
        "base": {"corpus_version": base_m["corpus_version"], "chunks": len(base["index"]["chunks"]),
                 "manifest_sha256": base["manifest_sha256"],
                 "chunks_bin_sha256": base_m["outputs"]["chunks.bin"],
                 "exact_prefix": "chunks 0..%d of this corpus are the base chunks.bin byte for "
                                 "byte (tokens 0..%d), in the same order with the same ids"
                                 % (len(base["index"]["chunks"]) - 1, len(base_tokens) - 1)},
        "seed": args.seed, "chunk_tokens": chunk, "doc_tokens_cap": args.doc_tokens,
        "chat_doc_tokens_cap": args.chat_doc_tokens, "min_hangul": args.min_hangul,
        "mix": EXTEND_MIX, "chunks": len(chunks), "added_chunks": added,
        "total_tokens": len(tokens), "chunks_per_stream": {s: len(r) for s, r in by_stream.items()},
        "korean_token_share": sum(korean[t] for t in tokens) / len(tokens),
        "tool_call_chunk_share": sum(tool_call in r for rows in by_stream.values()
                                     for r in rows) / len(chunks),
        "text_roundtrip_exact": tok.encode(text) == tokens,
        "outputs": {"chunks.bin": index["tokens_sha256"],
                    "chunks.json": hashlib.sha256(index_text.encode()).hexdigest(),
                    "calibration.txt": hashlib.sha256(text.encode("utf-8")).hexdigest()},
        "evaluation_exclusion": {
            "rule": "No source is a file of this repository, so no text is shared with "
                    "the R0 evaluation corpus a01 (repository files only).",
            "content_check": built["exclusion"]},
        "sources": {"base": base_m["sources"], "added": EXTEND_SOURCES},
        "source_list_note": "Every source of every chunk (base and added); an evaluation "
                            "corpus for Flash-Next must avoid all of them.",
        "streams": streams,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return manifest


def read_tokens_bytes(data: bytes) -> list[int]:
    return [int.from_bytes(data[i:i + 4], "little", signed=True) for i in range(0, len(data), 4)]


def run_extend(args) -> int:
    base = load_base(args.base)
    if base["index"]["chunk_tokens"] != args.chunk_tokens:
        raise SystemExit("--chunk-tokens differs from the base corpus")
    paths = check_sources(args.sources, EXTEND_SOURCES)
    v6 = check_sources(args.sources, {"v6": SOURCES["v6"]})["v6"]
    tok = Tokenizer(args.model_dir / "tokenizer.json")
    if tok.sha256 != base["manifest"]["tokenizer"]["sha256"]:
        raise SystemExit("tokenizer differs from the base corpus")
    renders = [Renderer(args.model_dir)] + [Renderer(args.model_dir, t) for t in args.alt_template]
    built = extend(paths, base, tok, renders, v6.read_text(encoding="utf-8"), args)
    added = [c for s in EXTEND_MIX for c in built["streams"][s]]
    built["exclusion"] = []
    for path in args.exclude_windows:
        hits = overlap(added, read_tokens(path), args.overlap_tokens)
        built["exclusion"].append({"file": path.name, "sha256": sha256_file(path),
                                   "ngram_tokens": args.overlap_tokens, "shared_ngrams": hits,
                                   "scope": "added chunks; the base corpus checked itself"})
        if hits:
            raise SystemExit(f"{path}: {hits} shared {args.overlap_tokens}-token n-grams")
    base_checks = {c["sha256"]: c for c in
                   base["manifest"]["evaluation_exclusion"]["content_check"]}
    for check in built["exclusion"]:
        prior = base_checks.get(check["sha256"])
        if not prior or prior["shared_ngrams"] or prior["ngram_tokens"] != check["ngram_tokens"]:
            raise SystemExit(f"base corpus has no clean check against {check['file']}")
    manifest = write_extended(args.out, tok, renders, base, built, args)
    print(json.dumps({k: manifest[k] for k in (
        "chunks", "added_chunks", "total_tokens", "chunks_per_stream", "korean_token_share",
        "tool_call_chunk_share", "evaluation_exclusion", "outputs")}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources", type=Path, required=True,
                   help="directory holding the SOURCES files")
    b.add_argument("--model-dir", type=Path, required=True,
                   help="HF model directory with tokenizer.json and chat_template.jinja")
    b.add_argument("--exclude-windows", type=Path, action="append", required=True,
                   help="evaluation token file (int32) the chunks must not overlap")
    b.add_argument("--overlap-tokens", type=int, default=16)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--korean-share", type=float, default=0.3)
    b.add_argument("--chat-fraction", type=float, default=0.25)
    b.add_argument("--chunk-tokens", type=int, default=512)
    b.add_argument("--doc-tokens", type=int, default=512)
    b.add_argument("--min-hangul", type=int, default=200)
    b.add_argument("--seed", type=int, default=20261029)
    x = sub.add_parser("extend", help="c02: a base corpus as exact prefix plus wider domains")
    x.add_argument("--base", type=Path, required=True, help="base corpus directory (c01)")
    x.add_argument("--sources", type=Path, required=True)
    x.add_argument("--model-dir", type=Path, required=True)
    x.add_argument("--alt-template", type=Path, action="append", default=[],
                   help="another model's chat template that must render the same text")
    x.add_argument("--exclude-windows", type=Path, action="append", required=True)
    x.add_argument("--overlap-tokens", type=int, default=16)
    x.add_argument("--out", type=Path, required=True)
    x.add_argument("--add-chunks", type=int, default=3073)
    x.add_argument("--chunk-tokens", type=int, default=512)
    x.add_argument("--doc-tokens", type=int, default=512)
    x.add_argument("--chat-doc-tokens", type=int, default=4096)
    x.add_argument("--min-hangul", type=int, default=200)
    x.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args(argv)
    if args.command == "extend":
        return run_extend(args)
    if not 0.0 <= args.korean_share < 1.0:
        parser.error("--korean-share must be in [0, 1)")
    if not 0.0 <= args.chat_fraction <= 1.0:
        parser.error("--chat-fraction must be in [0, 1]")
    if args.chunk_tokens < 2 or args.doc_tokens < 2:
        parser.error("chunk and document sizes must be at least 2")
    paths = check_sources(args.sources)
    tok = Tokenizer(args.model_dir / "tokenizer.json")
    render = Renderer(args.model_dir)
    built = build(paths, tok, render, args)
    chunks = [c for s in STREAMS for c in built["streams"][s]]
    built["exclusion"] = []
    for path in args.exclude_windows:
        hits = overlap(chunks, read_tokens(path), args.overlap_tokens)
        built["exclusion"].append({"file": path.name, "sha256": sha256_file(path),
                                   "ngram_tokens": args.overlap_tokens, "shared_ngrams": hits})
        if hits:
            raise SystemExit(f"{path}: {hits} shared {args.overlap_tokens}-token n-grams")
    manifest = write_outputs(args.out, tok, render, built, args)
    print(json.dumps({k: manifest[k] for k in (
        "chunks", "total_tokens", "chunks_per_stream", "korean_share", "korean_token_share",
        "evaluation_exclusion",
        "text_roundtrip_exact", "outputs")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
