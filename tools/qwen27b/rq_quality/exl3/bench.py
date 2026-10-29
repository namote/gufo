#!/usr/bin/env python3
"""W8 ExLlamaV3 speed measurements on gfx1151 (plan/rotated-quant/W8-BRIEF.md "속도 측정").

Modes (one model load per call):

- `pptg`: prompt processing and plain decode with corpus a01 tokens, like `gufo bench -p 2048 -n 128
  -d 0,32768 -r 3`: pp = one `prefill` of `--pp` tokens after `depth` cached tokens, tg = `--tg`
  single-token forwards after `depth` cached tokens (argmax, synchronized each step), each repeated
  `--reps` times after one warm-up; tok/s per repetition is reported.
- `chat`: the 64 RQ acceptance prompts (tools/qwen27b/rq_quality/accept_prompts.json) through the
  ExLlamaV3 Generator, greedy, thinking off, `--max-tokens` new tokens, one warm-up prompt and
  `--reps` passes; `--draft mtp` drafts with the model's own MTP head (the gufo DFlash2 comparison),
  `--draft none` is plain decoding. Per prompt: new tokens, generate time, accepted and rejected
  draft tokens, token sha256 (repetitions must match). Summary: decode tok/s = sum tokens / sum
  generate time, acceptance = accepted / (accepted + rejected), accepted per output token.
- `kernels`: a short fixed workload (prefill 512, 16 decode steps) for `rocprofv3 --kernel-trace`.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np

PAGE = 256


def corpus_tokens(windows_json: Path) -> np.ndarray:
    index = json.loads(windows_json.read_text())
    raw = (windows_json.parent / index["tokens_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != index["tokens_sha256"]:
        raise SystemExit("corpus sha256 mismatch")
    return np.frombuffer(raw, dtype="<i4").astype(np.int64)


def need_tokens(tokens: np.ndarray, n: int) -> np.ndarray:
    reps = -(-n // len(tokens))
    return np.tile(tokens, reps)[:n]


def env_report(torch) -> dict:
    return dict(torch=torch.__version__, hip=torch.version.hip, exllamav3=importlib.metadata.version("exllamav3"),
                device=torch.cuda.get_device_properties(0).gcnArchName)


def mode_pptg(a, torch) -> dict:
    from exllamav3 import Cache, Config, Model
    from exllamav3.cache import CacheLayer_fp16
    toks = corpus_tokens(a.windows)
    depths = [int(x) for x in a.depths.split(",")]
    cap = -(-(max(depths) + max(a.pp, a.tg) + 1) // PAGE) * PAGE
    cap = max(cap, 4096)
    t0 = time.perf_counter()
    config = Config.from_directory(a.model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=cap, layer_type=CacheLayer_fp16, max_batch_size=1)
    # --fill-single prefills the depth context in one chunk from an empty cache: upstream 1.6.5's
    # paged prefill kernel with cached past context asks for 128 KB of LDS at head_dim 256 on
    # gfx1151 (64 KB) and fails (w8sp-a01), while a prefill from position 0 runs.
    load_chunk = max(a.chunk, max(depths)) if a.fill_single else a.chunk
    model.load(progressbar=False, max_chunk_size=load_chunk, max_batch_size=1)
    load_s = time.perf_counter() - t0
    ids = torch.as_tensor(need_tokens(toks, cap)[None])
    kinds = a.kinds.split(",")
    out = dict(mode="pptg", load_s=load_s, cache_tokens=cap, pp=a.pp, tg=a.tg, chunk=a.chunk,
               fill_single=bool(a.fill_single), kinds=kinds, results=[])

    def fill(state, depth):
        base = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, cap), "recurrent_states": [state]}
        step = depth if a.fill_single else a.chunk
        for s in range(0, depth, max(step, 1)):
            e = min(depth, s + step)
            model.prefill(ids[:, s:e], dict(base, past_len=s))
        return base

    for depth in depths:
        for kind in kinds:
            rates = []
            for rep in range(a.reps + 1):  # first is warm-up
                state = cache.get_new_state()
                base = fill(state, depth)
                torch.cuda.synchronize()
                t = time.perf_counter()
                if kind == "pp":
                    for s in range(depth, depth + a.pp, a.chunk):
                        e = min(depth + a.pp, s + a.chunk)
                        model.prefill(ids[:, s:e], dict(base, past_len=s))
                    torch.cuda.synchronize()
                    n = a.pp
                else:
                    for i in range(a.tg):
                        logits = model.forward(ids[:, depth + i:depth + i + 1], dict(base, past_len=depth + i))
                        int(torch.argmax(logits[0, -1]).item())  # host sync per token, as a generator would
                    n = a.tg
                dt = time.perf_counter() - t
                cache.release_state(state)
                if rep:
                    rates.append(n / dt)
            out["results"].append(dict(kind=kind, depth=depth, tok_s=rates, mean=float(np.mean(rates)),
                                       spread=float((max(rates) - min(rates)) / np.mean(rates))))
            print(f"{kind} depth {depth}: {np.mean(rates):.2f} tok/s {['%.2f' % r for r in rates]}", flush=True)
    return out


def mode_chat(a, torch) -> dict:
    from exllamav3 import Generator, GreedySampler, Job, model_init
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache=True, default_cache_size=a.cache_tokens, add_draft_model_args=True)
    argv = ["-m", a.model_dir, "-cs", str(a.cache_tokens)]
    if a.draft == "mtp":
        argv += ["-mtp"]
        if a.num_draft_tokens:
            argv += ["-ndt", str(a.num_draft_tokens)]
        if a.dynamic_draft:
            argv += ["-dds"]
    margs = parser.parse_args(argv)
    t0 = time.perf_counter()
    init = model_init.init(margs)
    load_s = time.perf_counter() - t0
    if a.draft == "mtp":
        model, config, cache, tokenizer, draft_model, draft_config, draft_cache = init
        gen = Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft_model,
                        draft_cache=draft_cache, num_draft_tokens=margs.num_draft_tokens,
                        dynamic_draft_tokens=margs.dynamic_draft)
    else:
        model, config, cache, tokenizer = init[:4]
        gen = Generator(model=model, cache=cache, tokenizer=tokenizer)
    prompts = json.loads(a.prompts.read_text())["prompts"]

    def one(text):
        # The pack's chat_template.jinja for one user turn, add_generation_prompt, enable_thinking false
        # (no system block without tools or reasoning instructions). hf_chat_template needs transformers,
        # which the image does not carry (w8sp-a01).
        prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ids = tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)
        job = Job(input_ids=ids, max_new_tokens=a.max_tokens, stop_conditions=config.eos_token_id_list,
                  sampler=GreedySampler())
        gen.enqueue(job)
        result, out_ids = None, []
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r["stage"] == "error":
                    raise r["error"]
                if "token_ids" in r and r["token_ids"] is not None:
                    out_ids += [int(x) for x in r["token_ids"].flatten().tolist()]
                if r.get("eos"):
                    result = r
        return dict(prompt_tokens=int(ids.shape[-1]), new_tokens=int(result["new_tokens"]),
                    time_generate=float(result["time_generate"]), time_prefill=float(result.get("time_prefill", 0.0)),
                    accepted=result.get("accepted_draft_tokens"), rejected=result.get("rejected_draft_tokens"),
                    tokens_sha256=hashlib.sha256(np.asarray(out_ids, dtype="<i4").tobytes()).hexdigest())

    one(prompts[0]["text"])  # warm-up
    passes = []
    for rep in range(a.reps):
        rows = []
        for p in prompts:
            r = one(p["text"])
            r["id"] = p["id"]
            rows.append(r)
        passes.append(rows)
        tok = sum(r["new_tokens"] for r in rows)
        gt = sum(r["time_generate"] for r in rows)
        print(f"pass {rep}: {tok} tokens, decode {tok / gt:.2f} tok/s", flush=True)
    first = passes[0]
    same = all([r["tokens_sha256"] for r in rows] == [r["tokens_sha256"] for r in first] for rows in passes)
    tok = sum(r["new_tokens"] for rows in passes for r in rows)
    gt = sum(r["time_generate"] for rows in passes for r in rows)
    acc = sum(r["accepted"] or 0 for r in first)
    rej = sum(r["rejected"] or 0 for r in first)
    summary = dict(decode_tok_s=tok / gt, output_tokens_first_pass=sum(r["new_tokens"] for r in first),
                   acceptance=acc / (acc + rej) if acc + rej else None,
                   accepted_per_output_token=acc / sum(r["new_tokens"] for r in first) if a.draft == "mtp" else None,
                   repetitions_identical=same)
    print(json.dumps(summary), flush=True)
    return dict(mode="chat", draft=a.draft, num_draft_tokens=margs.num_draft_tokens if a.draft == "mtp" else None,
                dynamic_draft=bool(a.dynamic_draft), load_s=load_s, max_tokens=a.max_tokens, reps=a.reps,
                prompts_file=str(a.prompts), prompts_sha256=hashlib.sha256(a.prompts.read_bytes()).hexdigest(),
                summary=summary, passes=passes)


def mode_kernels(a, torch) -> dict:
    from exllamav3 import Cache, Config, Model
    from exllamav3.cache import CacheLayer_fp16
    toks = torch.as_tensor(need_tokens(corpus_tokens(a.windows), 1024)[None])
    config = Config.from_directory(a.model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096, layer_type=CacheLayer_fp16, max_batch_size=1)
    model.load(progressbar=False, max_chunk_size=4096, max_batch_size=1)
    for _ in range(2):  # the second pass is the one to read in the trace (JIT done)
        state = cache.get_new_state()
        base = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, 4096), "recurrent_states": [state]}
        model.prefill(toks[:, :512], dict(base, past_len=0))
        for i in range(16):
            model.forward(toks[:, 512 + i:513 + i], dict(base, past_len=512 + i))
        torch.cuda.synchronize()
        cache.release_state(state)
    return dict(mode="kernels", prefill=512, decode_steps=16, passes=2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("pptg", "chat", "kernels"))
    ap.add_argument("model_dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--windows", type=Path, default=Path("/workspace/storage/evaluations/rq-r0-20261029/corpus-a01/windows.json"))
    ap.add_argument("--pp", type=int, default=2048)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--depths", default="0,32768")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--kinds", default="pp,tg", help="pptg: which of pp,tg to run at every depth")
    ap.add_argument("--fill-single", action="store_true", help="pptg: prefill the depth context in one chunk")
    ap.add_argument("--prompts", type=Path, default=Path("/src/tools/qwen27b/rq_quality/accept_prompts.json"))
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--draft", choices=("none", "mtp"), default="none")
    ap.add_argument("--num-draft-tokens", type=int, default=0)
    ap.add_argument("--dynamic-draft", action="store_true")
    ap.add_argument("--cache-tokens", type=int, default=8192)
    a = ap.parse_args(argv)
    import torch
    with torch.inference_mode():
        report = dict(env_report(torch), model_dir=a.model_dir, argv=sys.argv[1:] if argv is None else argv)
        report.update({"pptg": mode_pptg, "chat": mode_chat, "kernels": mode_kernels}[a.mode](a, torch))
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
