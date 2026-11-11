#!/usr/bin/env python3
"""Exactness check for Chat Completions `score_assistant` on a running server.

For each prompt (abliteration OFF and, with --abliteration, ON) the check
generates a greedy thinking-disabled reply with `logprobs`, then scores the
generated tokens' text with `score_assistant` and compares token IDs and
logprobs. Generation that ends by length keeps every generated token visible,
so the scored content is the concatenation of the generated token strings.

The server is not started here; point --base-url at a test instance, never
at the production one.
"""

import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "Name the three primary colours in one sentence.",
    "What is 17 multiplied by 23? Answer briefly.",
    "Write one sentence about the sea.",
]


def post(base_url: str, body: dict) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def check(base_url: str, model: str, prompt: str, ablation: dict | None,
          max_tokens: int, tolerance: float) -> list[str]:
    messages = [{"role": "user", "content": prompt}]
    common = {"model": model, "chat_template_kwargs": {"enable_thinking": False}}
    if ablation is not None:
        common["abliteration"] = ablation
    generated = post(base_url, {**common, "messages": messages,
                                "temperature": 0, "max_tokens": max_tokens,
                                "logprobs": True})
    choice = generated["choices"][0]
    entries = choice["logprobs"]["content"]
    if choice["finish_reason"] != "length":
        # A stop token's metadata follows the visible tokens.
        entries = entries[:-1]
    answer = "".join(entry["token"] for entry in entries)
    scored = post(base_url, {**common, "score_assistant": True,
                             "messages": messages + [
                                 {"role": "assistant", "content": answer}]})
    tokens = scored["score"]["tokens"]
    failures = []
    if [t["token"] for t in tokens] != [e["token"] for e in entries]:
        failures.append(f"tokenization differs: generated "
                        f"{[e['token'] for e in entries]} scored "
                        f"{[t['token'] for t in tokens]}")
        return failures
    worst = 0.0
    for index, (entry, token) in enumerate(zip(entries, tokens)):
        delta = abs(entry["logprob"] - token["logprob"])
        worst = max(worst, delta)
        if delta > tolerance:
            failures.append(f"token {index} {token['token']!r}: generated "
                            f"{entry['logprob']} scored {token['logprob']}")
    print(f"{'ON ' if ablation else 'OFF'} {len(tokens):3d} tokens "
          f"max|delta|={worst:.3g} {prompt!r}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="largest accepted |logprob difference|")
    parser.add_argument("--abliteration", action="store_true",
                        help="also check with the loaded profile's defaults")
    args = parser.parse_args()
    settings = [None]
    if args.abliteration:
        settings.append({"enabled": True, "profile": "default"})
    failures = []
    for ablation in settings:
        for prompt in PROMPTS:
            failures += check(args.base_url, args.model, prompt, ablation,
                              args.max_tokens, args.tolerance)
    for failure in failures:
        print("FAIL", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
