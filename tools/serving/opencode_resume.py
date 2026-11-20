#!/usr/bin/env python3
"""Resume an exported OpenCode assistant turn using its captured HTTP request.

Export shape and request construction inspected at OpenCode revision
3a90639cb57619a21e59f544b3e8d23ffed56f48:
  packages/opencode/src/cli/cmd/export.ts (info + messages[{info, parts}])
  packages/opencode/src/session/message-v2.ts (text/reasoning parts)
  packages/opencode/src/session/llm/request.ts (system/plugins/tools)
  packages/opencode/src/session/llm.ts (provider transformation before sending)
Source: https://github.com/anomalyco/opencode/tree/3a90639cb57619a21e59f544b3e8d23ffed56f48

An export is not a wire request: it lacks the resolved system prompt, tool
schemas and provider/plugin transformations. --request MUST be the original
Chat Completions JSON body captured for the cut generation, including its
complete messages and tools. Gufo currently has no request-body dump option.
This helper never runs OpenCode, edits its database, or executes returned tools.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import urllib.request


def rebuild(export, captured, session_id):
    if export.get("info", {}).get("id") != session_id:
        raise ValueError("export session id does not match the requested session")
    messages = export.get("messages", [])
    if not messages or messages[-1].get("info", {}).get("role") != "assistant":
        raise ValueError("export must end at the cut assistant message")
    message = {"role": "assistant"}
    for part in messages[-1].get("parts", []):
        kind = part.get("type")
        if kind in ("text", "reasoning"):
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("text/reasoning part must contain string text")
            if text.startswith("[redacted:"):
                raise ValueError("sanitized exports cannot restore a generation")
            if kind == "reasoning" and "content" in message:
                raise ValueError("reasoning after answer text cannot be flattened")
            key = "content" if kind == "text" else "reasoning_content"
            message[key] = message.get(key, "") + text
        elif kind not in ("step-start", "step-finish", "snapshot", "patch"):
            raise ValueError(f"cannot recover exact assistant bytes from {kind!r} part; "
                             "use a raw text prefix, not structured tool parts")
    if "content" not in message and "reasoning_content" not in message:
        raise ValueError("the exported assistant has no partial text")
    body = copy.deepcopy(captured)
    history = body.get("messages")
    if not isinstance(history, list) or not history or not body.get("model"):
        raise ValueError("--request must contain the original model and messages")
    if history[-1].get("role") == "assistant":
        raise ValueError("capture must be the request before the cut assistant turn")
    history.append(message)
    body["continue_final_message"] = True
    body["add_generation_prompt"] = False
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    return body


def events(lines):
    data = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


def consume(lines, joined, emit):
    calls = {}
    result = {"complete": False, "finish_reason": None}
    for data in events(lines):
        if data == "[DONE]":
            result["complete"] = result["finish_reason"] is not None
            return result
        event = json.loads(data)
        if "error" in event:
            raise ValueError(f"server error: {event['error']}")
        if event.get("usage"):
            result["usage"] = event["usage"]
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            for key in ("reasoning_content", "content"):
                if key in delta and delta[key] is not None:
                    text = delta[key]
                    joined[key] = (joined.get(key) or "") + text
                    emit(text)
            for item in delta.get("tool_calls", []):
                index = item["index"]
                if index not in calls:
                    calls[index] = {"id": item.get("id", ""), "type": "function",
                                    "function": {"name": "", "arguments": ""}}
                call = calls[index]
                for key in ("name", "arguments"):
                    call["function"][key] += item.get("function", {}).get(key, "")
                joined["tool_calls"] = [calls[i] for i in sorted(calls)]
            if choice.get("finish_reason"):
                result["finish_reason"] = choice["finish_reason"]
        reasoning_open = event.get("gufo", {}).get("reasoning_open")
        if isinstance(reasoning_open, bool):
            result["reasoning_open"] = reasoning_open
            # A close followed only by separator whitespace emits no content
            # delta. Preserve the explicit final phase for another resume.
            joined["content"] = None if reasoning_open else (joined.get("content") or "")
    raise ValueError("stream ended without [DONE]; joined partial text was retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id")
    parser.add_argument("--export", default="-", help="export JSON file; default stdin")
    parser.add_argument("--request", required=True, help="captured original HTTP JSON body")
    parser.add_argument("--endpoint", help="explicit /v1/chat/completions URL")
    parser.add_argument("--output", required=True, help="save joined turn and request JSON")
    parser.add_argument("--dry-run", action="store_true", help="save request without sending")
    parser.add_argument("--max-tokens", type=int, help="new generation token limit")
    args = parser.parse_args()
    if not args.dry_run and not args.endpoint:
        parser.error("--endpoint is required when sending")
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if Path(args.output).resolve() in {
        Path(args.request).resolve(), Path(args.export).resolve()
    }:
        parser.error("--output must not overwrite an input file")
    with (sys.stdin if args.export == "-" else open(args.export, encoding="utf-8")) as source:
        export = json.load(source)
    body = rebuild(export, json.loads(Path(args.request).read_text()), args.session_id)
    if args.max_tokens is not None:
        body.pop("max_completion_tokens", None)
        body["max_tokens"] = args.max_tokens
    joined = copy.deepcopy(body["messages"][-1])
    saved = {"session_id": args.session_id, "request": body,
             "joined": joined, "complete": False}
    try:
        if args.dry_run:
            return
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if os.environ.get("GUFO_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["GUFO_API_KEY"]
        request = urllib.request.Request(args.endpoint, json.dumps(body).encode(), headers)
        with urllib.request.urlopen(request, timeout=300) as response:
            saved.update(consume(response, joined,
                                 lambda text: print(text, end="", flush=True)))
    except Exception as error:
        saved["error"] = str(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(str(error))
