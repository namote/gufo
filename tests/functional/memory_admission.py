#!/usr/bin/env python3
"""Bounded Flash-Next memory admission checks against an already owned server.

The caller owns GPU exclusion, health guards, server lifetime and its log.
This runner never retries inference or starts a second model. Timings are
observations, not matched performance qualification.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import re
from pathlib import Path
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18740")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "requests": [], "checks": {}}
    mutex = threading.Lock()

    def save():
        (args.output / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")

    def request(name, messages, extra=None):
        body = {"model": "gufo-flash-next", "messages": messages,
                "max_tokens": 128, "temperature": 0, "seed": 314159,
                "chat_template_kwargs": {"enable_thinking": False}}
        body.update(extra or {})
        started = time.monotonic()
        req = urllib.request.Request(args.url + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            data = json.load(response)
            headers = dict(response.headers)
        entry = {"name": name, "seconds": time.monotonic() - started,
                 "request": body, "response": data, "headers": headers}
        with mutex:
            report["requests"].append(entry)
            save()
        print(name, round(entry["seconds"], 3), data.get("usage"), flush=True)
        return data

    def message(text):
        return [{"role": "user", "content": text}]

    def content(data):
        return data["choices"][0]["message"]["content"]

    try:
        with urllib.request.urlopen(args.url + "/v1/models", timeout=5) as response:
            report["models"] = json.load(response)
        assert "image" in report["models"]["data"][0]["architecture"]["input_modalities"]
        # Four unlimited reservations must make progress together on a pool
        # holding two contexts. Stop streams after visible output, not EOS.
        barrier = threading.Barrier(4)
        first_outputs = threading.Barrier(4)

        def stream(row):
            body = {"model": "gufo-flash-next", "stream": True,
                    "messages": message(f"Start at {100 + row * 100} and count consecutive integers up to 10000. Output only numbers separated by spaces. Do not stop early."),
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}
            barrier.wait(timeout=10)
            started = time.monotonic()
            req = urllib.request.Request(args.url + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            chunks = []
            first = None
            with urllib.request.urlopen(req, timeout=180) as response:
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    if line.strip() == b"data: [DONE]":
                        break
                    data = json.loads(line[6:])
                    assert "error" not in data, data
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        if first is None:
                            first = time.monotonic()
                            first_outputs.wait(timeout=90)
                        chunks.append(delta["content"])
                    if len(chunks) >= 64:
                        break
            assert chunks, row
            return {"row": row, "started": started, "first": first,
                    "closed": time.monotonic(), "chunks": len(chunks),
                    "text": "".join(chunks)}

        with ThreadPoolExecutor(max_workers=4) as pool:
            streams = list(pool.map(stream, range(4)))
        report["checks"]["unlimited_c4"] = streams
        save()
        assert max(x["first"] for x in streams) < min(x["closed"] for x in streams), streams
        assert "READY" in content(request("after-cancel", message("Reply with exactly READY.")))
        # C1 repeat is strict. C4 scheduling can change batch arithmetic, so
        # concurrent checks require valid code and progress, not RNG equivalence.
        for temperature in (0, 1):
            def item(row):
                return message(f"Write a short Python function named add_{row} that adds two numbers. Output code only.")
            extra = {"temperature": temperature, "top_p": 0.95, "top_k": 20,
                     "seed": 20261103, "cache_prompt": False}
            first = request(f"mtp-{temperature}-serial-first", item(0), extra)
            second = request(f"mtp-{temperature}-serial-repeat", item(0), extra)
            assert first["choices"][0]["message"] == second["choices"][0]["message"]
            assert first["choices"][0]["finish_reason"] == second["choices"][0]["finish_reason"]
            with ThreadPoolExecutor(max_workers=4) as pool:
                parallel = list(pool.map(lambda r: request(f"mtp-{temperature}-parallel-{r}", item(r), extra), range(4)))
            for row, data in enumerate(parallel):
                assert f"def add_{row}" in content(data) and "return" in content(data), data
            report["checks"][f"mtp-{temperature}-repeat"] = True
            save()
        long = "Remember the secret code is JADE-619.\n" + "This is a neutral filler sentence about a quiet library.\n" * 800 + "\nWhat is the secret code? Reply with the code only."
        messages = message(long)
        first = request("long-first", messages)
        repeat = request("long-repeat", messages)
        assert "JADE-619" in content(first) and content(first) == content(repeat)
        assert repeat["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        continued = messages + [first["choices"][0]["message"], {"role": "user", "content": "Repeat the same secret code only."}]
        cached = request("continuation-cached", continued)
        cold = request("continuation-cold", continued, {"cache_prompt": False})
        assert content(cached) == content(cold) and "JADE-619" in content(cached)
        report["checks"]["long-cache-continuation"] = True
        image = base64.b64encode(args.image.read_bytes()).decode()
        vision = [{"role": "user", "content": [
            {"type": "text", "text": "Name the colors of the left and right halves. Use: Left: color; Right: color."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}}]}]
        answer = content(request("vision", vision)).lower()
        assert "left: red" in answer and "right: blue" in answer, answer
        report["checks"]["vision"] = True
        answer = content(request("medium", message("What is 17 times 19? Answer with only the number."),
                                 {"max_tokens": 256, "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"}}))
        assert "323" in answer, answer
        report["checks"]["medium"] = True
        # Keep three real requests live. One full-context guarantor plus a
        # ~7K prompt borrower and a short peer fills a two-context pool. Once
        # a borrower's 4096-token window is exhausted, cancel the guarantor
        # and require a logged resume. CPU tests separately check exact tokens.
        log_start = args.server_log.stat().st_size
        stops = [threading.Event() for _ in range(3)]
        started = [threading.Event() for _ in range(3)]
        pressure_results = [{} for _ in range(3)]

        def pressure(row):
            filler = ("This is a neutral filler sentence about a quiet library.\n" * 580) if row == 2 else ""
            body = {"model": "gufo-flash-next", "stream": True,
                    "messages": message(filler + f"\nStart at {100 + row * 100} and count consecutive integers up to 50000. Output only numbers separated by spaces. Do not stop early."),
                    "temperature": 0, "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(args.url + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            chunks = 0
            try:
                with urllib.request.urlopen(req, timeout=330) as response:
                    for line in response:
                        if not line.startswith(b"data: "):
                            continue
                        if line.strip() == b"data: [DONE]":
                            break
                        data = json.loads(line[6:])
                        assert "error" not in data, data
                        if data.get("choices", [{}])[0].get("delta", {}).get("content"):
                            chunks += 1
                            started[row].set()
                        if stops[row].is_set():
                            break
                pressure_results[row] = {"chunks": chunks, "closed_by_test": stops[row].is_set()}
            except BaseException as error:
                pressure_results[row] = {"chunks": chunks, "error": repr(error)}
                raise

        def recent_log():
            with args.server_log.open("rb") as log:
                log.seek(log_start)
                return log.read().decode(errors="replace")

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = []
            try:
                for row in range(3):
                    futures.append(pool.submit(pressure, row))
                    assert started[row].wait(60), (row, pressure_results)
                deadline = time.monotonic() + 300
                while "event=kv_wait" not in recent_log():
                    assert all(not f.done() for f in futures), pressure_results
                    assert time.monotonic() < deadline, "No GPU KV pressure reached"
                    time.sleep(0.5)
                before_cancel = recent_log()
                waiter = re.findall(r"event=kv_wait request=(\d+)", before_cancel)[-1]
                cancel_offset = len(before_cancel)
                stops[0].set()
                resume_deadline = time.monotonic() + 45
                while not re.search(rf"event=kv_resume request={waiter}\b", recent_log()[cancel_offset:]):
                    assert time.monotonic() < resume_deadline, "KV waiter did not resume"
                    time.sleep(0.5)
                report["checks"]["gpu-kv-wait-resume"] = True
                report["pressure_waiter"] = waiter
            finally:
                for stop in stops:
                    stop.set()
                try:
                    for future in futures:
                        future.result(timeout=60)
                finally:
                    report["pressure"] = pressure_results
                    (args.output / "pressure.log").write_text(recent_log())
                    save()
        assert "READY" in content(request("after-pressure", message("Reply with exactly READY.")))
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
