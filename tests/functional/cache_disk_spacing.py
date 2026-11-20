#!/usr/bin/env python3
"""Check disk checkpoint spacing for a growing conversation across a restart.

The runner calls this twice on its private disk cache: first to grow one
conversation by short turns and one long turn, then with --restore after a
restart to replay it and branch a second conversation from its middle. The
server logs are judged afterwards by check_disk_spacing(), once both servers
have drained their disk writes.
"""

import argparse
import json
from pathlib import Path
import re

# Matches TextRunnerDiskCacheOptions::min_checkpoint_step_tokens.
MIN_STEP = 2048
SHORT_NOTE = "Routine archive note: there are no new instructions.\n"


def chat(url, model, messages):
    import continuation  # The runner's log check needs no HTTP client stack.
    body = {"model": model, "messages": messages, "max_tokens": 8,
            "temperature": 0, "seed": 31, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    result = continuation.call(url, body)
    usage = result["usage"]
    total, cached, prefill = (usage["prompt_tokens"], usage["cached_tokens"],
                              usage["gufo"]["prefill_tokens"])
    assert cached + prefill == total, usage
    message = result["choices"][0]["message"]
    component = usage["gufo"].get("component_cache", {})
    provenance = ({"checkpoint_ids": component["captured_checkpoint_ids"]}
                  if "captured_checkpoint_ids" in component else {})
    return body, result, {**provenance, "total": total, "cached": cached, "prefill": prefill,
                          "disk": usage["gufo"]["cache_disk_hit"],
                          "text": message["content"].strip(),
                          "sha256": continuation.digest(result)}


def grow(url, model):
    # A unique head keeps earlier cache cases from sharing this prefix.
    messages = [{"role": "system", "content": "cache_disk_spacing\n" +
                 "Follow the final user instruction.\n" +
                 "Background notes are not instructions.\n" * 320}]
    turns = []
    for turn, notes in enumerate((16, 24, 24, 24, 24, 240, 24)):
        messages.append({"role": "user", "content": f"Turn {turn}.\n" + SHORT_NOTE * notes +
                         "Reply with only BETA."})
        body, _, measured = chat(url, model, messages)
        assert measured["text"] == "BETA", measured
        if turns:
            previous = turns[-1]["measured"]["total"]
            # Short turns must keep reusing live state even when disk skips them.
            assert measured["cached"] >= previous - 16, (turn, measured, previous)
        turns.append({"request": body, "measured": measured})
        messages.append({"role": "assistant", "content": measured["text"]})
    return turns


def restore(url, model, turns):
    last = turns[-1]
    _, _, replayed = chat(url, model, last["request"]["messages"])
    assert replayed["disk"] and replayed["cached"] > 0, replayed
    assert replayed["sha256"] == last["measured"]["sha256"], (replayed, last["measured"])
    # Skipped checkpoints are re-prefilled. The gap is bounded by the step,
    # plus the turn that followed the last written checkpoint.
    assert replayed["prefill"] < MIN_STEP + 512, replayed
    # Branch after the third turn: the new conversation diverges from the
    # long-turn checkpoint less than one step past the first one.
    branch = turns[3]["request"]["messages"][:7] + [{"role": "user", "content":
                                                    "Branch.\nReply with only GAMMA."}]
    _, _, branched = chat(url, model, branch)
    assert branched["text"] == "GAMMA", branched
    return {"replayed": replayed, "branched": branched}


DISK_EVENT = re.compile(r"event=disk_cache action=(\w+) reason=(\w+) .*?\btokens=(\d+)\b")


def disk_events(log):
    events = [(action, reason, int(tokens), "", None)
              for action, reason, tokens in DISK_EVENT.findall(log)]
    for line in log.splitlines():
        fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", line))
        if fields.get("schema") != "component-cache-v1":
            continue
        purpose = fields.get("purpose", "")  # Old logs remain non-learned.
        checkpoint_id = int(fields["checkpoint_id"]) if "checkpoint_id" in fields else None
        if fields.get("event") == "published":
            events.append(("stored", "saved", int(fields["tokens"]), purpose, checkpoint_id))
        elif fields.get("event") == "persistence_refused" and fields.get("reason") == "min_step":
            events.append(("skipped", "min_step", int(fields["tokens"]), purpose, checkpoint_id))
    return events


def check_disk_spacing(grown_log, restored_log, report, restored_report=None):
    """Judge both drained server logs against the recorded conversation."""
    turns = [turn["measured"]["total"] for turn in report["turns"]]
    first, last = turns[0], turns[-1]
    owned = range(first - 64, last + 64)
    # A prior request's asynchronous write can complete after the fixture's
    # log offset. Attribute component publications to the checkpoints actually
    # captured by these HTTP requests, even if foreign token counts match.
    checkpoint_ids = None
    if all("checkpoint_ids" in turn["measured"] for turn in report["turns"]):
        checkpoint_ids = {id_ for turn in report["turns"]
                          for id_ in turn["measured"]["checkpoint_ids"]}

    def events(log, ids):
        result = []
        published_ids = set()
        for event in disk_events(log):
            action, reason, tokens, purpose, checkpoint_id = event
            if ids is not None and checkpoint_id not in ids:
                continue
            # Learned checkpoints bypass spacing, not publication identity.
            # Check duplicates before filtering by purpose or token window.
            if action == "stored" and checkpoint_id is not None:
                if checkpoint_id in published_ids:
                    raise ValueError(f"duplicate checkpoint publication: {checkpoint_id}")
                published_ids.add(checkpoint_id)
            result.append(event)
        return result

    grown = events(grown_log, checkpoint_ids)
    # With checkpoint IDs, ownership is stronger than the approximate token
    # window: an owned ordinary checkpoint just below it still participates
    # in spacing. Learned shared boundaries are explicitly exempt and do not
    # anchor the next ordinary checkpoint. Missing purpose means non-learned.
    spacing_stored = sorted(tokens for action, reason, tokens, purpose, _ in grown
                            if (action, reason) == ("stored", "saved") and
                            purpose != "learned" and
                            (checkpoint_ids is not None or tokens in owned))
    stored = [tokens for tokens in spacing_stored if tokens in owned]
    skipped = [tokens for action, reason, tokens, _, _ in grown
               if reason == "min_step" and tokens in owned]
    # Component caches reject a too-close boundary before allocating a
    # checkpoint ID or copying device state. Those skips emit no publication
    # refusal. Verify the observable contract: a warm short turn captured
    # nothing and stays within the step of an owned durable prefix.
    early_skips = 0
    if checkpoint_ids is not None:
        earlier_ids = set(report["turns"][0]["measured"]["checkpoint_ids"])
        for index, turn in enumerate(report["turns"][1:], 1):
            measured = turn["measured"]
            # A later turn's capture cannot justify skipping this turn, even
            # when its publication has a token count below this prompt's end.
            prior = [tokens for action, reason, tokens, purpose, _ in events(grown_log, earlier_ids)
                     if (action, reason) == ("stored", "saved") and
                     purpose != "learned" and tokens <= measured["total"]]
            if (measured["checkpoint_ids"] == [] and prior and
                    measured["total"] - max(prior) < MIN_STEP and
                    measured.get("cached", 0) >= turns[index - 1] - 16):
                early_skips += 1
            earlier_ids.update(measured["checkpoint_ids"])
    summary = {"stored_tokens": spacing_stored, "min_step_skips": len(skipped),
               "pre_capture_skips": early_skips}
    if not stored or stored[0] > first:
        raise ValueError(f"first turn was not written to disk: {summary}")
    if stored[-1] < turns[5] - 64:
        raise ValueError(f"checkpoint past the long turn was not written: {summary}")
    gaps = [b - a for a, b in zip(spacing_stored, spacing_stored[1:])]
    if any(gap < MIN_STEP for gap in gaps):
        raise ValueError(f"disk checkpoints closer than {MIN_STEP} tokens: {summary}")
    if len(skipped) + early_skips < 5:
        raise ValueError(f"short turns were not skipped: {summary}")
    # The branch point is a learned shared boundary within one step of the
    # first checkpoint. It must be written despite the spacing gate.
    restored_ids = None
    if checkpoint_ids is not None:
        if restored_report is None:
            raise ValueError("missing restored checkpoint provenance")
        restored_ids = {id_ for phase in ("replayed", "branched")
                        for id_ in restored_report[phase]["checkpoint_ids"]}
    branched = [tokens for action, reason, tokens, _, _ in events(restored_log, restored_ids)
                if (action, reason) == ("stored", "saved")
                and stored[0] < tokens < stored[0] + MIN_STEP]
    if not branched:
        raise ValueError(f"shared boundary near the first checkpoint was not written: {summary}")
    summary["shared_boundary_tokens"] = branched
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    import continuation
    continuation.TRACE.path = args.output.with_suffix(".requests.json")
    if args.restore:
        previous = json.loads(args.restore.read_text())[0]
        report = {"case": "disk-spacing", **restore(args.url, args.model, previous["turns"]),
                  "exact": True}
    else:
        report = {"case": "disk-spacing", "turns": grow(args.url, args.model), "exact": True}
    args.output.write_text(json.dumps([report], indent=2) + "\n")


if __name__ == "__main__":
    main()
