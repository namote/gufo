#!/usr/bin/env python3
"""Records the W9 held-out suite's overlap with the tuning prompts (tools/qwen27b/rq_quality/accept.py check_overlap:
40-character fragments every 20 characters, exact substring) into suite_h.json's manifest.overlap.

Against: every prompt text of accept_prompts.json (A, includes the D4 set), suite_k.json (K) and suite_d.json (D),
the D4 Korean source file, and the other held-out prompts (duplicates inside the set). The decoded R0/R3 quality
corpora are not checked here (they are stored as token windows). Usage: python3 -I check_heldout.py [--write]
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools/qwen27b/rq_quality"))
import accept  # noqa: E402


def texts(path):
    return "\n\n".join(p["text"] for p in json.loads(path.read_text(encoding="utf-8"))["prompts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    path = HERE / "suite_h.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    against = {
        "accept_prompts": texts(ROOT / "tools/qwen27b/rq_quality/accept_prompts.json"),
        "suite_k": texts(HERE / "suite_k.json"),
        "suite_d": texts(HERE / "suite_d.json"),
        "d4_ko_prompts": (ROOT / "tools/qwen27b/rq_quality/d4_ko_prompts.json").read_text(encoding="utf-8"),
    }
    result = accept.check_overlap(doc, against)
    inner = {}
    for p in doc["prompts"]:
        others = "\n\n".join(q["text"] for q in doc["prompts"] if q["id"] != p["id"])
        hits = accept.check_overlap({"prompts": [p]}, {"x": others})["against"]["x"]["hits"]
        inner.update(hits)
    result["within_set_hits"] = inner
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if a.write:
        doc["manifest"] = {"overlap": result}
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
