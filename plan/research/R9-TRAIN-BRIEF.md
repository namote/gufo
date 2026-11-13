# Brief R9-train: fixed-direction strength training against re-check loops

Goal (user): with `reasoning_effort` xhigh plus a trained suppression profile, get **more correct answers than low at similar or fewer tokens** on AIME 2025. The direction is fixed; only per-layer strengths are learned, by SPSA, as for the political profile.

Read:
- `plan/research/IDEAS.md` R9, and `docs/GFX1151-KNOWHOW.md` ("Thinking-depth baselines", "Thinking directions", "Re-check suppression grid").
- `tools/qwen38_flash_next/abliteration/spsa.py` (objective, `--facts` guard, `--answer-tokens`, `--block-size`, `--minibatch`).
- `docs/SERVER.md` (`score_assistant`: how the assistant message and reasoning are framed and scored).
- `tools/steer/` (train job wrapper).
- `build/b1/think/` (read only): `think_eval.py` (HES regex, records with `reasoning`/`content`), `build_think_dirs.py` (thought split and RECHECK regex), `mkavg.py` (the averaged direction), `rt/abliteration/recheck-avg.json` (the profile to train: projected, direction_layer 45, residual 1.0).

## Data

Traces are xhigh runs being collected now:
- `build/b1/think/results/aime24-xhigh.jsonl` (30 problems);
- `results/xhigh32.jsonl` (the 100 MATH problems in `math.jsonl`).

Until they exist, develop on the medium traces: `results/aime24-medium.jsonl` and `results/base.jsonl`. AIME 2025 (`aime25*.jsonl`) is the held-out evaluation set and must never be used.

## Build

1. **Three-way labels for re-check thoughts** (independent review advice). For each thought that opens with a re-check (the RECHECK regex), find the running answer candidate before and after the re-check episode (the thoughts until the next forward thought). Use the last explicit numeric result or "answer is"/boxed value seen so far; document the heuristic and its failure modes. Labels:
   - **useful:** the candidate changed to the correct answer. Keep.
   - **harmful:** it changed from correct to wrong. Suppress.
   - **neutral:** no change, or no candidate. Excluded from the objective, but report counts.
   - Also report how often the heuristic finds a candidate at all, per set. Label AIME 2024 and MATH alike.
2. **Objective, by teacher-forced scoring only** (no generation). The quantity is the log-probability of the next thought's opening tokens at the thought boundary, given the trace prefix. Terms:
   - suppress: the opener of harmful and redundant re-checks, plus the forward alternative where one exists;
   - keep: the opener of useful re-checks;
   - guard: the per-token log-likelihood of the final answer content (`content`), only on problems the OFF run got right. The same hinge idea as spsa.py's facts term; a fully solvable-only guard.

   Write the files spsa.py can consume, or extend spsa.py minimally. Requirements:
   - It must score only the continuation tokens after a long shared prefix (the trace so far, in the reasoning channel). Check what `score_assistant` supports for reasoning prefixes and partial scoring. If it cannot score a suffix of the reasoning, add the smallest server or tool change that can, with CPU tests.
   - Prefix lengths can reach ~30k tokens. Plan cost with prompt caching (same prefix for many boundary rows), a cap on boundaries per trace, and `--minibatch`.
3. **SPSA run plan:** direction fixed from `recheck-avg` (direction_layer 45, projected). Parameters: `residual` per layer, attention/mlp 0, `--block-size` 4 or so. Init 1.0 and also 0.5. Write the exact command and the server settings it needs (session count, context: the traces are long, so the training server needs at least 32k context; production runs 1M YaRN, which must be kept so numerics match generation).
4. **Evaluation plan:** the trained profile goes to `rt/abliteration/` (or is uploaded via `--allow-profile-updates` on a new binary) and runs AIME 2025 at xhigh, medium and low with `think_eval.py`, compared with the existing baselines:
   - low 28/30, 7,738 tokens;
   - medium 27, 8,989;
   - xhigh 24, 13,447;
   - low with the untrained L45 profile 22, 7,795.
5. **Tests (CPU):** the labeler on hand-made traces (useful, harmful, neutral, no candidate), the data builder, and any spsa.py change with a fake server.

## Do not

- Use the GPU, call any live server (18740 is in use by an experiment), start/stop `llm` or any `gufo-*` container, push, or use `pgrep -f`/`pkill -f`.
- Modify `build/b1/think/` (write your outputs under `build/b1/r9/`), `~/.local/share/gufo/`, `tools/exl3_quant/hc/`, `plan/exl3/HC-Q8-CALIBRATION.md`.

## Commit and review

Commit tool or server changes with single-line Conventional Commit messages ending with:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016Kh8svoeuCWVkepKUHx2XF
```
Data stays uncommitted under `build/b1/r9/`. Before finishing, have a second reviewer agent review the code, fix, and re-review.

## Report

Label counts per set and label, with 3 examples each; how the objective is scored and its cost estimate; the exact SPSA and evaluation commands and the server settings; tests; and anything left open.
