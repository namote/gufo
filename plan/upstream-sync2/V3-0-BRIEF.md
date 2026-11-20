# V3-0 brief: keep the previous prompt reusable when clients drop reasoning

Read `plan/upstream-sync2/PLAN.md` (V3-0, V2 results), `docs/KV-CACHE.md` (what gets retained, disk spacing, live frontier), and the KNOWHOW entries "V2 upstream v0.35 medium items, combined pass", "Short-replay and cancelled-conversation cache misses fixed" and "RAM checkpoint tier removed". Branch `ai-tools/port-u4f`, HEAD e7fee9ae. Production now runs this code (runtime-20261120b).

## Problem (measured, `build/b1/pass-20261120v2/r4/func/cand-growth/` and current `func/cand-growth/`)

Flash-Next (GDN recurrent state + attention KV), 4 sessions, disk cache on, no RAM checkpoint tier (c597eed1).
`tests/functional/cache_growth.py` replays a growing conversation in modes `drop_reasoning`, `keep_reasoning`, `discard_reasoning`, `thinking_off`.
In `drop_reasoning`/`discard_reasoning`, the client sends history without the previous assistant reasoning. The live slot's frontier contains the generated reasoning, so the next prompt diverges at the previous assistant turn. The recurrent state cannot rewind, so reuse falls back to the nearest disk checkpoint. Disk checkpoints are spaced ≥2,048 tokens (`min_step`), so the previous turn's complete prompt was not written. Measured per turn: `cached=2924, previous_prompt=3132` (turn 2), `cached=3132, previous_prompt=3338` (turn 3): ~200–400 tokens re-prefilled each turn here, up to ~2K in general. Coding agents that drop reasoning pay this every turn.

## Goal

The next turn of such a conversation reuses at least up to the previous request's prompt end (minus the assistant opening that may re-render), without restoring a RAM checkpoint tier and without unbounded disk write growth.

Candidate designs (evaluate, pick the cheapest correct one, explain why):
1. **Device-side prompt-end point in the slot:** at the end of prefill (before generation) keep a copy of the recurrent/private state (GDN state, conv tails; tens of MB) in the slot's device memory. Attention KV rows up to the prompt end stay valid in place. A later prompt that diverges inside the generated part rewinds to that point (restore state, truncate KV) instead of going to disk. Check `session_rollback_bytes`, existing rollback/snapshot machinery, slot memory budgets with 4 sessions at 1M context, and what happens on reassignment (write that point to disk as today's live-frontier write-back does).
2. **Exempt the prompt-end capture from `min_step`** when the generated suffix is likely to be dropped (reasoning enabled), so it goes to disk. Cost: one ~170 MB disk write per turn. Compare with 1.
3. Anything better you find in the code.

## Rules

- **No GPU, model or server runs. No production changes.** Do not touch `~/.local/share/gufo`, `llm`, or docker containers other than the build.
- Build only with `build/b1/build.sh <targets>`; CPU tests via `build/b1/build.sh --run /build/obj/<path>`; format with `build/b1/fmt.sh -i`, check with `build/b1/checkfmt.sh`; `python3 tests/functional/test_runner.py`.
- Never run anything from `~/projects/_ai.tools` or any `ai-tools/*` docker image.
- Do not touch `tools/exl3_quant/hc/`, `plan/exl3/HC-Q8-CALIBRATION.md`, or the untracked `plan/research/*.md`. No `pgrep -f`/`pkill -f`.
- Commit when done (single-line Conventional Commit, trailers below), do not push:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_016Kh8svPWb3TTEKxGwdHJTN
  ```
- Keep exactness: a rewind must produce the same tokens as a cold prefill of the same prompt (the session/serving tests check exact replay). Greedy output and seeded sampling must not change.
- Update `docs/KV-CACHE.md` for the new behavior and log events.

## Done

- Builds; CPU tests for the cache/scheduler/runner pass, plus new CPU tests that reproduce the drop-reasoning divergence and show reuse up to the prompt end; checkfmt passes.
- Report in English: chosen design and why, memory cost per session (at 1M context, 4 sessions), commit hash, tests, risks, and exact GPU checks (functional `cache-growth`, `cache`, `long-context`, `batch`, session test, a pi_agent replay, and decode speed cells to confirm no regression).
