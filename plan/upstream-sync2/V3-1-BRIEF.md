# V3-1 brief: lone prompts in wide (14K-row) prefill passes on transient scratch

Read `plan/upstream-sync2/PLAN.md` (V3-1), `plan/upstream-sync2/survey-upstream.md` §17, and in `docs/GFX1151-KNOWHOW.md` the prefill chunk-cap experiment ("Prefill chunk cap (`kPrefillChunkTokens`...", 4096/8192/16384: +1.3% at 8192, +2.8% at 16384, −0.5% at 2048, +6.38 GiB peak, rejected for its memory). Branch `ai-tools/port-u4f`, HEAD b85a48ad. Prefill speed is the user's top priority.

## Upstream

- `2e50748f` (#1265): run lone prompts in 14K-row passes on transient scratch. `src/core/hip/scratch_arena.{hpp,cpp}` (~260 lines), executor scratch carve-up, engine/runner StepCapacity/PassChunk plumbing, `tests/mem/scratch_arena_test.cpp`. Upstream measured served prompts of 28K–85K tokens 2–7% faster than 4096-row passes (GGUF).
- `425e3513` (#1274): release the wide scratch when the server goes idle (scheduler/runner idle hook).
- `842295e4` (#1251): only the demand-mapped VMM arena part that `ScratchArena` needs (`mem::DemandMapped` or equivalent). Do **not** port the rest of #1251 (session infrastructure, ~7K lines).
Use `git show <hash>`; port by content into our layout (`src/models/qwen38_flash_next/runtime/`, `engine.cpp` with `kPrefillChunkTokens = 4096` at ~:37, our EXL3 executor scratch in `runtime/executor*.cpp`).

## Requirements

- The wide pass is used only when one prompt prefills with **no decoder and no other prefill beside it**; the extra row buffers are mapped only for that pass and unmapped at the next decode step or when idle. Otherwise behavior stays exactly as today (4096-row passes).
- If demand mapping or the wide allocation fails (VMM unsupported on gfx1151, not enough memory with 4 sessions × 1M context K4V4 + KV pool), fall back to 4096-row passes silently except for one INFO log line; never fail a request.
- Keep our U2-2 alignment and in-pass checkpoint rules (capture boundaries, 2,048-token grid, stable boundary stops, V3-0 rewind points) valid at the new pass width. A capture boundary inside a wide pass must split it exactly as today.
- Output must stay exact (same tokens as 4096-row passes for greedy; the session tests check exact replay). Choose the wide width so the expert GEMMs stay efficient on gfx1151 (upstream 14K; our measurement favoured 16K over 8K); expose it as one constant.
- Log one INFO line per wide pass decision (`event=wide_prefill rows=... mapped_bytes=...` or a refusal reason) so the GPU pass can see it.

## Rules

- **No GPU, model or server runs. No production changes.** Do not touch `~/.local/share/gufo`, `llm`, or docker containers other than the build.
- Build only with `build/b1/build.sh <targets>`; CPU tests via `build/b1/build.sh --run /build/obj/<path>`; `build/b1/fmt.sh -i`, `build/b1/checkfmt.sh`; `python3 tests/functional/test_runner.py`.
- Never run anything from `~/projects/_ai.tools` or any `ai-tools/*` docker image. No `pgrep -f`/`pkill -f`.
- Do not touch `tools/exl3_quant/hc/`, `plan/exl3/HC-Q8-CALIBRATION.md`, or the untracked `plan/research/*.md`.
- Have a subagent review your design before implementing and the diff after; fix and re-review.
- Commit (single-line Conventional Commits citing upstream PR numbers; one commit per upstream item is fine), do not push. Trailers:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_016Kh8svPWb3TTEKxGwdHJTN
  ```

## Done

- Builds; scratch-arena CPU tests (ported), scheduler/runner/cache CPU tests pass; checkfmt passes.
- Report in English: what was ported, commit hashes, peak extra memory of a wide pass, fallback behavior, risks, and exact GPU checks: `qwen38_flash_next_session_test`, `mtp_ops_test`, functional `long-context`/`cache`/`batch`/`cache-growth`, prefill timing at ~8K, 16K, 32K, 64K and the 116,803-token prompt (A B A B against runtime-20261120b at production config), a check that decode and C4 are unchanged, and the mapping/unmapping log lines.
