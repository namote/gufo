# V3-4 brief: GDN recurrent state stored in FP16, computed in FP32

Read `plan/upstream-sync2/PLAN.md` (V3-4) and `plan/upstream-sync2/survey-kyojin-exl3.md` §4 (exllamav3 `0244652`, Kyojin `fce7501`). In `docs/GFX1151-KNOWHOW.md` find the 27B BF16 state rejection (#138, `686a0928`, `GUFO_QWEN_RECURRENT_STATE=bf16`: full-logit RMSE 0.1126 → 0.1396, 200-token greedy reply diverged after 824 bytes) and the B2 profile GDN decode cost. Branch `ai-tools/port-u4f`. The user's rule: speed over bit-exactness; a quality pass decides.

## Goal

Store GDN/linear-attention recurrent state in FP16 (compute stays FP32; round once per pass on store), on both models:
- **Qwen3.8 27B first** (cheaper to qualify): add FP16 next to the existing BF16 `QwenRecurrentStateStorage` in `src/models/qwen/hip/`, reusing the #138 plumbing.
- **Flash-Next**: `src/models/qwen38_flash_next/runtime/` recurrent kernels (decode, verify, replay, `RestoreGdnStateKernel`, `GdnRowSplitKernel` prefill), session state allocation, snapshots/rewind points/disk payloads.

Expected effect: state per Flash-Next session ~108 → 54 MiB; C1 AR +1.5–2%, about twice that at C4 (estimate); checkpoint payloads and rewind points shrink accordingly.

## Requirements

- One canonical selection while it is being qualified: a server option `--recurrent-state fp32|fp16` (and the 27B BF16 value if it already exists under another name, unify it under this option; drop the env var). Default stays **fp32** until the quality pass; the combined pass decides whether fp16 becomes the default.
- Bump the continuation/component-cache layout version for fp16 payloads; old or mismatched snapshots must be rejected cleanly (cache miss, never a failed request), and the disk cache identity must include the state format.
- FP16 overflow: GDN decay keeps the state bounded, but clamp or saturate on store if any path can exceed FP16 range; say what you did.
- Keep MTP, lookup, batched C2+ verify, V3-0 rewind points and V2-4 paths working in both formats.
- Provide a quality tool or reuse an existing one (`tools/qwen27b/check.py`, `tools/qwen38_flash_next/f32_reference.py`, the #138 full-logit RMSE method) that the GPU pass can run: full-logit RMSE and KL of fp16 vs fp32 state over a long prompt (≥32K and ≥128K positions), plus greedy divergence position. Document the exact command.

## Rules

- **No GPU, model or server runs. No production changes.** Do not touch `~/.local/share/gufo`, `llm`, or docker containers other than the build.
- Build only with `build/b1/build.sh <targets>`; CPU tests via `build/b1/build.sh --run /build/obj/<path>`; `build/b1/fmt.sh -i`, `build/b1/checkfmt.sh`; `python3 tests/functional/test_runner.py`.
- Never run anything from `~/projects/_ai.tools`, any `ai-tools/*` docker image, or Kyojin. No `pgrep -f`/`pkill -f`.
- Do not touch `tools/exl3_quant/hc/`, `plan/exl3/HC-Q8-CALIBRATION.md`, or the untracked `plan/research/*.md`. Do not quantize weights.
- **Stage and commit only your own files** (`git commit -- <paths>`; other sessions commit on this branch). One commit per model, single-line Conventional Commits, do not push. Trailers:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_016Kh8svPWb3TTEKxGwdHJTN
  ```
- Have a subagent review the design and the diff; fix and re-review.

## Done

- Builds; CPU tests for continuation layout/cache identity and any kernel reference tests pass; checkfmt passes.
- Report in English: commits, option name, layout version change, memory per session in each format, overflow handling, risks, and exact GPU checks: 27B and Flash-Next session tests in both formats, the quality tool at 32K/128K (RMSE, KL, greedy divergence), decode C1/C4 and prefill timing fp16 vs fp32 at production config, disk restore of a fp16 checkpoint, rejection of a fp32 checkpoint by a fp16 server.
