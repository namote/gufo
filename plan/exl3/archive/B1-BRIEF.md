# B1 brief: Flash-Next EXL3 decode/prefill launch and traffic cuts (2026-11-07)

Goal: reduce Flash-Next EXL3 decode (R=1 AR and R≤8 MTP verify) and prefill
time. **Rule change (user, 2026-11-07): output bits may change.** More
accurate results are welcome; a small numerical error is acceptable when it buys
speed. Bit-exactness is no longer required. Source audit (read-only) estimated:
decode ≈1,750 graph nodes/token (≈1,150 small non-GEMV kernels), kernel-active
≈90%; prefill HC family ≈54 GB/forward at ≈190 GB/s.

## Rules

- Read `AGENTS.md`, `docs/GFX1151-KNOWHOW.md` (sections "Numerical contracts and
  bit-exactness", "Reusable kernel patterns", "Tried and rejected", and the
  Qwen3.8-Flash-Next part of "Other") before editing. Do not revive a rejected
  mechanism unchanged.
- **Numerical policy (replaces the old bit-exact contract):** outputs may
  change. Keep FP32 accumulation and FP32 residual streams (BF16 residuals are
  a measured quality loss). For every change, state in the commit what can
  differ (e.g. "FMA association in X", "HC-down weights W8A8 instead of F16")
  and the expected error scale. Prefer changes that are exact or more accurate;
  lossy ones are allowed when they cut real time or bytes, and the parent's
  final quality pass (BF16-teacher KL/NLL/top-1 on 32,760 positions, plus
  AR/MTP output sanity) decides whether each lossy change stays. Keep each
  lossy change in its own commit so it can be reverted alone.
- Preserve: graph capture/replay semantics, cached-activation readiness flags
  (and the pre-capture save/restore tuple at `executor.cpp` ≈4447 — add any new
  cached flag to it), rollback/snapshot, ragged rows, R=1..8 verify widths, the
  GGUF path (do not break it; EXL3 is the target), K3–K8 and K3.5 rates.
- No new options/switches: a successful change becomes the default path.
- Match surrounding style and comment density. Keep `src/core/platform` rules.
- **No GPU runs, no model runs, no timing.** Only build and CPU/unit checks.
  The parent runs one combined GPU correctness+performance pass at the end.
- Build (CPU-only container, pinned Nix ROCm10.1, incremental):
  `build/b1/build.sh` (default flash-next targets) or `build/b1/build.sh <targets>`.
  CPU checks: `build/b1/build.sh --run /build/obj/quant_exl3_format_test` and
  `build/b1/build.sh --run /build/obj/tests/models/qwen38_flash_next/qwen38_flash_next_exl3_loader_test`.
  Format: `build/b1/build.sh --run clang-format --dry-run --Werror <files>`
  (use `-i` to fix). Only one build at a time; keep the script default (--parallel 4, 16 GiB; a live server shares this host).
- Add or extend GPU unit tests (they run in the final pass) that compare the
  new path against the old sequence or a CPU oracle (byte-equal where the
  change is exact, otherwise with a stated tolerance justified by the error
  scale), covering
  R=1..8, ragged/odd sizes, K3.5 and an integer rate, signed zero/subnormal
  inputs where the moved arithmetic is sensitive.
- Commit each item separately on the current branch (Conventional Commits,
  single line, end with the attribution lines the parent gives you). Do not
  push. Do not touch `docs/GFX1151-KNOWHOW.md` (parent records results after
  measurement).
- Report: per item, what fused/removed (node count per token/layer before→after),
  files:lines, pinned expressions, new tests, build/CPU check results, and any
  item you rejected with the reason.

## Items

All items are done in the main checkout `/home/m2on/projects/gufo` on branch
`ai-tools/port-u4f`, one item at a time. A3 is already committed (`5c328695`).
Partial unreviewed WIP from an earlier split attempt may be used as a starting
point, not trusted: `B1-WIP-A.patch` (A items after A3) and `B1-WIP-B.patch`
(B items) in the scratchpad path the parent gives you.

Decode:

- **A1 GDN decode glue.** In the n≤8 decode DeltaNet sequence
  (`executor.cpp` ≈3319; kernels ≈`kernels.hip.cpp:8289-8377`): fold
  HistoryShift+Copy into SsmConv4 and GdnPrep into GdnKernel.
  (Prefill row-split fusion `024a2d5b` is the precedent; decode
  is not fused yet.)
- **A2 X2b input rotation into producers (decode).** The shared Qwen path
  already accepts pre-rotated inputs (`input_rotated`, `src/models/qwen/hip/
  .../gemv.hip` ≈770-800). Have the producers (HcMixEpilogue → GDN qkv/z and
  attention q/k/v and shared-MLP input; GdnEpilogue → ssm_out; SigmoidMul →
  attn out) write the rotated FP16 input directly, removing the separate
  rotate launches, only where the rotation's rounding equals the current
  separate kernel's.
- **A3 HC-down GEMV memory-level parallelism.** `kernels.hip.cpp` ≈1461-1482:
  HC-down (K=10240, N=320) runs 80 blocks with kSteps=4 → latency-bound. Raise
  loads in flight (kSteps 8–10, or two rows/wave) for this shape without
  changing each lane's element order or the accumulate order.
- **A4 GPU greedy argmax for AR.** AR currently copies 1 MB FP32 logits to the
  host and samples on the CPU (`executor.cpp` ≈4864, `engine.cpp` ≈855). For
  requests whose effective sampler is pure greedy with no penalties/
  constraints/logprobs, use the existing GPU argmax (`GreedyMtpPredictions`/
  `LaunchBatchedGPUArgmax`, `executor.cpp` ≈5758-5812) with identical tie
  order to the CPU sampler; otherwise unchanged.

MoE + prefill HC:

- **B1 Routed decode glue.** `exl3_moe.hip` ≈1550-1606 and `executor.cpp`
  ≈3901: fold memset+BuildTiles into TopK, ScaleDownRows into
  RestoreActivateRotate, RestoreDownAssignments into MoeEpilogue.
  Keep deterministic per-row outputs (tile order from atomics is
  fine only because each tile writes disjoint rows).
- **B2 RoutedGemv block packing.** Gate/up grid (40,10,2) and down (160,10) use
  one-wave blocks. Pack 4–8 tiles (waves) per block with identical per-wave
  arithmetic; check VGPR/LDS/scratch in the compiler resource output (watch
  the past 40 KiB LDS promotion from route descriptors) before keeping.
- **B3 EXL3 prefill deferred attention-combine residual.** GGUF path has
  `defer_w`/`HcApplyDeferred` (`kernels.hip.cpp` ≈6630, `kPrior` pinning
  ≈1180). EXL3 (`defer_residual_ = WmmaExperts()` false, `executor.cpp` ≈4775)
  writes the full FP32 4×2560 residual after attention then rereads it.
  Port the deferral (kPrior variant of `HcCombineVec4Kernel` and
  a prior term in the FromRes mix epilogue). Net ≈2 GB/forward.
- **B4 Prefill routing glue.** Have the FFN mix epilogue also emit the FP16
  `mixed` the router's `PrepareHalfInput` produces (drop that pass); vectorize
  `RotateGateUpTokens` loads (`exl3_moe.hip` ≈606-621 scalar) with unchanged
  arithmetic; fuse router top-k.

Lossy/precision items (now allowed; each its own commit):

- **C1 Q8_0 quantizer rounding.** `QuantizeHalfQ8_0Kernel` (`kernels.hip.cpp`
  ≈6919) uses fast-math `amax/127` and `1/d`: make scale/inverse correctly
  rounded (`__fdiv_rn` or host-equivalent) so codes stop flipping between
  compiler builds. Do this before C2/C3.
- **C2 Decode HC-down uses the resident Q8_0 copy** (`device_model.cpp`
  ≈362-390 `down_q8`; currently gated by `prefill_phase` at `executor.cpp`
  ≈2975) via the W8A8 path the GGUF decode already uses (`executor.cpp`
  ≈3042-3052). −≈295 MB/token, no new memory.
- **C3 HC-up Q8_0 for EXL3 decode/prefill**, built like `down_q8`, replacing
  the F16 copy in memory if no remaining reader needs F16 (else note the +MiB).
- **C4 Prefill HC gate stored FP16** instead of FP32 (T×10240 per mixer),
  or better, **C5 fuse HC up GEMM with the mix epilogue** so the gate never
  reaches memory: a token-major bandwidth-oriented kernel (lo tile in LDS, up
  weights from L2, coalesced along hidden). Note the earlier epilogue-style
  fusion was rejected for speed (+28 ms, KNOWHOW S:34), not for numerics; pre-
  screen design against the ≈0.5 ms/call bandwidth floor. If C5 is done, C4 is
  unnecessary.
- With exactness relaxed, B4's router top-k fusion and any A/B item that was
  blocked only by rounding are now allowed.

Order is free; skip an item only if it does not pay off or is unsafe.
