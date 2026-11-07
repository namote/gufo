# B2 brief: Flash-Next EXL3 decode/verify/prefill, second batch (2026-11-07)

Same rules as `plan/exl3/B1-BRIEF.md` (read it: Rules, Numerical policy, build
procedure `build/b1/build.sh`, commit/attribution, report format). Bits may
change; lossy changes each in their own commit with the expected error stated;
FP32 accumulation/residuals kept. No GPU/model/timing runs while iterating.
Work in `/home/m2on/projects/gufo` on `ai-tools/port-u4f`, one item at a time.

B1 result (KNOWHOW "Flash-Next B1 decode glue and HC Q8"): AR 27.3→31.0,
MTP d0 51.4→60.7, PP2048 1471→1509 tok/s; quality unchanged within noise.
Learned: fusing q/k normalization into GdnKernel broke chunk/checkpoint
invariance — fused paths must give the same result regardless of how tokens are
split into calls (chunk invariance is a correctness contract, not a bit rule).

## Items

- **E1 (was A2) producer-side X2b input rotation, decode.** Producers
  (HcMixEpilogue → GDN qkv/z, attention q/k/v, shared-MLP input; GdnEpilogue →
  ssm_out; SigmoidMul → attention out) write the rotated FP16 input the EXL3
  GEMV consumes, removing separate rotate launches. Earlier blocker: up to 3
  different suh vectors and `input_rotated` accepted only at R=1. Extend the
  GEMV entry to accept pre-rotated inputs for R≤8 and have the producer emit one
  slab per distinct suh. Count nodes removed per layer.
- **E2 (was B3) EXL3 prefill deferred attention-combine residual.** Port the
  GGUF `defer_w`/`HcApplyDeferred`/`kPrior` mechanism (`kernels.hip.cpp`
  ≈6630, ≈1180) to the EXL3 four-lane combine and the FromRes mix epilogue;
  audit every reader of `res` between the deferred write and its application.
- **E3 (was C5) fused HC up + mix, prefill.** Token-major bandwidth kernel that
  computes the HC up projection from the low-rank tile and applies the mix
  epilogue without storing the gate. It must recompute the norm from the
  residual (EXL3 route), so it needs a from-residual epilogue. Pre-screen the
  design against ≈0.5 ms/call floor (read residual + write mixed) before
  wiring; if a standalone version can't plausibly beat the current C4 path,
  report and skip.
- **E4 routed live-row bodies for R5–R8 (MTP verify).** `exl3_moe.hip`
  `TileRows` maps 5–8 to the generic 8-row body; extend the exact live-row
  selection (kept for R1–R4) to R5–R8. Audit VGPR/LDS/scratch of every new
  instantiation from the compiler resource output (past 40 KiB LDS promotion).
- **E5 MTP draft chain without per-draft host sync.** Each draft step syncs and
  copies a 4-byte token (`executor.cpp` ≈4406, host loop `engine.cpp` ≈1232).
  Where the draft policy permits (greedy argmax / the existing confidence rule
  computable on device), chain draft steps on the device (graph or device-side
  token feed) and read back the whole chain once. Keep sampled-draft semantics
  identical where on-device is not possible.
- **E6 GDN verify snapshot cost.** First verify token snapshots full FP32 state
  ≈3.1 MB/layer × 36 (`kernels.hip.cpp` ≈2319-2327 at B1 time). Replace with
  cheaper rollback (e.g. ping-pong state buffers + replay of accepted prefix, or
  snapshot only on rejection paths) preserving rollback correctness.
- **E7 shared expert concurrent with routed in decode.** Overlap is gated to
  large batches or Q8_0 (`executor.cpp` ≈3976). Add a parallel graph branch for
  EXL3 decode shared MLP vs routed chain, joining before the epilogue; scratch
  must not alias across branches.

Skip an item only if it does not pay off or is unsafe; say why. Prefer E1, E4,
E5, E6, E7 (decode/MTP) first, then E2, E3 (prefill).
