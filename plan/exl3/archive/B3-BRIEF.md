# B3 brief: Flash-Next decode dispatch count, launch gaps, routed bandwidth (2026-11-07)

Same rules as `plan/exl3/archive/B1-BRIEF.md` (Rules, Numerical policy — bits may
change, lossy changes isolated and stated, FP32 accumulation/residuals kept;
build `build/b1/build.sh`; formatter: the pinned clang-format under
`/nix/store/*clang-tools*/bin/clang-format` via `build/b1/build.sh --run <abs
path>`; commit/attribution; report). Chunk/split invariance of fused paths is a
correctness contract (B1 lesson). No GPU/model/timing runs while iterating.
Work in `/home/m2on/projects/gufo` on `ai-tools/port-u4f`, one item at a time,
second-reviewer subagent review per item.

Current decode (B2 profile, KNOWHOW "Flash-Next B2 …"; raw
`build/b1/gpu-b2/prof/decode-summary.json`): AR 31.8 tok/s, 31.78 ms/token,
**1442 dispatches/token, kernel-active 67% → ≈10.5 ms/token idle ≈7 µs per
dispatch**. Calls/token: HC combine/mix 290, dense/shared GEMV 228, HC Q8
down/up 194, GDN 144, routed glue 144, attention 120, routed GEMV 96, small
GEMM 86, rotation 49, router top-k 48, norm 30. Routed GEMV gate/up 4.64 ms
(≈170 GB/s), down 2.63 ms (≈150 GB/s); dense ≈200 GB/s; head ≈230 GB/s.
Goal: halve dispatches (<800/token) and lift routed GEMV toward 200+ GB/s.

## Items

- **G0 launch-gap microbenchmark** `tools/bench/gfx1151_launch_gap.hip`
  (build with `tools/bench/build.sh`; it runs in the final pass). Measure per
  dispatch overhead for ≈1500 tiny dependent kernels: plain stream vs captured
  hipGraph, with/without a forked side stream, small vs 1 KiB kernargs, and
  report µs/dispatch. Also measure a persistent kernel's grid-wide barrier cost
  (40 CUs, atomic counter + `__threadfence`, one block per CU and 2–4 blocks
  per CU) per barrier. Print a table. This decides G6/G7.
- **G1 HC decode chain fusion (largest count).** Per layer the decode HC path
  is ≈6 combine/mix + ≈4 Q8 kernels per mixer pair (combine, norm/Q8_1
  quantize, Q8 down GEMV, SiluScaleQ8_1, Q8 up GEMV, mix epilogue …). Fuse
  adjacent elementwise/quantize stages into their producer/consumer and use
  the "last arriving block runs the tail" pattern (atomic arrival counter, as
  B1's `RouterTopKTiles`) to chain the small N=320 down → SiLU/requant → next
  stage inside one launch where a grid dependency was the only reason for a
  separate kernel. Keep counter reset graph-safe.
- **G2 attention decode** (≈10 dispatches per full-attention layer): fuse
  q/k norm+RoPE+KV write, sparse indexer scoring/selection glue, and output
  gating where adjacent.
- **G3 GDN decode** (4/layer): fuse remaining conv/prep/epilogue launches
  where chunk invariance is kept (q/k norm stays in GdnPrepKernel — B1 lesson).
- **G4 router + routed glue** (router GEMV, top-k+tiles, restore/activate,
  combine ≈4–5/layer): fold top-k into the router GEMV via last-block tail;
  fold RestoreActivateScale into the gate/up GEMV tail or the down GEMV head.
- **G5 routed GEMV bandwidth.** `RoutedGemv<4,false,1,false>` (gate/up,
  grid 10 experts × 2 × N-tiles) and `<4,false,1,true>` (down). Analyze loads
  in flight per wave, prefetch distance, waves/CU and the down GEMV's short K
  (640) and propose/implement changes that keep per-row summation order where
  cheap (order may change if it pays). Check VGPR/LDS/scratch.
- **G6 runtime launch settings (conditional).** If a runtime setting such as
  device-resident kernargs (`HIP_FORCE_DEV_KERNARG`) or graph instantiation
  flags plausibly cut dispatch latency, implement it as the default in the
  appropriate `src/core/platform` leaf / HIP init path (no new CLI option), in
  its own commit, so the final pass can keep or revert it based on G0 and
  end-to-end numbers.
- **G7 persistent decode design (doc only).** Write
  `plan/exl3/archive/B3-PERSISTENT-DECODE.md`: what a persistent R=1 (and R≤8) layer
  kernel would look like for this model (stages, grid barriers per layer,
  register/LDS budget, which GEMVs stay separate), expected saving using G0's
  numbers once measured, and risks. No implementation.

Prefer G1, G4, G2, G3 (dispatch count), G5, then G0/G6/G7. Report dispatches
per token before→after per item (count from the code path), resources, lossy
changes, and the GPU tests/bench the final pass must run.
