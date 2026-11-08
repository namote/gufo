# F3 — Flash-Next EXL3 speed plan

2026-10-31; reviewed HEAD `3712dbe7`. Read-only planning; no benchmarks run.

**Recommendation:** run three streams: shared K7/X3 primitives, Flash-Next expert kernels, and executor/MTP integration. Prioritize replacing reference prefill, accelerating the K7 trunk, and removing full-body cache catch-up. Expert-only tuning cannot close the whole gap.

Below, `K/` means `/home/m2on/projects/gufo/build/w9-src/kyojin/`. Reported measurements are source evidence; proposed gains are estimates.

## Targets and present gap

Every target must pass individually; averaging categories is insufficient.

| Workload | Required tok/s | Maximum ms/output token | Historical Gufo comparison* | Required rate gain |
|---|---:|---:|---:|---:|
| Chat MTP | 52.8 | 18.94 | 40.5 | 30.4% |
| Prose MTP | 47.5 | 21.05 | 38.6 | 23.1% |
| Code MTP | 73.2 | 13.66 | 56.4 | 29.8% |
| 8K MTP | 52.2 | 19.16 | 39.0 | 33.8% |
| 32K MTP | 51.1 | 19.57 | 39.8 | 28.4% |
| Cold 8K prefill | 1621 | 0.617/prompt token | No matched EXL3 result | Measure first |

\*W9 GGUF results, **not this EXL3 revision**, and server timing rather than Kyojin’s client timing (`plan/archive/rotated-quant/W9-RESULTS.md:107`). Published GGUF d0 pp2048/AR are 1801/28.29 tok/s; neither establishes cold-8K EXL3 performance (`docs/models/qwen3.8-flash-next/BENCHMARKS.md:17`).

Kyojin’s card recipe uses greedy, thinking off, 256 generated tokens, four prompts/category and median-of-three repetitions. Its depth recipe uses cold prompts and client-timed Qwen decoding (`K/doc/benchmarks.md:11`). Reproduce that scope alongside Gufo’s standard pp2048/tg128 benchmark.

The requested approximately 6.0 versus 4.6 GB/token handicap is about **31% more weight traffic**. It does not relax the targets. There is no measured current-EXL3 gap yet.

## Byte and time budget

### AR and MTP

Calculated decimal weight bytes, excluding activation, recurrence, KV, PLE I/O and launch overhead:

| AR component | GB/token | At 200 GB/s |
|---|---:|---:|
| Routed top-10 experts | 1.484 | 7.42 ms |
| HC, included in always-used weights | 1.28 | 6.40 ms |
| Remaining always-used weights | ≈3.25 | 16.25 ms |
| **Total** | **≈6.02** | **≈30.1 ms** |

Thus about **75% is always-used traffic**, with HC alone about 21%. At 200 GB/s plus 2–7 ms other work, AR is approximately 27–31 tok/s—not a promise to beat GGUF (`plan/exl3/archive/F2-DESIGN.md:202`).

For verification, use the actual number of expert tiles read:

`Bverify ≈ 4.53 + 0.1484 × U GB`

Here `U` is distinct experts per layer when each fits one eight-row tile. For larger cohorts, count repeated tiles too. Uniform independent routing gives:

| Verify rows R | Expected U | GB/verify | Weight time at 200 GB/s | Ideal completed tok/s† |
|---|---:|---:|---:|---:|
| 2 | 19.80 | 7.47 | 37.35 ms | 49.6 |
| 3 | 29.42 | 8.90 | 44.48 ms | 59.5 |
| 4 | 38.84 | 10.29 | 51.47 ms | 66.2 |
| 8 | 74.74 | 15.62 | 78.11 ms | 80.8 |

†Perfect acceptance, `R−1` drafts, full heads, zero other overhead. Each draft budgets approximately 0.119 GB body plus up to 0.477 GB head. These are scenarios, not forecasts (`plan/exl3/archive/F2-DESIGN.md:219`).

**Code-target constraint:** four completed tokens require a cycle ≤54.64 ms. Independent-route verification alone consumes 51.47 ms; three full-head drafts add 8.94 ms. At 200 GB/s this scenario cannot reach 73.2 tok/s even before overhead. Compact heads, correlated routing, faster streaming, or a better measured cycle are necessary.

Instrument per cycle: committed tokens, drafts, verify rows, expert-tile reads, selected head segments, catch-up rows, phase times and wall span. Optimize `committed tokens / complete cycle time`, not acceptance fraction.

### Prefill

For a chunk of P rows, an optimistic once-per-chunk weight-read budget is:

`Bpp/token ≈ [4.53 + 75.98 × (1 − (1−10/512)^P)] / P GB`

| Chunk rows | Mean rows/expert | Ideal MB/prompt token | Weight-only time at 200 GB/s |
|---|---:|---:|---:|
| 512 | 10 | 157 | 0.786 ms |
| 2048 | 40 | 39.3 | 0.197 ms |
| 8192 | 160 | 9.83 | 0.049 ms |

These are **reuse lower bounds**, not predictions: output tiles reread weights, buckets are ragged, and compute/state traffic remains. An 8K request processed as four 2K chunks retains the 2K budget.

Current prefill has additional FP32 dequantization traffic and per-expert submission: both dense and expert projections fall back to reference execution (`src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hip:326`; `executor.cpp:3412`). Do not treat those temporary-buffer bytes as compulsory DRAM traffic; bounded slabs may remain cached.

For scale, an earlier GGUF pp2048 profile spent approximately 0.189 ms/token in experts, 0.171 in dense F16 GEMMs, 0.065 in HC combines, 0.047 in DeltaNet, 0.030 in W8A8 and 0.017 in attention (`docs/models/qwen3.8-flash-next/EXPERIMENTS.md:100`). The EXL3 target is **≤0.617 ms/token including applicable predictor catch-up**; profile the new path before assigning tighter phase budgets.

## Ranked optimizations

Expected gains below are engineering estimates, are not additive, and require complete-model confirmation.

| Rank | Concrete change and source | Expected gain | Risk and correctness bar |
|---|---|---|---|
| **1** | **Replace reference prefill with X3.** W3 supplies trellis→LDS WMMA; expert worker adds device bucket descriptors, gather/scatter and bucket-row dispatch. Start with 16/32/64-row tiles and preserve 640-wide expert outputs. **Gufo-27B.** | Several-fold improvement in affected reference projections; total uplift unmeasured. Essential prerequisite for 1621. | Medium/high. Current X3 accepts only K≤6 and `n%256==0`; K7 and 640 outputs need implementation (`prefill_fp16.hip:1028`). Keep FP32 butterflies, specified FP16 boundaries, output rotation and ordered MoE combination. |
| **2** | **Tune K7 dense AR/verify at actual Flash-Next shapes.** Reuse X2b grouped q/k/v, qkv/z and fused shared MLP. Compare current dot2 against WMMA for R=2–8; optimize cyclic K7 extraction and inspect LDS banks/spills. **Gufo-27B + Kyojin.** | Target ≥200 GB/s on cold production shapes; estimated 0–10% whole-cycle gain until profiled. | Medium. K7 currently extracts individual cyclic windows (`src/core/hip/detail/exl3_gemv.hpp:147`). X3’s three-dword producer cannot gain K7 merely by changing its guard (`prefill_fp16.hip:427`). Preserve scalar/batched arithmetic; do not silently route verification through prefill WMMA. |
| **3** | **Recover packed routed-down arithmetic, then fuse its producer chain.** Scale the already-rounded FP16 row by an exact power of two, undo before restoration, and compare against widened-FMA control. Fuse restored gate/up→SwiGLU→down rotation; specialize actual R, especially 3. **New + Gufo-27B + Kyojin union idea.** | Down kernel approximately 1.3–2×; estimated 2–8% complete cycle. | High numerical risk. Zero rows, overflow headroom and remaining subnormals need explicit handling. Keep widened FMA for unsupported cases. Require original stage limits, independent FP64 and producer-bit checks; scaling before FP16 rounding is a separate arithmetic change. Current widened route: `exl3_moe.hip:97`, `:422`. |
| **4** | **Implement EXL3 K/V-only MTP catch-up and retain compact-head eligibility.** Group separate K/V projections, update indexer/cache, skip unused Q/O/FFN and later HC work. Recalibrate EXL3 cycle costs only after kernels settle. **Gufo existing behavior + new integration.** | Estimated 50–90% reduction in cache-only body work; whole-cycle benefit limited by measured catch-up share. | Medium state risk. Both scalar and batch guards currently require packed `attn_qkv`, excluding EXL3 (`executor.cpp:5109`; `batch.cpp:252`). Compare every surviving KV/indexer row and subsequent logits; preserve final carried hidden state and sampled replay. |
| **5** | **HC traffic reduction study.** First improve exact skinny F16 scheduling/fusion. Separately evaluate Kyojin-style per-row Q8 HC weights against the quality gate. **Kyojin + Gufo skinny kernels.** | Q8 could save ≈0.64 GB/forward, about 3.2 ms at 200 GB/s; exact scheduling likely smaller. | High for Q8: it changes weights. Kyojin reports 36.1→40.0 tok/s after mixer quantization (`K/README.strix-halo.md:82`), but unchanged single PPL is insufficient. No gain booked until full-vocabulary quality/tails pass. |
| **6** | **Larger prefill chunks and exact HC/GDN fusions.** After bucket WMMA works, test 2K→4K chunks, then justified 8K. Preserve checkpoint cadence and cancellation responsiveness. Fuse only existing arithmetic/rounding boundaries. **Kyojin + Gufo.** | Estimated 2–10% prefill; shape dependent. | Medium/high state risk. Kyojin’s big-chunk rationale is increased rows/expert (`K/exllamav3/generator/job.py:1314`). Do not import its KDA algebra as a GDN replacement. |
| **7** | **Host/launch cleanup after profiling.** Persistent descriptors, fewer redundant resets/copies, graph coverage verification. **Kyojin + Gufo.** | Expect 0–2% unless wall/GPU gap proves otherwise. | Low/medium. Preserve descriptor lifetime, concurrent scratch isolation and capture coverage. |

### What transfers—and what does not

- **Already present:** SAD decoding, grouped dense launches and fused shared MLP. SAD is explicitly used in `src/core/hip/detail/exl3_gemv.hpp:113`; do not claim another SAD gain.
- **Kyojin WMMA evidence:** Flash-Next approximately 17→27.9 tok/s versus reconstruct/GEMM fallback; LDS padding later 32.7→34.9, skinny projections 34.9→36.1 (`K/README.strix-halo.md:75`). These do not predict uplift over Gufo’s optimized dot2 path.
- **Union verification:** Gufo already builds device expert tiles (`exl3_moe.hip:43`). Kyojin’s R=3 specialization reports +1.46% served throughput, but in the GLM measurements (`K/README.strix-halo.md:44`). Its union kernel requires intermediate width divisible by 512, so Flash-Next’s 640 cannot use it unchanged (`K/exllamav3/exllamav3_ext/quant/exl3_dec.cu:3051`).
- **Prefill fusions:** Kyojin’s cited HC fusion gain is GLM +1.46–1.58%, not Flash-Next (`K/README.strix-halo.md:59`). Its KDA defaults lack an isolated Flash-Next gain in that note.
- **Avoid revived failures:** Gufo removed chunked DeltaNet after a tool prompt flipped a 1.4-nat greedy decision (`docs/models/qwen3.8-flash-next/EXPERIMENTS.md:3`). Also inspect generated code before tuning: bounded scheduling previously recovered 27B Q8 pp2048 from 291→464 tok/s by removing compiler spills (`docs/models/qwen3.8-27b/EXPERIMENTS.md:130`).

## Three parallel workstreams

File ownership is exclusive; primitive APIs are agreed first, and the integration owner wires them.

| Owner | Owned files | Deliverable |
|---|---|---|
| **W3: shared primitives** | `src/core/hip/detail/exl3_gemv.hpp`; `src/models/qwen/hip/kernels/{exl3_gemv.*,prefill_fp16.hip}`; `src/models/qwen/hip/ops/prefill_fp16.hpp`; shared `exl3_dispatch.hpp`; 27B EXL3 tests | K7 prefill producer, narrow/tail-capable WMMA API, dense decode/verify tuning. Preserve 27B/GGUF controls. |
| **Codex expert worker** | Flash-Next `kernels/rocm/exl3_moe.{hip,hpp}`; new dedicated Flash-Next expert-kernel test source | Scaled routed-down, fused producer chain, bucket kernels and device descriptors. No executor edits. |
| **Codex integration worker** | Flash-Next `executor.{cpp,hpp}`, `batch.cpp`, `device_model.{cpp,hpp}`, `kernels.hip.cpp`, `kernels.hpp`, model/test CMake, existing model operator tests and `mtp_costs.hpp` | Dispatch/scratch integration, K/V-only catch-up, HC experiments, chunk policy, profiles and final qualification. |

## Promotion gates

1. Establish retained EXL3 production baseline and measured hardware ceiling; profile separately. Rotate cold weights beyond MALL and match production allocation.
2. Keep independent decode-stage 1e-5 checks and the existing frozen-input E2E contract (`plan/exl3/archive/F2-IMPLEMENTATION.md:167`). X3 has a distinct documented WMMA arithmetic bound; **do not widen either gate to accept a candidate** (`plan/exl3/archive/X3-RESULTS.md:64`).
3. Cover awkward rows/tails, shared/disjoint routing, exact producer boundaries, finite outputs, graph replay and allocation-boundary canaries. Preserve HC’s full 10240-wide normalization; the existing 5120-wide RMSNorm fusion is not directly reusable (`exl3_gemv.hpp:107`).
4. Qualify fixed-history full-vocabulary KL/ΔNLL/top-1/tails, real MTP proposals and every rejection prefix; seeded sampling, EOS, images, cancellation, continuation, snapshots, `long-context` and `cache`. Keep the upstream MTP normalization mismatch visible (`plan/exl3/archive/F2-DESIGN.md:159`).
5. Retain standard speed and numerical-quality tests. Compare affected requests/phases at **5% and 10 ms**, retaining inconclusive results. Include C1 and affected C2/C4/C6/C8, plus an actual `pi_agent.py` workflow.
6. Publish only when all requested speed cells pass and quality passes. Successful kernels become defaults; remove rejected routes and experimental switches.

## F3-API (W3, 2026-10-31)

Model-independent shared EXL3 primitives for the Flash-Next expert and integration workers. Headers:

- `src/core/hip/detail/exl3_gemv.hpp`: device decode
- `src/models/qwen/hip/kernels/exl3_gemv.hpp`: decode/verify GEMV and rotation
- `src/models/qwen/hip/ops/prefill_fp16.hpp`: WMMA GEMM

All take `quant::exl3::DeviceLinear` or raw device pointers in checkpoint layout (trellis `[k/16][n/16][16K]`, FP16 `suh[k]` and `svh[n]`). The codebook must be mul1.

### Device decode (`exl3_gemv.hpp`, namespace `gufo::hip::exl3`)

- `MakeLaneWindow<K>(lane)`: K1–K7.
- `LaneStates<K>(d0, d1, d2, shift, state[8])`: the lane's eight 16-bit states from its three dwords. K7 takes the 65-bit window plus bit 64, so it no longer needs a separate per-state extraction.
- `HashMul1Packed(src)`: full-rate hash.
- `DecodePairMul1(p0, p1)` and `DecodeLaneMul1<K>(tile, lw, w[4])`: bit-exact with the CPU `DequantTile` (the kernel test checks K1–K7).
- `Butterfly128`.
- Lane layout: lane L holds positions 8L..8L+7, at rows r0, r0+1, r0+8, r0+9 (r0 = 2(L%4)) of columns L/4 and L/4+8.

### Decode and verify (GEMV, rows 1–64 in 8-row passes)

- `Exl3FastWorkspace`, `Exl3FastTarget`, `Exl3FastSupported` (mul1, K1–7, k and n multiples of 128).
- `LaunchExl3FastProjection(s)`: up to 3 matrices with the same k and rate, one launch.
- `LaunchExl3FastMlp`: gate/up, then SwiGLU, then the down input in the epilogue.
- `LaunchExl3FastRmsNorm`: 5120-wide only; Flash-Next's 2560/10240 do not apply.
- Per-row results do not depend on the batch.
- Arithmetic: FP16 rotated input; `v_dot2acc_f32_f16` with FP32 accumulation; `svh` and FP32 butterflies at the output.
- Subnormal FP16 inputs lose accuracy (X3-RESULTS "dot2 명령 형태와 서브노멀").

### Rotation

- `LaunchExl3RotateRows(x, x_stride, rows, k, suh, out)`: FP32 → FP16 `H(suh ⊙ x)`, row-major `[rows][k]`.
- `LaunchExl3RotateRowsGrouped(problems, count, max_rows, k)` with `Exl3RotateProblem {x, x_stride, row_index (gather, nullable), suh, out, rows}`. One launch for all experts' rows.

### WMMA GEMM (prefill and expert buckets, `ops/prefill_fp16.hpp`, namespace `gufo::hip`)

- `Exl3GemmShapeSupported(k, n, rate, codebook)`: mul1, integer K1–7, n and k positive multiples of 128.
- `LaunchExl3PrefillGemm(linear, x_fp16_rotated, y, rows, residual)`.
- `LaunchExl3PrefillGemmGrouped(problems, count, max_rows, k, n, rate, residual)` with `Exl3GemmProblem {trellis, svh, x, y, rows}` (a device array; `rows = 0` skips). All problems share k, n and rate.
- Tile, chosen by rows (`max_rows` for grouped):
  - ≤ 64 rows: 128 channels × 64 tokens (the weights are read once)
  - otherwise: 256 × 256, or 128 × 256 when n % 256 ≠ 0 (640-wide experts)
- Output: `y[r][n]` FP32 = `svh ⊙ H(xW')`, residual: `+= `. `y` is dense `[rows][n]`; scatter/combine belongs to the caller.
- Error bound: relative L2 ≤ (k / 16) × 2⁻²³ against FP64 fed the same FP16 rows (WMMA accumulation, X3-RESULTS).
- Not covered: the grouped gate/up SwiGLU fusion (the expert worker's producer chain), and a per-row power-of-two scale.

### Guarantee

- The GGUF `HalfPrefillGemmKernel` instances are unchanged in resources and instruction text (ISA comparison).
- The 27B GEMV/prefill kernels changed only in instruction scheduling of the integer decode (same ops), so the results are the same. The 27B kernel test values and fingerprints are reconfirmed in the next unit window.

### F3-API status (W3, 2026-10-31 17:50)

- **#1:** done. The GPU unit window f3u1 (45462f43) passed every production check:
  - K7 2560→512 and 6144→2560 GEMV
  - K7 prefill at 300 and 33 rows
  - K5 640-wide (128×256 tile)
  - 640→2560
  - grouped 7/33/300
  - 27B fingerprints identical to rq2h
- **The new lane-decode test failed** because of a compiler miscompile, which is a warning for workers. hipcc 7.2 (gfx1151, true16) compiles `__builtin_bit_cast(std::uint16_t, half2.y)` to a `global_store_b16` of the low half. Verified in the ISA. Split the pair as a dword (`bit_cast<uint32_t>(pair) >> 16`) instead. The fixed test stores correctly (`global_store_d16_hi_b16`); it will be confirmed in the combined pass.
- **#2 code:** the shared K7 lane window takes the K7 GEMV kernel (M8 W4 TN2) from 1379 to 1015 instructions, the same as K6 (1010). Neither has spills. VGPRs: 84 for K7 vs 81 for K6.
- **Remaining #2 work goes to the combined pass:**
  - FN-shape timing with `tools/bench/exl3_k7_bench.hip`: GEMV vs WMMA at R=1..8, and the 2×4 / 2×8 / 2×16 / 4×4 shape sweep with checks against production output.
  - Tuning `PickShape` from those results. With few blocks (o, shared down: 80 blocks), wider waves are the candidate.

### F3-API: expert bucket raw inner GEMM (W3, 2026-10-31)

- **Header** `src/models/qwen/hip/ops/exl3_prefill_bucket.hpp`, HIP-free: `gufo::hip::Exl3PrefillBucket {expert, count, first}` (12 B). The expert worker aliases its `Exl3ExpertBucket` to this type with `using Exl3ExpertBucket = gufo::hip::Exl3PrefillBucket;` so the `Exl3BucketGemm` function pointer matches without a cast.
- **Entry point** `LaunchExl3PrefillBucketInner(bank, buckets, bucket_count, max_buckets, tile_rows, input, inner, k, n, stream)` in `ops/prefill_fp16.hpp`:
  - Bank: mul1 K5; k and n multiples of 128.
  - Input: FP16 rotated, packed `[rows][k]`. Output: raw FP32 inner products, packed `[rows][n]`, with no H, svh or suh.
  - Masking: descriptors at or past the device count (`*bucket_count`, read on the device) are ignored; rows past `count` are neither read nor written; nothing is padded.
  - Bank-entry check on the device: an entry with the wrong rate, k, n or codebook writes NaN to that bucket's rows.
- **Tiles:** 128 channels × 16, 32 or 64 tokens, the smallest that holds `tile_rows`. The grid is `(1, n/128, max_buckets)`. No spills. VGPRs are 129, 170 and 218, for occupancy of 10, 8 and 6.
- **Arithmetic:** every tile accumulates each element over k in the same WMMA order, so values do not depend on the bucket partition or the position in a tile. The test checks this bit for bit.
- **Test:** `qwen27b_exl3_kernel_test` `TestBuckets`, for 2560→640 and 640→2560. Experts have 1, 37, 0 and 70 rows, with partitions of 16, 32, 64 and 7 rows. Checks:
  - bit identity across partitions
  - FP64 at the (k/16)·2⁻²³ bar
  - rows outside every bucket stay untouched
  - descriptors past the count are ignored
  - K6 entry poisoning
- **Status:** built; the GPU run waits for the combined pass.

### F3-API: byte-stable plain prefill GEMM (W3, 2026-10-31)

- **API:** `gufo::hip::LaunchPlainPrefillStable(core::GgmlType type, weights, input, output, rows, n, k, stream)`. Header: `src/models/qwen/hip/ops/plain_prefill_stable.hpp`. Kernel: `src/models/qwen/hip/kernels/plain_prefill_stable.hip`.
  - Weights: F16, BF16 or F32, row-major `[n][k]`. Input and output: FP32, packed `[rows][k]` and `[rows][n]`.
  - Ragged rows, n and k are masked. Rows are split across launches in chunks of 65535×16.
- **Arithmetic:** the output is byte-equal to Flash-Next `SmallGemmStable` (f3-integ 03a6afba) for every row count and tile position. It reproduces the same per-output reduction:
  - Lane class L keeps an fmaf chain from +0, in element order, over elements 4L..4L+3 of every 128-element step. A step starting at or past k runs no FMA (the reference's `break`), which preserves underflowed −0 chains. A partially valid step runs all four FMAs with zeros.
  - The xor butterfly (16, 8, 4, 2, 1) follows, and lane 0's sum is stored.
- **Not included:** the fused HC-down SiLU. Apply the existing `SiluScale(…, 0.25F)` kernel after this GEMM; it matches the fused epilogue byte for byte.
- **Tiling:**
  - Blocks: 16 tokens × 32 weight rows, 8 waves.
  - LDS: one 128-element step of the block's inputs and widened weights, double-buffered (48 KiB).
  - Registers: each lane extends the chains of an 8 tokens × 8 rows tile.
- **ISA:** no spills, 140 VGPRs, occupancy 4. Per step: 256 `v_fmac_f32` (including dual issue). Reduction: 320 `v_add_f32`. No `v_mul_f32`, so no other contraction. FP32 denorm mode 3, the same as the reference.
- **Test:** `qwen27b_plain_prefill_stable_test`.
  - Reference: a verbatim extraction of the pinned SmallGemmStable code (`tests/models/qwen27b/small_gemm_stable_reference.hip`), built with Flash-Next's `-O3 -ffast-math -fno-finite-math-only`.
  - Shapes: F16 320×10240, BF16 67×2597, F32 40×300 and F16 512×2560, at rows 1/8/9/64/65/2048 with offsets 0 and 1.
  - Checks: byte equality, finite outputs, no writes past the rows, an FP64 oracle at (k/32 + 5)·2⁻²³ Σ|wx|, and −0 chains.
  - Status: built; the GPU run waits for the combined pass.
- **Speed estimate** (unmeasured) at 2048×320×10240, 6.7 G FMA:
  - The FP32 VALU ceiling is about 7–15 T FMA/s. With LDS reuse the kernel is compute-bound, about 1–2 ms.
  - SmallGemmStable re-reads the weight row for every 8-token group (about 1.7 GB of weights) and the input slab for every 4-row block (about 6.7 GB). That is cache-bound, about 8–15 ms.
  - Estimated speedup: about 5–10×. The combined pass measures it.

## Status at hold (2026-11-01 14:30)

Code lives on `ai-tools/f3-integ` (HEAD `966f84b8`, not merged into port-u4f yet). All prefill changes are chunk-invariant; bit changes passed the q6 gate (5.05 vs BF16 KL 0.0264, top-1 92.1 %).

| Probe (docker 7.2.4 RelWithDebInfo, 5.05 + KVarN) | pp2048 | pp8192 | tg128 |
|---|---:|---:|---:|
| 68d4e3c8 (start) | 454 | 466 | 24.7 |
| hc7/att2 (now) | ≈1,255–1,260 | ≈1,290–1,324 | ≈24.7 |
| UD-Q4_K_XL, same docker binary | 1,567 | — | — |
| UD-Q4_K_XL, nix | 1,762 | 1,796 | 29.2 |

Not yet measured end to end: W3's 8x1 tile, hot-expert buckets and att2 together on `966f84b8`.

Remaining pp2048 gap vs UD (kernel ms/pass, hc7 trace): HC combine/mix +119, MoE glue +71, routed GEMM +69, attention non-GEMM +32 (att2 ≈ −22), HC GEMMs +24; EXL3 wins on dense (−135) and GDN (−66).

Measured dead ends (numbers in the agent reports and benches): HC int8 decode (+3.5 % tg, below the +10 % bar), split-K/320/LDS double buffer/fragment ping-pong/stagger/global-space trellis loads, producer–consumer decode (WMMA and VALU do not overlap on a SIMD), two-stream micro-batch overlap (12–20 %), hipBLASLt (not faster, not shift-invariant), mix fused into HC up (+28 ms).

Next on resume: one T1 probe on `966f84b8`; then the HC combine/mix gather latency, the remaining MoE glue, the kWideHalf (n % 256 ≠ 0) dense tile at 20–29 TFLOP/s; then the nix final images and the combined final pass (q3, q6, FN timing, 27B timing).
