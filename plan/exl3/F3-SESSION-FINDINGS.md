# Session findings (2026-10-31/11-01) that may not be in repo files
Source = agent reports in the supervisor session. Numbers are measured unless marked [estimate]. Verify against repo files where a commit/bench is named.

## Hardware behaviour (gfx1151 / RDNA3.5, Strix Halo)
- WMMA and VALU do NOT overlap on one SIMD: wave A WMMA-only 3.07–3.20 ms, wave B VALU-only 3.25 ms, both on same SIMD 6.44 ms = sum (bench tools/bench/gfx1151_wmma_valu_overlap.hip, f3-integ 5e20dc48). So producer/consumer decode→LDS-ring→WMMA wave specialisation cannot help; "WMMA → VALU dual-issue" conversion cannot raise the ceiling either (VALU dot2 dual ≈ WMMA FP16 peak ≈ 55–59 TFLOPS; decode VALU competes).
- LDS-fed WMMA rate vs fragment reads per WMMA: 2.0→28–30, 1.5→36–38, 1.25→39–44, 1.0→46–48, 0.75→46–49, 0.5→47–50, regs only→48–54 TFLOPS; ceiling ≈128 B/clk LDS per WGP (tools/bench/gfx1151_wmma_lds_bench.hip, a6847817). rocprofv3: zero LDS bank conflicts in production X3 tiles.
- WMMA f32_16x16x16_f16 does not cancel exactly: 1·1+(−1)·1 = −2^-24 (relative residue); test oracles must allow ≤2^-23 of |product sum| (6fde4b45). FP16 subnormal inputs lose relative precision in dot2/WMMA (X3-RESULTS / fp16 probes).
- Two-stream kernel overlap (GEMM vs bandwidth HC pass) only 12–20 % (bench 10c55dde). The initial assessment shelved micro-batch overlap for that mix; it does not cover the current EXL3 branch pairs. See the unmeasured [stream schedule audit](F3-STREAM-OVERLAP.md).
- Dirty-cache read penalty: a GEMM reading data written just before is ~2× slower. HC down 10240→320 M=2048: warm 0.77 ms, after 256 MB read flush 1.04, after 256 MB write flush 1.93 ms. Producer-side non-temporal stores (global_store … glc slc dlc) cut HC combine 1.47→1.00 ms/call and HC down to 1.78 ms; consumer-side NT loads: −11 % dirty but +48 % warm, +19 % read-flushed → not kept.
- flat_load vs global_load for runtime (struct-loaded) pointers: ISA changed, timing neutral (routed bucket).
- __builtin_amdgcn_iglp_opt(0) on dense X3 256×256: GDN out 2.19→2.02 ms (+8 %), others neutral (cc0dd4c5). iglp_opt(1) on bucket kernel: compiler crash.
- hipcc 7.2 miscompiles __builtin_bit_cast(std::uint16_t, half2.y) (stores low half). Avoid.
- No packed 16-bit mul-hi on RDNA3.5 → trellis hash already minimal 2-op form; byte sum via v_sad, FMA via v_pk_fma.
- rocprofv3 on this ROCm build exposes no SQ stall counters.

## Fast-math / bit-exactness traps
- Flash-Next kernels.hip.cpp is -O3 -ffast-math -fno-finite-math-only; plain prefill / qwen libs are -O2 without fast-math. Reproducing a fast-math kernel's bits in another TU requires matching its compiled association (read the ISA).
- Observed reassociations that broke bit identity: v*sc*gm compiled to v*(sc*gm); first lane-exchange folded into fma with partner's square (x²+x'² in one rounding); RmsNorm (x·scale)·γ became (x·γ)·scale; RoPE contracted differently. Fix: Materialize/pin the exact expression; write RoPE as fma(a,c,−bs), fma(a,s,bc).
- Unit tests can pass while the model differs: the inlined norm folded into neighbouring products only inside the model (hc4). The 8-window collector .bin byte check is the authoritative gate for "bit-exact" claims.
- A non-fast-math TU can still contract a*b+c into FMA (clang HIP default fp-contract).
- Device-side __fdiv_rn / float_control pragma inside a fast-math TU still gave 1-ulp-low scales → quantize on host.
- Materialize pins can serialize independent loads → hoist all loads before pinned products (recompute mix epilogue 139→93 ms/pass).

## GEMM kernels (gufo EXL3 / X3)
- Dense X3 256×256 8x4: 0.75 reads/WMMA, 31–36 TFLOPS (~61 % peak); 4x4 at 0.5 reads (242 VGPR, occ 5) no gain (±2 %). M=2048 vs 4096 linear. Reconstruct-to-FP16 + our FP16 core: same speed (core-limited, not decode-limited).
- kWideHalf follow-up (Nix10.1, 2026-11-01): masked256×256 at K2560/N640 lost19–26%; K32 plus smaller workgroups mostly lost. Single-group128×128 4×2 K32 was16.48% faster, but two groups2.34% slower. Pack confirms indexer parentK5 versus sharedgate/upK7; key prefill reaches the complete parent even before sparse attention. Subsequently qualified through2048 rows in F3-PREFILL-BATCH2.md; other routes unchanged.
- Codex-written FN DenseF16GEMMKernel has an M=4096 cliff (6144→2560: 26.0→15.5 TFLOPS, 3.35× time for 2× work; 640→2560 29.1→21.5). X3, plain-X3, bucket cores scale linearly. Production dense now on X3.
- hipBLASLt (docker ROCm 7.2.4): only 7 FP16→F32 solutions; best MT96x96x32 MI16x16 20–33 TFLOPS; heuristic default MT32x32x64 at 5.6 TFLOPS; pinned best is not shift-invariant (1-row shift changes bits) → not usable under chunk invariance. Our core matches/beats it except two M=4096 shapes.
- Routed bucket GEMM (2560→640, 512 K5 experts, 2048×10 assignments): base 3.6 ms; floor ≈2.05 ms (WMMA 1.45 ideal + decode ~0.6, serialized). Ablation: decode removed 3.25, weight fetch removed 3.4, both 2.85, WMMA removed 3.04, skeleton only 1.06 (0.84 without B staging), WMMA+LDS only 2.21–2.29, syncs removed no change, B path ≈0.5 ms. Decode ≈23 % of stage, ~38 VALU per trellis tile.
- Routed variants, all bit-identical: 8x1 eight-wave tile −7..−12 % (kept, df1f7409); hot-expert 128-token class −1..−8 % (kept, 56202401); 256-token class worse (238–247 VGPR, occ 4); 320-wide tile ~60 % slower (L2 hits halve but misses unchanged → re-reads were L2 hits); K128 stages 4–8 % slower; LDS double buffer 5.3–7.6 vs 3.5–4.6 ms; two-stage register prefetch neutral; per-wave private decode 4.2 vs 3.6; 16-B trellis staged through LDS 5.0; 128-row buckets for all slower at 2048; grid z changes none; padding-skip of 16-row subtiles kept (09f6b8a4); BHOIST fragment hoist 5–15 % slower; 2-deep fragment ping-pong +13–17 %; wave-parity stagger ~4× slower (guarded WMMA copies, >200 VGPR); B fragments direct from global 45 % slower; s_setprio neutral; persistent scheduling subsequently screened on Nix10.1 (2026-11-01,80/160blocks): exact, no qualified gain, artifacts retained; stream-K still untried.
- Flash-Next routing is heavily skewed at pp2048 (20,480 assignments, 512 experts): experts >64 tokens hold 51–67 % (early), 70–75 % (mid), 80–85 % (late) of assignments; max 360–1,483 tokens; 151–199 zero-token experts late (expert-counts.log in fn-hc9 receipts).
- HC down 10240→320: split-K slower (1.17 vs 0.91 ms; activation-read bound); finer tiles (64×64, 128×32, 64×32) slower; 96-block underfill not the cost; FP32-input fused fetch slower warm (1.466 vs 1.417); W8A8 Q8_0 HC down (UD's path) 1.78→0.46 ms/call, −109 ms/pass, q6 pass (8737db4e).
- Plain prefill operands (HC/router/PLE): FP32-input SmallGemmStable was 53 % of prefill; byte-stable row-tiled FP32 GEMM still ~1 TFMA/s; FP16 WMMA on all prefill rows fixed it (HC down 10.4→0.92 ms) — FP16 narrowing matched the precision the q6 quality had been measured with.
- Mix fused into HC-up GEMM epilogue: bit-exact but net +28 ms (fused GEMM 72→150–157 ms; mix 93→37) → reverted.
- MoE glue: MoveRows removed via fused scatter; fused restore+combine 3.10→1.42 ms/layer; per-token gate/up rotation 1.79→1.21 ms; fused activate+scale 0.67→0.60; MoE sum fused into HC combine −19 ms/pass; folding rotation into the GEMM producer modelled as +38 % on gate/up (rejected).
- Paired gate/up launch (UD RoutedGatedF16Gemm style): no shared operand for EXL3 (different suh per expert) → ≈−10..−20 ms only; shelved.

## GDN, attention, HC (Flash-Next)
- GDN prefill ≈650 MB/call at ≈220 GB/s (bandwidth-bound overall); row-split VALU-bound (1,833 instr: 360 v_fmac, 123 v_dual_fmac, 271 s_delay_alu); fixed FMA order limits VOPD. conv+PrepKq fused into row-split −15 % (3.3→2.8 ms/call); head remap no change; 8 waves/block slower (3.63 vs 3.30). Chunked DeltaNet algebra forbidden (EXPERIMENTS.md:3).
- HC combine four-lane at every EXL3 prefill width (c8625608); F16 xn written once + recompute in mix (0d051491) −85 ms/pass.
- Attention extras (UnpackQGate, q norm, rope, rotations, SigmoidMul) fused, 45→23 ms/pass, bit-exact (2e054760).

## Decode (Flash-Next EXL3 5.05, tg128 ≈24.7 t/s vs UD 29.2)
- Per token ≈5.9 GB weights; Exl3Gemv trunk/head shapes already 205–221 GB/s; routed experts 141–158 GB/s; HC plain F16 126–165 GB/s (HC = 1.26 GB/token, 21 % of bytes).
- Routed decode GEMV 4-tile-ahead fetch: 216→271 GB/s, ≈−1.3 ms/token (04269373). Row-tiled stable GEMV for HC/router (D1) slower; K7 PickShape: production shapes already best (only shared down 2x8 −0.1 %).
- HC int8 (Kyojin group format g128/g64, plain rounding): tg +3.5 % d0 / +2.4 % d32K, KL +0.0019, CI [+0.0003, +0.0034] (upper bound fails the ≤+0.002 bar; F1-RESULTS.md:506), +659 MB → dropped (user bar: ≥+10 %).

## System / process
- Packs must be on internal NVMe: 5.05 load 300 s from USB (370 MB/s, nothing page-cached) vs ~30 s internal; n-gram table is pread at runtime.
- n-gram gather is not on the pp2048 critical path (6 ms idle in a timed pass); first-request lazy code-object gaps 27/11/11 ms (TTFT only).
- Docker RelWithDebInfo build vs nix release: UD pp2048 1,567 vs 1,762 (~11 % lower) → docker probes are guides only.
- rocprofv3 7.2.4 (image) cannot trace the nix ROCm 10.1 binary (two HSA runtimes); building nix rocprofiler-sdk pulls a ROCm clang source build (hours) → profile a docker build instead.
- rocprofv3 counter replay across many bench variants runs far too long; use one dispatch, in-container `timeout -s KILL`, and the lock; killing the docker client leaves the container running.
- Docker image gufo-eval:64849ca6ec7b needs --entrypoint bash; --group-add 44 990 for /dev/kfd; exFAT external disk cannot hardlink; root must stay ≥150 GiB for windows.
- pgrep -f "<cmd>" wait loops match themselves; use background-task notifications.
- window.py refuses steps whose timeout exceeds the remaining window budget.

## Packs / quality (Flash-Next, vs HF BF16, 32,760 rows, FP16 KV)
- UD-Q4_K_XL KL 0.0392 top-1 90.9 %; EXL3 5.05 KL 0.0264–0.0269 top-1 92.1 %; turboderp 4.05 KL 0.0424 (≈UD); Kyojin pack (yamz-labs, actually ≈4.0 bpw, not 3.05) KL 0.0598 top-1 87.3 %.
- EXL3 vs UD prefill class table (pp2048 ms/pass, same docker binary): EXL3 wins dense −124..−135 and GDN −66; loses HC, MoE glue, routed.
