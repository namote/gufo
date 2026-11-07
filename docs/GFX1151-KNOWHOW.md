# gfx1151 optimization know-how

## Flash-Next B4 serial tails undone, routed LDS staging, November 8, 2026

Commits `517445a3..dc24fd89` (brief `plan/exl3/B4-BRIEF.md`). Kept: G1 HC
tails reverted (H1); device-kernarg default reverted (H2, no effect in B3);
routed GEMV activations staged in LDS per K chunk so the K loop keeps the next
trellis group in flight (H4a); HC down finishes each Q8_1 group in its own
block, no counter, SiLU launch removed (H3). Reverted after measurement
(structure vs speed): G3/H5 conv q/k norm, G4a router+top-k, G4b/H4b routed
restore in the gate/up GEMV, G4d MoE output in the next HC combine — together
−178 dispatches/token and −1.8 ms idle, but +1.9 ms busy (G4d alone +0.51 ms,
a redundant per-stream combine), AR within ±0.4%, and H4b changed decode bits
(d0 hash `75a056e35e70`, not caught by `exl3_moe_kernel`), which made MTP d0
7% slower through a different text path. HC up+mix fusion skipped by design
screen (one block owning a 256-column chunk needs 1024 gate rows on 10 WGPs).

Same window, c03, Nix10.1, FP16 KV, B C S rotation r2, baseline B2
`9599b4c6`, adopted tree S: AR128 d0/d32K 31.77/31.09→32.49/31.75
(+2.3/+2.2%); MTP128 d0/d32K 63.95/99.01→67.39/102.75 (+5.4/+3.8%); PP
2048/8192 unchanged (1508/1541). Outputs, hashes (`6d067a1254ff`,
`d856f78e53b9`) and MTP work counts equal B2; prefill logits equal B3 (quality
unchanged, KL256 0.041742).

HC decode bench (`tests/.../qwen38_flash_next_hc_decode_bench`, µs/call, R=1/4/8):
B2 down+SiLU 20.78/21.37/21.83 vs fused down 20.08/20.09/22.35; up 18.4–22.5;
mix 2.8–3.0; chain 41.8→40.0 at R=1. Decode profile AR d0 (S): 30.93 ms,
1273 dispatches/token, kernel-active 66.8%, busy 20.65, idle 10.28 ms/token;
routed gate/up 4.54 ms ≈174 GB/s + restore 0.22, down 2.55 ≈155 GB/s (B3 5.08
≈155 / 2.66 ≈148); HC chain 3.87, combine 0.68. Lesson confirmed: on this GPU a
removed dispatch is worth ≈3 µs idle; a fusion only pays if busy time does not
grow, and redundant or serial tail work quickly exceeds that. Evidence:
`build/b1/gpu-b4/RESULT.json`.

## Flash-Next B3 decode dispatch fusion and launch gaps, November 8, 2026

Commits `addf9dd3..5bda5f57` (brief `plan/exl3/B3-BRIEF.md`): HC-down tail
SiLU/Q8_1 and HC-up mix-epilogue tail (G1, −192 dispatches/token); router GEMV
last block runs top-k + tiles (G4a, −48); restore/scale in the gate/up GEMV
tail (G4b, −48); decode MoE output formed in the next HC combine (G4d, −47);
one-launch attention prep (G2, −72); q/k norm in SsmConv4's last block (G3,
−36); routed GEMV next-group prefetch and global trellis loads (G5);
`HIP_FORCE_DEV_KERNARG=1` default (G6); persistent-decode design doc (G7);
single-row Q8_0 GEMV rounding unified with batched rows (`6b09c93c`).

Same window, c03, Nix10.1, FP16 KV, B C B C r2, baseline B2 `9599b4c6`:
PP2048/8192 off 1519.5/1551.1→1519.1/1549.4; MTP-PP 1451.1/1476.3→
1450.4/1475.6; AR128 d0/d32K 31.74/30.99→31.36/30.69 (−1.2/−1.0%); MTP128
d0/d32K 63.92/98.74→62.17/95.69 (−2.8/−3.1%). All below the 5%/10 ms gate but
consistently slower on decode. AR/MTP hashes, MTP work counts and prefill
logits equal B2; quality unchanged (KL256 0.041742).

Decode profile (AR d0): 998 dispatches/token (B2 1442), kernel-active
67.0→71.5%, idle 10.5→9.11 ms/token, but kernel busy 21.3→22.9 ms/token. Removing
444 dispatches saved only ≈1.4 ms idle (≈3.2 µs each). Busy by group (ms/token,
B2→B3): HC chain q8+combine/mix 4.64→6.23 incl. 0.45 moved MoE combine
(HcUpMix 2.60 vs Q8 up GEMV 1.71; HcDownSiluQ8 1.98 vs 1.71) — G1's
last-block tails serialize and cost more than the gaps they remove; router
GEMV+top-k 1.24→1.25 (neutral, −48 launches); routed gate/up+restore
≈5.3→5.08 (≈155 GB/s), down 2.63→2.66 (≈148 GB/s); attention 0.74→0.54.
Isolation (TG, interleaved): G6 env 0 vs 1 ±0.06% (no effect); G5 reverted is
slower by 0.2–0.3% AR, 1.5–1.9% MTP (keep G5).

Launch gap microbench `tools/bench/gfx1151_launch_gap.hip` (µs/dispatch,
1500 dependent tiny kernels): stream 1.95–2.16, captured graph 1.83–1.98;
1 KiB vs 16 B kernargs and 1×32 vs 40×256 grids make no difference; a forked
side stream adds ≈0.1. `HIP_FORCE_DEV_KERNARG` changes nothing. Cooperative
grid barrier: 0.91–0.93 µs at 1 block/CU, 0.94–0.97 at 2, 0.98–1.10 at 4 (tool
reports 20 compute units — WGP count). So the ≈7 µs/dispatch decode idle is
mostly ramp/drain and dependency latency, not launch cost; a persistent kernel
saves launch cost ≈0.9 µs/stage but only wins if its stages avoid the same
serial tails. Fusion pays only when the fused kernel's busy time does not grow.

Bugs found by the pass: inlining SsmConv4 let fast-math reorder its 4-tap FMA
chain (projection_ops/gdn_ops bit mismatch); fixed by calling the pinned
`SsmConv4Value` (`abd66204`). Evidence: `build/b1/gpu-b3/RESULT.json`,
`prof/decode-summary.json`.

## Flash-Next B2 producer rotation, verify state and side stream, November 7, 2026

Commits `c49ea976..9599b4c6` (brief `plan/exl3/B2-BRIEF.md`): producer epilogues
write the rotated X2b input for R≤8 (E1, −144 rotate launches/token); exact
R5–R8 routed live-row bodies (E4, MaxRows-8 VGPR 39–69→66–84, no scratch/LDS);
GDN verify keeps state and replays the kept prefix from per-token records
instead of a 3.1 MB/layer snapshot (E6, −≈113 MB/session); EXL3 decode shared
MLP on a captured side-stream branch beside the routed chain (E7); EXL3 prefill
attention-combine residual write deferred to its readers (E2).

Same window, c03, Nix10.1, FP16 KV, B C B C r2, baseline B1 `cc86343e`:
PP2048/8192 off 1507.96/1534.37→1516.66/1547.74 (+0.6/+0.9%); MTP-PP
1440.77/1460.82→1450.14/1476.09; AR128 d0/d32K 31.04/30.38→31.80/31.10
(+2.5/+2.4%); MTP128 d0/d32K 61.28/94.61→63.89/99.19 (+4.3/+4.9%). Repeatable
but each below the 5% gate. E7 alone (interleaved no-E7 variant): TG +1.1–1.5%;
side-stream overlap does pay on decode here. AR/MTP hashes, MTP work counts and
the 32,760-position prefill logits are byte-identical to B1 (E2's explicit FMA
changed no bits). Rejected: E3 HC-up+mix fusion (≈13.4 GFLOP/call makes it a
WMMA epilogue fusion, the rejected S:34 design); E5 device-chained drafts (host
decides confidence stop/lookup/EOS each step; chaining wastes draft work).

Decode profile (rocprofv3 7.2.4 over the Nix runtime, AR d0, 31.46 tok/s under
the tracer): 31.78 ms and 1442 dispatches/token, kernel-active 67.0% (tracer
may inflate gaps). ms/token: EXL3 dense+shared GEMV 10.48 (≈200 GB/s),
routed 7.27 (≈165 GB/s), HC Q8 3.42 (≈190 GB/s), lm head 2.74 (≈230 GB/s),
HC combine/mix 1.22, GDN 1.16, small GEMM 1.14, routed glue 1.12, attention
0.74, router top-k 0.58. Next levers: inter-kernel idle (launch count/
persistent decode) and routed GEMV bandwidth.
Evidence: `build/b1/gpu-b2/RESULT.json`, `perf.json`, `prof/decode-summary.json`.

## Flash-Next B1 decode glue and HC Q8, November 7, 2026

User rule change: output bits may change; slight error is acceptable for speed.
Commits `5c328695..cc86343e` (brief `plan/exl3/B1-BRIEF.md`): HC-down decode
deep steps (A3); GDN decode history/rollback writes folded into SsmConv4 (A1);
routed decode memset/BuildTiles/ScaleDown/RestoreDown folded into router,
restore and combine (B1); four routed GEMV waves per block (B2); prefill mix
epilogue emits F16 mixed (B4); IEEE-rounded Q8_0 copy scales (C1); EXL3 decode
HC-down reads the resident Q8_0 copy (C2); EXL3 HC-up stored Q8_0, F16 freed
(C3); prefill HC gate stored F16 (C4).

Same window, c03 pack, pinned Nix10.1, FP16 KV, B C B C, r2, baseline
`f24b10ea` build vs candidate `cc86343e`:

| Row | Baseline tok/s | Candidate tok/s | Change |
|---|---:|---:|---:|
| PP2048 / 8192 off | 1471.5 / 1499.7 | 1508.7 / 1535.0 | +2.5% / +2.4% |
| PP2048 / 8192 MTP | 1395.9 / 1419.8 | 1440.5 / 1461.7 | +3.2% / +3.0% |
| AR128 d0 / d32768 | 27.31 / 26.75 | 31.02 / 30.37 | +13.6% / +13.5% |
| MTP128 d0 / d32768 | 51.36 / 87.62 | 60.72 / 94.51 | +18.2% / +7.9% |

PP gains are consistent but below the 5% gate. Candidate AR/MTP greedy hashes
agree at both depths; d0 differs from baseline (allowed), d32768 is identical
and lookup-heavy. Quality on the 32,760 BF16-teacher positions vs saved c03 FP16
reference: KL256 0.041522→0.041742, top-1 90.8394→90.8303%, NLL
2.1626503→2.162053; every paired token/block/window interval spans zero, so no
C2/C3/C4 isolation was needed. Contributions of individual items are not
isolated.

Rejected: fusing q/k normalization into GdnKernel (`kPrep`) broke chunk and
checkpoint invariance (state ≤1.5e-8, output ≤6e-7 across {3}→{1,2}); reverted
in `cc86343e`, history fusion kept. A4 GPU greedy argmax saves only ≈50 µs/token
because the host still syncs for the token. C5 full HC-up+mix fusion and A2
producer-side rotation, B3 EXL3 deferred residual remain open. Pre-existing:
gdn_ops "fused stable GDN differs from the separate convolution" (n=1, few-ulp
outputs, identical state) also fails at `f24b10ea`.
Evidence: `build/b1/gpu/RESULT.json`, `perf.json`, `quality/`, `units2/`.

## c03 final-pack runtime compatibility, November 7, 2026

Final assembly passed payload checks, but nativevision image e7a761a40187 /
ELF62f0c59a… is older than K8 support. Actual smoke002 failed before inference
on the valid K8 head. Disassembly of the actual ELF (not current source) shows
BindExl3LinearBank K3..7, Exl3FastSupported/GEMV K1..7 and zero K8 GEMV symbols.
The final pack has42 K8 projections, so changing only a head loader bound is
unsafe. Source commit2095529d adds coherent K8 loader/decode/GEMV/prefill/MoE
support. The newer refactor-gpu binary was built with ROCm7.2.4, so it is not
a Nix10.1 replacement. Prepare a coherent current-source Nix10.1 build and
matched quality collector, preserving all completed packs and references.
This is runtime incompatibility, not a measured quantization quality failure.
Evidence: flash-next-quant-20261104/mtp-final-validation-preparation/
loader-diagnosis/DIAGNOSIS.json; final-validation/smoke-002/.

## c03 terminal head recovery, November 7, 2026

The completed K8 head failed only when row1's 7793×248320 FP16 logits
exceeded the 2GiB asynchronous write budget. Reused the existing packed head;
recomputed only five lost raw references and packed forwards, with original
CPU error reduction/dtype/scales. No H collection or quantization repeated.
Five finite full-row comparisons took 554.55s; mean RFN 0.00386195,
range 0.00273402–0.00499969, mean SQNR 48.5118dB. This is a local head
comparison on calibration inputs, not whole-model quality or throughput.
Private synchronous writes retained 9.18GiB logits. Independent review,
cleanup zero and unchanged protected metadata passed; CPU commit hashed all
five saved outputs then atomically promoted idx53, preserving idx52/idx51 and
failed module062 evidence. Final assembly and bundled inference validation
are separate stages. Evidence: `flash-next-quant-20261104/head-recovery/`
`RESULT-REVIEW.json`, `COMMIT-JOURNAL.json`, and recovery001 result `SUMMARY.json`.

## Offline EXL3 K8 ngram conversion, November 4, 2026

Kyojin v1.8.6 (`fbde8a36`) with the retained ROCm 10.1 Torch/native extension
defaults K8/L160/mul1 trellis conversion to its base kernel on gfx1151. Its
already-compiled optimized kernel cut the same 131,072 mixed-hash-head rows
from 17.837 s (base, 8192-row chunks) to 2.975 s (opt, 32768-row chunks),
with all packed bytes equal. Increasing base chunks to 131,072 only gave
17.690 s; extra memory alone did not fix the kernel bottleneck. Two workers,
each with a private stream and TLS scratch, took 2.520 s; four took 2.560 s.
Zero, constant, tiny, ramp and alternating test rows also matched base exactly.
These are conversion measurements, not inference speed or final model quality.
The 1.71-hour full-table extrapolation excludes I/O and production checks.
Adopt opt/32768/two streams only for this qualified K8/L160 conversion;
do not globally force the kernel family for other bitrates or tile lengths.
Evidence: `ai.tools/tmp/test-results/flash-next-quant-20261104/`
`ngram-speed/RESULTS.json`, `ngram-streams/RESULTS.json`, and retained
`ngram-streams/opt-reference.safetensors`; execution plan
[`F3-QUANT-BIT-ALLOCATION.md`](../plan/exl3/F3-QUANT-BIT-ALLOCATION.md).

## Vemloir Flash-Next EXL3 pack, November 4, 2026

`Vemloir/Qwen3.8-Flash-Next-sc-exl3` 4.05bpw_sc_h8_ng8 (`fe8ad64b`) needed
three loader/kernel additions: n-gram tables split over
`ngram_embedding-NNNNN-of-MMMMM.safetensors`, integer K8 (head, shared
experts, attention k/v, indexer) in the shared decode GEMV, X3 WMMA and
buckets, and K3.5 routed gate/up experts (decode `RoutedGemv` and bucket
WMMA; rate code 16 + K). K8 lane decode takes state 7's low byte from the
third dword (offset always 24); K + 0.5 lane windows span 7K + 20 bits. Lane
decode K1-K8 and K1.5-K3.5 matches the CPU dequant bit for bit; K8/K3.5
projections, MLPs, buckets and routed experts pass their FP64 bars. Real pack:
the n-gram row oracle passes over both parts; greedy text, image (embedded K8
vision) and MTP prompts are correct; `conversation`/`batch`/`long-context`/
`state-edges` pass in MTP mode (1,417 of 1,702 drafts accepted).
RelWithDebInfo `gufo bench`: pp512 1081.9, pp2048 1395.4, tg128 26.21 tok/s
(AR); MTP tg128 38.55. Kyojin on the same build: 1093.8/1393.2/30.50. The
lower Vemloir TG follows its bit allocation (dense GDN/attention K6-K7 and K8
shared experts versus Kyojin's K4 dense): its K6-K8 GEMVs take ~918 ms versus
~578 ms of K4 GEMVs per 64-token profile at similar bandwidth, not a slower
kernel. Kyojin and 27B EXL3 pp/tg stay within 1% of `c74430b1` (one
cold-cache Kyojin pp512 outlier did not reproduce over six reps).

## EXL3 refactor and native 27B combined pass, November 4, 2026

Base `e5002432` versus `76e46955`, identical RelWithDebInfo docker builds,
`tests/functional/run.py` with recorded baselines. All suites pass functionally
with no quality/coverage change: 27B EXL3 5.00bpw (base through a GGUF
sidecar, candidate from the pack directory, same `--model` string and BF16
`--mmproj` on both) `conversation`/`batch`/`long-context`; 27B UD-Q4_K_M and
Flash-Next Kyojin native `conversation`/`batch`. Summed per run, decode and
server time stay within 1% (27B EXL3 prefill -4% from the WMMA threshold).
`gufo bench` 27B EXL3 pp512/pp2048/tg128 is unchanged within 1% across two
alternations. Every retained per-request flag is inconclusive and moves both
ways: `batch_c2_greedy_short_*` cache snapshots take a whole number of
~80 ms steps (175/257, 489/570/813 ms), so the step a request pays shifts
between runs; `short_pairs` TTFT/inter-token flags follow which of the two
concurrent requests is admitted first. A bisect over every intermediate commit
shows the same pattern, not a code regression. Starting a native-pack 27B
server without `--mmproj` loads its embedded EXL3 vision (927,587,264 B), so
compare text timing only with matching vision residency.

## Qwen3.8 27B EXL3 prefill WMMA threshold, November 4, 2026

Per-projection prefill time, X2b GEMV (8-row passes) versus X3/X3b trellis
WMMA (rotation + GEMM), actual `turboderp/Qwen3.8-27B-exl3` 5.00bpw weights,
1-64 rows, two runs (50 and 100 reps; `tools/bench/exl3_wmma_threshold_bench.hip`,
RelWithDebInfo; each matrix timed repeatedly, so small ones may stay warm in
the MALL). One WMMA launch costs about 1.7-1.9x a GEMV pass for the
5120x17408 FFN matrices (9-16 rows: gate/up WMMA 449-469 vs 2-pass GEMV
522-542 us, down 490-507 vs 528-538 us), but 2.1-4.2x for q (5120x12288),
qkv, z, ssm_out, o and k/v. Retained:
projections of at least 2^26 elements switch at 9 rows, the others keep 17.
Model projection time at 9-16 rows falls about 8%; the per-projection optimum
adds only 0.1-0.7% at 17-32 rows (k/v prefer the GEMV through 32). Matched
`gufo bench` (alternating base/candidate, 5 reps each, same build flags):
pp9 57.6/57.9 to 63.7/64.0 tok/s (+10%), pp12 76.1/76.2 to 82.4/83.3 (+9%),
pp16 98.4/98.6 to 107.0/108.1 (+9%); pp8, pp17 and pp32 unchanged within 1%.
The large WMMA path is the kernel already qualified for 17+ rows; a raw
15-token greedy prompt gives identical output. Only 5.00bpw was measured;
other bit rates may move the crossover.

## Native EXL3 vision, November 4, 2026

Embedded vision now loads directly from native packs. Strict geometry and
selected-tensor identity, split QKV precedence, FP16 operands/FP32 residuals
and padded FFN channels preserve the pack's numerical contract. K6 uses the
shared packed GEMM; K8 uses one temporary decoded-inner slab, never a permanent
dense encoder copy. Image admission includes the additional scratch.

K6/K8 independent same-pack vision comparison: all 29 stages pass 1% relative RMS
on 320x224; final 0.6023%/0.6212%. Existing GGUF all 29 stages byte-exact against
retained baseline. Full Gufo C2/native 256K image+repeat succeeds without any
GGUF mappings,91/91 cached tokens and 0 prefill on repeat. K6 vision residence
352601088B vs historical BF16 903984064 B saves 525.84 MiB; no matched speed claim.
[Scope, provenance, failures and limits](../plan/exl3/F3-VISION-NATIVE.md).

## Request memory budgeting and KV admission candidate, November 3, 2026

Fixed physical KV capacity is independent of concurrency. For Flash-Next's
host-only checkpoints, keep one active request entitled to a full context;
peers reserve their prompt plus4096 output tokens and pause before growing
past their window. Promote a resident after the guarantor finishes. Reclaim
idle rows with reuse=false when growing an already acquired lease: protecting
a duplicate idle prefix here can deadlock promotion. Do not apply this policy
to device-reference checkpoints without qualifying their shared-row accounting.

Image admission uses actual prepared dimensions, summed persistent embeddings
and the maximum serialized encoder scratch. Keep request promises through
completion and cache-segment promises through physical commit. Both consult
current free memory with4GiB headroom. A background callback must not read
mutable executor scratch pointers; capture a conservative bound instead.
Already materialized images are conservatively double-counted until request
completion. BLAS workspace and other lazy allocations are not fully measured.
Memory logs separate budgets, device totals/subsets, graph counts, host vector
capacity, cache ledger categories, RSS and host/device free; graph bytes remain
unmeasured. They run at admission/release and at most once per10seconds.

Candidate CPU checks: request_memory_budget_test, text_model_runner_test,
component_text_cache_test and text_generation_scheduler_test pass (50.23s).
Actual host block allocator exercises C4 on one/two-context pools,20K outputs,
AR/fake3-token-MTP, preview, promotion and cancellation; matching idle-prefix
reclaim and failed growth reservation preservation pass. This is not actual
7-draft GPU/RNG, multimodal or performance qualification. Existing baselines
and Nix10.1 intermediates retained. Candidate evidence: build/memory-efficiency/.
GPU qualification on November4 passed at16K/C4/two-context pool: four
unlimited-output streams admitted together and cancelled;20 completed requests
cover C1 greedy/seeded MTP exact replay, C4 response smoke,9637-token cache and
continuation, vision left/right and medium. Three live streams reached real
reservation pressure: request25 waited, then was promoted/resumed after the
guarantor was cancelled. MTP draft_limit=7 and accepted/rejected drafts observed.
No new GPU errors; compute owners released. Exit quiet flag is false because
desktop GPU busy=11%, so timings remain advisory, not matched qualification.
Long cold prefill8082ms; exact cached replay0ms prefill/140ms total;
continuation cached285ms vs cold8302ms, same output. These are different cache
histories, not candidate-vs-baseline speedups. Full256K inference not repeated.
Evidence: build/memory-efficiency/gpu/http-a03/ and its retained server logs.

Incremental link maps must normalize compile output paths: the first candidate
compiled the new scheduler but linked a retained old relative-path object.
That attempt is INVALID_ARTIFACT (not engine failure); LINK-a01.json and the
old image are retained. Corrected LINK.json asserts all five changed objects
occur exactly once. The earlier HTTP max_tokens=-1 attempt was invalid input;
omit the HTTP field to inherit the server's unlimited default. Both failed
attempts remain recorded. Default launch image is the corrected candidate;
C2/native256K/K6V6/vision/medium/MTP and the fixed6.123GiB pool are retained.


## Flash-Next K6V6 KV, November 3, 2026

Implemented explicit K6V6 token-major KVarN with existing low-nibble words
plus a high-two-bit plane per row. Keep K4 specializations, FP16 sink/tail,
and FP16 MTP KV. Propagate precision through pool geometry, component-cache
identity and native snapshot magic; changing only CLI/quantizer corrupts restores.
C2 native262144, YaRN off, two-context pool:2050*3207168=6574694400B (6.123GiB).

Independent K4/K6 operator checks pass; K6 read path14 launches maxabs0.000151
within existing tolerance, not bit-exact. Shared AR/MTP fragmented mappings,
boundary rewinds, saturation, native materialized replay and wrong-codec
snapshot rejection pass. CPU parser/layout and764-file formatting pass;
independent code review approved. Saved BF16 reference reused:32760 scored
positions, NLL2.170561, KL2560.024987, top1agreement0.920910. This is not a
matched K4-to-K6 quality gain or a full256K quality test.

Single candidate timings under desktop graphics (prebusy11%, no foreign KFD):
PP2048/8192=1374.42/1403.11, AR128 d0/d32K=24.33/23.58,
MTP128=35.66/76.06tok/s. Saved quiet K4 rates1425.09/1462.00,25.20/24.47,
40.92/79.27 are historical; current repeats1 vs saved2 and desktop activity
prevent causal attribution. Short-context MTP observation is12.9% lower;
do not claim performance neutrality. No baseline rerun or discarded failure.
Nix ROCm10.1 compiler/flags and previous objects retained in build/k6v6.
Evidence: `/home/m2on/.local/state/gufo/k6v6-256k-20261103/qualification/RESULT.json`, `quality-score.json`;
source diff/build/review logs in `build/k6v6/`.

## How to use

**Read before experimenting. Measured means do not repeat unchanged work.**
Re-measure only when the relevant code, compiler, allocation or workload changed;
name that change and reuse unaffected evidence. This ledger records transferable
lessons, not a claim that every cited branch is integrated or fully qualified.
Model-only details are retained in the final **Other (model-specific)** section.

**This is a living document. Update it as you find things.** In the same commit
as any measured result, whether kept or rejected, add or update its entry here
with a citation (commit, bench file, receipt path or file:line).
- Put model-agnostic lessons in the main sections and model-only results under
  **Other**.
- Replace or correct a stale entry rather than appending a duplicate, and note
  the change that invalidated it.
- When an open lead is resolved, move it into a patterns row or the rejected
  table with its numbers.

Scope: AMD Strix Halo, RDNA3.5 `gfx1151`, 128 GB LPDDR5X unified memory
(platform identity: BFN:3). The source audit used `gufo@d286c866` and
`gufo-f3-integ@966f84b8`. Source rules are not new timing measurements.
No GPU experiments were run for this document.

Evidence notation below is `key:line` or `key:start–end`; each key expands to
an exact file in this table. Commit IDs identify implementations, not proof of
an unreported speedup. **S is session-report evidence** (agent reports from the 2026-10-31/11-01 F3
session, with later receipt-backed annotations); it is weaker provenance than raw receipts.

| Key | Source, relative to repository root unless stated otherwise |
|---|---|
| S | `plan/exl3/F3-SESSION-FINDINGS.md` |
| R0 | `plan/rotated-quant/R0-RESULTS.md` |
| R0b | `plan/rotated-quant/R0B-RESULTS.md` |
| R1b | `plan/rotated-quant/R1B-RESULTS.md` |
| R2 | `plan/rotated-quant/R2-RESULTS.md` |
| W8 | `plan/rotated-quant/EXL3-RESULTS.md` |
| W9 | `plan/rotated-quant/W9-RESULTS.md` |
| X3 | `plan/exl3/X3-RESULTS.md` |
| XB | `plan/exl3/XB-RESULTS.md` |
| XA | `plan/exl3/XA-RESULTS.md` |
| XQ | `plan/exl3/XQ-RESULTS.md` |
| F1 | `plan/exl3/F1-RESULTS.md` |
| F2I | `plan/exl3/F2-IMPLEMENTATION.md` |
| F2D | `plan/exl3/F2-DESIGN.md` |
| F2Q | `plan/exl3/F2-QUALITY.md` |
| F3 | `plan/exl3/F3-PLAN.md` |
| K | `plan/gufo-upstream-port/KVARN.md` |
| Skinny | `plan/gufo-upstream-port/SKINNY.md` |
| E27 | `docs/models/qwen3.8-27b/EXPERIMENTS.md` |
| EFN | `docs/models/qwen3.8-flash-next/EXPERIMENTS.md` |
| Q27 | `docs/models/qwen3.8-27b/QUALITY.md` |
| QFN | `docs/models/qwen3.8-flash-next/QUALITY.md` |
| B27 | `docs/models/qwen3.8-27b/BENCHMARKS.md` |
| BFN | `docs/models/qwen3.8-flash-next/BENCHMARKS.md` |
| Format | `docs/EXL3-FORMAT.md` |
| Testing | `docs/TESTING.md` |

Raw GPU-window evidence usually lives under
`~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/`;
older retained port evidence is under
`~/projects/ai.tools/storage/evaluations/gufo-upstream-port-final-20261023/`.
Keep the named window's `receipt/run`, binary hashes, plan, health result and
rejected attempts. R0 quality records live under
`~/projects/ai.tools/storage/evaluations/rq-r0-20261029/` (R0:255; XQ:382–390).

`F3bench/` below means `/home/m2on/projects/gufo-f3-integ/tools/bench/`;
`GDNtests/` means
`/home/m2on/projects/gufo-gdn/tests/models/qwen38_flash_next/`.
`F3src/` means `/home/m2on/projects/gufo-f3-integ/src/`.
Additional source prefixes: `QH/` = `src/models/qwen/hip/` in port-u4f;
`FN/` = `F3src/models/qwen38_flash_next/kernels/rocm/`;
`FM/` = `F3src/models/qwen38_flash_next/`;
`FT/` = `/home/m2on/projects/gufo-f3-integ/tests/models/qwen38_flash_next/`.
Use the cited snapshot and line numbers; identically named files in another
worktree may implement a different path.
These worktrees contain newer benches absent from port-u4f. Recover them from
the cited commit if the worktree disappears. Do not substitute an older bench
with the same name without comparing its source.

## Starting a new model's kernels on gfx1151

- Read the hardware, numerical and rejection sections first. Reuse the measured
  issue/overlap/cache lessons; do not start by repeating the same tile, wave
  specialization or instruction-spelling searches.
- Inventory real K/N/token shapes, ragged tails, routing/bucket histograms and
  allocation placement. Profile one representative phase and one control;
  separate weight bytes, activation traffic, compute and launch/wait costs
  (R0b:20–28,199–208; S:29–31; Testing:151–158).
- Fix arithmetic contracts before tuning: stored dtypes, rounding points,
  product/FMA association, reduction tree, tie order, phase selection and
  invariance under row shifts, chunking, batching and restore
  (Testing:114–149; XB:197–227).
- Establish an independent operator oracle and a pinned model teacher. Cover
  signed zero, subnormals, cancellations, awkward allocation-boundary tails,
  full logits/distributions and state transitions; agreement between two
  local paths is insufficient (Testing:88–98,114–145; F3:205–223).
- Keep baseline/candidate production binaries and emitted flags/ISA. Use
  `tools/bench/build.sh <name>`, focused builds/unit checks during iteration,
  then one combined correctness/integration/performance pass. Retain every
  failed or inconclusive result (`AGENTS.md:21–35,65–79`).
- Promote only after the component gain survives complete-request work and
  the unchanged quality gates. Make the winner default; do not preserve
  rejected experiments as production switches (Testing:3–5,88–102; F3:119).

## Hardware facts, ceilings and toolchain quirks

These are sustained microbenchmarks, not model-throughput forecasts. UMA does
not make allocation types equivalent: the peak probe uses device allocations;
production-style registered read-only weights were measured separately
(R0b:20–28,52).

| Resource / condition | Measured result | Evidence |
|---|---|---|
| DRAM read / write / copy | 240.07 / 218.37 / 209.23 GB/s | R0b:43–52; `tools/bench/gfx1151_peak.hip` |
| WMMA int8 / FP16 / BF16 | 57.02 TOPS / 55.34 / 55.29 TFLOPS | R0b:43–50 |
| VALU FP32 FMA | 24.82 TFLOPS | R0b:50 |
| MALL relevant to these probes | 32 MiB; repeatedly reused small buffers can measure cache rather than DRAM | `tools/bench/exl3_gemv_bench.hip:24–26`; R0b:195 |
| HIP-visible capacity on the measured host | 112.00 GiB; earlier planning's 124 GiB came from another configuration | F1:60 |
| Workgroup LDS limit | 65,536 B; a request for 131,072 B failed | W8:179 |
| Same-SIMD WMMA/VALU overlap | WMMA alone 3.07–3.20 ms, VALU 3.25 ms, combined completion 6.44 ms; interleaved wave 6.35 ms | `F3bench/gfx1151_wmma_valu_overlap.hip:12–15`; `5e20dc48` |
| Separate GEMM and bandwidth-pass streams | Only 12–20% overlap in the measured mix | S:8; `10c55dde`; `GDNtests/overlap_bench.cpp` |

There is no demonstrated free same-SIMD compute engine for a scalar decode
producer beside a WMMA consumer. The cross-stream result is narrower: overlap
can depend on compiler, resources and neighboring work, so measure complete
span rather than adding isolated kernel times (S:5,8; EFN:52–71,107–108).
Source audit clarification (no new measurement): that generic probe and the
existing Q8-weight production overlaps do not cover current EXL3 branch pairs.
[EXL3 stream candidate audit](../plan/exl3/F3-STREAM-OVERLAP.md) records their
joins, workspace aliases and priority. Shared/routed overlap is now qualified;
GDN Z had weak/negative evidence. See the final stream qualification below.

LDS-fed WMMA calibration:

| Fragment reads / WMMA | Sustained TFLOPS |
|---:|---:|
| 2.0 | 28–30 |
| 1.5 | 36–38 |
| 1.25 | 39–44 |
| 1.0 | 46–48 |
| 0.75 | 46–49 |
| 0.5 | 47–50 |
| Registers only | 48–54 |

Source for every entry: `F3bench/gfx1151_wmma_lds_bench.hip:11–13`,
`a6847817`. The inferred ceiling is approximately 128 B/clock/WGP of LDS
reads. The near-plateau around 0.75 reads/WMMA means fewer reads alone need
not improve a full core. The measured production tile had zero LDS bank
conflicts (S:6); do not assume padding is its missing optimization.

### Numerical hardware and compiler bugs

- WMMA FP16 cancellation is not exact: opposing unit products left a residue
  of −2^-24. Bound absolute error using the sum of absolute products, not a
  cancelled output; the recorded allowance is at the 2^-23 scale
  (S:7; `6fde4b45`). A zero partner can also affect the live product in a
  WMMA pair: causal tail padding needs a qualified pairing, not merely a
  zero mask (`tools/bench/kvarn_decode_bench.hip:20–24`; K:229).
- FP16 subnormals are not simply flushed. Unit-weight probes passed, but
  non-power-of-two products showed relative errors up to 6.7e-5 in selected
  cases and 1.3e-5 randomly. VOP2 dot2acc, VOP3 dot2 and WMMA share this;
  changing instruction spelling is not a repair (X3:82–109).
- The recorded hipcc half-pair bit-cast bug stores the low half when extracting
  `half2.y` as `uint16_t`. Extract the high half from a dword; ISA confirmed
  the correction (F3:181).
- Packed integer instruction availability matters: the trellis hash already
  uses packed low multiply plus mad; no packed halfword mul-high replacement
  was found. SAD byte sums and paired FP16 FMA are already used
  (S:13; `src/core/hip/detail/exl3_gemv.hpp:94–115`).
- Full-debug iterative-ILP compilation hit an LLVM register-pressure crash;
  line tables retain source profiling without that crash
  (`src/models/qwen/CMakeLists.txt:105–112`). Another recorded bucket
  scheduling mode crashed compilation (S:11).
- The recorded rocprofv3 build exposes no SQ stall counters. Static ISA counts
  or idle gaps do not establish a specific hardware stall (S:14).

### Compilation is part of the experiment

| Transferable rule | One repository example / evidence |
|---|---|
| Record emitted flags, not just preset names | The release preset is RelWithDebInfo, conventionally `-O2`; selected HIP targets override to `-O3 -ffast-math -fno-finite-math-only`. These are different numerical environments. `CMakePresets.json:59–68`; `FM/CMakeLists.txt:53–70` |
| Finite-only and reassociation are separate | The fast-math target explicitly disables finite-only assumptions; that does not restore stored rounding boundaries. `FM/CMakeLists.txt:68–70`; S:17–20 |
| Match helper and caller wave size | Separate wave64 translation units keep their helpers consistent while other kernels remain wave32. `src/models/qwen/CMakeLists.txt:97–112` |
| Scope scheduling/vectorization fixes narrowly | ILP helped selected widths; SLP scale/offset tuples spilled affected prefill tiles. `src/models/qwen/CMakeLists.txt:105–121` |
| Keep the CPU oracle strict | A test applies fast-math only on the device side. `FT/CMakeLists.txt:58–60` |
| Launch bounds are compiler/shape evidence | An unconstrained attention variant used 241 VGPRs; targeted minimum residency avoided its allocation cliff. `QH/kernels/attention_wmma.hip:144–147` |
| Arrays can become private scratch | Named vectors replaced an array that generated 112 B scratch, reloaded every window. `FN/kernels.hip.cpp:1946–1950` |

## Numerical contracts and bit-exactness

Preserve the compiled rounded operations, not just an algebraic formula.
Moving code between TUs, inlining it, changing tile position or recording an
intermediate can alter contraction (S:17–22; XB:197–227).

| Rule | Measured case or code evidence |
|---|---|
| Pin each intended product/store before the next expression | Input-times-scale had fused with the first butterfly; a register dependency restored the contract. XB:201–213; `c0da594e` |
| Match the input oracle's precision | FP64 rotation rounded to FP16 differed from ordered FP32 rotation rounded to FP16. Test accumulation with the exact same rounded input. XB:197–227 |
| Make FMA association explicit | Scale/gamma order, square-plus-partner reductions and RoPE changed under reassociation. S:18–20; `QH/kernels/ssm_row_split.hip:332–358` |
| Pins have scheduling costs | Hoist independent loads before pinned products; a recompute epilogue fell 139→93 ms/pass after this repair. S:22 |
| A division intrinsic/pragma may not defeat fast-math | Device scale generation stayed one ulp low; the recorded quantizer moved to host. F1:493–495 |
| Standalone operator equality is not model equality | Inlining folded a norm into neighboring products only inside the model; full record-byte comparison caught it. S:19 |
| Preserve geometry used by reductions | Dropping members from a grouped launch must retain its original shape basis when that basis chooses split-K. XB:116–119; `FN/executor.cpp:2185–2194` |
| Store precision is a separate contract from accumulation | Bit-exact decoded weights do not imply bit-exact effective weights after additional rotations/roundings. Format:139–151 |
| Phase selection may be semantic | A prefill tail must not switch to a different reduction solely because it became short. `FN/executor.cpp:2101–2105,3887–3894` |
| Keep ranking and ties exact | Narrowing selector queries can exchange near-tied winners; retain the original precision, reduction and tie order. `FN/kernels.hip.cpp:3042–3045,3892–3894` |

Specify independent tolerances before measuring. An accumulation gate fed the
same low-precision inputs and an end-to-end gate spanning quantized boundaries
answer different questions. Freeze the first changed intermediate to isolate
bin crossings; do not widen a gate because an optimization fails
(Testing:121–145; F2I:155–173). Model-specific bounds remain in Other.

Validate shifted rows, ragged K/N tails, signed zero, subnormals, cancellation,
finite outputs and allocation-boundary guards. Test every affected phase and
state transition: chunk partitions, scalar/batched execution, graph replay,
snapshot restore, cancellation and rollback prefixes (F3:205–223;
Testing:88–98,147–149). A legacy helper test does not cover a production
executor that now bypasses it (`FT/exl3_invariance_test.cpp:65–101`;
`FN/executor.cpp:2255–2268`).

Use fixed token histories and a pinned independent teacher. Truncated KL is a
lower bound and can hide a copy failure whose correct token is outside the
retained set; report coverage, NLL/loss tails, categories and correlated-sample
intervals. Identical free-running tokens prove consistency, not teacher
accuracy (Testing:114–149; W8:128–135; F2Q:10–16,157–163).

Do not assume runtime determinism from operator determinism: repeated upstream
collection produced different record hashes despite the same pack, image and
collector. Keep process/reload controls and paired comparisons
(XQ:414–418; correction to `plan/rotated-quant/PLAN.md:111`).

## Reusable kernel patterns

Here N means output features, K input features, R independent input rows.
Shape values below illustrate a reusable mechanism; they are not default
thresholds for a new model. Read the cited rejection before reviving a variant.

### Matrix cores, decoding and ragged outputs

| Pattern | One measured example / limit | Evidence |
|---|---|---|
| Decode weights into a proven WMMA frame | Reuse its LDS layout and dead scratch for an epilogue; existing non-trellis instances retained identical ISA/resources. | X3:39–49,72–80; `9d5c937b`, `abe9ee24` |
| Reconstructing weights first is not automatically faster | Fused-decode and reconstruct-to-FP16 plus the same core had equivalent speed; the core, not decode alone, limited this case. | S:25; `F3bench/exl3_prefill_shapes_bench.hip` |
| Distinguish core throughput from feed bandwidth | The 256×256 core delivered 31–36 TFLOPS with the calibrated near-plateau LDS reuse. See hardware calibration instead of assuming more reuse must win. | S:25; `F3bench/exl3_x3_tiles_bench.hip` |
| Preserve reduction order across tile classes | Token/feature tiling may change with shape, but live rows must retain the qualified arithmetic. Existing dispatch values are recorded in Other. | `F3src/models/qwen/hip/kernels/prefill_fp16.hip:1161–1201` |
| Specialize header lifetime and decode placement | Reuse packed headers across stages, but delaying one mixed-format decoder changed rounding; different decoders retained different output orientations. | `F3src/models/qwen/hip/kernels/prefill_fp16.hip:250–274,696–698` |
| Carry a qualified schedule into the actual grouped instantiation | Grouped EXL3 dense 256×256 now shares the existing direct-path hint; exact K7 checks and aggregate record equality pass. No isolated end-to-end gain attributed to this one change. | `fn-prefill-batch-nix101`; final Nix prefill section below |
| Bound compiler scheduling | A per-K fence/no-unroll/no-SLP repair recovered affected prefill throughput; exact build-specific numbers stay in Other. | E27:130 |
| Test full-library kernels under row shifts | A pinned best library solution changed bits when a row moved; exact dimensions and a custom invariant reduction were required. | S:28; `FN/blaslt.cpp:137–173` |

### MoE and irregular work

| Pattern | One cited example | Evidence |
|---|---|---|
| Measure routing skew before choosing bucket classes | Hot experts held most assignments while many experts were unused; uniform averages were misleading. Concrete histograms stay in Other. | S:31 |
| Use hot/cold bucket classes rather than one universal tile | Hot-expert 128-token buckets improved 1–8%; keep compact cold classes. | S:30; `56202401` |
| Reduce redundant decoding without inflating live fragments | An eight-wave strip arrangement improved the affected class 7–12%, same K order. | S:30; `df1f7409` |
| Keep descriptors/counts on device | A bounded grid ignores descriptors beyond the device count and rows past each bucket; no host count download or physical padding required. | F3:187–203; `d2dbcb3c` |
| Skip entirely dead subtiles | Padding-only token subtiles can be omitted without changing live outputs. | S:30; `09f6b8a4` |
| Share preparation at the actual reuse level | Per-token input rotation replaced per-assignment repetition: 1.79→1.21 ms. | S:35; `5a7e2280` |
| Fuse restore/scatter/reduction at a consumer boundary | Restore-plus-combine fell 3.10→1.42 ms/layer; folding the MoE sum into the next residual/combine saved another reported 19 ms/pass. | S:35; `713cd166`, `a4ec352c` |
| Reuse original routing IDs instead of recovering them from offsets | Removes the per-segment/slot binary search while preserving assignment order and descriptor lifetime; independent frozen-row oracle and collected records agree exactly. | `fn-prefill-operator-screen-nix101` oracle; `fn-prefill-batch-nix101` qualification |
| Preserve assignment sum order in fusion | Inverse slot map, ordered weighted additions and pinned final values retain the unfused store contract. | `FN/exl3_prefill_combine.hpp:48–101` |

Ablations remove correctness and expose dependencies; their deltas are not
additive. A recorded bucket baseline was 3.6 ms, WMMA+LDS-only 2.21–2.29 ms,
and removing barriers was neutral. The estimated serialized floor was not an
achieved implementation (S:29). Inspect the whole critical path before deciding
that one removed instruction family is the bottleneck.

### Loads, memory placement and scratch

| Pattern | One cited example | Evidence |
|---|---|---|
| Distinguish dirty consumer input from cold weights | A 10240→320 GEMM at R=2048 took 0.77 ms warm, 1.04 after a 256 MB read flush, 1.93 after a write flush. | S:9 |
| Test producer-side store policy | Non-temporal producer stores reduced a bandwidth producer 1.47→1.00 ms/call and its consumer to 1.78 ms. Model-level retention was not established. | S:9 |
| Match production allocation | Device-copy weights lost 3–8% cold throughput versus the mapped control; huge-page-backed immutable weights gave about 2% AR benefit in another retained control. | E27:100,107 |
| Tune per-wave memory-level parallelism without changing sum order | Fetch-ahead improved effective trellis throughput 216→271 GB/s; it is not a new physical DRAM ceiling. | S:45; `04269373` |
| Coalesce stored format bytes before decoding | Whole aligned stored blocks improved a routed down total 160.5→153.0 ms with identical hashes. | EFN:262–263 |
| Cache small reused matrices; stream large one-use weights | A small injection matrix deliberately keeps a different cache hint from feature/decode matrices. | `QH/kernels/prefill_gemm.hip:167–178` |
| Reuse scratch only after its last consumer | Sequential stages alias their largest existing slabs; row-offset views leave shared transient staging at its base. | `FN/executor.cpp:1228–1231,2057–2068`; `FN/batch.cpp:110–116` |
| Invalidate cached activation identities on every rewrite | Alternating slots and pointer/row/shape readiness are part of graph semantics, not just host bookkeeping. | `FN/executor.cpp:1905–1914,2807–2811,4116–4133`; `FN/batch.cpp:119–128` |
| Preserve graph-captured addresses | Growing rollback storage retains old addresses; releasing storage invalidates only graphs that captured it. | `FN/kernels.hpp:141–144`; `FN/executor.cpp:931–940` |
| Place weights with dispatch headroom | The weight-placement helper leaves a 4 GiB reserve when choosing hipMalloc, otherwise managed storage. This is not a guarantee for arbitrary allocations. | `src/core/hip/heap_placement.hpp:8–24` |

### Attention, reductions and precision-safe fusion

- Cover more independent query work per K/V load when registers allow it;
  prefetch behind existing compute rather than at point of use. Recorded
  next-tile prefetch reduced one call 3.250→2.500 ms shallow and
  51.595→46.107 ms deep without changing staged bytes/order
  (`QH/kernels/attention_wmma.hip:10–52`).
- LDS transpose layout can outweigh global coalescing. Keep padding justified
  by actual bank conflicts; neither a lower resource count nor higher nominal
  occupancy proves a win (`QH/kernels/attention_wmma.hip:17–23,54–59`).
- LDS-only barriers may retain read-only global prefetch in flight. Use the
  convergent LDS wait/barrier only when no global publication is required;
  ordinary block sync serialized the recorded pipeline
  (`FN/kernels.hip.cpp:4462–4469`; `QH/kernels/small_batch_gemm.hpp:71–78`).
- Pack more logical reductions into a hardware wave only while preserving
  original lane partials and the original final addition. A narrower logical
  head explicitly keeps both old partials until their original combination
  (`QH/kernels/attention_decode_split_k.hpp:146–171`).
- Bound unrolling/live query loads to avoid spills without changing ranking.
  The recorded selector improved 109.3→76.3 µs while retaining every FMA and
  sum (`FN/kernels.hip.cpp:3082–3108`; EFN:172–185).
- Normalizing small FP16 operands with an exact power-of-two row scale can
  preserve the rounded input while avoiding unsafe subnormal arithmetic.
  Choose a safe range, undo in FP32, and retain widened-FMA fallback when
  unsafe; do not scale before the specified FP16 rounding
  (`FN/exl3_moe.hip:177–207,230–288,605–608`; `f6471551`).
- Fuse only across qualified storage boundaries. Recompute-versus-store can
  save traffic, but recording a different intermediate or contracting across
  a removed store changes results. Keep recurrent/history ownership explicit
  (`FN/kernels.hip.cpp:1799–1815,2279–2282,8108–8113`).

## Measurement and process know-how

- During implementation, build and run the smallest relevant unit checks;
  finish with one combined correctness/integration/performance pass. Broaden
  only for changed shared behavior or a discovered risk
  (`AGENTS.md:25–35`; Testing:3–5,88–102).
- Time production binaries, preserve compiler/dependency identity and keep both
  baseline/candidate artifacts (`AGENTS.md:21–23`). Match inputs, depth,
  quantization, sampling, history, cache state, concurrency and actual work;
  shortened setup histories are not matched controls (Testing:151–158; EFN:256).
- Judge individual requests/phases at **5% and 10 ms**, not an average across
  requests. Retain inconclusive results; do not silently apply current margins
  to historical reports using another absolute threshold
  (`AGENTS.md:65–79`; F3:118; Skinny:126–127).
- Rotate buffers beyond cache and match allocation. Effective GB/s divides
  declared bytes by elapsed time; it need not equal DRAM traffic. Small reused
  matrices reported 400–860 GB/s while the cold pass sustained 124 GB/s
  (`tools/bench/asr_decode_gemv_bench.hip:8–13`).
- Separate warmup, first-use and steady work. Choose the actually timed trace
  pass, inspect wall span/GPU busy/idle gaps, and profile separately from
  headline timings (`tools/prof/prof.py:4–26`).
- Docker and Nix results are not interchangeable. A same-model example was
  1567 versus 1762 tok/s; the build configuration change does not isolate a
  compiler effect (S:51; F3:233–238).
- A Nix executable can still load the base image's HIP/BLAS through inherited
  `LD_LIBRARY_PATH=/opt/rocm/lib`. The 2026-11-01 resume found that in the
  candidate image. Clearing the image default and retaining only the wrapper's
  patched ROCr prefix makes clr/hipblas/rocblas/hipblaslt resolve to Nix 10.1.
  The corrected dense-prefill unit passed (`fn-indexer-nix-unit-runtime101`).
  Audit actual baseline receipt environments before reusing timings; some old
  plans already override this variable. See F3-RESUME-RESULTS in gufo-f3-integ.
- Match profiler and binary runtimes. The recorded container profiler could
  not trace the other HSA runtime; building the missing profiling stack would
  pull an hours-long compiler build (S:52).
- Counter replay over a variant matrix can overrun the window. Use a
  single-dispatch mode (`5d994265`), shared lock and in-container hard timeout.
  Killing the Docker client can leave GPU work running (S:53).
- Respect the runner's remaining budget. Lock rejection before launch is not
  a failed kernel run; keep rejected attempts and health receipts
  (R0b:34–36; S:56).
- rocprof finalization can hang after SIGTERM. The recorded driver waited for
  its finalization marker before killing; retain complete trace data rather
  than confusing shutdown with missing measurements (Skinny:188–193).
- The recorded evaluation image needs a shell entrypoint override and GPU
  groups 44/990; its windows required 150 GiB free root space. These are
  environment facts, not authorization to change another host (S:54).
- Use internal NVMe for large active artifacts. A recorded load took about
  300 seconds from USB versus 30 internal; later on-demand reads still matter
  even after startup (S:49).
- Large write/delete cycles changed SSD latency independently of GPU code.
  TRIM-associated controls restored stable reads; the internal QLC mechanism
  was an inference. Record storage history and device latency
  (`plan/gufo-upstream-port/DISKREAD-RESULTS.md:9–23,48–55`).
- Separate anonymous allocations, precommitted backing and reclaimable file cache.
  A supposed budget-independent memory floor partly came from precommitted
  cache backing (`plan/gufo-upstream-port/ANON-FLOOR-RESULTS.md:27–43`).
- Run a memory-heavy CPU oracle alone where possible. Co-resident weights
  starved page cache; conversion/build work also invalidated timing windows
  (EFN:129–131; R2:527; K:338).
- Diagnostics may disable graphs or omit accounting. Confirm actual execution
  mode before interpreting diagnostic timings as serving performance
  (`FN/executor.cpp:3895–3899,4100–4105`).
- Warm shapes eagerly before capture; record only after required allocations.
  Thread-local capture permits frozen peer copies on independent nonblocking
  streams; readiness/slot state must match the executed pass
  (`FN/executor.cpp:4103–4139`).
- Persist BLAS solution IDs, not opaque process-dependent algorithm bytes.
  Reconstruct and validate hardware/runtime/library identity, exact dimensions
  and workspace. Raw descriptor replay once segfaulted at prewarm
  (`QH/kernels/hipblaslt_gemm.hip:169–220,297–315`;
  `src/core/hip/detail/hipblaslt_plan_database.hpp:18–35`; `FN/blaslt.cpp:165–173`).
- Use explicit bench names: build.sh with no arguments builds every bench.
  Its default `-O3` and separate wave64 objects do not automatically reproduce
  every target flag (`tools/bench/build.sh:8–10,56–76,90–108`).
- Analyze the correct trace pass with
  `tools/prof/prof.py show DB --passes GAP_MS --pass N`; static ISA counts
  are not dynamic trip counts or stall counters. The actual analyzer path is
  `tools/prof/isa_mix.py out.s [substring] [--all]`
  (`tools/prof/prof.py:15–26,44`; `tools/prof/isa_mix.py:2–15,23–36`).

Tool fidelity issues still worth fixing: the ISA docstring prints an obsolete
path; the bench script omits CMake's no-SLP option for one wave64 prefill
object; the newer shared matrix/decode names are absent from one stage map.
Verify classification before trusting stage rollups
(`tools/prof/isa_mix.py:10–12`; `tools/bench/build.sh:90–108`;
`src/models/qwen/CMakeLists.txt:113–121`; `tools/prof/prof.py:129–163,232–236`).

## Tried and rejected — general lessons

These are measured negative examples for transferable mechanisms, not proofs
that every implementation of an idea must fail. Repeat only with a materially
changed path and an explicit reason. Model-only searches are preserved in Other.

| Idea | Recorded result | Transferable lesson / evidence |
|---|---|---|
| Fewer LDS reads at any register cost | Alternate dense tile stayed within ±2% despite 242 VGPRs/occupancy 5 | The LDS plateau is not the whole kernel; S:25, `b831b1e5` |
| Producer/consumer WMMA wave specialization | No same-SIMD overlap; see hardware calibration | Decode VALU competes with WMMA issue; S:5, `5e20dc48` |
| Treat library GEMM as a universal upgrade | Best 20–33 TFLOPS versus default 5.6, yet pinned best failed a one-row shift | Speed selection must preserve row/chunk contracts; S:28 |
| Larger output tile to remove weight rereads | Recorded bucket tile was about 60% slower; fewer L2 hits, unchanged misses | Distinguish cache hits from compulsory traffic; S:30 |
| Larger staging or double buffering without proven overlap | K128 4–8% slower; LDS double buffer 5.3–7.6 versus 3.5–4.6 ms | More stages can add resource/latency cost; S:30 |
| Duplicate decoding privately per wave or stage codes through LDS | 4.2 versus 3.6 ms private decode; LDS-staged trellis 5.0 ms | Decode reuse must repay extra work/traffic; S:30 |
| Universal hot-expert buckets | Cold experts slowed with forced large buckets; larger hot class also lost at 238–247 VGPRs/occupancy 4 | Choose by measured skew and live resources; S:30 |
| Fragment hoist/ping-pong/stagger | Hoist 5–15% slower, ping-pong 13–17% slower, stagger about 4× slower | Source-level overlap may inflate live fragments or guarded copies; S:30 |
| Direct-global operand fragments | 45% slower | Removing LDS can discard useful reuse; S:30 |
| Grid/priority/address-space spelling alone | Grid-z, priority and global-versus-flat probes were neutral | Changed ISA is not a measured gain; S:10,30 |
| Split-K for an activation-read-bound skinny GEMM | 1.17 versus 0.91 ms; finer tiles also slower | More blocks do not repair activation traffic; S:32 |
| Fuse a large epilogue regardless of occupancy | Net +28 ms/pass although standalone mix work shrank | Price the enlarged producer as well as removed pass; S:34 |
| Consumer non-temporal loads for every cache state | Dirty input −11% time, warm +48%, read-flushed +19% | Producer/consumer policies and cache histories differ; S:9 |
| Wider per-lane quant decode | Sixteen→thirty-two values raised 88→120–136 VGPRs and lost 13% end to end | Less repeated unpacking can lose occupancy; `QH/quant_ops.hpp:772–782` |
| Smaller staging solely to reduce resources | A compact staging variant used less LDS/VGPR but lost 3–5% | Resource count alone is not the objective; E27:8 |
| Wider weight prefetch universally | Cold-weight control lost 8–12% | Match production residency and live registers; E27:52 |
| Complete device copy / streaming hints as blanket fixes | Copy loss is recorded in memory patterns; cold streaming wins became about 1% model AR loss | Full traffic/cache history is authoritative; E27:100,103,106,109 |
| Coalesced attention loads or no padding everywhere | One V remap helped shallow 4% but lost deep 30%; removing padding lost to bank conflicts | Qualify both depth and LDS transpose behavior; `QH/kernels/attention_wmma.hip:54–59` |
| Tensor alignment alone as a speed feature | Decode about ±0.2%, prefill drift dominated | Repair format correctness separately from assuming speed; R2:529–554 |
| Global memory-policy disable / extra transfer queues | Short prefill or individual decode phases regressed | Prefer the narrow demonstrated runtime fix; EFN:231–234 |

## Open leads — shared mechanisms

These remain open opportunities, not promised end-to-end speedups. Check later
commits before acting; model-specific reachability and pending work are in Other.

| Lead | Bounded next action | Evidence |
|---|---|---|
| Deferred K5 indexer narrow tile | The 128×128 4×2 K32 single-group candidate reduced synthetic time 16.48%; qualify actual weights and model impact only with a later larger batch. The tested masked-wide K5 variant was slower. | `plan/exl3/F3-NEXT-MATRIX-SCREEN.md` |
| Stream-K bucket scheduling | Persistent 80/160-block scheduling has now been screened without a qualified benefit. Stream-K remains untried and would need its own reduction-order/quality proof; inspect skewed layer tail span first. | S:29–31; final Nix prefill section below |
| Reuse low-bit decode across small batches | Existing upstream mmvq repeats decoding per activation row; shared-row decoding is a new path requiring independent quality and real verification timing. | R0b:200,208 |
| Prepare actual first-use functions/graph keys | Profile one affected first use; broad warmup already exists and some stalls remain inconclusive. | W9:563–592; EFN:229,243 |
| Chunk reuse versus responsiveness | Use real bucket histograms and state hashes; larger chunks must preserve cancellation/checkpoint cadence and row arithmetic. | F3:91; S:31 |
| Storage-aware checkpoint selection | Compare shallower RAM plus compute against deeper disk restore under measured device latency, preserving state/lineage correctness. | `plan/gufo-upstream-port/DISKREAD-RESULTS.md:43–55` |

## Build and measurement provenance

- Preserve persistent CMake/Ninja objects; derivation temporary build trees can
  disappear even after a post-install failure. CMake legitimately rewrites RPATH
  on install: compare production/test build-tree ELFs before installation, not
  byte-identical installed versus build-tree ELFs. Separate assertion-enabled
  unit and release directories avoid recompiling shared libraries on every toggle.
- A failed health window cannot qualify timings. The first Nix10.1 operator
  screen has exact-output goldens and provenance, but its timings are advisory.
  Subsequent read-only cache reuse verifies bytes, runtime/toolchain/input identity
  and the full local header closure; metadata enrichment does not rehabilitate
  failed health. Final speedup must use the saved clean full-model T1 baseline.

## Other (model-specific)

Full model-specific results live in `docs/models/<model>/EXPERIMENTS.md`, `plan/exl3/*-RESULTS.md`, `plan/rotated-quant/*-RESULTS.md`.

The bullets below preserve the moved model-specific record, including rejected
searches and unqualified work. They are not policies or thresholds for a new model.

### Qwen3.8-27B

- **Quality/defaults:** the published GGUF sweep used FP16 KV, while the later
  default is KVarN k4v2; reproduce the historical precision explicitly
  (B27:3–17). Final same-pack engine KL means were about 0.7–4.2e-4;
  BF16-reference order was custom c02 approximately stock 5.00, ahead of
  stock 4.00. BF16 embedding default `3712dbe7` rejects a 1.11 GiB saving
  with a significant 0.04-percentage-point top-1 loss (XA:323–354).
- **RQ/interoperability:** FP32 rotated-GGUF and FP16-input EXL3 have separate
  contracts. Sign-seed effects and sensitive late-layer positions prevent
  claiming a general rotation gain from one seed (R0:142–190; R2:480–504).
  Prism signs are per input width, so old per-group/layer bytes cannot simply
  be relabeled. Alpha/beta are folded at load outside the upstream allow-list;
  binding errors were BF16 1.61e-4 and Q8_0 5.81e-4. Loader checks used
  identical-file/token llama.cpp controls
  (`plan/rotated-quant/R1C-PRISM-MAPPING.md:20–33`; R2:449–466; R0:113–139).
- **Alignment correction:** the quantizer copied declared alignment but laid
  bytes at its default alignment. The old preservation claim was retracted;
  converter `fe77eaa4` repacks output (R1b:344–351).
- **Shared decode dispatch:** X2b supports up to three same-K/rate matrices,
  eight-row passes and a full-group shape basis for subsets. PickShape uses
  two-tile-block count [352,512)→four tiles/four waves; <64→two/sixteen;
  otherwise K/16≥384→two/eight, default two/four. The prefill crossover
  R≥17 remains explicitly provisional; fast decode/verify covers R≤64,
  unsupported calls fall back (`QH/kernels/exl3_gemv.hip:455–469,677–725`;
  `QH/exl3_dispatch.hpp:30–88`).
- **Prefill dispatch:** FP16 low-bit GGUF prefill starts at R≥1024 for exact
  supported geometry and at least one low-bit FFN; all-Q8 retains quantized
  activations. Max prompt batch is 4096. Direct FP16 tiles choose 256×256
  for SwiGLU, N≥4096, or selected types N≥1024/R≥1024; only Q4/Q5 use the
  complete-tile bounds-free specialization. Final uncaptured-layer output
  keeps its original 256-token tile, after all cache updates
  (`QH/prefill_chunk.cpp:19–49,211–224`; `QH/arena.cpp:23,1124`;
  `QH/kernels/prefill_fp16.hip:1017–1048`).
- **Compiler/source specializations:** quant16 wave64 alone uses iterative ILP;
  both K-quant prefill units disable SLP. Affected Q8 prefill recovered
  291→464 tok/s (E27:130; `src/models/qwen/CMakeLists.txt:97–121`).
  Q5/IQ4 header caching, mixed-Q6 fetch-stage decode, IQ3 orientation and
  format/K-specific swizzles remain intentional
  (`F3src/models/qwen/hip/kernels/prefill_fp16.hip:250–310,679–698`).
- **Shared-core tile policy:** F3 dense dispatch uses 128×64 at R≤64,
  otherwise 256×256 or 128×256 when N%256≠0; these are audited existing
  values, not a new-model tuning recommendation
  (`F3src/models/qwen/hip/kernels/prefill_fp16.hip:1161–1201`).
- **EXL3 shape reachability:** q 5120→12288; k/v 5120→1024; out 6144→5120;
  GDN input 5120→10240/6144, output 6144→5120; FFN 5120→17408→5120;
  head 5120→248320; MTP FC 10240→5120. All output widths are divisible by
  256; none of these production projections reaches kWideHalf
  (`src/models/qwen/exl3_weights.cpp:96–155`; Format:74–77;
  `plan/exl3/XB-DESIGN.md:20–28`).
- **Numerical gates/tests:** X2b matched-input stage bound 1e-5 measured
  1.4e-7–3.4e-7; X3 prefill uses `(k/16) × 2^-23`, not a moving ratio to
  another Gufo GEMM. Verify remains GEMV. Packed-hash exhaustive checks,
  shape-basis equality, poisoning and bucket guards are in the kernel test
  (XB:222–227; X3:52–80;
  `tests/models/qwen27b/exl3_kernel_test.hip:602–627,1100–1237,1251–1326`).
- **Synthetic quant survey:** AR ms/token / verify-eight ms/step were
  Q8_0 126.2/128.2, Q5_K 83.4/90.2, Q4_K 73.4/89.0, IQ4_XS 64.0/89.1,
  Q3_K 84.8/133.2, IQ3_S 85.0/120.5, Q2_K upstream mmvq 42.5/186.1.
  These are whole-matrix synthetic sums, not service rates; low-bit grids
  and repeated per-row decode can erase byte savings (R0b:54–71,199–208).
- **Retained decode/skinny:** X2b lane-major/DPP measured 217 GB/s at one row,
  209 at eight; grouped projections and fused MLP/norm passed exact checks
  (XB:165–195,231–263; `c90bedea`, `4e040f65`, `2beae8f2`, `0c09450c`).
  Balanced selected 9–16-row groups (`4fc0211d`) leave smallest widths alone;
  selected Q5/Q6 narrow/query cases extend through 64. Four-row Q4 gate/up
  improved 615–641→578–582 µs; five-row IQ4 cold gains of about 7%/14%
  translated to prose 24.67→24.84 tok/s (Skinny:98–104; E27:9,18;
  `QH/kernels/prefill_quant_gemm.hip:1840–1866`).
- **Skinny qualification:** serving gains were small; batch-overlap assertions
  also varied on unchanged controls. Cost retune `eb78772f` failed
  full-acceptance and per-case timing checks, with no aggregate gain; it
  remained on branch pending a decision, not recorded as reverted
  (Skinny:125–167,319–355).
- **Rejected dense/verification searches:** broader Q4 row-first/smaller token
  groups, Q5 row-first/wave32/native wider rows/compact LDS/lane coefficients,
  IQ4 padding/staging, Q5 bit rotations, Q8 staging/packing, extra gate/up
  fusion, residual/norm fusion and split-attention graph gave flat/mixed
  components or no full-model gain. Keep their exact shape-specific routes
  (E27:8,19–20,53–61,79–85,91–93,103–109,114–116).
- **Exact FP32 prefill rejected:** Q4 prefill 695→90/138 tok/s was too costly
  despite decode-equivalent arithmetic (E27:27).
- **Attention:** wave64 verification preserves old partials; shared-KV
  partitions accelerated the eight-row component 2.95× at 32K and 3.66× at
  64K. Ordered draft V prefetch/paired heads gave about 0.8–1.0% complete-model
  TG; visible scalar causal tails preserve chunk invariance. Wider prefetch,
  row-sharing/partition variants did not generalize (E27:29,45–49,58,66–78,98,128).
  Optimized attempt starts at visible context 1024; split-K starts at 128,
  uses 32 splits and head_dim=256; score LDS capacity subtracts reduction
  scratch (`QH/detail/attention_policy.hpp:12–24,38–40,56–74`).
- **KVarN:** canonical rotated FP16 sink/tail, token-major K, channel-major V
  and scales remain a layout contract. Split count depends only on groups:
  <24→6, <48→8, <88→10, else 20, capped by groups. Prefill uses pass attention
  for R≤64, or R≤128 when not all heads fit
  (`tools/bench/kvarn_decode_bench.hip:38–67`;
  `QH/kvarn_layout.hpp:75–90,123–136`). Biased-code/late-V prototype reduced
  group-loop instructions 2248→1151 and 32K call 186.7→139.4 µs; deep tiny
  appends avoided whole-prefix repack, one row 1163→139 ms and 33 rows
  1223→399 ms at 256K (K:214–229,336–343; `cf7447b0`, `4a7046b2`).
- **KVarN dead ends/open work:** int8 WMMA, LDS dequant sharing, AR dot2,
  compact quantizer LDS/occupancy caps and all-layer staging did not qualify.
  Verifier issue cost and residual quantizer launch latency remain open;
  do not repeat those variants unchanged (K:218–229,344–346).
- **Speculation:** width-weighted acceptance, censored full acceptance and
  context cost were separately qualified. Fixed-width, joint allocation,
  extra q filtering, full-draft-then-truncate and saturated-only retunes lost
  on ordinary or perfect-acceptance cases. Prompt lookup must preserve message
  boundaries/private history/actual q; repetitive gains are not ordinary
  coding gains (E27:12–20,68–71,87–97,118–124).
- **Memory:** a small BF16 injection matrix deliberately stays cached
  (`QH/kernels/prefill_gemm.hip:167–178`); shared draft-feature backing removed
  duplicated slot allocation (`plan/gufo-upstream-port/U4D-RESULTS.md:70`).
  Standalone rotated output producers and disabled SwiGLU fusions remain
  candidates only if their measured share justifies work (R2:625–630).

- **Conversion/open studies:** resume must repeat `--out_scales always`
  and chosen `-cpi`; headers did not reveal silently changed scale policy.
  Custom calibration versus quantizer-version effects were not isolated;
  compute missing category intervals from retained records before another
  conversion (XQ:237–245,439–453).

### Qwen3.8-Flash-Next

- **Build boundary:** host executor stays `-O2`; MMQ keeps the C++17 device
  dialect and its MoE unit uses wave64, as does the W8A8 unit. Other HIP flags
  are covered by the shared compile lesson (`FM/CMakeLists.txt:28–70`).
- **Status/provenance:** F3 hold was `966f84b8`; Docker prefill improved about
  454→1255–1260 tok/s, but latest tile/hot-bucket/attention changes had not
  been measured together. Remaining qualification is verification debt, not
  another broad tile sweep (F3:229–246). Public GGUF results are an older
  revision (BFN:3–7).
- **Packs/quality:** fixed-corpus FP16-KV prefill without MTP measured
  KL_256: EXL3 5.05 0.0269, UD 0.0392, stock 4.05 0.0424, Kyojin
  approximately-four-bpw 0.0598. The pack label was not its actual bitrate.
  The actual BF16 teacher run took 2.84 h/8.2 GB peak RSS and supersedes the
  earlier runtime estimates (F1:420–436,482–487; F2Q:18–74).
- **Phase dispatch:** kVecBatch=8; MatrixRows = prefill OR R>8;
  ExpertMatrixRows = prefill OR R>32; WidePrefill = prefill AND R>8.
  EXL3 prefill always takes X3 including singleton tails; equal N/rate GEMMs
  group after separate suh rotations; slices require a single projection
  (`FN/executor.cpp:82–85,132–152,2035–2055,2093–2105,2134–2139,3887–3894`).
  The vector-threshold comment needs its decode/verify scope clarified;
  the “tile never changes” comment predates the hot class.
- **Expert classes/rates:** requested tile sizes are 16/32/64; production
  requests 64 and splits hot experts into balanced ≤128-row buckets. Device
  class-filtered kernels record 97–106 VGPRs/no scratch for the smaller
  class and 150/no scratch hot. Routed decode specializes widths 1/2/3,
  then 4/8; expert adapter integer K3–K7 is narrower than shared X3 K1–K7
  (`FN/executor.cpp:85,3891–3894`; `FN/exl3_moe.hip:365–414,1021–1055`;
  `F3src/models/qwen/hip/kernels/prefill_fp16.hip:1298–1319`).
- **Routing/decode budget:** at pp2048, experts above 64 assignments held
  51–67% early and 80–85% late, with 151–199 unused late experts. About
  5.9 GB weights/token streamed: trunk/head 205–221 GB/s, routed 141–158,
  plain HC 126–165 (S:31,43–45). K7 window decoding reduced instruction count
  1379→1015 without spills; PickShape sweep found no useful retune beyond
  about 0.1% for shared-down (F3:182–185; S:45).
- **MoE glue:** activate-plus-scale 0.67→0.60 ms (`bf28112b`); fused scatter
  removes MoveRows. Exact down scaling is already implemented after FP16
  rounding with widened fallback (`f6471551`); do not rediscover it
  (S:35; `FN/exl3_moe.hip:177–207,230–288,605–608`). The recorded bucket
  ablations also include no decode 3.25, no weights 3.4, neither 2.85,
  no WMMA 3.04, skeleton 1.06 ms; estimated floor near 2.05 is not achieved
  throughput (S:29).
- **MoE dead ends:** two-stage register prefetch was neutral; paired gate/up
  was initially shelved at estimated 10–20 ms/pass because independent expert
  suh loses operand sharing. A later K5 single-launch grouping screen, preserving
  separate inputs, also found no useful gain (see November 2 entry below). Producer-fused rotation was analytically rejected at a
  modeled 38% gate/up increase; it was not a measured candidate (S:30,35–36).
  Rejected parameterizations included output width 320, hot 256-token class,
  forced 128-row buckets, 16-B trellis-through-LDS and two-deep fragment
  ping-pong; see the general rejection table for transferable outcomes (S:30).
- **Dense shapes:** shared gate/up each 2560→640 and combined indexer
  2560→640 have N%256=128, so dense calls above R=64 use kWideHalf.
  The indexer exposes q/k 512/128 views but projects the full descriptor
  (`FM/config.hpp:25,48–49,64`; `FM/exl3_weights.cpp:309–314,341–346`;
  `FN/executor.cpp:2078–2118`). Routed N=640 uses bucket tiles, HC N=320
  is plain storage. Other dense widths—q 12288, k/v 512, qkv 10240,
  z 6144, out/shared-down/FC 2560, head 248320—are divisible by 256
  (`FM/config.hpp:25–26,36–38,55–58`;
  `FM/exl3_weights.cpp:280–306,332–354`).
- **Dense core migration/open path:** EXL3 plain projections use X3; GGUF
  shared-down still calls DenseF16Gemm. Old core at 6144→2560 fell
  26.0→15.5 TFLOPS on doubling rows (3.35× time); shared-down fell
  29.1→21.5. Confirm surviving GGUF reachability before assuming every
  production path migrated (`2097eb14`; S:27;
  `FN/executor.cpp:2012–2016,3670–3674`).
- **GGUF thresholds:** DenseF16Route tests output N≥2048, K≤2560 and staging
  capacity plus MatrixRows; kDenseF16MinRows is not a token threshold.
  Outside WidePrefill, assignment count ≥16×expert count picks 48-token
  routed tiles, else 16; WidePrefill instead uses down 64 and paired gate/up
  128 (`FN/executor.cpp:164–192,2012–2016,2414–2424`). Earlier mid-size
  wide-prefill dispatch started too late; above-eight prefill fixed that
  cliff (EFN:202–206).
- **Plain operands/tests:** FP32 SmallGemmStable was 53% of prefill and tiled
  stable FP32 still about 1 TFMA/s; restored FP16 WMMA changed a down call
  10.4→0.92 ms (`2fcecf6e`, `6fde4b45`; S:33). Direct
  exl3_invariance still checks legacy stable-helper cohort/chunk bytes;
  production executor prefill has separate dense/plain/expert/GDN tests.
  The name/scope should be clarified, not that old gate removed
  (`FT/exl3_invariance_test.cpp:65–101`; `FT/CMakeLists.txt:78–87`;
  `FN/executor.cpp:2255–2274`). Trace standalone-helper callers before
  treating the old stable fallback as a production prefill bottleneck.
- **Routed numerical bounds:** freeze the FP16 intermediate; final E2E bound
  is `control_L2 × 1.05 + 1e-5` with exact producer bits and strict stage
  checks. Early small-down diagnosis was unconfirmed; later subnormal probes
  establish the hardware issue, not universal attribution (F2I:155–201).
- **HC retained:** four-lane combine at every EXL3 prefill width (`c8625608`),
  normalized input written once as FP16 and recomputed in mix saved
  85 ms/pass (`0d051491`), prefill W8A8 Q8_0 down copies cut
  1.78→0.46 ms/call and 109 ms/pass with reported q6 pass (`8737db4e`;
  S:32,40). Keep full-width normalization; this is separate from decode int8.
- **HC specifics/open work:** inject/projection fusion begins at 96 rows;
  HcDownSiluF16 intentionally runs stable GEMM then SiLU separately because
  fusion changed row bits. Gather latency and producer store policy remained
  open at hold (`FN/executor.cpp:2593–2594,2741–2750`;
  `FN/kernels.hip.cpp:8108–8113`; F3:242–246).
- **HC rejected:** FP32-input fused fetch 1.466 versus 1.417 ms warm;
  stable row-tiled decode was slower. Group-int8 decode gained only
  3.5% shallow/2.4% deep, added 659 MB and missed the requested 10% bar;
  confidence intervals crossed KL/top-1 limits even though the mean KL
  delta looked below its nominal limit. It was not merged; the experiment
  remains a separate branch. Failed HC tile shapes were 64×64, 128×32 and
  64×32 (S:32,45; F1:489–517).
- **GDN:** conv/qk prep fusion improved 3.3→2.8 ms/call (`024a2d5b`). Whole
  path about 650 MB/call at 220 GB/s coexists with a VALU-bound recurrence:
  1833 instructions, 360 ordinary FMAs, 123 dual FMAs, 271 delay instructions
  (S:39). Two state rows/lane share q/k; four-token windows/three named
  windows in flight; fused conv reads history until the later shift
  (`FN/kernels.hip.cpp:1782–1815,1946–1950`). The GGUF Q8_0 output path
  aliases FP16 output with FP32 scratch for the 6144→2560 shape; at ≥1024
  rows it pads the 6144-wide pitch by 32 halves
  (`FN/executor.cpp:3126–3145`).
- **GDN dead ends/tests:** head remap neutral; eight waves 3.63 versus
  3.30 ms (S:39). Chunked WMMA DeltaNet/fused norm was removed: about 4%
  prefill did not justify changed greedy tool output. BF16 residual streams
  also lost quality (EFN:3–32,73–92,109–112). gdn_ops covers fused/separate
  conv, ragged chunks, checkpoint history and short rollback prefixes;
  attention_rows checks unfused norm/RoPE/rotation/gating with/without YaRN
  (`FT/gdn_ops_test.cpp:742–797`; `FT/attention_rows_test.cpp:79–126`).
- **Attention/selection:** fused prologue/epilogue extras 45→23 ms/pass
  (`2e054760`; S:41). Exact selection scan/head groups retained masks/hashes
  (EFN:264–265). Direct V/per-call packed V, union/lane-permute variants,
  query/key sharing, sparse scheduling/scorer grouping failed to qualify;
  membership-sorted tiles changed bits despite fewer WMMAs
  (EFN:47,251,259,261,264). Score scratch is capped near cache size;
  ranked queries stay FP32 (`FN/executor.cpp:1265–1271,3363–3375`).
- **Duplicate indexer work (resolved):** indexer_k/q share one descriptor and
  input; the original sparse prefill computed it separately for both slices.
  The retained implementation reuses the complete result and preserves its
  full-descriptor rotation. Initial source audit (`FM/exl3_weights.cpp:309–314`;
  `FN/executor.cpp:2078–2118,3338–3352`).
  Resume 2026-11-01: a candidate on `gufo-f3-integ@966f84b8` now shares the
  parent projection and copies both slices, retaining singleton chunk sizing.
  Independent code review and the focused dense-prefill unit passed; FP16-KV
  and KVarN collector records (32,760 rows each) match hc7 byte-for-byte.
  Nix ROCm 10.1 qualification subsequently passed in the saved q3/q6 and
  final prefill-batch receipts; the following Docker observations are historical.
  Baseline Docker 7.2.4 T1,
  one run: pp2048 971.66, pp8192 1354.23, tg128 24.87 tok/s; lower pp2048
  versus historical probes is unexplained, not a matched regression finding.
  Candidate Docker probe: pp2048 1285.56, pp8192 1339.10, tg128 25.04.
  Reuse only activates beyond 2048 tokens, so the pp2048 change is not an
  optimization gain. pp8192 difference is -1.1%; no baseline repeat was run.
  See `gufo-f3-integ/plan/exl3/F3-RESUME-RESULTS.md` and receipts
  `fn-resume-966f84b8-stamped/`, `fn-indexer/` under the raw-evidence root.
- **GGUF MoE dead ends:** smaller dense tiles/shared rows-first tiles did not
  improve full span; high-nibble half-subnormal trick was noise to 2% slower;
  repacks need all decode/verify/fallback readers rewritten; deeper routed
  prefetch/lane epilogue/wave caps did not beat retained coalescing
  (EFN:48–50,113–114,263). Side-stream value is compiler-dependent
  (EFN:52–71); only qualifying Q8 shared weights overlap, with scratch joined
  before reuse (`FN/executor.cpp:3636–3658`).
- **KVarN:** k4v4 quality differs from 27B k4v2. Chunk-matched fake quant is
  authoritative; fake-after-prefill misses prior quantized-chunk feedback.
  Cache/snapshot/persistence were disabled, so server suites stopped at
  expected cache assertions; later checks were not exercised. Continuation
  support is unfinished qualification (F1:171–197,438–473).
- **MTP/reference semantics:** Gufo full-HC hidden norm differs from upstream
  per-stream normalization. Keep that mismatch visible, not silently change
  Gufo to fit the oracle (F2D:159–168). EXL3 KV-only catch-up exists
  (`5156e280`), and subset QKV retains full-group geometry (`3d725c83`).
- **MTP/serving policy:** confidence-stop `92fcaf37` gained 8.9–10.4% C1
  greedy with held-out coverage, but two first-use cases remained inconclusive;
  this does not qualify sampled/concurrent policy. Global sampled cost retune
  helped code but hurt thinking; private prompt lookup and compact sampled
  transfers have distinct eligibility/replay rules (W9:551–586;
  EFN:172–174,191–195,246–250). Cost tables still cite older Q4/Q8 provenance;
  EXL3-specific recalibration is an open lead, not an approved retune
  (`FM/mtp_costs.hpp:11–16`).
- **Catch-up/chunk rules:** normal prefill chunk 4096 may absorb tail 128;
  old served prompts motivated this, not a new EXL3 sweep. Predictor pruning
  retains aligned wide tiles/at least 96 rows, or two short rows to retain
  multi-row arithmetic, after all KV/indexer writes
  (`FM/engine.cpp:26–34,256–262`;
  `FN/executor.cpp:4436–4439,5720–5736`).
- **PLE/I/O/graphs:** table is already disk-backed; first gather waits, next
  prefetch overlaps. Pool has 128 readers, batches eight reads at ≥1024 jobs,
  keeps original quantized cached rows and avoids waking readers for small
  cached gathers (`FM/ngram.cpp:18–25,175–199,220–224,318–341`). Cold long
  single-session prefill showed I/O cost despite a short pass being off the
  critical path (F1:85–101; S:50). Flat scratch construction removed an
  87.35 ms allocation stall; indexer plan startup was 104–115 ms
  (EFN:243,245). PLE splits graph execution before injection so earlier
  layers hide the wait; eligibility includes pool backlog/image prefix,
  and diagnostics disable graphs/mark bucket tile-read accounting incomplete
  (`FN/executor.cpp:3895–3899,4100–4105,4288–4329`).
- **Serving/memory dead ends:** completed-prompt capture sharing and smaller
  reserve regressed individual phases; do not accept total request gain as
  a phase pass (EFN:213,252). The old anonymous-floor interpretation mixed
  precommitted checkpoint backing and per-slot buffers; preserve actual allocation/cache
  accounting (`plan/gufo-upstream-port/ANON-FLOOR-RESULTS.md:27–43`).
- **Historical status:** earlier reference-prefill/rate-support limitations
  are superseded by the later implementation/API and hold record
  (F3:76,86,155–203,229–246).
- **Draft-vocabulary fixture framing:** EXL3 5.05 on Nix ROCm 10.1 interpreted
  the lifecycle test's raw repeated Thai prompt as a user request and switched
  to English reasoning. Its 19 output IDs matched AR exactly, but only six
  were rare, failing the unchanged one-third coverage requirement. Keep
  coverage failures distinct from numerical/token mismatches. Frame repetition
  as an assistant continuation using the existing non-thinking chat template;
  retain rarity, actual neural drafts, acceptance and restore assertions.
  Evidence: `fn-draft-diagnostic-nix101/receipt/run/draft_vocab.log`; revised
  fixture qualification is recorded in the F3 resume results, not implied by
  the diagnostic equality alone.

- **EXL3 CPU oracle storage boundary:** ideal FP64 input rotation omits the
  GPU FP32 producer and FP16 rotated-input store. Preserve the ideal mode,
  but use explicit producer/storage semantics for a decode-stage audit.
  Verify independently with dyadic inputs and a dense Sylvester transform;
  include block permutations and split predictor FC calls. Use EXL3 head
  projection for EXL3 weights; a GGUF Q8 head oracle cannot read that layout.
- **Nonlinear-stage isolation:** tiny upstream rounding differences can cross
  a downstream FP16 bin. Keep the CPU-computed FFN input in the trace and
  check it independently (2e-6), then replay observed GPU input for the
  downstream FFN/carry comparison. Never inject FFN output to make carry pass.
  This staged audit is not end-to-end ideal quality. The unchanged carry
  limit 5e-5 passed with max4.71e-5 across eight text/image states in
  `fn-final-q3-mtpinput-nix101`; all full-head/cache/batch exact checks passed.
- **Bound CPU reference cost:** even a three-token full-model/full-vocabulary
  CPU probe took 604.27 s on the 5.05 pack. All three top-1s matched; printed
  KL was 0, 0.00001, 0.00175. The probe has no numerical quality threshold
  and saves summary metrics, not reference vectors. Do not routinely repeat;
  use saved BF16 vectors for the pending q6 full-model quality gate. q3
  evidence is indexed in that receipt's `q3-summary.json`, with health_PASS.
  Passed sampling/prefill/32K and old baseline timing were not repeated.

- **Compiler migration and quality:** Nix10.1 q6 on5.05 preserved the broad
  historical UD advantage (KL2560.026481 vs0.039227; top1 92.0543% vs90.8608%),
  but had a resolved mean NLL increase+0.006600 against hc7. Do not hide it
  behind unchanged meanKL; inspect target-loss tails. User prefers performance
  over correcting small aggregate drift; no precision patch was applied.
  `fn-final-q6-nix101/Q6-SUMMARY.json` records the decision and full evidence.
- **HC weight quantization ties:** an isolated exact-source quantizer, on a
  captured real320x10240 F16 matrix, differed in2153 int8 codes across compiler
  builds under one Nix10.1 runtime; scales matched. All differing inputs were
  |v|=amax/2, legacy +/-63 versus Nix +/-64. This does not prove the production
  q6 cause. Regenerated ISA alone did not explain it; do not claim a product
  barrier preserves legacy arithmetic. Preserve the payloads and contract
  distinction before any future repair. Diagnostic recipes are in the F3
  worktree's `build/f3-q6-quant-diagnostic/`; no full baseline was rerun.

- **Prefill first (2026-11-01):** user defers TG/MTP tuning. Preserve Nix10.1
  AR pp2048 1372.10 / pp8192 1411.52 tok/s from
  `fn-final-t1-x-ar-nix101` as the current baseline; no repeat baseline run.
  A pending MoE-combine optimization can use original routing IDs because
  inverse[assignment] was constructed in exactly that expert's bucket.
  Preserve IDs through deferred combine and shifted final-layer RowScratch;
  retain slot order/arithmetic and qualify output bytes before claiming
  exactness. Saved older profiles identify leads but do not establish a
  same-toolchain speedup. TG results and intentional T3 cancellation are in
  the F3 resume results; they are outside this prefill iteration.

### Flash-Next Nix ROCm 10.1 prefill batch (2026-11-01)

Direct routing IDs, pre-pruning WMMA/KVarN eligibility, and grouped dense 256x256
iglp0 are retained after `fn-prefill-batch-nix101` COMPLETE/health_PASS. The K7
operator checks and32,760 collector records/64 head checks pass; payload equals the
saved baseline exactly. Saved pre-batch Nix T1 → one candidate r2 window:
pp512 1039.61→1061.85, pp2048 1372.10→1400.15, pp8192 1411.52→1442.28,
d32K pp2048 1257.46→1280.80 tok/s (+1.86–2.18%). This is a combined observation,
not per-change attribution; existing hc7-relative quality drift remains.

The225-row synthetic tail case exposed generic attention reading uninitialized
masks when unsupported WMMA pruning was requested. Gate pruning before query
projection/selection on WMMA geometry, sparse live extent, and KVarN pass.prefill.
Its97-of225 expert-row diagnostics and chunk logits pass exactly in 0.564 s.

Routed iglp/split32, wider BM256 16x1/8x2 and persistent80/160 candidates remain
unpromoted: exact outputs but added complexity and no qualified benefit. First
operator timings came from a health-failed window; later healthy candidate
comparisons cannot qualify those old control times. Artifacts remain in f3-integ
`build/f3-prefill-matrix`, including `archive-wide`/`archive-persistent`.
Performance-neutral simplification or correctness can still justify retention.
Final image `f3-prefill-batch-f712b2d0-rocm101`, code patchf712b2d0, preserves all
previously completed work; no new TG/MTP qualification or baseline inference.
See f3-integ F3-RESUME-RESULTS.md final section and BATCH-QUALIFICATION.json.

### Flash-Next matrix follow-up screen (2026-11-01)

No production change; the qualified pp2048 baseline remains 1400.15 tok/s.
Two healthy, short Nix 10.1 operator windows reused saved outputs and avoided
full-model inference. [Detailed tables and identities](../plan/exl3/F3-NEXT-MATRIX-SCREEN.md).

- K32 staging paired with fewer waves was mostly slower: wide dense +21–59%,
  routed BM64 +11–22%; routed BM128 gave gate +2.11% / down −1.97%.
  Masked dense 256×256 at N640 lost 19–26%; the tested K5 variant is rejected.
- Single-group K5 K2560/N640 with 128×128 4×2 K32 improved 0.255579→0.213457 ms
  (−16.48%), but two groups lost 2.34%. Preserve as a deferred candidate.
  Pack metadata identifies the indexer parent as K5 and shared gate/up as K7.
  Indexer key prefill computes the complete parent even before sparse attention,
  so pp2048 d0 can reach this candidate; its model impact remains unmeasured.
  Do not generalize the K5 signal to shared experts.
- Production's 4224-row scratch launches 1172 bucket slots per class, versus
  the old operator's 832. A first 1172-slot measurement gave gate/down
  2.979682/3.198049 ms. Tightening to668 changed the 2×gate+down estimate +0.32%;
  cold512/hot315 descriptor partition changed it −0.12%, excluding builder
  cost. No meaningful aggregate benefit; neither route is promoted.
- All screened matrix outputs matched exact finite bytes. Only the separate
  unequal65/129-row dense tail check exercised allocation guards; generic
  timed cases did not. These K5 checks do not qualify full-model quality.

Receipts: `fn-prefill-stage-screen-nix101` and
`fn-prefill-partition-screen-nix101`, both COMPLETE/health_PASS. Source,
objects, resources, and controls stay in f3-integ `build/f3-prefill-stage` and
`build/f3-prefill-partition`; production incremental trees remain intact.

### Flash-Next narrow dense and output-slice batch (2026-11-01)

- K5 singleton N640/K2560 StageK2/128²/4×2 is exact and faster at R512
  (−29.22%) andR2048 (−15.31%); R4096 regressed1.78%, so dispatch stops
  at2048. Grouped/residual/other-bit paths remain unchanged.
- Singleton output slices aligned to128 channels can preserve the parent
  trellis K-stride while computing only the slice. Keep original input
  rotation, offset trellis by begin×bits uint16 elements, and offset svh by
  begin. Paired q/k outputs still need one complete parent calculation.
- Routed WMMA operand transpose plus vector stores: gate+2.18%, down+.01%;
  null-exponent scaling removal did not improve the aggregate. Wave64 with
  corrected32-lane decoder subgroups: gate+1.08%, down−.21%; wider hot groups
  lost8.92%/9.45%. No routed variant promoted. Use saved1172-slot controls,
  not the older832-slot timings printed alongside cached-output identities.
- Dense K7 epilogue Stage64 reusing full64KiB LDS was+.08%/+3.03% on the
  large shapes; XOR output swizzle/combinations were worse. Exactness and
  output guards passed, but these variants remain prototypes.
- Cached raw FP16 W′ feeding the *current* X3 producer/consumer (distinct
  from the old UnquantizedF16Gemm reconstruction test) passed exact outputs
  and guards but large shapes lost57.30%/7.31%, excluding initialization.
  Smaller tile consumers spilled and lost37.97–190.10%. K640/N2560 improved
  3.06% locally; reconstruction adds.010621ms. Expanding all300 trunk K7
  parents costs5.42GiB and2.286× weight bytes; no model cache is implemented.
- Qualified model batch: pp5121062.82±.08, pp20481405.25±.83,
  pp81921446.11±1.33, d32K pp20481282.89±2.18 tok/s, respectively
  +.09/+.36/+.27/+.16% versus saved prior batch. The32760-record payload
  and64 head checks remain exact; hc7 NLL drift+.006600 is not removed.

All detailed operator values, exactness limits, source/runtime fingerprints,
and receipts: [batch2](../plan/exl3/F3-PREFILL-BATCH2.md) and its linked JSON.
Units/collector passed before a wrapper0644 packaging fault; mode0755 repair
changed no ELF bytes, and only the unexecuted benchmarks resumed. Both windows
passed health fences; the first remains a recorded failure, not relabeled.

### EXL3 register-pressure evidence scope (2026-11-01)

Static inspection, no new build/GPU measurement: 13 matrix specializations in
preserved Nix ROCm 10.1 stage-screen assembly have 89–214 VGPRs, zero private
frame, and zero SGPR/VGPR spill counts. This does not cover every kernel in the
final qualified image. [Metadata and provenance](../plan/exl3/F3-REGISTER-AUDIT.json).
[Candidate scope](../plan/exl3/F3-SEARCH-STATUS.md#register-pressure-and-kernel-fission)
separates actual spill removal, array scalarization, independent output tiling,
stage fission, and resource fit for streams. Existing smaller-tile regressions
remain evidence; high register use alone is not a demonstrated spill bottleneck.

### Flash-Next dense traversal sweep (2026-11-01)

All seven actual trunk K7 shape/group combinations were screened with exact
outputs and guards. Singleton K6144/N2560/R2048 packed traversal group1 changed
1.930083→1.639585 ms (−15.05%); a narrowly guarded production dispatch is
qualified below. Other best packed alternatives: K2560/N6144 +2.37%,
K640/N2560 +1.49%, K2560/N10240 +4.23%, K2560/N12288 −.48%, grouped N512 +.57%,
grouped N640 −1.74%. These do not warrant extra branches. Controls for the four
newly needed shape/group combinations were saved once; prior controls reused.

Cache-fed current X3 with tuned traversal still lost17.82%/2.70% on the large
shapes; small K640/N2560 gained at most4.55% before initialization. No persistent
model cache is added. Partial-group mapping was independently checked; the
actual release object also passed untimed saved-output checks. Reuse batch512
collector evidence separately from new2048 operator evidence. See
[detailed traversal results](../plan/exl3/F3-PREFILL-GRID.md) and its linked JSON.

The dense traversal batch subsequently qualified: pp5121063.05±.22,
pp20481416.51±.82, pp81921447.47±1.36, d32K pp20481291.03±1.09 tok/s.
That is +.11/+1.17/+.36/+.80% versus the saved starting batch; pp2048 gains
+.80% over batch2 alone. Actual release-object exact/guard checks passed;
batch512 collector evidence was reused under its explicit unchanged scope.
The >=2000 tok/s target remains unmet. [Qualification](../plan/exl3/F3-PREFILL-GRID-QUALIFICATION.json).

### Flash-Next compact cold-class closure (2026-11-01)

Healthy cached-control screen: raw128×16/32/64/128 8×1 kernels with disjoint
bucket classes, unchanged K16 order and packed-row offsets. Compact32/64/128
measured gate3.027125/down3.184357 ms; compact16/32/64/128 measured
3.115747/3.220175. Against saved1172-slot2.979682/3.198049 controls, the weighted
2×gate+down estimate loses.89%/3.21%. All launches and realistic512-per-cold,
315-hot bounds are included; device partition builder cost is excluded. Exact
finite bytes and guards passed. No promotion or another model rerun.

Live classes:127/56/62/155 buckets, carrying4.63/6.50/14.08/74.79% of assignments.
Existing padding-skip already avoids unused WMMAs; four classes reduce staged
B rows22.21%, but only5.72% beyond split32. The older split32 health-failed
window remains advisory; this new healthy compact test closes the distinct
partition/LDS-pressure idea. [Search status and limits](../plan/exl3/F3-SEARCH-STATUS.md).

### Flash-Next stream scheduling screens (2026-11-01, initial operator evidence)

Healthy pinned Nix10.1 screens reused controls and exact production matrix
objects. At2048 rows, Q-side Q/KV projections improved4.40757→4.28673ms(2.74%);
shared MLP beside raw routed MoE+combine improved15.0755→14.7609ms(2.09%).
These are candidates, not model throughput claims. Shared/routed excludes
router/top-k; production integration starts shared before those operations.
At512 rows these gains were only.70/.74%. Grouped heterogeneous Q/KV was
slightly slower than streams, so no extra GEMM batching mode is promoted.

GDN late Z only gained.54%(7.89030→7.84739ms); simultaneous Z lost.51%.
The GDN screen freezes alpha/beta and excludes their projection. Smaller-LDS
Z tiles lost2.72–5.16%, grouped GDN lost1.79%; fewer resources did not produce
a better complete span. Routed gate/up overlap gained only.61%; sharing the
single side stream with simultaneous shared/routed work would be unsafe.

Expanding Q/KV overlap through norm/RoPE/cache preparation gained2.34–2.42%
against5.18853ms, versus2.25% for the projection-only fork in that same scope.
The extra refactor is not justified by that small difference. Attention and
KVarN quantization were excluded. The65-row projection/branch gains24–25% are
not extrapolated to large-model prefill. All candidate outputs/guards were
exact, including GDN recurrent/history state and immutable KV prefix. Details,
512-row results and provenance: [stream ledger](../plan/exl3/F3-STREAM-OVERLAP.md).

### Actual release resources and KVarN register selection (2026-11-01)

Static extraction of the qualified115MB release ELF covers3383 kernels. All79
EXL3 prefill matrix specializations have zero private bytes and zero SGPR/VGPR
spill counts; max219VGPR, routed128×64/128×128 use105/149 and dense K7 256²188.
Relevant plain F16, HC and GDN kernels also have no private/spills. This does
not describe the whole binary: SelectScore retains18SGPR spills/private0,
unrelated Q4_K has16Bprivate/3VGPR spills, and packed fused F16 attention has
544Bprivate but is outside the current EXL3 separate-QKV dispatch. No new
spill-driven fission of those hot matrix kernels is justified by this audit.

KVarN group/boundary quantizers have80Bprivate despite zero spill counters.
ISA shows34static scratch accesses: the compiler reconstructs indexed reads
of two eight-element standard-deviation arrays. Unconditional empty `+v`
barriers after the unrolled selections keep them in registers without changing
arithmetic. The copied matching-flags prototype has0private,166VGPR vs162,
unchanged36096B LDS and no spills. Actual-release-control screen:1/4/16groups
.116067/.116459/.380499→.104360/.112471/.328080ms (−10.09/−3.42/−13.78%).
All quantized record bytes, prefix and guards match; health_PASS. New matched
controls are saved once. Production qualification and selection follow below.

The saved hc7 profile puts this quantizer at10.13ms/24calls versus987+726ms
for its two largest matrix entries; it is a bounded cleanup, not a claim of
large whole-model speedup. No new profile was collected.
[Actual release audit and screen provenance](../plan/exl3/F3-RELEASE-RESOURCE-AUDIT.json).

### Whole-model stream selection and final closure (2026-11-02)

Do not promote the initial Q/shared combination: it was exact but measured
pp5121047.19, pp20481393.02, pp81921423.83, d32K1274.65tok/s, losing
1.49/1.66/1.63/1.27% to saved controls. Same-allocation pp2048 isolation gave
quantizer-only1417.09±1.37, Q-only1417.67±.20, shared-only1425.06±.51.
Separate Q event handles still gave1399.91±.26. The interaction's cause is
unproved; independent local wins did not compose. The first isolation attempt
failed to mount model metadata and ran no inference; it is not a timing sample.

Retain shared/routed overlap plus register selection, restore baseline Q/KV
launching, and remove the extra Q scheduling/events/counter. Final actual image
`f3-shared5-703d3e77-rocm101` qualified COMPLETE/health_PASS:

| Case | Saved baseline | Final tok/s | Change |
|---|---:|---:|---:|
| pp512 |1063.05±.22|1068.58±3.09|+.52%|
| pp2048 |1416.51±.82|1429.13±1.13|+.89%|
| pp8192 |1447.47±1.36|1460.97±.73|+.93%|
| pp2048,d32K |1291.03±1.09|1301.99±1.41|+.85%|

Final-image1025-row test covers two full512-row chunks, four shared forks, a
serial tail, all residuals/logits and GDN state against the serial fallback.
Both KVarN format CPU-oracle code/scale tests pass; allfour final quantizers
have0private/spills. Supporting32760-record/64replay FP16-KV collector evidence
is reused from the combined image: retained MoE source matches and Q code is
restored baseline. No fresh fullcollector on the cleaned image is claimed.
KVarN performance usesk4v4; its correctness is separate operator evidence.
Baseline inference/profile runs:0. Target2000 remains unmet.

[All bounded numeric records](../plan/exl3/F3-STREAM-SCREENS.json),
[whole-model qualification and scope bridge](../plan/exl3/F3-STREAM-QUALIFICATION.json).


### Routed K5 gate/up grouped launch screen (2026-11-02)

A new paired grid selects bank/input/output per block, retaining independent
rotations, K16 accumulation and FP16 rounding. Cold64 and hot128 launches each
cover both banks: four matrix launches become two. Compiler resources remain
105/149VGPR, 24/32KiB LDS, zero private bytes and both spill counters zero;
no device workspace is added.

The healthy 4.92-second synthetic shared-MLP + routed-MoE + combine screen
reused the saved512/2048-row controls and all payloads. Serial-shared grouped
latency was8.71150/15.08650ms versus8.61897/15.07550ms (+1.07/+.07%). With
the retained shared side stream it was8.65352/14.89320ms versus saved shared-
side8.55525/14.76090ms (+1.15/+.90%). Small historical differences are advisory,
not a proven regression; there is no useful speed signal. Unlike a neutral
simplification, this introduces paired dispatch/grid cases. Not promoted.

Shared, combined and raw-down outputs are exact and finite; output guards pass.
Gate/up intermediate values were not separately compared. Five linked objects
match old controls; the kernels TU differs. No fresh independent oracle,
full-model qualification, baseline run or profile. Prototype/source/objects
remain in `build/f3-grouped-moe`; [all results and provenance](../plan/exl3/F3-GROUPED-MOE-SCREEN.json).


### Current shared5 prefill profile (2026-11-02, user requested)

Native Nix roctracer10.1 timestamps now cover the exact qualified shared5 ELF
(SHA b6b28244…) under patched ROCr88j/CLRal4. A bounded collector activates at
first embedding: startup-inclusive collection twice hit its2Mrecord cap;
those failed receipts remain. The successful32.81s process includes29.31s
loading, one2048 warmup and one2048 measured forward, d0/k4v4/AR/context4096.
Exactly4798 GPU launches match4798 HIP API correlations,2399 per forward,
with no invalid timestamps or graphs. No prior baseline was rerun.

Measured GPU span1434.413ms, kernel union1429.764ms (**99.676% busy**), gap
4.648ms. Kernel+copy union1430.050ms;94 barrier-wait intervals are excluded.
The main stream holds1428.485ms and the shared side stream342.044ms, with
340.764ms temporal overlap and1.280ms side-only activity. Summed kernel time
1770.529ms is not latency. Temporal overlap does not quantify net savings.
Kernel-active coverage is not SM/ALU utilization; the1386.046ms host stream
synchronize call waits on the GPU and is not additive CPU overhead.

| Main-stream work | ms | % of measured GPU span |
|---|---:|---:|
| Routed MoE matrices | 463.445 | 32.31 |
| HC-family mixing/projections, including fused routed combine | 326.251 | 22.74 |
| Dense EXL3 matrices | 295.706 | 20.62 |
| MoE routing/rotation/activation | 104.580 | 7.29 |
| GDN recurrence/epilogue | 100.231 | 6.99 |
| Attention/KV preparation | 57.339 | 4.00 |

Routed gate/up account for303.137ms; down160.307ms. Largest main dense shapes:
K2560→N10240 (GDN QKV)100.094ms, K6144→N256086.207ms, K2560→N6144 (Z)62.545ms,
K2560→N12288 (attention Q)40.461ms. HC-family mix/combine costs229.990ms, including86.710ms in the fused
routed-output restore/combine specialization; it is not pure HC math.
Main-stream MoE+dense matrices total52.92% of span. Prioritize their kernel
work and HC passes; pure host/launch idle is a small opportunity at this shape.

No hardware counters were collected: timing alone does not distinguish ALU,
LDS, dequantization and DRAM stalls. Reuse the actual release no-spill audit.
Profiled1426.48tok/s is advisory (context4096 versus earlier benchmark8193);
qualified clean1429.13 remains current. Existing `tools/prof/prof.py` exactly
cross-checks interval totals; its generic stage map is not used for EXL3
attribution. [Raw receipt, shapes, coverage and limitations](../plan/exl3/F3-CURRENT-PREFILL-PROFILE.json).

### Routed MoE ceiling diagnosis (2026-11-02, offline only)

Current gate/up is303.137ms for6.31662e12 useful FLOPs (20.838TFLOPS):47 full
2048-row FFNs plus the final128-row FFN. The earlier21.253 estimate incorrectly
counted48 full layers. The
55.34TFLOPS register-fed reference used ROCm7.2.4; their ratio is not hardware
utilization. Saved synthetic routing has20,480 useful rows,23,648 WMMA rows and
35,520 B staging slots; current kernels skip dead16-row WMMA subtiles but still
prepare full64/128-row inputs. This is logical work, not measured DRAM traffic
or the current model's routing distribution. Decode/issue competition and LDS
reuse have historical support, but current causal percentages remain unknown.
No GPU measurement, production change or quality qualification was added.
[Calculations/provenance](../plan/exl3/F3-MOE-CEILING-DIAGNOSIS.json) and
[ordered plan](../plan/exl3/F3-SEARCH-STATUS.md#ordered-improvement-plan) preserve
existing rejected experiments and prioritize dead-subtile preparation before
conditional trellis-address work, with one final combined qualification.

### Routed input staging follow-up (2026-11-02)

Implemented two exact-work candidates: skip B global/LDS preparation for
wholly dead16-row subtiles, and remove the K-tail test guaranteed false by
validated EXL3 k%128 with StageK2/4. Live padded tiles, A work, barriers and
per-output K16 order stay unchanged. Nix10.1 raw K2560/N640:2.979682→2.940020ms
(dead B) or2.948984ms (combined); K640/N2560:3.198049→3.158803/3.164221ms.
Saved1172-slot controls are reused; old832 control labels in logs are not the
comparison denominator. The extra K simplification has no established gain.
Cold/hot VGPR105/149→104/149→103/148; LDS24/32KiB unchanged, no private/spills.

Complete shared-side512/2048 operator spans8.59501/14.6878ms versus saved
8.55525/14.7609ms (+.46/−.50%). Synthetic output/guards and144 separate
boundary/exponent refactor comparisons pass. Small historical differences
remain advisory; no new model, baseline timing or profile. Preserve the narrow
K5 raw-half candidate for a later combined qualification; do not repeatedly
measure it or claim the original33.4% logical staging reduction as time saved.
Trellis ISA already uses shared bases/immediate offsets/pointer induction;
there is no concrete additional address-hoist candidate.
[All outcomes, artifacts and limits](../plan/exl3/F3-MOE-STAGING-SCREENS.json).

### HC MoE-combine operand loads, November 2, 2026

Selected in source, pending combined model qualification: group four adjacent
raw FP16, SVH FP16 and shared FP32 loads before arithmetic pins. The original
compiled restore order needs an explicit intermediate pin; the first unpinned
prototype failed exactness before timing and is preserved. Fixed Nix10.1
synthetic HC combine2048/hidden2560/used10:2.02531→1.84565ms (−8.87%);
after256MiB write flush2.09518→1.90804ms (−8.93%). Extra DPP transport
1.84522/1.91000ms adds no useful gain and is not selected. Prototype96VGPR,
zero private/spills throughout; no additional workspace.

Both actual production TUs compile and match all six saved fused-output
hashes on19×1,19×16,2048×10 cases, including following norm/mix outputs.
Independent Sylvester oracle4 passes with irregular scales/full5/tail1;
prototype standalone parity24 also passes. Guards/finite/poison included.
No baseline replay, model run or profile for the production check. These are
operator timings, not a whole-prefill speedup; qualified1429.13tok/s remains.
HcMix gate loads are already grouped in ISA; do not repeat that lead.
[Failures, all candidates, hashes and receipts](../plan/exl3/F3-HC-LOAD-SCREENS.json).

### Dense X3 K-tail audit (2026-11-02)

Audited the actual qualified shared5 K7 grouped256² ISA, reusing the saved
profile. Trellis pointer induction/immediate offsets and grouped LDS reads
with descending WMMA waits are already present;188VGPR,64KiBLDS,zero spills.
One distinct removal remained: the B K-tail branch despite validated k%128.
A narrow K7/Batching==1 grouped-descriptor mode/256²/kStore prototype removes it, retaining all row bounds
and per-output K16 order. All six saved-output/finite/guard checks pass.

Saved-control timings: out1.639585→1.637545ms, Z1.819499→1.799713,
sharedDown.215308→.212555, QKV2.854322→3.025093, Q3.709895→3.723071,
group2KV.352216→.364407. Candidate187VGPR/no spills. Not selected: mixed
historical evidence with the largest QKV worse, no aggregate win justifying
another specialization. These are advisory operator timings, not a confirmed
model regression. No baseline/profile/model rerun for this screen. Preserve
prototype and failed pre-GPU manifest check; move to combined HC/MoE batch.
[All numbers, ISA sites and receipts](../plan/exl3/F3-DENSE-KTAIL-SCREEN.json).

### HC/MoE combined qualification, November 2, 2026

HC operand-load grouping and K5 Mode3 dead-subtile/K-tail staging are now
qualified together on pinned Nix ROCm10.1. Dense K7 K-tail removal is not
selected. Current image: `ai-tools/gufo-eval:f3-prefill6-f15434b2-rocm101`,
as-built `1dbee4d6-prefill6-f15434b2`.

| Case | Saved shared5 tok/s | Qualified tok/s | Change |
|---|---:|---:|---:|
| pp512 | 1068.58 ±3.09 | 1078.15 ±.40 | +.90% |
| pp2048 | 1429.13 ±1.13 | 1435.59 ±.07 | +.45% |
| pp8192 | 1460.97 ±.73 | 1466.18 ±1.12 | +.36% |
| pp2048 at d32768 | 1301.99 ±1.41 | 1311.98 ±1.03 | +.77% |

One fresh32760-record/64-head-replay FP16-KV collector is byte-exact to the
saved Nix baseline; affected full-FFN1025/shared-state and expert/final-tail
checks pass. Performance uses k4v4/temp0/seed314159/r2. The initial pp2048
1342.22±103.86 was inconclusive. A shortened-list replay inadvertently changed
context8193→4096 and is diagnostic only; the final d0 replay restored the
original512,2048,8192 list and8193 capacity. All attempts are retained; no
baseline, correctness or d32K rerun. Target2000 is still unmet.
[Qualification, runtime/code identities and all attempts](../plan/exl3/F3-PREFILL-BATCH6-QUALIFICATION.json).

## TG/MTP routed and HC optimization, November 2, 2026

Flash-Next EXL3 decode/verify exposed an LDS problem that a private-spill-only
audit missed: routed widths2/3/4/8 reserved40960B per one-wave block while
width1 reserved0. Native batch6TG32 profile attributed74.34% of MTPspan to
RoutedGemv (28drafts/22accepted). Static row indices removed LDS for all75
K3–K7/dot-mode variants with no new private/spills.40-byte route descriptor ×
compiler-max1024threads is the inferred promotion source; do not call it a
measured register spill. Keep valid-row guards, per-row K order and exact
scaled-down fallback. The actualshape frozen-module, independent oracle,
partial-row poison and sentinel checks pass.

HC-up K320 did not reach the existing K%512 vector-load path. Its retained
stableF16 path loads four valid weights/activations at a time, skips whole absent
steps and keeps the original4step FMA chain. Cold52MiB component time dropped
11.37% atR1 and43.63% atR8; row1–8 frozen bytes/orderedFMA oracle and signedzero/
subnormal8/9boundaries pass. This is distinct from the rejected prefill HC
prefetch/tiling mechanisms. The grouped>8 path is unchanged.

Pinned Nix ROCm10.1, EXL3 5.05, C1 TG128, KVarN k4v4, temperature0,
seed314159, r2, depth0/32768 and common context capacity32897. Frozen
batch6 was measured once per mode and saved as the current control; historical
T1/T2 had different source/capacity34817 and are not reused as matched controls.
Both repetitions retain identical output hashes and draft/acceptance/lookup
counts before and after; candidate AR and MTP outputs also agree.

| Mode | Depth | Frozen batch6 tok/s | Candidate tok/s | Change |
|---|---:|---:|---:|---:|
| AR | 0 | 24.95 ±0.06 | 25.24 ±0.04 | +1.16% |
| AR | 32768 | 24.30 ±0.02 | 24.51 ±0.03 | +0.86% |
| MTP + lookup | 0 | 17.05 ±0.58 | 46.70 ±3.90 | +173.90% |
| MTP + lookup | 32768 | 22.23 ±0.01 | 79.31 ±0.94 | +256.77% |

Shallow MTP has material spread (46.70±3.90, only2repetitions): the large
improvement is clear, but neither precise steady-state speed nor the variance
cause is established. Deep MTP is the complete flow on the fixed repetitive
history:111accepted includes101lookup tokens and10neural drafts,17cycles.
It is not pure neural-MTP throughput or evidence that depth itself speeds decoding.
C2/4/6/8 correctness passed; their serving throughput was not measured here.

Complete sampling/batch/rejection/lookup/snapshot/deep-state checks pass.
Draft policy/costtables and precision are unchanged.75of77codeobjects remain
identical; only FlashNext general kernels and routed-MoE TUs changed.
[Qualification](../plan/exl3/F3-TGMTP-QUALIFICATION.json),
[screens](../plan/exl3/F3-TGMTP-SCREENS.json),
[flow inventory](../plan/exl3/F3-TGMTP-FLOW.json).

## Kyojin target comparison, November 2, 2026

The qualified TG/MTP image completed the frozen K17/D8 serving suites in one
window (40.1s ready,95.6s K,175.9s D);25 measured requests and GPU health pass.
No rebuild, baseline rerun or new numerical qualification. All17 historical K
completion hashes agree; that is output continuity, not an independent oracle.

C1, context49152, EXL3 5.05, pinned Nix ROCm10.1, KVarN k4v4, MTP+lookup,
thinking off, temperature0, up to256tokens, prompt cache off. Client rate uses
(completion_tokens−1)/(first-to-last useful SSE event span), matching pinned
Kyojin bench.py; server decode rate is retained separately. Medians below are
across distinct prompts, one pass each, not repeated-run confidence estimates.

| Category | Prompts | Gufo client tok/s | Gufo server tok/s | Published Kyojin target | Client gap |
|---|---:|---:|---:|---:|---:|
| Chat | 3 | 41.67 | 41.83 | 52.8 | −21.08% |
| Prose | 3 | 34.65 | 34.79 | 47.5 | −27.05% |
| Code | 3 | 57.25 | 57.48 | 73.2 | −21.79% |
| Multilingual | 4 | 38.94 | 39.09 | — | — |
| Copy/edit | 4 | 64.63 | 65.00 | — | — |

All three published category targets remain unmet. This is a cross-machine,
cross-pack reference: Kyojin had not been run locally at that stage
(the later local run is recorded below). Frozen source2132bd2 has
three card prompts/category, while today's public description says four.
The saved published table is not a matched performance control.

D labels are approximate: d8k actually contains7274–7509tokens; d32k contains
27158–39256tokens. Client mixed-task medians are40.55 and35.61tok/s, respectively.
Story subgroups are35.06/31.91; documentation subgroups47.56/43.26 (two each).
Published Kyojin8K/32K essay rates52.2/51.1 are different workloads and exact
depths. Do not call this a matched depth comparison. Server prefill medians
1293.51/1280.75 also belong to these documentation inputs and the serving path;
they do not replace the frozen fixed-pattern prefill benchmark.

This confirms that79.31tok/s on the earlier repetitive lookup fixture is not
a general-generation rate. Preserve current kernel improvements. Any next
optimization should first attribute the remaining cost on a real prose prompt
using this optimized ELF: dense/HC, draft+verify cost per committed token, and
PLE/host waits. The prior native profile predates verifier LDS removal and
cannot establish the new bottleneck; no new profile was run in this comparison.

Receipt `fn-tgmtp-kd-nix101`; per-request client/server rates, output hashes,
acceptance/lookup counts, runtime maps, frozen inputs and published-page SHA
are retained in the comparison report. Existing numerical qualification and
its NLL caveat remain unchanged. No Kyojin quality parity is claimed.

[Comparison](../plan/exl3/F3-TGMTP-KYOJIN-COMPARISON.json).


## Native EXL3 PLE transition, November 2, 2026

EXL3 n-gram rings are distinct from EXL3 matrix tiles: cyclic symbols, mul1
FP16 codebook, row scale and per-head bias, then a final FP16 embedding store.
There is no matrix Hadamard transform. A sliding16-bit state avoids repeatedly
reconstructing all preceding symbols. Keep compressed rows in the existing
bounded asynchronous cache; support single and split tensor ranges independently
of head boundaries. Current native5.05 K6 and Kyojin K4 boundary rows agree with
an independent bit-at-a-time oracle (51200elements); preserve both tests.

One native5.05 combined window:32760 quality records/64head replays plus
sampling/EOS, C2/4/6/8 AR/MTP/lookup,32K and cache/policy serialization pass.
Native PLE changes weights: old output hashes are not the quality oracle.
Saved-reference paired NLL improves by.007451, window95%CI[−.017370,−.001770];
KL256+.rest.026481→.025464; top-1−.0794pp unresolved. PP2048/8192 rates are
1423.76/1461.67tok/s (saved1435.59/1466.18); AR25.20/24.43 (saved25.24/24.51).
Short MTP40.09 vs46.70 reflects changed generated text, cycles40→54 and
accepted88/113→74/114. This is a material end-to-end loss, not established
loader/kernel slowdown. Same-output32K MTP78.53 vs79.31 keeps17cycles and
111accepted,101from lookup. Never generalize that repetitive rate to prose.
All ROCr/CLR paths match pinned Nix10.1; no GGUF mappings and healthPASS.

Native directory metadata can use an owned typed adapter without writing a
GGUF sidecar. Validate NFC/pretokenizer/added-token contracts, distinguish PLE
EOS from generation EOS, match native MTP to trunk shards, and bind persistent
identity to full payloads plus all metadata and source stamps. Preserve the
existing Nix volume/objects; only the completed export's new unused BuildKit
records were cleaned after packaging. [Evidence](../plan/exl3/F3-NATIVE-PACK.json).

## Native PLE host attribution, November 2, 2026

Prior GPU profiles began at EmbedKernel and omitted pre-embedding host work.
Ordinary embeddings run on GPU; native PLE reads/dequantizes on CPU, even on
compressed-cache hits, into coherent mapped memory consumed by the GPU.
A separate host-only diagnostic retains production graphs/kernels. Nonoverlap
PleFetch+WaitPle+PrefetchPle was6.54/1420.88ms forPP2048 and25.48/5558.23ms
forPP8192 (both0.46%); AR12888.31/5073.33ms (1.74%), MTP12853.73/3182.50ms
(1.69%). Prefix/warmup excluded as appropriate; nested read scopes not summed.
AR/MTP outputs match saved native64c0a14b, MTP54cycles/114drafts/74accepted.
Health/runtime/no-GGUF checks pass. No production change or baseline rerun.
These are instrumented host durations, partly overlapped with GPU work; logging
can hide waits. No uninstrumented-stall bound, worker/SSD bandwidth interference,
GPU mapped-quantizer attribution or cold/prose generalization is established.
Observed host cost is small here; profile those unresolved costs before a large
loader rewrite. [Evidence and scope](../plan/exl3/F3-PLE-HOST-PROFILE.md).

## Local Kyojin, November 2, 2026

Frozen source2132bd2 built once (458.03s CPU-only), then preserved with its
source-pinned ROCm10.1.0a20260822 wheel SDK/clang23. This is not the Gufo
Nix10.1 release/clang24 runtime; Torch reports HIP7.16.26332. Internal NVMe
Kyojin≈4bpw/K4 PLE, context16384, quant K4V4, HC sidecar and draft2s active.
One resident server, twelve requests, no saved Gufo baseline rerun.
Real-text PP2048/8192=1550.88/1386.46tok/s after one warm pass each; distinct
from Gufo fixed-pattern PP. Client AR prose/chat/code=31.30/31.33/31.28;
MTP=42.52/49.44/63.22tok/s, 256 output tokens per case. Startup172.05s excluded.

All requests completed and healthPASS, but paired greedy AR/MTP exact token
and text equality failed3/3, first differing zero-based token57/76/149.
Independent raw-request review confirms speculative is the only request
parameter difference; actual accepted/rejected drafts138/105,160/61,194/36.
Cause and general quality impact are not isolated; no repeat/determinism claim.
Retain as measured evidence, not an accepted production replacement. Historical
Gufo K-suite rates differ in pack/PLE/KV/runtime/capacity; do not call the raw
12–20% difference pure engine gain or current native parity.

User-authorized five-payload internal GGUF removal funded88.82GiB local copy,
including24.4GiB ngram; external originals/manifests remain. Streamed source
hashes and destination sizes/fsync checked, no redundant full reread. Existing
Gufo artifacts preserved. Infra432pass; no Gufo inference change.
[Receipt and limitations](../plan/exl3/F3-KYOJIN-LOCAL.md).

## Same-pack TG/MTP attribution, November 2, 2026

Same internal Kyojin≈4bpw/K4 PLE, C1/context16384/neutral greedy256tokens:
Gufo nativeb4ebe27c AR30.49/30.51/30.23, MTP42.55/46.94/62.66tok/s for
prose/chat/code. Saved Kyojin MTP42.52/49.44/63.22 reused. Gufo text equality
passes3/3; no rawoutput-ID claim. Effective HC/int8sidecar, drafthead, KVcodec
and runtime still differ; cross-engine text differs. Single-pass MTP gaps
+0.08/−5.06/−0.89% do not establish statistical/pure-engine/quality parity.
No inference change or model switch. First partial run inheritedpresence1.5,
was stopped and preserved; corrected Gufo explicitlyneutral, Kyojin not rerun.
Greedy temperature0 does not disable Gufo penalties: inspect effective defaults.

Current native5.05 prose64 afterwarm16 (postverifierLDSfix): ARdense35.44%,
routed20.39%,otherstable small14.82%,HCup7.20%; MTP routed27.43%,dense22.21%,
otherstable9.88%,HCup5.12%,draftsubsethead2.77% of decodekernelspan.
AR/MTP spans2579.785/2064.302ms, kernelactive90.13/78.24%; bothtextmatch and
healthPASS. MTP28cycles/63drafts/35accepted.19,141directlaunches/4mallocs observed;
possiblefirst-use/capture is not separated from expected eager work, and
nonkernelspan is not proven removable CPU overhead.
Range selection by Embed sequence excludeswarm/prefill: AR18,7; MTP18,17,1,7,6,
then1-row decodeanchor. No unverifiedPython/roctracer clock join; serverdecode
agrees within1ms. Replaynames retained, missinggridnotinvented. Tracer reused
unchanged and preserved under ai.tools test/.../native_decode_trace.

Prioritize distinct denseK7 load/reuse and routedK5R3/4 gate/up candidates;
reuse rejected tile/prefetch ledger, preserve arithmetic and qualify realragged
routes. Do not start with2.77% drafthead approximation or previouslyfailed HC
int8. C1greedy uses confidence afterinitialcycle; oldcostcurves alone do not
establish a retuning opportunity. Batch small screens and final qualification.
[All evidence and limitations](../plan/exl3/F3-KYOJIN-SAME-PACK.md).

## Exact live-row routed verification, November 2, 2026

Repeated ragged-row checks inside every K step were a material K5 R3/R4 cost.
Selecting an exact live-row body once per tile retained dot/reduction/scaling
order and improved cold gate/up21.38/32.03%, scaled down9.60/13.90%.
Keep negative-exponent FP32 down and generic other rates/widths/full-precision.
The retained path uses gate VGPR40/46, down54/58, zero private/LDS. Occupancy
alone is not decisive: flat→global reduced gate VGPR38/40 to32/34 but R3/R4
changed only+0.32/+0.33%; aligned-K additionally changed−0.44/−0.35%.
Those casts/assumes were excluded. Prior prefill global-address-space neutrality
was reused as context, not confused with this decode-path evaluation.
Dense K7 TN2 already issues six raw loads before dot; no redundant experiment.

Three small screens pass112 cases each, reusing one saved cold control series;
64experts/62.5MiB avoids MALL-only timings. Final independent operators include
K3..K7, partial/poison rows, zero/overcapacity/count guards and distinct z2 banks.
One combined native5.05 window passes32760 identical collector rows, sampling,
batches,32K/cache, and identical AR/MTP output hashes/work counts. MTP d0
40.09→40.85tok/s (+1.90%); d32K78.53→79.48 is lookup-heavy and overlaps saved
baseline spread. PP1426.88/1460.75 and AR25.20/24.49 are effectively unchanged.
Do not equate component gains with whole-model gains. All baseline/prototype
objects remain; no baseline/model remeasurement or extra pipeline profile.
[Complete evidence and all measured shapes](../plan/exl3/F3-TGMTP-ROWS.md).

November 7 generalization removes the K5-only R3/R4 gate, including K3.5/K8,
and qualifies it together with all-rate Mode3 dead-B staging/K-tail elimination
and stable grouped SmallGemm vector loads. Format-specific unpack, FMA/rounding,
alignment, negative-exponent FP32 fallback and unstable grouped path remain
unchanged. Nix10.1 incremental build, CPU format/loader and four focused GPU
suites pass; representative boundaries include K2560/N128 Mode3 and grouped
input misalignment, not exhaustive shapes or misaligned weights. Compiler
resource comparison: 0 worse, 5 better, 3509 unchanged, 4 unmatched; scratch
increase0. This resource result alone is not a speed measurement.

Saved c03 32,760-position prefill payload is byte-identical; repeated AR/MTP
emitted hashes and draft work match stored controls. CLI two-repeat observations:
PP-off2048/8192 1460.97/1501.37→1457.86/1495.46tok/s; AR d0/d32K
27.33/26.72→27.32/26.76; PP-MTP1370.97/1403.43→1385.03/1416.51;
MTP d0/d32K50.70/87.30→52.27/87.69. These are small stored-baseline deltas,
not isolated per-change or confirmed universal gains; d32K is lookup-heavy.
HTTP greedy/sampled512 C1/C2, off/MTP pass; greedy outputs match, with no
individual prefill/decode slowdown crossing both5% and10ms. Sampled-output
identity is not asserted. C2 MTP draft/accept counts differ from the
stored baseline despite identical greedy output; its timing difference cannot
be attributed solely to kernel work. Group-wall aggregate MTP C1 greedy/sample
41.70/41.97→43.29/43.80 and C2 45.55/50.08→45.90/52.48tok/s.

Keep these default source paths, preserving original server deployment after
the approved test window. All baseline/model data and incremental build
artifacts remain. Attempt001 lacked render-device permission before computation;
attempt002's completed TG-MTP hash was interrupted by stdout headers, repaired
only for CPU parsing and independently checked against all four baseline
hashes/counters. Raw evidence stays intact; attempt003 ran only the two unrun
HTTP cases. Evidence: `build/f3-general-opts/{runtime,validation-002,validation-003}`,
`QUALIFICATION.json`, `INDEPENDENT-FINAL-REVIEW.json` and
`validation-002/tg-mtp/PARSE-REPAIR.json`.

## Prefill live-subtile compiler screen, November 2, 2026

Do not directly clone the successful routed GEMV live-row loops into X3
prefill. Existing dead-B/WMMA skipping already removes inactive subtiles;
selecting exact live-subtile loops once adds substantial register pressure.
Production BN64/128 uses103/148VGPR, zero private/spills. Four-way loop clones
use231/256VGPR and BN128 private244B/spill108; full-body scopes still use
239/256 and private136B/spill68. A final two-way full-subtile/generic refinement
uses191/256 and private148B/spill71. LDS remains24/32KiB throughout.
All three compile with production Nix10.1/O2 and are rejected before GPU
execution. Compiler evidence does not establish actual latency regression or
the precise allocator cause. No new correctness/speed claim; existing
prefill/TG/MTP implementation and saved baselines remain selected.
[Scope, patches and resources](../plan/exl3/F3-PREFILL-LIVE-TILES.md).

## Flash-Next physical prefix references, November 4, 2026

The pool below originally shared capacity, with exclusive per-session blocks.
It now shares physical immutable prefix blocks as Gufo27B does. Retain only
blocks complete in target, draft and pooled-indexer planes: the sparse
transition writes an old pooled backlog, and capturing a writable frontier
after admission otherwise creates unreserved COW debt. KVarN append group0
maps to physical group1, so use one-group chunks; pooled keys use64-row chunks
for256-token blocks. COW copies every plane/layer and keeps the device table
pointer stable. Recurrent/raw/sink/tail state stays private.

For admission, intersect retained token ranges across planes and align to
whole blocks; do not lose later GPU capacity behind early RAM chunks. Reclaim
only preserves the actually selected context-aware checkpoint/live frontier.
Protecting every token-prefix match retained duplicate cold histories and
rejected a valid image branch with a misleading context-length400. a03 fixes
this; failures and the targeted replay remain preserved.

88 GPU cases (K4/K6/FP16 AR+MTP) and exact32760 saved-K6 numerical records pass;
new reservations followed by capture incur zero COW debt. CPU tests include
mixed row units, disjoint spans, cross-plane intersection and image-aware
reclaim. C4 pressure/resume,13 branch/image requests, disk4867+0 prefill and
Pi2tasks/6tools pass. PP2048/8192=1372.59/1398.58, AR d0/d32K=24.30/23.59,
MTP=33.90/76.67tok/s vs saved K6=1374.42/1403.11,24.33/23.58,35.66/76.06.
Desktop-active timing is advisory; no baseline rerun or full256K replay.
Pool backing size is unchanged; sharing reduces duplicate block occupancy.
[Full evidence and limits](../plan/exl3/F3-KV-REFERENCES.md).

## Flash-Next shared K4V4 pool, November 2, 2026

Retained: existing KvBlockPool shared across sessions, with exclusive blocks,
stable tables, KVarN group-offset-aware transfers and canonical sink/tail state.
Include the pooled-index guard in mapped/admission counts: two256K contexts
need2050 blocks (4.62GiB). Host-backed checkpoints retain no GPU blocks; short
restore must return excess mappings. Coalesce only physically and logically
adjacent copies to preserve private-session transfer cost. Unsupported shared
or KVarN in-pass checkpoints must return zero bytes/reject throughout the stack.

Shared GPU fixtures92.04s and final103.511s pass, including fragmented mappings
and saturated final-token admission. Numerical32760 rows/101424960 bytes exact
against saved baseline. Remaining sampling/batches/32K/cache pass; initial
aggregate mixed EOS assertion lacked clause diagnostics, focused diagnostic
and serving sampling pass unchanged (not a proven noise diagnosis). An initial
Python module-path failure resumed only unstarted checks; both failures retained.
PP2048/8192 1426.88/1460.75→1425.09/1462.00, AR25.20/24.49→25.20/24.47,
MTP40.85/79.48→40.92/79.27tok/s; hashes exact, maximum difference0.264%.
No baseline rerun;32K MTP is lookup-heavy. No full two-times-256K inference.
Actual C4 HTTP,12K long context, process-restart disk restore12352+7 tokens
(0.486s, same output), Pi1.4.0 three tasks/nine tools pass. Cold/warm identity
startup85.34/35.15s. Current image4757e04f, Nix10.1 incremental objects retained.
[All results and provenance](../plan/exl3/F3-SHARED-KV.md).


### 2026-11-04: Gufo FP16/native256K/C2 serving memory

User-requested precision switch on the already qualified Flash-Next KV-sharing
image `fn-kv-references-a03-rocm101`; no kernel/build/baseline rerun. The shared
FP16 pool is14,408,908,800B=13.419GiB (2,050×7,028,736B),+7.296GiB over K6V6.
Vision, medium and MTP7 remain enabled. Two concurrent2,431/9,011-token prompts
with96-token output caps and one224px image (90tokens, correct red/blue answer)
completed; host MemAvailable minimum10.687GiB over70 half-second samples and
after checks10.725GiB. Not a full256K worst-case test or speed/quality comparison.
KV backing is allocated, but scratch/graphs/images can grow. Keep the4GiB host
shutdown floor visible; cgroup memory excludes most UMA GPU allocations.
Gufo remains ready, run d46df13dd50a; Kyojin stopped,default alias unchanged.
Evidence: `/home/m2on/.local/share/gufo/fp16-c2-256k-20261104/RESULT.json`.

### 2026-11-05: EXL3 c03 long calibration preflight

Pinned Kyojin converter/ROCm10.1 image79e3e0b7, c03 SHA36c8eddb,
BF16 source de4b8e4d, completed K8 ngram reused. Independent8192/512-token
rows pass embedding/HC/GDN/PLE/first full-attention. All-expert capture output
is discarded; normal routing advances state. Five B1 packed operators pass
the existing5% relative activation-error limit: GDNqkvK5 0.5451%,
first expert-downK4 4.3159%, attentionqK6 0.2406%, kK8 0.1947%,
full-attention-block expert-downK4 4.8279%. FP32 output shortcut disabled;
finite H/output/scales, nonnegative proxy, no uncalibrated fallback.
Pilot169.1s, peak GPU allocated15.829GiB, host available minimum89.87GiB,
returnedKFD0. This is a two-row operator/long-shape screen, not full-model
quality or peak-memory qualification. Existing512 baseline not repeated.
Main retains8GiB host reserve/100GiB container and module checkpoints.
Evidence: `ai.tools/tmp/test-results/flash-next-quant-20261104/long-row-pilot/`;
[conversion/watch](../plan/exl3/F3-CONVERSION-WATCH.md).

### 2026-11-05: c03 calibration state residency repair

Full calibration checkpoint is81,899,571,619B (~76.28GiB). The converter's
eager CPU reader retained all rows as anonymous RAM; attempt003 stopped at
first block loading when MemAvailable fell below8GiB. No computation failure
was recorded; c03 idx2 checkpoint and K8 ngram retained, KFD returned0.
Use a one-row disk-backed cache and stream unchanged tensor values into the
same checkpoint format. CPU seven contracts pass including official reader,
BF16 byte-view flattening, rotation/resume and unchanged source. Metadata-only
open of the real76GiB checkpoint adds2MiB RSS. GPU staging on actual3217/8192
rows preserves exact values, including GPU-only head eviction;3.13s,
peak320MiB, host available minimum108.34GiB, KFD0. Independent review passed.
This fixes storage residency; full conversion peak/time/quality remain to be
measured. Existing long-row/512 baseline reused, runtime/kernels unchanged.
Evidence: `ai.tools/tmp/test-results/flash-next-quant-20261104/model-job-c03/repair-001-eager-state/`.

### 2026-11-05: completed row checkpoint recovery without recomputation

Attempt004 completed layer0 quantization and all2309 forward rows in3576.52s;
RFN0.017948/cos0.000078/SQNR40.802051, minimum host available73.95GiB.
The3600s module guard interrupted redundant76GiB aggregate checkpoint writing.
Final per-row outputs survived. Verified exact original calibration token IDs,
frozen input/recipe hashes, every row's extent/shape/dtype/attempt provenance,
and all1542 packed targets (6184 total saved tensor entries), then recovered
idx3 using immutable hardlinks in1.229s. No GPU work repeated. CPU7contracts,
actual recovered row0/364/2308 exact reads, infra432/18.38s and independent
review pass. Reader accepts old aggregate and new row manifests; private input
links survive old checkpoint deletion, output replace preserves snapshots.
Fixed stop/kill auto-remove race so cleanup errors no longer hide initial cause.
Failed scratch is retained until recovery; successful scratch links are retired.
Full model quality/performance still pending; time/memory/disk guards unchanged.
Evidence: `ai.tools/tmp/test-results/flash-next-quant-20261104/model-job-c03/repair-002-checkpoint/`.

### 2026-11-05: Docker missing-object diagnostic case

Attempt005 completed PLE in345.05s and committed idx4; supervisor stopped only
because the installed Docker CLI emits lowercase `error: no such object` after
`--rm`. Normalize diagnostic case while retaining ownership/reap/KFD checks and
rejecting unrelated daemon/permission/socket errors. CPU11 regression cases,
including the real installed CLI, pass; infra432/18.60s and independent review
pass. Confirmed all2309 checkpoint row headers/shapes/dtypes and unchanged input
IDs; KFD0. Resume idx4 without repeating model/GPU tests or completed work.
Evidence: `ai.tools/tmp/test-results/flash-next-quant-20261104/model-job-c03/repair-003-cleanup-case/`.

### 2026-11-05: CPU checkpoint commit after bounded GPU window

Attempt006 completed layer1 and2309 forward rows in3592.60s, RFN0.008642,
cos0.000148/SQNR52.849043, minimum host available74.108GiB. GPU3600s guard
expired just before checkpoint commit. Reaped owned container/KFD0, then
validated completed TransformerBlock outputs and committed unchanged rows in
2.337s on CPU, without repeating compute. Use a separate120s CPU window with
8GiB RAM/100GiB disk floors, only for explicit GPU window timeout and complete
new rows/IDs/recipe/packed tensors/feedback. Other failures and head are rejected.
Durable commit journal and incomplete-stage admission guard protect interrupted
rename. CPU10 checks and infra432/18.82s pass; independent review pass.
Retire redundant links only after commit; actual disk free292.34GiB. Full model
quality/performance still pending. A converter commit during stop grace can
still require manual audit; it is never silently treated as success.
Evidence: `ai.tools/tmp/test-results/flash-next-quant-20261104/model-job-c03/module-006/CPU-COMMIT.json`;
`repair-004-cpu-commit/`. GPU window and numerical settings unchanged.


### 2026-11-05: Elide discarded expert-down outputs during EXL3 calibration

The pinned converter computes all512 experts while collecting Hessians, then
throws away that forward output and separately computes normal routed reference
rows/state. Keep Linear.forward padding/capture_H intact and skip only routed
down LinearFP16.inner GEMMs inside this explicitly discarded terminal-MoE scope.
Shared experts, normal routing, FP32 arithmetic, calibration, recipe and N2
quantization remain unchanged. Require the audited terminal structure and pinned
unwrapped cm.main caller/source; the torch.inference_mode decorator must be
included in the source identity. An initial guard mismatch stopped before any
baseline computation; the CPU fixture now tests the real decorator too.

One retained baseline per GDN/full-attention block at33/512/8192 rows: all516 H
keys/tensors/counters, preserved inputs and normal outputs match byte-for-byte.
At512 rows, capture1.37779/1.21895s becomes0.52963/0.53058s; at8192,
33.58238/35.01159s becomes24.19120/23.78136s. The206.45s combined GPU screen
returns KFD0. First-shape/library/allocator warmup can favor later candidates;
these are provisional capture timings, not full-conversion or inference gains.
Save baseline H and outputs; do not repeat them for unchanged comparisons.

Foreach H/count additions also match exactly but add no material8K gain
(24.15114/23.78973s), with attention512 at0.54635s. Preserve experiment evidence;
do not deploy this extra machinery. Keep the original flush. Use host-only60s
progress instead of EXL3_QUANT_PROFILE=1, which synchronizes quant-worker stages.

CPU scope15/foreach18/progress9/guardian8 checks pass. The existing first
attention block finished in4886.74s, RFN0.008242, and committed idx7: a bounded
supervisor-only handoff preserved its GPU calculation past the old1h guard.
The new module limit is2h with unchanged RAM8GiB/disk100GiB running floors.
Full model accuracy/performance remain the final combined qualification.
Evidence: `ai.tools/tmp/test-results/flash-next-quant-20261104/capture-speed/`,
`capture-optimization-cpu-active-runtime/`, and `model-job-c03/repair-005-capture-speed/`.
[Conversion watch and rollout](../plan/exl3/F3-CONVERSION-WATCH.md).

Production activation is confirmed on resumed module009: the first host progress
record reports4 capture scopes and2048 skipped down calls (9,900,032 expert rows).
`capture-speed/ACTIVATION.json` distinguishes this from the explicit probe scope.
Post-integration infra432 checks pass in19.17s; full conversion remains running.

09:20 qualification correction: module009 capture projects about63min, whereas
old complete GDN blocks took58.6–59.9min. This is a suspected real regression,
not an established ETA-only effect. Earlier probe omitted production inference_mode,
used resident inputs and original-first timing without balanced warmup. These
are confirmed measurement limitations, not yet a causal diagnosis. A bounded
matched diagnostic is reserved at the next natural checkpoint boundary, with
saved initial H, real512/8192/3217 rows and separate read/H2D/forward/flush timing.
See `capture-regression/` and the conversion watch; no full baseline repeat.

09:45 matched diagnostic completed in136.11s with exact H/counters/normal outputs
and clean GPU return. Resident capture means512:0.55880→0.42625s,
8192:15.00277→13.16548s,3217:4.94569→4.15319s. Original-first warmup inflated
earlier gains; new measurements still favor skip-down, but do not explain the
4424.52s complete module009 (capture3732.49s). I/O is excluded from these gains;
512 pipeline also contains a read-time outlier. Overall regression remains
unresolved. Independent review retained code unchanged and resumed idx8;
`capture-regression/REVIEW.json` records the decision.

Actual-loop follow-up: pinned cm.main64-row ABBA with automatic scope, DiskRows,
first five normal references and no per-stage synchronization gives original
66.7137/67.1036s versus skip54.9655/54.8417s (17.9% less time). All516H/counters,
references/inputs/RNG are exact. Private byte-identical copies preserve protected
checkpoint mounts; page caches are warm. Cleanup errors0/KFD0, independent review
keeps skip-down and resumes idx9. This qualifies the sampled capture loop, not
complete conversion or the cause of earlier whole-block slowdown. Evidence:
`capture-pipeline/{SUMMARY,STATE,REVIEW}.json`.

Direct FP32 pack follow-up: keep the original flush. Actual64-row capture ABBA
with deployed skip-down gives55.2241/55.2856s original versus55.0169/54.9570s
pack (0.485% mean reduction, below5% criterion). All516H/counters/references/
inputs/RNG match the saved original control; no whole baseline rerun. Separate
3217/7793/512-row flush event intervals sum to729.09ms preprocessing,
6091.85ms BMM and50.97ms H/count accumulation. These three samples are not a
whole-capture cost fraction and may include host submission gaps. No claim
about the historical whole-block slowdown follows. Cleanup errors0/KFD0;
independent review resumed idx10 unchanged. `capture-pack/{SUMMARY,STATE,REVIEW}.json`.

November5 follow-up profiling did not qualify numerically: the64-row loop with
all-row events and a row4 torch.profiler trace took75.917s, then failed saved-control
byte equality at `capture_H.block.attn.input.H`. This is the first comparison
failure, not proof of a unique bad tensor or first causal layer. No magnitude was
retained by that initial harness; raw numeric preservation was corrected before
retry. Do not interpret the profiled wall time as a regression or deploy a changed
arithmetic contract. Original checkpoint idx10 is untouched; cleanup errors0/KFD0.
Evidence: `capture-profile-quarter-attempt-001/`. Retry isolates events-only first,
then trace only if exact, with provisional profile/difference/numeric persistence.
Quarter calibration is an independent two-projection screen using retained full2M
packed weights; its first attempt had not started when this profile failed.

Quarter c03 operator screen (577rows/503322tokens) completed: prefix capture401.48s,
full diagnostic488.97s, K6 GDNout/K4expert0 down retained the full2M rotation signs.
Same-c03 mean document error increases1.181%/0.556% (biased toward full calibration).
Fresh fixed8-input screen reusing both packs took37.31s: mean document error ratios
1.007627/0.994610, maxima1.013991/1.001267; neither flags the predeclared advisory
1.02mean/1.05max. Includes synthetic Korean4K/local C++8K and samples max256positions;
some sampled p99 errors rise. This supports a subset-mapping check, not all-layer
quality equivalence or inference speed. Prefix401s is65H/64experts, not full512expert
conversion timing. Both cleanups have0errors/0KFD. Evidence: `capture-profile-quarter/`
and `capture-quarter-holdout/`, each REVIEW.json. Full2309-row state remains intact.

Same-shape Hessian chunk128 rejected for deployment: six real64-row historical
GDN capture cases match all516 saved H/counts/references/IDs/RNG exactly. Warmed
ABBA chunk64 54.912/55.085s vs chunk128 55.474/55.069s; mean +0.495% time is
inconclusive, not a confirmed regression or improvement. Peak allocated GPU
memory rises15.23→19.10GiB (+3.87GiB). First candidate warmup77.44s is excluded;
its initialization cause was not identified. Keep chunk64, full2M calibration,
FP32 and idx10 stopped. Cleanup0/KFD0; production frozen and protected inputs
unchanged. Independent review: `capture-chunk/{SUMMARY,STATE,REVIEW}.json` under
`ai.tools/tmp/test-results/flash-next-quant-20261104/`. Do not repeat this unchanged
comparison or extrapolate its capture-only timing to full conversion.

Gram operator follow-up (`capture-gram/`,97.35s,cleanup0/KFD0): direct hipBLASLt
FP32 heuristics2/3 improve resident64×512×640 Gram ~2.1–2.2x,1×512×6144
~3.2–3.5x,64×8192×640 ~2.4–3.7x. Relative differences vs Torch FP32 BMM are
4.05e-7/4.07e-7/8.25e-7; FP64 samples(first2batches,8columns) improve on expert
shapes and change dense4.84e-7→4.91e-7. Heuristic1 is byteexact and ~7% faster
on8K only. Original-FP16-input Lt reaches7.2/9.9/17.5x, but8K sample error grows
9.61e-7→~1.71e-5. Custom WMMA achieves3.2/6.0/6.1x, slower than fastest half Lt;
its dyadic analytic mismatch is preserved, not accepted by relaxing exact checks.
No FP32 input was cast down. All outputs FP32; accumulation order changes matter.
Operator-only timing excludes packing/masking/H addition. Torch preference is Lt,
but profiler exported no GPU kernel events; actual kernel attribution remains
unproven. Prefer matched flush/quantization quality checks before any deployment;
production unchanged,full2M maintained,idx10 stopped. Preserve real-inputs.pt and
per-case control.pt; no recapture or whole baseline rerun. REVIEW.json records
independent assessment; first analytic attempt retained separately.

Complete-flush and matched-quantization follow-up (`capture-lt/`,37.37s): reuse
saved64-expert512/8192 and dense512 inputs; keep stack/cast/finite mask/H-add/count
ordering. Original flush0.73544s,FP32Lt3 0.36042s(~2.04x),original-halfLt1
0.12585s(~5.84x). Original agrees byte-exact with saved Gram controls; candidate
repeats/counts are exact. Same small H samples, weights, RNG and worker stream
for two projections: mean document error increase dense+0.0435%/+0.1032%,
expert+0.0154%/+0.0140%(FP32/half); worst sampled p99 rises~1.92%. Rotations
match/no fallback. Dense512 is rank-deficient before damping, so not full-model
quality qualification. Cleanup0/KFD0 and production unchanged. Next real-loop
qualification must bound Lt plan/workspace caching: current per-shape32MiB plans
cannot accumulate without a limit across all ragged calibration lengths.
Independent assessment in REVIEW.json; retain H/packed/evaluation inputs.

Bounded Lt actual-loop follow-up failed before matched timing: first FP32Lt
warmup72.64s completed64 rows but normal refs differed from saved control.
CPU retained-output audit: all5 finite, relativeL2 7.85e-7..8.42e-6;516H counters/
metadata, input hashes, IDs, preserves and RNG match. No causal claim: H is not
read by the normal reference forward. No valid A/B/C comparison/cache-peak result
was recorded; do not use warmup as a speed result. Cleanup0/KFD0, mainidx10 stopped,
25% deferred, candidate uninstalled. Retain failed-numeric.pt and REVIEW.json in
capture-lt-pipeline/. Focused row0 same-process normal repeats before/after original
flush, Lt flush and plan close will separate reproducibility from intervention;
no full64 baseline replay or completed-checkpoint change.

Focused Lt failure isolation retained original baseline: warmed row0 (32.00s),
cold Lt-first row0 with original host wrapper/entry sync (21.05s), and first5-row
prefix (60.70s) all match saved normal references byte-exact. These elapsed spans
include CPU copies/hash/validation and are not performance timings. Prefix
references remain unchanged after subsequent rows;516H counters each19848,
inputs/preserves/IDs/RNG exact. Native LRU exercised12evictions, peak8plans,
explicit workspace cap256MiB; Torch peak15.20GiB excludes native workspace and
library reservations. All diagnostics cleanup0/KFD0. Evidence capture-lt-ref/,
capture-lt-cold/,capture-lt-prefix/ under the same retained November4 results.
This does not reproduce or explain the first64-row failure. No arithmetic guard
was relaxed, no production code installed, no full baseline repeated. Keep
FP32Lt3 as the next qualification candidate; FP16Lt1 remains separately unqualified
for the complete loop. Any next candidate-only64-row diagnostic should preserve
normal refs both immediately and at loop end to distinguish later mutation from
cross-process reproducibility; it has not started. Mainidx10/25% hold remain.

User-authorized numerical-quality evaluation separates arithmetic differences
from data integrity (`capture-lt-evaluate/`). Original saved baseline was reused;
observer+half warmup excluded, candidate-only B/C/C/B capture means FP32Lt3
53.516s, original-halfLt1 44.853s (16.19% less wall time between candidates).
All6 cases preserve saved normal references exactly, candidate repeats H/refs
byte-exact, observer refs unchanged at loop end. H differences vs saved original
max relativeL2 FP32 5.39e-8 / half 2.03e-5.12evictions,peak8plans,explicit256MiB
workspace; Torch peak15.23GiB excludes native reservations. Cleanup0/KFD0;
381.90s diagnostic. Historical original~55s is not a concurrent matched control.
The first separate64-row mismatch remains unexplained and its failure receipt
is retained. Next quality comparison uses saved64-row H (50056tokens) to quantize
two layer6 projections identically, fixed8 documents and actualBF16 source weight
projection teacher with sampledCPUFP64. No calibration reduction/main resume.

Layer6 paired quality follow-up (`capture-lt-quality/`,177.35s,cleanup0/KFD0):
reuse original/new64-row H (50056tokens), quantize GDNoutK6 and expert0downK4
with identical RNG/worker arithmetic. Fixed8 reused docs traverse original loader
weights plus fixedK8 ngram; the projection teacher uses actual sourceBF16 weights
in FP32 and sampledCPUFP64 cross-check. BF16→loadedFP16 max2.98e-8 recorded.
Mean document error ratios vs original-H pack: dense FP32 .99744961 (-.2550%),
original-half 1.00001649 (+.00165%); expert both1.0 with ALL packed bytes identical.
Half dense worst document+0.615%, worst sampled-p99+1.882%; not all metrics improve.
All6quantizations CPUFP32 Cholesky first attempt,rotation signs same,no fallback.
Prefer original-half Lt for next integration qualification given16.19% less
capture wall time than FP32Lt, while retaining FP32 inputs as FP32. No production
installation/main resume: full-attention/unseen shape/heuristic support and
explicit unsupported handling remain before integration. Two projections/fixed8
docs are not whole-model quality equivalence. Original initial64 mismatch remains
unexplained; later6candidate captures are repeatable/exact in normal references.
Preserve H/prefix states/evaluation activations/BF16 teachers/packs for reuse.


Lt integration qualification (`capture-lt-support/`,50.09s,cleanup0/KFD0):
152 calibrated/diagnostic lengths ×5 batch/dimension pairs ×2 input precisions
support the selected algorithms (1520 creations). Actual attention33/512/8192
keeps saved normal refs exact, H/ref repeats byte-exact,516 counters8737;
max H relativeL2 vs original2.45e-5.8plan/256MiB explicit bound,16evictions.
This is selected heuristic support, not numerical execution of all1520 shapes.

User-requested Lt64 versus128 actual64-row ABBA (`capture-lt-chunk/`):
64first65.7576s,12844.7009/44.6832s,64last44.6663s. All four cases match
saved qualified Lt64 H/refs/counts/IDs/RNG byte-exact; B128304 new support
combinations pass; cleanup0/KFD0. First-A discrepancy is cold/order-associated,
not evidence of128 gain. Warm128 shows no gain and peakTorch19.096GiB vs
64last15.231GiB (+3.865GiB), so retain64. Original baseline/quality not rerun.
Original-half Lt1/F32Lt3 is now integrated at idx10 with full1,999,496tokens,
chunk64, bounded8plans, explicit unsupported errors. Prior393-file freeze and
2caller sources/checkpoint metadata preserved; new405-file freeze. Final whole
model correctness/performance still required after conversion. Evidence:
`capture-lt-chunk/REVIEW.json`, `model-job-c03/repair-006-capture-lt/`.


Capture-owner correctness regression discovered by full-token guard (module013,
layer7): capture completes1969.08s, then indexer count audit fails before quant.
Shared block.attn.input chooses first_key once. A long first row3217 executes
conditional QSA indexer before QKV (threshold2051); following512 rows skip that
owner. Predicted full-c03 input-H coverage898184/1999496tokens. Existing layer3
uses the same owner history; prior completion did not establish full coverage.
Lt arithmetic is independent of this preexisting ordering defect.
`capture-owner/`: real layer7 long3217→short512 produces count3217→3217;
canonical always-run q_proj owner produces3217→3729, matching valid short-first
for ALL516 H bytes and normal refs. CPU7/infra432, cleanup0/KFD0, independent review.
GPU run retained failed additional assumption (other515 H equal broken control):
mlp.input differs8.34e-8 relative; valid short-first comparison is exact. Saved
CPU audit avoids GPU repeat. FP64 actual-input16×16 Gram relative7.83e-6.
Production remains pausedidx10; restore layer3 input via retainedidx2+existing
0/PLE/1/2 packed forward, then repair3 onward. Do not re-quantize healthy0..2 or
silently retain invalid full-calibration claims. Evidence capture-owner/REVIEW.json.


Owner-bug recovery authorization: user discarded unusable idx10 and old layer3~6
packs; exact deletion manifest retained under repair-007-capture-owner (81.29GiB).
Preserve idx2 and healthy layer0/PLE/layer1/layer2 packs. Prefix replay clones the
pinned converter, replacing only capture/quant/save-pack section with existing
packed load; original state advance/preserves/finite/checkpoint code retained.
Pilot first3 actual rows passes in45.74s,idx6,source/IDs intact,cleanup0/KFD0;
CPU AST6/owner integration7/infra432. First pilot failed before forward on a
premature display-only weights_numel lookup; preserved attempt-001, no math change.
Full2309-row replay completed in1813.25s,idx6,1999496tokens; first3 final rows
match pilot byte-for-byte, all rows finite, original protection and cleanup pass.
Independent review qualified promotion; rename preserves all2309 row inodes and
header/extents. Owner integration changes only the two production callers in the
old405-file freeze, adding14 dependencies (419 total); CPU admission passes.
Full calibration and Lt64 remain unchanged. No async state I/O change is included.
This qualifies state recovery, not complete layer0..2 model quality.
Evidence prefix-recovery/{SUMMARY,REVIEW,PROMOTION}.json and
model-job-c03/repair-007-capture-owner/{MIGRATION,ADMISSION}.json;
full model quality/performance remains final.


Conversion state I/O overlap (user requested stop/restart at layer3): CPU reader1
and writer1, each queue bounded2GiB, prefetch8rows. Writer owns cloned CPU
snapshot; stale reads invalidate/join, read-after-write waits, checkpoint drains
both workers before fsync/link. Current row/GPU/clone temporaries are additional.
CPU9 includes speculative-read failure blocking checkpoint. Real idx6 first16rows
GPU transfer+elementwise+I/O ABBA .83949/.74239/.72310/.72396s, all output hashes
exact, original metadata/stats preserved, cleanup0/KFD0. Warm timing is equal/noisy;
no end-to-end model speed gain established. User-requested overlap structure
adopted; arithmetic/full calibration/Lt64 unchanged, idx6 preserved.
Independent review + infra432/18.95s +428-file admission. Evidence capture-io/
{RESULTS,REVIEW,CPU}.json and repair-008-state-io/. One-off completed probe runner
lacks SIGTERM handler; recorded limitation, do not reuse. Production supervisor
retains existing signal cleanup. Main layer3 restarted for actual timing.


User-requested larger read queue integrated at healthy layer3→layer4 boundary:
reuse async I/O mechanics, cap800rows/10GiB (whichever first), writer2GiB unchanged.
CPU9 contracts plus3 policy checks pass: actual first window319rows/
10,602,127,360bytes; pure512-token rows fit512 at10GiB. Original428-file freeze
preserved in repair-009-large-prefetch; two callers changed, new433-file freeze.
No GPU rebenchmark or measured whole-conversion speed gain. Queue cap excludes
current tensors. Evidence capture-io-large/ and repair-009-large-prefetch/.

Remaining capture forward operator screen completed, not deployed. Actual
LinearFP16.forward controls retain expert FP16 and dense FP32 output. Shared-input
expert gate/up B2 Lt0:512 rows .118–.130→.096–.097ms;8192 rows1.654–1.662→
1.273–1.327ms. Sampled FP64 relative error is about2.25e-4/2.11e-4 for both paths;
outputs differ by relL2 about4e-5. Dense512 Lt4:2.55→.557–.565ms, but sampled
FP64 relL2 rises1.03e-6→1.90e-5. All candidates finite and repeat-byte-exact.
Output-buffer reuse and hot small weights confound pure-GEMM/end-to-end claims;
no actual capture/H/normal-reference/quantized quality qualification yet.
WMMA excluded pending its small analytic rounding discrepancy (max1.91e-6),
not proof of worse model quality. Original small-M hgemm also has a separate
M9..32/CPL2→CPL1 stale-grid defect: N64 second32 columns unwritten; this c03
uses minimum512 rows, so unaffected. Preserve attempt-003 and CPU audit; runtime
was not changed. Evidence capture-forward/{SUMMARY,STATE,LT-ANALYTIC,
WMMA-REVIEW,SMALL-M-CPU-AUDIT}.json and saved inputs/control/candidate tensors.
Next qualification: actual capture H/counters/normal references for promising
expert batching; independent error evaluation for dense; remove redundant FP32
preparation in original-half Gram and discarded-routing traffic. No25% subset.


Expert paired capture candidate prepared separately while main conversion runs:
only discarded up/gate LinearFP16 inners use shared-input B2 Lt0; original outer
H capture and normal references remain intact. Fresh output per expert is
mandatory because down H retains activated up storage until flush. That view
also retains the consumed gate half: up to5GiB extra at8192 rows, plus3.125GiB
packed weights, bounded8plans/256MiB workspace and other allocations.
No output-scratch reuse; inference tensors use pinned adjacent-call/source and
identity/stream guards rather than unavailable _version counters. CPU19 checks
include retained H storage, scalar/signed-zero byte checks and failure cleanup;
infra432/19.41s, independent review qualified_for_capture_screen. No new GPU
results yet, production433-file freeze unchanged. Natural-boundary runner waits
for main completion, then candidate-only64-row capture twice against saved
capture-lt-evaluate/1-original_f16/numeric.pt; no baseline recomputation.
Input reader differs from historical timing, so no speedup inference from this
qualification. Evidence capture-forward-pair/{CPU,INPUT,READINESS-REVIEW,
RESERVATION}.json. H/normal references and quantized quality remain unqualified.


Layer4 module016 disk-reserve incident: capture/quant/forward completed2528.59s,
then100GiB free-space guard stopped checkpoint publication. This was not a numeric
or pair-kernel failure; paired probe never reached GPU admission. New state output
is76.275GiB, so150GiB admission minus100GiB runtime reserve leaves too little
space; budget at least185GiB before next module including new packed tensors.
Independent complete2309row/1542packed/IDs/extents/mtime audit passed. Isolated
recover_disk_checkpoint reuses frozen validation/commit with exact disk-stop+
complete-forward eligibility; CPU9/independent review, idx8 committed1.68s with
no GPU recomputation. Old input/partial/scratch links rotated only after new inode
checks. Unused Docker buildcache older24h reclaimed28.85GB; free200.92GiB after
rotation. Production433 freeze unchanged. Evidence module016/DISK-RECOVERY.json,
CPU-COMMIT.json and repair-010-disk-checkpoint/. Pair first pre-admission failure
preserved in capture-forward-pair/attempt-001-boundary-disk; narrow reviewed CPU
recovery gate permits its first actual GPU screen from this healthy boundary.


Paired expert Lt actual capture completed (candidate-only64rows/50056tokens,
retained layer6 control reused):516 H counters/5normal refs/IDs/RNG exact,
512expert-down H changed (maxrelL2 1.260194e-5), other4H byte-exact; second
candidate full numeric repeat byte-exact. Pack3.125GiB/0.162s, GPU peak23.08→
23.24GiB across cases (old halfLt control peak15.23GiB). Capture64.0717 then
43.0643s; first-use/cache effects and changed read-only row loader prevent matched
speedup/regression claims against old44.78s. Finite/integrity passed, not quantized
quality qualification. Keep as uninstalled candidate pending convincing pipeline
gain and downstream quality; no full baseline rerun or new quant sweep yet.
GPU window133.30s, cleanup0/KFD0. Evidence capture-forward-pair/{SUMMARY,
DIFFERENCES-0,DIFFERENCES-1,STATE}.json, retained case numeric and originalcontrol.


Per-module disk preflight prepared at user request (not yet integrated): validate
fixed calibration row/token shape and current checkpoint, retire only completed
attempt transient directories with durable deletion records and current-link
protection. Failed unresolved and baseline/model data remain protected. Required
free bytes=100GiB+actual state bytes+16GiB (currently192.275GiB). Final main module
also budgets current qtensors plus20GiB for new head/side qtensors and compiled
copies/auxiliary slack; existing ngram is not recopied. Low space permits only
unused Docker buildcache older24h pruning, then alerts before GPU work if still
insufficient. CPU20 filesystem/negative cases, infra432/19.58s and independent
review pass; natural-boundary integration reserved, production433 unchanged.
Evidence disk-boundary/{CPU,REVIEW,INTEGRATION-CPU,BOUNDARY}.json.


Capture traffic removal qualified and integrated (2026-11-05): original-half
Gram uses its private stack for masking, eliminating unused F32 cast/mask and
second half buffer. Pinned standard all-expert discarded pass reuses one owning
gather, retains per-expert activated up for H, and removes empty routed-down
zero/multiply/scatter consumers. Normal forward and Lt1/3, chunk64, H addition
order/calibration/shape guards stay unchanged. Saved64-row/50056-token control
reused: all516H,5normalrefs,counters,IDs,preserve,RNG byte-exact in two runs; repeat
byte-exact. CPU exhausts65536 half patterns, mixed bad rows/float types, flush
tail/alias cases and runtime guards (7+5 groups), infra432/18.80s; independent
review passed. GPU128.23s window, cleanup0/KFD0. Capture59.13/37.48s and peak
13.41/13.57GiB are diagnostic cold/warm observations; historical control reader
differs, so no matched/full-conversion speed claim. Retained as exact traffic
reduction. No new full baseline or quantization quality rerun.
Disk preflight integrated alongside this change: past successful transient state
cleanup and dynamic space budget before every module. Previous433 freeze/callers
and idx9 retained; new450 freeze. Evidence capture-lean/{STATE,SUMMARY,REVIEW,
CPU,RUNTIME-CPU}.json and model-job-c03/repair-012-lean-disk/.

Disk preflight activation caught retired module011 idx10 log after approved
rollback to idx6/currentidx9; no GPU work started. Ignore past entries only when
all three cleanup trees are absent, while unresolved journals remain unconditional
blocks and newer attempts with actual targets remain blocked. Dangling cleanup
symlinks rejected. CPU23 + independent re-review + infra432 pass; four affected
freeze hashes updated with old450 freeze preserved. Real preflight ready276.88GiB
vs required192.275GiB, cleanup0; module018 Layer6 resumed. Evidence
model-job-c03/repair-013-disk-empty-history/.

First production layer with capture traffic removal completed: Layer6
2207.66s total (capture1565.66s, quant432.72s, forward206.60s),2309rows,
idx10,bad_rows[],cleanup0/KFD0. Actual gather2309/reuses1179899 and dead
down/scatter removals1182208 each. State IO read wait81.72s. Next Layer7
capture completed1598.89s and quantization began. These differ in layer/IO
from Layer5, so total differences are observations, not matched speedup proof.
Evidence model-job-c03/module-018/{STATE.json,run.log}, module-019/run.log.

Static all-expert capture candidate prepared separately during module020: skip
router/softmax/T×E assignment sort/count readback under pinned all-expert guards.
Aggregate repeated immutable-gather gate/up Inf/NaN counts once and multiply the
integer diagnostic by1024, preserving H queue/num_total and every expert GEMM.
Native counter uses uint64 atomic counts; final bounded int64 totals remain exact.
CPU9 groups include real capture_H mocked counter comparison with Inf/NaN and
nonzero initial counters; runtime5 groups pass. Not yet GPU qualified or deployed.
Next boundary combines saved64-row control exact/repeat checks and small count
ABBA; no whole baseline recapture. Evidence capture-static/{CPU,RUNTIME-CPU}.json.

Static all-expert candidate qualified/integrated at idx12 after Layer8:
saved64rows/50056tokens, all516H and metadata (Inf/NaN,num_total,count),5normal
refs,preserve,IDs,RNG byte-exact in two runs, repeat exact.64routes/64count
kernels/65472count reuses/64commits, bounded8plans. Capture42.43/20.75s vs
historical lean59.13/37.48s is encouraging, not matched whole-conversion timing.
Synthetic1024 repeated counts ABBA:512rows old168/115ms,new7.42/.700ms;
8192rows old4.289/4.280s,new4.263/4.205ms, all invalid-value integer counts exact.
These times cover count diagnostics only. Small retained benchmark input adds
approx40MiB to candidate peak. GPU window106.84s,cleanup0/KFD0; CPU9+runtime5
and infra432/19.67s, independent review pass. Original450 freeze/callers and
checkpoint retained; two callers changed, new464 freeze. Evidence capture-static/
{STATE,SUMMARY,COUNT-ABBA,REVIEW}.json and repair-014-static-capture/.

Production static capture: Layer10 completed1475.16s total, capture855.81s,
quant427.24s, forward170.72s; count kernels2309/reuses2362107. Full2309rows,
idx14,bad_rows[],cleanup0/KFD0. Following attention Layer11 capture893.20s
completed, so guarded static routing also executed in its real capture path.
No extra benchmark: observed different-layer timings, not matched speedup.
Evidence model-job-c03/module-022 and module-023/{STATE.json,run.log}.


## 2026-11-06 — Flash-Next MTP calibration preparation (CPU only)

- User selected full c03 MTP calibration and K4/K6/K8 role-based allocation,
  plus original BF16 vision. No running converter source was changed.
- Pinned MTP input FCs originally share `block.mtp.fc` although their inputs
  differ. A calibrated adapter must split their H maps: hidden FC sees four
  HC streams per token (7,997,984 samples), embedding FC sees 1,999,496.
- Runtime prompt alignment is zero first hidden + previous token's pre-mixer
  FP16 HC stack, with unchanged full IDs. idx51 is the last preserved input
  before the final mixer collapses that stack. Reuse it; do not replay main.
- CPU contract checks cover lengths1/2/33/512/8192, FP16 overflow/nonfinite,
  source immutability, actual1546-target recipe coverage, distinct H owners,
  count mismatch and uncalibrated fallback rejection. GPU qualification pending.
- Read-only idx51 boundary reservation:7CPU tests + independent review pass;
  tolerate checkpoint rename gaps, retain the lease on notification failures.
- Original vision payload from source safetensors headers:333 tensors,
  448,931,056 parameters,897,862,112 bytes(0.8362GiB). Existing vision_bits16
  compiler path copies raw extras with BF16 enabled; final payload audit pending.
- Evidence: flash-next-quant-20261104/mtp-calibration-candidate/ and
  mtp-calibration-boundary/. This records preparation, not measured GPU speed,
  MTP acceptance improvement, or full-model quality parity.

### Correction: target Gufo MTP semantics differ from pinned Kyojin

Subsequent source review found Gufo executor/reference use full HC*H FP32 RMSNorm
and pairs ids[:,1:] with pre-mixer hidden[:,:-1]. The initial per-stream/FP16/
zero-prefix CPU checks above validate the Kyojin adapter only. They do not qualify
Gufo calibration. Correct counts are1,997,187 valid pairs across all2309 rows,
7,988,748 hidden-FC samples. FP32 input must reach input Hadamard before rounding
rotated operands toFP16; output staysFP32. New T-1 decoder Gram shapes require
new bounded qualification, not an unreviewed allowlist extension. The candidate
was explicitly gated against GPU use while this adapter is corrected; live main
and idx51 reservation are unchanged. See MTP-GUFO-INPUT-MISMATCH.json and current
F3-MTP-CALIBRATION.md. No GPU/acceptance/throughput result is claimed.

### 2026-11-06 EXL3 Layer44 scale-search completion stall

Module056 stopped progressing at1142/1542 for>53min; two py-spy snapshots
showed worker K4 [2560,640] g_scale_search_batch mse2.tolist waiting for ROCr
completion. This does not identify the preceding failing kernel. Only the owned
container was stopped; cleanup KFD0, checkpoint47/2309rows and464 frozen hashes
verified; completed layers0–43 preserved. Fresh-context add and synthetic same
K4/mul1 g-scale workers1→2→2 passed finite/exact repeat (0.443/0.205/0.199s,
diagnostic only, no performance conclusion). Driver busy remained100 despite
KFD0 and no increasing available DRM engine counters over8.11s. No reset or
production quiet-gate waiver; root cause unresolved. Evidence: ai.tools
`tmp/test-results/flash-next-quant-20261104/model-job-c03/repair-015-quant-stall/`
STACK-1/STACK-LOCALS,PROBE-WINDOW,PROBE,CHECKPOINT-AUDIT,GPU-OWNER-DELTA,OWNER-REVIEW.

Layer44 stall follow-up: explicit user-approved MODE2 reset at2026-11-06 18:15KST
reported device-wedged recovery. Same bounded K4 probe passes again and exactly
matches pre-reset scale/MSE; recovery of device does not identify original kernel
cause. Post-reset10–11% is actual Chrome/KWin graphics (306ms/41ms over3.12s),
not the previous zero-counter/busy100 state. Production still stopped atidx47
because existing <=4 quiet admission is unmet; no app killed or gate changed.
Receipts same repair directory: GPU-RESET,RESET-KERNEL,POST-RESET-OWNER-DELTA,RECOVERY-STATUS.


### 2026-11-06: 변환 시작 시 화면렌더링과 GPU wedge 구분

KFD0/작은 연산 성공만으로 GPU 건강을 단정하지 않는다. 이번 busy100+DRM engine
카운터 정지 상태는 승인 MODE2 reset 후 해소됐고 커널도 wedged recovery를 보고했다.
reset 후 busy10~11%는 실제 Chrome/KWin gfx 증분으로 귀속됐다. 최초 원인은 미확정.
변환에만 고정 boot/pdev/전체 DRM client·proc 신원, 두 coherent 짧은 표본의
단조 카운터·non-gfx 증분0·낮은 busy와 gfx 귀속 조건을 추가했다. malformed counter,
새 client, reset counter, 설명되지 않는 activity는 차단한다. CPU7/독립리뷰 통과.
GPU/JOB락·KFD/NPU·RAM/디스크 guard 및 성능측정 quiet는 유지한다.
증거: flash-next-quant-20261104/model-job-c03/repair-016-desktop-admission.


### 2026-11-06: MTP 입력 보정과 shifted Lt 소형 검사

Gufo MTP는 F32 HC 전체10240차원 RMSNorm 및 hidden[:-1]/token[1:] 정렬이다.
일반 변환기의 per-stream/zero-first 경로를 그대로 쓰지 않는다. 실제idx51 네위치에서
Gufo source native와비교: rawFC 상대L2 최대9.01e-7, K8 packed 최대1.087e-6;
실제codec FP16 내측가중치와입력회전은byte-exact. BF16→FP16 로드의아주작은값은
최대2.98e-8 차이라원본가중치전체byte-exact라고하지않는다. 전체품질/속도증명은아니다.

T-1 Lt1510shape 지원,20edge 수치검사/반복성확인. 긴FP16 5shape는원래1e-5
screen실패(최대2.2100e-5), 기존8192 FP16Lt 약1.9902e-5와같은규모다.
저장8191input CPU FP64합산에서 대각후보편향 약-1.63e-5; FP32 control이더가깝다.
실패를지우지않고별도3e-5 후속screen으로실제MTP검사까지허용했다. 속도·수락률·
최종모델품질은미검증. calibration Gram속도와실제draft+verify token/s는구분한다.

호출부를cm.main밖으로옮기면capture_fast의caller-code guard로자동scope가비활성이다.
별도run_module에서는검토된index1 discarded capture에만명시scope를연결하고
normal reference를밖에둔다. 실제2행에서staticroutes2/emptydown1024/H516카운터정상.
exec생성flush는inspect.getsource가실패하므로실제marshal코드/Python버전/고정leafSHA로
신원을기록한다. 기존입력pack/H/derived는명시input-preserving migration으로재사용했다.
읽기전용source와RW별도bind간hardlink는같은host디스크라도EXDEV다. 원본예약을유지한
borrowed read-only행+private output overlay로대체하고완전advance전checkpoint를금지한다.
근거: flash-next-quant-20261104/mtp-calibration-runs/, mtp-borrowed-source/,
mtp-calibration-work/main/{gram,gram-followup,native-raw-002,native-packed,probe}/.


### 2026-11-06: MTP 전체 입력 보정 완료·결과 재사용

전체2309행/1997187 shifted쌍에서 입력FC 두개 K8 보정1441.10초 완료.
저장로그 구간: 통계1221.96초/양자화12.31초/전체derived전진204.60초.
hidden H7988748·embedding H1997187, H/refs/packed finite·hash검사통과.
첫5보정참조RFN0.01080~0.01382, SQNR37.80~40.10dB는 입력결합블록 지표다.
전체품질·초안수락률·추론성능으로 일반화하지 않는다. 원본idx51보존/cleanup0/KFD0.
2309개F32 T-1 derived행의header/inode신원과실제FCpack을연결확인한뒤 재사용하면,
디코더에새76.19GiB할당을중복예산잡을필요없다. 현재검토예산122.05GiB에
runtime100GiB floor·H/refs·packed상한·margin을포함한다. 행payload전체재해시없음.
근거 flash-next-quant-20261104/mtp-full-input-review/{REVIEW,CPU-AUDIT}.json 및
mtp-calibration-runner/contracts/full-decoder-001-admission.json.


### 2026-11-06: MTP 전체 디코더 보정 완료

2309행/1997187쌍 통계907.01초,1544투영양자화424.12초,5정상참조5.58초,
모듈1364.01초. 516H/카운터/finite/inf_nan0·실제혼합1546=K4 1536/K6 5/K8 5
저장팩/refs/Hpayload/입력FC→derived→H→packed 연결독립감사통과.
첫5보정참조RFN3.711~5.571%/SQNR27.05~30.85dB는블록출력오차이며
초안수락률·전체품질저하율·추론속도아님. 2행pilot와H표본이달라동등비교금지.
원본idx51/저장H/packed/refs보존, baseline/입력보정재수집없음.
근거 flash-next-quant-20261104/mtp-full-decoder-review/.


### 2026-11-06: 보정된 side model을 위한 마지막 본체 단계 분리

완료MTP재사용시본체converter와supervisor를함께main-only로바꿔야한다.
converter는EXL3_SKIP_SIDE_MODELS=1/vision16 및MAIN-CONVERSION·compileargs를
기록하고supervisor는idx53+그marker만완료로본다. marker없는idx53은head재실행금지.
이모드에서head시점옛packed+20GiB compile예산은실제로실행되지않는다.
검토된정확한caller/source/c03와idx51/52에만예산분리;100GiB실행여유+
실제state+16GiBmargin 및원래cleanup유지, 최종assembler는자체완성팩예산을요구.
CPU예산4+완료4/독립리뷰/통합CPUadmission/infra432(18.91초)통과,470동결로재개.
근거 flash-next-quant-20261104/model-job-c03/repair-018-final-mtp-vision/ 및
mtp-final-compile-candidate/main-only-disk/. GPU산술변경없음.


### 2026-11-07: terminal head 로그잇과 비동기쓰기큐 크기

module062 head H수집2309행2405.63초·K8양자화637.64초는완료됐지만,
첫5참조출력전진중7793×248320 FP16로그잇3.60GiB가2GiBasyncwrite budget을넘었다.
이는캡처/양자화/메모리부족실패가아니며완료qtensors/lm_head.safetensors는보존됐다.
첫5head상태는일반hidden보다크다(총9.18GiB);쓰지않는2304행은원래prehead입력유지.
복구는저장pack+원본idx52로잃어버린5RAM참조만재비교하고직접파일쓰기,
완료H/양자화전체반복없음. 후보CPU5·감독CPU7·독립리뷰/infra43218.73초통과,
현재GPU실결과대기. 새완료marker는복구결과검토후에만생성한다.
근거 flash-next-quant-20261104/head-recovery/.


### 2026-11-07: c03 조립팩의 K8 런타임 및 최종 품질 검증

최종17shard 조립159.31초, MTP1546투영(K4 1536/K6 5/K8 5)과비전333개
BF16 payload exact. 과거native-vision 이미지의실제ELF는K8 GPU커널이없어
정상lm_head K8팩을거부했다. 로더상한만완화하지않고현재HEAD a81c0f93…의
Gufo/rq_logits를같은NixROCm10.1로빌드했다.319작업완료,각K8심볼64개,
실제빌드의존성과실행동적라이브러리Nix경로확인. Nix호스트없는환경에서는
보존gufo-nix volume을RO로붙이고실행ELF와baseimage신원을구분한다.

새ELF6edf0d44…로HTTP10요청은MTPoff/on 모두본문·종료·토큰수동일,
산술42/17·BF16비전/이미지이력·prefix반복1821/branch1811캐시통과.
RAMprefix재사용검사이지물리GPU블록zero-copy증명은아니다.
bench stdout/stderr합친로그는output_sha256=와실제SHA사이에표를끼울수있다.
smoke003실패영수증보존,저장off실제SHA복구후누락MTP만별도확인한다.
성능수집기는stdout/stderr분리하여같은파싱실패를방지한다.

저장HF BF16원본32760위치(8×4095)와새팩FP16KV/MTPoff출력을비교:
candidate수집155.37초,CPUscore5.14초,64head replay exact/finite/cleanup통과.
평균ΔNLL -0.010195878, KL256 0.041522177, top1일치0.908394383,
기준top256평균확률질량0.970305786. 전체어휘KL·독립holdout·decode정답률이아니다.
참조는HF CPU BF16, KV실제dtype은기록부족으로단정하지않는다.
기존UD자료와같은위치CPUpaired비교:ΔNLL차이 -0.016849077,
KL차이+0.002295360, top1차이-7/32760. KL/top1 block/window CI는0포함,
기존UD실행환경ROCm7.2.4/GCC13이달라양자화만의인과효과로해석하지않는다.
원본/기존baseline재수집없음. 전체속도·MTP성능검증은별도진행한다.
근거 flash-next-quant-20261104/final-current-nix101/ 및
final-validation/{smoke-003,quality-current-001}/.


### 2026-11-07: c03 혼합 MTP 최종 생성 속도

현재동일Nix10.1/FP16KV팩에서off→MTP최대7:TG128 depth0
27.33±0.03→50.70±0.35tok/s,depth32768 26.72±0.01→87.30±0.05.
각2반복최종tokenSHA동일. 32K샘플111/111초안수락이라일반긴문맥속도아님.
HTTPgreedy4요청전체wall C1 26.13→41.70,C2합산36.56→45.55tok/s,
C1off기준C2off/C1on/C2on본문/종료/토큰수모두동일.
PP2048 1460.97→1370.97,PP8192 1501.37→1403.43으로약6–7%비용도존재.
동일자료의순차off/on관측이며희소온도표본/적은반복한계유지.
8case cleanup0/returnedKFD0/팩불변,최소가용39.65GiB(32K설정).
근거 final-validation/throughput-001/RESULT-REVIEW.json,FINAL-QUALIFICATION.json.
기존baseline전체반복없이완료,시간별감시종료. 서비스자동배포없음.


## 2026-11-07 c03 KVarN K4V4 versus retained FP16

Same assembled c03 pack and frozen a81c0f93 Nix ROCm10.1 runtime; no re-quantization,
rebuild, or FP16/BF16 baseline rerun. Evidence root:
`/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104/final-validation/`.
Quality: `quality-kv4-001/{RESULT-REVIEW.json,quality/compare-fp16.json}`.
Performance/correctness: `kv4/throughput-001/{STATE,RESULTS,RESULT-REVIEW}.json`.

| Metric | Saved FP16 | K4V4 |
|---|---:|---:|
| Prefill 2048, MTP off (tok/s) | 1460.97 | 1456.81 |
| Prefill 8192, MTP off (tok/s) | 1501.37 | 1487.07 |
| TG128 depth0, MTP off (tok/s) | 27.33 | 27.10 |
| TG128 depth32768, MTP off (tok/s) | 26.72 | 26.33 |
| TG128 depth0, adaptive mixed MTP max7 (tok/s) | 50.70 | 39.30 |
| TG128 depth32768, adaptive mixed MTP max7 (tok/s) | 87.30 | 86.26 |
| HTTP C1 greedy, MTP on group-wall tok/s | 41.7024 | 42.6084 |
| HTTP C1 sampled, MTP on group-wall tok/s | 41.9665 | 44.4836 |
| HTTP C2 greedy, MTP on aggregate group-wall tok/s | 45.5548 | 47.0233 |
| HTTP C2 sampled, MTP on aggregate group-wall tok/s | 50.0779 | 48.6803 |

All eight cases passed; K4 internal off/on native emitted SHA and C1/C2 HTTP greedy
signatures match. Actual KVarN K4V4 metadata and repeat/branch cache reuse verified.
Cross-KV texts differ: these historical timing ratios are not isolated kernel gains.
Depth32768 is a repetitive lookup-heavy case, not general neural-MTP throughput.
Two native repeats and one HTTP group per mode do not resolve small timing changes.

Same configured pool 1932735283 bytes: off 297→1028 blocks (76032→263168 tokens);
MTP on 274→798 blocks (70144→204288 tokens). Pool bytes were held fixed, so this
measures capacity gain, not whole-process RAM reduction. FP16 sink/tail, MTP draft KV,
GDN states and other allocations remain. Minimum host availability across cases 39.47GiB.

On the same 32760 teacher-forced prefill positions (8×4K windows), BF16-reference
compressed top256+rest KL 0.04152218→0.04497509; top1 agreement 90.8394→90.3327%.
Both paired differences exclude zero in block/window 95% bootstrap intervals.
NLL 2.1626503→2.1645008, paired +0.0018505 has intervals crossing zero.
This is a small measured fidelity loss, not quality equivalence, and not independent
holdout/general task/long-context/vision quality proof. K4 favors KV capacity; it is
not a uniform speed win. Persistent service/default deployment was not performed.


## Format-general paired prefill and EXL3 dispatch, November 7, 2026

**Final scope after user-requested rollback:** retain Dense K1..8 and K3.5
head, but revert the GGUF49-pair dispatch, expanded header caching, associated
K-tail change and candidate-only GGUF tests. Original14-pair behavior returns.
The measurements below are preserved historical evidence for the rejected
GGUF candidate. `build/f3-general-formats/rollback/` records the selective
rollback; no GPU/model baseline or service restart is needed for it.

Retained for generality: dense traversal's K7 gate becomes integer K1..8 at
unchanged6144→2560/R2048/singleton geometry; draft subset head adds K3.5 via
the existing half-rate decoder, plus zero-work/prefix-only launch guards.
GGUF paired prefill expands14→49 combinations among Q3/4/5/6_K, Q8_0 and
IQ4_NL/XS. Header reuse follows each operand's format, not its partner.
Mixed-Q6 decode placement and accumulator ordering remain intact. IQ3_S/F16
remain unsupported by this paired route because their coordinate/loader
contracts differ. New32-weight combinations required eliminating invalid
up fetches in the final partial K64 stage; covered K32/96/160.

All focused outputs match independent/direct and preserved controls: GGUF
424,157,714B and dense167,772,160B byte-exact, head K3.5 K128/K256 and integer
K3..8 with IDs/guards/repeat/empty-prefix tests pass. Nix10.1 incremental
build/CPU checks and independent review pass. Dense eight-rate warm latency
changes-10.36%..+0.32%. Actual two-launch production fallback versus new fused
GGUF35 pairs gives16.35–32.63% lower latency on small shapes,39.46–58.25%
on tested admission boundaries and12.73–21.88% on the larger shape. These
are warm component measurements; no model-level or universal gain claim.
An old/new A/B/B/A check retains existing Q5/IQ4_NL's small+3.96%/+2.11us cost
under the user's explicit generality-first criterion.

Measurement correction: the initial four-launch independent oracle is valid
for correctness but is not production fallback timing. Its apparent new-pair
small-shape slowdown must not drive dispatch policy. Preserve the initial
screen, then reuse its numeric outputs and only fix affected timing controls.
Original model/reference and successful dense/head checks are not repeated.
Evidence: `build/f3-general-formats/{QUALIFICATION.json,SCREEN.json,ADMISSION.json,
validation-001,validation-002,validation-003,INDEPENDENT-FINAL-REVIEW.json}`.
The original serving runtime is restored after the approved windows.


### EXL3 hot256 with more waves, November 7, 2026

K4 Mode3 screen with128×256/8×2/K32,8×2/K64,8×4/K64 avoids the old
hot256 register pressure:114/137/90VGPR, zero scratch/spills (currenthot128140).
All six three-/two-class candidates preserve output bytes/finite/guards/repeat,
but the three-class complete matrix launches lose2.67–9.87% across synthetic
pp512/2048/4096. Removing the hot128 launch also fails to give a useful signal
(pp2048/4096 +14.81–38.36% versus stored control, cross-window advisory).
Keep cold64/hot128; fewer registers and fewer weight decodes do not by
themselves offset altered workgroup shape, padding and scheduling. Their
individual causal contributions were not profiled or established.

Fixed1172-slot capacity, null gate/up exponents, scaled down, deterministic
512-expert histograms;13 controls stored once and reused. Output guard
shift2bytes and excluded descriptor-building/glue costs limit timing scope.
No production MoE edit or model-baseline repeat. Detailed scope and source/ELF
evidence: `build/f3-moe-wide/QUALIFICATION.json`; model experiment entry
“Wider EXL3 MoE token tiles”.

### Fused EXL3 gate/up and down-input preparation, November 7, 2026

Rejected an isolated K4 prototype that kept gate intermediates inside the
workgroup, preserving Mode3 FP16 rounding, Hadamard/scale/SwiGLU arithmetic,
and separate whole-row down exponents. Use a disjoint dead output slab:
writing into gate input while other output-channel blocks still read it is
unsafe. LDS transpose can reuse dead operand storage after block barriers.
Keeping rounded gate half bits in two uint32 registers instead of four restored
FP32 values reduces register use without adding quantization.

These savings did not produce a useful speed signal. Restored-FP32 hot128
initially spilled (256VGPR/32B scratch); moving to16 waves removed spills
(159VGPR). Packed-half candidates use111VGPR for64 rows/16 waves,143 for128
rows/16 waves,147 for split64/8 waves, all scratch/spill-free. Across synthetic
pp2048/4096, the fastest packed candidate is still12.0–17.2% slower than stored
complete-stage controls; other packed variants lose18.5–36.7%. Cross-window
numbers are advisory, not an isolated causal or model-performance result.
Removing stores alone does not establish a gain; the cost of longer fused
kernels and changed workgroup layouts must also be measured.

Seven shapes include awkward counts through513, signed scales, zeros/tiny
inputs and512-expert histograms. All42 packed candidate/repeat comparisons
match saved half outputs and row exponents exactly; guards/finite pass.
Positive exponents are covered, negative-exponent fallback is not. The first
window completed numeric/timing work but its runtime observer rejected a stale
expected ELF prepared before the signed-fixture build completed. Keep that
failure: it does not qualify deployment. The second window waits for completed
build/hash, verifies actual Nix10.1 ELF, and reuses controls without recomputing
them. No production changes, full-model rerun or deployment. Evidence:
`build/f3-moe-fused/{SCREEN-FIRST,SCREEN-PACKED,QUALIFICATION}.json`.
