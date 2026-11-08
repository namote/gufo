# Qwen3.8 Flash-Next benchmarks

## Original BF16 / pure F16 HC comparison, November 9, 2026

Only200 HC down/up matrices differ; c03 body/MTP, FP16 KV, Nix10.1,
r2, temperature0, top_p1, top_k0, seed314159 and draft7 are fixed.
Installed calibrated-hybrid results are saved controls, not a new same-window
control. Each candidate uses a fresh process/cache. Rates are completed tok/s.

| Case | Installed Q8 hybrid (saved) | Original BF16 | Pure F16 |
| --- | ---: | ---: | ---: |
| PP2048 | 1522.02 | 1273.07 | 1405.17 |
| PP8192 | 1554.38 | 1251.23 | 1457.40 |
| AR128 d0 | 32.78 | 27.88 | 29.04 |
| AR128 d32768 | 32.44 | 27.92 | 28.80 |
| MTP128 d0 | 65.69 | 52.50 | 59.51 |
| MTP128 d32768 | 104.39 | 83.06 | 96.84 |

BF16 loses16.36–19.50% PP,13.93–14.95% AR and20.08–20.43% MTP;
F16 loses6.24–7.68% PP,11.22–11.41% AR and7.23–9.41% MTP.
Both pure variants emit the same tokens, and each matches its own AR/MTP
hashes at both depths/repetitions. Shallow text differs from installed hybrid,
so hybrid free-generation rates are not identical-work kernel timings.
Deep output is identical across all:17cycles,111accepts including101lookup
accepts. This lookup-heavy rate is not general neural-MTP throughput.
Shallow F16 is39cycles/88of111drafts accepted; BF16 is37/90of111,
both with1of2lookup accepts. Better BF16 acceptance does not offset its cost.

Saved fixed-token histories provide a separate cost observation (milliseconds):

| Full forward scope | Hybrid | BF16 | F16 |
| --- | ---: | ---: | ---: |
| Prefill32760 scored positions | 46887.31 | 56120.02 | 47609.75 |
| Eight2048-prefix +128scalar sequences | 48051.53 | 57158.07 | 52657.36 |

These timers include forward/logit transfer and prefixes where applicable;
they exclude separate head-replay checking and CPU scoring. They are not
pure TG kernels. Historical timing uncertainty remains visible; no baseline
forward or whole-model timing matrix was repeated.

The current pure16 implementations are not a datatype hardware ceiling.
F16 uses fused HC-down+SiLU; BF16 uses generic BF16 projection then SiLU.
BF16 prefill materializes the deferred FP32 norm before BF16 narrowing because
the existing combine produces F16. Contributions were not individually profiled.
All200 HC weight allocations total1.258850GiB for installed F16-down+Q8-pair,
versus1.220703GiB pure16 (39.06MiB less). Conversely pure16 projection
weight payload is1.882× Q8. Allocation capacity is not traffic; these are
logical weight bytes, not whole-server RSS or measured DRAM bandwidth.

Evidence: `build/exl3-hc/precision-001/{RESULT,INPUTS,INPUTS-002}.json`,
`window/f16-*`, `window-002/bf16-*`. Model/default settings are unchanged.

## Rejected standalone HC prefill Q8 candidates, November 8, 2026

Pinned Nix10.1, same c03 all200 GPTQ sidecar, FP16 KV, r2. Saved installed
controls from `runtime-qualification-001` reused, never remeasured. A replaces
remaining HC F16 down with Q8×Q8; B uses Q8 weights × F16 activations there.
The existing producer-fed W8A8 route is unchanged. Values below are tok/s.

| Case | Saved installed control | A | B |
| --- | ---: | ---: | ---: |
| PP2048 | 1522.02 | 1528.63 | 1525.18 |
| PP8192 | 1554.38 | 1560.33 | 1555.40 |
| AR128 d0 | 32.78 | 32.06 | 32.18 |
| AR128 d32768 | 32.44 | 31.51 | 31.46 |
| MTP128 d0 | 65.69 | 47.88 | 36.05 |
| MTP128 d32768 | 104.39 | 83.37 | 77.71 |

PP observations are only +0.38..0.43% (A) and +0.07..0.21% (B), not a
resolved speed gain. Both candidates have large MTP timing flags. Shallow
greedy outputs and acceptance differ from the saved control, so those rates
are not an isolated arithmetic comparison. Deep emitted tokens and total
17cycles/111accepted remain equal, but deep timing still regresses; neither
text change nor acceptance alone explains all of the observed slowdown.
No causal attribution or same-window baseline confirmation is claimed.
Quality independently rejects both candidates (see QUALITY.md), so no extra
baseline timing or third candidate was run. The qualified runtime is retained.

Candidate AR/MTP emitted-token hashes match at both depths and repeats.
Pure neural acceptance must exclude lookup proposals; deep total acceptance
is dominated by101lookup tokens, as in the original benchmark.
Evidence: `build/exl3-hc/prefill-q8-{001,002}/RESULT.json` and per-case logs.


## All200 GPTQ HC Q8 sidecar, November 8, 2026

Same B10 compute kernels, pinned Nix10.1, c03, FP16 KV, r2; saved Cr1/Cr2
controls reused. Candidate only; no whole baseline remeasurement.

| Case | Saved B10 mean tok/s | Calibrated HC tok/s | Change |
| --- | ---: | ---: | ---: |
| PP2048 | 1516.34 | 1522.02 | +0.37% |
| PP8192 | 1549.30 | 1554.38 | +0.33% |
| AR128 d0 | 32.87 | 32.78 | −0.27% |
| AR128 d32768 | 32.56 | 32.44 | −0.37% |
| MTP128 d0 | 67.91 | 65.69 | −3.27% |
| MTP128 d32768 | 104.69 | 104.39 | −0.29% |

Candidate AR/MTP emitted-token hashes match at both depths and both repeats.
Shallow output differs from old B10 after changing HC weights: MTP cycles36→38,
accepted91/111→90/114 (81.98→78.95%). This is a prompt-specific output/acceptance
tradeoff, not an isolated kernel speed comparison. After subtracting lookup, shallow neural MTP is89/113. Deep repetitive lookup case
retains17 cycles,111/111 total drafts and101 lookup accepts (neural MTP10/10); it is not general
generation speed. No speedup claim from sub-percent differences.
Evidence: `build/exl3-hc/runtime-qualification-001`, `build/b1/b10/perf`.


## B10 deployed runtime (B9 + GDN k.q order), November 8, 2026

Same-window B C B C vs deployed B5 (`31cee0b6`), c03, Nix10.1, FP16 KV, r2:
PP2048/8192 1513.5/1545.2→1516.3/1549.3 tok/s (+0.2/+0.3%); AR128 d0/d32K
32.92/32.62→32.87/32.56; MTP128 d0/d32K 67.02/104.53→67.91/104.69. Output
hashes equal B5; prefill logits changed (GDN k.q order), quality unchanged
within noise (KL256 0.041926). Now deployed (`1d35af4f`). Details:
[know-how](../../GFX1151-KNOWHOW.md).

## B6 prefill streaming stores and packed attention gate, November 8, 2026

Same-window rotation vs deployed B5 (`31cee0b6`), c03, Nix10.1, FP16 KV, r2 ×2:
PP2048/8192 1515.4/1546.0→1520.5/1553.0 tok/s (+0.3/+0.5%); PP MTP and TG
unchanged. Outputs and prefill logits equal B5. CU-masked shared-MLP streams
lost 6–17%. Details: [know-how](../../GFX1151-KNOWHOW.md).

## B5 routed down tiles and decode side branches, November 8, 2026

Same-window B C B C vs B4 adopted tree, c03, Nix10.1, FP16 KV, r2; adopted
tree `eb789c77` is the measured C−J1 binary: AR128 d0/d32K ≈32.9/32.6 tok/s
(+1.3/+2.7%), MTP128 d0/d32K ≈67.0/104.6 (−0.6/+0.9%), PP unchanged. All within
the gate. Outputs and quality equal B4. Details: [know-how](../../GFX1151-KNOWHOW.md).

## B4 routed LDS staging and simplified decode, November 8, 2026

Same-window rotation vs B2 `9599b4c6` (adopted tree `dc24fd89`), c03, Nix10.1,
FP16 KV, r2: AR128 d0/d32K 31.77/31.09→32.49/31.75 tok/s (+2.3/+2.2%);
MTP128 d0/d32K 63.95/99.01→67.39/102.75 (+5.4/+3.8%); PP2048/8192 unchanged.
Outputs and quality equal B2. B3's coupled fusions were reverted (no wall-clock
gain). Versus the pre-B1 baseline: AR +19.0%, MTP d0 +31.2%.
Details: [know-how](../../GFX1151-KNOWHOW.md).

## B3 decode dispatch fusion, November 8, 2026

Same-window B C B C vs B2 `9599b4c6` (candidate `5bda5f57`), c03, Nix10.1,
FP16 KV, r2: PP2048/8192 off 1519.5/1551.1→1519.1/1549.4 tok/s; AR128 d0/d32K
31.74/30.99→31.36/30.69 (−1.2/−1.0%); MTP128 d0/d32K 63.92/98.74→62.17/95.69
(−2.8/−3.1%). Dispatches/token 1442→998 but fused HC kernels are slower; within
the gate, not adopted as a gain. Outputs, prefill logits and quality equal B2.
Details: [know-how](../../GFX1151-KNOWHOW.md).

## B2 producer rotation, verify state and side stream, November 7, 2026

Same-window B C B C vs B1 `cc86343e` (candidate `9599b4c6`), c03, Nix10.1, FP16
KV, r2: PP2048/8192 off 1507.96/1534.37→1516.66/1547.74 tok/s; AR128 d0/d32K
31.04/30.38→31.80/31.10; MTP128 d0/d32K 61.28/94.61→63.89/99.19. Outputs and
prefill logits byte-identical to B1. Versus the pre-B1 baseline: AR +16.4%,
MTP d0 +24.4%, PP2048 +3.1%. Details: [know-how](../../GFX1151-KNOWHOW.md).

## B1 decode glue and HC Q8, November 7, 2026

Same-window B C B C control (baseline `f24b10ea`, candidate `cc86343e`), c03,
Nix10.1, FP16 KV, r2: PP2048/8192 off 1471.5/1499.7→1508.7/1535.0 tok/s
(+2.5/+2.4%), MTP 1395.9/1419.8→1440.5/1461.7; AR128 d0/d32K
27.31/26.75→31.02/30.37 (+13.6/+13.5%); MTP128 d0/d32K 51.36/87.62→60.72/94.51
(+18.2/+7.9%, d32K lookup-heavy). Outputs may differ from baseline (Q8_0 HC,
F16 prefill HC gate); BF16-teacher KL256 0.041522→0.041742, top-1
90.8394→90.8303%, NLL 2.16265→2.16205, all paired intervals span zero.
Details: [know-how](../../GFX1151-KNOWHOW.md).

## Generalized kernel paths, November 7, 2026

Combined all-rate R3/R4 live rows, Mode3 staging and stable grouped vector
loads; c03 pack, pinned Nix10.1, FP16 KV, two CLI repetitions. Existing matched
runtime/model/settings baseline is reused; these small changes are observations
against stored results, not a fresh alternating speedup experiment.

| Scope | Saved baseline tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| PP2048, MTP off | 1460.97 | 1457.86 | -0.21% |
| PP8192, MTP off | 1501.37 | 1495.46 | -0.39% |
| AR128 d0 | 27.33 | 27.32 | -0.04% |
| AR128 d32768 | 26.72 | 26.76 | +0.15% |
| PP2048, MTP7 | 1370.97 | 1385.03 | +1.03% |
| PP8192, MTP7 | 1403.43 | 1416.51 | +0.93% |
| MTP128 d0 | 50.70 | 52.27 | +3.10% |
| MTP128 d32768 | 87.30 | 87.69 | +0.45% |

All AR/MTP token hashes and draft work match saved controls. The d32768 MTP
fixture is lookup-heavy. No individual mechanism's contribution is isolated.
Receipts: `build/f3-general-opts/validation-002/` and its preserved TG-MTP
`PARSE-REPAIR.json`/`REVIEWED-RESULT.json`; no successful measurement rerun.
HTTP group-wall aggregate MTP C1 greedy/sampled41.70/41.97→43.29/43.80tok/s;
C2 45.55/50.08→45.90/52.48. No individual prefill/decode slowdown crossed
both5% and10ms across off/MTP C1/C2. Sampled outputs are not an equality
oracle; these aggregate rates are not sums of individual decode rates. C2 MTP draft/accept counts differ from the
stored baseline despite identical greedy output; its timing difference cannot
be attributed solely to kernel work.


## Native EXL3 vision (2026-11-04)

Native K6 vision residency 352601088 B versus historical BF16 903984064 B,
a 525.84 MiB reduction. Quantization/weights differ; not a matched accuracy or
latency comparison. K8 generic decoding uses temporary per-image memory.
[Evidence and limits](../../../plan/exl3/archive/F3-VISION-NATIVE.md).

## Physical KV prefix references, November 4, 2026

Same Nix10.1/native5.05-NG6/K6V6 settings, one repetition; saved K6 results
reused without another baseline run. Desktop graphics remain active, so these
are observations rather than matched speedup qualification. The final serving
admission fixes do not participate in these standalone bench paths.

| Scope | Saved K6 tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| PP2048 | 1374.42 | 1372.59 | -0.13% |
| PP8192 | 1403.11 | 1398.58 | -0.32% |
| AR128 d0 | 24.33 | 24.30 | -0.12% |
| AR128 d32768 | 23.58 | 23.59 | +0.04% |
| MTP128 d0 | 35.66 | 33.90 | -4.94% |
| MTP128 d32768 | 76.06 | 76.67 | +0.80% |

AR/MTP output hashes remain exact at both depths; d32K MTP is lookup-heavy.
Sharing changes physical block occupancy and restore work inside the fixed
pool, not the size of its initial backing allocation. Direct GPU fixtures
restore shared4K branches with at most two newly allocated blocks, including
MTP's private frontier/guard. All88 boundary/reference cases pass.
Evidence: `build/fn-kv-sharing/RESULT.json`, `gpu/` and `gpu-reservation/`;
[implementation and limits](../../../plan/exl3/archive/F3-KV-REFERENCES.md).


## K6V6 candidate observations, November 3, 2026

One repetition, native5.05/NG6, Nix10.1, K6V6, YaRN off; desktop graphics
active (11% at admission), no competing KFD compute process. These are advisory,
not matched performance qualification. Existing baselines were reused, not rerun.

| Scope | K6V6 tok/s | Saved quiet K4V4 tok/s |
| --- | ---: | ---: |
| PP2048 | 1374.42 | 1425.09 |
| PP8192 | 1403.11 | 1462.00 |
| AR128 d0 / d32768 | 24.33 / 23.58 | 25.20 / 24.47 |
| MTP128 d0 / d32768 | 35.66 / 76.06 | 40.92 / 79.27 |

MTP d0 observation is12.9% lower; precision, compiler specialization, desktop
activity and repeat count are not separated. AR and MTP output hashes agree
at both depths. The d32K fixture is lookup-heavy. All checks and GPU health pass.
Evidence: `/home/m2on/.local/state/gufo/k6v6-256k-20261103/qualification/`.

## Shared KV and durable continuation, November 2, 2026

Shared KV implementation preserves native5.05/K6 standard model rates against
saved baseline (no rerun): PP2048/8192 1426.88/1460.75 →1425.09/1462.00;
AR128 d0/d32K25.20/24.49 →25.20/24.47; MTP40.85/79.48 →40.92/79.27tok/s.
All hashes exact, maximum change0.264%; d32K MTP is lookup-heavy.
C4 shared serving passes and actual disk restore uses12352 cached+7 prefill
in0.486s with identical output. No full two-times-256K timing claim.
[Evidence and limitations](../../../plan/exl3/archive/F3-SHARED-KV.md).

## Exact live-row verification, November 2, 2026

Native5.05/K6 n-gram, Nix ROCm10.1, kvarn-k4v4, two repeats per configured
benchmark, one final candidate window against the saved native baseline:

| Scope | Baseline tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| PP2048 / PP8192 | 1423.76 / 1461.67 | 1426.88 / 1460.75 | +0.22% / -0.06% |
| AR128 d0 / d32K | 25.20 / 24.43 | 25.20 / 24.49 | 0.00% / +0.25% |
| MTP128 d0 / d32K | 40.09 / 78.53 | 40.85 / 79.48 | +1.90% / +1.21% |

Outputs and cycle/draft/acceptance counts are unchanged. PP/AR are effectively
neutral; d32K is lookup-heavy and its gain overlaps baseline spread.
The21–32% routed component gain is not a whole-model claim. No new Kyojin run.
Image `ai-tools/gufo-eval:f3-rows-91fce75a-rocm101`.
[Full results and rejected candidates](../../../plan/exl3/archive/F3-TGMTP-ROWS.md).

## Same-pack local Kyojin follow-up, November 2, 2026

Same-pack Gufo MTP42.55/46.94/62.66tok/s versus saved Kyojin42.52/49.44/63.22
(prose/chat/code,256tokens,neutral greedy,C1/context16384). Gufo text equality
passes3/3; no rawtoken or cross-engine quality parity claim. HC effective
quantization/head/KV/runtime differ; all cross-engine texts differ. No new
kernel change or default model switch. Current native5.05 profile attributes
ARdense35.44% and MTPexperts27.43%/dense22.21%; retain as diagnostic evidence,
not unprofiled speed. Both completed windows healthPASS; prior invalid default
penalty attempt is preserved. Existing numerical qualification is not repeated.
[Exact scope, receipts and next batch](../../../plan/exl3/archive/F3-KYOJIN-SAME-PACK.md).


## Native EXL3 n-gram, November 2, 2026

Native directory loading removes GGUF metadata/PLE dependencies for text/MTP.
The selected5.05 pack now uses its own K6 ring table, with upstream FP16 output
rounding; Kyojin K4 sharding is CPU-covered. Existing GPU optimizations remain.
One combined window passes quality/sampling/batch/32K/cache/PP/TG and health;
no GGUF mappings or baseline rerun. PP2048/8192=1423.76/1461.67tok/s;
AR d0/d32K=25.20/24.43; MTP=40.09/78.53. PP/AR/d32K MTP are within1% of saved
rates, but d0 MTP falls14.15% with changed output and acceptance88/113→74/114.
This material workload-specific loss is retained; do not describe universal
performance parity. Native full-vocabulary NLL improves2.175202→2.167751,
paired window95%CI forΔNLL[−.017370,−.001770]. Top-1 change−.0794pp is unresolved.

[Complete scope, caveats and evidence](../../../plan/exl3/archive/F3-NATIVE-PACK.md).


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
cross-pack reference: Kyojin had not been run locally at that stage. Frozen source2132bd2 has
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

[Comparison](../../../plan/exl3/archive/F3-TGMTP-KYOJIN-COMPARISON.json).

## TG/MTP routed and HC optimization, November 2, 2026

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

Prefill remains frozen at the batch6 evidence below; no new prefill speed or
full-corpus collector claim is made for this TG/MTP batch.

Image `ai-tools/gufo-eval:f3-tgmtp-9fc9a496-rocm101`; as-built `0ff1cb20-tgmtp-9fc9a496`.
Use `/opt/gufo-tgmtp/bin/gufo` (the image entrypoint); the inherited legacy
`/opt/gufo/src/build/release/gufo` is an older binary. Installed ELF SHA256
`f4dca2bca9efe4da63e8e1309f9b15dd31792c5460d5e27cff35ae19740578d5` matches the qualified mounted candidate.


[Qualification and identities](../../../plan/exl3/archive/F3-TGMTP-QUALIFICATION.json),
[operator screens](../../../plan/exl3/archive/F3-TGMTP-SCREENS.json),
[flow review](../../../plan/exl3/archive/F3-TGMTP-FLOW.json).

## HC/MoE combined qualification, November 2, 2026

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
[Qualification, runtime/code identities and all attempts](../../../plan/exl3/archive/F3-PREFILL-BATCH6-QUALIFICATION.json).

## EXL3 shared-MoE/KVarN qualified batch, November 2, 2026

Previous qualified image was `ai-tools/gufo-eval:f3-shared5-703d3e77-rocm101`
(`2184da9f…`), code `da1fd6a8`. Nix ROCm10.1, k4v4, speculative off,
temperature0, seed314159, r2; saved baseline reused.

| Case | Prior qualified tok/s | Shared5 tok/s | Change |
|---|---:|---:|---:|
| pp512 | 1063.05 ±.22 | 1068.58 ±3.09 | +.52% |
| pp2048 | 1416.51 ±.82 | 1429.13 ±1.13 | +.89% |
| pp8192 | 1447.47 ±1.36 | 1460.97 ±.73 | +.93% |
| pp2048 at d32768 | 1291.03 ±1.09 | 1301.99 ±1.41 | +.85% |

[Qualification and exact scope](../../../plan/exl3/archive/F3-STREAM-QUALIFICATION.json).
The later routed grouped-gate/up screen was not promoted and did not run a
model benchmark; this is the prior batch. Goal2000 tok/s is unmet.
A subsequent user-requested current-binary profile measured1426.48tok/s under
tracing at context4096, versus8193 here. This is advisory and does not replace
the clean table. [Bottleneck profile](../../../plan/exl3/archive/F3-CURRENT-PREFILL-PROFILE.json).

## EXL3 5.05 dense traversal follow-up, November 1, 2026

Earlier qualified image: `ai-tools/gufo-eval:f3-prefill3-67a50024-rocm101`
(`31fd777e…`). Nix ROCm10.1, KVarN k4v4, speculative off, temperature0,
seed314159, r2. Stored baselines reused; no new baseline inference.

| Case | Starting batch tok/s | Current tok/s | Change |
|---|---:|---:|---:|
| pp512 | 1061.85 ±.37 | 1063.05 ±.22 | +.11% |
| pp2048 | 1400.15 ±.83 | 1416.51 ±.82 | +1.17% |
| pp8192 | 1442.28 ±1.13 | 1447.47 ±1.36 | +.36% |
| pp2048 at d32768 | 1280.80 ±3.23 | 1291.03 ±1.09 | +.80% |

Only singleton K7 6144→2560 at2048 rows changes grid traversal. The actual
release object passed saved exact-output/guard comparisons. Batch512 model
collector evidence is reused because it cannot enter the new2048 predicate.
[Full scopes and qualification](../../../plan/exl3/archive/F3-PREFILL-GRID.md).
The2000 tok/s goal remains unmet; earlier batches below are historical.

## EXL3 5.05 narrow dense/output-slice batch, November 1, 2026

Prior batch2 qualified prefill: Nix ROCm10.1, KVarN k4v4, speculative off,
temperature0, seed314159, r2. One candidate benchmark pass; prior saved
baseline reused. Accuracy payload remained exact. Goal2000 tok/s is unmet.

| Case | Prior batch tok/s | Current tok/s | Change |
|---|---:|---:|---:|
| pp512 | 1061.85 ±.37 | 1062.82 ±.08 | +.09% |
| pp2048 | 1400.15 ±.83 | 1405.25 ±.83 | +.36% |
| pp8192 | 1442.28 ±1.13 | 1446.11 ±1.33 | +.27% |
| pp2048 at d32768 | 1280.80 ±3.23 | 1282.89 ±2.18 | +.16% |

Image `ai-tools/gufo-eval:f3-prefill2-f497677d-rocm101-r1` (`d256d2ab…`).
[Qualification and scope](../../../plan/exl3/archive/F3-PREFILL-BATCH2.md).
The earlier matrix screens below are historical; their narrow candidate is
now qualified in this batch. Completed TG/MTP work is retained.

## EXL3 5.05 prefill batch, November 1, 2026

Nix ROCm10.1, GCC16.2/Clang24, gfx1151; KVarN k4v4, speculative off,
temperature0, seed 314159. One candidate r2 window, saved pre-batch baseline reused.

| Prefill | Saved pre-batch (tok/s) | Candidate (tok/s) | Change |
|---|---:|---:|---:|
| pp512 | 1039.61 ± 0.73 | 1061.85 ± 0.37 | +2.14% |
| pp2048 | 1372.10 ± 0.11 | 1400.15 ± 0.83 | +2.04% |
| pp8192 | 1411.52 ± 1.17 | 1442.28 ± 1.13 | +2.18% |
| pp2048 @ d32768 | 1257.46 ± 0.39 | 1280.80 ± 3.23 | +1.86% |

`fn-prefill-batch-nix101` is healthy and complete. Contexts, prefill warmups and
case order match saved `fn-final-t1-x-ar-nix101`; generation cases were omitted.
These are observed batch-level gains; individual changes were not timed separately.
[Build and exactness evidence](../../../plan/exl3/archive/F3-RESUME-RESULTS.md#qualified-prefill-batch-2026-11-01).

## Earlier GGUF benchmark suite

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target
with shared-Q8_0 MTP; adaptive drafting. HTTP, greedy, thinking off.
Gufo `fec4759d` (October 18, 2026); llama.cpp `b11069` AR and `6fcaa16f` MTP
(September 22–23). Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1801.01 | 489.59 | +267.9% | 28.29 | 22.20 | +27.4% |
| 4,096 | 1727.95 | 455.24 | +279.6% | 28.27 | 21.21 | +33.3% |
| 8,192 | 1721.41 | 428.85 | +301.4% | 28.27 | 20.40 | +38.6% |
| 12,288 | 1710.64 | 400.21 | +327.4% | 28.21 | 19.64 | +43.6% |
| 16,384 | 1706.86 | 375.89 | +354.1% | 28.25 | 18.95 | +49.1% |
| 32,768 | 1672.60 | 301.31 | +455.1% | 28.18 | 16.54 | +70.4% |
| 65,536 | 1602.93 | 221.94 | +622.2% | 28.09 | 11.62 | +141.7% |
| 131,072 | 1557.18 | 144.78 | +975.5% | 27.85 | 7.98 | +249.0% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up.

<!-- bench:single-mtp -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1734.96 ± 0.85 | 468.76 | +270.1% | 35.79 | 31.73 | +12.8% | 116.93 ± 0.10 | 48.12 | +143.0% |
| 4,096 | 1676.83 ± 1.73 | 423.79 | +295.7% | 37.64 | 34.48 | +9.2% | 116.41 ± 0.07 | 45.71 | +154.7% |
| 8,192 | 1669.91 | 394.98 | +322.8% | 36.91 | 32.20 | +14.6% | 114.08 ± 0.01 | 44.31 | +157.5% |
| 12,288 | 1660.54 ± 1.22 | 365.91 | +353.8% | 39.28 | 32.32 | +21.5% | 105.48 ± 0.15 | 43.53 | +142.3% |
| 16,384 | 1635.18 | 341.98 | +378.2% | 40.54 | 32.07 | +26.4% | 113.32 ± 0.23 | 43.66 | +159.6% |
| 32,768 | 1618.63 | 277.61 | +483.1% | 39.97 | 25.41 | +57.3% | 105.38 ± 0.17 | 39.28 | +168.3% |
| 65,536 | 1556.32 | 208.04 | +648.1% | 39.23 | 19.65 | +99.6% | 104.78 ± 0.21 | 27.75 | +277.6% |
| 131,072 | 1481.19 | 135.42 | +993.8% | 39.87 | 14.63 | +172.5% | 108.37 | 20.96 | +417.0% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
llama.cpp re-evaluates its four-token checkpoint tail.

<!-- bench:multi-ar -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 28.24 | 22.34 | +26.4% |
| 2 | 50.66 | 37.08 | +36.6% |
| 4 | 86.01 | 54.82 | +56.9% |
| 6 | 108.34 | 65.34 | +65.8% |
| 8 | 125.48 | 68.79 | +82.4% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table.

<!-- bench:multi-mtp -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 35.74 | 31.77 | +12.5% | 116.50 | 46.93 | +148.2% |
| 2 | 61.30 | 42.79 | +43.3% | 165.50 | 52.57 | +214.8% |
| 4 | 88.77 | 48.78 | +82.0% | 218.12 | 48.23 | +352.2% |
| 6 | 109.74 | 52.59 | +108.7% | 231.85 | 48.51 | +377.9% |
| 8 | 127.02 | 61.92 | +105.1% | 245.81 | 58.13 | +322.9% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

## Loading time

C1, context capacity 262144, MTP. Cold target/sidecar files to HTTP readiness.

<!-- bench:loading -->
| Flash-Next Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 16.42 | 117.83 | +617.6% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 133121, AR. Peak memory reported by HIP.

<!-- bench:memory -->
| Flash-Next Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 88.78 | 84.84 | -4.4% |
| 16K prefix, pp4096 + tg128 | 90.53 | 85.31 | -5.8% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

## HC combine operator screen (subsequently qualified above), November 2, 2026

At the operator-screen checkpoint, qualified pp2048 was1429.13±1.13tok/s. The source-selected HC operand-load
change reduces one synthetic2048-row fused combine screen by8.87% (plain)
and8.93% (after256MiB write pressure); these are not model measurements.
No model baseline or full profile was repeated. Preserve saved controls and
Nix production objects for the next combined qualification.
[Scope, both variants and receipts](../../../plan/exl3/archive/F3-HC-LOAD-SCREENS.json).


## EXL3 conversion capture screen, November 5, 2026

These are conversion-only wall times, not serving prefill/decode benchmarks.
One saved baseline; possible first-use library/allocator warmup means the
percentage is provisional until complete conversion timings are available.

| Block | Tokens | Original capture (s) | Dead-down elision (s) | Reduction |
| --- | ---: | ---: | ---: | ---: |
| GDN | 512 | 1.37779 | 0.52963 | 61.56% |
| Full attention | 512 | 1.21895 | 0.53058 | 56.47% |
| GDN | 8192 | 33.58238 | 24.19120 | 27.96% |
| Full attention | 8192 | 35.01159 | 23.78136 | 32.08% |

All captured statistics and normal outputs are byte-exact. The unchanged first
attention conversion completed in4886.74s before adoption; later block timings
will be logged without quant-worker synchronization. [Scope and receipts](../../../plan/exl3/archive/F3-CONVERSION-WATCH.md).
