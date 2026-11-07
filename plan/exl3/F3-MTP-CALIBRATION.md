# c03 MTP calibration and final component changes

**현재 상태 (2026-11-07):** c03 본체·보정 MTP·원본 BF16 비전 조립과 최종 검증이 완료됐다. 사용자 승인으로 c03를 상시 서비스 중이며 변환 감시는 종료했다. 아래는 날짜별 계획·실행 이력이다. [최종 결과](F3-MTP-CALIBRATION.md).

2026-11-06 user decisions: keep original BF16 vision; use mixed MTP bitrates
matching the main full-attention block; calibrate MTP too. The running main
converter remains frozen. These decisions supersede the earlier uncalibrated
K4 MTP / K8 vision plan, but are not yet a statement of production integration.

## Boundary and preserved input

- Main idx50 is Layer47; checkpoint idx51 contains its pre-final-mixer HC output.
- Reserve GPU then JOB locks after Layer47 finishes, before idx51 mixer runs.
  Do not signal the converter, replay completed main layers, or delete its state.
- Run calibrated MTP before resuming main mixer/head. This avoids retaining an
  extra 76.275GiB HC checkpoint through final packing.
- Separate MTP work/progress and artifacts from main. Reuse successful stages on
  retry and protect main checkpoint/IDs, original BF16 source and baseline data.
- Boundary helper: `mtp_calibration_boundary.py`; receipts:
  `flash-next-quant-20261104/mtp-calibration-boundary/`.

## Numerical contract

- Use all 2,309 c03 rows / 1,999,496 tokens, same row order. No quarter subset.
- Match Gufo executor/reference semantics: retain FP32 pre-mixer HC states;
  pair state[:, :-1] with token IDs[:, 1:], normalize across the complete HC*H
  row. There is no synthetic zero-prefix MTP step in Gufo. All 2,309 documents
  remain; 1,997,187 valid next-token pairs are calibrated. This is teacher-forced
  calibration, not coverage of all multi-step self-generated draft distributions.
- The initial candidate followed pinned Kyojin semantics (per-stream norm and
  FP16 zero-prefix tap); it is historical CPU evidence and is NOT qualified for
  Gufo. The corrected adapter must preserve the FP32 input until EXL3 rotation
  and round rotated operands to FP16, retaining FP32 output.
- Input `fc_hidden` and `fc_embedding` currently share `block.mtp.fc`. Assign
  distinct capture maps before collecting H: their inputs are different.
- Expected hidden FC H count: 4 × 1,997,187 = 7,988,748; embedding FC and decoder
  H count: 1,997,187. Require finite H and inputs; reject missing/partial counts
  and uncalibrated fallback rather than silently reducing input.
- Original FP32 BMM is an explicit input-FC backend for new 4(T-1) shapes.
  Decoder rows are now T-1, so the old Lt allowlist is insufficient; qualify the
  required lengths in a bounded operator check before adding a new contract.
  Do not silently fall back or claim that old T-shape tests qualify T-1.
- The decoder activates every routed expert only for H capture; normal reference
  outputs use original routing. Preserve at least first five reference rows.
- Quantize input FCs then advance their output into separate disk rows; use those
  states for decoder calibration. Do not write all decoder output states solely
  for the terminal mixer, which has no quantization targets.

## Bitrates and vision

MTP: routed experts K4; shared experts K6; Q/O K6; K/V/indexer K8;
`fc_hidden` and `fc_embedding` K8. All 1,546 per-linear entries are recorded in
`MTP-MIXED-BITS-PROPOSAL.json`. The original allocator ignores per-key values for
`qbits_key=mtp_bits`, so an audited explicit mapping is required.

Embedding BF16 and output head K8 remain shared with main. Norm/HC/router tensors
retain source precision. Rotation and EXL3 trellis encoding remain enabled.

Vision: `vision_bits=16` omits vision quantization, and the existing compiler
copies original tensors with `allow_bf16=True`. Final audit must check all 333
vision tensors (448,931,056 parameters, 897,862,112 bytes): keys, shapes, dtypes
and exact payload against the original. Do not report the old K8 metadata.

## Verification and completion

CPU: actual key coverage/bit mapping, distinct H owners and count multipliers,
token alignment/no cross-document carry, failure on nonfinite/incomplete input,
resume/artifact identity, raw-vision copy path. Independent review required.

At the reserved boundary: validate GPU cleanup/main checkpoint and use bounded
small real-input capture/quantization tests before full MTP conversion. Keep
saved controls and retry only failed checks. The current main baseline is not
rerun. Preserve previous freeze; integrate only reviewed final-stage changes.

After MTP succeeds, prevent the old uncalibrated side loop from overwriting its
artifacts, apply vision16, compile, and record actual mixed-bit/calibration
metadata. Final text/image/MTP correctness and acceptance/throughput checks run
together after pack completion. Only then stop the hourly watch.

## User priority: MTP must improve end-to-end speed

The user explicitly emphasized MTP latency. Acceptance rate alone is not the
adoption criterion. At final qualification compare MTP off, the applicable K4
reference, and the mixed-bit candidate under the same completed main model,
hardware/runtime, prompts/cache history, KV settings, concurrency, sampling
and draft length. Reuse a saved baseline only if those conditions match; never
compare a different main pack and attribute the difference to MTP bitrates.

Record final emitted tokens / whole decode wall time (including draft,
verification, rejection/rollback and synchronization), draft latency,
verification latency, accepted/proposed counts, request latency and peak memory.
Check exact greedy verification and sampled correctness as well as speed.
Batch these into the existing final validation window. Do not silently select
a slower mixed-bit configuration just for its higher acceptance rate. If timing
is indistinguishable, report it as such and consider quality; if slower, revisit
the costly projections and use stored evidence for only the affected comparison.

2026-11-06 follow-up: the user accepts lower **draft prediction accuracy** when
faster drafts more than offset extra verification/recovery cost and improve final
token throughput. Apply this tradeoff to FP32 versus FP16 choices too, not only
bitrate. Preserve the main model's verification semantics; draft acceptance and
final output correctness are separate measures. Calibration Gram throughput is
one-time conversion cost, not evidence of inference acceleration. Keep original
strict numeric differences recorded, then judge practical candidates using the
combined timing/correctness window above. No separate long benchmark per tiny
precision change, and no automatic choice of FP32 merely for closer intermediates.

Source-header weight-count calculation: raising the ten non-routed-expert MTP
projections from uniform K4 to this recipe adds **20.703125 MiB of ideal packed
weight payload**, excluding scales/padding/format overhead. This does not predict
latency: these dense projections execute on every draft token, while only the
selected routed experts execute. In particular `fc_hidden` acts on four HC
streams. Profile draft/verification costs before claiming that the small total
storage increment is negligible for speed.

Persist pre-quantization H and reference outputs with exact source/IDs/state,
math/backend identity and counters. Reusing H for alternate decoder bitrates
requires unchanged actual input-combine output states. If input FC bits change,
the resulting decoder inputs change too; do not label the old H equivalent.
Alternate recipes require an explicit recorded comparison decision. Saving H
is a means to avoid repeated collection, not permission to ignore dependencies.

For any later input-FC bitrate alternatives, preserve or archive the protected
idx51 pre-mixer HC input until those alternatives are resolved; otherwise changing
FCs after main mixer advancement would require main replay. H caches alone cannot
reconstruct changed decoder inputs. Audit disk admission before retaining/copying
these large states. Actual candidate footprint: derived rows ~76.187 GiB plus
~4.05 GiB for H and duplicated stage references, excluding packs/probe/scratch.


2026-11-06 full calibration completed and independently reviewed:
input 1441.10s; decoder 1364.01s; all2309 rows/1997187 pairs;
input H counts7988748/1997187 and decoder516H each1997187; no nonfinite
H/reference/packed values. Actual1546 projection headers match K4:1536/K6:5/K8:5.
First5 decoder calibration-reference RFN0.03711–0.05571 is local block error,
not independent whole-model quality, draft acceptance or final throughput.
Evidence: `mtp-full-decoder-review/{REVIEW,CPU-AUDIT,LOG-AUDIT}.json` under run output.

Original idx51 now independently preserved at `mtp-protected-idx51/ckpt` via
immutable row hardlinks with independently copied metadata, no new76GiB payload.
After independent audit, only2309 completed derived state payloads were retired;
manifest, all H/refs/packs and input/decoder provenance remain. Future input-FC
variants need explicit replay from protectedidx51; changing FCs invalidates direct
reuse of old decoder H. See `mtp-full-decoder-review/DERIVED-RETIREMENT.json`.
Main-only mixer/head completion and final mixedMTP/BF16vision assembly remain;
whole inference correctness and draft+verify+recovery token/s still pending.

## Current final validation, 2026-11-07

User explicitly approved final assembly followed by validation. Main idx53 is
complete; final001 assembly took159.31s with mixed1546 MTP projections exact,
333 original BF16 vision tensors exact, protected artifacts unchanged and GPU
cleanup zero. No further H collection, quantization or baseline generation.
Final pack: `ai.tools/tmp/test-results/flash-next-quant-20261104/`
`mtp-final-assembly/data/final001/pack/`. Source-only provenance documents are
labelled separately under `source-provenance/`; they do not hash the new shards.

The selected implementation remains the existing native EXL3 vision production
image `e7a761a40187` with pinned Nix ROCm10.1 and retained object inputs.
No new kernel specialization, precision shortcut or quarter-calibration is
introduced. Quality collector compatibility is audited against those exact
objects/libraries; a small relink is preferred to a full rebuild if necessary.

One bounded smoke checks C1/4096 FP16 KV, no YaRN, Korean/English text,
MTP off/on7 and native BF16 image behavior. HTTP text/finish/count equality and
bench emitted-token digest equality are distinct evidence; HTTP does not expose
raw token IDs. Once smoke passes, the combined window measures PP2048/8192,
TG128 at0/32768 history, maximum draft7 with the existing adaptive policy, and representative fixed Korean/English/
code/tool requests at C1/C2 with32768 context and ~1.8GiB shared FP16 pool.
A short C2 run does not qualify two fully populated256K sessions.
Save the first matching c03 MTP-off control for reuse; old different-main speed
numbers are historical only. No matched uniform-K4 or alternate FP16/FP32 draft
arithmetic artifact is available, so those alternatives remain explicitly untested.

Quality uses the retained actual HF BF16 prefill reference on32760 positions,
8 evaluated4096-token windows, with c03 MTP off. Independence from c03 calibration is not established. Its101424960B payload SHA
was checked once on November7 and matches de728d88…; do not regenerate it.
Report deltaNLL, full-vocabulary top1, KL256+rest lower bound/coverage, tails,
and token/block/window confidence intervals. Saved UD-Q4 records can provide
historical numerical context, not a matched current-runtime timing control.
No saved full BF16 decode reference is claimed. Collector diagnostic replay
cost is excluded from speed claims.

Fresh exclusive GPU/JOB ownership per window, no conversion/assembly overlap;
models/reference mounts readonly, bounded private outputs,100GiB disk runtime
reserve and150GiB admission, host8GiB warning/4GiB floor. Reclaimed unused
Docker buildcache older24h before tests: rootfree about186.5GiB. No model,
checkpoint, H, packed baseline or final output was deleted. Final qualification
and watch completion require actual reviewed results, not assembly alone.

### Runtime selection corrected after actual load

The selected historical nativevision image e7a761a40187 was stale: smoke002
rejected the valid K8 lm_head. Actual ELF disassembly confirms loaderK3..7,
Exl3FastSupported/GEMV bits1..7, no K8 kernel instantiations. Current repository
commit2095529d supplies K8 loader and GPU decode/GEMV/prefill/MoE; a coherent
Nix10.1 runtime/collector is now required before final qualification. A loader
bound-only patch is unsafe. The existing refactor-gpu build is ROCm7.2.4 and is
not substituted for the user's requested10.1 runtime. Failed001/002 are retained;
no completed main/calibration/quantization is rerun and model metadata is valid.


### Completed numerical qualification, November 7, 2026

A coherent current-source a81c0f93 build replaced the obsolete image-contained
ELF: Gufo6edf0d44 and collectorbc989832, both using pinned Nix ROCm10.1.
The container base remains e7a761a40187; it does not identify the running Gufo.
319 build steps, format/loader CPU checks and runtime library identity passed.

| Saved-reference comparison | HF BF16 | New c03 EXL3 | Historical UD-Q4 |
| --- | ---: | ---: | ---: |
| Mean next-token NLL (lower is better) | 2.172846 | 2.162650 | 2.179499 |
| KL256+rest vs BF16 (lower is closer) | 0 | 0.041522 | 0.039227 |
| Highest-probability token agreement with BF16 | 100% | 90.8394% | 90.8608% |

The corpus is eight project-document/code/synthetic-document-chat windows,
32760 next-token positions, not an independent general-task benchmark.
Top256 reference probability mass averages97.03%; KL256+rest is a lower bound,
not full-vocabulary KL. Native HF cache dtype was not recorded; c03 KV is FP16.
The negative c03 deltaNLL has window95%CI[-0.024874,+0.000487], so this does
not establish general superiority over the unquantized model. Compared with
saved UD, the net top1 difference is only seven positions and KL/top1 paired
block/window intervals include zero. Historical UD also used a different
ROCm7.2.4/GCC13 runtime, preventing a quantization-only causal interpretation.

All64 native head replay checks are bitwise exact. c03 collection155.37s and
CPU scoring5.14s completed with no cleanup errors, unchanged pack and no
baseline regeneration. Source inputs/checkpoints and calibration H remain.
Evidence: final-validation/quality-current-001/QUALITY-RESULT-REVIEW.json,
historical-comparison/COMPARISON.json and FINAL-QUALITY-DRAFT-KO.json.

HTTP ten-request smoke passed text/arithmetic/BF16 vision/image-history and
prefix/branch cache behavior in both modes, with identical output text/counts.
The standalone32-token emitted-ID hash also matches with actual MTP proposals.
The original smoke003 parser-failure receipt is retained; the supplemental
run reused its saved off result and executed only the missing MTP case.
This checks observed RAM prefix reuse, not physical zero-copy GPU KV sharing.
Final longer throughput/equivalence validation completed as recorded below.


### Completed throughput and final disposition, November 7, 2026

Same c03 pack, coherent current Nix10.1 build, FP16 KV, no YaRN. MTP uses
maximum7 drafts and the existing adaptive policy. Native bench rows are inner
prefill/decode-loop rates (two repeats), not HTTP wall time. HTTP rows include
prefill/queue/transport and sum the emitted tokens of four fixed prompts.

| Measured scope | MTP off tok/s | Mixed MTP tok/s |
| --- | ---: | ---: |
| Native PP2048 | 1460.97 ± 0.07 | 1370.97 ± 1.56 |
| Native PP8192 | 1501.37 ± 1.56 | 1403.43 ± 1.60 |
| Native TG128, depth0 | 27.33 ± 0.03 | 50.70 ± 0.35 |
| Native TG128, depth32768 | 26.72 ± 0.01 | 87.30 ± 0.05 |
| HTTP C1, four greedy prompts, group wall | 26.13 | 41.70 |
| HTTP C2, four greedy prompts, aggregate group wall | 36.56 | 45.55 |

Every native final-token digest matches off/on at both depths and repetitions.
All four greedy HTTP outputs match the C1-off reference across C1/C2 and off/on.
The32K native sample accepted every draft, so its3.27x ratio is not a general
long-context forecast. C2 rates are total throughput, not each user's rate.
Sampled512-limit groups observed C1 26.81→41.97 and C2 37.53→50.08tok/s,
but differing output text/lengths prevent an identical-work speedup claim.
Sequential off-then-on order and sparse thermal readings limit small timing
conclusions; EPP samples remained balance_performance, not a continuous power
or temperature guarantee. No old external benchmark was rerun or replaced.

Observed interactive benefit supports retaining the calibrated mixed MTP,
with a measured6–7% prefill cost. This does not choose a nonexistent matched
uniform-K4 or alternate FP16/FP32 arithmetic candidate. All eight cases cleaned
up with no returned KFD contexts, unchanged pack and minimum39.65GiB available
host RAM under this32K configuration. Full256K/512K occupancy remains untested.

Final receipt: output/FINAL-QUALIFICATION.json, qualified_for_measured_scope.
Final throughput independent review: final-validation/throughput-001/RESULT-REVIEW.json.
Hourly watch stopped/disabled after completion; temporary servers are cleaned
up. No persistent service or default-model switch was performed. All successful
artifacts, checkpoints, baselines and incremental Nix build outputs are retained.
