# F3 resume, 2026-11-01

Source baseline: `966f84b8`, Docker `ai-tools/gufo-eval:64849ca6ec7b`,
ROCm 7.2.4, RelWithDebInfo. These probes are not final Nix qualification.

## Baseline T1 probe

EXL3 5.05, KVarN k4v4, speculative off, seed 314159, temperature 0,
one repetition, d0, pp2048/pp8192 and tg128:

| Measurement | tok/s |
|---|---:|
| pp2048 | 971.66 |
| pp8192 | 1354.23 |
| tg128 | 24.87 |

Load took 29.9852 s. Health checks passed. The lower pp2048 result versus
the handoff's historical probes is unexplained; do not treat those historical
runs as matched controls or claim a regression from this comparison alone.
Binary SHA256: `537bf7b38ef1d397271ea3e5af0795cf1ae82e2bf4f6afba96bcfa603c285e55`.
Preserved binary: `build/f3w3b/gufo-966f84b8-probe`.

Receipts under `/home/m2on/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/`:
- `fn-resume-966f84b8`: failed before model launch because `.gufo-revision`
  was missing; health passed. Retained as setup-failure evidence.
- `fn-resume-966f84b8-stamped`: successful baseline probe.

## Shared indexer slices

Sparse EXL3 prefill now computes the shared indexer parent once, copying
both Q/K slices before reusing scratch. Actual projection count remains one
for chunk sizing, rotation, GEMM and diagnostics. Other paths are unchanged.
An independent reviewer approved the change. The focused dense-prefill unit
passes exact full-parent byte comparisons, chunk sizes 1/9/33/64/65, output
guards, projection counters and rejection of distinct-parent sliced groups.

Builds pass. Modified files pass clang-format 23.1.3 and `git diff --check`.
The full repository formatting command fails on untouched `json_constraint.cpp`,
`openai_chat_test.cpp`, `deepseek_v4_flash/serving_test.cpp` and
`qwen/cpu/generator_test.cpp`; host Nix is unavailable. This was superseded
by the pinned Nix container formatting check: PASS, all 758 C++ files.

Combined collector/timing evidence is retained in `fn-indexer` (COMPLETE,
health_PASS). Both FP16 KV and KVarN collectors have 32,760 records and are
byte-identical to their stored hc7 controls. Candidate probe: pp2048 1285.56,
pp8192 1339.10, tg128 25.04 tok/s. Do not attribute the apparent pp2048 gain
to this change: sparse indexer reuse activates only beyond 2048 tokens.
The baseline pp2048 anomaly remains unexplained; no duplicate baseline run
was performed. pp8192 differs by about -1.1%, below the timing gate.
Its `source.patch` identifies the candidate relative to `966f84b8`; the
collector driver records binary hashes. This is Docker 7.2.4 evidence only.
No merge or push has been made; Nix candidate images are listed below.

## Nix ROCm 10.1 transition

The user requested Nix images and ROCm 10.1, with model validation batched
at the end and stored baselines reused. The existing flake already pins
ROCm 10.1.0 / HIP Clang 24. Reuse volume `gufo-nix` and the existing image
packaging pattern. Build snapshot/recipes: `build/f3-nix-resume/`.
Evaluated toolchain: gcc-wrapper-16.2.0, HIP Clang 24, ROCm 10.1, gfx1151.
Candidate patch SHA256:
`b03a1f1eb451fe06d91b48478e45c6b4387c4bec97b08d6761444dbbb93ac34f`.
Final q3/q6 and FN/27B timing remain pending; no per-item model windows
should be added. Reuse stored Nix UD timings while its code path is unchanged.

Production and focused test Nix builds passed. Candidate images:

- `ai-tools/gufo-eval:f3-indexer-b03a1f1e-rocm101`, image ID
  `sha256:9dd41fa63b576c077de1b9a8f5d4750e4550f53845beb01039a22078bbc8dfbd`.
- `ai-tools/gufo-eval:f3-indexer-b03a1f1e-tests-rocm101`, image ID
  `sha256:92764890d1bfbfba12a31bbd5606390ddaa83ea5fc719fa66838797092b0f88b`.

Release output: `/nix/store/5q39q6sw7h2nqp8nl72zsiv1lj1r7iif-gufo-0-unstable-dirty`.
Tests output: `/nix/store/vp64rvqzgmzflpqivnvrmnq096m3yal8-gufo-f3-indexer-tests-0-unstable-dirty`.
Tests include exl3_ops, rq_logits, session_test and gpu_probe. All four
wrappers select patched `rocm-runtime-10.1`. Image labels identify the base
commit and patch; the path-flake binary version itself reports `dirty`.

The inherited base image set `LD_LIBRARY_PATH=/opt/rocm/lib`, which makes
even a Nix binary load HIP/BLAS 7.2.4. The first Nix unit receipt
`fn-indexer-nix-unit` therefore records mixed runtime evidence, not pure 10.1.
Both candidate images now set `LD_LIBRARY_PATH` empty; their wrappers prepend
only the patched ROCr 10.1 directory. `ldd` confirms clr, hipblas, rocblas and
hipblaslt all resolve to Nix 10.1 paths. The affected focused unit was rerun
once in `fn-indexer-nix-unit-runtime101`: COMPLETE, health_PASS.
No model or timing rerun was added for this environment correction.

Historical unit plans explicitly override LD_LIBRARY_PATH, but the historical
timing plans require an actual receipt/environment audit before claiming
runtime equivalence. Preserve and reuse matching baselines; do not automatically
invalidate or rerun all prior measurements merely from the base image default.

Packaging initially failed with the Nix volume read-only; writable mounting
restored closure lookup. Only this task's four temporary Docker input-cache
records were pruned after packaging (5.99 GB); images and evidence remain.
Free root space recovered to about 151.4 GiB. Scripts and contexts retain the
exact recipes, including the environment-only overlay that needs no rebuild.

## q3 and merge audit (2026-11-01)

`fn-final-q3-indexer-nix101` stopped on its first failing subtest after
154.85 s; health_PASS. Batch C2/C4/C6/C8 AR/MTP logits and RNG, isolated
state, lookup C2..C8, snapshots, cancellation and restores passed. The
draft-vocabulary lifecycle then failed at session_test.cpp's requirement
that the free-running Thai continuation contain at least one-third rare
token IDs. It had already passed the checks that MTP drafting occurred and
that every drafting cycle used the small head. No output token dump was
recorded, so a fixture assumption versus model/predictor defect remains
unresolved. Remaining q3 subtests were not run; no tolerance was relaxed.

At the user's request, audited the suspected loss of prior ROCm 10.1 work:
`b15a040a` (Clang 24 direct WMMA), `c796ee2f` (spill-free verification steps),
`9974cb4c` (Nix ROCm10.1-only), `05e527d2` and `1d7b729e` are ancestors of
f3-integ, port-u4f and integ. Both optimization implementations remain in
the source. The running q3 process's /proc maps showed clr, ROCr, hipblas,
rocblas and hipblaslt all from Nix 10.1 store paths.

Non-checkout merge-tree previews: merging integ into f3-integ produces the
identical committed tree; merging port-u4f into f3-integ changes four plan
documents only. Neither adds missing code/build settings. This does not
perform an actual merge and excludes unrelated uncommitted work in port-u4f.
Formal F3 integration remains pending, but no restoration merge is needed
to recover the prior 10.1 optimizations. Retain current test evidence by
source/patch/binary identity rather than rerunning solely for a merge label.

## Draft-vocabulary fixture diagnosis

Focused `fn-draft-diagnostic-nix101` (37.41 s, health_PASS) recorded the
same 19 output IDs for autoregressive and speculative decoding. The
130-token raw Thai prompt produced six rare tokens, then both paths
emitted `<think>` and "The user has repeated the same Thai sentence multiple
times:". Six of nineteen is below the unchanged one-third coverage gate.
This is evidence of an unmet continuation-fixture assumption, not evidence
of a speculative token mismatch. The sparse indexer path is unreachable
at this prompt length.

The test now exposes `--draft-vocab-only` to resume just lifecycle and sampled
draft-head checks, with their required concurrency of one and two. Failure
diagnostics retain output text/IDs and an independent AR token comparison.
The revised fixed fixture uses the existing non-thinking chat template to
request 100 repetitions, then prefills complete copies as assistant output.
It retains the old minimum repeated history, miss-window coverage, and all
rarity/head-use/acceptance/restore assertions. The diagnostic and fixture
changes were independently reviewed. Production code was not changed by
this test repair.

`fn-draft-fixture-nix101` completed with health_PASS:

- EXL3 5.05: PASS, 123.67 s.
- GGUF UD-Q4: PASS, 65.00 s.
- Both exercised 3/3 actual neural drafting cycles through the small head
  and 38 lookup copies; all original rare-output and acceptance gates passed.
  Used/fresh table restores and mixed batches were exact. Sampled-head
  schedules, restores, reordered batches and greedy output equivalence passed.
- Pinned Nix formatting: PASS, 758 files. Independent review: approved.

Test patch SHA256: `80fc029c6392c547979c665ee36b079a6dfb0d73439af206842e51527c6b6928`.
Binary SHA256: `9b3232a98e28ecff9dde07fe60e16033fd5ce145e4c3fec7526eca99c9c48d89`.
Image: `ai-tools/gufo-eval:f3-indexer-draftfixture-rocm101`, ID
`sha256:77237ac95067ac9177006a5ae6790ea21266d0a1437410c0979aef482f2dc458`.
Build recipe/snapshot: `build/f3-nix-draft-fix/`; only session_test.cpp differs
from the original Nix source snapshot. The overlay adds only the new test
derivation; runtime dependencies remain unchanged at Nix ROCm 10.1.

The original failed receipts remain immutable. Its passed batch-prefix checks
plus this focused lifecycle/sampled-head tail cover the first q3 batch item;
do not rerun that whole item. Remaining q3: sampling, prefill, execution32k,
MTP audit, engine floor. Remaining overall: formal integration, q6, Nix FN/27B
timing and prefill/decode performance goals. No baseline timing was repeated.

## q3 remaining checks and EXL3 oracle repair

`fn-final-q3-remaining-nix101` retained three passes: sampling 147.47 s
(25 AR and MTP configurations, C2 and budgets exact), prefill 94.19 s
(chunk boundaries and prefetch hints exact), execution32k 84.60 s
(independent AR/MTP, mixed batches and RAM/serialized sampled replay exact).
The fourth check failed after 37.96 s: recursive carry relative RMS 0.0002127
exceeded the unchanged 0.00005 limit. Fusion was 0.0002149 and attention
0.0004285. Health passed; engine-floor had not run. This matches the older
unresolved EXL3 oracle symptom; by itself it does not establish a new GPU regression.

Independent review identified missing decode-storage semantics in the CPU
EXL3 oracle: it used ideal FP64 input rotation despite the GPU's FP32 producer
and FP16 rotated-input boundary. The default ideal reference is preserved;
only Storage::kDecode selects the explicit storage-aware formula, including
the split predictor FC calls. A dyadic-input CPU unit independently uses dense
Sylvester multiplication to verify the storage boundary and block permutations.
No existing numerical tolerances were changed. FFN stage errors are now logged.

The audit also now validates separate EXL3 FC trellis/scale uploads and uses
EXL3 full-head projection plus independent host sorting for exact candidate
IDs/logits, instead of applying the GGUF Q8 kernel to EXL3 storage. The GPU
inference implementation is unchanged by this repair. Review approved;
qualification pending. First build caught a missing include, now corrected;
failed build log/patch retained under `build/f3-nix-mtp-audit/`.

The pending engine-floor diagnostic uses fixed `The sky is`, batch32/context4096:
verify actual multirow token count, finite CPU/full-vocabulary comparisons and
completion. The probe has no numerical pass threshold; exit0 alone is not
quality qualification. This bounded prompt does not cover warmed histories or
chunk transitions; those remain covered by prefill/32K and upcoming q6.

The first storage-aware rerun, `fn-final-q3-audit-nix101`, stopped after
37.76 s (health_PASS): fusion improved to 0.0000012, but FFN input rounding
still propagated through the nonlinear/FP16 FFN boundary; carry 0.0000622
remained above 0.00005. Its failed receipt is retained. The next refinement
extends the existing matched-stage-input audit to the FFN input: independently
computed CPU FFN input is preserved and separately checked at relative RMS
<=0.000002, while the observed GPU input is supplied to the CPU FFN.
FFN output is never injected.
Thus the downstream comparison isolates the FFN/carry arithmetic; it is not
an end-to-end ideal-reference quality claim. All original thresholds remain.
This refinement and the new input gate were independently reviewed.

`fn-final-q3-mtpinput-nix101`: MTP audit PASS in 81.88 s. Eight text/image
sequence states passed; maximum recursive-carry relative RMS 0.0000471
(limit 0.00005), maximum printed FFN-input relative RMS 0.0000002
(limit 0.000002). Full-vocabulary candidate IDs/logits, cache chunks
1/8/9/32/33 with both KV-only settings, ragged/full short catch-up and
full/tail independent batches were exact. Sampling, prefill and execution32k
were reused from the earlier receipt, not repeated.

Final oracle patch SHA256:
`2b616033ee7b6b0ccf50d7864c92dececbbde09341839597267ee4a0a360570c`.
Snapshot/recipes: `build/f3-nix-mtp-input/`; image:
`ai-tools/gufo-eval:f3-indexer-mtpinput-rocm101`; Nix test output:
`/nix/store/22ihcdns0rwjwz16mibn275nx4xar0qw-gufo-f3-mtp-input-tests-0-unstable-dirty`.
CPU EXL3 reference unit passed after the storage repair; pinned formatting
passed (758 files) after the final stage-input change. No GPU inference
kernel or model weights changed during this oracle repair. Engine floor and window health completed as recorded below. q6 full-model
quality qualification remains pending.

Next-q6 preparation (no collection rerun): the existing BF16 and UD-Q4
`.bin`/`.json` prefixes are present under `final-fn/fn-bf16-prefill/receipt/run/`
and `final-fn/fn-q6a-ud-bf16/receipt/run/` respectively; both metadata files
cover eight 4095-scored-row windows (32760 total). Reuse these saved data.
`PLAN-FINAL-FN2-Q6B.json` still selects the old 7.2 collector and needs a new
Nix-10.1 candidate plan before execution. Its note calls the saved UD collector
Nix, but the UD metadata explicitly records a 7.2.4 build/runtime; preserve
that provenance and do not call it a matching Nix-10.1 control or timing
baseline without further evidence. Saved 68d4 Nix timing baselines are separate.


Final q3 resume completion: the same window completed with health_PASS and
no remaining container/KFD context or new GPU error. The engine-floor probe
completed in 604.27 s with exactly three prompt/logit rows:

| Position | CPU/GPU top-1 (both) | KL (printed) | max absolute logit difference |
|---|---:|---:|---:|
| 0 | 20438 | 0.00000 | 0.012 |
| 1 | 369 | 0.00001 | 0.029 |
| 2 | 279 | 0.00175 | 0.356 |

All metrics are finite and top-1 agrees 3/3. This is a bounded diagnostic,
not a formal numerical quality gate. CPU execution was much more costly
than estimated; do not routinely repeat it. Its log saves comparison metrics,
not full CPU reference vectors. The independent full-model q6 comparison will
reuse the already saved BF16 reference vectors rather than recomputing them.

Final image ID: `sha256:41ab9fd7d4c029d2dec6a492efe32c78b44e671ce512ad74de30ab370689639e`.
Probe binary SHA256: `4dc4c5def8d757c8acc611e426e01416d30a0ab0e86c726a611046a9fe6905fb`.
Actual loaded ROCm/rocBLAS/hipBLAS/CLR paths are saved in
`fn-final-q3-mtpinput-nix101/receipt/runtime-rocm-maps.txt`, all Nix 10.1.
The receipt's `q3-summary.json` indexes the preserved batch prefix/repaired
fixture tail, three prior passed checks, final MTP audit and bounded floor.
The q3 resume scope is complete. q6, Nix FN/27B timing, formal integration
and performance targets remain pending. No performance baseline was repeated.


## q6 Nix measurement and performance-priority decision

`fn-final-q6-nix101` completed, health_PASS. Exactly one candidate model
collection (154.01 s) produced 32760 records, eight complete 4095-row windows,
FP16 KV, batch512 and 64 exact head replay checks. Scoring/comparisons took
about13.3 s. BF16, UD and hc7 model inference were not repeated. Candidate
ELF SHA256 `e4da4316dbabf1949576d09c912137350aa6b4929bc167ef5bf7cdd626d3ab81`;
record bin SHA256 `789c46a2a6939dc980e23b2cbf78bac71ccbcd9b1472899be846a1d56584f2a0`.
Runtime maps confirm Nix ROCm10.1. Metadata HIP7.16.0 is the HIP API version,
not a fallback to ROCm7.2. The independent plan review approved the protocol.

| Against HF BF16 | Historical UD | Historical hc7 | Nix candidate |
|---|---:|---:|---:|
| KL256 | 0.039227 | 0.026434 | 0.026481 |
| Delta NLL | +0.006653 | -0.004244 | +0.002356 |
| Top-1 agreement | 90.8608% | 92.1245% | 92.0543% |
| Top-1 flips with KL>1 | 59 | 23 | 28 |

KL256 uses reference top256 plus a rest bucket, normalized over the full
vocabulary; it is not full KL. Mean reference mass in top256 is0.9703.
Candidate-minus-UD window95%CI: KL [-0.015466,-0.010096], top1
[+0.009126,+0.015781]. Candidate-minus-hc7 NLL is+0.006600,
window95%CI [+0.001729,+0.013902]: a measured increase, not dismissed as noise.
It corresponds to about+0.66% perplexity; top1 change is-0.0702 percentage
points and KL change+0.000047. Full tails/categories and paired intervals
are retained in the score/compare JSONs and `Q6-SUMMARY.json`.

Saved-record analysis (no inference) found differences already at token0,
before the sparse indexer path. Chat dominates the NLL increase; w7p3430
has the largest paired increase+8.38776. Removing the100 largest increases
still leaves mean+0.001953. Compatible older Nix EXL3 records also show
candidate-minus-old NLL+0.008746, but revisions differ. Do not assert a
specific kernel cause from these aggregate comparisons.

A reviewed isolated diagnostic captured layer0 HC down's320x10240 F16
matrix and compiled the exact QuantizeHalfQ8_0Kernel with both compiler
versions. Both ran under Nix10.1 libraries. The Nix output was retained from
`fn-q6-quant-diagnostic-nix101`; the legacy-compiler process initially failed
at glibc startup, then only that process resumed via the Nix loader in
`fn-q6-quant-legacy-nix101` (both health_PASS). Payloads are3481600 bytes;
2095 blocks/2153 int8 codes differ, zero scale bytes differ. Every difference
is an exact |v|=amax/2 tie: legacy +/-63, Nix +/-64. This establishes a
compiler-sensitive microkernel result, not the cause of full q6 drift.
Recipes/input/ISA: `build/f3-q6-quant-diagnostic/`. Regenerated .s files do
not fully explain the actual binary difference; no production repair was
applied. Preserve this unresolved distinction.

User direction after reviewing the small aggregate differences: performance
is important; do not sacrifice speed merely to eliminate small quality drift.
Stop further precision repair work and prioritize performance measurement.
q6 status is **measured with documented drift**, not an unconditional
no-regression PASS. Numerical differences and tails remain visible. No
production source changed during q6. Next: current-candidate Nix timing.

## Prefill-first continuation (2026-11-01)

The user's latest priority is **prefill first**; TG/MTP tuning is deferred.
No precision repair was applied for the documented q6 drift. Preserve these
completed pure-Nix10.1 current-candidate baseline receipts; do not rerun them:

| Receipt | pp512 d0 | pp2048 d0 | pp8192 d0 | pp2048 d32768 |
| --- | ---: | ---: | ---: | ---: |
| `fn-final-t1-x-ar-nix101` | 1039.61 | 1372.10 | 1411.52 | 1257.46 |
| `fn-final-t2-x-mtp-nix101` | — | 1291.82 | — | 1180.47 |

Rates are tok/s, r2, EXL3 5.05 + KVarN; both health_PASS. T1 TG128 was25.01
(d0)/24.37(d32K); T2 was16.93/22.16. These TG findings are preserved, not the
next optimization target. Historical UD timing images inherited /opt/rocm/lib,
so those numbers are context rather than demonstrated matched-runtime controls.

T3's first attempt failed before requests because the worktree git metadata
was outside its container. The reviewed read-only git-metadata wrapper fixed
that harness issue. `fn-final-t3-worktree-nix101` completed K17 in246.8946s;
D was intentionally stopped on the user's prefill-first direction. Its
`USER-PRIORITY-STOP.json` records the stop; container137/windowFAILED is an
intentional cancellation, with health_PASS, not a performance/correctness
regression. Preserve completed K and partial D. T4 was not launched.

Current prefill candidate reuses original routing IDs in EXL3 MoE combine
instead of repeatedly recovering the same expert from bucket offsets.
Independent review established the mapping/lifetime invariant; arithmetic,
slot order and FP32 barriers are retained. Validation and performance are
pending; no gain or bit-exact claim yet. Recipes/source provenance are in
`build/f3-nix-prefill-ids/`. Obsolete task-owned ROCm7.2 intermediate objects
and static archives were reclaimed for the150GiB free-space floor; all linked
baseline executables/images, source snapshots and receipts remain. A manifest
records every removed intermediate. Avoid building the Nix profiler SDK here:
its missing closure would also rebuild LLVM/Clang/hipify/rocprofiler/RCCL.
The saved hc7 profile is only a bottleneck lead, not a current timing control.

## Prefill batching and operator screening update

User direction (2026-11-01): preserve completed work (including TG/MTP), prioritize
MoE matrix operations and dense X3, batch small improvements, retain Nix ROCm10.1
intermediates, screen with small operators, and run one aggregate full-model
correctness/prefill performance pass at the end. Reuse the saved matched baseline;
do not repeat baseline inference or the expensive CPU diagnostic.

The ID-based expert combine is implemented/reviewed but not full-model qualified.
Its Nix build compiled all targets, then a packaging comparison falsely rejected
CMake's expected installed-RPATH change. Raw outputs remain in
`build/f3-nix-prefill-ids/recovered`; no repeat full build for that packaging check.
A persistent Nix10.1 tree now lives at `build/f3-nix-incremental/units` (187targets,
compressed debug sections, unchanged arithmetic/optimization flags). The separate
release tree will avoid toggling global test assertions and rebuilding both modes.
Matrix prototypes use `build/f3-prefill-matrix/objects` with Ninja dependencies;
an unchanged invocation already reported no work. Preserve these trees.

`fn-prefill-operator-screen-nix101`: independent frozen combine oracle PASS;
the new225-row final-tail fixture exposed a pre-existing attention-pruning bug.
Unsupported WMMA geometry prepared only final-query masks, then generic attention
read all masks, producing NaN routing IDs and a GPU memory fault. The window's
health failed; matrix timings from it remain advisory, not qualified speedups.
The reviewed fix gates pruning on WMMA geometry/live extent and KVarN pass.prefill.
`fn-prefill-wide-screen-nix101` COMPLETE/health_PASS: affected unit0.564s, including
97-of225 expert-tail diagnostics and exact chunk logits; matrix test2.666s, all
outputs exact. No passed full model or MoE24s suite was repeated.

All first-window controls remain immutable and are mounted read-only for reuse;
full quoted-header closure is recorded in a separate provenance extension.
Initial MoE iglp/split32 variants showed no useful consistent signal. BM256
16x1/8x2 candidates also showed no useful signal (advisory ratios+6..45%) and
were not promoted. Grouped dense iglp remains a candidate (advisory2..6% lower
operator time), pending aggregate qualification. Persistent MoE scheduling is
being screened next; it was explicitly untried in the previous findings.
No full-model window has been launched for this batch. Earlier full-quality and
pure-Nix T1 baseline receipts remain the authoritative baseline.

## Qualified prefill batch, 2026-11-01

Retained: direct original routing IDs in the expert combine, WMMA/KVarN eligibility
before last-query mask pruning, and grouped dense X3 scheduling. Independent
implementation/plan/evidence review passed. User criterion: performance-neutral
simplifications or correctness improvements can be retained; absence of a speedup
alone is not a rejection. The matrix prototypes added complexity and provided no
qualified benefit, so their source/binaries remain experimental artifacts.

One combined window `fn-prefill-batch-nix101`: COMPLETE, health_PASS, all 5 steps PASS.
The existing K7 and 64/65-row dense and independent cancellation checks passed (0.514s).
One new collector completed 32,760 records/64 head replays (152.758s); its101424960-byte
payload exactly matches saved Nix q6 SHA
`789c46a2a6939dc980e23b2cbf78bac71ccbcd9b1472899be846a1d56584f2a0`.
Prior hc7-relative NLL drift remains +0.006600 (~0.66% perplexity); this batch adds
no drift in the collected records. This is not a claim about every possible history.

| Prefill | Saved pre-batch (tok/s) | Candidate (tok/s) | Change |
|---|---:|---:|---:|
| pp512 | 1039.61 ± 0.73 | 1061.85 ± 0.37 | +2.14% |
| pp2048 | 1372.10 ± 0.11 | 1400.15 ± 0.83 | +2.04% |
| pp8192 | 1411.52 ± 1.17 | 1442.28 ± 1.13 | +2.18% |
| pp2048 @ d32768 | 1257.46 ± 0.39 | 1280.80 ± 3.23 | +1.86% |

EXL3 5.05, KVarN k4v4, speculative off, temperature 0, seed 314159, r2. Saved baseline:
`fn-final-t1-x-ar-nix101`; no baseline inference repeated. Dropping the generation
cases preserves prefill order, warmups and contexts 8193/34817. Observed 1.86–2.18%
gain is one candidate window versus saved pre-batch measurements, not a fresh
interleaved significance test or per-change attribution. No new TG/MTP checks.

Image: `ai-tools/gufo-eval:f3-prefill-batch-f712b2d0-rocm101`,
`sha256:931977351cdc6b604873272394d404497215523f68a1c5dea7dfbc01bc3605d5`. Installed production ELF SHA
`19448675aff38cd773eb1357f73231c28664bc2a53508fdd50a94c07df025f92`.
Source patch `f712b2d037d148c07f3e9466a2edd02ddf1ce0ab802afdda32324477eba182bd` is frozen in
`build/f3-nix-incremental/BATCH-SOURCE.patch`; it includes inherited work as well
as this batch's three incremental changes. Version/help/server smoke checks pass.
Actual collector and production mappings confirm Nix patched ROCr10.1/CLR10.1/
COMGR24.0.0; HIP API version7.16 is not ROCm7.2. Receiptroot
`PREFILL-COMPARISON.json` and `build/f3-nix-incremental/BATCH-QUALIFICATION.json`
contain raw pointers and deltas. No commit, merge or push performed.

Preserve `build/f3-nix-incremental/{release,units}` (initial release 306 targets;
updated unit build only 5 steps), `build/f3-prefill-matrix/objects`, recovered raw
Nix outputs and the baseline/candidate images. Only exact, task-owned duplicate
Docker packaging-cache IDs were reclaimed for the 150 GiB window floor; see
`PACKAGING-CACHE-CLEANUP.json`. No Nix objects, source, images or baseline receipts
were deleted. The persistent80/160 prototype passed exact outputs/health but was
not promoted; it and wider tiles are retained in `archive-persistent`/`archive-wide`.

## Local integration follow-up, 2026-11-01

Validated code and its evidence were committed as `2c159a12`; pending know-how
and instruction-pointer work received independent review as well. F3 was merged
into port-u4f (`9fcaffea`) and then into integ (`ffc0738a`), preserving the
port branch's documentation-only history. The old no-commit/no-merge statements
above describe the qualification snapshot, before this follow-up.

`F3-INTEGRATION.json` records exact tracked-blob identity across production
sources, tests, tools, CMake, Nix pins and runtime assets. The source/test diff
from base `966f84b8` still has the tested patch SHA `f712b2d0...`. The pre-commit
CI formatting check passed all 758 C++ files. No model tests, baseline inference,
builds or image creation were repeated solely for these merge labels.
The tested image `93197735...` and persistent Nix intermediates are unchanged;
main's pre-existing `.dockerignore` was preserved by content hash. No push.

Further prefill work starts from this integrated code and saved candidate
measurements. The >=2,000 tok/s goal remains unmet; completed TG/MTP work stays
in place and further TG/MTP tuning remains deferred under the user's priority.

## Qualified narrow dense/output-slice batch, 2026-11-01

[Batch2](F3-PREFILL-BATCH2.md) retains the small-row K5 narrow tile and avoids
unused aligned singleton parent outputs. All boundaries/slices/guards passed;
32760 records and64 replay checks exactly match saved q6. pp20481405.25±.83
is+.36% versus saved1400.15±.83; other prefill cases are+.09–.27%. The2000
objective is unmet. Nix incremental build reused old objects and preserved
changed predecessors; no baseline inference or new model profiling occurred.
Routed vector stores/wave64, full-LDS dense epilogues and cached internal FP16
weights had no useful large-shape benefit; artifacts and measurements remain.
See batch2 for the wrapper-permission-only recovery and exact installed hashes.

## Qualified dense traversal follow-up, 2026-11-01

[Final grid batch](F3-PREFILL-GRID.md): pp20481416.51±.82 tok/s, +.80% over
batch2 and+1.17% over the saved starting batch. Actual production-object
exact/guard checks passed; batch512 collector reused because the new predicate
requires2048. All seven actual K7 shapes were screened; only6144→2560 singleton
uses the new traversal. The cached-FP16 branch is rejected. Image31fd777e…,
source patch67a50024…, persistent Nix intermediates retained. Goal2000 unmet.


## Shared-MoE and KVarN closure, 2026-11-02

The two-phase stream/resource goal qualifies shared EXL3 MLP beside router and
routed MoE, plus exact register-held KVarN statistics. Final pp2048 is1429.13±1.13,
+.89% from saved1416.51; pp512/8192/d32K improve.52/.93/.85%. All final-image
bounded state and KVarN CPU-oracle checks pass. The combined-image fullcollector
32760/64 exact result is supporting evidence with a source/scope bridge; the
cleaned image did not rerun it. Target2000 remains unmet; TG/MTP preserved.

Initial Q/shared combination was exact but regressed1.27–1.66%. Isolation picked
shared-only; dedicated Q events did not fix the interaction. Baseline Q/KV
launching is restored. All bounded measurements, private/spill distinctions,
failed candidates and final artifacts/scopes are recorded in
[F3-STREAM-QUALIFICATION.json](F3-STREAM-QUALIFICATION.json),
[F3-STREAM-SCREENS.json](F3-STREAM-SCREENS.json) and
[F3-RELEASE-RESOURCE-AUDIT.json](F3-RELEASE-RESOURCE-AUDIT.json).
No saved baseline inference or full profile was repeated. Reuse Nix intermediates.
