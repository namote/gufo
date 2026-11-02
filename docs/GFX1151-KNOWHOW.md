# gfx1151 optimization know-how

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
