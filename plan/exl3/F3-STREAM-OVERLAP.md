# EXL3 prefill stream-overlap candidates, 2026-11-01

**Final disposition:** shared/routed MoE overlap and the KVarN register cleanup
qualified together at pp2048 **1429.13±1.13 tok/s (+.89%)**. Q/KV overlap is not
retained: alone it was whole-model neutral; together with shared overlap it
regressed despite local gains. Separate dense events did not recover baseline.
All raw bounded records are in [the screen ledger](F3-STREAM-SCREENS.json), and
whole-model selection/validation scopes in [qualification](F3-STREAM-QUALIFICATION.json).


Status: source/dependency review only, against code `b13b712a` (documentation
head `ab7ce1cb`). These EXL3 schedules are **not implemented or measured**.
The qualified image and saved performance/accuracy receipts are unchanged.
Matrix-kernel screening did not close this separate scheduling direction.

## Existing evidence and its limits

- Attention already rotates Q/K/V together and groups equal-shaped K/V GEMMs;
  Q is a separate GEMM on the same stream. GDN QKV is one projection, but its
  separate Z projection also runs on that stream.
- Shared EXL3 gate/up are grouped. Routed gate/up share a rotation launch but
  their GEMMs are sequential. Grouping alone does not establish concurrency.
- Production GDN Z and shared-expert side-stream paths exist for qualifying
  Q8_0 weight routes using FP16 arithmetic. EXL3 does not select those legacy
  paths; its newly qualified shared path uses dedicated workspace. Historical results
  in `docs/models/qwen3.8-flash-next/EXPERIMENTS.md` are not EXL3 results.
- The older `tests/models/qwen38_flash_next/overlap_bench.cpp` probes plain
  FP16 X3 or W8A8 HC-down beside HC combine/mix. Its measured 12–20% overlap
  does not accept or reject the EXL3 pairs below. Same-SIMD WMMA/VALU contention
  likewise does not prove that different kernels cannot benefit from streams.
- The shelved paired gate/up launch is a different proposal from two-stream
  gate/up execution. Neither is a measured success for this EXL3 workload.

## Candidate queue

Priority reflects dependency slack and implementation cost, not measured gain.

| Priority | Independent branches | Fork and join | Storage / performance concern |
|---|---|---|---|
| 1 | GDN Z versus QKV → alpha/beta → convolution/preparation → recurrence | After grouped rotations, expose Z on a side stream; also consider starting it after QKV to reduce GEMM competition. Existing `z_ready` wait joins immediately before `GdnEpilogueKernel`. | QKV and Z outputs are disjoint. Preserve their rotated inputs and device descriptors until Z finishes. Recurrence and Z can still contend; late consumption is an opportunity, not proof of speedup. |
| 2 | Shared MLP versus routing / routed MoE | Fork once normalized input is ready; retain the join before shared/routed combination, including deferred HC combination. | Shared outputs have separate allocations, but EXL3 dense rotations borrow the routed `exl3_prefill_.input` workspace. Allocate independent rotation storage or prove another region dead for the entire overlap. Enabling the existing Q8-only predicate alone would race. |
| 3 | Attention Q versus grouped K/V | Fork after grouped rotations; join before any consumer needing both branches. A later refinement can move branch-local normalization/KV preparation ahead of that join. | Per-target rotation segments and outputs are disjoint; shared descriptors remain live. Both GEMMs are compute-heavy, and K/V are much narrower, so recoverable overlap may be small. |
| 4 | Routed MoE gate versus up | Fork after bucket/gather and joint rotations; join before activation and down projection. | Gate reads `scratch.input`; up reads `scratch.out` reinterpreted as FP16. Outputs are separate. Activation reuses input and down overwrites the up-input slab. Both branches stress similar resources; do not infer a gain from independent operands. |

The indexer Q/K slices share a parent projection already computed once; they
are not a second independent Q/K GEMM pair to overlap.

## Required lifetime and scheduling rules

`Exl3DensePrefill` has one three-entry device descriptor set and reuses rotated
input storage between chunks/calls. It cannot return pending work to arbitrary
callers or start the next chunk over the same memory. Initially keep unsupported
multi-chunk/capture cases on the serial path, or explicitly provide stable
per-inflight descriptors/storage and joins. Do not simply redirect whole helper
calls to another stream: shared scratch, mutable executor stream state and
nested projection calls all need an ownership audit.

The attention, linear-attention and expert stages reuse a common arena. Keep
whole-stage ordering; overlap only the proven independent branches inside a
stage. Join before scratch reuse on normal and error/cancellation paths. If a
new path is graph-eligible, its fork/join must be captured correctly with stable
addresses/events and replay checked; otherwise retain the existing fallback.
Reuse of the existing side stream/events needs explicit non-overlapping lifetimes
between GDN and MoE. No host/device-wide synchronization in the steady path.

Full-LDS/large-workgroup GEMMs may leave little room for co-residency. A kernel
that loses alone is not automatically rejected for a new paired schedule, but
reopening it needs a concrete resource-fit argument and complete-span evidence.

## Bounded next experiment

1. Reuse existing profiles to choose representative shapes. Implement GDN Z
   first; investigate shared/routed storage separation alongside it. Keep all
   completed TG/MTP changes and the pinned Nix ROCm 10.1 build intermediates.
2. Screen with small exact-output/guard tests covering the new joins, tails,
   chunk transitions, GDN recurrent state and convolution history, plus captured
   replay where enabled. Measure the complete fork-to-joined-completion span,
   including preparation and synchronization; do not add isolated kernel times
   or stop timing while the side stream remains active.
3. Reuse applicable saved controls. If the new paired workload lacks a matched
   serial control with the same buffers/cache/harness, collect that control
   once and save it. This does not call for rerunning the full-model baseline
   or collecting another full-model profile.
4. Batch viable candidates into one final affected correctness and prefill
   performance window using the preserved baseline receipts. Report contention
   or inconclusive results explicitly. No gain, target attainment or universal
   hardware limit is claimed by this investigation.

## Source audit anchors

- `src/models/qwen38_flash_next/kernels/rocm/executor.cpp`:
  `Exl3DensePrefill`, `LinearAttention`, `Attention`, `Moe`, `Exl3Experts`;
  stage-arena allocation and separate `shexp_gate/up/out` allocations.
- `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp`:
  `PrepareExl3DensePrefillKernel`, `GatedDeltaNet`, late `z_ready` wait.
- `src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hip`:
  `Exl3PrefillExpertsRaw`, joint rotation, gate/up launches and down reuse.

A second source reviewer independently confirmed these dependencies and the
boundary between older generic/Q8-weight overlap evidence and the untested EXL3 paths.

## Bounded screening and initial integration (2026-11-01)

Four guarded Nix10.1 operator windows completed with health_PASS, finite exact
outputs and guards. No full model baseline or profile was rerun. Receipts live
under `ai.tools/tmp/test-results/gufo-upstream-port-20261023/`:

| Receipt suffix (`fn-prefill-stream-…-nix101`) | Scope and 2048-row result | Disposition |
|---|---|---|
| `gdn` | Rotate/prepare + QKV/Z + recurrent GDN: serial7.89030ms; late Z7.84739(−.54%); simultaneous7.93047(+.51%). Frozen alpha/beta; its projection excluded. | No promotion of Z alone |
| `resource` | Smaller-LDS Z tiles lost2.72–5.16%; heterogeneous grouped GDN lost1.79%. Q/KV projections: serial4.40757ms; Q-side4.28673(−2.74%); KV-side4.27553(−3.00%); heterogeneous group4.29973(−2.45%). | Q-side projection candidate; no new matrix specialization |
| `moe` | Full shared MLP + raw routed MoE + final combine; fixed Zipf routing, router/top-k excluded. Serial15.0755ms; shared-side14.7609(−2.09%); gate/up split14.9834(−.61%). | Shared-side candidate; no nested gate/up fork |
| `branches` | Q/KV plus Q norm/RoPE/Hadamard and K/V norm/RoPE/store/KVarN rotations: serial5.18853ms; KV projection5.07193(−2.25%); full KV branch5.06726(−2.34%); full Q branch5.06312(−2.42%). Attention and quantization excluded. | Projection fork avoids broader refactor for tiny incremental gain |

At512 rows, GDN late Z−.42%, simultaneous+1.63%; Q-side projections−.70%,
shared-side MoE−.74%, gate/up−.11%; full Q branch−.27%, KV projection+1.15%.
At65 rows, Q-side projections−24.96%; full Q/KV preparation branches improved
23.71/24.62%. These are complete local spans, not whole-model speedup estimates.
Two warmups and10 timed iterations per candidate. New matched controls were
saved once; the second screen reuses the first GDN controls. Source/object/binary
hashes, compiler, image and actual patched ROCr/CLR10.1 maps are frozen in each
receipt. Correctness poisons outputs before the fork, compares GDN recurrent and
history state, and verifies the immutable17-token KV prefix in branch tests.

The retained production path begins shared MLP before router/top-k, using its own
rotated-input and descriptor allocation. Extra storage is
`2 * max_batch * max(2 * hidden_size, shared_expert_ff)` bytes plus three rotate
and three GEMM descriptors (41.25MiB at4224×2560). The existing routed-scratch
estimator explicitly counts routed bucket slabs only and is unchanged.
The rejected initial Q/KV candidate forks the final width/rate group (Q) after common preparation and rejoins
inside the helper before consumers, descriptor reuse or return. Both paths are
prefill-only; TG/MTP dispatch and graph eligibility are retained. Independent
review checked buffer ownership, fork/join/event reuse and error cleanup.
The actual-executor512+1 bounded test completed health_PASS: identical-shaped
serial fallback and overlap have bit-identical residuals, logits and GDN state.
Final-layer pruning leaves only the first layer eligible for shared overlap.
Existing1/64/65 GDN chunk invariants also pass. The final retained binary uses
a stronger1025-row test: two full512-row FFNs per chunk, four shared forks,
then a serial tail. All residuals/logits/GDN state match the serial fallback.

### Additional dependency opportunities

QKV is not the complete opportunity list. GDN alpha/beta projection is also
independent until recurrence (EXL3 prefill uses PlainDensePrefill FP16 WMMA,
not the later BLAS branch); it needs separate narrowed input if moved to a side
stream. PLE query normalization can precede its embedding wait, and PLE key
normalization is independent of value projection. Indexer/selection can run
beside Q/KV with separate scratch but an older generic trial regressed; do not
repeat it unchanged. HC main projection→mix and output→combine are true edges;
a previous separate injection/shared experiment was exact but .5–.9% slower.
Cross-layer/microbatch pipelines need independent residual/arena ownership and
ordered recurrent/KV/checkpoint updates, so they are not simple stream switches.


## Whole-model selection (2026-11-02)

The initial combined image was exact (32760records/64head replays, FP16 KV),
but pp512/2048/8192/d32K throughput fell1.49/1.66/1.63/1.27%. Do not promote
operator wins without complete-workload validation. Same-allocation isolation
at pp2048 gave quantizer-only1417.09±1.37, Q-only1417.67±.20 and shared-only
1425.06±.51. Dedicated Q event handles still produced1399.91±.26; the cause of
the interaction is unproved. Preserve all diagnostic artifacts; no baseline
inference or new full profile was run. Initial isolation's missing-metadata
mount failure ran no inference and is not a timing result.

Final cleanup removes Q fork/event/counter code and restores the original
Q/KV grouping. Only explicit workspace/stream selection remains in the dense
helper for shared MLP. Final image `f3-shared5-703d3e77-rocm101` measured
pp5121068.58±3.09, pp20481429.13±1.13, pp81921460.97±.73 and d32Kpp2048
1301.99±1.41tok/s (+.52/.89/.93/.85%). All final-image bounded state and KVarN
CPU-oracle checks passed. Supporting fullcollector evidence is reused from the
combined image, with retained MoE source equality and restored baseline Q code;
**no fresh fullcollector on the cleaned final image is claimed**. KVarN model
performance uses k4v4; the reused fullcollector uses FP16 KV. Target2000 unmet.
