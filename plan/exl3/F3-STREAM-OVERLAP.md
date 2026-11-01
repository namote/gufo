# EXL3 prefill stream-overlap candidates, 2026-11-01

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
  Q8_0 weight routes using FP16 arithmetic. Current EXL3 does not select those paths. Historical results
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
