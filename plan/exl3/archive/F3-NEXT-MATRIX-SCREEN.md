# Prefill matrix screening, November 1, 2026

Production remains at the qualified `9dc1d15c` source and image
`f3-prefill-batch-f712b2d0-rocm101`. No production kernel or dispatch changed.
The saved full-model baseline remains pp2048 **1400.15 tok/s**; the 2000 tok/s
goal is not met. No full-model baseline, quality collection, or timing sweep
was repeated for these prototypes.

Both operator windows completed with health_PASS on the same Nix ROCm 10.1
runtime. Loaded mappings confirm patched ROCr `88jjsz00...` and CLR
`al4qwi7j...`. Compiler flags remain the production matrix TU's `-O2`, without
fast-math. Times use two warmups and ten HIP-event launches. These are warm
synthetic operators, not end-to-end inference measurements.

## Stage depth and workgroup geometry

`fn-prefill-stage-screen-nix101`: 2.766 s. Current controls were captured once
and saved with exact outputs, hashes, compiler/build commands, header closure,
runtime identity, shapes, and seeds. The older matrix control window had failed
health checks and predates the grouped scheduling change; its times were not
used. Dense and routed cases use K5, 2048 tokens, and unchanged ascending K16
accumulation. The new K32 candidates pair reduced staging with fewer waves;
they are distinct from the previously measured 32-wave K32 wide tile.

Time changes relative to each saved current operator control (negative is faster):

| Dense K→N, groups | 256×128 8×2 K32 | 128×256 4×4 K32 | 128×128 4×2 K32 | Masked 256×256 8×4 K64 |
|---|---:|---:|---:|---:|
| 6144→2560, 1 | +22.77% | +58.82% | +21.68% | — |
| 2560→6144, 1 | +26.03% | +32.61% | +20.87% | — |
| 2560→640, 2 | +44.74% | +9.22% | +2.34% | +18.91% |
| 2560→640, 1 | +49.29% | −9.94% | −16.48% | +26.05% |

Routed BM128 K32 was +2.11% for gate/up and −1.97% for down. BM64 K32 was
+22.10% / +11.30%. All compared output bytes were exact and finite/nonzero.
The separate unequal 65/129-row grouped N640 case also passed exact output
and allocation-tail guards for all four dense variants. Generic timed cases
allocated guard space but did not initialize/check it; do not describe their
coverage as guarded. This does not qualify other rates or full-model quality.

The single-group narrow candidate is preserved, not promoted. Pack metadata
confirms the shared indexer parent is K5, K2560/N640; shared gate/up are K7
and use two groups. Shared q/k reuse requires sparse attention, but prefill
also computes the complete parent for the indexer_k slice before sparse
attention activates (`executor.cpp:3365`, `batch.cpp:715`). The candidate can
therefore reach pp2048 d0; its local 16.48% time reduction is not a measured
pp2048 gain and does not apply to every dense GEMM.
Actual packed weights/cache traffic and other token widths remain unmeasured.
Do not spend a separate model window on this small candidate; combine any
later qualification with larger viable work.

## MoE launch bounds and descriptor partition

Inspection found a workload mismatch in earlier synthetic launch geometry:
production scratch capacity is 4224 tokens for sufficiently large contexts,
so both cold and hot kernels launch 1172 bucket slots, even for 2048 tokens.
The operator's historical 832-slot count came from its actual assignments.

`fn-prefill-partition-screen-nix101` reused the healthy frozen outputs without
rerunning the 832-slot control. It measured the actual 1172-slot launch once,
then two alternatives using unchanged matrix kernels:

| Launch geometry | Gate/up ms | Down ms | 2×gate + down change vs 1172 |
|---|---:|---:|---:|
| Current 1172 slots per class | 2.979682 | 3.198049 | — |
| Tight 668 slots per class | 3.015123 | 3.156427 | +0.32% |
| Cold 512 slots + hot 315 slots | 3.010771 | 3.125126 | −0.12% |

The printed raw `change_percent` fields compare against cached 832-slot
times; the table above and JSON `current_capacity_comparison` use the correct
1172-slot reference. The weighted sum is an operator estimate, not a model
measurement. Partition construction happened on the CPU outside timing, so
even that negligible result excludes a production device-builder cost.

For A assignments, E experts, and 64-row bucket mode, a safe total bound is
`active + (A-active)/128`, with `active=min(A,E)`. At most one cold bucket
exists per expert, including the balanced 129→64+65 split; hot buckets each
contain at least 65 rows. The screened histogram had 245 cold + 155 hot
buckets. Repartitioning preserved descriptor contents and every output byte.
The tested layout/bounds are not useful enough to justify more production
code; empty launched slots are not the dominant measured cost here.

## Preserved evidence and next work

Machine-readable times, identities, scope, and pack metadata are in
[F3-NEXT-MATRIX-SCREEN.json](F3-NEXT-MATRIX-SCREEN.json). Receipt roots are
`/home/m2on/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/`
`fn-prefill-stage-screen-nix101` and `fn-prefill-partition-screen-nix101`.
All prototypes, compiler resources, LLVM/ISA intermediates, and objects remain
under the f3-integ worktree's `build/f3-prefill-stage` and
`build/f3-prefill-partition`. Production `build/f3-nix-incremental` is intact.

Next work should target the matrix computation itself or reduce required
computation. Do not retry these stage/geometry, masked-wide, or descriptor
partition variants unchanged. Stream-K remains untested and would need an
explicit reduction-order/quality argument before any production adoption.
The narrow candidate can join a later larger batch. TG/MTP remains retained.
