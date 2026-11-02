# Exact live-row loops for EXL3 verification

2026-11-02. Base `add504fe`; native 5.05/K6 n-gram, Nix ROCm10.1 unchanged.
Retained after one combined qualification. Component gains are not full-model gains.

## Mechanism and scope

`RoutedGemv` previously tested each row against the expert tile's live count
inside every unrolled K step. For K5 capacity3/4, select a fixed live-row body
once per tile. The same dot products and reductions execute in the same order.
Gate/up and scaled down share the selection; negative down exponents still use
the pinned FP32 fallback. Other bit rates, R1/R2/R8, and unscaled full-precision
down keep the generic route. Zero/overcapacity descriptors retain generic behavior.
There is no quantization, weight layout, KV, draft-controller or prefill change.

## Batched candidate screening

All candidates used the retained Nix10.1 compiler and production device flags.
Current baseline module was reused. One new cold operator control series was
saved per shape, then reused for later candidates without timing it again.
Each timing has five batches of20 launches, 64 distinct experts/62.5MiB trellis,
above the32MiB MALL. Ragged counts cover1..capacity. Ordered microbenchmarks are
advisory; small differences remain noise, not regression or improvement claims.

| Candidate | Gate R3 / R4 throughput change | Scaled down R3 / R4 | Decision |
| --- | --- | --- | --- |
| Trellis global address space | +0.32% / +0.33% | +0.41% / +1.20% | No material gain; do not retain compiler-specific cast |
| Global + aligned K-tail elimination | -0.44% / -0.35% | -0.52% / +0.20% | No material gain; screen-only assume is not a production contract |
| Exact live rows, gate/up only | +21.54% / +32.94% | unchanged, within noise | Extend the same mechanism to down |
| Exact live rows, gate/up and down | +21.38% / +32.03% | +9.60% / +13.90% | Retained after final qualification |

Global-address-space changes had previously been neutral in the prefill bucket;
this was a separate decode-path evaluation, not a novel mechanism. It changed
12 flat loads to global loads and reduced gate R3/R4 VGPR38/40 to32/34, without
a material speed win. Aligned-K increased register pressure again. Exact-row
selection uses gate VGPR40/46 and scaled-down54/58, all with zero private/LDS.
Lower register counts alone did not predict performance.

Dense K7 TN2 same-iteration load grouping was excluded before implementation:
the current native binary already issues all six trellis dword loads before
decode/dot. Saved profiles were reused; no new pipeline profiling was needed.

Each of three screen windows passed112 byte/guard cases. First screen compares
two candidates against one baseline output (199 comparisons); later screens
compare one candidate (112 each). Rates3..7, capacities1/2/3/4/8, tiny
unsaturated inputs, partial counts, poisoned unused slots, K144 tail,
scaled exponents-1/0/+1 and full-precision down are covered. The aligned-K
prototype is deliberately skipped at K144. No tolerance was widened.

## Final qualification

Independent final operators pass DownFixtures, ExpertInvariance and
FusedProducer across rates and ragged/shared/disjoint routes. Additional frozen
checks cover count0, overcapacity count, blocks beyond descriptor count and
z2 with distinct gate/up weights and activations. These run against the actual
candidate object. Final object's device `.text` is byte-identical to the
screened gate+down prototype.

The full32760-position prefill collector is byte-identical to the saved native
baseline (`5cf3aa2a927d40c2d9f991aaa82c8ab8e27d7755d568a7226bdd5b560008a05b`).
Sampling, concurrency,32K state, cache and identical PP/TG benchmarks all pass
in the single final window; healthPASS and loaded Nix10.1 runtime confirmed.
Existing baseline/model/reference were reused. Every repeated AR/MTP output
hash and cycle/draft/acceptance/lookup count equals the saved baseline.

| Scope | Saved baseline tok/s | Candidate tok/s | Observed change |
| --- | ---: | ---: | ---: |
| PP2048 | 1423.76 | 1426.88 | +0.22% |
| PP8192 | 1461.67 | 1460.75 | -0.06% |
| AR128 d0 | 25.20 | 25.20 | 0.00% |
| AR128 d32768 | 24.43 | 24.49 | +0.25% |
| MTP128 d0 | 40.09 ±0.08 | 40.85 ±0.14 | +1.90% |
| MTP128 d32768 | 78.53 ±1.03 | 79.48 ±0.03 | +1.21% |

These are one candidate window against the saved matched native baseline.
PP/AR remain effectively unchanged. Short-context MTP improves modestly;
the32K case is lookup-heavy and its change overlaps the baseline spread, so
it is not a resolved broad long-context gain. Do not transfer the21–32%
component improvement to whole-model throughput or claim Kyojin parity.
The patch retains arithmetic and passes all gates, so the modest end-to-end
result is retained rather than repeating expensive checks for a small gain.

Selected local image: `ai-tools/gufo-eval:f3-rows-91fce75a-rocm101`.
Qualified ELF SHA256:
`91fce75a239ab0aa80fe532b52422fa5850cfe568e3d5df9de461bb3d8b7e5f1`.
The image keeps the prior Nix10.1 closure and native5.05 model selection.

## Evidence and reproduction

Gufo intermediates: `build/f3-tgmtp3/` (prototype source, objects, modules,
compiler/link argv, resource inventories, current dense ISA, final binaries).
All previous Nix intermediates and baseline ELF remain intact.
Ai.tools scripts/plans: `test/experiments/gufo_upstream_port_20261023/`:
`tgmtp_load_screen.hip`, `tgmtp_rows_qualify.py`, `tgmtp_rows_analyze.py`,
`PLAN-FN-TGMTP3-*.json`, `TGMTP-ROWS-RESULT.json`, and
`tgmtp_rows_artifacts/` with preserved candidate patches/build identities.
Receipts under `tmp/test-results/gufo-upstream-port-20261023/`:
`fn-tgmtp3-screen-nix101`, `fn-tgmtp3-rows-screen-nix101`,
`fn-tgmtp3-rows-down-screen-nix101`, `fn-tgmtp3-qualification-nix101`.

Independent second reviewer approved source, dispatch boundaries and final
harness linkage. Shared CI formatting check passes763files using the exact
pinned clang-tools21.1.8; direct Nix shell was blocked by container worktree
ownership/mount resolution, so the same check ran with that retained tool.
No full baseline or Kyojin benchmark was rerun. No baseline was deleted.

Final ai.tools infrastructure regression:432passed in19.73s. No additional GPU
run was needed after independent final review. Root disk remains166GiB free.
