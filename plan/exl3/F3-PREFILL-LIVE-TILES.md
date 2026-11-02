# Prefill live-subtile compiler screen, November 2, 2026

Decision: reject all three prototypes before GPU execution. The successful
TG/MTP exact-live-row mechanism does not transfer cheaply to the current X3
prefill kernel. Production source and the qualified rows-91fce75a image remain
unchanged. This is a compiler-resource result, not measured latency regression.

For K5 raw routed EXL3 experts, prefill uses HalfPrefillGemmKernel, including
singleton tails; decode uses RoutedGemv. Prefill already skips dead B subtiles
in fetch, LDS stores and WMMA.
The remaining opportunity was selecting the live 16-row subtile loop once,
instead of evaluating its guards during every K step. Existing baseline ISA
still contains these predicates. Earlier compact-class experiments changed
launch geometry; these candidates keep BN64/128, WM8/WN1, descriptor order,
two class launches and the original LDS allocation.

| Prototype | BN64 VGPR | BN128 VGPR | BN128 private bytes | BN128 VGPR spills |
| --- | ---: | ---: | ---: | ---: |
| Saved production | 103 | 148 | 0 | 0 |
| Four-way K-loop selection | 231 | 256 | 244 | 108 |
| Four-way whole-body selection | 239 | 256 | 136 | 68 |
| Two-way full-subtile / generic | 191 | 256 | 148 | 71 |

All variants retain 24/32 KiB LDS for BN64/128. The first shares accumulator
and staging state across specialized loops; the second scopes the full body
and epilogue per specialization with wrapper-owned LDS array references.
The third specializes only rows49–64 or113–128, retaining generic partial
buckets. Full-subtile does not mean all rows are valid: input clamp, output
bounds and exponent handling remain required. Complete is not enabled.

Multiple large loop bodies increase code size and register pressure in one
machine function. Changing C++ scope alone did not recover resources; the
precise register allocator cause was not established. The first prototype's
BN64/128 disassembly grows from1933/3296 to3745/5113 lines. Static instruction
and barrier duplication must not be described as extra dynamically executed
math or barriers. The bounded two-way refinement also spills and is rejected.

All three prototype versions of the affected TU compile with production Nix10.1 clang24,
O2, gfx1151 and no fast-math. Baseline metadata is extracted from the retained
batch6 prefill object. No GPU operator, model, baseline inference or profile
was rerun. No new numerical-quality or speed qualification is claimed.
Independent review agreed with the resource gate and the final bounded
refinement. Future work should not repeat these clones without a distinct
code-generation mechanism; additional launches/noinline calls are not an
adopted workaround.

Evidence: [resource records](F3-PREFILL-LIVE-TILES.json), retained intermediates
in build/f3-prefill-live-tiles, and ai.tools
test/experiments/gufo_upstream_port_20261023/prefill_live_tiles_artifacts
(three source patches, compiler argv/logs, result and artifact SHA manifest).
The qualified ELF remains
91fce75a239ab0aa80fe532b52422fa5850cfe568e3d5df9de461bb3d8b7e5f1.

All three saved patches reproduce their compiled sources; artifact hashes and
duplicate result records agree. Documentation whitespace check passes.
Ai.tools infrastructure checks:432passed in18.32s. No production C++ changed.
