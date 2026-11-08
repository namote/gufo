# B6 brief: Flash-Next EXL3 prefill (2026-11-08)

Same rules as `plan/exl3/archive/B1-BRIEF.md` (Rules, Numerical policy — bits may
change, lossy changes isolated in their own commit with the expected error
stated, FP32 accumulation/residuals kept; build `build/b1/build.sh`; pinned
clang-format via `build/b1/build.sh --run`; commit/attribution; report).
Chunk/split invariance is a correctness contract. No GPU/model/timing runs while
iterating. Main checkout `/home/m2on/projects/gufo`, branch `ai-tools/port-u4f`,
one item at a time, second-reviewer subagent review per item, each item in its
own commit(s) so the final pass can revert it alone. Changes that are meant to
be exact need a unit test comparing real-path bytes against the old path.

Goal (user priority: prefill first, beat Kyojin by a wide margin): PP8192
1544 → ≥1800 tok/s (5306 → ≤4550 ms per 8192 forward, −755 ms); PP2048 1511 →
≥1750. Decode must not regress.

**Read first:** `build/b1/prefill-prof-b6/summary.md` (+ `summary.json`,
`profile.json`, `gaps.json`, `contention.json`) — fresh c03 profile of the
deployed binary; `plan/exl3/archive/F3-SEARCH-STATUS.md` and
`plan/exl3/archive/F3-SESSION-FINDINGS.md` (rejected prefill experiments — do NOT
repeat them unchanged: K32 stages, smaller waves, wide/masked tiles, compact
MoE classes, cold/hot split, wave64 routed core, dense LDS output staging,
raw FP16 weight cache, persistent scheduling, deeper stages/double buffering/
fragment hoists/direct-global B, grouped single-launch routed gate/up, Q/KV
stream overlap, int8 HC); KNOWHOW "Current shared5 prefill profile" and
"Routed MoE ceiling diagnosis".

PP8192 main-stream ms (two 4096 chunks): routed gate/up 944 (26.8 TF), down
537 (23.5 TF), dense EXL3 1222 (≈35–37 TF), HC mix/combine 757 (≈210 GB/s),
HC projections 306, MoE routing/rotation/activation 492 (RotateGateUpTokens
229, router GEMM 14 TF), GDN 433 (GdnRowSplit 314 with only 96 workgroups),
attention + sparse indexer 391 (WMMA attention 317 at 13.7 TF useful), dense
input rotation/prep 127, idle 91 (recurring ≈0.24 ms gaps before large GEMMs /
HC / GDN kernels), shared MLP side stream 1970 (overlapped, 1.6–4.7 TF).

## Items (ranked by plausible saving at PP8192)

- **P1 MoE glue (−≈270 ms possible).** Fold `RotateGateUpTokens` into its
  producer (the mix epilogue / routed input staging, as decode E1 did with
  producer-side rotation) or into the routed gate/up A-staging; bring the
  router GEMM onto the efficient WMMA path or fuse it with the producer.
  Check how many times per layer the same activations are read/written.
- **P2 GdnRowSplit occupancy (−≈150 ms).** 96 workgroups on 40 CUs: split the
  recurrence work further (heads × value-columns, or more rows per chunk with a
  two-level scan) while keeping chunk/checkpoint invariance exactly (B1 lesson:
  the kPrep fusion broke it). Also look at GdnPrepAb → RowSplit gap.
- **P3 sparse attention at long prefill (−≈190 ms at 8192).** WMMA causal
  attention at 13.7 TF useful: examine tile shape vs selected blocks, masking
  waste, K/V reuse, and the indexer/selection chain (SelectMark → attention
  gaps). Keep the selection semantics identical.
- **P4 routed MoE GEMM (−≈400 ms if 35 TF).** Only genuinely new mechanisms
  (see rejected list). Candidates: measure/decide shared-MLP contention policy
  (the side stream runs at 1.6–4.7 TF while overlapping; consider giving it
  fewer CUs or scheduling it against non-WMMA phases — the stream ledger
  `F3-STREAM-OVERLAP.md` has prior evidence), decode/WMMA interleave schedules
  in the K5 cold/hot kernels found by ISA inspection, B-staging of ragged rows.
  Report the analysis even if nothing is implemented.
- **P5 HC mix/combine bytes (−≈100–300 ms, may be lossy).** Already near
  bandwidth; savings require fewer bytes: e.g. FP16/BF16 for intermediate
  (non-residual) HC tensors, fusing combine+mix reads of the same residual,
  or avoiding a re-read of `res` per stream. FP32 residual storage stays
  unless you can argue the error explicitly; a lossy step goes in its own
  commit with the expected error stated (quality pass decides).
- **P6 idle gaps (−≈50–90 ms).** Find why ≈0.24 ms gaps precede large GEMMs
  (host-side work, events, kernarg size, cache/TLB, or profiler artifact) —
  Nov 2 shared5 had 4.6 ms total idle. If real and fixable (e.g. host waits
  between chunks or per-layer syncs), fix it.

Also: for every kernel you change, add an old-vs-new operator bench target
(as `qwen38_flash_next_hc_decode_bench` / `qwen38_flash_next_dense_gemv_bench`)
with real c03 shapes at 2048 and 4096 rows so the final pass can measure each
change directly. Report per item: expected ms at PP2048/PP8192 with reasoning,
resources (VGPR/LDS/scratch), exactness or expected error, and the exact GPU
tests/benches the final pass runs. Skip items whose analysis says they cannot
pay; say why.

## B6 addendum: P7 side-stream contention control (user chose to add it before the GPU pass)

The prefill shared MLP runs on the side stream for 1970 ms at PP8192
(1.6–4.7 TF) fully overlapped with the routed MoE GEMMs (23–27 TF); both
compete for the same WGPs, LDS and VALU. Implement a prefill-only side stream
policy so the final pass can choose by measurement:
- **P7a stream priority:** the prefill shared-MLP stream created with a lower
  priority (`hipStreamCreateWithPriority`) than the main stream.
- **P7b CU mask:** a prefill-only side stream created with
  `hipExtStreamCreateWithCUMask`, restricting the shared MLP to a subset of
  CUs (keep the decode E7/J3 branch stream unchanged). gfx1151 runs WGP mode
  (20 WGPs = 40 CUs); mask whole WGPs (CU pairs). Check how ROCr maps mask bits
  to CUs/SEs on this chip (read the ROCr/HIP source in the Nix store) so a mask
  spreads across shader engines.
- Make the policy a single compile-time constant (no CLI option, no env var),
  default = current behaviour, so the final pass builds variants: none, P7a,
  P7b with 4, 8 and 12 CUs (and P7a+P7b-8 if cheap). Graph capture must still
  work if prefill uses graphs; check that masked streams are compatible with
  the existing event fork/join and with `hipStreamWaitEvent`.
- Own commit(s); exact (scheduling only). Add a CPU-buildable check that the
  stream is created and falls back cleanly if the call fails.
