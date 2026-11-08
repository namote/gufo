# B4 brief: Flash-Next decode, undo serial tails, parallel HC fusion, routed loads (2026-11-08)

Same rules as `plan/exl3/archive/B1-BRIEF.md` (Rules, Numerical policy — bits may
change, lossy changes isolated and stated, FP32 accumulation/residuals kept;
build `build/b1/build.sh`; pinned clang-format via `build/b1/build.sh --run`;
commit/attribution; report). Chunk/split invariance is a correctness contract.
No GPU/model/timing runs while iterating. Main checkout
`/home/m2on/projects/gufo`, branch `ai-tools/port-u4f`, one item at a time,
second-reviewer subagent review per item.

B3 result (read KNOWHOW "Flash-Next B3 decode dispatch fusion and launch
gaps"): dispatches 1442→998/token but AR −1.2%, MTP −2.8/−3.1% vs B2. Graph
dispatch costs ≈1.85 µs; the ≈7 µs/dispatch idle is mostly ramp/drain and
dependency latency. G1's "last arriving block runs the tail" HC kernels added
≈1.1 ms/token busy (HcUpMix 2.60 vs Q8 up GEMV 1.71 ms; HcDownSiluQ8 1.98 vs
1.71). **Lesson: a fusion must not add a serial single-block tail on the
critical path; the fused kernel's busy time must not grow.** G6 had no effect.
G5 kept (+0.2–1.9%). Routed gate/up ≈155 GB/s, down ≈148 GB/s.

## Items

- **H1 revert G1.** Revert `874675c7` and `b3dd2b97` (HcDownSiluQ8/HcUpMix
  and their executor wiring/tests), restoring the B2 HC decode path. Keep
  `6b09c93c`'s unified rounding in `qfn_mmq.hip.cpp` and keep a split-invariance
  check (row 0 identical across widths 1..8) on the restored unfused path in
  `hc_mix_ops_test`. Resolve conflicts with later commits (G4d's MoE-in-combine
  touches HcCombine, keep it).
- **H2 revert G6** (`47bfa4e6`, `ce5fbda0`): no measured effect; keep code simple.
  Keep the G0 bench tool.
- **H3 HC decode fusion without serial tails (redo of G1, optional by design
  screen).** Only fuse where each block owns complete outputs:
  HC down (N=320): give each block whole Q8_1 groups (32 outputs) so it applies
  SiLU/scale and quantizes its own outputs — no arrival counter. HC up + mix:
  only if one block can own a column chunk with all the gate rows it needs
  (all streams) so the mix epilogue runs in-block with full parallelism; if the
  gate layout makes that require redundant work, skip. Before wiring, write a
  cost estimate (bytes, waves/CU, instructions) vs the B2 kernels' measured
  ms; skip if it cannot plausibly be ≤ B2 busy time. Add both variants to a
  standalone bench `tools/bench/flash_next_hc_decode.hip` (B2-style unfused
  sequence vs fused, real shapes K=hc dim/N=320 and back, R=1 and R=8) so the
  final pass measures them directly; the final pass will keep the faster one.
- **H4 routed GEMV activation staging (open G5 lever).** Activation loads at
  their use force `vmcnt(0)` and drain the prefetched trellis group. Stage the
  block's activation slice (R rows × K FP16) in LDS (or registers) once before
  the K loop, with an explicit barrier, so the K loop only issues trellis
  global loads; check ISA for `s_waitcnt vmcnt` placement, VGPR/LDS/scratch for
  R=1..8 gate/up and down, and keep occupancy. Also look at the restore tail
  in `RoutedGateUpRestore` (≈0.45 ms/token in B3): if it is a serial
  last-block tail, make it parallel or move restore into the down GEMV head.
- **H5 G4a/G4d/G2/G3 tail audit.** Inspect the remaining last-block tails
  (router top-k, SsmConv4 q/k norm, HcCombine MoE) for serial work on the
  critical path; report estimated tail µs per call. Change only if clearly
  better (e.g. widen the tail to the whole last block's threads).

Order: H1, H2 (quick), H4, H3, H5. Report per item: dispatches/token, expected
busy-time change with reasoning, resources, lossy changes, and the exact GPU
tests/benches the final pass must run.
