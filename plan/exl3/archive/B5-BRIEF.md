# B5 brief: Flash-Next decode, dense EXL3 GEMV, routed down, hiding ramp/drain idle (2026-11-08)

Same rules as `plan/exl3/archive/B1-BRIEF.md` (Rules, Numerical policy, build
`build/b1/build.sh`, pinned clang-format via `build/b1/build.sh --run`,
commit/attribution, report). Chunk/split invariance is a contract. No
GPU/model/timing runs while iterating. Main checkout `/home/m2on/projects/gufo`,
branch `ai-tools/port-u4f`, one item at a time, second-reviewer subagent review
per item. Every item in its own commit(s) so the final pass can revert it alone.

Read KNOWHOW "Flash-Next B4 …" and "Flash-Next B3 …" first. Lessons: a removed
dispatch is worth only ≈3 µs idle; a fusion must not grow busy time (no serial
or redundant tail work); intended-exact changes must be proven exact by a unit
test that compares the real decode path bytes (B4's H4b changed decode bits and
no test caught it).

Current (B4 adopted tree, AR d0 profile `build/b1/gpu-b4/prof-S/decode-summary.json`):
30.93 ms/token, 1273 dispatches, busy 20.65 ms, idle 10.28 ms. ms/token:
dense/shared EXL3 GEMV 12.98 (Exl3GemvKernel<5,1,4,2,0> 4.37 over 36 calls,
<6,1,4,2,0> 2.97/48, <6,1,8,2,0> 2.75/48, <6,1,4,4,0> 1.30/12, <6,1,16,2,1>
1.24/48), routed gate/up 4.54 (≈174 GB/s) + restore, routed down 2.55
(≈155 GB/s), lm head K8 2.74 (≈230 GB/s), HC chain 3.87, SmallGemm 1.15/86
calls, GDN 0.83, HcCombine 0.68, router top-k 0.58, RoutedCombine 0.45.

## Items

- **J1 dense EXL3 GEMV bandwidth.** For each instantiation above, compute
  bytes/call from the real shapes and the achieved GB/s, then inspect ISA
  (`tools/prof/isa_mix.py`, `s_waitcnt vmcnt` placement, loads in flight,
  VGPR/occupancy). Apply what made B4 H4a work for the routed GEMV (activation
  slice staged in LDS so the K loop only issues trellis loads and keeps the next
  group in flight), or a better prefetch, where the analysis says the kernel is
  below ≈220 GB/s for latency reasons. Keep per-output summation order (exact).
- **J2 routed down short K (K=640, ≈155 GB/s).** Analyze why it trails gate/up
  (blocks per expert, waves/CU, per-block setup vs 640-element work, tail).
  Propose and implement a shape (more output rows per block, multiple experts
  per block, deeper prefetch) that keeps order exact.
- **J3 overlap independent decode work with graph branches.** Idle is ramp/
  drain and dependency latency, which concurrent branches hide (B2 E7 side-stream
  shared MLP paid +1.1–1.5%; a fork costs ≈0.1 µs). Find independent kernels in
  a decode layer (e.g. GDN projections that read the same input, attention q vs
  k/v, sparse indexer vs other prep, router GEMV vs anything not consuming it)
  and run them on captured side-stream branches joined before their consumer.
  Scratch must not alias across branches; keep the existing E7 branch working.
  Report the expected overlapped kernels per layer.
- **J4 SmallGemm (86 calls/token, 1.15 ms).** Identify each call site; if a
  call can join an existing branch/launch without growing busy time, do it;
  otherwise report.
- **J5 MTP draft head audit (report only).** Does each MTP draft step run the
  full-vocabulary K8 lm head? Count lm-head reads per MTP cycle and estimate
  what a K6 head (≈−0.7 ms per read) would give. No quantization.

Order: J1, J2, J3, J4, J5. Report per item: expected ms/token change with
reasoning, dispatches, VGPR/LDS/scratch, exactness evidence (which test
compares which bytes), and the exact GPU tests/benches the final pass runs.
If a standalone bench is cheap (`tools/bench/` or a CMake bench target as in
B4's `qwen38_flash_next_hc_decode_bench`), add it for J1/J2 old-vs-new.
