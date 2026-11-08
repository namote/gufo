# B7 brief: EXL3 prefill GEMM ceiling screen on gfx1151 (2026-11-08)

Decide with data whether a prefill GEMM redesign can pay before writing it.
Same rules as `plan/exl3/archive/B1-BRIEF.md` (build `build/b1/build.sh`, pinned
clang-format, commits/attribution). No GPU runs while implementing; one short
GPU window runs the benches afterwards. Main checkout, branch
`ai-tools/port-u4f`, second-reviewer subagent review per item.

Context (KNOWHOW "Flash-Next B6 …", "c03 prefill profile before B6",
"Routed MoE ceiling diagnosis"; `plan/exl3/archive/F3-SEARCH-STATUS.md`,
`F3-MOE-CEILING-DIAGNOSIS.json`, `F3-SESSION-FINDINGS.md`): PP8192 5276 ms,
routed gate/up 934 ms (27 TF), down 534 (23.6 TF), dense EXL3 ≈1220 (33–37
TF); WMMA FP16 register-fed reference ≈55 TF. Routed TF grew 22.4→26.8 from
2048- to 4096-row chunks (≈40→80 rows per expert). Historical ablation:
removing decode 3.6→3.25 ms, decode+weight fetch →2.85 ms (old kernels).
Hypotheses: (1) in-kernel trellis decode competes with WMMA for issue,
(2) too few rows per expert, (3) low LDS reuse in routed tiles.

## Items

- **Q1 ceiling bench** `tools/bench/gfx1151_exl3_wmma_ceiling.hip` (or a CMake
  bench target if it must link model kernels). Real c03 shapes: routed gate/up
  K2560→N640 (×2 projections) and down K640→N2560 at 40/80/160/320/640 rows per
  expert with 10-of-512 routing (synthetic, balanced and skewed); dense
  K2560→N10240, K6144→N2560, K2560→N6144 at 2048/4096/8192 rows. For each,
  time (µs, TF, exact-output check where applicable):
  a. the **production** kernels via their real launchers (current baseline);
  b. **no-decode**: same tiling/WMMA with pre-decoded FP16 weights read from
     global (upper bound with today's tiling and fetch);
  c. **WMMA-only**: weights already in LDS/registers (tiling ceiling);
  d. **decode-only**: the trellis decode with stores but no WMMA (decode cost);
  e. **warp-specialized prototype**: producer waves decode EXL3 tiles into an
     LDS ring (double/triple buffered), consumer waves only issue WMMA; at
     least for routed gate/up K4/K5 and dense K5/K6 at 4096 rows. Exact output
     vs (a) is required (same per-output K order), or state the difference.
  Print one table per shape; record VGPR/LDS/occupancy of each variant.
- **Q2 chunk-size lever.** Find the prefill chunk cap (why PP8192 runs as
  2×4096) and its memory cost. If raising it is a config/constant change,
  prepare a variant build with the cap at 8192 (and 16384 if memory allows)
  and compute the extra activation/workspace bytes. Do not change the default.
- **Q3 decision note** `plan/exl3/archive/B7-CEILING-RESULT.md` template: what result
  in Q1/Q2 justifies which redesign (e.g. if (e) ≥ 1.25× (a) on routed and
  dense, rewrite production kernels around it; if (b) ≈ (a), decode is not the
  limit; if (c) ≤ 35 TF, tiling is the limit), plus expected PP8192 ms.

Report: commits, how to build and run each bench (exact commands), the
variant build for Q2, resources, and what the GPU window must run.
