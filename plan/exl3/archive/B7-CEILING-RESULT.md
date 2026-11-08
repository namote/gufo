# B7 result: EXL3 prefill GEMM ceiling screen (2026-11-08)

Brief `plan/exl3/archive/B7-CEILING-BRIEF.md`. Filled from the GPU window of
2026-11-08 (live server stopped, restarted after). Every number is kept
in `build/b1/gpu-b7/` (bench stdout `SUMMARY` lines).
Bench `qwen38_flash_next_exl3_wmma_ceiling_bench`
(`tools/bench/gfx1151_exl3_wmma_ceiling.hip`). Chunk variants
`build/b1/b7-chunk{4096,8192,16384}-gufo`.

## Priors that bound the outcome (do not re-measure)

- WMMA and VALU do not overlap on one SIMD (sum, not max:
  `tools/bench/gfx1151_wmma_valu_overlap.hip`, F3-SESSION-FINDINGS; KNOWHOW
  "Tried and rejected": producer/consumer specialization). So (e) cannot
  win by co-issuing decode and WMMA. It can only win by hiding latency or
  barrier stalls that (a) cannot hide: the gap between `a` and
  `max(c, d)` or `c + d` below tells which.
- LDS-fed WMMA ceiling by fragment reads per WMMA: 1.0 → 46–48 TF, 2.0 →
  28–30 TF (`gfx1151_wmma_lds_bench.hip`). Routed 8x1 waves read 1 A + 1 B
  fragment per WMMA at 4 WMMAs per A, i.e. 1.25 per WMMA → ≈39–44 TF.
  Dense 8x4 waves (2 x 4 tiles) read 0.75 per WMMA → ≈46–49 TF.
- Cached raw FP16 W′ in today's X3 kernel (≈ variant b) was exact but slower
  on large dense K7 shapes (−57%/−7%) and +3% at K640/N2560 (KNOWHOW, Dense
  X3 notes). Expect `b` ≥ `a` on dense; routed down is the only shape where
  `b` < `a` was seen.


## Q1 tables

Raw: `build/b1/gpu-b7/ceiling/stdout.log` (504 `SUMMARY` lines), parsed
table `build/b1/gpu-b7/ceiling-table.txt`. TF columns are TF; `d` is µs
(its TF is GEMM-equivalent and meaningless). Exactness: every `a*`, `b`,
`e-bar` and `e-flag` row in all 58 shapes printed `EQUAL` (routed
gate/up `EQUAL / EQUAL`); e-flag never hung or timed out. `a*` within ±3%
of `a` everywhere (max gap −2.6%, dense out K5 2048 rows; at the table
shapes ≤ 1.7%). The bench reports routed a* at 111/156 VGPR (64/128 tile)
rather than the 121/169 noted above; times match `a`, so taken as
equivalent.

| Shape | rows | K | a TF (µs) | a* | b | c | c0 | d µs | e-bar 4P / 8P | e-flag | e exact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| routed gate/up balanced | 80/expert | K4 | 29.19 (9195) | 29.48 | 15.99 | 39.64 | 42.76 | 3579 | 22.10 / 25.69 | 22.40 | EQUAL |
| routed gate/up skewed | 80/expert | K4 | 27.11 (9903) | 27.55 | 15.46 | 36.00 | 38.70 | 3758 | 20.05 / 22.27 | 21.30 | EQUAL |
| routed gate/up balanced | 80/expert | K5 | 28.40 (9451) | 28.66 | 16.01 | 39.56 | 42.31 | 4533 | 21.02 / 24.03 | 21.28 | EQUAL |
| routed gate/up skewed | 80/expert | K5 | 26.44 (10153) | 26.60 | 15.50 | 35.96 | 38.50 | 4666 | 19.10 / 21.12 | 20.29 | EQUAL |
| routed down balanced | 80/expert | K4 | 26.96 (4979) | 26.79 | 15.84 | 34.15 | 35.42 | 1931 | 17.43 / 19.02 | 18.82 | EQUAL |
| routed down skewed | 80/expert | K4 | 24.58 (5461) | 24.60 | 15.24 | 31.26 | 32.52 | 1991 | 16.76 / 18.05 | 18.01 | EQUAL |
| routed down balanced | 80/expert | K5 | 26.13 (5136) | 25.98 | 15.84 | 33.56 | 35.44 | 2245 | 17.20 / 18.67 | 18.13 | EQUAL |
| dense GDN qkv 2560→10240 | 4096 | K5 | 36.73 (5846) | 36.33 | 38.52 | 38.89 | 38.70 | 637 | 28.02 (16C+8P r2) | 31.00 | EQUAL |
| dense GDN qkv 2560→10240 | 4096 | K6 | 36.34 (5909) | 36.62 | 38.63 | 38.89 | 38.59 | 629 | 28.41 | 30.92 | EQUAL |
| dense GDN out 6144→2560 | 4096 | K6 | 35.02 (3680) | 34.96 | 34.48 | 37.26 | 42.70 | 357 | 18.42 | 17.63 | EQUAL |
| dense GDN out 6144→2560 | 4096 | K5 | 34.17 (3770) | 34.44 | 34.68 | 36.38 | 42.67 | 376 | 19.05 | 19.15 | EQUAL |
| dense z/q 2560→6144 | 4096 | K5 | 34.95 (3686) | 35.28 | 37.36 | 37.90 | 37.41 | 386 | 27.65 | 29.58 | EQUAL |
| dense z/q 2560→6144 | 4096 | K6 | 35.23 (3657) | 34.81 | 37.16 | 37.99 | 37.40 | 367 | 27.42 | 29.50 | EQUAL |

Dense 2048/8192 rows match 4096 within a few % (a 34–39 TF, c0 38–43,
e 0.53–0.87x of a); see `ceiling-table.txt`.

Resources (threads / VGPR / LDS / blocks per CU; routed 64-tile | 128-tile):
a* 256/111/24 KiB/5 | 256/156/32 KiB/4; b 117 | 162 VGPR; c 86 | 130; c0
71 | 103; d 44 VGPR, 16 KiB, 8 blk; e-bar 8C+4P 384 thr 116 | 142 VGPR,
48 | 64 KiB, 2 blk; e-bar 8C+8P 512 thr 68 | 97 VGPR, 2 blk; e-flag
8C+4P r3 384 thr 72 | 95 VGPR, 36 | 48 KiB, 3 | 2 blk. Dense: a* 1024 thr
192 VGPR 64 KiB 1 blk; b 166; c 179 (occupancy API printed 0 blk/CU, the
kernel ran); c0 118; d 36 VGPR 32 KiB 2 blk; e-bar 16C+8P r2 768 thr 192
VGPR 64 KiB 1 blk; e-flag r3 768 thr 190 VGPR 48 KiB 1 blk.

Derived ratios (time; 80 rows/expert, dense 4096):

| Family | a/c | a / max(c,d) | a / (c+d) | (a−c)/d exposed decode | a/e best |
|---|---|---|---|---|---|
| routed gate/up | 1.33–1.39 | same (c > d) | 0.84–0.89 | 0.58–0.68 | 0.80–0.88 |
| routed down | 1.27–1.30 | same | 0.82–0.87 | 0.51–0.59 | 0.71–0.74 |
| dense (3 shapes, K5/K6) | 1.06–1.08 | same | 0.95–0.97 | 0.51–0.74 | 0.53–0.85 |

Rows-per-expert scaling, routed `a` TF (c / e-bar 8C+8P in brackets for K4):

| Shape | dist | 40 | 80 | 160 | 320 | 640 |
|---|---|---|---|---|---|---|
| gate/up K4 | balanced | 22.21 (31.98/15.52) | 29.19 (39.64/25.69) | 29.56 (40.14/26.08) | 31.49 (40.04/29.09) | 34.24 (42.45/32.01) |
| gate/up K4 | skewed | 21.42 | 27.11 | 30.46 | 31.42 | 32.38 |
| gate/up K5 | balanced | 21.09 | 28.40 | 28.95 | 30.84 | 33.52 |
| gate/up K5 | skewed | 20.16 | 26.44 | 29.77 | 30.79 | 31.89 |
| down K4 | balanced | 19.54 | 26.96 | 27.23 | 28.95 | 31.19 |
| down K4 | skewed | 19.07 | 24.58 | 27.60 | 28.53 | 29.48 |
| down K5 | balanced | 18.99 | 26.13 | 26.79 | 28.65 | 30.94 |
| down K5 | skewed | 18.38 | 23.88 | 27.17 | 28.12 | 29.27 |

Predicted Q2 gain (rule 6 formula, gate/up 934 + down 534 ms): balanced
934·(1−29.19/29.56) + 534·(1−26.96/27.23) ≈ 17 ms; skewed ≈ 103 + 58 =
161 ms.

## Q2 chunk cap

Cap: `kPrefillChunkTokens = 4096` (`src/models/qwen38_flash_next/engine.cpp`)
→ `PrefillCapacity()`; `max_batch = PrefillThroughCapacity()` = cap + 128
sizes every executor buffer. PP8192 = 2 × 4096 (Session::Feed). Serve also
admits at most 2048 tokens per step while another session is free
(`kAdmissionPrefillTokens`), so a larger cap helps bench/lone-session serve.
Device bytes scale ≈ 537 KiB per row: T=4224 2216.8 MiB, 8320 4366.2 MiB
(+2.10 GiB), 16512 8665.0 MiB (+6.30 GiB), plus per-session indexer K
52/104/208 MiB and host pinned 85/167/332 MiB. A bench with `-p 8192 -d 0`
sizes T = 8193 (context = prompt + 1).

Bench `-r 2 --speculative off -d 0` (args in `build/b1/gpu-b7/perf.sh`),
rotated 4096 → 8192 → 16384, then reversed; `build/b1/gpu-b7/perf/pp-*`.
The bench has no prompt cache directory and prints no logits/output
hashes. tok/s per round (r1 / r2), mean:

| Binary | PP2048 | PP8192 | PP16384 | routed TF (profile) | Δ PP8192 vs 4096 |
|---|---|---|---|---|---|
| chunk4096 | 1522.1 / 1517.1 (1519.6) | 1554.7 / 1549.2 (1552.0, 5278 ms) | 1529.8 / 1525.6 (1527.7) | B6: 27.0 / 23.6 | — |
| chunk8192 | 1517.1 / 1508.7 (1512.9, −0.4%) | 1573.2 / 1571.1 (1572.2, 5211 ms) | — | 29.39 / 26.24 | +1.30%, −68 ms |
| chunk16384 | 1509.6 / 1507.6 (1508.6, −0.7%) | 1571.7 / 1570.1 (1570.9) | 1570.3 / 1569.7 (1570.0, +2.8%) | — | +1.22% |

Device memory (host-sampled `mem_info_gtt_used` peak; VRAM carve-out
flat at 0.88 GB): PP runs 4096 70.71 GB, 8192 72.75 GB (+1.90 GiB; that
run stopped at context 8193 while 4096 ran to 16385), 16384 77.56 GB
(+6.38 GiB). TG runs at d32768: 4096 71.13 GB, 8192 73.43 GB (+2.14 GiB,
matches the +2.10 GiB estimate).

rocprofv3 PP8192 profile of chunk8192 (`build/b1/gpu-b7/prof-c8192`,
`profile-c8192.json`, one 8192-row chunk): span 5214.6 ms (1571.0 tok/s,
B6 5276.4). Routed gate/up 858.9 ms at 29.39 TF (B6 933.9, 27.0 TF), down
481.1 ms at 26.24 TF (B6 534.0, 23.6 TF): −128 ms routed. Offsets: HC
projections 307 → 372 ms (+65), MoE routing/rotation/activation 484 →
509 ms (+25); dense EXL3 1211 (B6 1223). Net −62 ms.

TG n128 (`perf/tg-*`, two rounds, tok/s r1 / r2):

| Binary | AR d0 | AR d32768 | MTP d0 | MTP d32768 |
|---|---|---|---|---|
| chunk4096 | 32.95 / 32.90 | 32.61 / 32.64 | 66.91 / 66.97 | 104.48 / 104.70 |
| chunk8192 | 32.92 / 32.98 | 32.62 / 32.64 | 67.02 / 66.95 | 105.02 / 104.79 |

Decode unchanged (all within 0.3%).

## Q3 decision rules

Let `S_r` = a/e speedup on routed gate/up+down (time-weighted 934:534 at
80 rows/expert), `S_d` = dense (time-weighted over the three shapes, 4096).

1. **Redesign around (e)** only if e is EQUAL and `S_r ≥ 1.25` and `S_d ≥
   1.25` (or ≥ 1.25 on one family whose time saved ≥ 200 ms at PP8192).
   Expected PP8192 ms = 5276 − 1468·(1 − 1/S_r) − 1220·(1 − 1/S_d)
   (1.25/1.25 → ≈4738 ms ≈ 1729 tok/s).
2. **Decode is not the limit** if `c ≈ a` (within 5%): the tiling/LDS/
   B-staging path bounds today's kernel; neither (e) nor faster decode
   pays. Next lever: fewer fragment reads per WMMA (tile shape) or Q2. Do
   not use `b` for this: it removes decode but moves 2.7-4x the weight
   bytes (FP16 tiles), so `b > a` can mean weight bandwidth, not decode.
   `b < a` (routed down) shows decode cost exceeding the extra bytes.
3. **Decode is the limit** if `a/c ≥ 1.25` (time ratio: c is ≥ 1.25x
   faster) and `a ≈ c + d` (issue sum, no overlap). Then cut decode
   instructions (ISA, cheaper hash/sum); specialization cannot help (no
   co-issue).
4. **Latency-bound** if `a > max(c, d)` by ≥ 20% while `a < c + d`: (e) or
   deeper async staging can pay; take (e) only if it already shows it.
5. **Tiling is the limit** if `c ≤ 35 TF` (routed) / `c0 ≤ 40 TF` (dense):
   redesign tiles (fragment reuse) before anything else.
6. **Q2** pays if chunk8192 PP8192 ≥ +3% with no PP2048/TG change and the
   2.1 GiB fits the deployment budget; predicted from routed a at 160 vs 80
   rows/expert: Δms ≈ 1468·(1 − TF80/TF160) (dense unaffected).

## Decision

Applied rules (80 rows/expert balanced K4 for S_r; dense 4096, best e per
shape for S_d):

1. **(e) redesign: rejected.** e is EQUAL everywhere but slower:
   S_r = 1468 / (934/0.880 + 534/0.705) = 0.81, S_d = 13212 / 18280 =
   0.72. Expected PP8192 = 5276 + 351 + 467 ≈ 6094 ms (−13%). Matches the
   prior: no WMMA/VALU co-issue, and specialization halves occupancy
   (2 blk/CU routed, 768-thread dense).
2. **Dense: decode is not the limit** (a/c 1.06–1.08, just over the 5%
   bar; decode-free bound −76 ms at PP8192). Rule 5 also fires for
   dense qkv and z/q: c0 37.4–38.7 TF ≤ 40, while out proj c0 is 42.7. Dense
   is bounded by tiling/fragment reuse at ≈37–39 TF.
3. **Routed: a/c ≥ 1.25 holds** (1.27–1.39), but a is 0.82–0.89 of c + d,
   not ≈ c + d.
4. **Routed: latency-bound by the letter of rule 4** (a ≥ 1.27 max(c, d),
   a < c + d). Its action is gated on (e) already showing the win, and
   (e) loses 12–30%, so deeper async staging is not taken. 51–68% of the
   decode issue cost is exposed in a. Cheaper decode instructions are the
   lever that remains (rule 3 action): decode-free bound S = 1468 /
   (934/1.358 + 534/1.267) = 1.32 → −358 ms → PP8192 ≈ 4918 ms (1666
   tok/s). Halving the exposed decode cost gives about −180 ms → ≈5096 ms.
5. **Routed down: tiling is also a limit.** c 34.2 TF ≤ 35 (skewed 31.3);
   gate/up c 39.6 is not. Routed c0 at 42.8/35.4 TF caps decode work at
   the 1.25 fragment reads per WMMA tile shape.
6. **Q2: does not pay.** chunk8192 PP8192 +1.30% (−68 ms; profile −62 ms,
   inside the 17–161 ms prediction), below the +3% bar. PP2048 −0.4% and
   chunk16384 PP2048 −0.7%. TG unchanged. Costs +2.1 GiB device per
   session (+6.3 GiB at 16384). PP16384 +2.8% with the 16384 cap is also
   below 3%. Default stays 4096.

Decision: no (e) rewrite, no chunk-cap change. Next items, in order:
(i) cut routed EXL3 trellis decode instructions in the routed bucket
GEMM (bound −358 ms, PP8192 ≈ 4918 ms; realistic ≈ 5100 ms); (ii)
fragment reuse for routed down and dense qkv/z tiles (rule 5; dense
decode-free bound only −76 ms, so dense gains need tiles that reach the
46–49 TF prior: 1220·(1 − 36/47) ≈ −290 ms). Against the priors: `b` is
0.55–0.64x of `a` on routed (down included), and 1.05–1.07x faster than
`a` on dense qkv and z/q (out ≈ a). Measured today: PP8192
5278 ms (chunk4096), 5211 ms (chunk8192, not adopted).
