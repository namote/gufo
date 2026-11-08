# B7 result: EXL3 prefill GEMM ceiling screen (template, 2026-11-08)

Brief `plan/exl3/B7-CEILING-BRIEF.md`. Fill in after the GPU window; keep
every number with its row of `build/b1/gpu-b7/` (bench stdout `SUMMARY`
lines). Bench `qwen38_flash_next_exl3_wmma_ceiling_bench`
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

## Q1 tables (fill)

Per shape: µs and TF for a, a*, b, c, c0, d, e-bar (both routed producer
counts), e-flag; dg; exactness strings; resources. a* must equal a (EQUAL),
and `a*` time within ±3% of `a` (routed a* has production's VGPR/LDS:
121/169 VGPR, 24/32 KiB; dense a* 192 vs production 187 VGPR — note any
time gap).

| Shape | rows | K | a TF | a* | b | c | c0 | d µs | e-bar | e-flag | e exact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| routed gate/up balanced | 80/expert | K4 | | | | | | | | | |
| routed gate/up skewed | 80/expert | K4 | | | | | | | | | |
| routed down balanced | 80/expert | K4 | | | | | | | | | |
| dense GDN qkv 2560→10240 | 4096 | K5 | | | | | | | | | |
| dense GDN out 6144→2560 | 4096 | K6 | | | | | | | | | |
| dense z/q 2560→6144 | 4096 | K5 | | | | | | | | | |

Rows-per-expert scaling (routed a TF at 40/80/160/320/640, balanced and
skewed) goes in its own table; it predicts the Q2 gain without a model run.

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

| Binary | PP2048 | PP8192 | PP16384 | routed TF (profile) | Δ vs 4096 |
|---|---|---|---|---|---|
| chunk4096 | | | | | |
| chunk8192 | | | | | |
| chunk16384 | | | — | | |

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

Decision: (fill) — chosen item(s), predicted and measured PP8192 ms.
