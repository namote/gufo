# B3 persistent decode design (Flash-Next EXL3, design only)

Status: design, no implementation. Inputs: B2 decode profile
(`build/b1/gpu-b2/prof/decode-summary.json`: 31.78 ms/token, 1442
dispatches/token, kernel-active 67 %), the B3 dispatch cuts (G1–G4) and the
G0 microbenchmark `tools/bench/gfx1151_launch_gap.hip` (numbers to be filled
in by the final GPU pass; symbols below).

## 1. Where decode stands after B3 (R = 1, counted from the code path)

Per GDN layer (36 layers), main stream unless noted:

| # | Launch | Kind |
|---|---|---|
| 1 | HcCombineKernel (mixer combine; forms the previous MoE output in LDS) | elementwise + 4 row norms |
| 2 | HcDownSiluQ8Kernel (320 × 10240 Q8_0, SiLU/Q8_1 tail) | GEMV, 3.4 MB |
| 3 | HcUpMixKernel (10240 × 320 Q8_0, mix epilogue + rotation tail) | GEMV, 3.4 MB |
| 4 | Exl3Gemv qkvz (K5) | GEMV, ≈ 8 MB |
| 5 | SmallGemm alpha/beta | GEMV, small |
| 6 | SsmConv4PrepKernel (conv + q/k norm tail) | elementwise |
| 7 | GdnKernel | recurrence, state 3.1 MB R/W |
| 8 | GdnEpilogue (+ rotation for ssm_out) | elementwise |
| 9 | Exl3Gemv ssm_out (K6) | GEMV |
| 10–12 | FFN combine, HC down, HC up (as 1–3) | |
| 13 | RouterGemvTopKTiles (router + top-k + tiles tail) | GEMV + select |
| 14 | RotateAssignments | elementwise |
| 15 | RoutedGateUpRestore (gate/up + restore/scale tail) | GEMV, ≈ 16 MB |
| 16 | RoutedGemv down | GEMV, ≈ 8 MB |
| s1–s2 | shared gate/up, shared down (side stream, joined before 1) | GEMV |

18 launches per GDN layer (16 main + 2 side). Attention layers (12) replace
4–9 by q/k/v GEMVs, PrepareAttention, attention + merge, output gate,
attn_out GEMV and the indexer launches: ≈ 22 per layer. Head, embedding,
PLE: ≈ 10 per token. Total ≈ 36 × 18 + 12 × 22 + 10 ≈ 920–1000 per token
(B2: 1442).

## 2. What a persistent layer kernel looks like

One cooperative launch per layer (or per two layers), `G = 40 × b` blocks
(b = 1–2 blocks per CU, 256 threads), staying resident for the whole layer.
Each stage is a grid-strided loop over the same work items the current
kernels have (rows of a GEMV, 128-column segments of an elementwise stage);
stages are separated by a grid barrier (atomic arrival + generation word, as
in the G0 bench). Per-wave arithmetic stays the current kernels' (the same
device bodies, called with a virtual block index), so outputs keep their bits.

Barriers needed per GDN layer (a barrier is needed only where a stage reads
a full row another block produced):

| Boundary | Why a barrier |
|---|---|
| combine → HC down | down reads the whole 10240-wide norm |
| HC down → HC up | up reads all 320 lo values (Q8_1 tail) |
| HC up → qkvz GEMV | GEMV reads the whole mixed row (rotation slabs) |
| qkvz + alpha/beta → conv/prep | conv reads all channels of the token |
| prep → GdnKernel | recurrence reads normalized q/k of every head |
| GdnKernel → epilogue | per-head; can fuse into GdnKernel tail per head (no grid barrier) |
| epilogue → ssm_out | GEMV reads whole row |
| ssm_out → FFN combine | combine row norm |
| combine → down → up | 2 (as above) |
| up → router | router reads whole mixed row |
| router/top-k → gate/up | tile list |
| gate/up → down | per-expert tile (already tail-fused; a tile-local flag suffices) |
| down → next layer combine | combine reads all experts' rows |

≈ 12 grid barriers per GDN layer instead of 16 launches. The saving is
therefore not "launches → zero" but `launch gap − barrier cost` per
boundary, plus the tail work that today runs in one block (top-k, restore,
Q8_1 quantization) spreading over the grid.

## 3. Expected saving (fill in from G0)

Let `g` = graph-replayed µs/dispatch for a dependent chain (G0 Part 1,
40 × 256 grid, 16 B args, with fork) and `β` = µs/barrier at the chosen
blocks per CU (G0 Part 2). Then per token

    saving ≈ N_boundaries × (g − β) − overheads,
    N_boundaries ≈ 36 × 12 + 12 × 16 ≈ 620.

The B2 profile implies ≈ 7 µs idle per dispatch under the tracer
(≈ 10.5 ms idle / 1442). If G0 gives `g` ≈ 5–7 µs and `β` ≈ 1–2 µs, a
persistent layer saves ≈ 620 × 4–5 µs ≈ 2.5–3 ms/token (≈ 8–10 % at
31 ms/token) on top of B3. If `β` is close to `g` (barrier latency
dominated by the same L2 round trips as a dispatch), the design does not pay
and the remaining lever is fewer, larger fused kernels (G1–G4 style tails).

Overheads to subtract: lost occupancy of the bandwidth-bound GEMVs when
confined to `G` resident blocks (today HC down uses 320 one-wave blocks, HC
up 1280 × 8 waves, routed gate/up 200 × 4 waves); every GEMV stage must keep
≥ 8 waves per CU in flight, so `b ≥ 2` with 256 threads, or each wave
must carry several rows (MLP per wave as in A3/B2).

## 4. Register / LDS budget

The union kernel's VGPR count is the maximum over stages. Current peaks
(gfx1151 resource dump): routed gate/up + restore R=1 62 VGPR, R=8 82;
Exl3Gemv K6 ≈ 60–90; GdnKernel ≈ 70; HcUpMix R=1 44, R=2–8 87. A single
persistent kernel would sit at ≈ 90 VGPR → ≤ 16 waves/SIMD, which is fine
for occupancy (G0's `b ≤ 4` × 8 waves = 32 waves/CU = 8 per SIMD). LDS: the
largest stage users are HcCombine-with-MoE (10 KB), RouterTopK (4 KB probs +
small) and the rotation rows (1 KB); ≤ 16 KB per block, no limit at b ≤ 4.
Scratch must stay 0: stage bodies must not be selected through a runtime
switch that keeps every stage's arrays live (B1 lesson: dynamically indexed
locals promoted to 40 KiB LDS).

## 5. What stays separate

- The lm head (2.7 ms, 7760 blocks, ≈ 230 GB/s): already bandwidth-bound,
  one launch per token; nothing to gain.
- The shared MLP on the side stream: a persistent main-stream kernel would
  have to absorb it (it currently overlaps the routed GEMVs, +1.1–1.5 % TG
  in B2). Either fold it into the routed stages as extra work items (shared
  expert = an eleventh "expert" with its own weights) or keep the side stream
  and give the persistent kernel `40 × b − k` blocks.
- Attention (AttentionKernel, split-K merge): keep as its own launch at first;
  its grid (12 × 8) and LDS use differ from the GEMV stages.
- Verify widths R > 2: the tails that run per token in one block (top-k,
  GdnPrep) do not scale; in a persistent kernel they become grid-strided,
  which is the main benefit for MTP.

## 6. R ≤ 8 (MTP verify)

Same stage list; GEMV stages already take 1–8 rows per wave. Stage work
items grow with R only for elementwise stages. Graph keys per R stay (one
cooperative launch per layer per R).

## 7. Risks

- Co-residency: a cooperative launch is required (`hipLaunchCooperativeKernel`,
  refused when the grid cannot be resident); inside hipGraph capture the
  cooperative node must be supported by the pinned runtime (check first).
- Barrier cost under load: G0 measures an idle GPU; with GEMV traffic in
  flight the atomic round trip to L2 lengthens.
- Lost overlap: today consecutive kernels can overlap at their edges
  (B2 overlap ≈ 1.1 ms/token of concurrent kernel time); a barrier forbids
  that.
- Bit stability: every stage must call the existing device bodies with the
  same per-wave mapping; virtual block indices must reproduce `blockIdx`
  exactly where reductions depend on block geometry (HC mix inject parts,
  BlockSum orders).
- Rollback/snapshot and graph replay semantics are unchanged as long as the
  kernel writes the same buffers; counters must wrap like the B3 tails.
- Debuggability: a hang in one stage hangs the layer; keep bounded spins with
  an error flag in development builds.

## 8. Suggested order if G0 says it pays

1. Prototype the FFN half (combine → HC down → HC up → router → gate/up →
   down) as one cooperative kernel in a bench against the B3 launch chain,
   R = 1, with the measured `g` and `β`.
2. Extend to the GDN mixer; fold GdnEpilogue into a per-head GdnKernel tail.
3. Attention layers last.
