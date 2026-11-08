# Flash-Next EXL3 B-series summary (B1–B8, 2026-11-07 to 2026-11-08)

Index of the c03 Flash-Next decode/prefill optimization batches on
`ai-tools/port-u4f`. Measurements and lessons live in
[docs/GFX1151-KNOWHOW.md](../../docs/GFX1151-KNOWHOW.md) (entries
"Flash-Next B1" … "B7"); briefs and GPU-pass notes are linked per batch.
All numbers: c03 pack, pinned Nix ROCm 10.1, FP16 KV, matched windows.

| Batch | Goal | Kept | Reverted / rejected | Measured result (vs previous batch) | Docs |
|---|---|---|---|---|---|
| B1 | Decode glue, HC Q8 traffic | A3 HC-down deep steps, A1 GDN history fused into SsmConv4, B1 routed glue folded, B2 four routed waves/block, B4 F16 mixed, C1 IEEE Q8_0 scales, C2/C3 Q8_0 HC down/up, C4 F16 HC gate | A1 kPrep q/k norm in GdnKernel (chunk/checkpoint invariance); A4 GPU argmax (≈50 µs) | PP +2.4–3.2%, AR +13.6%, MTP +7.9–18.2%; quality unchanged (KL256 0.041742) | [brief](B1-BRIEF.md), [pass](B1-GPU-PASS.md) |
| B2 | Producer rotation, verify state, side stream | E1 producer-side X2b rotation, E4 R5–R8 live-row bodies, E6 GDN verify records, E7 shared MLP side branch, E2 deferred attention residual | E3 HC-up+mix fusion, E5 device-chained drafts | PP +0.6–0.9%, AR +2.5%, MTP +4.3–4.9%; bits equal B1 | [brief](B2-BRIEF.md), [pass](B2-GPU-PASS.md) |
| B3 | Decode dispatch count, launch gaps | G2 one-launch attention prep, G5 routed next-group loads, single-row Q8_0 rounding unified | G1 HC tails, G3, G4a/b/d, G6 device kernargs (all undone in B4) | AR −1.0..−1.2%, MTP −2.8..−3.1% (busy grew); launch gap ≈2 µs, idle is ramp/drain | [brief](B3-BRIEF.md), [pass](B3-GPU-PASS.md), [persistent decode (not pursued)](B3-PERSISTENT-DECODE.md) |
| B4 | Undo serial tails, routed loads | H1/H2 reverts of G1/G6, H4a routed LDS staging, H3 HC down finishes its Q8_1 group per block | G3/H5, G4a, G4b/H4b (changed decode bits), G4d; HC up+mix fusion skipped | AR +2.2–2.3%, MTP +3.8–5.4% vs B2; PP unchanged | [brief](B4-BRIEF.md), [pass](B4-GPU-PASS.md) |
| B5 | Dense GEMV, routed down, idle | J2a route tiles out of LDS, J2b two tiles per routed down wave, J3 alpha/beta and indexer side branches | J1 dense GEMV prefetch (slower on large shapes) | AR +1.3/+2.7%, MTP −0.6/+0.9% vs B4; bits equal | [brief](B5-BRIEF.md), [pass](B5-GPU-PASS.md) |
| B6 | Prefill | P1 streaming MoE slab stores, P3 packed attention gate, P5 streaming HC residual stores | P7 shared-MLP side-stream policy (CU masks −6..−17%) | PP2048/8192 +0.3/+0.5%; bits equal B5 | [brief](B6-BRIEF.md), [pass](B6-GPU-PASS.md) |
| B7 | Prefill GEMM ceiling screen | bench `gfx1151_exl3_wmma_ceiling.hip` | warp-specialized rewrite (−12..−47%), 8192 chunk cap (+1.3% for +2.1 GiB) | best case ≈1770 tok/s PP8192 with this GEMM structure | [brief](B7-CEILING-BRIEF.md), [result](B7-CEILING-RESULT.md) |
| B8 | Wrap-up cleanup | — | dense GEMV reference copy, its test and bench | no production change (stripped code compared) | [brief](B8-CLEANUP-BRIEF.md) |
