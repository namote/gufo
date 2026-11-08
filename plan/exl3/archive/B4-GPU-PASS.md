# B4 combined GPU pass (2026-11-08)

Same procedure as `plan/exl3/archive/B3-GPU-PASS.md` (→ B2 → B1 procedure), with
these B4 changes (they override). Results go to `build/b1/gpu-b4/`. The user
already stopped the live server for this window (`llm status` shows stopped);
restarting it at the end is mandatory.

Trees (build all three binaries before any GPU timing; save copies):
- **B** baseline = B2 candidate binary `build/b1/b2-cand-gufo` (`9599b4c6`).
- **C** candidate = `ai-tools/port-u4f` HEAD (B4 items H1–H5 on top of B3).
  Save as `build/b1/b4-cand-gufo`.
- **S** simple = branch `b4-simple` (C with G3/G4a/G4b/G4d and the B4 code
  touching them reverted). Build it by switching the main checkout to
  `b4-simple`, building into a copy of `build/b1/obj` (e.g.
  `build/b1/obj-simple`, adapt build.sh via a copy), save as
  `build/b1/b4-simple-gufo`, then switch back to `ai-tools/port-u4f`. No
  worktrees. Leave the checkout on `ai-tools/port-u4f`.

Steps:
1. GPU unit tests on C and on S (S: only the tests its reverts touch:
   `exl3_moe_kernel`, `gdn_ops`, `projection_ops`, `hc_mix_ops`,
   `exl3_invariance`, `mtp_kv_executor`): `ctest --preset gpu-full -R
   '^qwen38_flash_next\.(hc_mix_ops|gdn_ops|exl3_moe_kernel|projection_ops|
   exl3_ops|exl3_invariance|exl3_shared_invariance|exl3_producer_rotation|
   attention_ops|mtp_kv_executor|mtp_kv|kvarn_ops|moe_epilogue_ops|
   routed_wmma_ops|snapshot_copy|exl3_prefill_executor|exl3_gdn_prefill|
   exl3_prefill_overlap)$'` plus `qwen38_flash_next_exl3_moe_kernel_test
   --rate-generalization`. Known pre-existing failure: gdn_ops "fused stable GDN
   differs from the separate convolution". Fix real failures (separate commits,
   on the right branch), rebuild only affected targets.
2. HC bench (C tree): build target `qwen38_flash_next_hc_decode_bench`, run
   it; keep its table (B2 down+SiLU vs fused down, chain, R=1..8).
3. Quality: collect once on C (32,760 positions); compare to B2/B3 (B3 logits
   sha `17dc8f9f…`). B4 is designed bit-identical; if C's logits sha equals
   B3's, S need not be collected. Otherwise collect S too.
4. Perf: rotate B C S B C S, r2 each, same bench args: TG n128 d0/d32768 AR
   and MTP (draft 7); PP 2048/8192 off and MTP only for B and C. Report
   per-row tok/s, % vs B, C vs S, verdict at 5% / 10 ms. Hashes and MTP
   draft/accept/cycle counts for each.
5. Decode profile (AR d0) for C and S: dispatches/token, kernel-active %,
   idle ms/token, busy ms/token, top kernels/categories, routed gate/up and
   down ms and GB/s (B3: 5.08 ms ≈155 GB/s, 2.66 ms ≈148 GB/s), HC chain ms.
6. Restart the service (`llm start`, ready, a short chat answers).

Do not edit `docs/`; do not merge branches or revert items — report data and a
recommendation (C, S, or C with specific items reverted, e.g. H3 if the HC
bench shows the fused down slower than B2's down+SiLU). The parent decides.
Write `build/b1/gpu-b4/RESULT.json`.
