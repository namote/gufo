# B5 combined GPU pass (2026-11-08)

Same procedure as `plan/exl3/B4-GPU-PASS.md` (→ B3 → B2 → B1), with these
B5 changes (they override). Results in `build/b1/gpu-b5/`. Restart the service
at the end (mandatory), normal configuration, no new binary deployed.

Trees (build all binaries before any GPU timing; main checkout only, use
temporary local branches + separate obj copies for variants, delete the
branches afterwards, leave the checkout on `ai-tools/port-u4f`):
- **B** = B4 adopted tree binary `build/b1/b4-simple-gufo` (source equals
  `e8e938ed` minus docs).
- **C** = HEAD (J1 `62e9b525`, J2a `4fca55f5`, J2b `3a62d24c`, J3 `f38465cc`).
  Save `build/b1/b5-cand-gufo`.
- Isolation variants, each C with one commit reverted: **C−J1**, **C−J2b**,
  **C−J3** (J2a is a fix and stays).

Steps:
1. GPU unit tests on C: `ctest --preset gpu-full -R
   '^qwen38_flash_next\.(exl3_ops|exl3_moe_kernel|exl3_dense_gemv_pipeline|
   exl3_decode_side_branches|exl3_shared_invariance|exl3_invariance|
   exl3_producer_rotation|mtp_kv_group|mtp_kv_executor|projection_ops|
   hc_mix_ops|gdn_ops|attention_ops)$'`, plus `exl3_moe_kernel_test
   --rate-generalization`, plus the qwen27b EXL3 kernel test (shared dense
   GEMV; find the target in `tests/models/qwen*/CMakeLists.txt`). Known
   pre-existing: gdn_ops "fused stable GDN differs from the separate
   convolution" (run the rest non-fatally as B4 did). Fix real failures.
2. Bench: `qwen38_flash_next_dense_gemv_bench` (old vs new table).
3. Quality: B5 is designed exact. Check decode hashes first; collect quality
   logits on C only if prefill could change (J1 touches the shared dense GEMV
   used in prefill at M≤8? check) — if logits sha equals `17dc8f9f…` stop.
4. Perf rotation B C B C r2: TG n128 d0/d32768 AR+MTP, PP 2048/8192 off+MTP.
   Then TG-only rotation C, C−J1, C−J2b, C−J3 (r2, two rounds). Report per
   item delta (C vs variant) and hashes/MTP counts for every binary
   (expected equal to B4: `6d067a1254ff` d0, `d856f78e53b9` d32768).
5. Decode profile AR d0 on C (and on any variant whose delta is surprising):
   dispatches, busy, idle, per-kernel GB/s for dense GEMV instantiations and
   routed gate/up/down, using `build/b1/gpu-b4/analyze.py`.
6. Restart the service and confirm a short chat answers.

Do not edit docs/, do not revert on ai-tools/port-u4f. Report data and a
keep/revert recommendation per item (keep if exact and its isolation shows a
gain on TG with no row worse beyond noise). Write `RESULT.json`.
