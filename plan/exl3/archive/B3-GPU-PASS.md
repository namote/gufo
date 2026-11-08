# B3 combined GPU pass (2026-11-07)

Same procedure as `plan/exl3/archive/B2-GPU-PASS.md` (which reuses the B1 procedure),
with these B3 changes (they override). Results go to `build/b1/gpu-b3/`.

- Commits under test: `addf9dd3..HEAD` (`plan/exl3/archive/B3-BRIEF.md`, G0–G7 plus
  fixes, including `6b09c93c` which changes single-row Q8_0 GEMV rounding).
  Baseline = B2 candidate binary already saved: `build/b1/b2-cand-gufo`
  (source `9599b4c6`, same toolchain). Do not rebuild it.
- Unit tests (CTest names): `qwen38_flash_next.(hc_mix_ops|exl3_moe_kernel|
  attention_ops|gdn_ops|exl3_ops|exl3_invariance|exl3_producer_rotation|
  mtp_kv|mtp_kv_group|mtp_kv_executor|kvarn_ops|projection_ops|
  moe_epilogue_ops|routed_wmma_ops|snapshot_copy|exl3_prefill_executor|
  exl3_gdn_prefill|exl3_prefill_overlap)` plus `gpu_queues_test`. Check names
  in `tests/models/qwen38_flash_next/CMakeLists.txt`. Known pre-existing
  failure: gdn_ops "fused stable GDN differs from the separate convolution".
  The new hc_mix_ops "split" check (row 0 identical across widths) must pass.
- G0: build `tools/bench/gfx1151_launch_gap.hip` with `tools/bench/build.sh`
  (inside the GPU container / nix env; link failed in the CPU container on an
  impure-path check, fix the invocation not the tool if needed). Run it twice:
  `HIP_FORCE_DEV_KERNARG=0` and `=1`. Keep the tables (µs/dispatch stream vs
  graph, fork, kernarg size; µs/barrier at 1/2/4 blocks per CU).
- Bit expectations: G2 (RoPE explicit FMA) and `6b09c93c` change decode bits,
  so token hashes vs B2 may differ. Candidate AR vs MTP greedy outputs must
  still agree with each other; report MTP draft/accept/cycle counts.
- Quality: collect once on the 32,760 positions; compare against B2 candidate
  quality (`build/b1/gpu-b2/` quality dir if present, else B1
  `build/b1/gpu/quality/`) and the saved c03 reference. Same thresholds. If
  worse, isolate by reverting `6b09c93c` and G2 (`906242a3`, test fix part of
  `9983db83` not needed) individually.
- Perf: B C B C r2 as before (PP 2048/8192 off+MTP; TG n128 d0/d32768 AR+MTP).
  Isolation variants, TG rows only, interleaved with candidate:
  (a) G6: candidate with `HIP_FORCE_DEV_KERNARG=0` in the environment;
  (b) G5: candidate with `8c0d17f7` and `13e379cf` reverted (build in a
  temporary worktree with its own obj copy, remove after). Report each delta.
  Recommend keep/revert for G6 and G5 from these numbers.
- Decode profile required: candidate AR TG d0 with `tools/prof/prof.py`;
  report dispatches/token (expected ≈1000, B2 was 1442), kernel-active %,
  idle ms/token, top kernel categories (ms/token), and routed GEMV gate/up and
  down ms/token with GB/s (B2: 4.64 ms ≈170 GB/s, 2.63 ms ≈150 GB/s).
- Restart the service at the end (mandatory even on failure).

Write `build/b1/gpu-b3/RESULT.json`. Do not edit `docs/` — the parent records.
