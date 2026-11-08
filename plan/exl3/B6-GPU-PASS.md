# B6 combined GPU pass (2026-11-08)

Same procedure as `plan/exl3/B5-GPU-PASS.md` (→ B4 → B3 → B2 → B1), with
these B6 changes (they override). Results in `build/b1/gpu-b6/`. Stop the
service only with user approval (granted for this window); restart it at the
end (mandatory) with its current config (runtime-b5-20261108); deploy nothing.

Trees (build all before any GPU timing; main checkout only; variants via
temporary local branches + separate obj copies; delete branches afterwards;
leave the checkout on `ai-tools/port-u4f`):
- **B** baseline = deployed B5 binary `build/b1/b5-noj1-gufo` (sha 31cee0b6).
- **C** = HEAD (P1 `32b3f95e`, P3 `127599a7`, P5 `54c51533`, tests
  `74575503`/`a61b20f0`, P7 `035e675f` with default `kShared`).
- **C−P1, C−P3, C−P5**: C with that one commit's src reverted.
- **P7 variants** (C with `src/models/qwen38_flash_next/kernels/rocm/prefill_side_stream.hpp`
  constants edited): `kLowPriority`; `kCuMask` with `kPrefillSideWgps` 2, 4, 6.

Steps:
1. GPU unit tests on C: `ctest --preset gpu-full -R
   '^qwen38_flash_next\.(attention_ops|exl3_moe_kernel|hc_mix_ops|kvarn_ops|
   exl3_ops|exl3_invariance|exl3_prefill_executor|exl3_gdn_prefill|
   exl3_prefill_overlap|prefill_side_stream|projection_ops|gdn_ops)$'`
   (known pre-existing gdn_ops failure). Benches: `qwen38_flash_next_moe_glue_bench`,
   `qwen38_flash_next_attn_bench 4096 5 0 4096 8192`,
   `qwen38_flash_next_hc_prefill_bench` (must print equal/PASS; keep µs).
   Also run `exl3_prefill_overlap` on each P7 variant binary/test build.
2. Exactness: prefill logits sha on C must equal `17dc8f9f…` (collect quality
   logits once; stop if equal). Decode hashes must equal B5 (`6d067a1254ff`,
   `d856f78e53b9`).
3. Perf: PP 2048/8192 off (r2) rotated across B, C, C−P1, C−P3, C−P5 and the
   four P7 variants, two rounds; also PP with MTP and TG AR/MTP d0 for B vs C
   (and vs the best P7 variant) to confirm no decode regression.
4. Profile PP8192 for C and the best P7 variant: routed gate/up/down TF,
   shared-MLP side-stream span, overlap, idle.
5. Restart the service and confirm a short chat answers.

Report per-item deltas and a keep/revert recommendation; recommend the P7
default value (keep `kShared` unless a variant wins PP without hurting TG).
Write `RESULT.json`. Do not edit docs/; do not change ai-tools/port-u4f.
