# Qwen3.8 Flash-Next tests

Targets are declared in [CMakeLists.txt](CMakeLists.txt). Small operator
checks build by default. Full-model tests, probes and benchmarks are
`EXCLUDE_FROM_ALL` and are built by name. Benchmark sources live in
[tools/qwen38_flash_next/bench/](../../../tools/qwen38_flash_next/README.md),
and their targets are declared here because they link the same libraries.

| Kind | CTest names / targets | Sources |
| --- | --- | --- |
| CPU, hosted in `check-pr` (`cmake/Checks.cmake`) | `qwen38_flash_next.{config, continuation_layout, mtp_sampling, rope_scaling}` | `<name>_test.cpp` |
| CPU, not hosted (`cpu` label) | `qwen38_flash_next.{ngram, native, rope_scaling_oracle, draft_vocab, exl3_reference, exl3_loader, exl3_prefill_scratch}` | `<name>_test.cpp`. `exl3_prefill_scratch` is a mode of `exl3_ops_test` (built only with HIP) |
| GPU operator (`hip;gpu;gfx1151`) | `qwen38_flash_next.{mtp_ops, blaslt_ops, hc_mix_ops, gdn_ops, attention_ops, attention_rows, moe_epilogue_ops, moe_ids_ops, projection_ops, routed_wmma_ops, select_ops, verification_sampling, kvarn_ops}` | `<name>_test.cpp` |
| GPU EXL3 | `qwen38_flash_next.{exl3_ops, exl3_moe_kernel, exl3_invariance}` and the `exl3_ops`/`attention_ops` modes registered after them (`exl3_shared_invariance`, `mtp_kv*`, `exl3_producer_rotation`, `exl3_decode_side_branches`, `exl3_*_prefill`, `exl3_prefill_overlap`, `hc_down_f16`, `gdn_stable_route_equality`) | `exl3_ops_test.cpp` (decode cases and main) with `exl3_prefill_ops.cpp`, `exl3_mtp_ops.cpp` and `exl3_ops_fixture.hpp`, `exl3_moe_kernel_test.cpp`, `exl3_invariance_test.cpp`; frozen kernel copies in `exl3_routed_reference.hpp` and `hc_combine_reference.hpp` |
| Full model (needs a model; target `qwen38_flash_next_model_tests`) | `qwen38_flash_next_session_test`, `_snapshot_test`, `_continuation_adapter_test` | `session_test.cpp`, `snapshot_test.cpp` (+ `device_headroom_fault.cpp`), `continuation_adapter_test.cpp` (+ `reset_submission_fault.cpp`) |
| Probes and studies (manual) | `qwen38_flash_next_{reference_probe, gpu_probe, rq_logits, continuation_probe, kvarn_k0}` | `reference_probe.cpp`, `gpu_probe.cpp` (+ `mtp_audit.cpp`, `mtp_trace.cpp`), `fn_rq_logits.cpp`, `continuation_probe.cpp`, `kvarn_k0.cpp` |
| Disk oracle (manual, Python) | `python3 -I tests/models/qwen38_flash_next/rq_quality_test.py` | Header checks, and an optional small HF numerical replay that needs torch and transformers; not registered in CTest |

`rope_scaling_golden.hpp` holds the HF YaRN golden values the rope tests share,
and `mtp_replay.hpp` the MTP replay helpers of the session and adapter tests. The F32 reference fixtures are in
`tests/fixtures/qwen38_flash_next/f32_reference/`; their writer and scorer is
`tools/qwen38_flash_next/f32_reference.py`.
