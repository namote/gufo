# Small operator checks are the default. Full-model qualification and
# diagnostic probes are built explicitly so routine iterations stay focused.
add_custom_target(qwen38_flash_next_tests)

# Model benchmarks live in tools/qwen38_flash_next/bench; their targets are
# declared here, beside the tests that share their libraries.
set(QFN_BENCH_DIR ${PROJECT_SOURCE_DIR}/tools/qwen38_flash_next/bench)

function(add_qwen38_flash_next_test name library labels)
  set(target qwen38_flash_next_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME qwen38_flash_next.${name} COMMAND ${target})
  set_tests_properties(qwen38_flash_next.${name} PROPERTIES
    LABELS "qwen38_flash_next;${labels}")
  add_dependencies(qwen38_flash_next_tests ${target})
endfunction()

add_qwen38_flash_next_test(exl3_reference gufo_qwen38_flash_next_reference "cpu;exl3")

add_qwen38_flash_next_test(exl3_loader gufo_qwen38_flash_next "cpu;exl3")

add_qwen38_flash_next_test(config gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(continuation_layout gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(ngram gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(native gufo_qwen38_flash_next "cpu;exl3")
add_qwen38_flash_next_test(mtp_sampling gufo_core "cpu")
add_qwen38_flash_next_test(rope_scaling gufo_qwen38_flash_next "cpu")
# The HF-oracle/off-bit-identity checks link the scalar model oracle, so
# (like ngram) this stays out of cmake/Checks.cmake's hosted check-pr set.
add_qwen38_flash_next_test(rope_scaling_oracle
  gufo_qwen38_flash_next_reference "cpu")
add_qwen38_flash_next_test(draft_vocab gufo_core "cpu")

add_executable(qwen38_flash_next_reference_probe EXCLUDE_FROM_ALL reference_probe.cpp)
target_link_libraries(qwen38_flash_next_reference_probe PRIVATE
  gufo_qwen38_flash_next_reference)
set_source_files_properties(reference_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  add_executable(qwen38_flash_next_rq_logits_probe EXCLUDE_FROM_ALL rq_logits_probe.cpp)
  target_link_libraries(qwen38_flash_next_rq_logits_probe PRIVATE
    gufo_qwen38_flash_next hip::host)

  # Decode HC projection timing (tools/qwen38_flash_next/bench), on the production kernels.
  add_executable(qwen38_flash_next_hc_decode_bench EXCLUDE_FROM_ALL
    ${QFN_BENCH_DIR}/hc_decode_bench.hip)
  set_source_files_properties(
    ${QFN_BENCH_DIR}/hc_decode_bench.hip
    PROPERTIES LANGUAGE HIP)
  target_link_libraries(qwen38_flash_next_hc_decode_bench PRIVATE
    gufo_qwen38_flash_next)

  # U4-5 screen: upstream #894's one-launch decode HC mixer (a copy, not
  # production) against the production decode mixer sequence.
  add_executable(qwen38_flash_next_hc_mix_fused_bench EXCLUDE_FROM_ALL
    ${QFN_BENCH_DIR}/experiments/hc_mix_fused_bench.hip)
  set_source_files_properties(
    ${QFN_BENCH_DIR}/experiments/hc_mix_fused_bench.hip
    PROPERTIES LANGUAGE HIP
    COMPILE_OPTIONS "-O3;-ffast-math;-fno-finite-math-only")
  target_link_libraries(qwen38_flash_next_hc_mix_fused_bench PRIVATE
    gufo_qwen38_flash_next)

  # Prefill MoE glue, plain-store kernel copies vs production (tools/qwen38_flash_next/bench).
  # The copies compile with the production library's HIP flags.
  add_executable(qwen38_flash_next_moe_glue_bench EXCLUDE_FROM_ALL
    ${QFN_BENCH_DIR}/moe_glue_bench.hip)
  set_source_files_properties(
    ${QFN_BENCH_DIR}/moe_glue_bench.hip
    PROPERTIES LANGUAGE HIP
    COMPILE_OPTIONS "-O3;-ffast-math;-fno-finite-math-only")
  target_link_libraries(qwen38_flash_next_moe_glue_bench PRIVATE
    gufo_qwen38_flash_next)

  # Prefill HC combine + W8A8 HC down against the frozen plain-store combine
  # (tools/qwen38_flash_next/bench), compiled with the production library's HIP flags.
  add_executable(qwen38_flash_next_hc_prefill_bench EXCLUDE_FROM_ALL
    ${QFN_BENCH_DIR}/hc_prefill_bench.hip)
  set_source_files_properties(
    ${QFN_BENCH_DIR}/hc_prefill_bench.hip
    PROPERTIES LANGUAGE HIP
    COMPILE_OPTIONS "-O3;-ffast-math;-fno-finite-math-only")
  target_link_libraries(qwen38_flash_next_hc_prefill_bench PRIVATE
    gufo_qwen38_flash_next)

  # EXL3 prefill GEMM ceiling screen (tools/qwen38_flash_next/bench, plan/exl3/archive/B7-CEILING-BRIEF.md):
  # production launchers against no-decode/WMMA-only/decode-only/warp-
  # specialized variants. No extra options: the kernel copies keep the
  # production prefill_fp16.hip flags, so their outputs compare bit for bit.
  add_executable(qwen38_flash_next_exl3_wmma_ceiling_bench EXCLUDE_FROM_ALL
    ${QFN_BENCH_DIR}/exl3_wmma_ceiling_bench.hip)
  set_source_files_properties(
    ${QFN_BENCH_DIR}/exl3_wmma_ceiling_bench.hip
    PROPERTIES LANGUAGE HIP)
  target_link_libraries(qwen38_flash_next_exl3_wmma_ceiling_bench PRIVATE
    gufo_qwen38_flash_next)

  # One binary, cases by stage: decode (exl3_ops_test.cpp, main), prefill and
  # MTP, sharing exl3_ops_fixture.hpp.
  add_qwen38_flash_next_test(exl3_ops gufo_qwen38_flash_next_reference "hip;gpu;gfx1151;exl3")
  target_sources(qwen38_flash_next_exl3_ops_test PRIVATE
    exl3_prefill_ops.cpp exl3_mtp_ops.cpp exl3_wide_ops.cpp)
  set_source_files_properties(exl3_ops_test.cpp exl3_prefill_ops.cpp
    exl3_mtp_ops.cpp exl3_wide_ops.cpp PROPERTIES LANGUAGE HIP)
  add_qwen38_flash_next_test(exl3_moe_kernel gufo_qwen38_flash_next "hip;gpu;gfx1151;exl3")
  set_source_files_properties(exl3_moe_kernel_test.cpp PROPERTIES LANGUAGE HIP)
  # Match production device arithmetic while keeping CPU oracles strict.
  set_source_files_properties(exl3_moe_kernel_test.cpp PROPERTIES
    COMPILE_OPTIONS "-O3;-Xarch_device;-ffast-math;-fno-finite-math-only")
  add_test(NAME qwen38_flash_next.exl3_shared_invariance
    COMMAND qwen38_flash_next_exl3_ops_test --shared-invariance-only)
  set_tests_properties(qwen38_flash_next.exl3_shared_invariance PROPERTIES
    LABELS "qwen38_flash_next;hip;gpu;gfx1151;exl3")
  add_qwen38_flash_next_test(exl3_invariance gufo_qwen38_flash_next "hip;gpu;gfx1151;exl3")
  set_source_files_properties(exl3_invariance_test.cpp PROPERTIES LANGUAGE HIP)
  foreach(name IN ITEMS mtp_ops blaslt_ops hc_mix_ops gdn_ops
      attention_ops attention_rows moe_epilogue_ops moe_ids_ops projection_ops routed_wmma_ops
      select_ops verification_sampling kvarn_ops)
    add_qwen38_flash_next_test(${name} gufo_qwen38_flash_next "hip;gpu;gfx1151")
  endforeach()
  add_test(NAME qwen38_flash_next.mtp_kv
    COMMAND qwen38_flash_next_attention_ops_test --mtp-kv)
  add_test(NAME qwen38_flash_next.mtp_kv_group
    COMMAND qwen38_flash_next_exl3_ops_test --mtp-kv-group)
  add_test(NAME qwen38_flash_next.mtp_kv_executor
    COMMAND qwen38_flash_next_exl3_ops_test --mtp-kv-executor)
  add_test(NAME qwen38_flash_next.exl3_wide_verification
    COMMAND qwen38_flash_next_exl3_ops_test --wide-verification)
  add_test(NAME qwen38_flash_next.exl3_producer_rotation
    COMMAND qwen38_flash_next_exl3_ops_test --producer-rotation)
  add_test(NAME qwen38_flash_next.exl3_decode_side_branches
    COMMAND qwen38_flash_next_exl3_ops_test --decode-side-branches)
  add_test(NAME qwen38_flash_next.exl3_prefill_executor
    COMMAND qwen38_flash_next_exl3_ops_test --prefill-experts-only)
  add_test(NAME qwen38_flash_next.exl3_dense_prefill
    COMMAND qwen38_flash_next_exl3_ops_test --dense-prefill-only)
  add_test(NAME qwen38_flash_next.exl3_plain_prefill
    COMMAND qwen38_flash_next_exl3_ops_test --plain-prefill-only)
  add_test(NAME qwen38_flash_next.exl3_gdn_prefill
    COMMAND qwen38_flash_next_exl3_ops_test --gdn-prefill-only)
  add_test(NAME qwen38_flash_next.exl3_prefill_overlap
    COMMAND qwen38_flash_next_exl3_ops_test --prefill-overlap-only)
  add_test(NAME qwen38_flash_next.exl3_prefill_scratch
    COMMAND qwen38_flash_next_exl3_ops_test --prefill-scratch-only)
  set_tests_properties(qwen38_flash_next.exl3_prefill_scratch PROPERTIES
    LABELS "qwen38_flash_next;cpu;exl3")
  add_test(NAME qwen38_flash_next.hc_down_f16
    COMMAND qwen38_flash_next_projection_ops_test --hc-down-f16)
  # Fused versus separate stable GDN route: see RunStableCase in gdn_ops_test.cpp.
  add_test(NAME qwen38_flash_next.gdn_stable_route_equality
    COMMAND qwen38_flash_next_gdn_ops_test --stable-route-equality)
  set_tests_properties(qwen38_flash_next.mtp_kv qwen38_flash_next.mtp_kv_group
    qwen38_flash_next.mtp_kv_executor qwen38_flash_next.exl3_wide_verification
    qwen38_flash_next.hc_down_f16
    qwen38_flash_next.gdn_stable_route_equality
    qwen38_flash_next.exl3_producer_rotation
    qwen38_flash_next.exl3_decode_side_branches
    qwen38_flash_next.exl3_prefill_executor
    qwen38_flash_next.exl3_dense_prefill
    qwen38_flash_next.exl3_plain_prefill
    qwen38_flash_next.exl3_gdn_prefill
    qwen38_flash_next.exl3_prefill_overlap
    PROPERTIES LABELS "qwen38_flash_next;hip;gpu;gfx1151;exl3")
  # The rope phase and RMS checks compare against the scalar CPU oracle.
  target_link_libraries(qwen38_flash_next_attention_ops_test PRIVATE
    gufo_qwen38_flash_next_reference)

  add_executable(qwen38_flash_next_session_test EXCLUDE_FROM_ALL session_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_disk_store.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/component_text_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_model_runner.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_generation_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/inference_backend.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/persistence_identity.cpp)
  target_link_libraries(qwen38_flash_next_session_test PRIVATE gufo_logging gufo_model_artifact
    gufo_kv_cache_type gufo_qwen38_flash_next gufo_deepseek_v4_flash gufo_cache_hip hip::host)
  add_executable(qwen38_flash_next_snapshot_test EXCLUDE_FROM_ALL snapshot_test.cpp
    device_headroom_fault.cpp)
  target_link_libraries(qwen38_flash_next_snapshot_test PRIVATE
    gufo_qwen38_flash_next ${CMAKE_DL_LIBS})
  add_executable(qwen38_flash_next_continuation_adapter_test EXCLUDE_FROM_ALL
    continuation_adapter_test.cpp reset_submission_fault.cpp)
  target_link_libraries(qwen38_flash_next_continuation_adapter_test PRIVATE
    gufo_qwen38_flash_next ${CMAKE_DL_LIBS})
  add_executable(qwen38_flash_next_continuation_probe EXCLUDE_FROM_ALL
    continuation_probe.cpp)
  target_link_libraries(qwen38_flash_next_continuation_probe PRIVATE
    gufo_qwen38_flash_next)
  # KVarN K0 accuracy study (plan/exl3/archive/F1-DESIGN.md 3.2); needs the model.
  add_executable(qwen38_flash_next_kvarn_k0_probe EXCLUDE_FROM_ALL kvarn_k0_probe.cpp)
  target_link_libraries(qwen38_flash_next_kvarn_k0_probe PRIVATE
    gufo_qwen38_flash_next_reference)
  add_custom_target(qwen38_flash_next_model_tests
    DEPENDS qwen38_flash_next_session_test qwen38_flash_next_snapshot_test
      qwen38_flash_next_continuation_adapter_test)

  add_executable(qwen38_flash_next_dense_gemm_bench EXCLUDE_FROM_ALL
    ${QFN_BENCH_DIR}/dense_gemm_bench.cpp)
  target_link_libraries(qwen38_flash_next_dense_gemm_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_attn_bench EXCLUDE_FROM_ALL ${QFN_BENCH_DIR}/attn_bench.cpp)
  target_link_libraries(qwen38_flash_next_attn_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_decode_attn_bench EXCLUDE_FROM_ALL
    ${QFN_BENCH_DIR}/decode_attn_bench.cpp)
  target_link_libraries(qwen38_flash_next_decode_attn_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_overlap_bench EXCLUDE_FROM_ALL
    ${QFN_BENCH_DIR}/overlap_bench.cpp)
  target_link_libraries(qwen38_flash_next_overlap_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_f16_gemm_bench EXCLUDE_FROM_ALL
    ${QFN_BENCH_DIR}/f16_gemm_bench.cpp)
  target_link_libraries(qwen38_flash_next_f16_gemm_bench PRIVATE
    gufo_qwen38_flash_next hip::host roc::hipblaslt)

  add_executable(qwen38_flash_next_gdn_prefill_bench EXCLUDE_FROM_ALL
    ${QFN_BENCH_DIR}/gdn_prefill_bench.cpp)
  target_link_libraries(qwen38_flash_next_gdn_prefill_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_routed_gemm_bench EXCLUDE_FROM_ALL
    ${QFN_BENCH_DIR}/routed_gemm_bench.cpp)
  target_link_libraries(qwen38_flash_next_routed_gemm_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_gpu_probe EXCLUDE_FROM_ALL gpu_probe.cpp
    mtp_audit.cpp mtp_trace.cpp)
  target_link_libraries(qwen38_flash_next_gpu_probe PRIVATE
    gufo_qwen38_flash_next_reference)
endif()
