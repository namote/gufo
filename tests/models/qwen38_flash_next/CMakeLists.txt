add_test(NAME qwen38_flash_next.exl3_metadata
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/exl3_metadata_test.py)
set_tests_properties(qwen38_flash_next.exl3_metadata PROPERTIES
  LABELS "cpu;qwen38_flash_next;exl3")

# Small operator checks are the default. Full-model qualification and
# diagnostic probes are built explicitly so routine iterations stay focused.
add_custom_target(qwen38_flash_next_tests)

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
add_qwen38_flash_next_test(exl3_ple_digest gufo_qwen38_flash_next "cpu;exl3")

add_qwen38_flash_next_test(config gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(continuation_layout gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(ngram gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(mtp_sampling gufo_core "cpu")
add_qwen38_flash_next_test(rope_scaling gufo_qwen38_flash_next "cpu")
# The HF-oracle/off-bit-identity checks link the scalar model oracle, so
# (like ngram) this stays out of cmake/Checks.cmake's hosted check-pr set.
add_qwen38_flash_next_test(rope_scaling_oracle
  gufo_qwen38_flash_next_reference "cpu")
add_qwen38_flash_next_test(draft_vocab gufo_core "cpu")
add_qwen38_flash_next_test(prompt_lookup gufo_core "cpu")

add_executable(qwen38_flash_next_reference_probe EXCLUDE_FROM_ALL reference_probe.cpp)
target_link_libraries(qwen38_flash_next_reference_probe PRIVATE
  gufo_qwen38_flash_next_reference)
set_source_files_properties(reference_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  add_executable(qwen38_flash_next_rq_logits EXCLUDE_FROM_ALL fn_rq_logits.cpp)
  target_link_libraries(qwen38_flash_next_rq_logits PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_qwen38_flash_next_test(exl3_ops gufo_qwen38_flash_next_reference "hip;gpu;gfx1151;exl3")
  set_source_files_properties(exl3_ops_test.cpp PROPERTIES LANGUAGE HIP)
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
      attention_ops moe_epilogue_ops moe_ids_ops projection_ops routed_wmma_ops
      select_ops snapshot_copy verification_sampling kvarn_ops)
    add_qwen38_flash_next_test(${name} gufo_qwen38_flash_next "hip;gpu;gfx1151")
  endforeach()
  # The rope phase and RMS checks compare against the scalar CPU oracle.
  target_link_libraries(qwen38_flash_next_attention_ops_test PRIVATE
    gufo_qwen38_flash_next_reference)

  add_executable(qwen38_flash_next_session_test EXCLUDE_FROM_ALL session_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_disk_store.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/component_text_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_model_runner.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_generation_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/inference_backend.cpp)
  target_link_libraries(qwen38_flash_next_session_test PRIVATE gufo_logging
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
  # KVarN K0 accuracy study (plan/exl3/F1-DESIGN.md 3.2); needs the model.
  add_executable(qwen38_flash_next_kvarn_k0 EXCLUDE_FROM_ALL kvarn_k0.cpp)
  target_link_libraries(qwen38_flash_next_kvarn_k0 PRIVATE
    gufo_qwen38_flash_next_reference)
  add_custom_target(qwen38_flash_next_model_tests
    DEPENDS qwen38_flash_next_session_test qwen38_flash_next_snapshot_test
      qwen38_flash_next_continuation_adapter_test)

  add_executable(qwen38_flash_next_dense_gemm_bench EXCLUDE_FROM_ALL
    dense_gemm_bench.cpp)
  target_link_libraries(qwen38_flash_next_dense_gemm_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_attn_bench EXCLUDE_FROM_ALL attn_bench.cpp)
  target_link_libraries(qwen38_flash_next_attn_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_decode_attn_bench EXCLUDE_FROM_ALL
    decode_attn_bench.cpp)
  target_link_libraries(qwen38_flash_next_decode_attn_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_routed_gemm_bench EXCLUDE_FROM_ALL
    routed_gemm_bench.cpp)
  target_link_libraries(qwen38_flash_next_routed_gemm_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_gpu_probe EXCLUDE_FROM_ALL gpu_probe.cpp
    mtp_audit.cpp mtp_trace.cpp)
  target_link_libraries(qwen38_flash_next_gpu_probe PRIVATE
    gufo_qwen38_flash_next_reference)
endif()
