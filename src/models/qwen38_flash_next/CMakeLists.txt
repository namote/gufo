# Qwen3.8-Flash-Next (qwen4exp) model package. The scalar oracle is built
# only by explicit validation targets, not by production or routine checks.
add_library(gufo_qwen38_flash_next STATIC
  abliteration.cpp
  exl3_hc_q8.cpp
  exl3_ngram.cpp
  exl3_weights.cpp
  config.cpp
  continuation_layout.cpp
  ngram.cpp
  native_metadata.cpp
  rope_scaling.cpp
  weights.cpp)

target_include_directories(gufo_qwen38_flash_next PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen38_flash_next PUBLIC gufo_core Threads::Threads)

add_library(gufo_qwen38_flash_next_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_qwen38_flash_next_reference PUBLIC
  gufo_qwen38_flash_next)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  set(QFN_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
  set(QFN_RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/runtime")
  set(QFN_MMQ_DIR "${QFN_ROCM_DIR}/mmq")

  # HIP quantized matrix kernels adapted from llama.cpp, isolated in the
  # qfn_mmq namespace. The adapter retains its C++17 dialect.
  set(QFN_MMQ_SOURCES
    ${QFN_MMQ_DIR}/qfn_ggml_stubs.hip.cpp
    ${QFN_MMQ_DIR}/qfn_mmq.hip.cpp
    ${QFN_MMQ_DIR}/mmid.hip.cpp
    ${QFN_MMQ_DIR}/mmvq.hip.cpp
    ${QFN_MMQ_DIR}/moe_wave64.hip.cpp
    ${QFN_MMQ_DIR}/quantize.hip.cpp)

  add_library(gufo_qwen38_flash_next_mmq STATIC ${QFN_MMQ_SOURCES})
  set_source_files_properties(${QFN_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${QFN_MMQ_DIR}/moe_wave64.hip.cpp
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  set_target_properties(gufo_qwen38_flash_next_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_qwen38_flash_next_mmq PRIVATE
    GGML_HIP_NO_VMM ENGINE_ENABLE_HIP=1)
  target_compile_options(gufo_qwen38_flash_next_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen38_flash_next_mmq SYSTEM PUBLIC ${QFN_MMQ_DIR})
  target_link_libraries(gufo_qwen38_flash_next_mmq PUBLIC hip::host roc::hipblas)

  # Model kernels by stage, sharing the private kernels_internal.hpp;
  # kernels.hpp declares their launchers.
  set(QFN_KERNEL_SOURCES
    ${QFN_ROCM_DIR}/embed_norm.hip.cpp
    ${QFN_ROCM_DIR}/hc.hip.cpp
    ${QFN_ROCM_DIR}/activation.hip.cpp
    ${QFN_ROCM_DIR}/abliteration.hip.cpp
    ${QFN_ROCM_DIR}/small_gemm.hip.cpp
    ${QFN_ROCM_DIR}/w8a8.hip.cpp
    ${QFN_ROCM_DIR}/routed_f16.hip.cpp
    ${QFN_ROCM_DIR}/dense_gemm.hip.cpp
    ${QFN_ROCM_DIR}/recurrent.hip.cpp
    ${QFN_ROCM_DIR}/attention_prep.hip.cpp
    ${QFN_ROCM_DIR}/attention_select.hip.cpp
    ${QFN_ROCM_DIR}/attention.hip.cpp
    ${QFN_ROCM_DIR}/moe_glue.hip.cpp
    ${QFN_ROCM_DIR}/mtp.hip.cpp)
  # Host runtime: the device-resident model, the executor by stage and the
  # batched multi-session decode.
  set(QFN_RUNTIME_SOURCES
    ${QFN_RUNTIME_DIR}/blaslt.cpp
    ${QFN_RUNTIME_DIR}/device_model.cpp
    ${QFN_RUNTIME_DIR}/executor.cpp
    ${QFN_RUNTIME_DIR}/executor_snapshot.cpp
    ${QFN_RUNTIME_DIR}/executor_dense.cpp
    ${QFN_RUNTIME_DIR}/executor_hc.cpp
    ${QFN_RUNTIME_DIR}/executor_ple.cpp
    ${QFN_RUNTIME_DIR}/executor_gdn.cpp
    ${QFN_RUNTIME_DIR}/executor_attention.cpp
    ${QFN_RUNTIME_DIR}/executor_moe.cpp
    ${QFN_RUNTIME_DIR}/executor_forward.cpp
    ${QFN_RUNTIME_DIR}/executor_mtp.cpp
    ${QFN_RUNTIME_DIR}/executor_batch.cpp)
  set(QFN_HIP_SOURCES
    continuation_adapter.cpp
    ${QFN_ROCM_DIR}/exl3_moe.hip.cpp
    ${QFN_KERNEL_SOURCES}
    ${QFN_ROCM_DIR}/w8a8_wave64.hip.cpp
    ${QFN_RUNTIME_SOURCES}
    engine.cpp)
  target_sources(gufo_qwen38_flash_next PRIVATE ${QFN_HIP_SOURCES})
  set_source_files_properties(${QFN_KERNEL_SOURCES} ${QFN_ROCM_DIR}/exl3_moe.hip.cpp
    ${QFN_ROCM_DIR}/w8a8_wave64.hip.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${QFN_ROCM_DIR}/w8a8_wave64.hip.cpp
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  target_compile_definitions(gufo_qwen38_flash_next PRIVATE QFN_ROCM_BUILD=1)
  target_compile_options(gufo_qwen38_flash_next PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen38_flash_next SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
  target_link_libraries(gufo_qwen38_flash_next PUBLIC
    gufo_cache_hip
    gufo_qwen38_flash_next_mmq hip::host roc::hipblas roc::hipblaslt)

  # Tool builds without tests still get the continuation probe, which
  # tests/models/qwen38_flash_next declares when testing is on.
  if (GUFO_BUILD_TOOLS AND NOT BUILD_TESTING)
    add_executable(qwen38_flash_next_continuation_probe EXCLUDE_FROM_ALL
      ${PROJECT_SOURCE_DIR}/tests/models/qwen38_flash_next/continuation_probe.cpp)
    target_link_libraries(qwen38_flash_next_continuation_probe PRIVATE
      gufo_qwen38_flash_next)
  endif()
endif()
