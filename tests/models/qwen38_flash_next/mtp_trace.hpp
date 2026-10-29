#ifndef GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_MTP_TRACE_HPP_
#define GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_MTP_TRACE_HPP_

#include <cstdint>
#include <string>

#include "src/models/qwen38_flash_next/mtp_policy.hpp"

/// gpu_probe --mtp-trace (mtp_trace.cpp).
struct MtpTraceOptions {
  std::string model_path;
  std::string mtp_path;
  std::string suite_path;
  std::string out_path;
  std::uint32_t context = 49152;
  std::uint32_t max_tokens = 256;
  std::uint32_t depth = gufo::models::qwen38_flash_next::kMaxMtpDraftTokens;
};

int RunMtpTrace(const MtpTraceOptions& options);

#endif  // GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_MTP_TRACE_HPP_
