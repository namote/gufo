#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXL3_MOE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXL3_MOE_HPP_

#include <hip/hip_runtime.h>

#include "src/core/quant/exl3/reference.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

struct Exl3RouteTile {
  std::uint32_t expert, count;
  std::uint32_t slots[8];
};

struct Exl3Scratch {
  quant::exl3::ReferenceScratch projection;
  float* slice{nullptr};
  std::size_t slice_elements{0};
  float* gather{nullptr};
  float* gate{nullptr};
  float* up{nullptr};
  float* out{nullptr};
  std::uint16_t* routed_input{nullptr};
  Exl3RouteTile* tiles{nullptr};
  std::uint32_t* tile_count{nullptr};
  std::uint32_t* head_segments{nullptr};
  float* head_values{nullptr};
};

/// Compressed projections, or the pack's plain operands with fixed FP32 math.
void Exl3Projection(const DeviceTensor& weight, const float* x, float* out,
                    std::uint32_t rows, const Exl3Scratch& scratch,
                    hipStream_t stream, bool fast = true);
void Exl3Gather(const float* x, const std::int32_t* rows, float* out,
                std::uint32_t count, std::uint32_t width, hipStream_t stream);
void Exl3Scatter(const float* x, const std::int32_t* slots, float* out,
                 std::uint32_t count, std::uint32_t width, hipStream_t stream);
/// Optional device destinations for operator qualification. Copies occur
/// between launches; kernels and their arithmetic do not depend on tracing.
struct Exl3RoutedTrace {
  std::uint16_t* gate_up_input{nullptr};  ///< [2][rows * used][hidden]
  float* gate_inner{nullptr};             ///< [rows * used][expert_ff]
  float* up_inner{nullptr};               ///< [rows * used][expert_ff]
  float* activation{nullptr};             ///< [rows * used][expert_ff]
  std::uint16_t* down_input{nullptr};     ///< [rows * used][expert_ff]
  float* down_inner{nullptr};             ///< [rows * used][hidden]
  float* down_packed_inner{nullptr};      ///< Same input, original packed dot.
};

/// Shared expert's decode/verify contract, common to serial and cohort calls.
/// Returns false without launches when the fused fast MLP cannot apply.
bool Exl3SharedMlp(const DeviceLayer& layer, const float* x, float* gate,
                   float* up, float* out, std::uint32_t rows,
                   const Exl3Scratch& scratch, hipStream_t stream);

/// Groups actual assignments by expert into 1..8-row tiles on device. Every
/// slot keeps its own rotation and output; the caller performs ordered MoE sum.
void Exl3RoutedMlp(const DeviceLayer& layer, const float* x,
                   const std::int32_t* ids, float* gate, float* up, float* out,
                   std::uint32_t rows, std::uint32_t used,
                   const Exl3Scratch& scratch, hipStream_t stream,
                   const Exl3RoutedTrace* trace = nullptr);
/// Prefix plus unique complete 128-column segments for the dynamic IDs.
/// Original trellis N stride is retained, and only final logits are gathered.
void Exl3DraftHead(const DeviceTensor& head, const float* x, float* logits,
                   std::uint32_t prefix, const std::int32_t* ids,
                   std::uint32_t extras, const Exl3Scratch& scratch,
                   hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm
#endif
