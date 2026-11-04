#ifndef GUFO_MODELS_QWEN_VISION_MEMORY_HPP_
#define GUFO_MODELS_QWEN_VISION_MEMORY_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace gufo::models::qwen::vision {

struct ImageMemory {
  std::size_t embedding_bytes;
  std::size_t scratch_bytes;
};

/// Allocation sizes in Encoder::Encode, excluding resident weights, CPU input
/// and implementation-owned BLAS workspace. Admission retains separate device
/// and host headroom for the runtime. RGB upload's pinned copy is included.
inline ImageMemory EstimateImageMemory(std::uint32_t width,
                                       std::uint32_t height,
                                       std::uint32_t output_width,
                                       bool native_exl3 = false) {
  const std::uint64_t pixels = std::uint64_t{width} * height;
  if (!width || !height || width % 32 || height % 32 || pixels > 16777216 ||
      output_width > 65536 || output_width == 0)
    throw std::invalid_argument("invalid vision memory geometry");
  const std::size_t count = pixels / 256;
  const std::size_t tile =
      std::min(count, count == 4096 ? std::size_t{64} : std::size_t{128});
  const std::size_t embedding = count / 4 * output_width * sizeof(float);
  const std::size_t scratch = pixels * 3 + count * 1152 * 4 + count * 4304 * 4 +
                              count * 4304 * 2 + 3 * count * 1152 * 2 +
                              count * 1152 * 2 + 16 * tile * count * (4 + 2) +
                              tile * 1152 * 4;
  // All image embeddings are charged separately, including this one. The
  // pinned upload exists before its output; conservative overlap is bounded.
  // Native projections retain packed resident weights. Non-WMMA EXL3 rates
  // reuse one decoded-inner slab per image; input/output rotation buffers are
  // also temporary. Charge the largest supported matrix, even for K6 packs
  // whose fast kernels do not need that slab.
  const std::size_t native_scratch =
      native_exl3 ? count * (4608 * 6 + 5120 * 4 + (4352 - 4304) * 6) +
                        std::size_t{4608} * std::max(4608U, output_width) * 6
                  : 0;
  return {embedding, scratch + pixels * 3 + native_scratch};
}

}  // namespace gufo::models::qwen::vision
#endif
