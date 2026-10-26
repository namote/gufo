#ifndef GUFO_MODELS_QWEN_CONTINUATION_LAYOUT_HPP_
#define GUFO_MODELS_QWEN_CONTINUATION_LAYOUT_HPP_

#include <span>
#include <vector>

#include "src/cache/types.hpp"
#include "src/core/model_config.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"

namespace gufo::models::qwen {

enum class ContinuationPart {
  kMetadata,
  kLogits,
  kFrontier,
  kKey,
  kValue,
  kConvolution,
  kRecurrent,
  kDraftMetadata,
  kDraftKey,
  kDraftValue,
  kDraftFeatures,
  // KVarN: one attention layer's quantized group records as token rows (a
  // pool block's record bytes, KvarnRecord(bits).bytes * kv heads per 128
  // rows), and every layer's FP16 sink and tail rows with the tail state.
  kKvarnRecords,
  kKvarnRows
};
struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t layer{};
};

inline constexpr std::size_t kContinuationMetadataBytes = 64;
inline constexpr std::size_t kContinuationFrontierMetadataBytes = 96;
inline constexpr std::size_t kContinuationPendingTokens = 9;
// Leads the kKvarnRows state: version, flushed_end, written_end.
inline constexpr std::size_t kContinuationKvarnHeaderBytes = 64;
// Layout version of the attention KV row components: FP16 K and V rows 1,
// FP32 2, KVarN group records 3 (k4v2) and 4 (k4v4).
[[nodiscard]] constexpr std::uint32_t ContinuationKvLayoutVersion(
    hip::QwenKvCacheStorage storage) noexcept {
  switch (storage) {
    case hip::QwenKvCacheStorage::kFp16:
      return 1;
    case hip::QwenKvCacheStorage::kFp32:
      return 2;
    case hip::QwenKvCacheStorage::kKvarnK4V2:
      return 3;
    case hip::QwenKvCacheStorage::kKvarnK4V4:
      return 4;
  }
  return 0;
}

struct ContinuationBackingClass {
  std::size_t block_bytes{}, blocks{}, metadata_bytes{};
};

// KV rows have a canonical token/head/channel order. FP32's native head-major
// planes are gathered without conversion; their layout version differs from
// FP16. The recurrent storage policy also has its own layout version.
class ContinuationLayout {
public:
  ContinuationLayout(const core::ModelConfig&, hip::QwenExecutionPolicy,
                     std::uint32_t context);
  [[nodiscard]] std::span<const ContinuationComponent> Components() const {
    return components_;
  }
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      cache::Rows) const;
  [[nodiscard]] std::size_t PrivateBytes() const noexcept;
  [[nodiscard]] std::size_t BytesPerToken() const noexcept;
  [[nodiscard]] std::vector<ContinuationBackingClass> BackingPlan(
      std::size_t ram_bytes) const;

private:
  std::uint32_t context_;
  std::vector<ContinuationComponent> components_;
};

}  // namespace gufo::models::qwen
#endif
