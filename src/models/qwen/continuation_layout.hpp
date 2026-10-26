#ifndef GUFO_MODELS_QWEN_CONTINUATION_LAYOUT_HPP_
#define GUFO_MODELS_QWEN_CONTINUATION_LAYOUT_HPP_

#include <span>
#include <vector>

#include "src/cache/types.hpp"
#include "src/core/model_config.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/kvarn_checkpoint.hpp"

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
  // rows); the FP16 sink rows [0, min(position, 128)), one position of
  // every layer (K then V) per row; and, as private state, the tail state
  // and every layer's FP16 tail ring. Tail rows change in place when a group
  // is quantized, so they cannot be append rows (cache histories only grow);
  // keeping only [base, position) needs private state of varying size.
  kKvarnRecords,
  kKvarnSink,
  kKvarnTail
};
struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t layer{};
};

inline constexpr std::size_t kContinuationMetadataBytes = 64;
inline constexpr std::size_t kContinuationFrontierMetadataBytes = 96;
inline constexpr std::size_t kContinuationPendingTokens = 9;
// Leads the kKvarnTail state: version, flushed_end, written_end.
inline constexpr std::size_t kContinuationKvarnHeaderBytes = 64;
// Layout version of the attention KV row components: FP16 K and V rows 1,
// FP32 2, KVarN 5 (k4v2) and 6 (k4v4). KVarN 3 and 4 kept the sink in the
// private FP16 rows too; their checkpoints are refused.
[[nodiscard]] constexpr std::uint32_t ContinuationKvLayoutVersion(
    hip::QwenKvCacheStorage storage) noexcept {
  switch (storage) {
    case hip::QwenKvCacheStorage::kFp16:
      return 1;
    case hip::QwenKvCacheStorage::kFp32:
      return 2;
    case hip::QwenKvCacheStorage::kKvarnK4V2:
      return 5;
    case hip::QwenKvCacheStorage::kKvarnK4V4:
      return 6;
  }
  return 0;
}

using hip::KvarnCheckpointRows;
using hip::KvarnRestoredTail;
using hip::KvarnRowsAt;
using hip::KvarnTailBase;

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
