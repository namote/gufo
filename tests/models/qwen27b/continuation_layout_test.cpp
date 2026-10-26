#include "src/models/qwen/continuation_layout.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace qwen = gufo::models::qwen;
namespace hip = gufo::hip;
namespace cache = gufo::cache;
namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
template<class F>
void Reject(F&& fn) {
  bool rejected = false;
  try {
    fn();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "invalid geometry/frontier was accepted");
}
void Check(bool fp16, bool bf16) {
  gufo::core::ModelConfig c;
  c.num_layers = 64;
  c.hidden_size = 5120;
  c.num_attention_heads = 24;
  c.intermediate_size = 17408;
  c.context_length = 262144;
  c.ssm_time_step_rank = 48;
  c.ssm_inner_size = 6144;
  hip::QwenExecutionPolicy policy;
  policy.kv_cache_storage =
      fp16 ? hip::QwenKvCacheStorage::kFp16 : hip::QwenKvCacheStorage::kFp32;
  policy.recurrent_state_storage = bf16 ? hip::QwenRecurrentStateStorage::kBf16
                                        : hip::QwenRecurrentStateStorage::kFp32;
  qwen::ContinuationLayout layout(c, policy, 100000);
  Require(layout.Components().size() == 131, "incomplete state inventory");
  // 48 independent recurrent layers: 10240 * 4 FP32 convolution elements,
  // plus 48 heads * 128 * 128 state elements. Logits and host metadata are
  // fixed.
  const std::size_t fixed =
      48 * (163840 + 786432 * (bf16 ? 2 : 4)) + 2 * 993280 + 64 + 96;
  Require(layout.PrivateBytes() == fixed, "incorrect private state claim");
  Require(layout.BytesPerToken() == (fp16 ? 65536 : 131072),
          "incorrect KV stride");
  for (auto position : {0U, 32768U, 100000U}) {
    const auto frontiers = layout.Positions(position);
    for (std::size_t i = 0; i < frontiers.size(); ++i) {
      const auto& part = layout.Components()[i];
      Require(part.descriptor.id.value == i + 1, "unstable component IDs");
      Require(frontiers[i].valid_rows == position,
              "component boundary differs");
      if (part.part == qwen::ContinuationPart::kKey ||
          part.part == qwen::ContinuationPart::kValue) {
        Require(
            part.descriptor.layout_version == (fp16 ? 1U : 2U) &&
                part.descriptor.layout_version ==
                    qwen::ContinuationKvLayoutVersion(policy.kv_cache_storage),
            "KV layouts alias");
        Require(part.descriptor.rows_per_chunk == 256,
                "unbounded chunk geometry");
      }
    }
  }
  Reject([&] { (void)layout.Positions(100001); });
  Reject([&] { qwen::ContinuationLayout bad(c, policy, 0); });
  c.full_attention_interval = 0;
  Reject([&] { qwen::ContinuationLayout bad(c, policy, 100000); });
}
// KVarN: one record row component per attention layer, the FP16 sink rows
// (one position of every layer per row) and the private tail state with the
// FP16 tail ring, layout version 5 for k4v2 and 6 for k4v4; FP16's
// components keep their versions and bytes.
void CheckKvarn() {
  gufo::core::ModelConfig c;
  c.num_layers = 64;
  c.hidden_size = 5120;
  c.num_attention_heads = 24;
  c.intermediate_size = 17408;
  c.context_length = 262144;
  c.ssm_time_step_rank = 48;
  c.ssm_inner_size = 6144;
  for (const auto [storage, row, version] :
       {std::tuple{hip::QwenKvCacheStorage::kKvarnK4V2, 840U, 5U},
        std::tuple{hip::QwenKvCacheStorage::kKvarnK4V4, 1096U, 6U}}) {
    hip::QwenExecutionPolicy policy;
    policy.kv_cache_storage = storage;
    qwen::ContinuationLayout layout(c, policy, 100000);
    Require(layout.Components().size() == 117, "KVarN state inventory");
    std::size_t records = 0, rows = 0;
    for (const auto& part : layout.Components()) {
      Require(part.part != qwen::ContinuationPart::kKey &&
                  part.part != qwen::ContinuationPart::kValue,
              "KVarN keeps FP16 K/V rows");
      if (part.part == qwen::ContinuationPart::kKvarnRecords) {
        ++records;
        Require(part.descriptor.row_bytes == row &&
                    part.descriptor.rows_per_chunk == 256 &&
                    part.descriptor.layout_version == version &&
                    part.descriptor.kind == cache::ComponentKind::kAppendRows,
                "KVarN record rows");
      }
      if (part.part == qwen::ContinuationPart::kKvarnSink) {
        ++rows;
        // K and V of 16 layers x 4 heads x 256 channels.
        Require(part.descriptor.row_bytes == 65536 &&
                    part.descriptor.rows_per_chunk == 128 &&
                    part.descriptor.layout_version == version &&
                    part.descriptor.kind == cache::ComponentKind::kAppendRows,
                "KVarN FP16 sink rows");
      }
      if (part.part == qwen::ContinuationPart::kKvarnTail) {
        ++rows;
        // Header, then 256 ring rows of K and V of 16 layers.
        Require(part.descriptor.state_bytes == 64 + 256 * 65536 &&
                    part.descriptor.layout_version == version &&
                    part.descriptor.kind == cache::ComponentKind::kPrivateState,
                "KVarN FP16 tail state");
      }
    }
    Require(records == 16 && rows == 2, "KVarN components per layer");
    Require(layout.BytesPerToken() == 16 * row + 65536,
            "KVarN stride per token");
    // The tail ring without the sink: 16 MiB instead of 24.
    const std::size_t fixed =
        48 * (163840 + 786432 * 4) + 2 * 993280 + 64 + 96 + 64 + 256 * 65536;
    Require(layout.PrivateBytes() == fixed, "KVarN private state claim");
    Require(qwen::ContinuationKvLayoutVersion(storage) == version,
            "KVarN layout version");
  }
}
// Which rows a checkpoint holds and how a restore reads the tail back.
void CheckKvarnRows() {
  using qwen::KvarnCheckpointRows;
  struct Case {
    cache::Rows position, flushed;
    KvarnCheckpointRows rows;
  };
  for (const auto& [position, flushed, rows] : {
           Case{0, 128, {0, 0, 0}},
           Case{70, 128, {70, 70, 0}},
           Case{128, 128, {128, 128, 0}},
           Case{200, 128, {128, 128, 72}},
           Case{646, 640, {640, 128, 6}},
           Case{530, 512, {512, 128, 18}},
           // A group ended at the last row; the next pass quantizes it.
           Case{640, 512, {512, 128, 128}},
           // A rejected draft: the tail is flushed past the position.
           Case{638, 640, {512, 128, 126}},
       }) {
    Require(qwen::KvarnRowsAt(position, flushed) == rows, "KVarN rows");
    const auto tail = qwen::KvarnRestoredTail(position, rows);
    Require(tail && tail->written_end == position &&
                tail->flushed_end == (position < 128 ? 128 : rows.records) &&
                qwen::KvarnRowsAt(position, tail->flushed_end) == rows,
            "KVarN restored tail");
  }
  // Rows of no checkpoint at the position are refused.
  for (const auto& [position, rows] :
       {std::pair<cache::Rows, KvarnCheckpointRows>{70, {64, 70, 0}},
        {70, {70, 70, 1}},
        {646, {512, 128, 6}},
        {646, {640, 127, 6}},
        {646, {646, 128, 0}},
        {700, {384, 128, 316}},
        {200, {72, 128, 128}}})
    Require(!qwen::KvarnRestoredTail(position, rows), "KVarN rows refused");
}
}  // namespace
int main() {
  try {
    for (bool fp16 : {false, true})
      for (bool bf16 : {false, true})
        Check(fp16, bf16);
    CheckKvarn();
    CheckKvarnRows();
    std::cout << "Qwen continuation geometry: PASS\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
