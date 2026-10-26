// A Qwen27B continuation slot whose whole state can be captured to host,
// changed and restored (the K0 study's KV access): fake-quantized attention
// K/V rows go back in as FP16 rows. Shared by qwen27b_kvarn_k0 and the KVarN
// decode test.
#ifndef GUFO_TESTS_MODELS_QWEN27B_KVARN_SESSION_HPP_
#define GUFO_TESTS_MODELS_QWEN27B_KVARN_SESSION_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/hip/transfer_pool.hpp"
#include "src/models/qwen/continuation_adapter.hpp"
#include "tests/models/qwen27b/kvarn_fake_quant.hpp"

namespace gufo::testing::kvarn {

namespace session_detail {
inline void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}
}  // namespace session_detail

struct Guard final : gufo::cache::MutationGuard {
  void BeforeOverwrite(gufo::cache::ComponentId, gufo::cache::Rows,
                       gufo::cache::Rows) override {}
  void BeforeRelease(gufo::cache::ComponentId, gufo::cache::Rows,
                     gufo::cache::Rows) noexcept override {}
};

// The whole retained state of a slot: one host buffer per component.
struct State {
  std::vector<gufo::cache::ComponentPosition> positions;
  std::vector<std::vector<std::byte>> components;
};

class Session {
public:
  Session(std::shared_ptr<const gufo::hip::QwenGpuModel> model,
          std::uint32_t context,
          gufo::hip::QwenExecutionPolicy policy =
              gufo::hip::QwenExecutionPolicy::Production(),
          std::shared_ptr<gufo::hip::KvBlockPool> kv_pool = nullptr)
      : adapter_(std::move(model), context, {'k', 'v', 'a', 'r', 'n'}, policy,
                 nullptr, {}, nullptr, nullptr, std::move(kv_pool)),
        slot_(adapter_.CreateSlot(guard_)),
        streams_(1) {}

  gufo::hip::QwenGpuExecutor& Executor() {
    return adapter_.GetExecutor(*slot_);
  }

  // Attention K/V components in layer order, key then value.
  [[nodiscard]] bool IsRows(std::size_t index) const {
    return adapter_.Components()[index].row_bytes != 0;
  }

  State Capture() {
    State state{adapter_.Positions(*slot_), {}};
    const auto components = adapter_.Components();
    for (std::size_t i = 0; i < components.size(); ++i) {
      const auto& component = components[i];
      const auto rows = state.positions[i].valid_rows;
      auto& bytes = state.components.emplace_back(
          component.row_bytes ? rows * component.row_bytes
                              : component.state_bytes);
      auto stream = Stream();
      Wait(component.row_bytes
               ? adapter_.CopyRowsOut(*slot_, component.id, 0, rows, bytes,
                                      *stream)
               : adapter_.CapturePrivate(*slot_, component.id, bytes, *stream));
    }
    return state;
  }

  void Restore(const State& state) {
    adapter_.BeginRestore(*slot_, state.positions);
    const auto components = adapter_.Components();
    for (std::size_t i = 0; i < components.size(); ++i) {
      const auto& component = components[i];
      auto stream = Stream();
      Wait(component.row_bytes
               ? adapter_.CopyRowsIn(*slot_, component.id, 0,
                                     state.positions[i].valid_rows,
                                     state.components[i], *stream)
               : adapter_.LoadPrivate(*slot_, component.id, state.components[i],
                                      *stream));
    }
    session_detail::Require(adapter_.Validate(*slot_, state.positions),
                            "restore failed");
  }

private:
  std::unique_ptr<gufo::hip::TransferStream> Stream() {
    auto stream = streams_.TryAcquire();
    session_detail::Require(bool(stream), "transfer stream exhausted");
    return stream;
  }
  static void Wait(gufo::cache::Completion done) {
    session_detail::Require(
        done.Wait() == gufo::cache::TransferResult::kSucceeded,
        "transfer failed");
  }

  gufo::models::qwen::ContinuationAdapter adapter_;
  Guard guard_;
  std::unique_ptr<gufo::cache::Slot> slot_;
  gufo::hip::TransferPool streams_;
};

struct SessionError {
  Error key, value;
};

// Fake-quantizes groups [first_group, end_group) of every attention K/V
// component of `state`.
inline SessionError Quantize(Session& session, State& state,
                             const Config& config, std::uint32_t heads,
                             std::uint32_t dim, std::size_t first_group,
                             std::size_t end_group) {
  SessionError error;
  bool key = true;
  for (std::size_t i = 0; i < state.components.size(); ++i) {
    if (!session.IsRows(i))
      continue;
    auto& bytes = state.components[i];
    const std::span rows(reinterpret_cast<std::uint16_t*>(bytes.data()),
                         bytes.size() / sizeof(std::uint16_t));
    const auto e =
        FakeQuantize(rows, heads, dim, key, config, first_group, end_group);
    auto& total = key ? error.key : error.value;
    total.squared += e.squared;
    total.reference += e.reference;
    key = !key;
  }
  return error;
}

inline double Relative(const Error& error) {
  return error.reference > 0 ? std::sqrt(error.squared / error.reference) : 0;
}

}  // namespace gufo::testing::kvarn

#endif  // GUFO_TESTS_MODELS_QWEN27B_KVARN_SESSION_HPP_
