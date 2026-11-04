// Qwen3.8 27B EXL3 prefill projection time at 1-64 rows: the X2b trellis
// GEMV (LaunchExl3FastProjection, 8-row passes) against the X3/X3b trellis
// WMMA (LaunchExl3RotateRows + LaunchExl3PrefillGemm), on the actual weights
// of one linear-attention and one full-attention layer of a native pack. It
// decides kExl3WmmaMinRows (src/models/qwen/hip/exl3_dispatch.hpp); prefill
// calls each projection separately (prefill_chunk.cpp gemm_weight), so the
// per-projection launches are what production runs. Built against a
// configured tree by tools/bench/exl3_wmma_threshold_bench.sh.
//
//   exl3_wmma_threshold_bench EXL3_DIR [reps]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "src/core/quant/exl3/checkpoint.hpp"
#include "src/core/quant/exl3/gemv.hpp"
#include "src/models/qwen/exl3_weights.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen/native_metadata.hpp"

namespace {

namespace qe = gufo::models::qwen_exl3;
using gufo::quant::exl3::DeviceLinear;

void Check(hipError_t e) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(e));
    std::exit(1);
  }
}

template<class T>
T* Upload(const std::vector<T>& v) {
  T* p = nullptr;
  Check(hipMalloc(&p, v.size() * sizeof(T)));
  Check(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  return p;
}

struct Projection {
  std::string name;
  DeviceLinear linear;
  unsigned count;  // per model forward (layers of this kind)
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s EXL3_DIR [reps]\n", argv[0]);
    return 2;
  }
  const int reps = argc > 2 ? std::atoi(argv[2]) : 50;
  std::string error;
  const auto metadata = gufo::models::qwen::OpenNativeMetadata(argv[1], &error);
  if (!metadata) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  const auto geometry = qe::GeometryFromGguf(*metadata, &error);
  const auto config = metadata->ExtractModelConfig(&error);
  const auto checkpoint = gufo::quant::exl3::Checkpoint::Open(argv[1]);
  const qe::Source source(checkpoint, *geometry);
  const unsigned layers = config->num_layers;
  const unsigned full = layers / config->full_attention_interval;
  const std::pair<const char*, unsigned> names[] = {
      {"blk.0.attn_qkv.weight", layers - full},
      {"blk.0.attn_gate.weight", layers - full},
      {"blk.0.ssm_out.weight", layers - full},
      {"blk.3.attn_q.weight", full},
      {"blk.3.attn_k.weight", full},
      {"blk.3.attn_v.weight", full},
      {"blk.3.attn_output.weight", full},
      {"blk.0.ffn_gate.weight", layers},
      {"blk.0.ffn_up.weight", layers},
      {"blk.0.ffn_down.weight", layers}};
  std::vector<Projection> projections;
  std::size_t max_k = 0, max_n = 0;
  for (const auto& [name, count] : names) {
    const auto host = source.PrepareLinear(name);
    DeviceLinear d = host.view;
    d.trellis = Upload(host.trellis);
    d.suh = Upload(host.suh);
    d.svh = Upload(host.svh);
    if (!gufo::hip::Exl3FastSupported(d) ||
        !gufo::hip::Exl3PrefillSupported(d)) {
      std::fprintf(stderr, "%s: unsupported by a path\n", name);
      return 1;
    }
    projections.push_back({name, d, count});
    max_k = std::max<std::size_t>(max_k, d.k);
    max_n = std::max<std::size_t>(max_n, d.n);
  }
  constexpr std::size_t kMaxRows = 64;
  std::vector<float> host_x(kMaxRows * max_k);
  std::mt19937 rng(1);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  for (auto& v : host_x)
    v = dist(rng);
  float* x = Upload(host_x);
  float* y = nullptr;
  Check(hipMalloc(&y, kMaxRows * max_n * sizeof(float)));
  std::uint16_t* slab = nullptr;
  Check(hipMalloc(&slab, 8 * max_k * sizeof(std::uint16_t)));
  std::uint32_t* counters = nullptr;
  const std::size_t counter_count = (max_n + 127) / 128;
  Check(hipMalloc(&counters, counter_count * sizeof(std::uint32_t)));
  Check(hipMemset(counters, 0, counter_count * sizeof(std::uint32_t)));
  std::uint16_t* prefill_input = nullptr;
  Check(hipMalloc(&prefill_input, kMaxRows * max_k * sizeof(std::uint16_t)));
  const gufo::hip::Exl3FastWorkspace workspace{
      slab, 8 * max_k, counters, counter_count, nullptr, 0};
  hipStream_t stream;
  Check(hipStreamCreate(&stream));
  hipEvent_t start, stop;
  Check(hipEventCreate(&start));
  Check(hipEventCreate(&stop));

  const auto time = [&](auto&& launch) {
    for (int i = 0; i < 5; ++i)
      launch();
    Check(hipEventRecord(start, stream));
    for (int i = 0; i < reps; ++i)
      launch();
    Check(hipEventRecord(stop, stream));
    Check(hipEventSynchronize(stop));
    float ms = 0;
    Check(hipEventElapsedTime(&ms, start, stop));
    return ms * 1000.0F / static_cast<float>(reps);  // us per launch set
  };

  std::printf("rows");
  for (const auto& p : projections)
    std::printf(",%s_gemv_us,%s_wmma_us", p.name.c_str(), p.name.c_str());
  std::printf(",model_gemv_ms,model_wmma_ms\n");
  for (const std::size_t rows :
       {1, 4, 8, 9, 12, 16, 17, 20, 24, 32, 40, 48, 56, 64}) {
    double model_gemv = 0, model_wmma = 0;
    std::printf("%zu", rows);
    for (const auto& p : projections) {
      const auto& l = p.linear;
      const float gemv = time([&] {
        gufo::hip::LaunchExl3FastProjection(l, x, l.k, rows, y, l.n, workspace,
                                            stream);
      });
      const float wmma = time([&] {
        gufo::hip::LaunchExl3RotateRows(x, l.k, rows, l.k, l.suh, prefill_input,
                                        stream);
        gufo::hip::LaunchExl3PrefillGemm(l, prefill_input, y, rows, false,
                                         stream);
      });
      std::printf(",%.2f,%.2f", gemv, wmma);
      model_gemv += gemv * p.count / 1000.0;
      model_wmma += wmma * p.count / 1000.0;
    }
    std::printf(",%.3f,%.3f\n", model_gemv, model_wmma);
  }
  return 0;
}
