// Flash-Next EXL3 decode HC projections: the split sequence (Q8_0 HC
// down GEMV 320 x 10240, SiluScaleQ8_1, Q8_0 HC up GEMV 10240 x 320,
// HcMixEpilogue) against the fused down projection HcDownSiluQ8_1 (each
// block owns one 32-row Q8_1 group of lo and finishes it in-block) followed
// by the same up GEMV and epilogue. The production entry points are timed,
// not copies. Rows R = 1..8 (decode and MTP verify widths), weights rotated
// over enough copies to miss the GPU caches, each variant captured in one
// graph of kCalls calls like a decode step. The fused lo and lo Q8_1 bytes
// must equal the split sequence's for every R. Results:
// docs/GFX1151-KNOWHOW.md, "Flash-Next B4".
//
// The HC up + mix fusion is not a variant: one block would have to own a
// 256-column chunk (1024 gate rows, 32 per wave on 10 of 20 WGPs) to run
// the mix epilogue without a cross-block tail, and the design screen did not
// expect that to match the plain up GEMV, which is timed for scale.
//
// Build (CMake, links the production library):
//   cmake --build --preset gpu-test --target qwen38_flash_next_hc_decode_bench
// Run:
//   build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_hc_decode_bench
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace q = gufo::models::qwen38_flash_next::rocm;

namespace {

constexpr int kHidden = 2560, kStreams = 4, kHcDim = kHidden * kStreams;
constexpr int kRank = 320;
constexpr int kMaxRows = 8;
// 2 x 3.48 MB of Q8_0 weights per mixer; 24 copies (167 MB) defeat the
// 32 MB last-level cache, as consecutive decode mixers do.
constexpr int kCopies = 24;
constexpr int kCalls = 96;  // HC mixers per Flash-Next decode token
constexpr int kReps = 20;

void Ck(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

void Require(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "%s\n", what);
    std::exit(1);
  }
}

std::uint32_t Next(std::uint32_t* s) {
  *s ^= *s << 13;
  *s ^= *s >> 17;
  *s ^= *s << 5;
  return *s;
}

std::vector<std::uint8_t> Q8_0(std::size_t blocks, std::uint32_t seed) {
  std::vector<std::uint8_t> bytes(blocks * 34);
  for (std::size_t b = 0; b < blocks; ++b) {
    const __half d = __float2half(1.0e-3F * float(1 + Next(&seed) % 64));
    std::memcpy(bytes.data() + b * 34, &d, 2);
    for (int i = 0; i < 32; ++i)
      bytes[b * 34 + 2 + i] = static_cast<std::uint8_t>(Next(&seed));
  }
  return bytes;
}

// Mean microseconds per call of `body(copy)` over a captured graph of
// kCalls calls, best of kReps replays.
template<class F>
double TimeGraph(hipStream_t stream, F&& body) {
  hipGraph_t graph;
  hipGraphExec_t exec;
  Ck(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "capture");
  for (int c = 0; c < kCalls; ++c)
    body(c % kCopies, stream);
  Ck(hipStreamEndCapture(stream, &graph), "end capture");
  Ck(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
  hipEvent_t a, b;
  Ck(hipEventCreate(&a), "event");
  Ck(hipEventCreate(&b), "event");
  Ck(hipGraphLaunch(exec, stream), "warm");
  Ck(hipStreamSynchronize(stream), "warm sync");
  float best = 1e30F;
  for (int r = 0; r < kReps; ++r) {
    Ck(hipEventRecord(a, stream), "record");
    Ck(hipGraphLaunch(exec, stream), "launch");
    Ck(hipEventRecord(b, stream), "record");
    Ck(hipEventSynchronize(b), "sync");
    float ms = 0;
    Ck(hipEventElapsedTime(&ms, a, b), "elapsed");
    best = ms < best ? ms : best;
  }
  Ck(hipEventDestroy(a), "destroy");
  Ck(hipEventDestroy(b), "destroy");
  Ck(hipGraphExecDestroy(exec), "destroy");
  Ck(hipGraphDestroy(graph), "destroy");
  return 1e3 * best / kCalls;
}

}  // namespace

int main() {
  hipStream_t stream;
  Ck(hipStreamCreate(&stream), "stream");
  const std::size_t w_bytes = std::size_t{kRank} * kHcDim / 32 * 34;
  std::vector<void*> down(kCopies), up(kCopies);
  for (int c = 0; c < kCopies; ++c) {
    const auto d = Q8_0(std::size_t{kRank} * kHcDim / 32, 0x51EDU + c);
    const auto u = Q8_0(std::size_t{kHcDim} * kRank / 32, 0x7A11U + c);
    Ck(hipMalloc(&down[c], w_bytes), "malloc");
    Ck(hipMalloc(&up[c], w_bytes), "malloc");
    Ck(hipMemcpy(down[c], d.data(), w_bytes, hipMemcpyHostToDevice), "upload");
    Ck(hipMemcpy(up[c], u.data(), w_bytes, hipMemcpyHostToDevice), "upload");
  }
  std::vector<float> xn(std::size_t{kMaxRows} * kHcDim);
  std::uint32_t seed = 0xC0DEU;
  for (auto& v : xn)
    v = float(int(Next(&seed) % 2001) - 1000) * 5e-4F;
  float *d_xn, *lo, *lo_f, *gate, *mixed;
  void *xq, *lo_q8, *lo_q8_f, *mixed_q8;
  const std::size_t lo_bytes = qfn_mmq_q8_1_bytes(kMaxRows, kRank);
  const std::size_t lo_row = qfn_mmq_q8_1_bytes(1, kRank);
  const auto lo_cols = static_cast<std::uint32_t>(lo_row / 36 * 32);
  Ck(hipMalloc(&d_xn, xn.size() * 4), "malloc");
  Ck(hipMemcpy(d_xn, xn.data(), xn.size() * 4, hipMemcpyHostToDevice), "up");
  Ck(hipMalloc(&xq, qfn_mmq_q8_1_bytes(kMaxRows, kHcDim)), "malloc");
  Ck(hipMalloc(&lo, std::size_t{kMaxRows} * kRank * 4), "malloc");
  Ck(hipMalloc(&lo_f, std::size_t{kMaxRows} * kRank * 4), "malloc");
  Ck(hipMalloc(&lo_q8, lo_bytes), "malloc");
  Ck(hipMalloc(&lo_q8_f, lo_bytes), "malloc");
  Ck(hipMemset(lo_q8, 0, lo_bytes), "memset");
  Ck(hipMemset(lo_q8_f, 0, lo_bytes), "memset");
  Ck(hipMalloc(&gate, std::size_t{kMaxRows} * kHcDim * 4), "malloc");
  Ck(hipMalloc(&mixed, std::size_t{kMaxRows} * kHidden * 4), "malloc");
  Ck(hipMalloc(&mixed_q8, qfn_mmq_q8_1_bytes(kMaxRows, kHidden)), "malloc");
  const float scale = 1.0F / kStreams;
  std::printf(
      "R  down+silu split  fused down  (GB/s split/fused)  up GEMV  mix  "
      "chain split  chain fused  [us/call]\n");
  for (int rows = 1; rows <= kMaxRows; ++rows) {
    Require(qfn_mmq_quantize_q8_1(d_xn, xq, rows, kHcDim, stream) == 0,
            "quantize");
    const auto down_split = [&](int c, hipStream_t s) {
      Require(qfn_mmq_q8_0_dense_vec_preq(down[c], nullptr, xq, lo, kRank, rows,
                                          kHcDim, s) == 0 &&
                  q::SiluScaleQ8_1(lo, scale, rows, kRank, lo_cols, lo_q8, s),
              "split down");
    };
    const auto down_fused = [&](int c, hipStream_t s) {
      Require(q::HcDownSiluQ8_1(down[c], xq, lo_f, scale, rows, lo_cols,
                                lo_q8_f, s),
              "fused down");
    };
    const auto up_gemv = [&](int c, hipStream_t s) {
      Require(qfn_mmq_q8_0_dense_vec_preq(up[c], nullptr, lo_q8, gate, kHcDim,
                                          rows, kRank, s) == 0,
              "up GEMV");
    };
    const auto mix = [&](int, hipStream_t s) {
      q::HcMixEpilogue(d_xn, gate, nullptr, mixed, nullptr, rows, kHidden,
                       kStreams, s, mixed_q8, q::Exl3InputSlabs{});
    };
    // Equality first (one call of each, copy 0).
    down_split(0, stream);
    down_fused(0, stream);
    Ck(hipStreamSynchronize(stream), "sync");
    std::vector<float> a(std::size_t(rows) * kRank), b(a.size());
    std::vector<std::uint8_t> qa(rows * lo_row), qb(qa.size());
    Ck(hipMemcpy(a.data(), lo, a.size() * 4, hipMemcpyDeviceToHost), "down");
    Ck(hipMemcpy(b.data(), lo_f, b.size() * 4, hipMemcpyDeviceToHost), "down");
    Ck(hipMemcpy(qa.data(), lo_q8, qa.size(), hipMemcpyDeviceToHost), "down");
    Ck(hipMemcpy(qb.data(), lo_q8_f, qb.size(), hipMemcpyDeviceToHost), "down");
    Require(std::memcmp(a.data(), b.data(), a.size() * 4) == 0 && qa == qb,
            "fused HC down differs from the split sequence");
    const double t_split = TimeGraph(stream, down_split);
    const double t_fused = TimeGraph(stream, down_fused);
    const double t_up = TimeGraph(stream, up_gemv);
    const double t_mix = TimeGraph(stream, mix);
    const double t_chain_split = TimeGraph(stream, [&](int c, hipStream_t s) {
      down_split(c, s);
      up_gemv(c, s);
      mix(c, s);
    });
    const double t_chain_fused = TimeGraph(stream, [&](int c, hipStream_t s) {
      down_fused(c, s);
      // The up GEMV reads the fused launch's lo Q8_1 in production.
      Require(qfn_mmq_q8_0_dense_vec_preq(up[c], nullptr, lo_q8_f, gate, kHcDim,
                                          rows, kRank, s) == 0,
              "up GEMV");
      mix(c, s);
    });
    std::printf(
        "%d  %10.2f  %10.2f  (%5.0f/%5.0f)  %7.2f  %5.2f  %8.2f  "
        "%10.2f\n",
        rows, t_split, t_fused, w_bytes / t_split * 1e-3,
        w_bytes / t_fused * 1e-3, t_up, t_mix, t_chain_split, t_chain_fused);
  }
  std::printf("fused HC down: lo and lo Q8_1 byte-identical for R = 1..8\n");
  return 0;
}
