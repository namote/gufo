// FP16 GEMM core comparison at the Flash-Next dense prefill shapes:
// our kernels (UnquantizedF16Gemm, PlainPrefillGemm on F16 weights) against
// every hipBLASLt solution the library reports for FP16 x FP16 -> FP32
// (F32 compute), row-major out[M][N] = x[M][K] * w[N][K]^T.
//
// For each shape it prints TFLOPS of our kernels, the hipBLASLt heuristic's
// first choice and the three fastest library solutions (name, workspace),
// then checks the fastest workspace-free solution for chunk invariance: the
// first 2048 rows of an M=4096 run must equal an M=2048 run, and rows shifted
// by one position (input offset by a row) must reproduce the same outputs.
//
//   qwen38_flash_next_f16_gemm_bench [max_solutions]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

void CheckHip(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void CheckBlas(hipblasStatus_t s, const char* what) {
  if (s != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(what) + " failed");
}

struct Shape {
  int n, k;  // out features, in features
};

struct Layouts {
  hipblasLtMatmulDesc_t op{};
  hipblasLtMatrixLayout_t w{}, x{}, y{};
  Layouts(int m_tokens, int n, int k) {
    const hipblasOperation_t t = HIPBLAS_OP_T, nn = HIPBLAS_OP_N;
    CheckBlas(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F),
              "desc");
    CheckBlas(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA,
                                              &t, sizeof(t)),
              "transa");
    CheckBlas(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB,
                                              &nn, sizeof(nn)),
              "transb");
    CheckBlas(hipblasLtMatrixLayoutCreate(&w, HIP_R_16F, k, n, k), "w");
    CheckBlas(hipblasLtMatrixLayoutCreate(&x, HIP_R_16F, k, m_tokens, k), "x");
    CheckBlas(hipblasLtMatrixLayoutCreate(&y, HIP_R_32F, n, m_tokens, n), "y");
  }
  ~Layouts() {
    (void)hipblasLtMatrixLayoutDestroy(w);
    (void)hipblasLtMatrixLayoutDestroy(x);
    (void)hipblasLtMatrixLayoutDestroy(y);
    (void)hipblasLtMatmulDescDestroy(op);
  }
};

template<typename F>
double TimeMs(F&& f, int reps) {
  hipEvent_t a, b;
  CheckHip(hipEventCreate(&a), "event");
  CheckHip(hipEventCreate(&b), "event");
  f();  // warm-up
  CheckHip(hipDeviceSynchronize(), "sync");
  std::vector<float> t;
  for (int r = 0; r < reps; ++r) {
    CheckHip(hipEventRecord(a, nullptr), "record");
    f();
    CheckHip(hipEventRecord(b, nullptr), "record");
    CheckHip(hipEventSynchronize(b), "sync");
    float ms = 0;
    CheckHip(hipEventElapsedTime(&ms, a, b), "elapsed");
    t.push_back(ms);
  }
  (void)hipEventDestroy(a);
  (void)hipEventDestroy(b);
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::size_t max_solutions = argc > 1 ? std::stoul(argv[1]) : 100000;
    const std::vector<Shape> shapes = {{10240, 2560}, {6144, 2560},
                                       {12288, 2560}, {2560, 6144},
                                       {640, 2560},   {2560, 640}};
    constexpr int kMaxM = 4096;
    constexpr std::size_t kWorkspace = 256ULL << 20;
    hipblasLtHandle_t handle;
    CheckBlas(hipblasLtCreate(&handle), "handle");
    std::vector<hipblasLtMatmulHeuristicResult_t> all;
    CheckBlas(hipblaslt_ext::getAllAlgos(
                  handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, HIPBLAS_OP_T,
                  HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F, HIP_R_32F, HIP_R_32F,
                  HIPBLAS_COMPUTE_32F, all),
              "getAllAlgos");
    std::cout << "hipBLASLt FP16 solutions: " << all.size() << '\n';
    void* workspace = nullptr;
    CheckHip(hipMalloc(&workspace, kWorkspace), "workspace");
    std::mt19937 rng(7);
    std::normal_distribution<float> normal;
    for (const auto& s : shapes) {
      std::vector<__half> hw(static_cast<std::size_t>(s.n) * s.k),
          hx(static_cast<std::size_t>(kMaxM + 1) * s.k);
      for (auto& v : hw)
        v = __float2half(normal(rng) * 0.02F);
      for (auto& v : hx)
        v = __float2half(normal(rng));
      __half *dw, *dx;
      float* dy;
      CheckHip(hipMalloc(&dw, hw.size() * 2), "w");
      CheckHip(hipMalloc(&dx, hx.size() * 2), "x");
      CheckHip(hipMalloc(&dy, std::size_t{kMaxM + 1} * s.n * 4), "y");
      CheckHip(hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice),
               "w up");
      CheckHip(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice),
               "x up");
      for (const int m : {2048, 4096}) {
        const double flop = 2.0 * m * s.n * s.k;
        const auto tf = [&](double ms) { return flop / (ms * 1e-3) / 1e12; };
        std::cout << "shape M=" << m << " K=" << s.k << " N=" << s.n << '\n';
        std::cout << std::fixed << std::setprecision(2);
        // Our kernels.
        bool ok = q::UnquantizedF16Gemm(dw, dx, dy, m, s.n, s.k, nullptr);
        if (ok) {
          const double ms = TimeMs(
              [&] {
                (void)q::UnquantizedF16Gemm(dw, dx, dy, m, s.n, s.k, nullptr);
              },
              7);
          std::cout << "  ours UnquantizedF16Gemm " << ms << " ms " << tf(ms)
                    << " TFLOPS\n";
        } else {
          std::cout << "  ours UnquantizedF16Gemm unsupported\n";
        }
        ok = q::PlainPrefillGemm(dw, q::WeightType::kF16, dx, dy, m, s.n, s.k,
                                 nullptr);
        if (ok) {
          const double ms = TimeMs(
              [&] {
                (void)q::PlainPrefillGemm(dw, q::WeightType::kF16, dx, dy, m,
                                          s.n, s.k, nullptr);
              },
              7);
          std::cout << "  ours PlainPrefillGemm(F16) " << ms << " ms " << tf(ms)
                    << " TFLOPS\n";
        } else {
          std::cout << "  ours PlainPrefillGemm(F16) unsupported\n";
        }
        // hipBLASLt.
        Layouts l(m, s.n, s.k);
        const float one = 1.0F, zero = 0.0F;
        {
          hipblasLtMatmulPreference_t pref;
          CheckBlas(hipblasLtMatmulPreferenceCreate(&pref), "pref");
          (void)hipblasLtMatmulPreferenceSetAttribute(
              pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspace,
              sizeof(kWorkspace));
          hipblasLtMatmulHeuristicResult_t h[1];
          int count = 0;
          if (hipblasLtMatmulAlgoGetHeuristic(handle, l.op, l.w, l.x, l.y, l.y,
                                              pref, 1, h, &count) ==
                  HIPBLAS_STATUS_SUCCESS &&
              count > 0) {
            const double ms = TimeMs(
                [&] {
                  CheckBlas(
                      hipblasLtMatmul(handle, l.op, &one, dw, l.w, dx, l.x,
                                      &zero, dy, l.y, dy, l.y, &h[0].algo,
                                      workspace, kWorkspace, nullptr),
                      "heuristic matmul");
                },
                7);
            std::cout << "  hipBLASLt heuristic " << ms << " ms " << tf(ms)
                      << " TFLOPS ws=" << h[0].workspaceSize << " "
                      << hipblaslt_ext::getKernelNameFromAlgo(handle, h[0].algo)
                      << '\n';
          }
          (void)hipblasLtMatmulPreferenceDestroy(pref);
        }
        struct Result {
          double ms;
          std::size_t ws;
          std::string name;
          int index;
        };
        std::vector<Result> results;
        std::size_t tried = 0;
        for (std::size_t a = 0; a < all.size() && tried < max_solutions; ++a) {
          std::size_t ws = 0;
          if (hipblaslt_ext::matmulIsAlgoSupported(
                  handle, l.op, &one, l.w, l.x, &zero, l.y, l.y, all[a].algo,
                  ws) != HIPBLAS_STATUS_SUCCESS ||
              ws > kWorkspace)
            continue;
          ++tried;
          const double ms = TimeMs(
              [&] {
                CheckBlas(hipblasLtMatmul(handle, l.op, &one, dw, l.w, dx, l.x,
                                          &zero, dy, l.y, dy, l.y, &all[a].algo,
                                          workspace, ws, nullptr),
                          "solution matmul");
              },
              3);
          results.push_back(
              {ms, ws,
               hipblaslt_ext::getKernelNameFromAlgo(handle, all[a].algo),
               static_cast<int>(a)});
        }
        std::sort(results.begin(), results.end(),
                  [](const Result& x, const Result& y) { return x.ms < y.ms; });
        std::cout << "  hipBLASLt tried " << tried << " solutions\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(3, results.size());
             ++i)
          std::cout << "  hipBLASLt #" << i + 1 << " " << results[i].ms
                    << " ms " << tf(results[i].ms)
                    << " TFLOPS ws=" << results[i].ws << " " << results[i].name
                    << '\n';
        // Chunk invariance of the fastest workspace-free solution (M=4096 run
        // only, against M=2048 and a one-row shift).
        if (m != 4096)
          continue;
        for (const auto& r : results) {
          if (r.ws != 0)
            continue;
          auto& algo = all[r.index].algo;
          const auto run = [&](int rows, const __half* x, float* y) {
            Layouts lr(rows, s.n, s.k);
            std::size_t ws = 0;
            if (hipblaslt_ext::matmulIsAlgoSupported(
                    handle, lr.op, &one, lr.w, lr.x, &zero, lr.y, lr.y, algo,
                    ws) != HIPBLAS_STATUS_SUCCESS ||
                ws != 0)
              return false;
            CheckBlas(
                hipblasLtMatmul(handle, lr.op, &one, dw, lr.w, x, lr.x, &zero,
                                y, lr.y, y, lr.y, &algo, nullptr, 0, nullptr),
                "matmul");
            CheckHip(hipDeviceSynchronize(), "sync");
            return true;
          };
          float* y2;
          CheckHip(hipMalloc(&y2, std::size_t{kMaxM + 1} * s.n * 4), "y2");
          std::vector<float> full(std::size_t{kMaxM} * s.n),
              half(std::size_t{2048} * s.n), shifted(std::size_t{2048} * s.n);
          bool supported = run(4096, dx, dy);
          if (supported) {
            CheckHip(hipMemcpy(full.data(), dy, full.size() * 4,
                               hipMemcpyDeviceToHost),
                     "dl");
          }
          supported = supported && run(2048, dx, y2);
          if (supported)
            CheckHip(hipMemcpy(half.data(), y2, half.size() * 4,
                               hipMemcpyDeviceToHost),
                     "dl");
          // Rows 1..2048 of the input as a 2048-row batch: row r of this run
          // is row r + 1 of the full run.
          supported = supported && run(2048, dx + s.k, y2);
          if (supported)
            CheckHip(hipMemcpy(shifted.data(), y2, shifted.size() * 4,
                               hipMemcpyDeviceToHost),
                     "dl");
          (void)hipFree(y2);
          if (!supported) {
            std::cout << "  invariance: fastest workspace-free solution "
                      << r.name << " does not support M=2048\n";
            break;
          }
          const bool same_prefix =
              std::memcmp(full.data(), half.data(), half.size() * 4) == 0;
          const bool same_shift = std::memcmp(full.data() + s.n, shifted.data(),
                                              shifted.size() * 4) == 0;
          std::cout << "  invariance of " << r.name << " (" << r.ms << " ms, "
                    << tf(r.ms) << " TFLOPS): M=2048 prefix "
                    << (same_prefix ? "equal" : "DIFFERS") << ", one-row shift "
                    << (same_shift ? "equal" : "DIFFERS") << '\n';
          break;
        }
      }
      (void)hipFree(dw);
      (void)hipFree(dx);
      (void)hipFree(dy);
    }
    (void)hipFree(workspace);
    (void)hipblasLtDestroy(handle);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
