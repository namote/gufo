// Isolated ctypes bridge to existing Gufo implementations. Not production code.
// GPU pointers are caller-owned, contiguous, on the current HIP device; calls
// enqueue on the supplied stream. Caller synchronizes before inspecting output.
// CPU entry points require HOST pointers. No allocations on the GPU occur here.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <exception>
#include <span>
#include <stdexcept>

#include "src/core/quant/exl3/gemv.hpp"
#include "src/models/qwen38_flash_next/exl3_weights.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace ex = gufo::quant::exl3;
namespace {
thread_local char error_text[1024] = {};
void Require(bool ok, const char* message) {
  if (!ok) throw std::invalid_argument(message);
}
template<class F> int Guard(F&& fn) noexcept {
  error_text[0] = 0;
  try { fn(); return 0; }
  catch (const std::exception& error) {
    std::snprintf(error_text, sizeof(error_text), "%s", error.what());
  } catch (...) {
    std::snprintf(error_text, sizeof(error_text), "unknown native exception");
  }
  return -1;
}
void CheckLaunch() {
  const auto result = hipPeekAtLastError();
  if (result != hipSuccess) throw std::runtime_error(hipGetErrorString(result));
}
ex::DeviceLinear Linear(const std::uint16_t* trellis,
                        const std::uint16_t* suh, const std::uint16_t* svh,
                        std::uint32_t k, std::uint32_t n, int bits, int codebook) {
  Require(trellis && suh && svh, "missing packed operand");
  Require(k && n && k % 128 == 0 && n % 128 == 0, "invalid packed geometry");
  Require(bits >= 1 && bits <= 8 && codebook >= 0 && codebook <= 2,
          "invalid packed format");
  ex::DeviceLinear linear{trellis, suh, svh, k, n, {bits, false},
                          static_cast<ex::Codebook>(codebook)};
  Require(ex::Supports(linear.codebook, linear.rate), "unsupported codebook/rate");
  return linear;
}
}  // namespace

extern "C" {
const char* mtpg_error() noexcept { return error_text; }
int mtpg_abi_version() noexcept { return 1; }

// gamma is already (original source gamma converted to F32) + 1.
int mtpg_norm(const float* x, const float* gamma, float* y,
              std::uint32_t rows, std::uint32_t dim, float eps, void* stream) {
  return Guard([&] {
    Require(x && gamma && y && rows && (dim == 2560 || dim == 10240) && eps > 0,
            "invalid full-row norm operands");
    q::rocm::RmsNormRows(x, gamma, y, rows, dim, 1, eps,
                         reinterpret_cast<hipStream_t>(stream));
    CheckLaunch();
  });
}

// Weight matrix is contiguous [N,K], unlike Python inner.weight [K,N].
// type: 0 = F32, 1 = F16, 30 = BF16 (Gufo WeightType identifiers).
int mtpg_raw(const void* weight, int type, const float* x, float* y,
             std::uint32_t rows, std::uint32_t k, std::uint32_t n, void* stream) {
  return Guard([&] {
    Require(weight && x && y && rows && k && n, "invalid raw operands");
    Require(type == 0 || type == 1 || type == 30, "unsupported raw dtype");
    q::rocm::SmallGemmStable(weight, static_cast<q::rocm::WeightType>(type),
                             x, y, rows, n, k,
                             reinterpret_cast<hipStream_t>(stream));
    CheckLaunch();
  });
}

// Output is row-major FP16 [rows,K], useful for a byte-exact stage comparison.
int mtpg_rotate(const float* x, const std::uint16_t* suh, std::uint16_t* out,
                std::uint32_t rows, std::uint32_t k, void* stream) {
  return Guard([&] {
    Require(x && suh && out && rows && k && k % 128 == 0,
            "invalid rotation operands");
    gufo::hip::LaunchExl3RotateRows(x, k, rows, k, suh, out,
                                   reinterpret_cast<hipStream_t>(stream));
    CheckLaunch();
  });
}

// Native DECODE path (not the FP32-rotation reference GPU path).
// Workspace caller allocates input uint16[64*K] and counters uint32[N/128].
// Counters must be zero before use; the native kernel returns them to zero.
// Only supported fast codebooks are accepted; no silent fallback.
int mtpg_packed(const std::uint16_t* trellis, const std::uint16_t* suh,
                const std::uint16_t* svh, const float* x, float* y,
                std::uint32_t rows, std::uint32_t k, std::uint32_t n,
                int bits, int codebook, std::uint16_t* workspace_input,
                std::uint64_t input_elements, std::uint32_t* counters,
                std::uint64_t counter_count, void* stream) {
  return Guard([&] {
    const auto linear = Linear(trellis, suh, svh, k, n, bits, codebook);
    Require(x && y && workspace_input && counters && rows &&
                rows <= gufo::hip::kExl3FastMaxRows,
            "invalid packed launch operands");
    const gufo::hip::Exl3FastWorkspace workspace{
        workspace_input, static_cast<std::size_t>(input_elements), counters,
        static_cast<std::size_t>(counter_count), nullptr, 0};
    const gufo::hip::Exl3FastTarget target{&linear, y, n};
    Require(gufo::hip::Exl3FastGroupSupported({&target, 1}, workspace),
            "native fast projection unsupported; no fallback");
    gufo::hip::LaunchExl3FastProjection(linear, x, k, rows, y, n, workspace,
                                       reinterpret_cast<hipStream_t>(stream));
    CheckLaunch();
  });
}

// HOST-only real codec oracle. out uint16[K,N] holds exact inner FP16 bits.
int mtpg_cpu_dequant(const std::uint16_t* trellis, std::uint32_t k,
                     std::uint32_t n, int bits, int codebook,
                     std::uint16_t* out) {
  return Guard([&] {
    Require(trellis && out && k && n && k % 128 == 0 && n % 128 == 0,
            "invalid host codec operands");
    Require(bits >= 1 && bits <= 8 && codebook >= 0 && codebook <= 2,
            "invalid host codec format");
    const auto cb = static_cast<ex::Codebook>(codebook);
    Require(ex::Supports(cb, {bits, false}), "unsupported host codebook/rate");
    ex::DequantInner(trellis, k, n, {bits, false}, cb, out);
  });
}

// HOST-only scalar oracle, including Gufo FP32 rotation then FP16 storage.
// Inner/output reductions use the independent CPU implementation's double
// arithmetic; compare a numerical tolerance, not GPU bit identity.
int mtpg_cpu_matvec(const std::uint16_t* trellis, const std::uint16_t* suh,
                    const std::uint16_t* svh, const float* x, float* y,
                    std::uint32_t rows, std::uint32_t k, std::uint32_t n,
                    int bits, int codebook) {
  return Guard([&] {
    Require(x && y && rows && rows <= 8, "bounded CPU oracle requires 1..8 rows");
    q::Exl3HostLinear host;
    host.view = Linear(trellis, suh, svh, k, n, bits, codebook);
    for (std::uint32_t row = 0; row < rows; ++row)
      q::Exl3MatVec(host, {x + std::size_t{row} * k, k},
                    {y + std::size_t{row} * n, n}, true);
  });
}
}  // extern "C"
