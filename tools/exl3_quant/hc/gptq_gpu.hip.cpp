// Offline FP64 sequential GPTQ feedback. No production inference changes.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace {
__global__ void Feedback(const double* weights, const double* upper,
                         const double* scales, std::int8_t* codes, int* failed,
                         unsigned rows, unsigned cols) {
  const unsigned row = blockIdx.x, lane = threadIdx.x;
  if (row >= rows)
    return;
  double value[40];
  for (unsigned t = 0; t < 40; ++t) {
    unsigned j = lane + t * 256;
    value[t] = j < cols ? weights[std::size_t{row} * cols + j] : 0.;
  }
  __shared__ double error;
  for (unsigned col = 0; col < cols; ++col) {
    if (lane == col % 256) {
      const double d = scales[std::size_t{row} * (cols / 32) + col / 32];
      const double current = value[col / 256];
      if (!isfinite(current))
        atomicExch(failed, 1);
      double code = 0.;
      if (d != 0.) {
        const double ratio = current / d;
        if (!isfinite(ratio))
          atomicExch(failed, 1);
        const double a = fabs(ratio), lo = floor(a);
        code = copysign(lo + (a - lo >= .5 ? 1. : 0.), ratio);
        code = fmax(-127., fmin(127., code));
      }
      codes[std::size_t{row} * cols + col] = static_cast<std::int8_t>(code);
      error = (current - code * d) / upper[std::size_t{col} * cols + col];
      if (!isfinite(error))
        atomicExch(failed, 1);
    }
    __syncthreads();
    for (unsigned t = 0; t < 40; ++t) {
      const unsigned j = lane + t * 256;
      if (j >= col && j < cols)
        value[t] -= error * upper[std::size_t{col} * cols + j];
    }
    __syncthreads();
  }
  for (unsigned t = 0; t < 40; ++t)
    if (lane + t * 256 < cols && !isfinite(value[t]))
      atomicExch(failed, 1);
}
}  // namespace
extern "C" int gufo_hc_feedback(const double* weights, const double* upper,
                                const double* scales, std::int8_t* codes,
                                int* failed, unsigned rows, unsigned cols,
                                hipStream_t stream) {
  if (!weights || !upper || !scales || !codes || !failed || !rows || !cols ||
      cols % 32 || cols > 10240)
    return static_cast<int>(hipErrorInvalidValue);
  hipLaunchKernelGGL(Feedback, dim3(rows), dim3(256), 0, stream, weights, upper,
                     scales, codes, failed, rows, cols);
  return static_cast<int>(hipGetLastError());
}
