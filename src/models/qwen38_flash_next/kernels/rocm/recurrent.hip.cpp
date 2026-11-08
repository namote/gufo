// Recurrent layers: Gated DeltaNet prep, recurrence, epilogue and
// rollback, the SSM four-tap convolution, the PLE gate/convolution/inject,
// and the rolling history and snapshot copies they share.
// HIP kernels follow the layouts and operator formulas in reference.cpp.

#include "src/models/qwen38_flash_next/kernels/rocm/kernels_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

__global__ void PleGateKernel(const float* key_n, const float* query_n,
                              const float* value, float* gated,
                              std::uint32_t hidden, std::uint32_t streams) {
  __shared__ float shared[32];
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t s = blockIdx.y;
  const std::size_t base = (static_cast<std::size_t>(t) * streams + s) * hidden;
  float dot = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    dot += key_n[base + i] * query_n[base + i];
  }
  dot = BlockSum(dot, shared);
  const float sc = dot * rsqrtf(static_cast<float>(hidden));
  const float mag = sqrtf(fmaxf(fabsf(sc), 1e-6f));
  const float sign = sc < 0.0f ? -1.0f : (sc > 0.0f ? 1.0f : 0.0f);
  const float gate = SigmoidF(sign * mag);
  const float* v = value + static_cast<std::size_t>(t) * hidden;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    gated[base + i] = v[i] * gate;
  }
}

__global__ void PleConvKernel(const float* in, const float* w,
                              const float* history, float* out,
                              std::uint32_t n_tokens, std::uint32_t channels,
                              std::uint32_t kernel, std::uint32_t dilation,
                              std::uint32_t hist) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<std::size_t>(n_tokens) * channels) {
    return;
  }
  const std::uint32_t t = idx / channels;
  const std::uint32_t c = idx % channels;
  float acc = 0.0f;
  for (std::uint32_t k = 0; k < kernel; ++k) {
    const std::int32_t src_t =
        static_cast<std::int32_t>(t) -
        static_cast<std::int32_t>((kernel - 1 - k) * dilation);
    const float v =
        src_t >= 0
            ? in[static_cast<std::size_t>(src_t) * channels + c]
            : history[static_cast<std::size_t>(hist + src_t) * channels + c];
    acc += w[static_cast<std::size_t>(c) * kernel + k] * v;
  }
  out[idx] = SiluF(acc);
}

/// New history row j is row (n_tokens + j) of [history ; in].
__global__ void HistoryShiftKernel(const float* in, std::uint32_t in_stride,
                                   const float* history, float* scratch,
                                   std::uint32_t n_tokens,
                                   std::uint32_t channels, std::uint32_t hist) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<std::size_t>(hist) * channels) {
    return;
  }
  const std::uint32_t j = idx / channels;
  const std::uint32_t c = idx % channels;
  const std::uint32_t src = n_tokens + j;
  scratch[idx] = src < hist
                     ? history[static_cast<std::size_t>(src) * channels + c]
                     : in[static_cast<std::size_t>(src - hist) * in_stride + c];
}

__global__ void PleInjectKernel(float* res, const float* gated,
                                const float* conv, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    res[i] += gated[i] + conv[i];
  }
}

/// L2-normalizes the convolved q and k of one (token, key head) into the
/// packed [t][kh][d] buffers the recurrence streams from, so the serial loop
/// carries no block-wide reductions. One wave per (token, head) row.
constexpr unsigned kGdnDim = 128;

/// GdnPrepKernel's normalization of one (token, key head) row for one lane
/// of a wave32: qv/kv hold elements r * 32 + lane on entry and the scaled
/// values on return. Every step is pinned to the sequence the original
/// kernel compiled to under fast-math (product, then FMA chain, butterfly
/// sum whose last step adds eps to the partner first, rsqrt, product).
/// Normalizing inside GdnKernel instead did not keep these bits: chunked
/// decode calls and prefill checkpoints then differed in the last ulp.
__device__ __forceinline__ void GdnPrepRow(float (&qv)[kGdnDim / 32],
                                           float (&kv)[kGdnDim / 32],
                                           float eps) {
  float qs = Materialize(qv[0] * qv[0]);
  float ks = Materialize(kv[0] * kv[0]);
#pragma unroll
  for (std::uint32_t r = 1; r < kGdnDim / 32; ++r) {
    qs = Materialize(__builtin_fmaf(qv[r], qv[r], qs));
    ks = Materialize(__builtin_fmaf(kv[r], kv[r], ks));
  }
#pragma unroll
  for (int offset = 16; offset > 1; offset >>= 1) {
    qs = Materialize(qs + XorLane(qs, offset));
    ks = Materialize(ks + XorLane(ks, offset));
  }
  qs = Materialize(Materialize(eps + XorLane(qs, 1)) + qs);
  ks = Materialize(Materialize(eps + XorLane(ks, 1)) + ks);
  qs = Materialize(rsqrtf(qs));
  ks = Materialize(rsqrtf(ks));
#pragma unroll
  for (std::uint32_t r = 0; r < kGdnDim / 32; ++r) {
    qv[r] = Materialize(qv[r] * qs);
    kv[r] = Materialize(kv[r] * ks);
  }
}

template<bool kBatch>
__global__ void GdnPrepKernel(const float* conv_out, float* qn, float* kn,
                              std::uint32_t n_rows, std::uint32_t k_heads,
                              std::uint32_t channels, float eps,
                              const GdnBatchItem* batch, std::uint32_t active) {
  if constexpr (kBatch) {
    if ((active & (1U << blockIdx.z)) == 0)
      return;
    const auto& item = batch[blockIdx.z];
    conv_out = item.conv_scratch;
    qn = item.qn;
    kn = item.kn;
    n_rows = item.n_tokens * k_heads;
  }
  constexpr std::uint32_t d = kGdnDim;
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t row =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= n_rows) {
    return;
  }
  const std::uint32_t t = row / k_heads;
  const std::uint32_t kh = row % k_heads;
  const float* conv = conv_out + static_cast<std::size_t>(t) * channels;
  const std::uint32_t per_lane = d / warpSize;
  float qv[d / 32];
  float kv[d / 32];
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    const std::uint32_t i = r * warpSize + lane;
    qv[r] = conv[kh * d + i];
    kv[r] = conv[k_heads * d + kh * d + i];
  }
  GdnPrepRow(qv, kv, eps);
  float* q_out = qn + static_cast<std::size_t>(row) * d;
  float* k_out = kn + static_cast<std::size_t>(row) * d;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    const std::uint32_t i = r * warpSize + lane;
    q_out[i] = qv[r];
    k_out[i] = kv[r];
  }
}

__global__ void GdnPrepKqKernel(const float* conv_out, float* scales,
                                std::uint32_t n_tokens, std::uint32_t k_heads,
                                std::uint32_t channels, float eps) {
  constexpr std::uint32_t d = kGdnDim;
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t kh = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (t >= n_tokens || kh >= k_heads) {
    return;
  }
  const float* row = conv_out + static_cast<std::size_t>(t) * channels;
  const auto* q = reinterpret_cast<const float4*>(row + kh * d);
  const auto* k = reinterpret_cast<const float4*>(row + (k_heads + kh) * d);
  const float4 q4 = q[lane];
  const float4 k4 = k[lane];
  float qs = q4.x * q4.x + q4.y * q4.y + q4.z * q4.z + q4.w * q4.w;
  float ks = k4.x * k4.x + k4.y * k4.y + k4.z * k4.z + k4.w * k4.w;
  float kq = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
  for (unsigned offset = 16; offset > 0; offset >>= 1) {
    qs += XorLane(qs, offset);
    ks += XorLane(ks, offset);
    kq += XorLane(kq, offset);
  }
  if (lane == 0) {
    float* dst = scales + (static_cast<std::size_t>(t) * k_heads + kh) * 3;
    dst[0] = rsqrtf(ks + eps);
    dst[1] = rsqrtf(qs + eps) * rsqrtf(static_cast<float>(d));
    dst[2] = kq;
  }
}

__global__ void GdnPrepAbKernel(const float* alpha_beta, const float* a,
                                const float* dt, float* ab, std::size_t count,
                                std::uint32_t v_heads) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }
  const std::uint32_t h = i % v_heads;
  const std::size_t t = i / v_heads;
  const float alpha = alpha_beta[t * 2 * v_heads + h];
  const float beta = alpha_beta[t * 2 * v_heads + v_heads + h];
  ab[i * 2] = __expf(a[h] * SoftplusF(alpha + dt[h]));
  ab[i * 2 + 1] = SigmoidF(beta);
}

template<int kMask>
__device__ __forceinline__ float GdnXorAddDpp(float x) {
  const int y = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x),
                                            0x160 | kMask, 0xF, 0xF, false);
  return x + __builtin_bit_cast(float, y);
}

__device__ __forceinline__ void GdnRowReduce(float& u, float& p) {
  u = GdnXorAddDpp<1>(u);
  p = GdnXorAddDpp<1>(p);
  u = GdnXorAddDpp<2>(u);
  p = GdnXorAddDpp<2>(p);
}

// Two-row LDS reuse adapted from mikealanni/gufo f450218743bc (MIT).
/// Qwen row-split recurrence specialized to the model's 128x128 state. Four
/// lanes own the 32-key slices of a row, eight row groups share a wave and
/// each lane carries kRows state rows. The q/k slices of a token are read from
/// LDS once per lane and applied to all of its rows: with one row per lane
/// the kernel re-read the same 16 float4 for every row and saturated LDS
/// bandwidth. Every row keeps its original per-lane summation order and
/// DPP reduction. A block of kWaves waves covers 8 * kRows * kWaves rows of one
/// head. The block stages four tokens at a time through LDS and keeps three
/// such windows of loads in flight, one window ahead of the one being
/// committed (the conv output is far larger than the caches).
constexpr int kGdnWindow = 4;
constexpr int kGdnWindowsAhead = 3;
constexpr int kGdnRows = 2;   // state rows per lane
constexpr int kGdnWaves = 4;  // waves per block
constexpr int kGdnRowSplitRowsPerBlock = 8 * kGdnRows * kGdnWaves;
constexpr int kGdnConvHistory = 3;  // four-tap convolution

/// SsmConv4Kernel<..., stable=true>'s arithmetic for one channel and token:
/// w0..w2 are tokens t-3..t-1 and v is token t.
__device__ __forceinline__ float GdnStableConv4(float4 taps, float w0, float w1,
                                                float w2, float v) {
  float acc = Materialize(Materialize(taps.x) * Materialize(w0));
  acc = Materialize(__builtin_fmaf(Materialize(taps.y), Materialize(w1), acc));
  acc = Materialize(__builtin_fmaf(Materialize(taps.z), Materialize(w2), acc));
  acc = Materialize(__builtin_fmaf(Materialize(taps.w), Materialize(v), acc));
  return SiluF(acc);
}

/// kConv: the block convolves its own q, k and v channels from the raw
/// projection (EXL3's stable four-tap convolution, history from conv_state)
/// and computes GdnPrepKqKernel's scales per token from the staged q and k,
/// so neither the convolved rows nor the scales round-trip through memory.
/// Every value keeps the separate kernels' arithmetic; conv_state is read
/// only (the caller shifts history after this kernel).
template<bool kConv>
__launch_bounds__(kGdnWaves * 32) __global__
    void GdnRowSplitKernel(const float* conv_out, const float* scales,
                           const float* ab, float* state, float* raw,
                           std::uint32_t n_tokens, std::uint32_t k_heads,
                           std::uint32_t v_heads, GdnCheckpoint checkpoint,
                           const float* qkv, std::uint32_t qkv_stride,
                           const float* conv_w, const float* conv_state,
                           float eps) {
  constexpr int kRows = kGdnRows;
  constexpr int kWaves = kGdnWaves;
  constexpr int d = kGdnDim;
  constexpr int kThreadsPerBlock = kWaves * 32;
  constexpr int kKeysPerLane = 32;
  constexpr int kVec = kKeysPerLane / 4;
  constexpr int kLanesPerRow = d / kKeysPerLane;
  constexpr int kRowsPerWave = (32 / kLanesPerRow) * kRows;
  constexpr int kRowsPerBlock = kRowsPerWave * kWaves;
  constexpr int kW = kGdnWindow;
  constexpr int kQkChunks = kW * (d / 4) * 2;  // q and k float4 per window
  constexpr int kQkPerThread = kQkChunks / kThreadsPerBlock;
  constexpr int kVChunks = kW * kRowsPerBlock;
  constexpr int kVPerThread = kVChunks / kThreadsPerBlock;
  static_assert(kQkChunks % kThreadsPerBlock == 0);
  static_assert(kVChunks % kThreadsPerBlock == 0);
  static_assert(kW * 5 <= kThreadsPerBlock, "scales and decays per window");
  // kConv: two q|k channels per thread, and one v channel for each thread
  // below kRowsPerBlock.
  static_assert(!kConv || 2 * d == 2 * kThreadsPerBlock);
  static_assert(!kConv || kRowsPerBlock <= kThreadsPerBlock);
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int segment = lane % kLanesPerRow;
  const int row_group = lane / kLanesPerRow;
  const int kh = h % k_heads;
  const int vec0 = segment * kVec;
  const int channels = 2 * k_heads * d + v_heads * d;

  // Row rr of this lane within the block.
  const auto row_local = [&](int rr) {
    return wave * kRowsPerWave + rr * (32 / kLanesPerRow) + row_group;
  };

  // Two window slots: q and k rows, the block's v rows, the three scales
  // and two decays per token.
  __shared__ float4 s_qk[2][2][kW][d / 4];
  __shared__ float s_v[2][kW][kRowsPerBlock];
  __shared__ float s_misc[2][kW][5];

  float4 s[kRows][kVec];
#pragma unroll
  for (int rr = 0; rr < kRows; ++rr) {
    const int row = blockIdx.x * kRowsPerBlock + row_local(rr);
    const auto* state4 = reinterpret_cast<const float4*>(
        state + (static_cast<std::size_t>(h) * d + row) * d);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      s[rr][i] = state4[vec0 + i];
    }
  }

  // Window loads: chunk c of the q/k plane is (q or k, token, float4 index),
  // laid out so a token's 128 floats are 32 consecutive chunks.
  // kConv sources: channels cq_ch and cq_ch + 1 of the key head's q (or k)
  // row, stored into half cq_half of LDS float4 cq_slot; v channel cv_ch.
  const int cq_which = tid >> 6;
  const int cq_vec = (tid >> 1) & 31;
  const int cq_half = tid & 1;
  const int cq_ch =
      ((cq_which == 0 ? kh : k_heads + kh) * d) + (cq_vec * 4) + (cq_half * 2);
  const int cq_slot = ((cq_vec % kVec) * kLanesPerRow) + (cq_vec / kVec);
  const bool has_v = tid < kRowsPerBlock;
  const int cv_ch = (2 * k_heads * d) + (h * d) + (blockIdx.x * kRowsPerBlock) +
                    (has_v ? tid : 0);
  float4 tap_q0{};
  float4 tap_q1{};
  float4 tap_v{};
  // hist_*[i] holds raw token t0 - 3 + i of the next window to convolve.
  float2 hist_q[kGdnConvHistory]{};
  float hist_v[kGdnConvHistory]{};
  if constexpr (kConv) {
    const auto* taps = reinterpret_cast<const float4*>(conv_w);
    tap_q0 = taps[cq_ch];
    tap_q1 = taps[cq_ch + 1];
    tap_v = taps[cv_ch];
#pragma unroll
    for (int i = 0; i < kGdnConvHistory; ++i) {
      hist_q[i] = *reinterpret_cast<const float2*>(
          conv_state + (static_cast<std::size_t>(i) * channels) + cq_ch);
      hist_v[i] = conv_state[(static_cast<std::size_t>(i) * channels) + cv_ch];
    }
  }
  const float* qk_src[kQkPerThread];
  int qk_tok[kQkPerThread];
  int qk_which[kQkPerThread];
  int qk_slot[kQkPerThread];
#pragma unroll
  for (int j = 0; j < kQkPerThread; ++j) {
    const int c = tid + j * kThreadsPerBlock;
    qk_which[j] = c >> 7;
    qk_tok[j] = (c >> 5) & 3;
    const int qk_vec = c & 31;
    qk_src[j] =
        conv_out + (qk_which[j] == 0 ? kh : k_heads + kh) * d + (qk_vec * 4);
    // A token's 32 float4 are stored [i][segment] so the four segments of
    // a row's lanes read adjacent 16-byte chunks.
    qk_slot[j] = ((qk_vec % kVec) * kLanesPerRow) + (qk_vec / kVec);
  }
  const float* v_src[kVPerThread];
  int v_tok[kVPerThread];
  int v_row[kVPerThread];
#pragma unroll
  for (int j = 0; j < kVPerThread; ++j) {
    const int e = tid + j * kThreadsPerBlock;
    v_tok[j] = e / kRowsPerBlock;
    v_row[j] = e % kRowsPerBlock;
    v_src[j] = conv_out + 2 * k_heads * d + h * d + blockIdx.x * kRowsPerBlock +
               v_row[j];
  }
  // kConv loads only the decay and beta (s_misc[..][3..4]); it computes the
  // scales per token from LDS.
  constexpr int kMiscLoads = kConv ? 2 : 5;
  const int m_tok = tid / kMiscLoads;
  const int m_idx = (tid % kMiscLoads) + (kConv ? 3 : 0);
  const float* m_src =
      m_idx < 3 ? scales + kh * 3 + m_idx : ab + h * 2 + (m_idx - 3);
  const std::size_t m_stride = m_idx < 3 ? k_heads * 3 : v_heads * 2;

  // Three windows of loads in flight, as named registers (a ring array
  // lands in scratch).
  // Two named q/k vectors: an array of them stayed in private memory
  // (112 B of scratch, reloaded every window).
  static_assert(kQkPerThread == 2, "named q/k window vectors");
  struct PlainWindow {
    float4 qk0;
    float4 qk1;
    float v[kVPerThread];
    float m;
  };
  // kConv: the raw projection values of the window's tokens.
  struct ConvWindow {
    float2 q[kW];
    float v[kW];
    float m;
  };
  using Window = std::conditional_t<kConv, ConvWindow, PlainWindow>;
  Window r0;
  Window r1;
  Window r2;
  const auto load_window = [&](int w, Window& r) {
    const int last = static_cast<int>(n_tokens) - 1;
    if constexpr (kConv) {
#pragma unroll
      for (int tl = 0; tl < kW; ++tl) {
        const auto t =
            static_cast<std::size_t>(min((w * kW) + tl, last)) * qkv_stride;
        r.q[tl] = *reinterpret_cast<const float2*>(qkv + t + cq_ch);
        if (has_v)
          r.v[tl] = qkv[t + cv_ch];
      }
      if (tid < kMiscLoads * kW) {
        const int t_m = min((w * kW) + m_tok, last);
        r.m = m_src[static_cast<std::size_t>(t_m) * m_stride];
      }
      return;
    } else {
      const auto qk_row = [&](int j) {
        const int t_qk = min((w * kW) + qk_tok[j], last);
        return *reinterpret_cast<const float4*>(
            qk_src[j] + static_cast<std::size_t>(t_qk) * channels);
      };
      r.qk0 = qk_row(0);
      r.qk1 = qk_row(1);
#pragma unroll
      for (int j = 0; j < kVPerThread; ++j) {
        const int t_v = min((w * kW) + v_tok[j], last);
        r.v[j] = v_src[j][static_cast<std::size_t>(t_v) * channels];
      }
      if (tid < 5 * kW) {
        const int t_m = min((w * kW) + m_tok, last);
        r.m = m_src[static_cast<std::size_t>(t_m) * m_stride];
      }
    }
  };
  // Windows commit in token order, so the history registers always hold
  // the three raw tokens before the window being convolved.
  const auto commit_window = [&](const Window& r, int lds) {
    if constexpr (kConv) {
#pragma unroll
      for (int tl = 0; tl < kW; ++tl) {
        const float2 x = r.q[tl];
        float2 y;
        y.x =
            GdnStableConv4(tap_q0, hist_q[0].x, hist_q[1].x, hist_q[2].x, x.x);
        y.y =
            GdnStableConv4(tap_q1, hist_q[0].y, hist_q[1].y, hist_q[2].y, x.y);
        reinterpret_cast<float2*>(&s_qk[lds][cq_which][tl][cq_slot])[cq_half] =
            y;
        hist_q[0] = hist_q[1];
        hist_q[1] = hist_q[2];
        hist_q[2] = x;
        if (has_v) {
          s_v[lds][tl][tid] =
              GdnStableConv4(tap_v, hist_v[0], hist_v[1], hist_v[2], r.v[tl]);
          hist_v[0] = hist_v[1];
          hist_v[1] = hist_v[2];
          hist_v[2] = r.v[tl];
        }
      }
      if (tid < kMiscLoads * kW) {
        s_misc[lds][m_tok][m_idx] = r.m;
      }
      return;
    } else {
      s_qk[lds][qk_which[0]][qk_tok[0]][qk_slot[0]] = r.qk0;
      s_qk[lds][qk_which[1]][qk_tok[1]][qk_slot[1]] = r.qk1;
#pragma unroll
      for (int j = 0; j < kVPerThread; ++j) {
        s_v[lds][v_tok[j]][v_row[j]] = r.v[j];
      }
      if (tid < 5 * kW) {
        s_misc[lds][m_tok][m_idx] = r.m;
      }
    }
  };

  const int n_windows = (static_cast<int>(n_tokens) + kW - 1) / kW;
  load_window(0, r0);
  if (n_windows > 1) {
    load_window(1, r1);
  }
  if (n_windows > 2) {
    load_window(2, r2);
  }
  commit_window(r0, 0);
  __syncthreads();
  float* out_base = raw + h * d;
  // `cur` held window w (committed already) and takes window w + 3;
  // `next` holds window w + 1, committed after this window's tokens.
  const auto window = [&](int w, Window& cur, const Window& next) {
    if (w >= n_windows) {
      return;
    }
    const int lds = w & 1;
    if (w + kGdnWindowsAhead < n_windows) {
      load_window(w + kGdnWindowsAhead, cur);
    }
    const int t_end = min(kW, static_cast<int>(n_tokens) - (w * kW));
    for (int tl = 0; tl < t_end; ++tl) {
      const float4* q4 = s_qk[lds][0][tl];
      const float4* k4 = s_qk[lds][1][tl];
      const float decay = s_misc[lds][tl][3];
      float u[kRows];
      float p[kRows];
      float4 kc[kVec];
      // Each row's u and p are one sequential FMA chain over its 32 keys in
      // natural order, the first term a plain product: the order the
      // single-row kernel compiled to, written out so it cannot change.
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        const float4 qv = q4[(i * kLanesPerRow) + segment];
        kc[i] = k4[(i * kLanesPerRow) + segment];
#pragma unroll
        for (int rr = 0; rr < kRows; ++rr) {
          s[rr][i].x *= decay;
          s[rr][i].y *= decay;
          s[rr][i].z *= decay;
          s[rr][i].w *= decay;
          if (i == 0) {
            u[rr] = s[rr][i].x * kc[i].x;
            p[rr] = s[rr][i].x * qv.x;
          } else {
            u[rr] = __builtin_fmaf(s[rr][i].x, kc[i].x, u[rr]);
            p[rr] = __builtin_fmaf(s[rr][i].x, qv.x, p[rr]);
          }
          u[rr] = __builtin_fmaf(s[rr][i].y, kc[i].y, u[rr]);
          p[rr] = __builtin_fmaf(s[rr][i].y, qv.y, p[rr]);
          u[rr] = __builtin_fmaf(s[rr][i].z, kc[i].z, u[rr]);
          p[rr] = __builtin_fmaf(s[rr][i].z, qv.z, p[rr]);
          u[rr] = __builtin_fmaf(s[rr][i].w, kc[i].w, u[rr]);
          p[rr] = __builtin_fmaf(s[rr][i].w, qv.w, p[rr]);
        }
      }
      float inv_k;
      float q_scale;
      float kq;
      if constexpr (kConv) {
        // GdnPrepKqKernel for this token: lane L takes the head's float4 L.
        const int slot = ((lane % kVec) * kLanesPerRow) + (lane / kVec);
        const float4 q = q4[slot];
        const float4 k = k4[slot];
        float qs = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        float ks = k.x * k.x + k.y * k.y + k.z * k.z + k.w * k.w;
        float dot = k.x * q.x + k.y * q.y + k.z * q.z + k.w * q.w;
#pragma unroll
        for (unsigned offset = 16; offset > 0; offset >>= 1) {
          qs += XorLane(qs, offset);
          ks += XorLane(ks, offset);
          dot += XorLane(dot, offset);
        }
        // Lane 0's sums, as GdnPrepKqKernel stores them.
        const auto first = [](float x) {
          return __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(
                                               __builtin_bit_cast(int, x)));
        };
        inv_k = rsqrtf(first(ks) + eps);
        q_scale = rsqrtf(first(qs) + eps) * rsqrtf(static_cast<float>(d));
        kq = first(dot);
      } else {
        inv_k = s_misc[lds][tl][0];
        q_scale = s_misc[lds][tl][1];
        kq = s_misc[lds][tl][2];
      }
      const float beta = s_misc[lds][tl][4];
#pragma unroll
      for (int rr = 0; rr < kRows; ++rr) {
        GdnRowReduce(u[rr], p[rr]);
        // (v - u * inv_k) * beta with the product fused into the subtract.
        float delta =
            __builtin_fmaf(-u[rr], inv_k, s_v[lds][tl][row_local(rr)]) * beta;
        // Keeps fast-math from folding beta * inv_k across the two products.
        asm volatile("" : "+v"(delta));
        const float update = delta * inv_k;
        if (segment == 0) {
          out_base[blockIdx.x * kRowsPerBlock + row_local(rr)] =
              __builtin_fmaf(kq, update, p[rr]) * q_scale;
        }
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          s[rr][i].x = __builtin_fmaf(update, kc[i].x, s[rr][i].x);
          s[rr][i].y = __builtin_fmaf(update, kc[i].y, s[rr][i].y);
          s[rr][i].z = __builtin_fmaf(update, kc[i].z, s[rr][i].z);
          s[rr][i].w = __builtin_fmaf(update, kc[i].w, s[rr][i].w);
        }
        if (checkpoint.state != nullptr &&
            w * kW + tl + 1 == checkpoint.tokens) {
          const int row = blockIdx.x * kRowsPerBlock + row_local(rr);
          auto* target = reinterpret_cast<float4*>(
              checkpoint.state + (static_cast<std::size_t>(h) * d + row) * d);
#pragma unroll
          for (int i = 0; i < kVec; ++i)
            target[vec0 + i] = s[rr][i];
        }
      }
      out_base += v_heads * d;
    }
    // The other slot was last read one window ago (before the previous
    // barrier), so window w + 1 goes in without a second barrier.
    if (w + 1 < n_windows) {
      commit_window(next, lds ^ 1);
    }
    __syncthreads();
  };
  for (int w = 0; w < n_windows; w += kGdnWindowsAhead) {
    window(w, r0, r1);
    window(w + 1, r1, r2);
    window(w + 2, r2, r0);
  }
#pragma unroll
  for (int rr = 0; rr < kRows; ++rr) {
    const int row = blockIdx.x * kRowsPerBlock + row_local(rr);
    auto* state4 = reinterpret_cast<float4*>(
        state + (static_cast<std::size_t>(h) * d + row) * d);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      state4[vec0 + i] = s[rr][i];
    }
  }
}

/// grid (value heads, row groups): each block owns kGdnRowsPerBlock state
/// rows of one head, kGdnLanes threads per row, each lane a contiguous
/// slice of the key dimension. Rows of the delta rule are independent, so
/// splitting a head over blocks only re-reads its q/k. The token loop is
/// serial; every reduction stays inside a lane group, so the loop runs
/// barrier-free. The raw attention rows go out unnormalized;
/// GdnEpilogueKernel finishes them.
constexpr unsigned kGdnLanes = 4;
constexpr unsigned kGdnRowsPerBlock = 32;

/// Record t of an n-token deferred speculative batch (GdnRollbackRowFloats).
__device__ __forceinline__ float* GdnRecord(const RollbackRows& rows,
                                            std::uint32_t t, std::uint32_t n,
                                            std::uint32_t k_heads,
                                            std::uint32_t v_heads) {
  return t + 1 == n && t != 0
             ? rows.rows[0] + GdnRollbackRecordFloats(k_heads, v_heads, kGdnDim)
             : rows.rows[t];
}

// The original vector kernel rounds error*key and then beta*correction for
// the last element of each lane, fusing the state decay into that result.
// HIP's FP intrinsics alone still permit reassociation of this three-product
// expression under fast-math. Keep its actual instruction sequence explicit.
__device__ __forceinline__ float GdnLastUpdate(float state, float decay,
                                               float error, float key,
                                               float beta) {
  float value;
  asm volatile(
      "v_mul_f32 %0, %1, %2\n\t"
      "v_mul_f32 %0, %0, %3\n\t"
      "v_fma_f32 %0, %4, %5, %0"
      : "=&v"(value)
      : "v"(error), "v"(key), "v"(beta), "v"(state), "v"(decay));
  return value;
}

template<bool kBatch>
__global__ void GdnKernel(const float* conv_out, const float* qn,
                          const float* kn, const float* alpha_beta,
                          const float* a, const float* dt, float* state,
                          float* raw, RollbackRows snapshots,
                          std::uint32_t n_tokens, std::uint32_t k_heads,
                          std::uint32_t v_heads, const GdnBatchItem* batch,
                          std::uint32_t active, GdnCheckpoint checkpoint) {
  if constexpr (kBatch) {
    if ((active & (1U << blockIdx.z)) == 0)
      return;
    const auto& item = batch[blockIdx.z];
    conv_out = item.conv_scratch;
    qn = item.qn;
    kn = item.kn;
    alpha_beta = item.alpha_beta;
    state = item.state;
    raw = item.raw;
    n_tokens = item.n_tokens;
  }
  constexpr std::uint32_t d = kGdnDim;
  constexpr std::uint32_t slice = d / kGdnLanes;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t kh = h % k_heads;
  const std::uint32_t j =
      blockIdx.y * kGdnRowsPerBlock + threadIdx.x / kGdnLanes;
  const std::uint32_t lane = threadIdx.x % kGdnLanes;
  const std::uint32_t i0 = lane * slice;
  const std::uint32_t channels = 2 * k_heads * d + v_heads * d;
  const float a_h = a[h];
  const float dt_h = dt[h];
  float* S = state + static_cast<std::size_t>(h) * d * d + j * d + i0;
  float row[slice];
#pragma unroll
  for (std::uint32_t i = 0; i < slice; ++i) {
    row[i] = S[i];
  }
  const float q_scale = rsqrtf(static_cast<float>(d));
  const bool deferred = (kBatch ? batch[blockIdx.z].state_snapshots.rows[0]
                                : snapshots.rows[0]) != nullptr &&
                        n_tokens > 1;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const float* q = qn + (static_cast<std::size_t>(t) * k_heads + kh) * d + i0;
    const float* k = kn + (static_cast<std::size_t>(t) * k_heads + kh) * d + i0;
    const float vv = conv_out[static_cast<std::size_t>(t) * channels +
                              2 * k_heads * d + h * d + j];
    const float alpha = alpha_beta[t * 2 * v_heads + h];
    const float beta = alpha_beta[t * 2 * v_heads + v_heads + h];
    const float decay = __expf(a_h * SoftplusF(alpha + dt_h));
    const float b = SigmoidF(beta);
    float kr[slice];
    float u = 0.0f;
    const float last = row[slice - 1];
#pragma unroll
    for (std::uint32_t i = 0; i < slice; ++i) {
      kr[i] = k[i];
      row[i] *= decay;
      u += row[i] * kr[i];
    }
#pragma unroll
    for (unsigned off = kGdnLanes / 2; off > 0; off >>= 1) {
      u += XorLane(u, off);
    }
    const float error = vv - u;
    float acc = 0.0f;
#pragma unroll
    for (std::uint32_t i = 0; i < slice; ++i) {
      // Preserve the original kernel's contraction: round error*key, then
      // fuse beta*correction into the decayed state. Materializing error*beta
      // for rollback would change this order under fast-math.
      const float correction = __fmul_rn(error, kr[i]);
      row[i] = i + 1 == slice ? GdnLastUpdate(last, decay, error, kr[i], b)
                              : __fmaf_rn(b, correction, row[i]);
      acc += row[i] * q[i];
    }
#pragma unroll
    for (unsigned off = kGdnLanes / 2; off > 0; off >>= 1) {
      acc += XorLane(acc, off);
    }
    if (lane == 0) {
      raw[static_cast<std::size_t>(t) * v_heads * d + h * d + j] =
          acc * q_scale;
    }
    if constexpr (!kBatch) {
      if (checkpoint.state != nullptr && t + 1 == checkpoint.tokens) {
        float* target =
            checkpoint.state + static_cast<std::size_t>(h) * d * d + j * d + i0;
#pragma unroll
        for (std::uint32_t i = 0; i < slice; ++i)
          target[i] = row[i];
      }
    }
    if (deferred) {
      float* snap =
          GdnRecord(kBatch ? batch[blockIdx.z].state_snapshots : snapshots, t,
                    n_tokens, k_heads, v_heads);
      if (j == 0 && h < k_heads) {
#pragma unroll
        for (std::uint32_t i = 0; i < slice; ++i)
          snap[h * d + i0 + i] = kr[i];
      }
      if (lane == 0) {
        if (j == 0) {
          snap[k_heads * d + h] = decay;
          snap[k_heads * d + v_heads + h] = b;
        }
        snap[k_heads * d + 2 * v_heads + h * d + j] = error;
      }
    }
  }
  // A speculative batch keeps the state it started from; RestoreGdnState
  // commits the accepted prefix from the records (no 3 MB/layer snapshot).
  if (deferred)
    return;
#pragma unroll
  for (std::uint32_t i = 0; i < slice; ++i) {
    S[i] = row[i];
  }
}

__global__ void RestoreGdnStateKernel(float* state, RollbackRows snapshots,
                                      std::uint32_t keep, std::uint32_t n,
                                      std::uint32_t k_heads,
                                      std::uint32_t v_heads) {
  constexpr std::uint32_t d = kGdnDim;
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= std::size_t{v_heads} * d * d)
    return;
  const auto h = i / (d * d);
  const auto j = (i / d) % d;
  const auto k = i % d;
  float value = state[i];
  for (std::uint32_t t = 0; t < keep; ++t) {
    const auto* update = GdnRecord(snapshots, t, n, k_heads, v_heads);
    const float key = update[(h % k_heads) * d + k];
    const float decay = update[k_heads * d + h];
    const float beta = update[k_heads * d + v_heads + h];
    const float error = update[k_heads * d + 2 * v_heads + h * d + j];
    const float correction = __fmul_rn(error, key);
    value = (k + 1) % (d / kGdnLanes) == 0
                ? GdnLastUpdate(value, decay, error, key, beta)
                : __fmaf_rn(beta, correction, __fmul_rn(value, decay));
  }
  state[i] = value;
}

/// Per-head RMSNorm of the raw attention rows and the sigmoid output gate,
/// one wave per (token, head) row.
/// A non-null `out_q8` receives the row quantized into the tiled Q8 layout
/// of the ssm_out projection (K = v_heads * d) in place of the F32 row:
/// each wave-wide slice of 32 lanes is one K block. Otherwise a non-null
/// `out_q8_1` also receives the F32 rows' unpadded Q8_1 rows. Non-empty
/// `slabs` (F32 rows only; k == v_heads * d) also receive each token's F32
/// row rotated for the EXL3 output projection.
template<bool kBatch>
__global__ void GdnEpilogueKernel(const float* raw, const float* z,
                                  std::uint32_t z_stride, const float* norm_w,
                                  float* out, void* out_q8, __half* out_half,
                                  std::uint32_t n_rows, std::uint32_t v_heads,
                                  float eps, const GdnBatchItem* batch,
                                  std::uint32_t active, void* out_q8_1,
                                  std::uint32_t out_half_stride = 0,
                                  Exl3InputSlabs slabs = {}) {
  if constexpr (kBatch) {
    if ((active & (1U << blockIdx.z)) == 0)
      return;
    const auto& item = batch[blockIdx.z];
    raw = item.raw;
    z = item.z;
    out = item.out;
    n_rows = item.n_tokens * v_heads;
  }
  constexpr std::uint32_t d = kGdnDim;
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::size_t row =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= n_rows) {
    return;
  }
  // z rows are [t][v_heads*d] with a caller-side row stride.
  const float* zrow = z + (row / v_heads) * z_stride + (row % v_heads) * d;
  const std::uint32_t per_lane = d / warpSize;
  const float* src = raw + row * d;
  float v[d / 32];
  float ss = 0.0f;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    v[r] = src[r * warpSize + lane];
    ss += v[r] * v[r];
  }
  const float scale = rsqrtf(WaveSum(ss) / static_cast<float>(d) + eps);
  if (out_q8 == nullptr) {
    float values[d / 32];
#pragma unroll
    for (std::uint32_t r = 0; r < per_lane; ++r) {
      const std::uint32_t i = r * warpSize + lane;
      const float value = v[r] * scale * norm_w[i] * SigmoidF(zrow[i]);
      values[r] = value;
      if (out_half != nullptr) {
        const auto offset =
            out_half_stride != 0
                ? (row / v_heads) * out_half_stride + (row % v_heads) * d
                : row * d;
        out_half[offset + i] = __float2half_rn(value);
      } else {
        out[row * d + i] = value;
        if (out_q8_1 != nullptr)
          QuantizeQ8_1Lane(value, out_q8_1, row * d + i);
      }
    }
    if constexpr (!kBatch) {
      if (slabs.count != 0 && out_half == nullptr) {
        // The head row is one segment; lane l takes elements 4l .. 4l + 3,
        // held as values[l / 8] by lanes (4l + j) % 32.
        static_assert(d == hip::exl3::kSegment && d / 32 == 4);
        float seg[4];
#pragma unroll
        for (std::uint32_t j = 0; j < 4; ++j) {
          const int source = static_cast<int>((4 * lane + j) % warpSize);
          float pick = 0.0f;
#pragma unroll
          for (std::uint32_t r = 0; r < 4; ++r) {
            const float moved = __shfl(values[r], source, warpSize);
            pick = lane / 8 == r ? moved : pick;
          }
          seg[j] = pick;
        }
        RotateIntoSlabs(seg, lane, (row % v_heads) * d + 4 * lane,
                        row / v_heads, slabs);
      }
    }
    return;
  }
  const std::size_t tok = row / v_heads;
  const std::size_t head = row % v_heads;
  const std::size_t num_blocks = (static_cast<std::size_t>(v_heads) * d) / 32;
  const std::size_t tl = tok % kQ8ActTileTokens;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    const std::uint32_t i = r * warpSize + lane;
    const float n = v[r] * scale * norm_w[i] * SigmoidF(zrow[i]);
    float max_abs = fabsf(n);
    for (int off = 16; off > 0; off >>= 1) {
      max_abs = fmaxf(max_abs, XorLane(max_abs, off));
    }
    const float dq = max_abs / 127.0F;
    const float id = (dq != 0.0F) ? (1.0F / dq) : 0.0F;
    const auto q = static_cast<std::int8_t>(roundf(n * id));
    const std::size_t kb = ((head * d) + (r * warpSize)) / 32;
    std::int8_t* tile =
        Q8ActTile(out_q8, num_blocks, tok / kQ8ActTileTokens, kb);
    tile[((lane >> 4u) * 256) + (tl * 16) + (lane & 15u)] = q;
    if (lane == 0) {
      *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                (tl * sizeof(float))) = dq;
    }
  }
}

/// Row j of the rolling state after token t is row (t + 1 + j) of the
/// concatenation [history ; rows], for any history depth `hist`.
__global__ void RollingSnapshotKernel(
    const float* rows, std::uint32_t row_stride, const float* history,
    RollbackRows snapshots, std::uint32_t n_tokens, std::uint32_t channels,
    std::uint32_t hist) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t per_token = static_cast<std::size_t>(hist) * channels;
  if (idx >= per_token * n_tokens) {
    return;
  }
  const std::uint32_t t = idx / per_token;
  const std::uint32_t j = (idx % per_token) / channels;
  const std::uint32_t c = idx % channels;
  const std::uint32_t src = t + 1 + j;
  snapshots.rows[t][idx % per_token] =
      src < hist ? history[static_cast<std::size_t>(src) * channels + c]
                 : rows[static_cast<std::size_t>(src - hist) * row_stride + c];
}

/// Causal conv over the chunk with the rolling state, SiLU applied.
__global__ void SsmConvKernel(const float* qkv, std::uint32_t qkv_stride,
                              const float* w, const float* conv_state,
                              float* out, std::uint32_t n_tokens,
                              std::uint32_t channels, std::uint32_t kernel) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<std::size_t>(n_tokens) * channels) {
    return;
  }
  const std::uint32_t t = idx / channels;
  const std::uint32_t c = idx % channels;
  float acc = 0.0f;
  for (std::uint32_t k = 0; k < kernel; ++k) {
    const std::int32_t src_t = static_cast<std::int32_t>(t) -
                               static_cast<std::int32_t>(kernel - 1 - k);
    const float v =
        src_t >= 0 ? qkv[static_cast<std::size_t>(src_t) * qkv_stride + c]
                   : conv_state[static_cast<std::size_t>(kernel - 1 + src_t) *
                                    channels +
                                c];
    acc += w[static_cast<std::size_t>(c) * kernel + k] * v;
  }
  out[idx] = SiluF(acc);
}

constexpr std::uint32_t kSsmConvTokensPerThread = 8;
template<bool kSaveHistory, bool kBatch, bool stable = false>
__global__ void SsmConv4Kernel(const float* qkv, std::uint32_t qkv_stride,
                               const float* w, float* conv_state, float* out,
                               std::uint32_t n_tokens, std::uint32_t channels,
                               RollbackRows snapshots,
                               const GdnBatchItem* batch,
                               std::uint32_t active) {
  if constexpr (kBatch) {
    if ((active & (1U << blockIdx.z)) == 0)
      return;
    const auto& item = batch[blockIdx.z];
    qkv = item.qkv;
    conv_state = item.conv_state;
    out = item.conv_scratch;
    n_tokens = item.n_tokens;
  }
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) {
    return;
  }
  const std::uint32_t t0 = blockIdx.y * kSsmConvTokensPerThread;
  const float4 taps = *reinterpret_cast<const float4*>(w + c * kSsmConvTaps);
  // window[i] holds token t0 - 3 + i.
  float window[kSsmConvTaps - 1];
#pragma unroll
  for (std::uint32_t i = 0; i < kSsmConvTaps - 1; ++i) {
    const std::int32_t src_t =
        static_cast<std::int32_t>(t0) - 3 + static_cast<std::int32_t>(i);
    window[i] =
        src_t >= 0
            ? qkv[static_cast<std::size_t>(src_t) * qkv_stride + c]
            : conv_state[static_cast<std::size_t>(kSsmConvTaps - 1 + src_t) *
                             channels +
                         c];
  }
#pragma unroll
  for (std::uint32_t i = 0; i < kSsmConvTokensPerThread; ++i) {
    const std::uint32_t t = t0 + i;
    if (t >= n_tokens) {
      break;
    }
    const float v = qkv[static_cast<std::size_t>(t) * qkv_stride + c];
    float acc;
    if constexpr (stable) {
      // EXL3 keeps the same multiply then three FMA boundaries for every
      // token tile, independent of history-saving and chunk width.
      acc = Materialize(Materialize(taps.x) * Materialize(window[0]));
      acc = Materialize(
          __builtin_fmaf(Materialize(taps.y), Materialize(window[1]), acc));
      acc = Materialize(
          __builtin_fmaf(Materialize(taps.z), Materialize(window[2]), acc));
      acc =
          Materialize(__builtin_fmaf(Materialize(taps.w), Materialize(v), acc));
    } else {
      acc = taps.x * window[0] + taps.y * window[1] + taps.z * window[2] +
            taps.w * v;
    }
    out[static_cast<std::size_t>(t) * channels + c] = SiluF(acc);
    window[0] = window[1];
    window[1] = window[2];
    window[2] = v;
    if constexpr (kSaveHistory) {
      if ((kBatch ? batch[blockIdx.z].conv_snapshots.rows[0]
                  : snapshots.rows[0]) != nullptr &&
          t + 1 < n_tokens) {
        float* snapshot = kBatch ? batch[blockIdx.z].conv_snapshots.rows[t]
                                 : snapshots.rows[t];
#pragma unroll
        for (std::uint32_t j = 0; j < kSsmConvTaps - 1; ++j)
          snapshot[std::size_t{j} * channels + c] = window[j];
      }
    }
  }
  if constexpr (kSaveHistory) {
    // Only one token tile may update history: each channel then has one
    // owner, which has already consumed all three original history values.
#pragma unroll
    for (std::uint32_t j = 0; j < kSsmConvTaps - 1; ++j)
      conv_state[std::size_t{j} * channels + c] = window[j];
  }
}

__global__ void CopyKernel(const float* src, float* dst, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    dst[i] = src[i];
  }
}

}  // namespace

void PleGate(const float* key_n, const float* query_n, const float* value,
             float* gated, std::uint32_t n_tokens, std::uint32_t hidden,
             std::uint32_t streams, hipStream_t stream) {
  hipLaunchKernelGGL(PleGateKernel, dim3(n_tokens, streams), dim3(kThreads), 0,
                     stream, key_n, query_n, value, gated, hidden, streams);
}

void PleConv(const float* in, const float* w, float* history,
             float* history_scratch, float* out, RollbackRows snapshots,
             std::uint32_t n_tokens, std::uint32_t channels,
             std::uint32_t kernel, std::uint32_t dilation, hipStream_t stream) {
  const std::uint32_t hist = (kernel - 1) * dilation;
  const std::size_t count = static_cast<std::size_t>(n_tokens) * channels;
  hipLaunchKernelGGL(PleConvKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, in, w, history, out, n_tokens, channels, kernel,
                     dilation, hist);
  if (snapshots.rows[0] != nullptr && n_tokens > 1) {
    const std::size_t saved = static_cast<std::size_t>(n_tokens - 1) * channels;
    hipLaunchKernelGGL(RollingSnapshotKernel, dim3(Blocks(saved * hist)),
                       dim3(kThreads), 0, stream, in, channels, history,
                       snapshots, n_tokens - 1, channels, hist);
  }
  // Device copies stay kernels: a copy engine transfer is not reliably
  // ordered behind the kernels on this stream.
  const std::size_t hist_count = static_cast<std::size_t>(hist) * channels;
  hipLaunchKernelGGL(HistoryShiftKernel, dim3(Blocks(hist_count)),
                     dim3(kThreads), 0, stream, in, channels, history,
                     history_scratch, n_tokens, channels, hist);
  hipLaunchKernelGGL(CopyKernel, dim3(Blocks(hist_count)), dim3(kThreads), 0,
                     stream, history_scratch, history, hist_count);
}

void PleInject(float* res, const float* gated, const float* conv,
               std::size_t count, hipStream_t stream) {
  hipLaunchKernelGGL(PleInjectKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, res, gated, conv, count);
}

void RestoreGdnState(float* state, RollbackRows snapshots, std::uint32_t keep,
                     std::uint32_t n, std::uint32_t k_heads,
                     std::uint32_t v_heads, hipStream_t stream) {
  const auto count = std::size_t{v_heads} * kGdnDim * kGdnDim;
  RestoreGdnStateKernel<<<(count + 255) / 256, 256, 0, stream>>>(
      state, snapshots, keep, n, k_heads, v_heads);
}

void HistoryPrefix(const float* in, std::uint32_t stride, const float* history,
                   float* destination, std::uint32_t tokens,
                   std::uint32_t channels, std::uint32_t history_rows,
                   hipStream_t stream) {
  hipLaunchKernelGGL(HistoryShiftKernel,
                     dim3(Blocks(std::size_t{history_rows} * channels)),
                     dim3(kThreads), 0, stream, in, stride, history,
                     destination, tokens, channels, history_rows);
}

bool GatedDeltaNet(const float* qkv, std::uint32_t qkv_stride, const float* z,
                   std::uint32_t z_stride, const float* alpha_beta,
                   const float* conv_w, const float* a, const float* dt,
                   const float* norm_w, float* conv_state, float* conv_scratch,
                   float* qn, float* kn, float* raw, float* state, float* out,
                   void* out_q8, RollbackRows state_snapshots,
                   RollbackRows conv_snapshots, std::uint32_t n_tokens,
                   std::uint32_t k_heads, std::uint32_t v_heads,
                   std::uint32_t d, std::uint32_t kernel, bool row_split,
                   bool convolved, float eps, hipStream_t stream,
                   __half* out_half, GdnCheckpoint checkpoint,
                   std::uint32_t out_half_stride, void* out_q8_1,
                   hipEvent_t z_ready, bool stable_conv,
                   Exl3InputSlabs* out_slabs) {
  const std::uint32_t channels = 2 * k_heads * d + v_heads * d;
  const std::size_t count = static_cast<std::size_t>(n_tokens) * channels;
  if (checkpoint.history != nullptr)
    HistoryPrefix(qkv, qkv_stride, conv_state, checkpoint.history,
                  checkpoint.tokens, channels, kernel - 1, stream);
  // EXL3 prefill: the recurrence convolves its channels itself. It reads
  // q|k channel pairs as float2, so token rows must be 8-byte aligned.
  const bool fused_conv =
      row_split && d == kGdnDim && state_snapshots.rows[0] == nullptr &&
      stable_conv && !convolved && kernel == kSsmConvTaps &&
      conv_snapshots.rows[0] == nullptr && qkv_stride % 2 == 0 &&
      reinterpret_cast<std::uintptr_t>(qkv) % alignof(float2) == 0;
  // One token tile: each channel's single owner shifts its history (and
  // writes the rollback rows) in place, replacing the HistoryShift, Copy and
  // RollingSnapshot launches with the same copied values. EXL3's stable
  // instantiation keeps its pinned arithmetic.
  const bool saved_history = !convolved && !fused_conv &&
                             kernel == kSsmConvTaps &&
                             n_tokens <= kSsmConvTokensPerThread;
  if (!convolved && !fused_conv) {
    if (saved_history && stable_conv) {
      hipLaunchKernelGGL((SsmConv4Kernel<true, false, true>),
                         dim3(Blocks(channels)), dim3(kThreads), 0, stream, qkv,
                         qkv_stride, conv_w, conv_state, conv_scratch, n_tokens,
                         channels, conv_snapshots, nullptr, 0);
    } else if (stable_conv && kernel == kSsmConvTaps) {
      // Wider EXL3 chunks: the same arithmetic over several token tiles;
      // history updates below use the stream-ordered shift.
      hipLaunchKernelGGL(
          (SsmConv4Kernel<false, false, true>),
          dim3(Blocks(channels), (n_tokens + kSsmConvTokensPerThread - 1) /
                                     kSsmConvTokensPerThread),
          dim3(kThreads), 0, stream, qkv, qkv_stride, conv_w, conv_state,
          conv_scratch, n_tokens, channels, RollbackRows{}, nullptr, 0);
    } else if (saved_history) {
      hipLaunchKernelGGL((SsmConv4Kernel<true, false>), dim3(Blocks(channels)),
                         dim3(kThreads), 0, stream, qkv, qkv_stride, conv_w,
                         conv_state, conv_scratch, n_tokens, channels,
                         conv_snapshots, nullptr, 0);
    } else if (kernel == kSsmConvTaps) {
      hipLaunchKernelGGL(
          (SsmConv4Kernel<false, false>),
          dim3(Blocks(channels), (n_tokens + kSsmConvTokensPerThread - 1) /
                                     kSsmConvTokensPerThread),
          dim3(kThreads), 0, stream, qkv, qkv_stride, conv_w, conv_state,
          conv_scratch, n_tokens, channels, RollbackRows{}, nullptr, 0);
    } else {
      hipLaunchKernelGGL(SsmConvKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                         stream, qkv, qkv_stride, conv_w, conv_state,
                         conv_scratch, n_tokens, channels, kernel);
    }
  }
  // The rolling state is the last kernel-1 projections: [history ; qkv].
  const auto shift_history = [&] {
    const std::uint32_t hist = kernel - 1;
    const std::size_t hist_count = static_cast<std::size_t>(hist) * channels;
    hipLaunchKernelGGL(HistoryShiftKernel, dim3(Blocks(hist_count)),
                       dim3(kThreads), 0, stream, qkv, qkv_stride, conv_state,
                       conv_scratch + count, n_tokens, channels, hist);
    hipLaunchKernelGGL(CopyKernel, dim3(Blocks(hist_count)), dim3(kThreads), 0,
                       stream, conv_scratch + count, conv_state, hist_count);
  };
  const unsigned waves = kThreads / 32;
  if (fused_conv) {
    hipLaunchKernelGGL(
        GdnPrepAbKernel,
        dim3(Blocks(static_cast<std::size_t>(n_tokens) * v_heads)),
        dim3(kThreads), 0, stream, alpha_beta, a, dt, kn,
        static_cast<std::size_t>(n_tokens) * v_heads, v_heads);
    hipLaunchKernelGGL(GdnRowSplitKernel<true>,
                       dim3(kGdnDim / kGdnRowSplitRowsPerBlock, v_heads),
                       dim3(kGdnWaves * 32), 0, stream, conv_scratch, qn, kn,
                       state, raw, n_tokens, k_heads, v_heads, checkpoint, qkv,
                       qkv_stride, conv_w, conv_state, eps);
    shift_history();
  } else if (!saved_history) {
    if (conv_snapshots.rows[0] != nullptr && n_tokens > 1) {
      const std::size_t saved =
          static_cast<std::size_t>(n_tokens - 1) * channels;
      hipLaunchKernelGGL(RollingSnapshotKernel,
                         dim3(Blocks(saved * (kernel - 1))), dim3(kThreads), 0,
                         stream, qkv, qkv_stride, conv_state, conv_snapshots,
                         n_tokens - 1, channels, kernel - 1);
    }
    shift_history();
  }
  if (fused_conv) {
    // Recurrence launched above.
  } else if (row_split && d == kGdnDim && state_snapshots.rows[0] == nullptr) {
    hipLaunchKernelGGL(GdnPrepKqKernel, dim3(k_heads, n_tokens), dim3(32), 0,
                       stream, conv_scratch, qn, n_tokens, k_heads, channels,
                       eps);
    hipLaunchKernelGGL(
        GdnPrepAbKernel,
        dim3(Blocks(static_cast<std::size_t>(n_tokens) * v_heads)),
        dim3(kThreads), 0, stream, alpha_beta, a, dt, kn,
        static_cast<std::size_t>(n_tokens) * v_heads, v_heads);
    hipLaunchKernelGGL(GdnRowSplitKernel<false>,
                       dim3(kGdnDim / kGdnRowSplitRowsPerBlock, v_heads),
                       dim3(kGdnWaves * 32), 0, stream, conv_scratch, qn, kn,
                       state, raw, n_tokens, k_heads, v_heads, checkpoint,
                       nullptr, 0U, nullptr, nullptr, eps);
  } else {
    hipLaunchKernelGGL(GdnPrepKernel<false>,
                       dim3((n_tokens * k_heads + waves - 1) / waves),
                       dim3(kThreads), 0, stream, conv_scratch, qn, kn,
                       n_tokens * k_heads, k_heads, channels, eps, nullptr, 0);
    hipLaunchKernelGGL(
        GdnKernel<false>, dim3(v_heads, kGdnDim / kGdnRowsPerBlock),
        dim3(kGdnRowsPerBlock * kGdnLanes), 0, stream, conv_scratch, qn, kn,
        alpha_beta, a, dt, state, raw, state_snapshots, n_tokens, k_heads,
        v_heads, nullptr, 0, checkpoint);
  }
  if (z_ready != nullptr &&
      hipStreamWaitEvent(stream, z_ready, 0) != hipSuccess)
    return false;
  const bool rotate = out_slabs != nullptr && out_slabs->count == 1 &&
                      out_q8 == nullptr && out_half == nullptr &&
                      d == kGdnDim && out_slabs->k == v_heads * d &&
                      n_tokens <= out_slabs->pass_rows;
  if (out_slabs != nullptr && !rotate)
    out_slabs->count = 0;
  hipLaunchKernelGGL(
      GdnEpilogueKernel<false>, dim3((n_tokens * v_heads + waves - 1) / waves),
      dim3(kThreads), 0, stream, raw, z, z_stride, norm_w, out, out_q8,
      out_half, n_tokens * v_heads, v_heads, eps, nullptr, 0,
      out_q8 == nullptr && out_half == nullptr && d == kGdnDim ? out_q8_1
                                                               : nullptr,
      out_half_stride, rotate ? *out_slabs : Exl3InputSlabs{});
  return true;
}

bool GatedDeltaNetBatch(const GdnBatchItem* items, std::uint32_t count,
                        std::uint32_t max_tokens, std::uint32_t active,
                        std::uint32_t qkv_stride, std::uint32_t z_stride,
                        const float* conv_w, const float* a, const float* dt,
                        const float* norm_w, std::uint32_t k_heads,
                        std::uint32_t v_heads, float eps, hipStream_t stream,
                        bool stable_conv) {
  if (items == nullptr || count == 0 || count > 8 || max_tokens == 0 ||
      max_tokens > kSsmConvTokensPerThread)
    return false;
  const auto channels = (2 * k_heads + v_heads) * kGdnDim;
  constexpr auto waves = kThreads / 32;
  const auto conv = stable_conv ? SsmConv4Kernel<true, true, true>
                                : SsmConv4Kernel<true, true, false>;
  hipLaunchKernelGGL(conv, dim3(Blocks(channels), 1, count), dim3(kThreads), 0,
                     stream, nullptr, qkv_stride, conv_w, nullptr, nullptr,
                     max_tokens, channels, RollbackRows{}, items, active);
  hipLaunchKernelGGL(GdnPrepKernel<true>,
                     dim3((max_tokens * k_heads + waves - 1) / waves, 1, count),
                     dim3(kThreads), 0, stream, nullptr, nullptr, nullptr,
                     max_tokens * k_heads, k_heads, channels, eps, items,
                     active);
  hipLaunchKernelGGL(
      GdnKernel<true>, dim3(v_heads, kGdnDim / kGdnRowsPerBlock, count),
      dim3(kGdnRowsPerBlock * kGdnLanes), 0, stream, nullptr, nullptr, nullptr,
      nullptr, a, dt, nullptr, nullptr, RollbackRows{}, max_tokens, k_heads,
      v_heads, items, active, GdnCheckpoint{});
  hipLaunchKernelGGL(GdnEpilogueKernel<true>,
                     dim3((max_tokens * v_heads + waves - 1) / waves, 1, count),
                     dim3(kThreads), 0, stream, nullptr, nullptr, z_stride,
                     norm_w, nullptr, nullptr, nullptr, max_tokens * v_heads,
                     v_heads, eps, items, active, nullptr);
  return hipGetLastError() == hipSuccess;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
