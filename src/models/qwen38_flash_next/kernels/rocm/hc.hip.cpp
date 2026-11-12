// Hyper-connection (HC) stream mix epilogues, combines, deferred residual
// updates and the norm read back from the residual.
// HIP kernels follow the layouts and operator formulas in reference.cpp.

#include "src/models/qwen38_flash_next/kernels/rocm/kernels_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

/// grid (tokens, hidden chunks of kThreads): the stream loop stays inside
/// the thread. With `inject_w` every block also contributes its slice of the
/// inject dot products, written as partial sums [t][stream][chunk] that
/// HcCombine totals (a fixed-order reduction, so the result is stable).
/// `kStreams` fixes the stream count (0: the `streams` argument, up to 8) so
/// the stream loops unroll: every read is issued up front, the inject
/// weights before the `mixed` store, and the sums still run stream by
/// stream in order.
/// Non-empty `slabs` (hidden % 256 == 0: every block holds two whole
/// segments of its row) also receive the row rotated for the EXL3 decode
/// projections that read `mixed` next: wave w rotates segment w % 2 for
/// consumer w / 2 from the block's stored values.
template<std::uint32_t kStreams>
__global__ void HcMixEpilogueKernel(const float* xn, const float* gate,
                                    const float* inject_w, float* mixed,
                                    float* inject, std::uint32_t hidden,
                                    std::uint32_t streams, void* mixed_q8,
                                    Exl3InputSlabs slabs) {
  constexpr std::uint32_t kMaxStreams = 8;
  constexpr std::uint32_t kSlots = kStreams != 0 ? kStreams : kMaxStreams;
  if constexpr (kStreams != 0) {
    streams = kStreams;
  }
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
  const std::size_t hc_dim = static_cast<std::size_t>(streams) * hidden;
  const float* x = xn + static_cast<std::size_t>(t) * hc_dim;
  const float* g = gate + static_cast<std::size_t>(t) * hc_dim;
  const bool live = i < hidden;
  float v[kSlots];
  float gv[kSlots];
#pragma unroll
  for (std::uint32_t s = 0; s < kSlots; ++s) {
    const std::size_t idx = static_cast<std::size_t>(s) * hidden + i;
    v[s] = live && s < streams ? x[idx] : 0.0f;
    gv[s] = live && s < streams ? g[idx] : 0.0f;
  }
  float wv[kSlots][kSlots];
#pragma unroll
  for (std::uint32_t o = 0; o < kSlots; ++o) {
#pragma unroll
    for (std::uint32_t s = 0; s < kSlots; ++s) {
      wv[o][s] =
          inject_w != nullptr && live && o < streams && s < streams
              ? inject_w[o * hc_dim + static_cast<std::size_t>(s) * hidden + i]
              : 0.0f;
    }
  }
  float acc = 0.0f;
#pragma unroll
  for (std::uint32_t s = 0; s < kSlots; ++s) {
    if (live && s < streams) {
      acc = Materialize(acc + v[s] * SigmoidF(gv[s]));
    }
  }
  float out = 0.0f;
  if (live) {
    out = acc / static_cast<float>(streams);
    mixed[static_cast<std::size_t>(t) * hidden + i] = out;
    // hidden % 32 == 0, so whole 32-lane groups take this branch together.
    if (mixed_q8 != nullptr) {
      QuantizeQ8_1Lane(out, mixed_q8, static_cast<std::size_t>(t) * hidden + i);
    }
  }
  if (slabs.count != 0) {
    __shared__ float row[kThreads];
    row[threadIdx.x] = out;
    __syncthreads();
    const std::uint32_t w = threadIdx.x / warpSize;
    if (w < 2 * slabs.count) {
      const std::uint32_t l = threadIdx.x % warpSize;
      const std::uint32_t local = (w % 2) * hip::exl3::kSegment + 4 * l;
      float seg[4] = {row[local], row[local + 1], row[local + 2],
                      row[local + 3]};
      const SlabTarget target = SlabOf(slabs, w / 2);
      hip::exl3::RotateInputSegment(seg, l, blockIdx.y * kThreads + local,
                                    target.suh, target.out, t, slabs.pass_rows);
    }
  }
  if (inject_w == nullptr) {
    return;
  }
  // All inject dots reduce through one shared pass: wave sums first, then
  // one thread per logit totals the waves.
  __shared__ float partial[kMaxStreams][kThreads / 32];
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t wave = threadIdx.x / warpSize;
#pragma unroll
  for (std::uint32_t o = 0; o < kSlots; ++o) {
    if (o >= streams) {
      break;
    }
    float dot = 0.0f;
    if (live) {
#pragma unroll
      for (std::uint32_t s = 0; s < kSlots; ++s) {
        if (s < streams) {
          dot = Materialize(dot + wv[o][s] * v[s]);
        }
      }
    }
    dot = WaveSum(dot);
    if (lane == 0) {
      partial[o][wave] = dot;
    }
  }
  __syncthreads();
  if (threadIdx.x < streams) {
    float total = 0.0f;
    for (std::uint32_t w = 0; w < blockDim.x / warpSize; ++w) {
      total += partial[threadIdx.x][w];
    }
    inject[(static_cast<std::size_t>(t) * streams + threadIdx.x) * gridDim.y +
           blockIdx.y] = total;
  }
}

/// r + b * w as the four-lane combine forms the updated residual, one FMA
/// per element: also the expression every reader of a deferred update
/// (DeferredResidual) reapplies, so both give the same bits.
__device__ __forceinline__ float4 CombineAdd(float4 r, float4 b, float w) {
  return float4{__fmaf_rn(b.x, w, r.x), __fmaf_rn(b.y, w, r.y),
                __fmaf_rn(b.z, w, r.z), __fmaf_rn(b.w, w, r.w)};
}

/// One stream's normalized float4, as HcCombineVec4Kernel stores it and
/// its consumers recompute it: v * (scale * gamma) per element, the order
/// fast-math gave the combine's original v * scale * gamma.
__device__ __forceinline__ float4 HcNorm4(float4 v, float sc, float4 gm) {
  // The result is pinned too: a consumer's own products must not fold
  // into this one.
  return float4{Materialize(v.x * Materialize(sc * gm.x)),
                Materialize(v.y * Materialize(sc * gm.y)),
                Materialize(v.z * Materialize(sc * gm.z)),
                Materialize(v.w * Materialize(sc * gm.w))};
}

/// kFromRes: xn is the residual row and the norm is recomputed from it,
/// gamma and the combine's per-stream scales (HcNorm4).
/// GateT: the HC gate rows as stored by the up projection (FP32, or F16
/// from HcUpF16Gemm).
template<typename XnT, bool kFromRes = false, typename GateT = float>
__global__ void HcMixEpilogueVec4Kernel(const XnT* xn, const GateT* gate,
                                        const float* inject_w, float* mixed,
                                        float* inject, std::uint32_t hidden,
                                        const float* gamma = nullptr,
                                        const float* xn_scale = nullptr,
                                        __half* mixed_half = nullptr,
                                        DeferredResidual prior = {}) {
  constexpr std::uint32_t kStreams = 4;
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  const std::size_t hc_dim = static_cast<std::size_t>(kStreams) * hidden;
  const XnT* x = xn + static_cast<std::size_t>(t) * hc_dim;
  float4 v[kStreams];
  float4 acc{0.0F, 0.0F, 0.0F, 0.0F};
  if constexpr (kFromRes) {
    // Every load is issued before the first pinned product, which would
    // otherwise wait on each stream's row in turn.
    float4 gm[kStreams];
    float sc[kStreams];
#pragma unroll
    for (std::uint32_t s = 0; s < kStreams; ++s) {
      const std::size_t idx = static_cast<std::size_t>(s) * hidden + i;
      v[s] = i < hidden ? Load4(x + idx) : float4{0.0F, 0.0F, 0.0F, 0.0F};
      gm[s] = i < hidden ? *reinterpret_cast<const float4*>(gamma + idx)
                         : float4{0.0F, 0.0F, 0.0F, 0.0F};
      sc[s] = xn_scale[(static_cast<std::size_t>(t) * kStreams) + s];
    }
    // A deferred combine left its update out of res: the combine's sum.
    if (prior.block_out != nullptr && i < hidden) {
      const float4 b =
          Load4(prior.block_out + (static_cast<std::size_t>(t) * hidden) + i);
#pragma unroll
      for (std::uint32_t s = 0; s < kStreams; ++s)
        v[s] = CombineAdd(
            v[s], b, prior.w[(static_cast<std::size_t>(t) * kStreams) + s]);
    }
#pragma unroll
    for (std::uint32_t s = 0; s < kStreams; ++s) {
      if (i < hidden)
        v[s] = HcNorm4(v[s], sc[s], gm[s]);
    }
  }
#pragma unroll
  for (std::uint32_t s = 0; s < kStreams; ++s) {
    const std::size_t idx = static_cast<std::size_t>(s) * hidden + i;
    if constexpr (!kFromRes)
      v[s] = i < hidden ? Load4(x + idx) : float4{0.0F, 0.0F, 0.0F, 0.0F};
    if (i < hidden) {
      float4 g;
      if constexpr (std::is_same_v<GateT, __half>) {
        __half h[4];
        *reinterpret_cast<uint2*>(h) = *reinterpret_cast<const uint2*>(
            gate + static_cast<std::size_t>(t) * hc_dim + idx);
        g = float4{__half2float(h[0]), __half2float(h[1]), __half2float(h[2]),
                   __half2float(h[3])};
      } else {
        g = *reinterpret_cast<const float4*>(
            gate + static_cast<std::size_t>(t) * hc_dim + idx);
      }
      acc.x += v[s].x * SigmoidF(g.x);
      acc.y += v[s].y * SigmoidF(g.y);
      acc.z += v[s].z * SigmoidF(g.z);
      acc.w += v[s].w * SigmoidF(g.w);
    }
  }
  if (i < hidden) {
    constexpr float kInvStreams = 1.0F / static_cast<float>(kStreams);
    acc.x *= kInvStreams;
    acc.y *= kInvStreams;
    acc.z *= kInvStreams;
    acc.w *= kInvStreams;
    *reinterpret_cast<float4*>(mixed + static_cast<std::size_t>(t) * hidden +
                               i) = acc;
    // NarrowHalf4Kernel's conversion of the stored row, for an F16 consumer.
    if (mixed_half != nullptr)
      *reinterpret_cast<ushort4*>(mixed_half +
                                  static_cast<std::size_t>(t) * hidden + i) =
          make_ushort4(__half_as_ushort(__half(Materialize(acc.x))),
                       __half_as_ushort(__half(Materialize(acc.y))),
                       __half_as_ushort(__half(Materialize(acc.z))),
                       __half_as_ushort(__half(Materialize(acc.w))));
  }
  if (inject_w == nullptr) {
    return;
  }
  __shared__ float partial[kStreams][kThreads / 32];
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t wave = threadIdx.x / warpSize;
#pragma unroll
  for (std::uint32_t o = 0; o < kStreams; ++o) {
    const float* w = inject_w + static_cast<std::size_t>(o) * hc_dim;
    float dot = 0.0F;
    if (i < hidden) {
#pragma unroll
      for (std::uint32_t s = 0; s < kStreams; ++s) {
        const float4 q = *reinterpret_cast<const float4*>(
            w + static_cast<std::size_t>(s) * hidden + i);
        dot += q.x * v[s].x;
        dot += q.y * v[s].y;
        dot += q.z * v[s].z;
        dot += q.w * v[s].w;
      }
    }
    dot = WaveSum(dot);
    if (lane == 0) {
      partial[o][wave] = dot;
    }
  }
  __syncthreads();
  if (threadIdx.x < kStreams) {
    float total = 0.0F;
    for (std::uint32_t w = 0; w < blockDim.x / warpSize; ++w) {
      total += partial[threadIdx.x][w];
    }
    inject[(static_cast<std::size_t>(t) * kStreams + threadIdx.x) * gridDim.y +
           blockIdx.y] = total;
  }
}

/// One token per block keeps independent rows in flight. The FP32 mixed
/// row, optional F16/Q8 projection inputs and inject partials share one
/// read of the four residual streams. Grid: (inject parts, tokens).
template<bool kMix = true>
__global__ void HcMixEpilogueF16Kernel(const __half* xn, const float* gate,
                                       const float* inject_w, float* mixed,
                                       __half* mixed_half, void* mixed_q8,
                                       float* inject, std::uint32_t hidden) {
  constexpr std::uint32_t kStreams = 4;
  const std::uint32_t part = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t i = (part * blockDim.x + threadIdx.x) * 4;
  const std::size_t hc_dim = static_cast<std::size_t>(kStreams) * hidden;
  const bool live = i < hidden;
  float4 wq[kStreams][kStreams];
#pragma unroll
  for (std::uint32_t o = 0; o < kStreams; ++o) {
#pragma unroll
    for (std::uint32_t s = 0; s < kStreams; ++s) {
      wq[o][s] = live && inject_w != nullptr
                     ? *reinterpret_cast<const float4*>(
                           inject_w + (static_cast<std::size_t>(o) * hc_dim) +
                           (static_cast<std::size_t>(s) * hidden) + i)
                     : float4{0.0F, 0.0F, 0.0F, 0.0F};
    }
  }
  __shared__ float partial[kStreams][kThreads / 32];
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t wave = threadIdx.x / warpSize;
  constexpr float kInvStreams = 1.0F / static_cast<float>(kStreams);
  float4 v[kStreams];
  float4 acc{0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
  for (std::uint32_t s = 0; s < kStreams; ++s) {
    const std::size_t idx = (static_cast<std::size_t>(t) * hc_dim) +
                            (static_cast<std::size_t>(s) * hidden) + i;
    v[s] = live ? Load4(xn + idx) : float4{0.0F, 0.0F, 0.0F, 0.0F};
    if (kMix && live) {
      const float4 g = *reinterpret_cast<const float4*>(gate + idx);
      acc.x += v[s].x * SigmoidF(g.x);
      acc.y += v[s].y * SigmoidF(g.y);
      acc.z += v[s].z * SigmoidF(g.z);
      acc.w += v[s].w * SigmoidF(g.w);
    }
  }
  if (kMix && live) {
    acc.x *= kInvStreams;
    acc.y *= kInvStreams;
    acc.z *= kInvStreams;
    acc.w *= kInvStreams;
    *reinterpret_cast<float4*>(mixed + static_cast<std::size_t>(t) * hidden +
                               i) = acc;
    if (mixed_half != nullptr) {
      __half* out = mixed_half + (static_cast<std::size_t>(t) * hidden) + i;
      *reinterpret_cast<__half2*>(out) = __floats2half2_rn(acc.x, acc.y);
      *reinterpret_cast<__half2*>(out + 2) = __floats2half2_rn(acc.z, acc.w);
    }
  }
  if (kMix && mixed_q8 != nullptr) {
    // Eight lanes hold one 32-wide block (every lane joins the reduction).
    float max_abs = fmaxf(fmaxf(fabsf(acc.x), fabsf(acc.y)),
                          fmaxf(fabsf(acc.z), fabsf(acc.w)));
    for (int off = 4; off > 0; off >>= 1) {
      max_abs = fmaxf(max_abs, XorLane(max_abs, off));
    }
    if (live) {
      const float d = max_abs / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      const auto q0 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(acc.x * id))));
      const auto q1 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(acc.y * id))));
      const auto q2 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(acc.z * id))));
      const auto q3 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(acc.w * id))));
      const std::size_t kb = i / 32;
      const std::uint32_t pos = i % 32;  // 0, 4, ..., 28
      std::int8_t* tile =
          Q8ActTile(mixed_q8, hidden / 32, t / kQ8ActTileTokens, kb);
      const std::size_t tl = t % kQ8ActTileTokens;
      *reinterpret_cast<std::uint32_t*>(tile + ((pos >> 4u) * 256) + (tl * 16) +
                                        (pos & 15u)) =
          q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
      if (pos == 0) {
        *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                  (tl * sizeof(float))) = d;
      }
    }
  }
  if (inject_w == nullptr) {
    return;
  }
#pragma unroll
  for (std::uint32_t o = 0; o < kStreams; ++o) {
    // HcInjectPart's order, pinned: one product, then FMAs in stream and
    // component order. A plain `dot +=` chain is contracted and reassociated
    // differently per instantiation under fast-math, so a short prefill
    // pass (this epilogue) and a wide one (the combine's fused partials)
    // gave the same token different inject logits in the last bit.
    float dot = __fmul_rn(wq[o][0].x, v[0].x);
    dot = __fmaf_rn(wq[o][0].y, v[0].y, dot);
    dot = __fmaf_rn(wq[o][0].z, v[0].z, dot);
    dot = __fmaf_rn(wq[o][0].w, v[0].w, dot);
#pragma unroll
    for (std::uint32_t s = 1; s < kStreams; ++s) {
      dot = __fmaf_rn(wq[o][s].x, v[s].x, dot);
      dot = __fmaf_rn(wq[o][s].y, v[s].y, dot);
      dot = __fmaf_rn(wq[o][s].z, v[s].z, dot);
      dot = __fmaf_rn(wq[o][s].w, v[s].w, dot);
    }
    dot = WaveSum(dot);
    if (lane == 0) {
      partial[o][wave] = dot;
    }
  }

  __syncthreads();
  if (threadIdx.x < kStreams) {
    float total = 0.0F;
    for (std::uint32_t w = 0; w < kThreads / 32; ++w) {
      total = __fadd_rn(total, partial[threadIdx.x][w]);
    }
    inject[(static_cast<std::size_t>(t) * kStreams + threadIdx.x) * gridDim.x +
           part] = total;
  }
}

/// HcCombineKernel's tiled-Q8 store of element i of stream s: a wave holds
/// one 32-wide block of the row, quantized for the next mixer's W8A8 down
/// projection, K = streams * hidden.
__device__ __forceinline__ void QuantizeQ8TiledLane(
    float v, void* xn_q8, std::uint32_t t, std::uint32_t s, std::uint32_t i,
    std::uint32_t hidden, std::uint32_t streams) {
  float max_abs = fabsf(v);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, XorLane(max_abs, off));
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const auto q = static_cast<std::int8_t>(roundf(v * id));
  const std::size_t num_blocks =
      (static_cast<std::size_t>(streams) * hidden) / 32;
  const std::size_t kb = (static_cast<std::size_t>(s) * hidden + i) / 32;
  const std::size_t lane = threadIdx.x & 31u;
  std::int8_t* tile = Q8ActTile(xn_q8, num_blocks, t / kQ8ActTileTokens, kb);
  const std::size_t tl = t % kQ8ActTileTokens;
  tile[((lane >> 4u) * 256) + (tl * 16) + (lane & 15u)] = q;
  if (lane == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
}

/// Up to kMaxParts inject partials held in registers: every load issues
/// before the first add (indices clamped, so none waits behind a branch),
/// and Sum adds parts[0] + ... + parts[n - 1] in order from zero.
struct InjectParts {
  static constexpr std::uint32_t kMaxParts = 16;
  const float* parts;
  std::uint32_t n;
  float v[kMaxParts];

  __device__ __forceinline__ InjectParts(const float* p, std::uint32_t count)
      : parts(p), n(count) {
    if (n - 1 < kMaxParts) {
#pragma unroll
      for (std::uint32_t i = 0; i < kMaxParts; ++i) {
        v[i] = parts[i < n ? i : n - 1];
      }
    }
  }

  __device__ __forceinline__ float Sum() const {
    float total = Materialize(0.0f);
    if (n - 1 >= kMaxParts) {
      for (std::uint32_t i = 0; i < n; ++i) {
        total += parts[i];
      }
      return total;
    }
#pragma unroll
    for (std::uint32_t i = 0; i < kMaxParts; ++i) {
      if (i < n) {
        total = Materialize(total + v[i]);
      }
    }
    return total;
  }
};

/// grid (tokens, streams): one block owns one residual stream, so the
/// grouped norm of the next mixer reduces over exactly its own elements.
/// `XnT` is float for the reference route and __half for the F16 mixer
/// input route; a non-null `xn_q8` also receives the norm quantized into
/// the tiled Q8 layout (hidden % 32 == 0) for the W8A8 down projection.
template<typename XnT>
__global__ void HcCombineKernel(float* res, const float* block_out,
                                const float* inject, std::uint32_t inject_parts,
                                const float* gamma, XnT* xn, void* xn_q8,
                                std::uint32_t hidden, std::uint32_t streams,
                                float eps, void* xn_q8_1) {
  __shared__ float shared[32];
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t s = blockIdx.y;
  const float* parts =
      inject + (static_cast<std::size_t>(t) * streams + s) * inject_parts;
  const std::size_t base = (static_cast<std::size_t>(t) * streams + s) * hidden;
  float* dst = res + base;
  const float* src = block_out + static_cast<std::size_t>(t) * hidden;
  const float* g = gamma + static_cast<std::size_t>(s) * hidden;
  // A row of up to kHeld elements per thread stays in registers between the
  // two passes, every read issued up front (a strided loop of loads behind
  // stores to `res` would wait out each one): the inject partials first, as
  // the update needs their sum before any row element, then the row at
  // clamped indices (a load under `i < hidden` is a branch of its own, and
  // the compiler waits for it there). Lanes past `hidden` never use theirs.
  // Same arithmetic, same order.
  constexpr std::uint32_t kHeld = 10;
  if (hidden != 0 && hidden <= kHeld * blockDim.x) {
    const InjectParts inject_sum(parts, inject_parts);
    float r[kHeld];
    float b[kHeld];
    float gv[kHeld];
#pragma unroll
    for (std::uint32_t c = 0; c < kHeld; ++c) {
      const std::uint32_t i = threadIdx.x + c * blockDim.x;
      const std::uint32_t k = i < hidden ? i : hidden - 1;
      r[c] = dst[k];
      b[c] = src[k];
      gv[c] = gamma != nullptr ? g[k] : 0.0f;
    }
    const float w =
        2.0f * SigmoidF(inject_sum.Sum() / static_cast<float>(streams));
    float ss = 0.0f;
#pragma unroll
    for (std::uint32_t c = 0; c < kHeld; ++c) {
      const std::uint32_t i = threadIdx.x + c * blockDim.x;
      if (i < hidden) {
        r[c] = r[c] + b[c] * w;
        dst[i] = r[c];
        ss = Materialize(ss + r[c] * r[c]);
      }
    }
    if (gamma == nullptr) {
      return;
    }
    ss = BlockSum(ss, shared);
    const float scale = rsqrtf(ss / static_cast<float>(hidden) + eps);
#pragma unroll
    for (std::uint32_t c = 0; c < kHeld; ++c) {
      const std::uint32_t i = threadIdx.x + c * blockDim.x;
      if (i < hidden) {
        const float v = r[c] * scale * gv[c];
        xn[base + i] = static_cast<XnT>(v);
        if (xn_q8_1 != nullptr) {
          QuantizeQ8_1Lane(v, xn_q8_1, base + i);
        }
        if (xn_q8 != nullptr) {
          QuantizeQ8TiledLane(v, xn_q8, t, s, i, hidden, streams);
        }
      }
    }
    return;
  }
  const float w = 2.0f * SigmoidF(InjectParts(parts, inject_parts).Sum() /
                                  static_cast<float>(streams));
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = dst[i] + src[i] * w;
    dst[i] = v;
    ss += v * v;
  }
  if (gamma == nullptr) {
    return;
  }
  ss = BlockSum(ss, shared);
  const float scale = rsqrtf(ss / static_cast<float>(hidden) + eps);
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = dst[i] * scale * g[i];
    xn[base + i] = static_cast<XnT>(v);
    if (xn_q8_1 != nullptr) {
      // The small-batch mixer down projection's input, as mmq would
      // quantize xn (float rows only; hidden % 32 == 0).
      QuantizeQ8_1Lane(v, xn_q8_1, base + i);
    }
    if (xn_q8 != nullptr) {
      QuantizeQ8TiledLane(v, xn_q8, t, s, i, hidden, streams);
    }
  }
}

/// Writes inject partial `part` of token t (HcMixEpilogueF16Kernel<false>'s
/// grid (part, t)): `xh` holds this thread's four F16-rounded norm streams
/// at element (part * kThreads + threadIdx.x) * 4, zero past the row.
/// Reuses `shared` behind a barrier.
__device__ __forceinline__ void HcInjectPart(
    const float4 (&xh)[4], const float* inject_w, float* inject,
    float (*shared)[8], std::uint32_t t, std::uint32_t part,
    std::uint32_t parts, std::uint32_t hidden) {
  constexpr std::uint32_t kStreams = 4;
  const std::size_t hc_dim = static_cast<std::size_t>(kStreams) * hidden;
  const std::uint32_t i = (part * kThreads + threadIdx.x) * 4;
  const bool live = i < hidden;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t wave = threadIdx.x >> 5u;
  float dots[kStreams];
#pragma unroll
  for (std::uint32_t o = 0; o < kStreams; ++o) {
    float4 wq[kStreams];
#pragma unroll
    for (std::uint32_t s = 0; s < kStreams; ++s) {
      wq[s] = live ? *reinterpret_cast<const float4*>(
                         inject_w + (static_cast<std::size_t>(o) * hc_dim) +
                         (static_cast<std::size_t>(s) * hidden) + i)
                   : float4{0.0F, 0.0F, 0.0F, 0.0F};
    }
    // The separate kernel's sum from zero: one product, then FMAs in stream
    // and component order.
    float dot = __fmul_rn(wq[0].x, xh[0].x);
    dot = __fmaf_rn(wq[0].y, xh[0].y, dot);
    dot = __fmaf_rn(wq[0].z, xh[0].z, dot);
    dot = __fmaf_rn(wq[0].w, xh[0].w, dot);
#pragma unroll
    for (std::uint32_t s = 1; s < kStreams; ++s) {
      dot = __fmaf_rn(wq[s].x, xh[s].x, dot);
      dot = __fmaf_rn(wq[s].y, xh[s].y, dot);
      dot = __fmaf_rn(wq[s].z, xh[s].z, dot);
      dot = __fmaf_rn(wq[s].w, xh[s].w, dot);
    }
    dots[o] = WaveSum(dot);
  }
  __syncthreads();  // earlier readers of `shared`
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < kStreams; ++o)
      shared[o][wave] = dots[o];
  }
  __syncthreads();
  if (threadIdx.x < kStreams) {
    float total = 0.0F;
    for (std::uint32_t q = 0; q < kThreads / 32; ++q) {
      total = __fadd_rn(total, shared[threadIdx.x][q]);
    }
    inject[(static_cast<std::size_t>(t) * kStreams + threadIdx.x) * parts +
           part] = total;
  }
}

/// HcCombineKernel over one token's whole hyper-connection row (all streams,
/// float4 lanes): a block owns streams * hidden elements in up to
/// kMaxChunks float4 per thread, keeps the updated residual in registers
/// between the two passes, reduces the four stream norms in one block
/// reduction, and quantizes each 32-wide block over its eight lanes with a
/// four-code store. Rows of four streams up to 2,560 wide.
/// kMoe: the block output is the EXL3 prefill MoE output, formed per token
/// in LDS from the raw expert rows (Exl3PrefillMoeSegment) instead of read
/// from block_out.
template<typename XnT, bool kDefer = false, bool kInject = false,
         bool kMoe = false>
__global__ void HcCombineVec4Kernel(
    float* res, const float* block_out, const float* inject,
    std::uint32_t inject_parts, const float* gamma, XnT* xn, void* xn_q8,
    std::uint32_t hidden, float eps, float* defer_w,
    const float* inject_w = nullptr, float* inject_out = nullptr,
    __half* xn_half = nullptr, float* xn_scale = nullptr,
    Exl3PrefillMoeRows moe = {}, DeferredResidual prior = {}) {
  constexpr std::uint32_t kStreams = 4;
  constexpr std::uint32_t kMaxChunks = 10;  // 4 x 2560 / 4 / kThreads
  constexpr std::uint32_t kMaxHidden = kMaxChunks * kThreads;  // 2560
  __shared__ float shared[kStreams][kThreads / 32];
  __shared__ float s_moe[kMoe ? kMaxHidden : 1];
  // The F16 norm row, regathered for the next mixer's inject partials.
  __shared__ __half2 s_xn[kInject ? kMaxChunks * kThreads * 2 : 1];
  const std::uint32_t t = blockIdx.x;
  const std::size_t hc_dim = static_cast<std::size_t>(kStreams) * hidden;
  float w[kStreams];
#pragma unroll
  for (std::uint32_t s = 0; s < kStreams; ++s) {
    // The partials in order from zero, pinned like InjectParts::Sum: every
    // instantiation (deferring, inject-writing, MoE-fused) must give a token
    // the same stream weights whichever pass route computes it.
    float logit = 0.0f;
    for (std::uint32_t p = 0; p < inject_parts; ++p) {
      logit = __fadd_rn(
          logit,
          inject[(static_cast<std::size_t>(t) * kStreams + s) * inject_parts +
                 p]);
    }
    w[s] = 2.0f * SigmoidF(logit / static_cast<float>(kStreams));
  }
  // A deferred update leaves the residual as it was and keeps the stream
  // weights; the next combine reapplies it (HcCombineMoeF16Kernel).
  if constexpr (kDefer) {
    if (threadIdx.x < kStreams) {
      defer_w[static_cast<std::size_t>(t) * kStreams + threadIdx.x] =
          w[threadIdx.x];
    }
  }
  float* dst = res + static_cast<std::size_t>(t) * hc_dim;
  const float* src = block_out + static_cast<std::size_t>(t) * hidden;
  if constexpr (kMoe) {
    const std::uint32_t lane = threadIdx.x & 31u;
    for (std::uint32_t segment = threadIdx.x >> 5u; segment * 128 < hidden;
         segment += kThreads / 32) {
      float o[4];
      Exl3PrefillMoeSegment(moe, t, segment, lane, o);
#pragma unroll
      for (std::uint32_t j = 0; j < 4; ++j)
        s_moe[(segment * 128) + (lane * 4) + j] = o[j];
    }
    __syncthreads();
    src = s_moe;
  }
  const std::uint32_t chunks = static_cast<std::uint32_t>(hc_dim / 4);
  float4 v[kMaxChunks];
  float ss[kStreams] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
  for (std::uint32_t c = 0; c < kMaxChunks; ++c) {
    const std::uint32_t e = (c * kThreads + threadIdx.x) * 4;
    v[c] = float4{0.0F, 0.0F, 0.0F, 0.0F};
    if (e < hc_dim) {
      const std::uint32_t s = e / hidden;
      const std::uint32_t i = e - (s * hidden);
      const float ws = w[s];
      float4 r = *reinterpret_cast<const float4*>(dst + e);
      const float4 b = *reinterpret_cast<const float4*>(src + i);
      if constexpr (std::is_same_v<XnT, float>) {
        // The previous combine's deferred update first, then this one, each
        // as that combine would have stored it.
        if (prior.block_out != nullptr)
          r = CombineAdd(
              r,
              *reinterpret_cast<const float4*>(
                  prior.block_out + (static_cast<std::size_t>(t) * hidden) + i),
              prior.w[(static_cast<std::size_t>(t) * kStreams) + s]);
        v[c] = CombineAdd(r, b, ws);
      } else {
        v[c] = float4{r.x + b.x * ws, r.y + b.y * ws, r.z + b.z * ws,
                      r.w + b.w * ws};
      }
      if constexpr (!kDefer) {
        if (std::is_same_v<XnT, float> && gridDim.x >= 512) {
          // Wide FP32 prefill: a streaming store (one block per token). The
          // residual (168 MB at 4,096 tokens) is next read after the mixer's
          // two projections, long after it would have left the caches, and
          // dirty lines would otherwise be written back under those
          // projections' reads. Short chunks keep it cached.
          using Float4Bits = float __attribute__((ext_vector_type(4)));
          __builtin_nontemporal_store(
              Float4Bits{v[c].x, v[c].y, v[c].z, v[c].w},
              reinterpret_cast<Float4Bits*>(dst + e));
        } else {
          *reinterpret_cast<float4*>(dst + e) = v[c];
        }
      }
      const float sq =
          v[c].x * v[c].x + v[c].y * v[c].y + v[c].z * v[c].z + v[c].w * v[c].w;
#pragma unroll
      for (std::uint32_t k = 0; k < kStreams; ++k) {
        ss[k] += (k == s) ? sq : 0.0F;
      }
    }
  }
  if (gamma == nullptr) {
    return;
  }
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t wave = threadIdx.x >> 5u;
#pragma unroll
  for (std::uint32_t k = 0; k < kStreams; ++k) {
    ss[k] = WaveSum(ss[k]);
    if (lane == 0) {
      shared[k][wave] = ss[k];
    }
  }
  __syncthreads();
  float scale[kStreams];
#pragma unroll
  for (std::uint32_t k = 0; k < kStreams; ++k) {
    float total = 0.0F;
    for (std::uint32_t q = 0; q < kThreads / 32; ++q) {
      total += shared[k][q];
    }
    scale[k] = rsqrtf(total / static_cast<float>(hidden) + eps);
  }
  if (xn_scale != nullptr) {
#pragma unroll
    for (std::uint32_t k = 0; k < kStreams; ++k) {
      if (threadIdx.x == k)
        xn_scale[(static_cast<std::size_t>(t) * kStreams) + k] = scale[k];
    }
  }
  const std::size_t num_blocks = hc_dim / 32;
#pragma unroll
  for (std::uint32_t c = 0; c < kMaxChunks; ++c) {
    const std::uint32_t e = (c * kThreads + threadIdx.x) * 4;
    // Every lane of the wave takes part in the block reductions below, so
    // lanes past the row carry zeros instead of leaving.
    float4 n{0.0F, 0.0F, 0.0F, 0.0F};
    if (e < hc_dim) {
      const std::uint32_t s = e / hidden;
      const float sc = scale[s];
      const float4 gm = *reinterpret_cast<const float4*>(gamma + e);
      // The F32 route's consumers may recompute this norm (HcNorm4); the F16
      // route keeps its original expression.
      if constexpr (std::is_same_v<XnT, float>) {
        n = HcNorm4(v[c], sc, gm);
      } else {
        n = float4{v[c].x * sc * gm.x, v[c].y * sc * gm.y, v[c].z * sc * gm.z,
                   v[c].w * sc * gm.w};
      }
      XnT* out = xn + (static_cast<std::size_t>(t) * hc_dim) + e;
      if constexpr (std::is_same_v<XnT, float>) {
        if (xn_scale == nullptr)
          *reinterpret_cast<float4*>(out) = n;
        // The F16 copy a projection would otherwise narrow from `out`
        // (NarrowHalf4Kernel's conversion).
        // Streaming stores: the projection that reads these rows next
        // otherwise finds them dirty in the caches.
        if (xn_half != nullptr) {
          __half* h = xn_half + (static_cast<std::size_t>(t) * hc_dim) + e;
          using Half4Bits = std::uint16_t __attribute__((ext_vector_type(4)));
          const Half4Bits bits{
              __half_as_ushort(__half(n.x)), __half_as_ushort(__half(n.y)),
              __half_as_ushort(__half(n.z)), __half_as_ushort(__half(n.w))};
          __builtin_nontemporal_store(bits, reinterpret_cast<Half4Bits*>(h));
        }
      } else {
        const __half2 lo = __floats2half2_rn(n.x, n.y);
        const __half2 hi = __floats2half2_rn(n.z, n.w);
        *reinterpret_cast<__half2*>(out) = lo;
        *reinterpret_cast<__half2*>(out + 2) = hi;
        if constexpr (kInject) {
          s_xn[e / 2] = lo;
          s_xn[e / 2 + 1] = hi;
        }
      }
    }
    if (xn_q8 == nullptr) {
      continue;
    }
    // Eight lanes hold one 32-wide block: absmax over them, then each lane
    // stores its four codes as one word.
    float max_abs =
        fmaxf(fmaxf(fabsf(n.x), fabsf(n.y)), fmaxf(fabsf(n.z), fabsf(n.w)));
    for (int off = 4; off > 0; off >>= 1) {
      max_abs = fmaxf(max_abs, XorLane(max_abs, off));
    }
    if (e < hc_dim) {
      const float d = max_abs / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      const auto q0 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(n.x * id))));
      const auto q1 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(n.y * id))));
      const auto q2 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(n.z * id))));
      const auto q3 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(n.w * id))));
      const std::size_t kb = e / 32;
      const std::uint32_t pos = e % 32;  // 0, 4, ..., 28
      std::int8_t* tile =
          Q8ActTile(xn_q8, num_blocks, t / kQ8ActTileTokens, kb);
      const std::size_t tl = t % kQ8ActTileTokens;
      *reinterpret_cast<std::uint32_t*>(tile + ((pos >> 4u) * 256) + (tl * 16) +
                                        (pos & 15u)) =
          q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
      if (pos == 0) {
        *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                  (tl * sizeof(float))) = d;
      }
    }
  }
  if constexpr (kInject) {
    const std::uint32_t parts = (hidden + kThreads * 4 - 1) / (kThreads * 4);
    __syncthreads();
    for (std::uint32_t c = 0; c < parts; ++c) {
      const std::uint32_t i = (c * kThreads + threadIdx.x) * 4;
      float4 part[kStreams];
#pragma unroll
      for (std::uint32_t s = 0; s < kStreams; ++s) {
        part[s] = float4{0.0F, 0.0F, 0.0F, 0.0F};
        if (i < hidden) {
          const std::size_t e = static_cast<std::size_t>(s) * hidden + i;
          const __half2 lo = s_xn[e / 2];
          const __half2 hi = s_xn[e / 2 + 1];
          part[s] = float4{__low2float(lo), __high2float(lo), __low2float(hi),
                           __high2float(hi)};
        }
      }
      HcInjectPart(part, inject_w, inject_out, shared, t, c, parts, hidden);
    }
  }
}

/// HcCombineVec4Kernel<__half> with the MoE epilogue fused in: the block
/// output row is formed in registers from the routed experts' F16 rows,
/// their weights and the gated shared expert (MoeEpilogueVec4Kernel), so the
/// block output never round-trips memory. Threads own hidden lanes (up to
/// three float4 each) across the four streams.
template<bool kNormalize, bool kPrior = false, bool kInject = false>
__global__ void HcCombineMoeF16Kernel(
    float* res, const __half* expert_out, const float* weights,
    const float* shared_out, const float* gate, std::uint32_t gate_stride,
    std::uint32_t used, const float* inject, std::uint32_t inject_parts,
    const float* gamma, __half* xn, void* xn_q8, std::uint32_t hidden,
    float eps, const float* prior_out, const float* prior_w,
    const float* inject_w, float* inject_out) {
  constexpr std::uint32_t kStreams = 4;
  constexpr std::uint32_t kMaxChunks = 3;  // 2560 / 4 / kThreads, rounded up
  __shared__ float shared[kStreams][kThreads / 32];
  const std::uint32_t t = blockIdx.x;
  const std::size_t hc_dim = static_cast<std::size_t>(kStreams) * hidden;
  float w[kStreams];
#pragma unroll
  for (std::uint32_t s = 0; s < kStreams; ++s) {
    // The partials in order from zero, pinned like InjectParts::Sum: every
    // instantiation (deferring, inject-writing, MoE-fused) must give a token
    // the same stream weights whichever pass route computes it.
    float logit = 0.0f;
    for (std::uint32_t p = 0; p < inject_parts; ++p) {
      logit = __fadd_rn(
          logit,
          inject[(static_cast<std::size_t>(t) * kStreams + s) * inject_parts +
                 p]);
    }
    w[s] = 2.0f * SigmoidF(logit / static_cast<float>(kStreams));
  }
  float* dst = res + static_cast<std::size_t>(t) * hc_dim;
  // The block output: sum of the weighted expert rows plus the gated shared
  // expert, per hidden chunk.
  const float g = SigmoidF(gate[static_cast<std::size_t>(t) * gate_stride]);
  const __half* rows = expert_out + static_cast<std::size_t>(t) * used * hidden;
  float4 v[kStreams][kMaxChunks];
  float ss[kStreams] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
  for (std::uint32_t c = 0; c < kMaxChunks; ++c) {
    const std::uint32_t i = (c * kThreads + threadIdx.x) * 4;
    float4 prior{0.0F, 0.0F, 0.0F, 0.0F};
    if constexpr (kPrior) {
      if (i < hidden)
        prior = *reinterpret_cast<const float4*>(
            prior_out + static_cast<std::size_t>(t) * hidden + i);
    }
    float4 b{0.0F, 0.0F, 0.0F, 0.0F};
    if (i < hidden) {
      for (std::uint32_t k = 0; k < used; ++k) {
        const float wk = weights[t * used + k];
        const float4 e = Load4(rows + static_cast<std::size_t>(k) * hidden + i);
        b.x += wk * e.x;
        b.y += wk * e.y;
        b.z += wk * e.z;
        b.w += wk * e.w;
      }
      const float4 sh = *reinterpret_cast<const float4*>(
          shared_out + static_cast<std::size_t>(t) * hidden + i);
      b.x += g * sh.x;
      b.y += g * sh.y;
      b.z += g * sh.z;
      b.w += g * sh.w;
    }
#pragma unroll
    for (std::uint32_t s = 0; s < kStreams; ++s) {
      v[s][c] = float4{0.0F, 0.0F, 0.0F, 0.0F};
      if (i < hidden) {
        const std::size_t e = (static_cast<std::size_t>(s) * hidden) + i;
        float4 r = *reinterpret_cast<const float4*>(dst + e);
        if constexpr (kPrior) {
          const float pw = prior_w[static_cast<std::size_t>(t) * kStreams + s];
          r = float4{__fmaf_rn(prior.x, pw, r.x), __fmaf_rn(prior.y, pw, r.y),
                     __fmaf_rn(prior.z, pw, r.z), __fmaf_rn(prior.w, pw, r.w)};
          // Retain the previous combine's F32 rounding boundary without
          // storing and reloading the intermediate residual.
          asm volatile("" : "+v"(r.x), "+v"(r.y), "+v"(r.z), "+v"(r.w));
        }
        v[s][c] = float4{r.x + b.x * w[s], r.y + b.y * w[s], r.z + b.z * w[s],
                         r.w + b.w * w[s]};
        *reinterpret_cast<float4*>(dst + e) = v[s][c];
        {
          // One square-accumulation order for every instantiation. The
          // deferred path must stay bit-identical to the separate-store
          // path, and fast-math reorders a plain `ss += x*x + ...`
          // differently per template instantiation (gfx1151 stream 3), so
          // the two would start from sums differing in the last F32 bits
          // and disagree in the Q8 block scales. The pinned order is the
          // compiled order of the separate-store build this defers.
          const float4 n = v[s][c];
          if (s == 3) {
            ss[s] = __fmaf_rn(n.w, n.w, ss[s]);
            ss[s] = __fmaf_rn(n.y, n.y, ss[s]);
            ss[s] = __fmaf_rn(n.z, n.z, ss[s]);
            ss[s] = __fmaf_rn(n.x, n.x, ss[s]);
          } else {
            if (s == 0 && c == 0) {
              ss[s] = __fmul_rn(n.y, n.y);
              ss[s] = __fmaf_rn(n.x, n.x, ss[s]);
            } else {
              ss[s] = __fmaf_rn(n.x, n.x, ss[s]);
              ss[s] = __fmaf_rn(n.y, n.y, ss[s]);
            }
            ss[s] = __fmaf_rn(n.z, n.z, ss[s]);
            ss[s] = __fmaf_rn(n.w, n.w, ss[s]);
          }
        }
      }
    }
  }
  if constexpr (!kNormalize)
    return;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t wave = threadIdx.x >> 5u;
#pragma unroll
  for (std::uint32_t k = 0; k < kStreams; ++k) {
    ss[k] = WaveSum(ss[k]);
    if (lane == 0) {
      shared[k][wave] = ss[k];
    }
  }
  __syncthreads();
  float scale[kStreams];
#pragma unroll
  for (std::uint32_t k = 0; k < kStreams; ++k) {
    float total = 0.0F;
    for (std::uint32_t q = 0; q < kThreads / 32; ++q) {
      total += shared[k][q];
    }
    scale[k] = rsqrtf(total / static_cast<float>(hidden) + eps);
  }
  const std::size_t num_blocks = hc_dim / 32;
  const std::size_t tl = t % kQ8ActTileTokens;
  // The next mixer's inject partials (HcMixEpilogueF16Kernel<false>): its
  // part c gives thread x the four streams of the elements this thread
  // holds, so the F16 norm is read back from registers.
  float4 xh[kInject ? kStreams : 1][kInject ? kMaxChunks : 1] = {};
#pragma unroll
  for (std::uint32_t s = 0; s < kStreams; ++s) {
#pragma unroll
    for (std::uint32_t c = 0; c < kMaxChunks; ++c) {
      const std::uint32_t i = (c * kThreads + threadIdx.x) * 4;
      const std::size_t e = (static_cast<std::size_t>(s) * hidden) + i;
      // Every lane of the wave takes part in the block reductions below,
      // so lanes past the row carry zeros instead of leaving.
      float4 n{0.0F, 0.0F, 0.0F, 0.0F};
      if (i < hidden) {
        const float4 gm = *reinterpret_cast<const float4*>(gamma + e);
        const float4 g =
            float4{__fmul_rn(gm.x, scale[s]), __fmul_rn(gm.y, scale[s]),
                   __fmul_rn(gm.z, scale[s]), __fmul_rn(gm.w, scale[s])};
        n = float4{__fmul_rn(v[s][c].x, g.x), __fmul_rn(v[s][c].y, g.y),
                   __fmul_rn(v[s][c].z, g.z), __fmul_rn(v[s][c].w, g.w)};
        __half* out = xn + (static_cast<std::size_t>(t) * hc_dim) + e;
        const __half2 lo = __floats2half2_rn(n.x, n.y);
        const __half2 hi = __floats2half2_rn(n.z, n.w);
        *reinterpret_cast<__half2*>(out) = lo;
        *reinterpret_cast<__half2*>(out + 2) = hi;
        if constexpr (kInject) {
          xh[s][c] = float4{__low2float(lo), __high2float(lo), __low2float(hi),
                            __high2float(hi)};
        }
      }
      // Eight lanes hold one 32-wide block: absmax over them, then each
      // lane stores its four codes as one word.
      float max_abs =
          fmaxf(fmaxf(fabsf(n.x), fabsf(n.y)), fmaxf(fabsf(n.z), fabsf(n.w)));
      for (int off = 4; off > 0; off >>= 1) {
        max_abs = fmaxf(max_abs, XorLane(max_abs, off));
      }
      if (i < hidden) {
        const float d = max_abs / 127.0F;
        const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
        const auto q0 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
            static_cast<std::int8_t>(roundf(n.x * id))));
        const auto q1 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
            static_cast<std::int8_t>(roundf(n.y * id))));
        const auto q2 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
            static_cast<std::int8_t>(roundf(n.z * id))));
        const auto q3 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
            static_cast<std::int8_t>(roundf(n.w * id))));
        const std::size_t kb = e / 32;
        const std::uint32_t pos = e % 32;  // 0, 4, ..., 28
        std::int8_t* tile =
            Q8ActTile(xn_q8, num_blocks, t / kQ8ActTileTokens, kb);
        *reinterpret_cast<std::uint32_t*>(tile + ((pos >> 4u) * 256) +
                                          (tl * 16) + (pos & 15u)) =
            q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        if (pos == 0) {
          *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                    (tl * sizeof(float))) = d;
        }
      }
    }
  }
  if constexpr (kInject) {
    const std::uint32_t parts = (hidden + kThreads * 4 - 1) / (kThreads * 4);
#pragma unroll
    for (std::uint32_t c = 0; c < kMaxChunks; ++c) {
      if (c >= parts)
        break;
      const float4 part[kStreams] = {xh[0][c], xh[1][c], xh[2][c], xh[3][c]};
      HcInjectPart(part, inject_w, inject_out, shared, t, c, parts, hidden);
    }
  }
}

}  // namespace

std::uint32_t HcInjectParts(std::uint32_t hidden) {
  return Blocks(hidden);
}

std::uint32_t HcInjectPartsVec4(std::uint32_t hidden) {
  return Blocks((hidden + 3) / 4);
}

bool HcMixEpilogue(const float* xn, const float* gate, const float* inject_w,
                   float* mixed, float* inject, std::uint32_t n_tokens,
                   std::uint32_t hidden, std::uint32_t streams,
                   hipStream_t stream, void* mixed_q8,
                   const Exl3InputSlabs& slabs) {
  if (hidden % 32 != 0) {
    mixed_q8 = nullptr;
  }
  const bool rotate = slabs.count != 0 && streams == 4 && hidden % 256 == 0 &&
                      slabs.k == hidden && n_tokens <= slabs.pass_rows;
  const Exl3InputSlabs launch_slabs = rotate ? slabs : Exl3InputSlabs{};
  if (streams == 4) {
    hipLaunchKernelGGL(HcMixEpilogueKernel<4>, dim3(n_tokens, Blocks(hidden)),
                       dim3(kThreads), 0, stream, xn, gate, inject_w, mixed,
                       inject, hidden, streams, mixed_q8, launch_slabs);
    return rotate;
  }
  hipLaunchKernelGGL(HcMixEpilogueKernel<0>, dim3(n_tokens, Blocks(hidden)),
                     dim3(kThreads), 0, stream, xn, gate, inject_w, mixed,
                     inject, hidden, streams, mixed_q8, launch_slabs);
  return false;
}

bool HcMixEpilogueVec4(const float* xn, const float* gate,
                       const float* inject_w, float* mixed, float* inject,
                       std::uint32_t n_tokens, std::uint32_t hidden,
                       std::uint32_t streams, hipStream_t stream,
                       __half* mixed_half, const __half* gate_half) {
  if (gate_half == nullptr && (streams != 4 || hidden % 4 != 0)) {
    HcMixEpilogue(xn, gate, inject_w, mixed, inject, n_tokens, hidden, streams,
                  stream);
    return false;
  }
  const dim3 grid(n_tokens, HcInjectPartsVec4(hidden));
  if (gate_half != nullptr)
    hipLaunchKernelGGL((HcMixEpilogueVec4Kernel<float, false, __half>), grid,
                       dim3(kThreads), 0, stream, xn, gate_half, inject_w,
                       mixed, inject, hidden, nullptr, nullptr, mixed_half);
  else
    hipLaunchKernelGGL(HcMixEpilogueVec4Kernel<float>, grid, dim3(kThreads), 0,
                       stream, xn, gate, inject_w, mixed, inject, hidden,
                       nullptr, nullptr, mixed_half);
  return mixed_half != nullptr;
}

void HcMixEpilogueVec4FromRes(const float* res, const float* gamma,
                              const float* xn_scale, const float* gate,
                              const float* inject_w, float* mixed,
                              float* inject, std::uint32_t n_tokens,
                              std::uint32_t hidden, hipStream_t stream,
                              __half* mixed_half, const __half* gate_half,
                              DeferredResidual prior) {
  const dim3 grid(n_tokens, HcInjectPartsVec4(hidden));
  if (gate_half != nullptr)
    hipLaunchKernelGGL((HcMixEpilogueVec4Kernel<float, true, __half>), grid,
                       dim3(kThreads), 0, stream, res, gate_half, inject_w,
                       mixed, inject, hidden, gamma, xn_scale, mixed_half,
                       prior);
  else
    hipLaunchKernelGGL((HcMixEpilogueVec4Kernel<float, true>), grid,
                       dim3(kThreads), 0, stream, res, gate, inject_w, mixed,
                       inject, hidden, gamma, xn_scale, mixed_half, prior);
}

__global__ void HcNormFromResKernel(const float* res, const float* gamma,
                                    const float* xn_scale, float* xn,
                                    std::uint32_t hidden,
                                    DeferredResidual prior) {
  constexpr std::uint32_t kStreams = 4;
  const std::uint32_t t = blockIdx.x;
  const std::size_t hc_dim = static_cast<std::size_t>(kStreams) * hidden;
  const std::size_t row = static_cast<std::size_t>(t) * hc_dim;
  for (std::uint32_t e = threadIdx.x * 4; e < hc_dim; e += kThreads * 4) {
    const std::uint32_t s = e / hidden;
    float4 r = *reinterpret_cast<const float4*>(res + row + e);
    if (prior.block_out != nullptr)
      r = CombineAdd(
          r,
          *reinterpret_cast<const float4*>(
              prior.block_out + (static_cast<std::size_t>(t) * hidden) +
              (e - s * hidden)),
          prior.w[(static_cast<std::size_t>(t) * kStreams) + s]);
    *reinterpret_cast<float4*>(xn + row + e) =
        HcNorm4(r, xn_scale[(static_cast<std::size_t>(t) * kStreams) + s],
                *reinterpret_cast<const float4*>(gamma + e));
  }
}

void HcNormFromRes(const float* res, const float* gamma, const float* xn_scale,
                   float* xn, std::uint32_t n_tokens, std::uint32_t hidden,
                   hipStream_t stream, DeferredResidual prior) {
  hipLaunchKernelGGL(HcNormFromResKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, res, gamma, xn_scale, xn, hidden, prior);
}

void HcMixInjectF16(const __half* xn, const float* inject_w, float* inject,
                    std::uint32_t n_tokens, std::uint32_t hidden,
                    hipStream_t stream) {
  const dim3 inject_grid(HcInjectPartsVec4(hidden), n_tokens);
  hipLaunchKernelGGL((HcMixEpilogueF16Kernel<false>), inject_grid,
                     dim3(kThreads), 0, stream, xn, nullptr, inject_w, nullptr,
                     nullptr, nullptr, inject, hidden);
}

void HcMixEpilogueVec4F16(const __half* xn, const float* gate,
                          const float* inject_w, float* mixed,
                          __half* mixed_half, void* mixed_q8, float* inject,
                          std::uint32_t n_tokens, std::uint32_t hidden,
                          hipStream_t stream) {
  if (hidden % 32 == 0) {
    hipLaunchKernelGGL((HcMixEpilogueF16Kernel<>),
                       dim3(HcInjectPartsVec4(hidden), n_tokens),
                       dim3(kThreads), 0, stream, xn, gate, inject_w, mixed,
                       mixed_half, mixed_q8, inject, hidden);
    return;
  }
  hipLaunchKernelGGL(HcMixEpilogueVec4Kernel<__half>,
                     dim3(n_tokens, HcInjectPartsVec4(hidden)), dim3(kThreads),
                     0, stream, xn, gate, inject_w, mixed, inject, hidden);
  if (mixed_half != nullptr) {
    NarrowActivations(mixed, mixed_half, false,
                      static_cast<std::size_t>(n_tokens) * hidden, stream);
  }
  if (mixed_q8 != nullptr) {
    QuantizeQ8Tiled(mixed, mixed_q8, n_tokens, hidden, stream);
  }
}

bool HcCombine(float* res, const float* block_out, const float* inject,
               std::uint32_t inject_parts, const float* gamma, float* xn,
               std::uint32_t n_tokens, std::uint32_t hidden,
               std::uint32_t streams, float eps, hipStream_t stream,
               bool decode, bool any_width, __half* xn_half, float* xn_scale,
               const Exl3PrefillMoeRows* moe, void* xn_q8t, float* defer_w,
               DeferredResidual prior) {
  // One block per token wants a batch: a decode step keeps the
  // one-block-per-stream kernel's parallelism.
  if (!decode && (n_tokens >= 16 || any_width) && streams == 4 &&
      hidden % 128 == 0 && hidden <= 2560) {
    xn_half = gamma != nullptr ? xn_half : nullptr;
    xn_q8t = gamma != nullptr ? xn_q8t : nullptr;
    const bool side = xn_half != nullptr || xn_q8t != nullptr;
    xn_scale = side ? xn_scale : nullptr;
    // Deferral needs the recomputing consumers: a deferred norm (xn_scale).
    if (defer_w != nullptr && (xn_scale == nullptr || moe != nullptr))
      throw std::invalid_argument(
          "HC combine: a deferred update needs a deferred norm and no MoE");
    if (moe != nullptr) {
      hipLaunchKernelGGL((HcCombineVec4Kernel<float, false, false, true>),
                         dim3(n_tokens), dim3(kThreads), 0, stream, res,
                         block_out, inject, inject_parts, gamma, xn, xn_q8t,
                         hidden, eps, nullptr, nullptr, nullptr, xn_half,
                         xn_scale, *moe, prior);
    } else if (defer_w != nullptr) {
      hipLaunchKernelGGL((HcCombineVec4Kernel<float, true>), dim3(n_tokens),
                         dim3(kThreads), 0, stream, res, block_out, inject,
                         inject_parts, gamma, xn, xn_q8t, hidden, eps, defer_w,
                         nullptr, nullptr, xn_half, xn_scale,
                         Exl3PrefillMoeRows{}, prior);
    } else {
      hipLaunchKernelGGL(HcCombineVec4Kernel<float>, dim3(n_tokens),
                         dim3(kThreads), 0, stream, res, block_out, inject,
                         inject_parts, gamma, xn, xn_q8t, hidden, eps, nullptr,
                         nullptr, nullptr, xn_half, xn_scale,
                         Exl3PrefillMoeRows{}, prior);
    }
    return side;
  }
  if (moe != nullptr || defer_w != nullptr || prior.block_out != nullptr)
    throw std::invalid_argument(
        "HC combine: fused MoE and deferred updates need the four-lane "
        "prefill kernel");
  hipLaunchKernelGGL(HcCombineKernel<float>, dim3(n_tokens, streams),
                     dim3(kThreads), 0, stream, res, block_out, inject,
                     inject_parts, gamma, xn, nullptr, hidden, streams, eps,
                     nullptr);
  return false;
}

bool HcCombineQ8_1(float* res, const float* block_out, const float* inject,
                   std::uint32_t inject_parts, const float* gamma, float* xn,
                   void* xn_q8, std::uint32_t n_tokens, std::uint32_t hidden,
                   std::uint32_t streams, float eps, hipStream_t stream) {
  if (gamma == nullptr || hidden % 32 != 0) {
    return false;
  }
  hipLaunchKernelGGL(HcCombineKernel<float>, dim3(n_tokens, streams),
                     dim3(kThreads), 0, stream, res, block_out, inject,
                     inject_parts, gamma, xn, nullptr, hidden, streams, eps,
                     xn_q8);
  return true;
}

bool HcCombineMoeF16(float* res, const __half* expert_out, const float* weights,
                     const float* shared_out, const float* gate,
                     std::uint32_t gate_stride, std::uint32_t used,
                     const float* inject, std::uint32_t inject_parts,
                     const float* gamma, __half* xn, void* xn_q8,
                     std::uint32_t n_tokens, std::uint32_t hidden,
                     std::uint32_t streams, float eps, hipStream_t stream,
                     const float* prior_out, const float* prior_w,
                     float* inject_out, const float* inject_w) {
  if (streams != 4 || hidden % 128 != 0 || hidden > 3 * 4 * kThreads ||
      (gamma != nullptr && xn_q8 == nullptr) ||
      (inject_w != nullptr && (gamma == nullptr || prior_out == nullptr))) {
    return false;
  }
  const auto combine =
      inject_w != nullptr    ? HcCombineMoeF16Kernel<true, true, true>
      : prior_out != nullptr ? (gamma ? HcCombineMoeF16Kernel<true, true>
                                      : HcCombineMoeF16Kernel<false, true>)
                             : (gamma ? HcCombineMoeF16Kernel<true>
                                      : HcCombineMoeF16Kernel<false>);
  hipLaunchKernelGGL(combine, dim3(n_tokens), dim3(kThreads), 0, stream, res,
                     expert_out, weights, shared_out, gate, gate_stride, used,
                     inject, inject_parts, gamma, xn, xn_q8, hidden, eps,
                     prior_out, prior_w, inject_w, inject_out);
  return true;
}

/// res[t][s][i] += block_out[t][i] * w[t][s], the update a deferring
/// HcCombineVec4Kernel left out, in the same expression.
__global__ void HcApplyDeferredKernel(float* res, const float* block_out,
                                      const float* w, std::uint32_t hidden) {
  constexpr std::uint32_t kStreams = 4;
  const std::uint32_t t = blockIdx.x;
  const std::size_t hc_dim = static_cast<std::size_t>(kStreams) * hidden;
  float* dst = res + static_cast<std::size_t>(t) * hc_dim;
  const float* src = block_out + static_cast<std::size_t>(t) * hidden;
  for (std::uint32_t e = threadIdx.x * 4; e < hc_dim; e += kThreads * 4) {
    const std::uint32_t s = e / hidden;
    const std::uint32_t i = e - (s * hidden);
    const float ws = w[static_cast<std::size_t>(t) * kStreams + s];
    const float4 r = *reinterpret_cast<const float4*>(dst + e);
    const float4 b = *reinterpret_cast<const float4*>(src + i);
    *reinterpret_cast<float4*>(dst + e) =
        float4{__fmaf_rn(b.x, ws, r.x), __fmaf_rn(b.y, ws, r.y),
               __fmaf_rn(b.z, ws, r.z), __fmaf_rn(b.w, ws, r.w)};
  }
}

void HcApplyDeferred(float* res, const float* block_out, const float* w,
                     std::uint32_t n_tokens, std::uint32_t hidden,
                     hipStream_t stream) {
  hipLaunchKernelGGL(HcApplyDeferredKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, res, block_out, w, hidden);
}

bool HcCombineF16(float* res, const float* block_out, const float* inject,
                  std::uint32_t inject_parts, const float* gamma, __half* xn,
                  void* xn_q8, std::uint32_t n_tokens, std::uint32_t hidden,
                  std::uint32_t streams, float eps, hipStream_t stream,
                  float* defer_w, float* inject_out, const float* inject_w) {
  // F16 prefill keeps the same reduction even for a one-token tail.
  if (streams == 4 && hidden % 128 == 0 && hidden <= 2560) {
    if (inject_w != nullptr && defer_w != nullptr && gamma != nullptr) {
      // The inject partials may replace the logits this combine reads;
      // every thread reads them before its first barrier.
      hipLaunchKernelGGL((HcCombineVec4Kernel<__half, true, true>),
                         dim3(n_tokens), dim3(kThreads), 0, stream, res,
                         block_out, inject, inject_parts, gamma, xn, xn_q8,
                         hidden, eps, defer_w, inject_w, inject_out);
      return true;
    }
    const auto combine = defer_w != nullptr ? HcCombineVec4Kernel<__half, true>
                                            : HcCombineVec4Kernel<__half>;
    hipLaunchKernelGGL(combine, dim3(n_tokens), dim3(kThreads), 0, stream, res,
                       block_out, inject, inject_parts, gamma, xn, xn_q8,
                       hidden, eps, defer_w, nullptr, nullptr, nullptr, nullptr,
                       Exl3PrefillMoeRows{}, DeferredResidual{});
    return defer_w != nullptr;
  }
  hipLaunchKernelGGL(HcCombineKernel<__half>, dim3(n_tokens, streams),
                     dim3(kThreads), 0, stream, res, block_out, inject,
                     inject_parts, gamma, xn, xn_q8, hidden, streams, eps,
                     nullptr);
  return false;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
