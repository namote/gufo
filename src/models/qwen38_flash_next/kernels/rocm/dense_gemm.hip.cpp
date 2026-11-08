// The dense F16/BF16 WMMA GEMM (DenseF16GEMMKernel) and its fused
// epilogues: stacked QKV with head norm and RoPE, HC up and mix, the SSM
// projection with its convolution, EXL3 plain prefill, plus the EXL3 dense
// prefill problem setup.
// HIP kernels follow the layouts and operator formulas in reference.cpp.

#include "src/models/qwen38_flash_next/kernels/rocm/kernels_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

struct Exl3DensePrefillArguments {
  Exl3DensePrefillProblem problems[3];
  unsigned count;
};
__global__ void PrepareExl3DensePrefillKernel(
    Exl3DensePrefillArguments args, gufo::hip::Exl3RotateProblem* rotations,
    gufo::hip::Exl3GemmProblem* gemms) {
  const auto i = threadIdx.x;
  if (i >= args.count)
    return;
  const auto p = args.problems[i];
  rotations[i] = {p.x, p.x_stride, nullptr, p.suh, p.rotated, p.rows, 0};
  gemms[i] = {p.trellis, p.svh, p.rotated, p.y, p.rows, p.trellis_rows};
}

}  // namespace

// Keep the separate four-tap convolution's F32 rounding order when its
// inputs come from the projection's LDS tile.
__device__ __forceinline__ float SsmConv4Value(float4 w, float x0, float x1,
                                               float x2, float x3) {
  float acc = __fmaf_rn(w.x, x0, __fmul_rn(w.y, x1));
  acc = __fmaf_rn(w.z, x2, acc);
  acc = __fmaf_rn(w.w, x3, acc);
  return SiluF(acc);
}

constexpr unsigned kSsmProjectionTileTokens = 32;

// The fused projection leaves each 32-token tile's first and last three
// raw rows in qkv. Only the first three convolutions need another tile or
// the previous chunk's history; all other rows are produced in LDS. A
// thread takes four channels of a tile's three boundary tokens.
__global__ void SsmConvBoundaryKernel(const float* qkv, const float* w,
                                      const float* history, float* out,
                                      std::uint32_t n_tokens,
                                      std::uint32_t channels,
                                      std::uint32_t stride) {
  const std::uint32_t c = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const std::uint32_t t0 = blockIdx.y * kSsmProjectionTileTokens;
  if (c >= channels) {
    return;
  }
  float4 taps[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    taps[j] = *reinterpret_cast<const float4*>(w + (c + j) * 4);
  // Input rows t0 - 3 .. t0 + 2.
  float4 v[6];
#pragma unroll
  for (int j = 0; j < 6; ++j) {
    const int src = static_cast<int>(t0) - 3 + j;
    v[j] = float4{0.0F, 0.0F, 0.0F, 0.0F};
    if (src < 0)
      v[j] = *reinterpret_cast<const float4*>(
          history + static_cast<std::size_t>(src + 3) * channels + c);
    else if (src < static_cast<int>(n_tokens))
      v[j] = *reinterpret_cast<const float4*>(
          qkv + static_cast<std::size_t>(src) * stride + c);
  }
#pragma unroll
  for (int z = 0; z < 3; ++z) {
    const std::uint32_t t = t0 + z;
    if (t >= n_tokens)
      break;
    *reinterpret_cast<float4*>(out + static_cast<std::size_t>(t) * channels +
                               c) = float4{
        SsmConv4Value(taps[0], v[z].x, v[z + 1].x, v[z + 2].x, v[z + 3].x),
        SsmConv4Value(taps[1], v[z].y, v[z + 1].y, v[z + 2].y, v[z + 3].y),
        SsmConv4Value(taps[2], v[z].z, v[z + 1].z, v[z + 2].z, v[z + 3].z),
        SsmConv4Value(taps[3], v[z].w, v[z + 1].w, v[z + 2].w, v[z + 3].w)};
  }
}

// Optional output layout for the stacked QKV projection. The projection's
// 256-row tile covers one query, gate, key or value head.
struct AttentionProjectionOutput {
  const float* q_gamma;
  const float* k_gamma;
  float* query;
  float* gate;
  __half* keys;
  __half* values;
  const std::uint32_t* position;
  const std::uint32_t* kv_blocks;  ///< K/V block table, null for contiguous.
  float theta;
  float eps;
  const qwen::vision::DeviceRope* rope;
  YarnRope yarn;
};

/// Dense F16 WMMA GEMM over Q8_0 or F16 weights: block = BM rows x BN tokens,
/// BK 32-element K blocks per LDS stage, waves = WM row groups x WN token
/// groups. The codes are dequantized to F16 once per stage as they are
/// committed to LDS (magic-number F16 construction, exact for a Q8_0 code),
/// and the activations are F16 rows [batch][k], so the matrix cores
/// accumulate in F32 with no per-block scaling. SSM waves retain their
/// decoded weight rows in registers instead of staging them in LDS. F16 weights
/// skip decoding and use the same ordered K16 products. y is [batch][m].
template<int BM, int BN, int BK, int WM, int WN, int kRowGroup = 1,
         bool kHcMix = false, bool kSsmConv = false, bool kAttention = false,
         bool kHalfWeights = false, bool kBf16 = false, bool kStrided = false,
         bool kYarn = false, WeightType kPlainType = WeightType::kF16,
         bool kPlainPrefill = false, bool kHalfOut = false>
__launch_bounds__(256) __global__ void DenseF16GEMMKernel(
    const void* __restrict__ w, const __half* __restrict__ x,
    float* __restrict__ y, std::size_t batch, std::size_t m, std::size_t k,
    const __half* xn = nullptr, __half* mixed_half = nullptr,
    void* mixed_q8 = nullptr, const float* conv_w = nullptr,
    float* conv_out = nullptr, AttentionProjectionOutput attention = {},
    std::uint32_t checkpoint_tokens = 0, std::size_t input_stride = 0,
    std::uint32_t first_row_tile = 0) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert(!(kBf16 && kHalfWeights));
  // 16-bit weight rows staged as they are: F16, or BF16 with BF16
  // activations.
  constexpr bool kRawWeights = kHalfWeights || kBf16;
  constexpr int kWeightBlockBytes =
      kHalfWeights && kPlainType == WeightType::kF32 ? 128
      : kRawWeights                                  ? 64
                                                     : 34;
  constexpr bool kRegisterWeights = kSsmConv;
  static_assert(!kRegisterWeights ||
                (BM == 256 && BK == 2 && WM == 8 && WN == 1));
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;
  // Weight and activation K blocks staged per thread; a unit past the
  // stage's block count is idle.
  constexpr int kAUnits = BM * BK;
  constexpr int kBUnits = BN * BK;
  constexpr int kAPer = (kAUnits + 255) / 256;
  constexpr int kBPer = (kBUnits + 255) / 256;

  // One K block of one row or token is four 16-byte chunks; the chunks of
  // nearby rows are permuted so a fragment read (one row per lane, 64-byte
  // stride) covers all bank groups.
  constexpr int kLdsChunks = BK * ((kRegisterWeights ? 0 : BM) + BN) * 4;
  constexpr int kTransposeChunks = 8 *
                                   (kSsmConv ? kSsmProjectionTileTokens : 16) *
                                   36 * sizeof(float) / sizeof(uint4);
  constexpr int kLdsStorage =
      kLdsChunks < kTransposeChunks ? kTransposeChunks : kLdsChunks;
  __shared__ __attribute__((aligned(16))) uint4 s_lds[kLdsStorage];
  auto* s_a = reinterpret_cast<uint4(*)[BM][4]>(s_lds);
  auto* s_b = reinterpret_cast<uint4(*)[BN][4]>(
      s_lds + (kRegisterWeights ? 0 : BK * BM * 4));
  const auto swizzle = [](int row, int c) { return c ^ ((row >> 1) & 3); };

  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const auto* w_bytes = static_cast<const std::uint8_t*>(w);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;
  // Group row tiles for the narrow HC up projection. The tile grid's row
  // count is divisible by kRowGroup; every dot product keeps its K order.
  const unsigned row_group = blockIdx.y / kRowGroup;
  const unsigned within = (blockIdx.y % kRowGroup) * gridDim.x + blockIdx.x;
  const int r_block =
      (first_row_tile + row_group * kRowGroup + within % kRowGroup) * BM;
  const int t_block = (within / kRowGroup) * BN;
  const auto token_of = [&](int local) { return t_block + local; };

  // SSM thread tid owns row tid and K block p of the stage. Other paths
  // use row (p*256+tid)/BK and K block (p*256+tid)%BK; rows past m read
  // the last row with a zero scale.
  const std::uint8_t* a_ptr[kAPer];
  bool a_live[kAPer];
#pragma unroll
  for (int p = 0; p < kAPer; ++p) {
    const int idx = (p * 256) + tid;
    const int r = r_block + (kRegisterWeights ? tid : idx / BK);
    a_live[p] = idx < kAUnits && r < m_i;
    // Interleave the four gate streams within the tile, without repacking
    // weights. Each group of four rows produces one mixed hidden element.
    const int weight_row = kHcMix ? (r % 4) * (m_i / 4) + r / 4 : r;
    a_ptr[p] =
        w_bytes + static_cast<std::size_t>(a_live[p] ? weight_row : (m_i - 1)) *
                      static_cast<std::size_t>(num_kb) * kWeightBlockBytes;
  }
  const __half* b_ptr[kBPer];
#pragma unroll
  for (int p = 0; p < kBPer; ++p) {
    const int idx = (p * 256) + tid;
    const int t = token_of(idx / BK);
    b_ptr[p] =
        idx < kBUnits && t < static_cast<int>(batch)
            ? x + (static_cast<std::size_t>(t) * (kStrided ? input_stride : k))
            : nullptr;
  }
  uint4 a_codes[kAPer][2];
  uint4 a_half[kRawWeights ? kAPer : 1][4];
  std::uint32_t a_d[kAPer];
  uint4 b_data[kBPer][4];

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int kb = kb0 + (kRegisterWeights ? p : (((p * 256) + tid) % BK));
      const bool live = a_live[p] && kb < num_kb;
      if constexpr (kHalfWeights && kPlainType != WeightType::kF16) {
        const auto* src =
            a_ptr[p] + static_cast<std::size_t>(kb) * kWeightBlockBytes;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          std::uint16_t h[8];
#pragma unroll
          for (int i = 0; i < 8; ++i)
            h[i] = __half_as_ushort(__float2half_rn(
                live ? RowElement(src, kPlainType, c * 8 + i) : 0.0F));
          __builtin_memcpy(&a_half[p][c], h, sizeof(uint4));
        }
      } else if constexpr (kRawWeights) {
        const auto* src = reinterpret_cast<const uint4*>(
            a_ptr[p] + static_cast<std::size_t>(kb) * 64);
#pragma unroll
        for (int c = 0; c < 4; ++c)
          a_half[p][c] = live ? src[c] : make_uint4(0u, 0u, 0u, 0u);
      } else {
        const std::uint8_t* blk =
            a_ptr[p] +
            (static_cast<std::size_t>(live ? kb : (num_kb - 1)) * 34);
        a_d[p] = live ? *reinterpret_cast<const std::uint16_t*>(blk) : 0U;
        __builtin_memcpy(&a_codes[p][0], blk + 2, 16);
        __builtin_memcpy(&a_codes[p][1], blk + 18, 16);
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      if (b_ptr[p] != nullptr && kb < num_kb) {
        const auto* src = reinterpret_cast<const uint4*>(b_ptr[p] + (kb * 32));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = src[c];
        }
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  const __half2 magic = __floats2half2_rn(-1152.0F, -1152.0F);
  const __half2 zero2 = __floats2half2_rn(0.0F, 0.0F);
  // Preserve this stage's rounded fragments before prefetch overwrites
  // the raw Q8 codes. Each SSM wave owns both of its 16-row output tiles.
  v16h own_a_lo[BK];
  v16h own_a_hi[BK];
  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kAUnits) {
        break;
      }
      const int row = idx / BK;
      const int kk = idx % BK;
      if constexpr (kRawWeights) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          s_a[kk][row][swizzle(row, c)] = a_half[p][c];
      } else {
        // Q8_0 codes are signed; flipping the sign bit carries q + 128, which
        // the 1152 magic takes back out.
        const std::uint32_t words[8] = {
            a_codes[p][0].x, a_codes[p][0].y, a_codes[p][0].z, a_codes[p][0].w,
            a_codes[p][1].x, a_codes[p][1].y, a_codes[p][1].z, a_codes[p][1].w};
        const __half2 scale2 = __half2half2(
            __builtin_bit_cast(__half, static_cast<std::uint16_t>(a_d[p])));
        __half2 h[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          CodesToHalves(words[i] ^ 0x80808080U, magic, scale2, zero2, h[2 * i],
                        h[2 * i + 1]);
        }
        if constexpr (kRegisterWeights) {
          __builtin_memcpy(&own_a_lo[p], &h[0], 32);
          __builtin_memcpy(&own_a_hi[p], &h[8], 32);
        } else {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            uint4 v;
            __builtin_memcpy(&v, &h[4 * c], 16);
            s_a[kk][row][swizzle(row, c)] = v;
          }
        }
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kBUnits) {
        break;
      }
      const int t = idx / BK;
      const int kk = idx % BK;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        s_b[kk][t][swizzle(t, c)] = b_data[p][c];
      }
    }
  };

  v8f acc[kWaveRowTiles][kWaveTokTiles];
  // Separate K16 chains reduce FP32 accumulation error for the sensitive
  // unquantized router/gate projections, with the same order in every chunk.
  v8f acc_high[kHalfWeights ? kWaveRowTiles : 1]
              [kHalfWeights ? kWaveTokTiles : 1]{};
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      acc[i][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  // Keep full tiles free of per-K bounds branches. The wide SSM paths can
  // omit whole 16-token subtiles in their final partial tile while retaining
  // the original WMMA order for every live output. One launch preserves
  // weight reuse between neighboring token tiles.
  const auto accumulate = [&]<bool partial>() {
    fetch_stage(0);
    for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
      commit_stage();
      __syncthreads();
      if (kb0 + BK < num_kb) {
        fetch_stage(kb0 + BK);
      }
      // Keeps the next stage's loads ahead of the matrix work: scheduled
      // freely, the compiler sinks them below it and the commit waits on
      // the full memory latency.
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int kb = 0; kb < BK; ++kb) {
        v16h a_lo[kWaveRowTiles];
        v16h a_hi[kWaveRowTiles];
        if constexpr (kRegisterWeights) {
          using v8u = __attribute__((__vector_size__(32))) std::uint32_t;
          const v8u own_lo = __builtin_bit_cast(v8u, own_a_lo[kb]);
          const v8u own_hi = __builtin_bit_cast(v8u, own_a_hi[kb]);
          v8u peer_lo;
          v8u peer_hi;
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            peer_lo[q] = XorLane(own_lo[q], 16);
            peer_hi[q] = XorLane(own_hi[q], 16);
          }
#pragma unroll
          for (int i = 0; i < kWaveRowTiles; ++i) {
            a_lo[i] = __builtin_bit_cast(v16h, half_id == i ? own_lo : peer_lo);
            a_hi[i] = __builtin_bit_cast(v16h, half_id == i ? own_hi : peer_hi);
          }
        } else {
#pragma unroll
          for (int i = 0; i < kWaveRowTiles; ++i) {
            const int row = (((wave_row * kWaveRowTiles) + i) * 16) + sub_lane;
            uint4 c[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              c[q] = s_a[kb][row][swizzle(row, q)];
            }
            __builtin_memcpy(&a_lo[i], &c[0], 32);
            __builtin_memcpy(&a_hi[i], &c[2], 32);
          }
        }
#pragma unroll
        for (int j = 0; j < kWaveTokTiles; ++j) {
          if constexpr (partial) {
            if (token_of(((wave_tok * kWaveTokTiles) + j) * 16) >= batch)
              continue;
          }
          const int t = (((wave_tok * kWaveTokTiles) + j) * 16) + sub_lane;
          uint4 c[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            c[q] = s_b[kb][t][swizzle(t, q)];
          }
          v16h b_lo;
          v16h b_hi;
          __builtin_memcpy(&b_lo, &c[0], 32);
          __builtin_memcpy(&b_hi, &c[2], 32);
#pragma unroll
          for (int i = 0; i < kWaveRowTiles; ++i) {
            if constexpr (kBf16) {
              // One K16 chain in K order for every token.
              acc[i][j] = WmmaBf16(a_lo[i], b_lo, acc[i][j]);
              acc[i][j] = WmmaBf16(a_hi[i], b_hi, acc[i][j]);
            } else if constexpr (kHalfWeights) {
              acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
              acc_high[i][j] = Wmma(a_hi[i], b_hi, acc_high[i][j]);
            } else {
              acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
              acc[i][j] = Wmma(a_hi[i], b_hi, acc[i][j]);
            }
          }
        }
      }
      __syncthreads();
    }
  };
  if constexpr (kSsmConv || kStrided) {
    if (t_block + BN <= batch)
      accumulate.template operator()<false>();
    else
      accumulate.template operator()<true>();
  } else {
    accumulate.template operator()<false>();
  }

  if constexpr (kHalfWeights) {
#pragma unroll
    for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j)
        acc[i][j] += acc_high[i][j];
    }
  }

  if constexpr (kAttention) {
    static_assert(BM == 256 && BN == 128 && WM == 8 && WN == 1);
    static_assert(!kHcMix && !kSsmConv);
    constexpr unsigned stride = 36, dim = 256, width = 6144, kvwidth = 512;
    float* scratch = reinterpret_cast<float*>(s_lds);
    float* tile = scratch + wave_id * 16 * stride;
    const unsigned projection_head = r_block / 256;
    const bool query = projection_head < 48 && (projection_head % 2) == 0;
    const bool key = projection_head >= 48 && projection_head < 50;
    const bool gate = projection_head < 48 && (projection_head % 2) == 1;
    const unsigned head =
        projection_head < 48 ? projection_head / 2 : projection_head % 2;
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile[sub_lane * stride + 2 * l + half_id] = acc[0][j][l];
        tile[sub_lane * stride + 16 + 2 * l + half_id] = acc[1][j][l];
      }
      __syncthreads();
#pragma unroll
      for (unsigned phase = 0; phase < 2; ++phase) {
        // One wave handles a token, emulating its original eight-wave norm.
        const unsigned tok_local = wave_id * 2 + phase;
        const std::size_t tok = t_block + j * 16 + tok_local;
        if (tok < batch) {
          float v[8];
#pragma unroll
          for (unsigned c = 0; c < 8; ++c)
            v[c] = scratch[(c * 16 + tok_local) * stride + lane_id];
          if (query || key) {
            float total = 0.0F;
#pragma unroll
            for (unsigned c = 0; c < 8; ++c) {
              // Match the separate norm: round each square and each wave sum
              // before accumulating the eight partials in their original order.
              float sq = v[c] * v[c];
              asm volatile("" : "+v"(sq));
              float ss = WaveSum(sq);
              asm volatile("" : "+v"(ss));
              total += ss;
              asm volatile("" : "+v"(total));
            }
            const float scale = rsqrtf(total / 256.0F + attention.eps);
            const float* gamma = query ? attention.q_gamma : attention.k_gamma;
#pragma unroll
            for (unsigned c = 0; c < 8; ++c) {
              v[c] = v[c] * scale;
              asm volatile("" : "+v"(v[c]));
              v[c] = v[c] * gamma[c * 32 + lane_id];
              asm volatile("" : "+v"(v[c]));
            }
            float freq = powf(attention.theta,
                              -2.0F * static_cast<float>(lane_id) / 64.0F);
            if constexpr (kYarn)
              freq = YarnFrequency(freq, lane_id, attention.yarn);
            float sn = 0.0F, cs = 0.0F;
            sincosf(qwen::vision::RopePosition(
                        attention.rope, *attention.position + tok, lane_id) *
                        freq,
                    &sn, &cs);
            if constexpr (kYarn)
              YarnScale(sn, cs, attention.yarn);
            const float lo = __fmaf_rn(v[0], cs, -__fmul_rn(v[1], sn));
            const float hi = __fmaf_rn(v[0], sn, __fmul_rn(v[1], cs));
            v[0] = lo;
            v[1] = hi;
          }
#pragma unroll
          for (unsigned c = 0; c < 8; ++c) {
            const unsigned col = c * 32 + lane_id;
            if (query)
              attention.query[tok * width + head * dim + col] = v[c];
            else if (gate)
              attention.gate[tok * width + head * dim + col] = v[c];
            else if (key)
              attention.keys[KvWriteRow(attention.kv_blocks,
                                        *attention.position + tok) *
                                 kvwidth +
                             head * dim + col] = __float2half_rn(v[c]);
            else
              attention.values[KvWriteRow(attention.kv_blocks,
                                          *attention.position + tok) *
                                   kvwidth +
                               head * dim + col] = __float2half_rn(v[c]);
          }
        }
      }
      __syncthreads();
    }
    return;
  }

  if constexpr (kHcMix) {
    static_assert(BM == 256 && BN == 128 && BK == 1 && WM == 4 && WN == 2);
    constexpr unsigned kHiddenTile = BM / 4;
    constexpr unsigned kStreamStride = kHiddenTile + 1;
    constexpr unsigned kPlane = 16 * kStreamStride;
    static_assert(4 * kPlane * sizeof(float) <= sizeof(s_lds));
    float* gates = reinterpret_cast<float*>(s_lds);
    const std::size_t hidden = m / 4;
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int token_group = 0; token_group < WN; ++token_group) {
        if (wave_tok == token_group) {
#pragma unroll
          for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
              const unsigned row =
                  (wave_row * kWaveRowTiles + i) * 16 + 2 * l + half_id;
              gates[(row % 4) * kPlane + sub_lane * kStreamStride + row / 4] =
                  acc[i][j][l];
            }
          }
        }
        __syncthreads();
        // A half-wave writes all 64 hidden values of one token. The stream
        // planes are padded to avoid the gate transpose's LDS bank conflicts.
        const unsigned token_in_tile = tid / (kHiddenTile / 4);
        const unsigned h_local = (tid % (kHiddenTile / 4)) * 4;
        const std::size_t token =
            t_block + (token_group * kWaveTokTiles + j) * 16 + token_in_tile;
        const std::size_t h = r_block / 4 + h_local;
        if (token < batch) {
          float4 value{0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
          for (unsigned stream = 0; stream < 4; ++stream) {
            const float* g = gates + stream * kPlane +
                             token_in_tile * kStreamStride + h_local;
            const float4 v = Load4(xn + token * m + stream * hidden + h);
            // Preserve the separate mixer's F32 FMA rounding.
            value.x = __fmaf_rn(v.x, SigmoidF(g[0]), value.x);
            value.y = __fmaf_rn(v.y, SigmoidF(g[1]), value.y);
            value.z = __fmaf_rn(v.z, SigmoidF(g[2]), value.z);
            value.w = __fmaf_rn(v.w, SigmoidF(g[3]), value.w);
          }
          value.x *= 0.25F;
          value.y *= 0.25F;
          value.z *= 0.25F;
          value.w *= 0.25F;
          if (y != nullptr)
            *reinterpret_cast<float4*>(y + token * hidden + h) = value;
          if (mixed_half != nullptr) {
            *reinterpret_cast<__half2*>(mixed_half + token * hidden + h) =
                __floats2half2_rn(value.x, value.y);
            *reinterpret_cast<__half2*>(mixed_half + token * hidden + h + 2) =
                __floats2half2_rn(value.z, value.w);
          }
          if (mixed_q8 != nullptr) {
            float max_abs = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                  fmaxf(fabsf(value.z), fabsf(value.w)));
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
              max_abs = fmaxf(max_abs, XorLane(max_abs, off));
            }
            const float d = max_abs / 127.0F;
            const float id = d != 0.0F ? 1.0F / d : 0.0F;
            const auto q0 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.x * id))));
            const auto q1 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.y * id))));
            const auto q2 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.z * id))));
            const auto q3 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.w * id))));
            const unsigned pos = h % 32;
            auto* tile = Q8ActTile(mixed_q8, hidden / 32,
                                   token / kQ8ActTileTokens, h / 32);
            const std::size_t tl = token % kQ8ActTileTokens;
            *reinterpret_cast<unsigned*>(tile + (pos >> 4) * 256 + tl * 16 +
                                         (pos & 15)) =
                q0 | q1 << 8 | q2 << 16 | q3 << 24;
            if (pos == 0) {
              *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                        tl * sizeof(float)) = d;
            }
          }
        }
        __syncthreads();
      }
    }
    return;
  }

  // Transpose the result through LDS, two row tiles at a time, so every
  // global store covers 32 consecutive rows of one token: a full 128-byte
  // line (half lines cost a read-modify-write on the fabric).
  // Four padding floats reduce scatter bank conflicts and retain float4
  // alignment for both output stores and the fused convolution's reads.
  constexpr unsigned kOutputStride = 36;
  // Pair token tiles for SSM: half as many convolutions cross a tile edge.
  // The larger transpose still fits the projection's existing LDS allocation.
  constexpr unsigned kOutputTokens = kSsmConv ? kSsmProjectionTileTokens : 16;
  constexpr unsigned kOutputGroups = kOutputTokens / 16;
  static_assert(8 * kOutputTokens * kOutputStride * sizeof(float) <=
                sizeof(s_lds));
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  float* tile_scratch =
      reinterpret_cast<float*>(s_lds) + wave_id * kOutputTokens * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; j += kOutputGroups) {
#pragma unroll
      for (unsigned group = 0; group < kOutputGroups; ++group) {
        // scratch[token][row], with 32 output rows per token.
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          tile_scratch[((sub_lane + group * 16) * kOutputStride) + (2 * l) +
                       half_id] = acc[i][j + group][l];
          tile_scratch[((sub_lane + group * 16) * kOutputStride) + 16 +
                       (2 * l) + half_id] = acc[i + 1][j + group][l];
        }
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          static_cast<std::size_t>(r_block) +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 = static_cast<std::size_t>(
          token_of(((wave_tok * kWaveTokTiles) + j) * 16));
#pragma unroll
      for (unsigned group = 0; group < kOutputGroups; ++group) {
        // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
        const int tok_l = (lane_id >> 1) + group * 16;
        const int row_l = (lane_id & 1) * 16;
        const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
        const auto* src = reinterpret_cast<const float4*>(
            tile_scratch + (tok_l * kOutputStride) + row_l);
        if constexpr (kSsmConv) {
          static_assert(BM == 256 && BN == 128 && BK == 2 && WM == 8 &&
                        WN == 1);
          static_assert(!kHcMix);
          constexpr std::uint32_t channels = 10240;
          if (tok < batch) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              const unsigned row = r0 + row_l + v * 4;
              if (row >= m)
                continue;
              const float4 current = src[v];
              if (row >= channels || tok_l < 3 || tok_l >= kOutputTokens - 3 ||
                  tok + 3 >= batch ||
                  (tok < checkpoint_tokens && tok + 3 >= checkpoint_tokens)) {
                *reinterpret_cast<float4*>(y + tok * m + row) = current;
              }
              if (row < channels && tok_l >= 3) {
                const float4 x0 = *reinterpret_cast<const float4*>(
                    tile_scratch + (tok_l - 3) * kOutputStride + row_l + v * 4);
                const float4 x1 = *reinterpret_cast<const float4*>(
                    tile_scratch + (tok_l - 2) * kOutputStride + row_l + v * 4);
                const float4 x2 = *reinterpret_cast<const float4*>(
                    tile_scratch + (tok_l - 1) * kOutputStride + row_l + v * 4);
                const float4 w0 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 0) * 4);
                const float4 w1 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 1) * 4);
                const float4 w2 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 2) * 4);
                const float4 w3 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 3) * 4);
                const float4 value{
                    SsmConv4Value(w0, x0.x, x1.x, x2.x, current.x),
                    SsmConv4Value(w1, x0.y, x1.y, x2.y, current.y),
                    SsmConv4Value(w2, x0.z, x1.z, x2.z, current.z),
                    SsmConv4Value(w3, x0.w, x1.w, x2.w, current.w)};
                *reinterpret_cast<float4*>(conv_out + tok * channels + row) =
                    value;
              }
            }
          }
        } else if constexpr (kHalfOut) {
          // y holds F16 rows: each FP32 sum rounded once (RNE).
          auto* yh = reinterpret_cast<__half*>(y);
          if (tok < batch && r0 + 32 <= m && m % 8 == 0) {
            const float* v = tile_scratch + (tok_l * kOutputStride) + row_l;
            __align__(16) __half h[16];
#pragma unroll
            for (int q = 0; q < 16; ++q)
              h[q] = __float2half_rn(v[q]);
            auto* dst = reinterpret_cast<uint4*>(
                yh + (tok * m) + r0 + static_cast<std::size_t>(row_l));
            dst[0] = *reinterpret_cast<const uint4*>(h);
            dst[1] = *reinterpret_cast<const uint4*>(h + 8);
          } else if (tok < batch) {
#pragma unroll
            for (int q = 0; q < 16; ++q) {
              const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
              if (r < m)
                yh[(tok * m) + r] = __float2half_rn(
                    tile_scratch[(tok_l * kOutputStride) + row_l + q]);
            }
          }
        } else {
          if (tok < batch && r0 + 32 <= m && ((tok * m) % 4 == 0) &&
              (!kPlainPrefill ||
               reinterpret_cast<std::uintptr_t>(y) % 16 == 0)) {
            auto* dst = reinterpret_cast<float4*>(
                y + (tok * m) + r0 + static_cast<std::size_t>(row_l));
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              dst[q] = src[q];
            }
          } else if (tok < batch) {
#pragma unroll
            for (int q = 0; q < 16; ++q) {
              const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
              if (r < m) {
                y[(tok * m) + r] =
                    tile_scratch[(tok_l * kOutputStride) + row_l + q];
              }
            }
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

bool AttentionF16Gemm(const void* weights, const __half* input,
                      const float* q_gamma, const float* k_gamma, float* query,
                      float* gate, __half* keys, __half* values,
                      std::uint32_t n_tokens, const std::uint32_t* position,
                      float theta, float eps, hipStream_t stream,
                      const qwen::vision::DeviceRope* rope,
                      const YarnRope& yarn, const std::uint32_t* kv_blocks) {
  if (n_tokens == 0 || weights == nullptr || input == nullptr ||
      q_gamma == nullptr || k_gamma == nullptr || query == nullptr ||
      gate == nullptr || keys == nullptr || values == nullptr ||
      position == nullptr)
    return false;
  const AttentionProjectionOutput output{q_gamma, k_gamma, query,    gate,
                                         keys,    values,  position, kv_blocks,
                                         theta,   eps,     rope,     yarn};
  // One 256-row tile per query, gate, key or value head (52). Four row
  // tiles per token tile in launch order share each activation tile's read
  // (8.8 to 7.9 ms at 4,096 tokens). One K block per LDS stage halves the
  // stage to 24 KB, so three blocks share a WGP instead of two (7.8 to
  // 7.3 ms); the K order is unchanged.
  if (yarn.enabled != 0) {
    hipLaunchKernelGGL((DenseF16GEMMKernel<256, 128, 1, 8, 1, 4, false, false,
                                           true, false, false, false, true>),
                       dim3((n_tokens + 127) / 128, 52), dim3(kThreads), 0,
                       stream, weights, input, nullptr, n_tokens, 13312, 2560,
                       nullptr, nullptr, nullptr, nullptr, nullptr, output);
    return true;
  }
  hipLaunchKernelGGL(
      (DenseF16GEMMKernel<256, 128, 1, 8, 1, 4, false, false, true>),
      dim3((n_tokens + 127) / 128, 52), dim3(kThreads), 0, stream, weights,
      input, nullptr, n_tokens, 13312, 2560, nullptr, nullptr, nullptr, nullptr,
      nullptr, output);
  return true;
}

bool HcMixF16Gemm(const void* up, const __half* low_rank, const __half* xn,
                  const float* inject_w, float* mixed, __half* mixed_half,
                  void* mixed_q8, float* inject, std::uint32_t n_tokens,
                  std::uint32_t hidden, std::uint32_t rank,
                  hipStream_t stream) {
  if (n_tokens < 96 || hidden != 2560 || rank != 320 || up == nullptr ||
      low_rank == nullptr || xn == nullptr ||
      (mixed == nullptr && mixed_half == nullptr && mixed_q8 == nullptr) ||
      (inject_w != nullptr && inject == nullptr)) {
    return false;
  }
  const dim3 grid((n_tokens + 127) / 128, 4 * hidden / 256);
  hipLaunchKernelGGL((DenseF16GEMMKernel<256, 128, 1, 4, 2, 8, true>), grid,
                     dim3(kThreads), 0, stream, up, low_rank, mixed, n_tokens,
                     4 * hidden, rank, xn, mixed_half, mixed_q8);
  if (inject_w != nullptr) {
    HcMixInjectF16(xn, inject_w, inject, n_tokens, hidden, stream);
  }
  return true;
}

bool UnquantizedF16Gemm(const void* w, const __half* x, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream) {
  if (m == 0 || batch == 0 || k == 0 || k % 32 != 0)
    return false;
  // Wide batches reuse each staged fragment across twice the rows and
  // tokens (router 0.61 to 0.52 ms, GDN alpha/beta 0.14 to 0.09 ms at 4,096
  // tokens). Every output keeps its two K16 chains in K order.
  if (batch >= 1024) {
    hipLaunchKernelGGL(
        (DenseF16GEMMKernel<128, 128, 2, 2, 4, 1, false, false, false, true>),
        dim3((batch + 127) / 128, (m + 127) / 128), dim3(kThreads), 0, stream,
        w, x, out, batch, m, k);
    return true;
  }
  hipLaunchKernelGGL(
      (DenseF16GEMMKernel<64, 64, 2, 2, 4, 1, false, false, false, true>),
      dim3((batch + 63) / 64, (m + 63) / 64), dim3(kThreads), 0, stream, w, x,
      out, batch, m, k);
  return true;
}

bool PlainPrefillGemm(const void* w, WeightType type, const __half* x,
                      float* out, std::size_t batch, std::size_t m,
                      std::size_t k, hipStream_t stream) {
  if (m == 0 || batch == 0 || k == 0 || k % 32 != 0)
    return false;
  // The HC projections (F16, very wide or very long) take the X3 core's
  // FP16 tile: 0.59 / 0.76 ms vs 1.09 / 0.88 ms at 2,048 tokens. Chosen by
  // shape alone, so every chunk of a projection uses the same kernel.
  if (type == WeightType::kF16 && (m >= 4096 || k >= 8192)) {
    gufo::hip::LaunchPlainF16PrefillGemm(w, x, out, batch,
                                         static_cast<std::uint32_t>(m),
                                         static_cast<std::uint32_t>(k), stream);
    return true;
  }
  // Fixed at every M, including singleton/tail chunks: each output keeps
  // one K16 chain for each half of a K32 block, then adds the two totals.
  // Row/token tile coordinates only select operands, never rotate K.
  const auto launch = [&]<WeightType plain>() {
    hipLaunchKernelGGL(
        (DenseF16GEMMKernel<128, 128, 2, 2, 4, 1, false, false, false, true,
                            false, false, false, plain, true>),
        dim3((batch + 127) / 128, (m + 127) / 128), dim3(kThreads), 0, stream,
        w, x, out, batch, m, k);
  };
  switch (type) {
    case WeightType::kF16:
      launch.template operator()<WeightType::kF16>();
      break;
    case WeightType::kBF16:
      launch.template operator()<WeightType::kBF16>();
      break;
    case WeightType::kF32:
      launch.template operator()<WeightType::kF32>();
      break;
    default:
      return false;
  }
  return true;
}

bool DenseBf16Gemm(const void* w, const void* x, float* out, std::size_t batch,
                   std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || batch == 0 || k == 0 || k % 32 != 0)
    return false;
  hipLaunchKernelGGL((DenseF16GEMMKernel<64, 128, 2, 2, 4, 1, false, false,
                                         false, false, true>),
                     dim3((batch + 127) / 128, (m + 63) / 64), dim3(kThreads),
                     0, stream, w, static_cast<const __half*>(x), out, batch, m,
                     k);
  return true;
}

bool DenseF16Gemm(const void* w, const __half* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream,
                  std::size_t input_stride) {
  if (m == 0 || k == 0 || batch == 0 || k % 32 != 0) {
    return false;
  }
  if (input_stride != 0 && input_stride != k) {
    if (batch < 1024 || m != 2560 || k != 6144 || input_stride < k ||
        input_stride % 8 != 0)
      return false;
    hipLaunchKernelGGL((DenseF16GEMMKernel<256, 128, 2, 8, 1, 5, false, false,
                                           false, false, false, true>),
                       dim3((batch + 127) / 128, (m + 255) / 256),
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k,
                       nullptr, nullptr, nullptr, nullptr, nullptr,
                       AttentionProjectionOutput{}, 0, input_stride);
    return true;
  }
  // The same tile plan as the W8A8 route: a 128-token macro tile for wide
  // batches, 64-row tiles for narrow projections, 64 tokens below 96.
  constexpr int kBM = 128;
  if (m <= 512 && batch >= 96) {
    constexpr int kBN = 128;
    constexpr int kNarrowBM = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kNarrowBM - 1) / kNarrowBM));
    hipLaunchKernelGGL((DenseF16GEMMKernel<kNarrowBM, kBN, 2, 2, 4>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  } else if (batch >= 96) {
    // 64 x 64 wave tiles: half the LDS fragment bytes per matrix product
    // of the 32 x 64 tile (the F16 fragments are twice the int8 ones).
    constexpr int kWideBM = 256;
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kWideBM - 1) / kWideBM));
    if (m == 10240 && k == 320 && batch >= 1024) {
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 1, 4, 2, 8>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else if (batch >= 1024 && m == 2560 && k == 6144) {
      // The output projection's activation is 50 MB at 4,096 tokens. In
      // grid order the resident blocks each walk a different token tile of
      // it; walking five row tiles of a token tile first shares each
      // activation tile between them (5.7 to 3.4 ms). Only the launch order
      // changes.
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 2, 8, 1, 5>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else if (batch >= 1024 && (m == 16384 || m == 13312) && k == 2560) {
      // Eight row groups reuse each weight fragment across all token tiles
      // and keep fewer weight fragments live. K accumulation is unchanged.
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 2, 8, 1>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else if (batch >= 1024 && m == 2560) {
      // Shared-expert down: pairs of row tiles per token tile (-11 %).
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 1, 4, 2, 2>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else {
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 1, 4, 2>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    }
  } else {
    constexpr int kBN = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((DenseF16GEMMKernel<kBM, kBN, 4, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  }
  return true;
}

bool HcUpF16Gemm(const void* w, const __half* x, __half* gate,
                 std::size_t batch, std::size_t m, std::size_t k,
                 hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || k % 32 != 0 || m % 8 != 0)
    return false;
  // DenseF16Gemm's tiles for this shape, with F16 stores.
  auto* out = reinterpret_cast<float*>(gate);
  constexpr auto kF16 = WeightType::kF16;
  if (batch >= 96) {
    const dim3 grid(static_cast<unsigned int>((batch + 127) / 128),
                    static_cast<unsigned int>((m + 255) / 256));
    if (m == 10240 && k == 320 && batch >= 1024)
      hipLaunchKernelGGL(
          (DenseF16GEMMKernel<256, 128, 1, 4, 2, 8, false, false, false, false,
                              false, false, false, kF16, false, true>),
          grid, dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    else
      hipLaunchKernelGGL(
          (DenseF16GEMMKernel<256, 128, 1, 4, 2, 1, false, false, false, false,
                              false, false, false, kF16, false, true>),
          grid, dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  } else {
    const dim3 grid(static_cast<unsigned int>((batch + 63) / 64),
                    static_cast<unsigned int>((m + 127) / 128));
    hipLaunchKernelGGL(
        (DenseF16GEMMKernel<128, 64, 4, 4, 2, 1, false, false, false, false,
                            false, false, false, kF16, false, true>),
        grid, dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  }
  return true;
}

bool DenseF16SsmGemm(const void* w, const __half* x, const float* conv_w,
                     const float* history, float* qkvz, float* convolved,
                     std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                     std::uint32_t channels, std::uint32_t kernel,
                     hipStream_t stream, std::uint32_t checkpoint_tokens,
                     hipStream_t z_stream, hipEvent_t z_start) {
  if (n_tokens == 0 || m != 16384 || k != 2560 || channels != 10240 ||
      kernel != kSsmConvTaps) {
    return false;
  }
  // Row-tile pairs per token tile in launch order (-0.9 %).
  const unsigned token_tiles = (n_tokens + 127) / 128;
  const unsigned qkv_tiles = z_stream != nullptr ? channels / 256 : m / 256;
  hipLaunchKernelGGL((DenseF16GEMMKernel<256, 128, 2, 8, 1, 2, false, true>),
                     dim3(token_tiles, qkv_tiles), dim3(kThreads), 0, stream, w,
                     x, qkvz, n_tokens, m, k, nullptr, nullptr, nullptr, conv_w,
                     convolved, AttentionProjectionOutput{}, checkpoint_tokens);
  if (z_stream != nullptr) {
    // Start after the QKV tiles: both GEMMs are compute bound, the
    // recurrence that follows them is not. If the fork cannot be ordered,
    // the Z tiles stay on `stream`.
    if (hipEventRecord(z_start, stream) != hipSuccess ||
        hipStreamWaitEvent(z_stream, z_start, 0) != hipSuccess)
      z_stream = stream;
    hipLaunchKernelGGL(
        (DenseF16GEMMKernel<256, 128, 2, 8, 1, 2, false, true>),
        dim3(token_tiles, m / 256 - qkv_tiles), dim3(kThreads), 0, z_stream, w,
        x, qkvz, n_tokens, m, k, nullptr, nullptr, nullptr, conv_w, convolved,
        AttentionProjectionOutput{}, checkpoint_tokens, 0, qkv_tiles);
  }
  hipLaunchKernelGGL(
      SsmConvBoundaryKernel,
      dim3(Blocks(channels / 4), (n_tokens + kSsmProjectionTileTokens - 1) /
                                     kSsmProjectionTileTokens),
      dim3(kThreads), 0, stream, qkvz, conv_w, history, convolved, n_tokens,
      channels, m);
  return true;
}

void PrepareExl3DensePrefill(std::span<const Exl3DensePrefillProblem> problems,
                             gufo::hip::Exl3RotateProblem* rotations,
                             gufo::hip::Exl3GemmProblem* gemms,
                             hipStream_t stream) {
  if (problems.empty() || problems.size() > 3)
    throw std::invalid_argument("EXL3 dense prefill requires 1..3 projections");
  Exl3DensePrefillArguments args{};
  args.count = static_cast<unsigned>(problems.size());
  for (unsigned i = 0; i < args.count; ++i)
    args.problems[i] = problems[i];
  hipLaunchKernelGGL(PrepareExl3DensePrefillKernel, dim3(1), dim3(32), 0,
                     stream, args, rotations, gemms);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
