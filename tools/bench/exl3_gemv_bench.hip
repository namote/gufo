// X2b (plan/exl3/XB-DESIGN.md): standalone bandwidth experiment of the EXL3
// trellis decode GEMV core on gfx1151, before any production wiring.
//
// Kernel: one wave decodes 16 x 16 trellis tiles straight into registers (no
// LDS staging, no barriers in the K loop) and multiplies them with an FP16
// rotated input through v_dot2_f32_f16, accumulating in FP32. Lane L decodes
// tile positions 8L .. 8L + 7, which the tensor-core permutation of the EXL3
// format (docs/EXL3-FORMAT.md "Tile layout") places at rows (r0, r0+1, r0+8,
// r0+9), r0 = (L % 4) * 2, of columns L / 4 and L / 4 + 8. The lane's eight
// 16-bit states lie in one 64-bit window of the tile bit string (K <= 6), cut
// out of three dwords with two v_alignbit_b32, so every state is a constant
// shift of that window. The warps of a block split K over the same N tiles
// and reduce through LDS in warp order. The output epilogue is the design's
// "store" form: the block that completes a 128-column segment (self-resetting
// arrival counter) rotates it, y = svh * H(acc), and writes FP32.
// Row r of an M-row call accumulates in exactly the order of an M = 1 call.
//
// The lane layout and the mul1 pair decode (v_sad_u8 / v_sad_hi_u8 byte sums
// packed straight into an FP16 pair) follow upstream exllamav3 v1.6.5
// exllamav3_ext/rocm/quant/exl3_gemv_kernel_rdna.cuh and
// exl3_gemv_tiles_rdna.cuh (MIT, Copyright (c) 2025 Turboderp); the code here
// is written anew for this experiment.
//
// Weights are random trellis words (every 16-bit state is valid), rotated
// over copies well beyond the 32 MiB MALL. Times are per launch (hipEvent
// pairs) after warm-up; GB/s counts trellis bytes only.
//
// A launch may carry up to three matrices of the same k (the production
// groups q + k + v, gate + up, qkv + z), each with its own rotated input.
// The input is row-major [M][k] or lane-major (each lane's M rows of one k
// tile contiguous, written that way by the producer kernel).
//
// Modes: check | time [reps]
//   check  decoded tiles against the CPU reference (FP16 bits), GEMV outputs
//          against an FP64 reference fed the same FP16 input, every M, input
//          layout and N tiles per wave against M = 1 (bits), and every slot
//          of a multi-matrix launch against its own launch (bits).
//   time   case x M x warps x N tiles per wave x input layout, then a plain
//          read of the same buffers as the bandwidth ceiling.
// Build: plan/exl3/XB-RESULTS.md.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/core/quant/exl3/format.hpp"

#define HIP_OK(expr)                                                    \
  do {                                                                  \
    const hipError_t err_ = (expr);                                     \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

namespace {

namespace exl3 = gufo::quant::exl3;

using half2_t = _Float16 __attribute__((ext_vector_type(2)));
using half4_t = _Float16 __attribute__((ext_vector_type(4)));

constexpr float kHadamardScale = 0.0883883461356163F;
constexpr std::uint32_t kMul1 = 0x83DCD12DU;
constexpr int kMaxSegments = 4096;  // per launch site; lm_head has 1940

// ---------------------------------------------------------------------------
// Device decode
// ---------------------------------------------------------------------------

// Lane-constant placement of the lane's 64-bit state window in a tile of
// 8 * K dwords: dword offsets of the three source words and the bit offset of
// the first state (never 0 for K = 1..6, so both alignbit shifts are < 32).
struct LaneWindow {
  unsigned o0, o1, o2;
  unsigned shift;  // 32 - bit offset
};

template<int K>
__device__ __forceinline__ LaneWindow MakeLaneWindow(unsigned lane) {
  constexpr int kTileBits = 256 * K;
  constexpr unsigned kWords = 8 * K;
  const int start =
      ((8 * static_cast<int>(lane) + 1) * K - 16 + kTileBits) % kTileBits;
  const unsigned i0 = static_cast<unsigned>(start) / 32;
  LaneWindow w;
  w.o0 = i0;
  w.o1 = (i0 + 1) % kWords;
  w.o2 = (i0 + 2) % kWords;
  w.shift = 32U - static_cast<unsigned>(start) % 32;
  return w;
}

// x * kMul1 mod 2^32 for a 16-bit x. kHash 0: one v_mul_lo_u32 (quarter
// rate); kHash 1: two 24-bit multiplies and a shift, full rate.
template<int kHash>
__device__ __forceinline__ std::uint32_t HashMul1(std::uint32_t x) {
  if constexpr (kHash == 0) {
    return x * kMul1;
  } else {
    const std::uint32_t hi = __umul24(x, kMul1 >> 16) << 16;
    return __umul24(x, kMul1 & 0xffffU) + hi;
  }
}

// Two hashed states -> the decoded FP16 pair (lo = first state), bit for bit
// the mul1 codebook: fma(1024 + bytesum, 0x1eee, 0xc931) rounded once.
__device__ __forceinline__ half2_t DecodePairMul1(std::uint32_t p0,
                                                  std::uint32_t p1) {
  std::uint32_t s = __builtin_amdgcn_sad_u8(p0, 0U, 0x64006400U);
  s = __builtin_amdgcn_sad_hi_u8(p1, 0U, s);
  const half2_t u = __builtin_bit_cast(half2_t, s);
  const half2_t inv = __builtin_bit_cast(half2_t, 0x1eee1eeeU);
  const half2_t bias = __builtin_bit_cast(half2_t, 0xc931c931U);
  return __builtin_elementwise_fma(u, inv, bias);
}

// The lane's eight decoded weights as four pairs: w[0] = rows (r0, r0+1) and
// w[1] = rows (r0+8, r0+9) of column L / 4; w[2], w[3] the same rows of
// column L / 4 + 8.
template<int K, int kHash>
__device__ __forceinline__ void DecodeLane(
    const std::uint32_t* __restrict__ tile, const LaneWindow& lw,
    half2_t (&w)[4]) {
  const std::uint32_t d0 = tile[lw.o0];
  const std::uint32_t d1 = tile[lw.o1];
  const std::uint32_t d2 = tile[lw.o2];
  const std::uint32_t hi = __builtin_amdgcn_alignbit(d0, d1, lw.shift);
  const std::uint32_t lo = __builtin_amdgcn_alignbit(d1, d2, lw.shift);
  const std::uint64_t window = (static_cast<std::uint64_t>(hi) << 32) | lo;
  std::uint32_t p[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const std::uint32_t state =
        static_cast<std::uint32_t>(window >> (48 - K * j)) & 0xffffU;
    p[j] = HashMul1<kHash>(state);
  }
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    w[q] = DecodePairMul1(p[2 * q], p[2 * q + 1]);
  }
}

// ---------------------------------------------------------------------------
// GEMV kernel
// ---------------------------------------------------------------------------

// One matrix of a launch. Matrices sharing a launch share k, M and the
// launch shape; each has its own rotated input (its own suh), output and
// counters. Blocks [block_begin, next block_begin) work this matrix.
struct Slot {
  const std::uint16_t* trellis;  // [k/16][n/16][16K]
  const _Float16* a;             // rotated input, layout per kLayout
  const _Float16* svh;           // [n]
  float* partial;                // [M][n], pre-rotation sums
  float* y;                      // [M][n]
  unsigned* counters;            // [n / 128], zero between launches
  unsigned n;
  unsigned block_begin;
};

constexpr int kMaxSlots = 3;

struct GemvArgs {
  Slot slot[kMaxSlots];
  unsigned slots;
  unsigned k;
};

// Input layouts. kRowMajor: [M][k]. kLaneMajor: per 16-element k tile, four
// groups q = 0..3 (lanes with r0 = 2q), each [M rows][a[2q], a[2q+1],
// a[2q+8], a[2q+9]], so a lane reads its M rows as one contiguous 8M bytes.
constexpr int kRowMajor = 0;
constexpr int kLaneMajor = 1;

template<int K, int M, int W, int TN, int kHash, int kLayout>
__global__ __launch_bounds__(W * 32) void Exl3GemvKernel(GemvArgs args) {
  constexpr unsigned kTileDwords = 8 * K;
  __shared__ float red[W][TN][M][16];
  __shared__ unsigned last_block;

  // Wave-uniform slot selection (no dynamic indexing of the argument block).
  Slot sl = args.slot[0];
  if (args.slots > 1 && blockIdx.x >= args.slot[1].block_begin) {
    sl = args.slot[1];
  }
  if (args.slots > 2 && blockIdx.x >= args.slot[2].block_begin) {
    sl = args.slot[2];
  }

  const unsigned lane = threadIdx.x % 32;
  const unsigned warp = threadIdx.x / 32;
  const unsigned n_tiles = sl.n / 16;
  const unsigned k_tiles = args.k / 16;
  const unsigned nt0 = (blockIdx.x - sl.block_begin) * TN;
  const unsigned kb0 = k_tiles * warp / W;
  const unsigned kb1 = k_tiles * (warp + 1) / W;

  const LaneWindow lw = MakeLaneWindow<K>(lane);
  const unsigned r0 = (lane % 4) * 2;

  float acc_a[TN][M];
  float acc_b[TN][M];
#pragma unroll
  for (int t = 0; t < TN; ++t) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      acc_a[t][r] = 0.0F;
      acc_b[t][r] = 0.0F;
    }
  }

  const std::uint32_t* base =
      reinterpret_cast<const std::uint32_t*>(sl.trellis);
  for (unsigned kt = kb0; kt < kb1; ++kt) {
    half2_t a01[M];
    half2_t a89[M];
    if constexpr (kLayout == kRowMajor) {
#pragma unroll
      for (int r = 0; r < M; ++r) {
        const _Float16* row =
            sl.a + static_cast<std::size_t>(r) * args.k + kt * 16 + r0;
        a01[r] = *reinterpret_cast<const half2_t*>(row);
        a89[r] = *reinterpret_cast<const half2_t*>(row + 8);
      }
    } else {
      const half4_t* group = reinterpret_cast<const half4_t*>(
          sl.a + (static_cast<std::size_t>(kt) * 16 + r0 * 2) * M);
#pragma unroll
      for (int r = 0; r < M; ++r) {
        const half4_t v = group[r];
        a01[r] = half2_t{v.x, v.y};
        a89[r] = half2_t{v.z, v.w};
      }
    }
#pragma unroll
    for (int t = 0; t < TN; ++t) {
      const std::uint32_t* tile =
          base +
          (static_cast<std::size_t>(kt) * n_tiles + nt0 + t) * kTileDwords;
      half2_t w[4];
      DecodeLane<K, kHash>(tile, lw, w);
#pragma unroll
      for (int r = 0; r < M; ++r) {
        acc_a[t][r] = __builtin_amdgcn_fdot2(w[0], a01[r], acc_a[t][r], false);
        acc_a[t][r] = __builtin_amdgcn_fdot2(w[1], a89[r], acc_a[t][r], false);
        acc_b[t][r] = __builtin_amdgcn_fdot2(w[2], a01[r], acc_b[t][r], false);
        acc_b[t][r] = __builtin_amdgcn_fdot2(w[3], a89[r], acc_b[t][r], false);
      }
    }
  }

  // Quad reduction: lanes 4c .. 4c + 3 hold columns c and c + 8.
#pragma unroll
  for (int t = 0; t < TN; ++t) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      float va = acc_a[t][r];
      float vb = acc_b[t][r];
      va += __shfl_xor(va, 1, 32);
      vb += __shfl_xor(vb, 1, 32);
      va += __shfl_xor(va, 2, 32);
      vb += __shfl_xor(vb, 2, 32);
      if (lane % 4 == 0) {
        red[warp][t][r][lane / 4] = va;
        red[warp][t][r][lane / 4 + 8] = vb;
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    if (lane < 16) {
#pragma unroll
      for (int t = 0; t < TN; ++t) {
#pragma unroll
        for (int r = 0; r < M; ++r) {
          float sum = red[0][t][r][lane];
#pragma unroll
          for (int v = 1; v < W; ++v) {
            sum += red[v][t][r][lane];
          }
          sl.partial[static_cast<std::size_t>(r) * sl.n + (nt0 + t) * 16 +
                     lane] = sum;
        }
      }
    }
    __threadfence();
    unsigned last = 0;
    const unsigned segment = nt0 / 8;
    if (lane == 0) {
      constexpr unsigned kArrivals = 8 / TN;
      const unsigned old = atomicAdd(&sl.counters[segment], 1U);
      last = old == kArrivals - 1 ? 1U : 0U;
      if (last != 0) {
        sl.counters[segment] = 0;
      }
    }
    last_block = __shfl(last, 0, 32);
  }
  __syncthreads();
  if (last_block == 0) {
    return;
  }
  __threadfence();

  // Output rotation of the completed segment, one wave per row.
  const unsigned seg_base = (nt0 / 8) * 128;
  for (unsigned r = warp; r < M; r += W) {
    float v[4];
    const float* src =
        sl.partial + static_cast<std::size_t>(r) * sl.n + seg_base + lane * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[j] = __hip_atomic_load(src + j, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
    }
#pragma unroll
    for (unsigned length = 1; length < 4; length <<= 1) {
#pragma unroll
      for (unsigned i = 0; i < 4; i += length << 1) {
#pragma unroll
        for (unsigned j = i; j < i + length; ++j) {
          const float a = v[j];
          const float b = v[j + length];
          v[j] = a + b;
          v[j + length] = a - b;
        }
      }
    }
#pragma unroll
    for (unsigned mask = 1; mask < 32; mask <<= 1) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float other = __shfl_xor(v[j], static_cast<int>(mask), 32);
        v[j] = (lane & mask) != 0 ? other - v[j] : v[j] + other;
      }
    }
    float* dst =
        sl.y + static_cast<std::size_t>(r) * sl.n + seg_base + lane * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float scaled = v[j] * kHadamardScale;
      dst[j] = scaled * static_cast<float>(sl.svh[seg_base + lane * 4 + j]);
    }
  }
}

// Every lane writes its eight decoded weights into the row-major [k][n] FP16
// matrix: the decode check against the CPU reference.
template<int K>
__global__ void Exl3DecodeKernel(const std::uint16_t* trellis, unsigned k,
                                 unsigned n, _Float16* out) {
  const unsigned lane = threadIdx.x % 32;
  const unsigned tile_id = blockIdx.x;
  const unsigned n_tiles = n / 16;
  const unsigned kt = tile_id / n_tiles;
  const unsigned nt = tile_id % n_tiles;
  const LaneWindow lw = MakeLaneWindow<K>(lane);
  const std::uint32_t* tile = reinterpret_cast<const std::uint32_t*>(trellis) +
                              static_cast<std::size_t>(tile_id) * 8 * K;
  half2_t w[4];
  DecodeLane<K, 0>(tile, lw, w);
  const unsigned r0 = (lane % 4) * 2;
  const unsigned rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
  const unsigned cols[2] = {lane / 4, lane / 4 + 8};
  for (int q = 0; q < 4; ++q) {
    const unsigned col = cols[q / 2];
    const unsigned row_lo = rows[(q % 2) * 2];
    const unsigned row_hi = rows[(q % 2) * 2 + 1];
    out[(static_cast<std::size_t>(kt) * 16 + row_lo) * n + nt * 16 + col] =
        w[q].x;
    out[(static_cast<std::size_t>(kt) * 16 + row_hi) * n + nt * 16 + col] =
        w[q].y;
  }
}

// Plain streaming read of the same bytes: the bandwidth ceiling.
__global__ void ReadKernel(const uint4* __restrict__ data, std::size_t count,
                           unsigned* sink) {
  unsigned acc = 0;
  for (std::size_t i =
           blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       i < count; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const uint4 v = data[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9U) {
    sink[0] = acc;
  }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

// One launch: matrices of the same k and K that read different rotated
// copies of one activation (one matrix, or a production group).
struct Case {
  const char* name;
  unsigned k;
  int bits;
  std::vector<unsigned> ns;
};

const std::vector<Case>& TimeCases() {
  static const std::vector<Case> cases = {
      {"qkv", 5120, 5, {10240}},
      {"z", 5120, 5, {6144}},
      {"q", 5120, 5, {12288}},
      {"kv", 5120, 5, {1024}},
      {"gate_up", 5120, 5, {17408}},
      {"o_out", 6144, 5, {5120}},
      {"down", 17408, 5, {5120}},
      {"lm_head", 5120, 6, {248320}},
      {"gate_up_k4", 5120, 4, {17408}},
      {"down_k4", 17408, 4, {5120}},
      {"qkv+z", 5120, 5, {10240, 6144}},
      {"q+k+v", 5120, 5, {12288, 1024, 1024}},
      {"gate+up", 5120, 5, {17408, 17408}},
  };
  return cases;
}

std::size_t TrellisBytes(unsigned k, unsigned n, int bits) {
  return static_cast<std::size_t>(k / 16) * (n / 16) * 32 * bits;
}

std::size_t CaseBytes(const Case& c) {
  std::size_t total = 0;
  for (unsigned n : c.ns)
    total += TrellisBytes(c.k, n, c.bits);
  return total;
}

unsigned SplitWarps(unsigned k_tiles, unsigned blocks) {
  if (k_tiles <= 128)
    return 4;
  if (blocks >= 1024)
    return 4;
  if (blocks < 128 || k_tiles >= 1024)
    return 16;
  return 8;
}

using Launcher = void (*)(GemvArgs, hipStream_t);

template<int K, int M, int W, int TN, int kLayout>
void LaunchGemv(GemvArgs args, hipStream_t stream) {
  unsigned blocks = 0;
  for (unsigned i = 0; i < args.slots; ++i) {
    args.slot[i].block_begin = blocks;
    blocks += args.slot[i].n / 16 / TN;
  }
  hipLaunchKernelGGL((Exl3GemvKernel<K, M, W, TN, 0, kLayout>), dim3(blocks),
                     dim3(W * 32), 0, stream, args);
}

template<int K, int M, int TN, int kLayout>
Launcher PickWarps(unsigned w) {
  switch (w) {
    case 4:
      return LaunchGemv<K, M, 4, TN, kLayout>;
    case 8:
      return LaunchGemv<K, M, 8, TN, kLayout>;
    default:
      return LaunchGemv<K, M, 16, TN, kLayout>;
  }
}

template<int K, int M>
Launcher PickForm(unsigned w, int tn, int layout) {
  if (tn == 1) {
    return layout == 0 ? PickWarps<K, M, 1, kRowMajor>(w)
                       : PickWarps<K, M, 1, kLaneMajor>(w);
  }
  return layout == 0 ? PickWarps<K, M, 2, kRowMajor>(w)
                     : PickWarps<K, M, 2, kLaneMajor>(w);
}

template<int K>
Launcher PickRows(int m, unsigned w, int tn, int layout) {
  switch (m) {
    case 1:
      return PickForm<K, 1>(w, tn, layout);
    case 2:
      return PickForm<K, 2>(w, tn, layout);
    case 4:
      return PickForm<K, 4>(w, tn, layout);
    default:
      return PickForm<K, 8>(w, tn, layout);
  }
}

Launcher Pick(int bits, int m, unsigned w, int tn, int layout) {
  switch (bits) {
    case 4:
      return PickRows<4>(m, w, tn, layout);
    case 5:
      return PickRows<5>(m, w, tn, layout);
    default:
      return PickRows<6>(m, w, tn, layout);
  }
}

std::uint16_t RandomHalf(std::mt19937& rng, float lo, float hi) {
  std::uniform_real_distribution<float> dist(lo, hi);
  return exl3::HalfFromDouble(dist(rng));
}

// The lane-major copy of an [m][k] row-major input (see kLaneMajor).
std::vector<std::uint16_t> LaneMajor(const std::vector<std::uint16_t>& a,
                                     unsigned k, int m) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(m) * k);
  std::size_t o = 0;
  for (unsigned kt = 0; kt < k / 16; ++kt) {
    for (unsigned q = 0; q < 4; ++q) {
      for (int r = 0; r < m; ++r) {
        const std::uint16_t* row =
            a.data() + static_cast<std::size_t>(r) * k + kt * 16 + 2 * q;
        out[o++] = row[0];
        out[o++] = row[1];
        out[o++] = row[8];
        out[o++] = row[9];
      }
    }
  }
  return out;
}

struct SlotHost {
  std::vector<std::uint16_t> trellis;
  std::vector<std::uint16_t> a;  // [8][k] row-major
  std::vector<std::uint16_t> svh;
};

struct SlotDevice {
  std::uint16_t* trellis = nullptr;  // copies * words
  std::size_t copy_words = 0;
  _Float16* a_row = nullptr;  // [8][k]
  _Float16* a_lane[9] = {};   // lane-major per M (1, 2, 4, 8)
  _Float16* svh = nullptr;
  float* partial = nullptr;
  float* y = nullptr;
  unsigned* counters = nullptr;
};

struct Device {
  std::vector<SlotDevice> slots;
  int copies = 1;
};

template<typename T>
T* Upload(const std::vector<std::uint16_t>& host) {
  T* d = nullptr;
  HIP_OK(hipMalloc(&d, host.size() * 2));
  HIP_OK(hipMemcpy(d, host.data(), host.size() * 2, hipMemcpyHostToDevice));
  return d;
}

Device Allocate(const Case& c, int copies, std::mt19937& rng,
                std::vector<SlotHost>* hosts) {
  Device d;
  d.copies = copies;
  hosts->clear();
  for (unsigned n : c.ns) {
    SlotHost h;
    h.trellis.resize(TrellisBytes(c.k, n, c.bits) / 2);
    for (auto& w : h.trellis)
      w = static_cast<std::uint16_t>(rng());
    h.a.resize(static_cast<std::size_t>(8) * c.k);
    for (auto& v : h.a)
      v = RandomHalf(rng, -2.0F, 2.0F);
    h.svh.resize(n);
    for (auto& v : h.svh) {
      v = RandomHalf(rng, 0.002F, 0.02F);
      if ((rng() & 1U) != 0U)
        v ^= 0x8000U;
    }
    SlotDevice sd;
    sd.copy_words = h.trellis.size();
    HIP_OK(hipMalloc(&sd.trellis, sd.copy_words * 2 * copies));
    for (int i = 0; i < copies; ++i) {
      HIP_OK(hipMemcpy(sd.trellis + sd.copy_words * i, h.trellis.data(),
                       sd.copy_words * 2, hipMemcpyHostToDevice));
    }
    sd.a_row = Upload<_Float16>(h.a);
    for (int m : {1, 2, 4, 8}) {
      std::vector<std::uint16_t> first(
          h.a.begin(), h.a.begin() + static_cast<std::ptrdiff_t>(m) * c.k);
      sd.a_lane[m] = Upload<_Float16>(LaneMajor(first, c.k, m));
    }
    sd.svh = Upload<_Float16>(h.svh);
    HIP_OK(hipMalloc(&sd.partial, sizeof(float) * 8 * n));
    HIP_OK(hipMalloc(&sd.y, sizeof(float) * 8 * n));
    HIP_OK(hipMalloc(&sd.counters, sizeof(unsigned) * kMaxSegments));
    HIP_OK(hipMemset(sd.counters, 0, sizeof(unsigned) * kMaxSegments));
    d.slots.push_back(sd);
    hosts->push_back(std::move(h));
  }
  return d;
}

void Free(Device& d) {
  for (SlotDevice& sd : d.slots) {
    HIP_OK(hipFree(sd.trellis));
    HIP_OK(hipFree(sd.a_row));
    for (int m : {1, 2, 4, 8})
      HIP_OK(hipFree(sd.a_lane[m]));
    HIP_OK(hipFree(sd.svh));
    HIP_OK(hipFree(sd.partial));
    HIP_OK(hipFree(sd.y));
    HIP_OK(hipFree(sd.counters));
  }
}

// Arguments for slots [first, first + count) of `c`, rows m, input layout.
GemvArgs Args(const Case& c, const Device& d, int copy, int m, int layout,
              unsigned first = 0, unsigned count = 0) {
  if (count == 0)
    count = static_cast<unsigned>(c.ns.size());
  GemvArgs a{};
  a.slots = count;
  a.k = c.k;
  for (unsigned i = 0; i < count; ++i) {
    const SlotDevice& sd = d.slots[first + i];
    Slot& s = a.slot[i];
    s.trellis = sd.trellis + sd.copy_words * copy;
    s.a = layout == 0 ? sd.a_row : sd.a_lane[m];
    s.svh = sd.svh;
    s.partial = sd.partial;
    s.y = sd.y;
    s.counters = sd.counters;
    s.n = c.ns[first + i];
  }
  return a;
}

unsigned Blocks(const Case& c, int tn) {
  unsigned blocks = 0;
  for (unsigned n : c.ns)
    blocks += n / 16 / tn;
  return blocks;
}

std::vector<float> ReadY(const SlotDevice& sd, unsigned n, int m) {
  std::vector<float> y(static_cast<std::size_t>(m) * n);
  HIP_OK(hipMemcpy(y.data(), sd.y, y.size() * 4, hipMemcpyDeviceToHost));
  return y;
}

// FP64 reference: y = svh * H(A W') per row, A the FP16 input as given.
std::vector<double> Reference(unsigned k, unsigned n, int bits,
                              const SlotHost& h, int m) {
  const exl3::BitRate rate{bits, false};
  std::vector<std::uint16_t> inner(static_cast<std::size_t>(k) * n);
  exl3::DequantInner(h.trellis.data(), k, n, rate, exl3::Codebook::kMul1,
                     inner.data());
  std::vector<double> wd(inner.size());
  for (std::size_t i = 0; i < inner.size(); ++i) {
    wd[i] = exl3::HalfToDouble(inner[i]);
  }
  std::vector<double> y(static_cast<std::size_t>(m) * n, 0.0);
  for (int r = 0; r < m; ++r) {
    double* yr = y.data() + static_cast<std::size_t>(r) * n;
    for (unsigned i = 0; i < k; ++i) {
      const double x =
          exl3::HalfToDouble(h.a[static_cast<std::size_t>(r) * k + i]);
      const double* wr = wd.data() + static_cast<std::size_t>(i) * n;
      for (unsigned j = 0; j < n; ++j)
        yr[j] += x * wr[j];
    }
    exl3::HadamardBlocks(std::span<double>(yr, n));
    for (unsigned j = 0; j < n; ++j)
      yr[j] *= exl3::HalfToDouble(h.svh[j]);
  }
  return y;
}

int CheckDecode(std::mt19937& rng) {
  int failures = 0;
  for (int bits : {4, 5, 6}) {
    const Case c{"decode", 256, bits, {512}};
    std::vector<SlotHost> hosts;
    Device d = Allocate(c, 1, rng, &hosts);
    const unsigned n = c.ns[0];
    _Float16* out = nullptr;
    HIP_OK(hipMalloc(&out, sizeof(_Float16) * c.k * n));
    const unsigned tiles = (c.k / 16) * (n / 16);
    const std::uint16_t* t = d.slots[0].trellis;
    if (bits == 4) {
      hipLaunchKernelGGL(Exl3DecodeKernel<4>, dim3(tiles), dim3(32), 0, 0, t,
                         c.k, n, out);
    } else if (bits == 5) {
      hipLaunchKernelGGL(Exl3DecodeKernel<5>, dim3(tiles), dim3(32), 0, 0, t,
                         c.k, n, out);
    } else {
      hipLaunchKernelGGL(Exl3DecodeKernel<6>, dim3(tiles), dim3(32), 0, 0, t,
                         c.k, n, out);
    }
    HIP_OK(hipDeviceSynchronize());
    std::vector<std::uint16_t> got(static_cast<std::size_t>(c.k) * n);
    HIP_OK(hipMemcpy(got.data(), out, got.size() * 2, hipMemcpyDeviceToHost));
    std::vector<std::uint16_t> want(got.size());
    exl3::DequantInner(hosts[0].trellis.data(), c.k, n,
                       exl3::BitRate{bits, false}, exl3::Codebook::kMul1,
                       want.data());
    std::size_t diff = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
      diff += got[i] != want[i];
    std::printf("check decode K=%d: %zu / %zu FP16 values differ\n", bits, diff,
                got.size());
    failures += diff != 0;
    HIP_OK(hipFree(out));
    Free(d);
  }
  return failures;
}

// Single matrices: M = 1 per row (row-major) against FP64; every M, layout
// and N tiles per wave bit-identical to that.
int CheckSingles(std::mt19937& rng) {
  int failures = 0;
  const Case cases[] = {{"kv", 5120, 5, {1024}},
                        {"o_out", 6144, 5, {5120}},
                        {"head_part", 5120, 6, {2048}},
                        {"down_k4", 17408, 4, {1024}}};
  for (const Case& c : cases) {
    std::vector<SlotHost> hosts;
    Device d = Allocate(c, 1, rng, &hosts);
    const unsigned n = c.ns[0];
    const std::vector<double> ref = Reference(c.k, n, c.bits, hosts[0], 8);
    const unsigned k_tiles = c.k / 16;
    // The split-K warp count sets the summation order, so the M = 1
    // baseline is taken per N tiles per wave (each picks its own count).
    double rel = 0.0;
    int differ = 0;
    int runs = 0;
    for (int tn : {1, 2}) {
      const unsigned w = SplitWarps(k_tiles, Blocks(c, tn));
      std::vector<float> rows1(static_cast<std::size_t>(8) * n);
      for (int r = 0; r < 8; ++r) {
        GemvArgs args = Args(c, d, 0, 1, 0);
        args.slot[0].a = d.slots[0].a_row + static_cast<std::size_t>(r) * c.k;
        Pick(c.bits, 1, w, tn, 0)(args, nullptr);
        const std::vector<float> y = ReadY(d.slots[0], n, 1);
        std::copy(y.begin(), y.end(),
                  rows1.begin() + static_cast<std::ptrdiff_t>(r) * n);
      }
      double num = 0.0;
      double den = 0.0;
      for (std::size_t i = 0; i < rows1.size(); ++i) {
        num += (rows1[i] - ref[i]) * (rows1[i] - ref[i]);
        den += ref[i] * ref[i];
      }
      rel = std::max(rel, std::sqrt(num / den));
      for (int layout : {0, 1}) {
        for (int m : {1, 2, 4, 8}) {
          Pick(c.bits, m, w, tn, layout)(Args(c, d, 0, m, layout), nullptr);
          const std::vector<float> got = ReadY(d.slots[0], n, m);
          ++runs;
          differ += std::memcmp(got.data(), rows1.data(), got.size() * 4) != 0;
        }
      }
    }
    failures += !(rel < 1e-5);
    failures += differ != 0;
    std::printf(
        "check %s K=%d: max rel L2 vs FP64 %.3e; %d/%d (tn, layout, M) "
        "runs differ from M=1 row-major of the same tn\n",
        c.name, c.bits, rel, differ, runs);
    Free(d);
  }
  return failures;
}

// Multi-matrix launches: every slot bit-identical to its own single launch.
int CheckMulti(std::mt19937& rng) {
  int failures = 0;
  const Case c{"q+k+v small", 5120, 5, {2048, 1024, 1024}};
  std::vector<SlotHost> hosts;
  Device d = Allocate(c, 1, rng, &hosts);
  int differ = 0;
  int runs = 0;
  for (int tn : {1, 2}) {
    for (int layout : {0, 1}) {
      for (int m : {1, 8}) {
        const unsigned w = SplitWarps(c.k / 16, Blocks(c, tn));
        std::vector<std::vector<float>> single;
        for (unsigned s = 0; s < c.ns.size(); ++s) {
          Pick(c.bits, m, w, tn, layout)(Args(c, d, 0, m, layout, s, 1),
                                         nullptr);
          single.push_back(ReadY(d.slots[s], c.ns[s], m));
        }
        Pick(c.bits, m, w, tn, layout)(Args(c, d, 0, m, layout), nullptr);
        for (unsigned s = 0; s < c.ns.size(); ++s) {
          const std::vector<float> got = ReadY(d.slots[s], c.ns[s], m);
          ++runs;
          differ +=
              std::memcmp(got.data(), single[s].data(), got.size() * 4) != 0;
        }
      }
    }
  }
  failures += differ != 0;
  std::printf("check %s: %d/%d slot outputs differ from single launches\n",
              c.name, differ, runs);
  Free(d);
  return failures;
}

int Check() {
  std::mt19937 rng(20261030);
  int failures = CheckDecode(rng);
  failures += CheckSingles(rng);
  failures += CheckMulti(rng);
  std::printf("check: %s\n", failures == 0 ? "all passed" : "FAILED");
  return failures == 0 ? 0 : 1;
}

float TimeLaunch(const Case& c, const Device& d, Launcher launch, int m,
                 int layout, int reps) {
  hipEvent_t e0;
  hipEvent_t e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  for (int i = 0; i < 3; ++i)
    launch(Args(c, d, i % d.copies, m, layout), nullptr);
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> times;
  for (int i = 0; i < reps; ++i) {
    HIP_OK(hipEventRecord(e0, nullptr));
    launch(Args(c, d, i % d.copies, m, layout), nullptr);
    HIP_OK(hipEventRecord(e1, nullptr));
    HIP_OK(hipEventSynchronize(e1));
    float ms = 0.0F;
    HIP_OK(hipEventElapsedTime(&ms, e0, e1));
    times.push_back(ms);
  }
  std::sort(times.begin(), times.end());
  HIP_OK(hipEventDestroy(e0));
  HIP_OK(hipEventDestroy(e1));
  return times[times.size() / 2];
}

float TimeRead(const Device& d, const Case& c, int reps) {
  unsigned* sink = nullptr;
  HIP_OK(hipMalloc(&sink, 4));
  hipEvent_t e0;
  hipEvent_t e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  std::vector<float> times;
  for (int i = 0; i < reps + 3; ++i) {
    HIP_OK(hipEventRecord(e0, nullptr));
    for (std::size_t s = 0; s < d.slots.size(); ++s) {
      const SlotDevice& sd = d.slots[s];
      const uint4* data = reinterpret_cast<const uint4*>(
          sd.trellis + sd.copy_words * (i % d.copies));
      hipLaunchKernelGGL(ReadKernel, dim3(2048), dim3(256), 0, 0, data,
                         TrellisBytes(c.k, c.ns[s], c.bits) / 16, sink);
    }
    HIP_OK(hipEventRecord(e1, nullptr));
    HIP_OK(hipEventSynchronize(e1));
    float ms = 0.0F;
    HIP_OK(hipEventElapsedTime(&ms, e0, e1));
    if (i >= 3)
      times.push_back(ms);
  }
  std::sort(times.begin(), times.end());
  HIP_OK(hipEventDestroy(e0));
  HIP_OK(hipEventDestroy(e1));
  HIP_OK(hipFree(sink));
  return times[times.size() / 2];
}

int Time(int reps) {
  std::mt19937 rng(7);
  std::printf("case\tk\tn\tK\tM\twarps\ttn\tlayout\tmedian_us\tGB_s\n");
  for (const Case& c : TimeCases()) {
    const std::size_t bytes = CaseBytes(c);
    const int copies =
        static_cast<int>(std::max<std::size_t>(1, (256U << 20) / bytes + 1));
    std::vector<SlotHost> hosts;
    Device d = Allocate(c, copies, rng, &hosts);
    unsigned n_total = 0;
    for (unsigned n : c.ns)
      n_total += n;
    for (int m : {1, 2, 4, 8}) {
      for (int tn : {1, 2}) {
        for (unsigned w : {4U, 8U, 16U}) {
          for (int layout : {0, 1}) {
            const float ms = TimeLaunch(c, d, Pick(c.bits, m, w, tn, layout), m,
                                        layout, reps);
            std::printf("%s\t%u\t%u\t%d\t%d\t%u\t%d\t%d\t%.1f\t%.1f\n", c.name,
                        c.k, n_total, c.bits, m, w, tn, layout, ms * 1000.0F,
                        bytes / (ms * 1e-3) / 1e9);
          }
        }
      }
    }
    const float ms = TimeRead(d, c, reps);
    std::printf("%s\t%u\t%u\t%d\tread\t-\t-\t-\t%.1f\t%.1f\n", c.name, c.k,
                n_total, c.bits, ms * 1000.0F, bytes / (ms * 1e-3) / 1e9);
    Free(d);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "check";
  if (mode == "check")
    return Check();
  if (mode == "time")
    return Time(argc > 2 ? std::atoi(argv[2]) : 20);
  std::fprintf(stderr, "usage: %s check | time [reps]\n", argv[0]);
  return 2;
}
