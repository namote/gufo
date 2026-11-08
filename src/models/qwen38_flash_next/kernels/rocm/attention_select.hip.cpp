// Sparse-attention indexer: its key-row stores, pooled key blocks and
// per-query block selection.
// HIP kernels follow the layouts and operator formulas in reference.cpp.

#include "src/models/qwen38_flash_next/kernels/rocm/kernels_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

/// Row of pooled indexer block `b` (one row per kKvPooledRatio tokens) in one
/// layer of the pooled-key plane, sharing the token planes' block table.
__device__ __forceinline__ std::size_t PooledRow(const std::uint32_t* kv_blocks,
                                                 std::uint32_t b) {
  constexpr std::uint32_t kRows = hip::kKvBlockTokens / kKvPooledRatio;
  if (kv_blocks == nullptr)
    return b;
  return static_cast<std::size_t>(kv_blocks[b / kRows]) * kRows + b % kRows;
}

__global__ void StoreRowsKernel(const float* src, float* dst,
                                std::uint32_t row_dim,
                                const std::uint32_t* start_pos,
                                std::uint32_t capacity) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t row = (*start_pos + t) & (capacity - 1);
  for (std::uint32_t i = threadIdx.x; i < row_dim; i += blockDim.x) {
    dst[static_cast<std::size_t>(row) * row_dim + i] =
        src[static_cast<std::size_t>(t) * row_dim + i];
  }
}

/// grid: the most blocks a batch can complete. Block b pools raw keys
/// [b*ratio, (b+1)*ratio) once every one of them is stored, i.e. for
/// b in [*first_block, (*start_pos + n_tokens) / ratio).
template<bool kYarn>
__global__ void PoolBlocksKernel(
    const float* raw, const float* gamma, __half* blocks,
    const std::uint32_t* first_block, const std::uint32_t* start_pos,
    std::uint32_t n_tokens, std::uint32_t ratio, std::uint32_t dim,
    std::uint32_t rotary_dim, float theta, float eps, std::uint32_t capacity,
    const qwen::vision::DeviceRope* rope, [[maybe_unused]] YarnRope yarn,
    const std::uint32_t* kv_blocks) {
  __shared__ float v[256];
  __shared__ float shared[32];
  const std::uint32_t b = *first_block + blockIdx.x;
  if (b >= (*start_pos + n_tokens) / ratio) {
    return;
  }
  const std::uint32_t i = threadIdx.x;
  float mean = 0.0f;
  if (i < dim) {
    for (std::uint32_t r = 0; r < ratio; ++r) {
      const std::uint32_t row = (b * ratio + r) & (capacity - 1);
      mean += raw[static_cast<std::size_t>(row) * dim + i];
    }
    mean /= static_cast<float>(ratio);
  }
  const float ss = BlockSum(i < dim ? mean * mean : 0.0f, shared);
  const float scale = rsqrtf(ss / static_cast<float>(dim) + eps);
  if (i < dim) {
    v[i] = mean * scale * gamma[i];
  }
  __syncthreads();
  const std::uint32_t half = rotary_dim / 2;
  float outv = i < dim ? v[i] : 0.0f;
  if (i < rotary_dim) {
    const std::uint32_t p = i < half ? i : i - half;
    float freq = powf(
        theta, -2.0f * static_cast<float>(p) / static_cast<float>(rotary_dim));
    if constexpr (kYarn)
      freq = YarnFrequency(freq, p, yarn);
    float s = 0.0f;
    float c = 0.0f;
    sincosf(qwen::vision::RopePosition(rope, b * ratio, p) * freq, &s, &c);
    if constexpr (kYarn)
      YarnScale(s, c, yarn);
    const float a = v[p];
    const float bb = v[p + half];
    outv = i < half ? a * c - bb * s : a * s + bb * c;
  }
  if (i < dim) {
    blocks[PooledRow(kv_blocks, b) * dim + i] = __float2half_rn(outv);
  }
}

// Keep indexer queries in F32: narrowing them before ranking can swap blocks
// at the selection boundary. A thread scores one key, reusing it across all
// four heads. Preserve the original wave32 F32 accumulation and reduction
// tree: even tiny changes can exchange blocks at the top-k boundary.
constexpr std::uint32_t kSelectHeads = 4;
constexpr std::uint32_t kSelectDim = 128;

__global__ void SelectScoreKernel(const float* q, const __half* blocks,
                                  float* scores, std::uint32_t n_tokens,
                                  const std::uint32_t* start_pos,
                                  std::uint32_t first_token,
                                  std::uint32_t ratio, std::uint32_t budget,
                                  std::uint32_t max_blocks,
                                  const std::uint32_t* kv_blocks) {
#pragma clang fp reassociate(off)
  const auto t0 = blockIdx.x;
  const auto complete = (*start_pos + first_token + t0 + 1) / ratio;
  const auto b = blockIdx.y * blockDim.x + threadIdx.x;
  // The position comes from the device, the row size from the host: a block
  // past max_blocks has no score slot in this row.
  if (t0 >= n_tokens || complete <= budget || b >= complete || b >= max_blocks)
    return;
  // Load both stored halves together. Unpacking preserves their exact F32
  // values while avoiding a separate narrow memory load for each component.
  std::uint32_t key[kSelectDim / 2];
  const std::size_t key_row = PooledRow(kv_blocks, b);
#pragma unroll
  for (unsigned i = 0; i < kSelectDim / 2; ++i)
    key[i] = reinterpret_cast<const std::uint32_t*>(
        blocks)[key_row * kSelectDim / 2 + i];
  float total = 0.0F;
#if __clang_major__ >= 23
  // Newer LLVM versions hoist queries from all four unrolled heads,
  // overflowing the scalar register file. LLVM 23 also spills into private
  // memory. Keep one head live without changing the FMA or reduction order.
#pragma unroll 1
#else
#pragma unroll
#endif
  for (unsigned h = 0; h < kSelectHeads; ++h) {
    __builtin_amdgcn_sched_barrier(0);
    const auto* query = q + (std::size_t{t0} * kSelectHeads + h) * kSelectDim;
    float partial[32];
#pragma unroll
    for (unsigned lane = 0; lane < 32; ++lane) {
      float dot = 0.0F;
#pragma unroll
      for (unsigned i = 0; i < 4; ++i) {
        const auto bits = static_cast<unsigned short>(
            key[(lane + i * 32) / 2] >> ((lane & 1) * 16));
        dot = fmaf(query[lane + i * 32], __half2float(__ushort_as_half(bits)),
                   dot);
      }
      partial[lane] = dot;
      // Bound query-load hoisting: keeping all 128 scalar values live spills
      // registers on gfx1151. Scheduling groups retain every FMA and sum.
      if ((lane + 1) % 16 == 0)
        __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for (unsigned delta = 16; delta; delta >>= 1) {
#pragma unroll
      for (unsigned lane = 0; lane < delta; ++lane)
        partial[lane] += partial[lane + delta];
    }
    total += fmaxf(partial[0], 0.0F);
  }
  scores[std::size_t{t0} * max_blocks + b] = total;
}

/// Locate a descending histogram rank cooperatively over kBins bins with a
/// kBlock-thread block. Each thread owns consecutive bins (threads past the
/// bins own none); state receives the bin, its remaining rank and count.
template<std::uint32_t kBins, std::uint32_t kBlock>
__device__ void FindHistogramThreshold(const std::uint32_t* histogram,
                                       std::uint32_t wanted,
                                       std::uint32_t* wave_sums,
                                       std::uint32_t* state) {
  constexpr std::uint32_t kBinsPerThread = kBins >= kBlock ? kBins / kBlock : 1;
  static_assert(kBins % kBinsPerThread == 0, "bins split evenly");
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t wave = threadIdx.x >> 5;
  const bool owner = threadIdx.x < kBins / kBinsPerThread;
  const std::uint32_t first =
      owner ? kBins - 1 - threadIdx.x * kBinsPerThread : 0;
  std::uint32_t count = 0;
  if (owner) {
#pragma unroll
    for (std::uint32_t j = 0; j < kBinsPerThread; ++j)
      count += histogram[first - j];
  }
  std::uint32_t inclusive = count;
#pragma unroll
  for (std::uint32_t distance = 1; distance < 32; distance *= 2) {
    const std::uint32_t other = __shfl_up(inclusive, distance);
    if (lane >= distance)
      inclusive += other;
  }
  if (lane == 31)
    wave_sums[wave] = inclusive;
  __syncthreads();
  std::uint32_t before = inclusive - count;
  for (std::uint32_t w = 0; w < wave; ++w)
    before += wave_sums[w];
  if (owner && before < wanted && before + count >= wanted) {
    for (std::uint32_t j = 0; j < kBinsPerThread; ++j) {
      const std::uint32_t bin = first - j;
      const std::uint32_t bin_count = histogram[bin];
      if (before + bin_count >= wanted) {
        state[0] = bin;
        state[1] = wanted - before;
        state[2] = bin_count;
        break;
      }
      before += bin_count;
    }
  }
  __syncthreads();
}

/// Exact top-k over non-negative score bits. A sample chooses a histogram
/// window; a clipped threshold falls back to the full key range. At most
/// 4096 candidates are refined in LDS; larger bins use the original scores.
/// Every score participates, and the final mask keeps lowest-index ties.
/// Narrow batches use 1,024 threads per row: a single row's passes over its
/// scores are bound by how much one block keeps in flight.
template<std::uint32_t kBlock>
__launch_bounds__(kBlock) __global__
    void SelectMarkKernel(std::uint32_t* mask, const float* scores,
                          const std::uint32_t* start_pos,
                          std::uint32_t first_token, std::uint32_t ratio,
                          std::uint32_t budget, std::uint32_t mask_words,
                          std::uint32_t max_blocks) {
  __shared__ std::uint32_t candidates[4096];
  __shared__ std::uint32_t hist[256];
  __shared__ std::uint32_t candidate_count;
  __shared__ std::uint32_t wave_ties[kBlock / 32];
  __shared__ std::uint32_t state[3];  // threshold, remaining, bin count
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t pos = *start_pos + first_token + t;
  // Only max_blocks scores exist per row (see SelectScoreKernel).
  const std::uint32_t complete = min((pos + 1) / ratio, max_blocks);
  std::uint32_t* words = mask + static_cast<std::size_t>(t) * mask_words;
  for (std::uint32_t w = threadIdx.x; w < mask_words; w += blockDim.x) {
    words[w] = complete <= budget ? 0xFFFFFFFFu : 0u;
  }
  if (complete <= budget) {
    return;
  }
  const auto* sc = reinterpret_cast<const std::uint32_t*>(
      scores + static_cast<std::size_t>(t) * max_blocks);
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t wave = threadIdx.x >> 5;
  // Short rows already fit the original histogram cheaply.
  const bool windowed = complete >= 4096;
  std::uint32_t maximum = 0;
  if (windowed) {
    maximum = threadIdx.x < min(complete, kThreads) ? sc[threadIdx.x] : 0u;
#pragma unroll
    for (unsigned offset = 16; offset > 0; offset /= 2)
      maximum = max(maximum, XorLane(maximum, offset));
    if (lane == 0)
      wave_ties[wave] = maximum;
    __syncthreads();
    for (unsigned w = 0; w < kBlock / 32; ++w)
      maximum = max(maximum, wave_ties[w]);
  }
  // Finer bins reduce contention among scores with similar exponents.
  // The sample only places the window: clipped tails are still counted.
  unsigned prefix_bits = windowed ? 14 : 20;
  unsigned base =
      (maximum >> prefix_bits) > 2047 ? (maximum >> prefix_bits) - 2047 : 0;
  if (threadIdx.x == 0)
    candidate_count = 0;
  // Visit every score of the row. Aligned rows give each thread four
  // 16-byte loads per step, issued before any of them is used.
  const auto for_each_score = [&](auto&& visit) {
    constexpr std::uint32_t kLoads = 4;
    constexpr std::uint32_t kStep = kLoads * 4;
    for (std::uint32_t b = threadIdx.x * kStep; b < complete;
         b += blockDim.x * kStep) {
      if (b + kStep - 1 < complete && max_blocks % 4 == 0) {
        uint4 v[kLoads];
#pragma unroll
        for (std::uint32_t j = 0; j < kLoads; ++j)
          v[j] = *reinterpret_cast<const uint4*>(sc + b + (j * 4));
#pragma unroll
        for (std::uint32_t j = 0; j < kLoads; ++j) {
          visit(v[j].x);
          visit(v[j].y);
          visit(v[j].z);
          visit(v[j].w);
        }
      } else {
        for (std::uint32_t j = 0; j < kStep && b + j < complete; ++j)
          visit(sc[b + j]);
      }
    }
  };
  for (unsigned attempt = 0; attempt < 2; ++attempt) {
    for (unsigned i = threadIdx.x; i < 4096; i += blockDim.x)
      candidates[i] = 0;
    __syncthreads();
    const auto bin = [&](std::uint32_t value) {
      const auto key = value >> prefix_bits;
      return key > base ? min(key - base, 4095u) : 0u;
    };
    for_each_score(
        [&](std::uint32_t v) { atomicAdd(&candidates[bin(v)], 1u); });
    __syncthreads();
    FindHistogramThreshold<4096, kBlock>(candidates, budget, wave_ties, state);
    if (!windowed || attempt != 0 ||
        (state[0] != 4095 && (state[0] != 0 || base == 0)))
      break;
    // A clipped threshold needs a histogram over the full 32-bit keys.
    prefix_bits = 20;
    base = 0;
  }
  std::uint32_t prefix = (base + state[0]) << prefix_bits;
  std::uint32_t remaining = state[1];
  // Reuse the first histogram as a bounded candidate list. If its bin is
  // larger, later radix passes read the original scores instead.
  const bool compact = state[2] <= 4096 && remaining != state[2];
  const std::uint32_t input_count = compact ? state[2] : complete;
  if (compact) {
    // Candidate order is irrelevant: only their values are refined.
    const auto keep = [&](std::uint32_t v) {
      if ((v & (~0u << prefix_bits)) == prefix)
        candidates[atomicAdd(&candidate_count, 1u)] = v;
    };
    for_each_score(keep);
    __syncthreads();
  }
  // A whole bin can be accepted as soon as its count fills the remaining
  // budget. Otherwise refine the remaining bits, at most eight per pass.
  for (unsigned bits = prefix_bits; bits > 0 && remaining != state[2];) {
    const unsigned digit_bits = min(bits, 8u);
    const std::uint32_t prefix_mask = ~0u << bits;
    bits -= digit_bits;
    const std::uint32_t radix_mask = (1u << digit_bits) - 1;
    if (threadIdx.x < 256)
      hist[threadIdx.x] = 0;
    __syncthreads();
    const auto count_digit = [&](std::uint32_t v) {
      if ((v & prefix_mask) == prefix)
        atomicAdd(&hist[(v >> bits) & radix_mask], 1u);
    };
    if (compact) {
      for (std::uint32_t b = threadIdx.x; b < input_count; b += blockDim.x)
        count_digit(candidates[b]);
    } else {
      for_each_score(count_digit);
    }
    __syncthreads();
    FindHistogramThreshold<256, kBlock>(hist, remaining, wave_ties, state);
    prefix |= state[0] << bits;
    remaining = state[1];
    __syncthreads();
  }
  const std::uint32_t threshold = prefix;
  // Blocks strictly above the threshold are in; the first `remaining` ties
  // in index order fill the budget. Ties are ranked with a wave ballot and
  // a per-chunk scan over the eight waves.
  if (remaining == state[2]) {
    // Every threshold tie fits (in particular, a unique threshold). No
    // prefix scan is needed: each thread writes complete mask words.
    for (std::uint32_t w = threadIdx.x; w * 32 < complete; w += blockDim.x) {
      const std::uint32_t b0 = w * 32;
      std::uint32_t bits = 0;
      if (b0 + 31 < complete && max_blocks % 4 == 0) {
#pragma unroll
        for (std::uint32_t j = 0; j < 32; j += 4) {
          const uint4 v = *reinterpret_cast<const uint4*>(sc + b0 + j);
          bits |= (v.x >= threshold ? 1u : 0u) << j;
          bits |= (v.y >= threshold ? 1u : 0u) << (j + 1);
          bits |= (v.z >= threshold ? 1u : 0u) << (j + 2);
          bits |= (v.w >= threshold ? 1u : 0u) << (j + 3);
        }
      } else {
        for (std::uint32_t j = 0; j < 32 && b0 + j < complete; ++j)
          bits |= (sc[b0 + j] >= threshold ? 1u : 0u) << j;
      }
      words[w] = bits;
    }
    return;
  }
  std::uint32_t ties_before = 0;
  for (std::uint32_t chunk = 0; chunk < complete; chunk += blockDim.x) {
    const std::uint32_t b = chunk + threadIdx.x;
    const std::uint32_t v = b < complete ? sc[b] : 0u;
    const bool tie = b < complete && v == threshold;
    if (b < complete && v > threshold) {
      atomicOr(&words[b / 32], 1u << (b % 32));
    }
    const std::uint64_t ballot = __ballot(tie);
    if (lane == 0) {
      wave_ties[wave] = static_cast<std::uint32_t>(__popcll(ballot));
    }
    __syncthreads();
    std::uint32_t before = ties_before;
    for (std::uint32_t w = 0; w < wave; ++w) {
      before += wave_ties[w];
    }
    before += static_cast<std::uint32_t>(
        __popcll(ballot & ((std::uint64_t{1} << lane) - 1)));
    if (tie && before < remaining) {
      atomicOr(&words[b / 32], 1u << (b % 32));
    }
    std::uint32_t chunk_ties = 0;
    for (std::uint32_t w = 0; w < kBlock / 32; ++w) {
      chunk_ties += wave_ties[w];
    }
    ties_before += chunk_ties;
    __syncthreads();
  }
}

}  // namespace

void StoreRows(const float* src, float* dst, std::uint32_t n_tokens,
               std::uint32_t row_dim, const std::uint32_t* start_pos,
               std::uint32_t capacity, hipStream_t stream) {
  hipLaunchKernelGGL(StoreRowsKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     src, dst, row_dim, start_pos, capacity);
}

void PoolIndexerBlocks(const float* raw_keys, const float* gamma,
                       __half* blocks, const std::uint32_t* first_block,
                       const std::uint32_t* start_pos, std::uint32_t n_tokens,
                       std::uint32_t grid_blocks, std::uint32_t ratio,
                       std::uint32_t dim, std::uint32_t rotary_dim, float theta,
                       float eps, std::uint32_t capacity, hipStream_t stream,
                       const qwen::vision::DeviceRope* rope,
                       const YarnRope& yarn, const std::uint32_t* kv_blocks) {
  if (grid_blocks == 0) {
    return;
  }
  if (yarn.enabled != 0)
    hipLaunchKernelGGL(PoolBlocksKernel<true>, dim3(grid_blocks),
                       dim3(kThreads), 0, stream, raw_keys, gamma, blocks,
                       first_block, start_pos, n_tokens, ratio, dim, rotary_dim,
                       theta, eps, capacity, rope, yarn, kv_blocks);
  else
    hipLaunchKernelGGL(PoolBlocksKernel<false>, dim3(grid_blocks),
                       dim3(kThreads), 0, stream, raw_keys, gamma, blocks,
                       first_block, start_pos, n_tokens, ratio, dim, rotary_dim,
                       theta, eps, capacity, rope, yarn, kv_blocks);
}

void SelectBlocks(const float* q, const __half* blocks, std::uint32_t* mask,
                  float* scores, std::uint32_t n_tokens,
                  const std::uint32_t* start_pos, std::uint32_t first_token,
                  std::uint32_t heads, std::uint32_t dim, std::uint32_t ratio,
                  std::uint32_t budget, std::uint32_t mask_words,
                  std::uint32_t max_blocks, hipStream_t stream,
                  std::uint32_t live_blocks, const std::uint32_t* kv_blocks) {
  if (heads != kSelectHeads || dim != kSelectDim) {
    return;
  }
  // Graph grids are sized by max_blocks so a captured decode graph replays
  // at any position; blocks past the live range return at once. An eager
  // launch passes its live range, sparing the empty workgroups.
  const std::uint32_t grid_blocks =
      live_blocks != 0 ? std::min(live_blocks, max_blocks) : max_blocks;
  const dim3 grid(n_tokens, (grid_blocks + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(SelectScoreKernel, grid, dim3(kThreads), 0, stream, q,
                     blocks, scores, n_tokens, start_pos, first_token, ratio,
                     budget, max_blocks, kv_blocks);
  constexpr std::uint32_t kNarrowMarkThreads = 1024;
  const auto mark = n_tokens <= 4 ? SelectMarkKernel<kNarrowMarkThreads>
                                  : SelectMarkKernel<kThreads>;
  hipLaunchKernelGGL(mark, dim3(n_tokens),
                     dim3(n_tokens <= 4 ? kNarrowMarkThreads : kThreads), 0,
                     stream, mask, scores, start_pos, first_token, ratio,
                     budget, mask_words, max_blocks);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
