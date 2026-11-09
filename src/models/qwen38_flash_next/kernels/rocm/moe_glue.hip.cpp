// MoE glue around the expert GEMMs: router top-k, expert counts and the
// expert-output epilogues.
// HIP kernels follow the layouts and operator formulas in reference.cpp.

#include "src/models/qwen38_flash_next/kernels/rocm/kernels_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

// Keep the original block-wide softmax reduction. Selection then stays in
// one wave's registers, retaining probability ordering and lowest-index ties.
// Threads past the first wave return early.
template<unsigned MaxExperts>
__device__ __forceinline__ void RouterTopKBody(
    const float* logits, std::uint32_t stride, std::int32_t* ids,
    float* weights, std::uint32_t n_experts, std::uint32_t k) {
  __shared__ float probs[1024];
  __shared__ float shared[32];
  __shared__ std::uint32_t chosen[32];
  __shared__ float chosen_p[32];
  const std::uint32_t t = blockIdx.x;
  const float* src = logits + static_cast<std::size_t>(t) * stride;
  float local_max = -INFINITY;
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    local_max = fmaxf(local_max, src[e]);
  }
  const float max_logit = BlockMax(local_max, shared);
  float local_sum = 0.0f;
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    probs[e] = __expf(src[e] - max_logit);
    local_sum += probs[e];
  }
  const float denom = BlockSum(local_sum, shared);
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    probs[e] /= denom;
  }
  __syncthreads();

  if (threadIdx.x >= 32)
    return;
  constexpr unsigned Items = MaxExperts / 32;
  float values[Items];
#pragma unroll
  for (unsigned j = 0; j < Items; ++j) {
    unsigned e = threadIdx.x + j * 32;
    values[j] = e < n_experts ? probs[e] : -1.0f;
  }
  for (unsigned slot = 0; slot < k; ++slot) {
    float best = -1.0f;
    unsigned index = 0xffffffffu;
#pragma unroll
    for (unsigned j = 0; j < Items; ++j) {
      unsigned e = threadIdx.x + j * 32;
      if (values[j] > best) {
        best = values[j];
        index = e;
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other = XorLane(best, offset);
      unsigned oi = XorLane(index, offset);
      if (other > best || (other == best && oi < index)) {
        best = other;
        index = oi;
      }
    }
    // Every unused probability is NaN (any NaN or +Inf logit, or all -Inf):
    // take the lowest unused expert at weight 0, so no ID reaches past
    // n_experts.
    // The reduction above leaves the same index in every lane, and finite
    // rows never enter this branch.
    if (index == 0xffffffffu) {
      best = 0.0f;
#pragma unroll
      for (unsigned j = Items; j-- > 0;) {
        unsigned e = threadIdx.x + j * 32;
        if (e < n_experts && !(values[j] == -1.0f))
          index = e;
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        index = min(index, XorLane(index, offset));
    }
    if (threadIdx.x == 0) {
      chosen[slot] = index;
      chosen_p[slot] = best;
    }
#pragma unroll
    for (unsigned j = 0; j < Items; ++j) {
      if (threadIdx.x + j * 32 == index)
        values[j] = -1.0f;
    }
  }
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (std::uint32_t slot = 0; slot < k; ++slot) {
      sum += chosen_p[slot];
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (std::uint32_t slot = 0; slot < k; ++slot) {
      ids[t * k + slot] = static_cast<std::int32_t>(chosen[slot]);
      weights[t * k + slot] = chosen_p[slot] / sum;
    }
  }
}

template<unsigned MaxExperts>
__global__ void RouterTopKKernel(const float* logits, std::uint32_t stride,
                                 std::int32_t* ids, float* weights,
                                 std::uint32_t n_experts, std::uint32_t k) {
  RouterTopKBody<MaxExperts>(logits, stride, ids, weights, n_experts, k);
}

/// RouterTopKKernel, then the last block to finish groups all of the grid's
/// IDs into BuildTiles' tiles with its first wave: per expert in assignment
/// order, a new tile every tile_rows assignments. Tile positions follow the
/// leading assignment's order instead of BuildTiles' atomic order; every
/// tile still covers the same rows and writes disjoint outputs.
template<unsigned MaxExperts>
__global__ void RouterTopKTilesKernel(const float* logits, std::uint32_t stride,
                                      std::int32_t* ids, float* weights,
                                      std::uint32_t n_experts, std::uint32_t k,
                                      Exl3RouteTile* tiles,
                                      std::uint32_t* tile_count,
                                      std::uint32_t* arrivals,
                                      std::uint32_t tile_rows) {
  RouterTopKBody<MaxExperts>(logits, stride, ids, weights, n_experts, k);
  if (threadIdx.x >= 32)
    return;
  // Thread 0 wrote this block's IDs; publish them before arriving. The
  // counter wraps to zero on the last arrival, ready for the next launch.
  unsigned last = 0;
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicInc(arrivals, gridDim.x - 1) == gridDim.x - 1;
  }
  if (__shfl(last, 0, 32) == 0)
    return;
  __threadfence();
  constexpr unsigned kChunks = kExl3FusedTileAssignments / 32;
  const unsigned lane = threadIdx.x;
  const unsigned n = gridDim.x * k;
  // Lane l holds assignments l + 32 c in registers; each pass broadcasts
  // every ID in assignment order (fixed chunk indices, no local array).
  int mine[kChunks];
  unsigned rank[kChunks], total[kChunks];
#pragma unroll
  for (unsigned c = 0; c < kChunks; ++c) {
    const unsigned i = lane + c * 32;
    mine[c] = i < n ? __hip_atomic_load(ids + i, __ATOMIC_RELAXED,
                                        __HIP_MEMORY_SCOPE_AGENT)
                    : -1;
    rank[c] = 0;
    total[c] = 0;
  }
#pragma unroll
  for (unsigned cj = 0; cj < kChunks; ++cj) {
    for (unsigned l = 0; l < 32 && cj * 32 + l < n; ++l) {
      const int e = __shfl(mine[cj], l, 32);
      const unsigned j = cj * 32 + l;
#pragma unroll
      for (unsigned c = 0; c < kChunks; ++c) {
        if (mine[c] == e) {
          ++total[c];
          rank[c] += j < lane + c * 32 ? 1U : 0U;
        }
      }
    }
  }
  // A tile starts at every tile_rows-th assignment of its expert; leaders
  // take consecutive positions in assignment order.
  unsigned base = 0, position[kChunks], count[kChunks], filled[kChunks];
#pragma unroll
  for (unsigned c = 0; c < kChunks; ++c) {
    const bool leader = mine[c] >= 0 && rank[c] % tile_rows == 0;
    const auto ballot = static_cast<std::uint32_t>(__ballot(leader));
    position[c] = leader ? base + __popc(ballot & ((1U << lane) - 1U)) : ~0U;
    count[c] = leader ? min(tile_rows, total[c] - rank[c]) : 0U;
    filled[c] = 0;
    base += __popc(ballot);
  }
  // Each leader takes the next `count` assignments of its expert, itself
  // first, in assignment order.
#pragma unroll
  for (unsigned cj = 0; cj < kChunks; ++cj) {
    for (unsigned l = 0; l < 32 && cj * 32 + l < n; ++l) {
      const int e = __shfl(mine[cj], l, 32);
      const unsigned j = cj * 32 + l;
#pragma unroll
      for (unsigned c = 0; c < kChunks; ++c) {
        if (mine[c] == e && j >= lane + c * 32 && filled[c] < count[c]) {
          tiles[position[c]].slots[filled[c]] = j;
          ++filled[c];
        }
      }
    }
  }
#pragma unroll
  for (unsigned c = 0; c < kChunks; ++c) {
    if (count[c] != 0) {
      Exl3RouteTile& tile = tiles[position[c]];
      tile.expert = static_cast<std::uint32_t>(mine[c]);
      tile.count = count[c];
      for (unsigned r = count[c]; r < 8; ++r)
        tile.slots[r] = 0;
    }
  }
  if (lane == 0)
    *tile_count = base;
}

/// grid (tokens, dim chunks of kThreads). Up to kMaxSlots slots, every read
/// is issued before the first product (a slot loop waits out each load in
/// turn), and the sum still runs slot by slot from zero.
__global__ void MoeEpilogueKernel(const float* expert_out, const float* weights,
                                  const float* shared_out, const float* gate,
                                  std::uint32_t gate_stride, float* out,
                                  std::uint32_t k, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= dim) {
    return;
  }
  constexpr std::uint32_t kMaxSlots = 16;
  if (k <= kMaxSlots) {
    const float* rows = expert_out + static_cast<std::size_t>(t) * k * dim + i;
    float w[kMaxSlots];
    float v[kMaxSlots];
#pragma unroll
    for (std::uint32_t s = 0; s < kMaxSlots; ++s) {
      w[s] = s < k ? weights[t * k + s] : 0.0f;
      v[s] = s < k ? rows[static_cast<std::size_t>(s) * dim] : 0.0f;
    }
    const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
    const float sh = shared_out[idx];
    const float g = gate[static_cast<std::size_t>(t) * gate_stride];
    float acc = Materialize(0.0f);
#pragma unroll
    for (std::uint32_t s = 0; s < kMaxSlots; ++s) {
      if (s < k) {
        acc = Materialize(acc + w[s] * v[s]);
      }
    }
    out[idx] = acc + SigmoidF(g) * sh;
    return;
  }
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < k; ++s) {
    acc += weights[t * k + s] *
           expert_out[(static_cast<std::size_t>(t) * k + s) * dim + i];
  }
  const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
  out[idx] = acc + SigmoidF(gate[static_cast<std::size_t>(t) * gate_stride]) *
                       shared_out[idx];
}

/// MoeEpilogueKernel's k <= 16 arithmetic, element for element (the same
/// Materialize-bounded chain and final expression), for four adjacent
/// elements per thread with 16-byte loads and stores (dim % 4 == 0, aligned).
__global__ void MoeEpilogueExact4Kernel(const float* expert_out,
                                        const float* weights,
                                        const float* shared_out,
                                        const float* gate,
                                        std::uint32_t gate_stride, float* out,
                                        std::uint32_t k, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  if (i >= dim) {
    return;
  }
  constexpr std::uint32_t kMaxSlots = 16;
  const float* rows = expert_out + static_cast<std::size_t>(t) * k * dim + i;
  float w[kMaxSlots];
  float4 v[kMaxSlots];
#pragma unroll
  for (std::uint32_t s = 0; s < kMaxSlots; ++s) {
    w[s] = s < k ? weights[t * k + s] : 0.0f;
    v[s] = s < k ? *reinterpret_cast<const float4*>(
                       rows + static_cast<std::size_t>(s) * dim)
                 : float4{0.0f, 0.0f, 0.0f, 0.0f};
  }
  const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
  const float4 sh4 = *reinterpret_cast<const float4*>(shared_out + idx);
  const float g = gate[static_cast<std::size_t>(t) * gate_stride];
  float result[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float sh = e == 0 ? sh4.x : e == 1 ? sh4.y : e == 2 ? sh4.z : sh4.w;
    float acc = Materialize(0.0f);
#pragma unroll
    for (std::uint32_t s = 0; s < kMaxSlots; ++s) {
      if (s < k) {
        const float vs = e == 0   ? v[s].x
                         : e == 1 ? v[s].y
                         : e == 2 ? v[s].z
                                  : v[s].w;
        acc = Materialize(acc + w[s] * vs);
      }
    }
    result[e] = acc + SigmoidF(g) * sh;
  }
  *reinterpret_cast<float4*>(out + idx) =
      float4{result[0], result[1], result[2], result[3]};
}

/// Four adjacent output lanes per thread: the same reduction over the top-k
/// slots with a quarter of the waves, so the per-wave issue overhead no
/// longer hides the streaming reads.
/// `ExpertT` is float, or __half when the routed down projection wrote its
/// rows as F16.
template<typename ExpertT>
__global__ void MoeEpilogueVec4Kernel(const ExpertT* expert_out,
                                      const float* weights,
                                      const float* shared_out,
                                      const float* gate,
                                      std::uint32_t gate_stride, float* out,
                                      std::uint32_t k, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  if (i >= dim) {
    return;
  }
  float4 acc{0.0F, 0.0F, 0.0F, 0.0F};
  const ExpertT* rows = expert_out + static_cast<std::size_t>(t) * k * dim + i;
  for (std::uint32_t s = 0; s < k; ++s) {
    const float w = weights[t * k + s];
    const float4 v = Load4(rows + static_cast<std::size_t>(s) * dim);
    acc.x += w * v.x;
    acc.y += w * v.y;
    acc.z += w * v.z;
    acc.w += w * v.w;
  }
  const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
  const float g = SigmoidF(gate[static_cast<std::size_t>(t) * gate_stride]);
  const float4 sh = *reinterpret_cast<const float4*>(shared_out + idx);
  acc.x += g * sh.x;
  acc.y += g * sh.y;
  acc.z += g * sh.z;
  acc.w += g * sh.w;
  *reinterpret_cast<float4*>(out + idx) = acc;
}

}  // namespace

__global__ void ExpertCountsKernel(const std::int32_t* ids,
                                   std::uint32_t* counts, std::size_t slots) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < slots && ids[i] >= 0) {
    atomicAdd(counts + ids[i], 1u);
  }
}

void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t n_tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream) {
  if (n_experts <= 512) {
    hipLaunchKernelGGL((RouterTopKKernel<512>), dim3(n_tokens), dim3(kThreads),
                       0, stream, logits, stride, ids, weights, n_experts, k);
  } else {
    hipLaunchKernelGGL((RouterTopKKernel<1024>), dim3(n_tokens), dim3(kThreads),
                       0, stream, logits, stride, ids, weights, n_experts, k);
  }
}

void RouterTopKTiles(const float* logits, std::uint32_t stride,
                     std::int32_t* ids, float* weights, std::uint32_t n_tokens,
                     std::uint32_t n_experts, std::uint32_t k,
                     Exl3RouteTile* tiles, std::uint32_t* tile_count,
                     std::uint32_t* arrivals, std::uint32_t tile_rows,
                     hipStream_t stream) {
  if (n_tokens == 0 ||
      std::uint64_t{n_tokens} * k > kExl3FusedTileAssignments ||
      tile_rows == 0 || tile_rows > 8 || arrivals == nullptr)
    throw std::invalid_argument("fused router tiles: unsupported call");
  const auto kernel = n_experts <= 512 ? RouterTopKTilesKernel<512>
                                       : RouterTopKTilesKernel<1024>;
  hipLaunchKernelGGL(kernel, dim3(n_tokens), dim3(kThreads), 0, stream, logits,
                     stride, ids, weights, n_experts, k, tiles, tile_count,
                     arrivals, tile_rows);
}

void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t n_tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream) {
  (void)hipMemsetAsync(counts, 0, n_experts * sizeof(std::uint32_t), stream);
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  hipLaunchKernelGGL(ExpertCountsKernel, dim3(Blocks(slots)), dim3(kThreads), 0,
                     stream, ids, counts, slots);
}

void MoeEpilogue(const float* expert_out, const float* weights,
                 const float* shared, const float* gate,
                 std::uint32_t gate_stride, float* out, std::uint32_t n_tokens,
                 std::uint32_t k, std::uint32_t dim, hipStream_t stream) {
  // Same values: the vector kernel repeats the scalar arithmetic per element.
  const bool aligned = (reinterpret_cast<std::uintptr_t>(expert_out) |
                        reinterpret_cast<std::uintptr_t>(shared) |
                        reinterpret_cast<std::uintptr_t>(out)) %
                           16 ==
                       0;
  if (k <= 16 && dim % 4 == 0 && aligned) {
    hipLaunchKernelGGL(MoeEpilogueExact4Kernel, dim3(n_tokens, Blocks(dim / 4)),
                       dim3(kThreads), 0, stream, expert_out, weights, shared,
                       gate, gate_stride, out, k, dim);
    return;
  }
  hipLaunchKernelGGL(MoeEpilogueKernel, dim3(n_tokens, Blocks(dim)),
                     dim3(kThreads), 0, stream, expert_out, weights, shared,
                     gate, gate_stride, out, k, dim);
}

void MoeEpilogueVec4(const float* expert_out, const float* weights,
                     const float* shared, const float* gate,
                     std::uint32_t gate_stride, float* out,
                     std::uint32_t n_tokens, std::uint32_t k, std::uint32_t dim,
                     hipStream_t stream) {
  if (dim % 4 != 0) {
    MoeEpilogue(expert_out, weights, shared, gate, gate_stride, out, n_tokens,
                k, dim, stream);
    return;
  }
  hipLaunchKernelGGL(MoeEpilogueVec4Kernel<float>,
                     dim3(n_tokens, Blocks(dim / 4)), dim3(kThreads), 0, stream,
                     expert_out, weights, shared, gate, gate_stride, out, k,
                     dim);
}

void MoeEpilogueVec4F16(const __half* expert_out, const float* weights,
                        const float* shared, const float* gate,
                        std::uint32_t gate_stride, float* out,
                        std::uint32_t n_tokens, std::uint32_t k,
                        std::uint32_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(MoeEpilogueVec4Kernel<__half>,
                     dim3(n_tokens, Blocks(dim / 4)), dim3(kThreads), 0, stream,
                     expert_out, weights, shared, gate, gate_stride, out, k,
                     dim);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
