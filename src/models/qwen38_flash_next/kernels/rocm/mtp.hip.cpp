// MTP head glue (hidden, embedding add), greedy and penalized argmax, draft
// logit masking and gathering, and MTP/verification candidate selection.
// HIP kernels follow the layouts and operator formulas in reference.cpp.

#include <hipcub/block/block_radix_sort.hpp>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

/// dst[t] = row < 0 ? alt[t] : base[(row + t)]: the draft block's hidden
/// input, a kept trunk row or its own carried residual.
__global__ void MtpHiddenKernel(const float* base, const float* alt,
                                const std::int32_t* row, float* dst,
                                std::uint32_t width) {
  const std::uint32_t t = blockIdx.x;
  const float* src = *row < 0
                         ? alt + static_cast<std::size_t>(t) * width
                         : base + (static_cast<std::size_t>(*row) + t) * width;
  for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    dst[static_cast<std::size_t>(t) * width + i] = src[i];
  }
}

/// MtpHiddenKernel in 16-byte pieces over grid (tokens, chunks of kThreads).
__global__ void MtpHiddenVec4Kernel(const float4* base, const float4* alt,
                                    const std::int32_t* row, float4* dst,
                                    std::uint32_t width4) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= width4) {
    return;
  }
  const float4* src =
      *row < 0 ? alt + static_cast<std::size_t>(t) * width4
               : base + (static_cast<std::size_t>(*row) + t) * width4;
  dst[static_cast<std::size_t>(t) * width4 + i] = src[i];
}

__global__ void MtpAddEmbeddingKernel(const float* embedding, float* residual,
                                      std::uint32_t hidden,
                                      std::uint32_t streams) {
  const std::uint32_t t = blockIdx.x;
  for (std::uint32_t i = threadIdx.x; i < streams * hidden; i += blockDim.x) {
    residual[static_cast<std::size_t>(t) * streams * hidden + i] +=
        embedding[static_cast<std::size_t>(t) * hidden + i % hidden];
  }
}

/// MtpAddEmbeddingKernel in 16-byte pieces over grid (tokens, chunks of
/// kThreads): the same one add per element. `hidden4` is hidden / 4.
__global__ void MtpAddEmbeddingVec4Kernel(const float4* embedding,
                                          float4* residual,
                                          std::uint32_t hidden4,
                                          std::uint32_t streams) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= streams * hidden4) {
    return;
  }
  const float4 a =
      embedding[static_cast<std::size_t>(t) * hidden4 + i % hidden4];
  float4& r = residual[static_cast<std::size_t>(t) * streams * hidden4 + i];
  float4 v = r;
  v.x += a.x;
  v.y += a.y;
  v.z += a.z;
  v.w += a.w;
  r = v;
}

__device__ __forceinline__ ArgmaxCandidate BetterCandidate(ArgmaxCandidate a,
                                                           ArgmaxCandidate b) {
  return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
}

__device__ ArgmaxCandidate ArgmaxBlock(ArgmaxCandidate best,
                                       ArgmaxCandidate* shared) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned wave = threadIdx.x >> 5u;
  for (int offset = 16; offset > 0; offset >>= 1) {
    best = BetterCandidate(
        best, {XorLane(best.value, offset), XorLane(best.index, offset)});
  }
  if (lane == 0) {
    shared[wave] = best;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (unsigned w = 1; w < kThreads / 32; ++w) {
      best = BetterCandidate(best, shared[w]);
    }
  }
  return best;
}

__global__ void ArgmaxPartialKernel(const float* logits,
                                    ArgmaxCandidate* partial,
                                    std::uint32_t vocab) {
  __shared__ ArgmaxCandidate shared[kThreads / 32];
  const std::uint32_t t = blockIdx.y;
  const float* src = logits + static_cast<std::size_t>(t) * vocab;
  ArgmaxCandidate best{-INFINITY, INT32_MAX};
  for (std::uint32_t i = blockIdx.x * kThreads + threadIdx.x; i < vocab;
       i += kArgmaxParts * kThreads) {
    best = BetterCandidate(best, {src[i], static_cast<std::int32_t>(i)});
  }
  best = ArgmaxBlock(best, shared);
  if (threadIdx.x == 0) {
    partial[t * kArgmaxParts + blockIdx.x] = best;
  }
}

__global__ void ArgmaxFinishKernel(const float* logits,
                                   const ArgmaxCandidate* partial,
                                   std::int32_t* out, std::uint32_t vocab) {
  __shared__ ArgmaxCandidate shared[kThreads / 32];
  const std::uint32_t t = blockIdx.x;
  ArgmaxCandidate best{-INFINITY, INT32_MAX};
  if (threadIdx.x < kArgmaxParts) {
    best = partial[t * kArgmaxParts + threadIdx.x];
  }
  best = ArgmaxBlock(best, shared);
  if (threadIdx.x == 0) {
    // std::max_element keeps element zero when it is NaN; later NaNs
    // never replace a finite candidate.
    out[t] =
        isnan(logits[static_cast<std::size_t>(t) * vocab]) ? 0 : best.index;
  }
}

__device__ __forceinline__ PenaltyArgmaxCandidate
BetterPenaltyCandidate(PenaltyArgmaxCandidate a, PenaltyArgmaxCandidate b) {
  return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
}

__device__ PenaltyArgmaxCandidate PenaltyArgmaxBlock(
    PenaltyArgmaxCandidate best, PenaltyArgmaxCandidate* shared) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned wave = threadIdx.x >> 5u;
  for (int offset = 16; offset > 0; offset >>= 1)
    best = BetterPenaltyCandidate(
        best, {__shfl_xor(best.value, offset), __shfl_xor(best.index, offset)});
  if (lane == 0)
    shared[wave] = best;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (unsigned w = 1; w < kThreads / 32; ++w)
      best = BetterPenaltyCandidate(best, shared[w]);
  }
  return best;
}

__global__ void PenaltyArgmaxPartialKernel(const float* logits,
                                           GreedyPenaltyRows penalties,
                                           float repeat, float frequency,
                                           float presence,
                                           PenaltyArgmaxCandidate* partial,
                                           std::uint32_t vocab) {
  __shared__ PenaltyArgmaxCandidate shared[kThreads / 32];
  const unsigned row = blockIdx.y;
  const auto* src = logits + std::size_t(row) * vocab;
  PenaltyArgmaxCandidate best{-INFINITY, INT32_MAX};
  for (unsigned token = blockIdx.x * kThreads + threadIdx.x; token < vocab;
       token += kArgmaxParts * kThreads) {
    if (!isfinite(src[token]))
      continue;
    auto begin = penalties.offsets[row];
    auto end = penalties.offsets[row + 1];
    while (begin < end) {
      const auto mid = begin + (end - begin) / 2;
      if (penalties.penalties[mid].token < token)
        begin = mid + 1;
      else
        end = mid;
    }
    double value = src[token];
    if (begin < penalties.offsets[row + 1] &&
        penalties.penalties[begin].token == token) {
      const auto penalty = penalties.penalties[begin];
      // Preserve the CPU's FP64 operations. In particular, do not narrow
      // the adjusted logit before comparing close candidates.
      if (penalty.repeated && repeat != 1.0F)
        value = value <= 0 ? __dmul_rn(value, double(repeat))
                           : __ddiv_rn(value, double(repeat));
      value = __dadd_rn(value, -__dmul_rn(double(frequency),
                                          double(penalty.generated_count)));
      if (penalty.generated_count != 0)
        value = __dadd_rn(value, -double(presence));
    }
    if (penalties.han_tokens != nullptr &&
        ((penalties.han_tokens[token / 32U] >> (token % 32U)) & 1U))
      value = __dadd_rn(value, penalties.han_bias[row]);
    best =
        BetterPenaltyCandidate(best, {value, static_cast<std::int32_t>(token)});
  }
  best = PenaltyArgmaxBlock(best, shared);
  if (threadIdx.x == 0)
    partial[row * kArgmaxParts + blockIdx.x] = best;
}

__global__ void PenaltyArgmaxFinishKernel(const float* logits,
                                          const PenaltyArgmaxCandidate* partial,
                                          ArgmaxCandidate* out,
                                          std::uint32_t vocab) {
  __shared__ PenaltyArgmaxCandidate shared[kThreads / 32];
  const auto row = blockIdx.x;
  PenaltyArgmaxCandidate best{-INFINITY, INT32_MAX};
  if (threadIdx.x < kArgmaxParts)
    best = partial[row * kArgmaxParts + threadIdx.x];
  best = PenaltyArgmaxBlock(best, shared);
  if (threadIdx.x == 0)
    out[row] =
        best.index < vocab
            ? ArgmaxCandidate{logits[std::size_t(row) * vocab + best.index],
                              best.index}
            : ArgmaxCandidate{NAN, -1};
}

constexpr unsigned kMtpCandidateTile = 1024;

__global__ void GatherArgmaxCandidatesKernel(const float* logits,
                                             const std::uint32_t* ids,
                                             ArgmaxCandidate* out,
                                             std::uint32_t rows,
                                             std::uint32_t vocab) {
  const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    const auto id = ids[row];
    out[row] = id < vocab
                   ? ArgmaxCandidate{logits[std::size_t(row) * vocab + id],
                                     static_cast<std::int32_t>(id)}
                   : ArgmaxCandidate{NAN, -1};
  }
}

// A token excluded from its tile's top Keep cannot enter the global top Keep.
// Reduce tiles repeatedly, retaining the original order of equal scores.
// Only token IDs need to survive between passes; the original logit buffer
// supplies scores and preserves the sign of zero in the final output.
template<unsigned Keep>
__global__ void MtpCandidateTileKernel(const float* logits,
                                       const std::uint32_t* input,
                                       std::uint32_t* output, float* scores,
                                       std::uint32_t size,
                                       std::uint32_t vocab) {
  using Sort = hipcub::BlockRadixSort<float, kThreads, 4, std::uint32_t>;
  __shared__ Sort::TempStorage scratch;
  float keys[4];
  std::uint32_t ids[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const std::size_t i =
        std::size_t(blockIdx.x) * kMtpCandidateTile + threadIdx.x * 4 + j;
    const auto id = i < size ? (input != nullptr ? input[i] : i) : UINT32_MAX;
    ids[j] = static_cast<std::uint32_t>(id);
    const float value = id < vocab ? logits[id] : -INFINITY;
    keys[j] = isfinite(value) ? (value == 0.0F ? 0.0F : value) : -INFINITY;
  }
  Sort(scratch).SortDescending(keys, ids);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const unsigned rank = threadIdx.x * 4 + j;
    if (rank < min(Keep, size)) {
      output[blockIdx.x * Keep + rank] = ids[j];
      if (scores != nullptr) {
        const float value = ids[j] < vocab ? logits[ids[j]] : -INFINITY;
        scores[rank] = isfinite(value) ? value : -INFINITY;
      }
    }
  }
}

template<bool First>
__global__ void VerificationCandidateTileKernel(
    const float* logits, GreedyPenaltyRows penalties,
    const std::uint32_t* input, std::uint32_t* output, std::uint32_t size,
    std::uint32_t vocab, std::uint32_t stride) {
  using Sort = hipcub::BlockRadixSort<float, kThreads, 4, std::uint32_t>;
  __shared__ Sort::TempStorage scratch;
  __shared__ unsigned excluded[kMtpCandidateTile / 32];
  __shared__ unsigned first_penalty;
  const unsigned row = blockIdx.y;
  const auto begin = blockIdx.x * kMtpCandidateTile;
  if constexpr (First) {
    if (threadIdx.x < kMtpCandidateTile / 32)
      excluded[threadIdx.x] = 0;
    if (threadIdx.x == 0) {
      auto lo = penalties.offsets[row];
      auto hi = penalties.offsets[row + 1];
      while (lo < hi) {
        const auto mid = lo + (hi - lo) / 2;
        if (penalties.penalties[mid].token < begin)
          lo = mid + 1;
        else
          hi = mid;
      }
      first_penalty = lo;
    }
    __syncthreads();
    for (auto i = first_penalty + threadIdx.x; i < penalties.offsets[row + 1];
         i += blockDim.x) {
      const auto token = penalties.penalties[i].token;
      if (token >= begin + kMtpCandidateTile)
        break;
      const auto offset = token - begin;
      atomicOr(&excluded[offset / 32], 1U << (offset % 32));
    }
    __syncthreads();
  }
  float keys[4];
  std::uint32_t ids[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const auto offset = threadIdx.x * 4 + j;
    const auto i = begin + offset;
    auto id = i < size ? (First ? i : input[std::size_t(row) * stride + i])
                       : UINT32_MAX;
    if constexpr (First) {
      if ((excluded[offset / 32] >> (offset % 32)) & 1U)
        id = UINT32_MAX;
    }
    float value =
        id < vocab ? logits[std::size_t(row) * vocab + id] : -INFINITY;
    // Every pass ranks the same key: the logit plus its Han offset.
    if (penalties.han_tokens != nullptr && id < vocab &&
        ((penalties.han_tokens[id / 32U] >> (id % 32U)) & 1U))
      value += static_cast<float>(penalties.han_bias[row]);
    // Drop the ID too: a subsequent pass must never resurrect an excluded
    // or nonfinite entry by reading its original logit.
    ids[j] = isfinite(value) ? id : UINT32_MAX;
    keys[j] = isfinite(value) ? (value == 0.0F ? 0.0F : value) : -INFINITY;
  }
  Sort(scratch).SortDescending(keys, ids);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const auto rank = threadIdx.x * 4 + j;
    if (rank < kMtpCandidates)
      output[std::size_t(row) * stride + blockIdx.x * kMtpCandidates + rank] =
          ids[j];
  }
}

__global__ void GatherVerificationCandidatesKernel(
    const float* logits, GreedyPenaltyRows penalties, const std::uint32_t* ids,
    sampling::LogitCandidate* out, std::uint32_t vocab, std::uint32_t stride) {
  const auto row = blockIdx.x;
  const auto count =
      kMtpCandidates + penalties.offsets[row + 1] - penalties.offsets[row];
  auto* destination = out + row * kMtpCandidates + penalties.offsets[row];
  for (unsigned i = threadIdx.x; i < count; i += blockDim.x) {
    const auto id =
        i < kMtpCandidates
            ? ids[std::size_t(row) * stride + i]
            : penalties.penalties[penalties.offsets[row] + i - kMtpCandidates]
                  .token;
    destination[i] =
        id < vocab
            ? sampling::LogitCandidate{id,
                                       logits[std::size_t(row) * vocab + id]}
            : sampling::LogitCandidate{UINT32_MAX, -INFINITY};
  }
}

/// ScoreLogprobs: one block per row. Pass 1 takes the row maximum and flags
/// NaN or +inf; pass 2 sums exp(x - max) in FP64 from FP32 exponentials, so
/// the sum carries only the exponentials' rounding (about 1e-7 relative),
/// not the order of about 250k additions. The target term is formed in FP64
/// from the original logits, as sampling::CollectLogprobs does.
__global__ void ScoreLogprobsKernel(const float* logits,
                                    const std::int32_t* targets,
                                    ScoredLogprob* out, std::uint32_t vocab) {
  __shared__ float shared_max[kThreads / 32];
  __shared__ unsigned shared_bad[kThreads / 32];
  __shared__ double shared_sum[kThreads / 32];
  const std::uint32_t r = blockIdx.x;
  const float* row = logits + static_cast<std::size_t>(r) * vocab;
  const unsigned lane = threadIdx.x & 31u;
  const unsigned wave = threadIdx.x >> 5u;
  float maximum = -INFINITY;
  unsigned bad = 0;
  for (std::uint32_t i = threadIdx.x; i < vocab; i += kThreads) {
    const float x = row[i];
    bad |= (isnan(x) || x == INFINITY) ? 1u : 0u;
    maximum = fmaxf(maximum, x);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = fmaxf(maximum, XorLane(maximum, offset));
    bad |= XorLane(bad, offset);
  }
  if (lane == 0) {
    shared_max[wave] = maximum;
    shared_bad[wave] = bad;
  }
  __syncthreads();
  maximum = shared_max[0];
  bad = shared_bad[0];
  for (unsigned w = 1; w < kThreads / 32; ++w) {
    maximum = fmaxf(maximum, shared_max[w]);
    bad |= shared_bad[w];
  }
  double sum = 0.0;
  for (std::uint32_t i = threadIdx.x; i < vocab; i += kThreads)
    sum += static_cast<double>(expf(row[i] - maximum));
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_xor(sum, offset);
  if (lane == 0)
    shared_sum[wave] = sum;
  __syncthreads();
  if (threadIdx.x != 0)
    return;
  sum = 0.0;
  for (unsigned w = 0; w < kThreads / 32; ++w)
    sum += shared_sum[w];
  const std::int32_t target = targets[r];
  const bool valid = bad == 0 && isfinite(maximum) && target >= 0 &&
                     static_cast<std::uint32_t>(target) < vocab;
  out[r].nonfinite = valid ? 0u : 1u;
  out[r].logprob =
      valid
          ? (static_cast<double>(row[target]) - static_cast<double>(maximum)) -
                log(sum)
          : 0.0;
}

}  // namespace

void MtpHidden(const float* base, const float* alt, const std::int32_t* row,
               float* dst, std::uint32_t n_tokens, std::uint32_t width,
               hipStream_t stream) {
  const bool aligned = (reinterpret_cast<std::uintptr_t>(base) |
                        reinterpret_cast<std::uintptr_t>(alt) |
                        reinterpret_cast<std::uintptr_t>(dst)) %
                           16 ==
                       0;
  if (aligned && width % 4 == 0) {
    hipLaunchKernelGGL(MtpHiddenVec4Kernel, dim3(n_tokens, Blocks(width / 4)),
                       dim3(kThreads), 0, stream,
                       reinterpret_cast<const float4*>(base),
                       reinterpret_cast<const float4*>(alt), row,
                       reinterpret_cast<float4*>(dst), width / 4);
    return;
  }
  hipLaunchKernelGGL(MtpHiddenKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     base, alt, row, dst, width);
}

void MtpAddEmbedding(const float* embedding, float* residual,
                     std::uint32_t n_tokens, std::uint32_t hidden,
                     std::uint32_t streams, hipStream_t stream) {
  const bool aligned = (reinterpret_cast<std::uintptr_t>(embedding) |
                        reinterpret_cast<std::uintptr_t>(residual)) %
                           16 ==
                       0;
  if (aligned && hidden % 4 == 0) {
    hipLaunchKernelGGL(
        MtpAddEmbeddingVec4Kernel, dim3(n_tokens, Blocks(streams * hidden / 4)),
        dim3(kThreads), 0, stream, reinterpret_cast<const float4*>(embedding),
        reinterpret_cast<float4*>(residual), hidden / 4, streams);
    return;
  }
  hipLaunchKernelGGL(MtpAddEmbeddingKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, embedding, residual, hidden, streams);
}

void Argmax(const float* logits, ArgmaxCandidate* scratch, std::int32_t* out,
            std::uint32_t n_tokens, std::uint32_t vocab, hipStream_t stream) {
  hipLaunchKernelGGL(ArgmaxPartialKernel, dim3(kArgmaxParts, n_tokens),
                     dim3(kThreads), 0, stream, logits, scratch, vocab);
  hipLaunchKernelGGL(ArgmaxFinishKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, logits, scratch, out, vocab);
}

void PenalizedArgmax(const float* logits, GreedyPenaltyRows penalties,
                     float repeat, float frequency, float presence,
                     PenaltyArgmaxCandidate* partial, ArgmaxCandidate* out,
                     std::uint32_t rows, std::uint32_t vocab,
                     hipStream_t stream) {
  if (rows == 0 || rows > 7 || vocab == 0)
    throw std::invalid_argument("invalid penalty argmax shape");
  PenaltyArgmaxPartialKernel<<<dim3(kArgmaxParts, rows), kThreads, 0, stream>>>(
      logits, penalties, repeat, frequency, presence, partial, vocab);
  PenaltyArgmaxFinishKernel<<<rows, kThreads, 0, stream>>>(logits, partial, out,
                                                           vocab);
}

__global__ void MaskDraftLogitsKernel(float* logits, const std::int32_t* table,
                                      std::uint32_t n) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && table[i] < 0)
    logits[i] = -INFINITY;
}

void MaskDraftLogits(float* logits, const std::int32_t* table, std::uint32_t n,
                     hipStream_t stream) {
  hipLaunchKernelGGL(MaskDraftLogitsKernel, dim3(Blocks(n)), dim3(kThreads), 0,
                     stream, logits, table, n);
}

__global__ void GatherDraftLogitsKernel(float* logits,
                                        const std::int32_t* table,
                                        std::uint32_t prefix, std::uint32_t n) {
  // One block: every read finishes before any write, so the table rows can
  // overwrite the logits they are gathered from.
  const std::uint32_t i = threadIdx.x;
  float value = -INFINITY;
  if (i < n && table[i] >= 0)
    value = logits[table[i]];
  __syncthreads();
  if (i < n)
    logits[prefix + i] = value;
}

void GatherDraftLogits(float* logits, const std::int32_t* table,
                       std::uint32_t prefix, std::uint32_t n,
                       hipStream_t stream) {
  hipLaunchKernelGGL(GatherDraftLogitsKernel, dim3(1), dim3(n), 0, stream,
                     logits, table, prefix, n);
}

__global__ void RemapDraftIdsKernel(std::uint32_t* ids, std::uint32_t n,
                                    std::uint32_t prefix,
                                    const std::int32_t* table,
                                    std::uint32_t table_rows) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || ids[i] < prefix)
    return;
  // A row past the table reads as an empty slot (-1), not past the table.
  const std::uint32_t row = ids[i] - prefix;
  ids[i] =
      row < table_rows ? static_cast<std::uint32_t>(table[row]) : UINT32_MAX;
}

void RemapDraftIds(std::uint32_t* ids, std::uint32_t n, std::uint32_t prefix,
                   const std::int32_t* table, std::uint32_t table_rows,
                   hipStream_t stream) {
  hipLaunchKernelGGL(RemapDraftIdsKernel, dim3(Blocks(n)), dim3(kThreads), 0,
                     stream, ids, n, prefix, table, table_rows);
}

void GatherArgmaxCandidates(const float* logits, const std::uint32_t* ids,
                            ArgmaxCandidate* out, std::uint32_t rows,
                            std::uint32_t vocab, hipStream_t stream) {
  hipLaunchKernelGGL(GatherArgmaxCandidatesKernel, dim3(Blocks(rows)),
                     dim3(kThreads), 0, stream, logits, ids, out, rows, vocab);
}

template<unsigned Keep>
void SelectMtpCandidates(const float* logits, std::uint32_t* ids,
                         std::uint32_t* scratch_ids, float* scores,
                         std::uint32_t vocab, hipStream_t stream) {
  if (logits == nullptr || ids == nullptr || scratch_ids == nullptr ||
      vocab == 0) {
    throw std::invalid_argument("invalid MTP candidate selection");
  }
  const auto tiles = [](std::uint32_t n) {
    return 1U + (n - 1U) / kMtpCandidateTile;
  };
  unsigned passes = 1;
  for (auto size = vocab; size > kMtpCandidateTile; size = tiles(size) * Keep) {
    ++passes;
  }
  // Choose the first buffer so the final pass always lands in `ids`.
  auto* destination = passes % 2 != 0 ? ids : scratch_ids;
  const std::uint32_t* source = nullptr;
  auto size = vocab;
  for (;;) {
    const auto blocks = tiles(size);
    hipLaunchKernelGGL((MtpCandidateTileKernel<Keep>), dim3(blocks),
                       dim3(kThreads), 0, stream, logits, source, destination,
                       blocks == 1 ? scores : nullptr, size, vocab);
    if (blocks == 1) {
      break;
    }
    size = blocks * Keep;
    source = destination;
    destination = destination == ids ? scratch_ids : ids;
  }
}

std::uint32_t MtpCandidateWorkspaceSize(std::uint32_t vocab) {
  const auto tiles = (vocab + 1023) / 1024;
  return (tiles > 2 ? tiles : 2) * kMtpCandidates;
}

void VerificationCandidates(const float* logits, GreedyPenaltyRows penalties,
                            std::uint32_t* ids, std::uint32_t* scratch_ids,
                            sampling::LogitCandidate* out, std::uint32_t rows,
                            std::uint32_t vocab, hipStream_t stream) {
  if (!logits || !ids || !scratch_ids || !out || !vocab || !rows || rows > 7)
    throw std::invalid_argument("invalid verification candidate shape");
  const auto stride = MtpCandidateWorkspaceSize(vocab);
  const auto tiles = [](std::uint32_t n) { return 1U + (n - 1U) / 1024; };
  auto size = vocab;
  auto blocks = tiles(size);
  VerificationCandidateTileKernel<true>
      <<<dim3(blocks, rows), kThreads, 0, stream>>>(logits, penalties, nullptr,
                                                    ids, size, vocab, stride);
  auto* source = ids;
  auto* destination = scratch_ids;
  while (blocks > 1) {
    size = blocks * kMtpCandidates;
    blocks = tiles(size);
    VerificationCandidateTileKernel<false>
        <<<dim3(blocks, rows), kThreads, 0, stream>>>(
            logits, penalties, source, destination, size, vocab, stride);
    std::swap(source, destination);
  }
  GatherVerificationCandidatesKernel<<<rows, kThreads, 0, stream>>>(
      logits, penalties, source, out, vocab, stride);
}

void MtpTopCandidates(const float* logits, std::uint32_t* ids,
                      std::uint32_t* scratch_ids, float* scores,
                      std::uint32_t vocab, hipStream_t stream) {
  if (scores == nullptr)
    throw std::invalid_argument("invalid MTP candidate scores");
  SelectMtpCandidates<kMtpCandidates>(logits, ids, scratch_ids, scores, vocab,
                                      stream);
}

void ScoreLogprobs(const float* logits, const std::int32_t* targets,
                   ScoredLogprob* out, std::uint32_t rows, std::uint32_t vocab,
                   hipStream_t stream) {
  if (rows == 0)
    return;
  hipLaunchKernelGGL(ScoreLogprobsKernel, dim3(rows), dim3(kThreads), 0, stream,
                     logits, targets, out, vocab);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
