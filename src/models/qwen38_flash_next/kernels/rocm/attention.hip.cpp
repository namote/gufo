// Attention over the cache: the selected-block attention of decode and
// short batches with the merge of its split partials, and the masked causal
// prefill attention on the WMMA matrix cores.
// HIP kernels follow the layouts and operator formulas in reference.cpp.

#include "src/models/qwen38_flash_next/kernels/rocm/kernels_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

// F16 fragments are used by attention and projection kernels below.
__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

/// grid n_tokens rows, block 256. Writes row t's selected complete blocks, in
/// ascending order, to lists[t * capacity ..] and their number to counts[t].
/// AttentionKernel would otherwise rebuild this list in every (head, split)
/// block, walking all of the row's mask words each time.
/// Invariant: a row has at most `capacity` selected blocks. SelectMarkKernel
/// marks at most budget = indexer_top_k / compress_ratio blocks per row and
/// the executor passes that budget as `capacity`, so the guards below never
/// trigger; they only keep a broken invariant from writing out of bounds.
__global__ void CompactSelectionKernel(
    const std::uint32_t* mask, std::uint32_t mask_words,
    const std::uint32_t* start_pos, std::uint32_t ratio, std::uint32_t* lists,
    std::uint32_t* counts, std::uint32_t capacity) {
  // Each thread owns kWords consecutive mask words, so one pass covers
  // 256 * kWords * 32 blocks (131,072 tokens at ratio 4) and the loads are
  // issued together instead of one dependent window at a time.
  constexpr std::uint32_t kWords = 4;
  constexpr std::uint32_t kChunk = 256 * kWords;  // words per pass
  __shared__ std::uint32_t wave_total[8];
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = threadIdx.x;
  const std::uint32_t lane = i & 31u;
  const std::uint32_t wave = i >> 5u;
  const std::uint32_t n_kv = *start_pos + t + 1;
  const std::uint32_t n_complete = n_kv / ratio;  // tail_start / ratio
  // The position comes from the device, the row size from the host: blocks
  // past mask_words have no mask bits and are not selected.
  const std::uint32_t n_words = min((n_complete + 31) / 32, mask_words);
  const std::uint32_t* words = mask + static_cast<std::size_t>(t) * mask_words;
  std::uint32_t* out = lists + static_cast<std::size_t>(t) * capacity;
  std::uint32_t written = 0;  // selected blocks of the earlier passes
  for (std::uint32_t c0 = 0; c0 < n_words; c0 += kChunk) {
    std::uint32_t bits[kWords];
    std::uint32_t count = 0;
#pragma unroll
    for (std::uint32_t j = 0; j < kWords; ++j) {
      const std::uint32_t w = c0 + (i * kWords) + j;
      bits[j] = w < n_words ? words[w] : 0u;
      // Only complete blocks are listed.
      if ((w + 1) * 32 > n_complete) {
        bits[j] &= w * 32 >= n_complete ? 0u : (1u << (n_complete % 32)) - 1u;
      }
      count += __popc(bits[j]);
    }
    std::uint32_t incl = count;
    for (std::uint32_t off = 1; off < 32; off <<= 1) {
      const std::uint32_t v = __shfl_up(incl, off);
      incl += lane >= off ? v : 0u;
    }
    if (lane == 31) {
      wave_total[wave] = incl;
    }
    __syncthreads();
    std::uint32_t base = 0;
    for (std::uint32_t w = 0; w < wave; ++w) {
      base += wave_total[w];
    }
    std::uint32_t total = 0;
    for (std::uint32_t w = 0; w < 8; ++w) {
      total += wave_total[w];
    }
    std::uint32_t slot = written + base + incl - count;
#pragma unroll
    for (std::uint32_t j = 0; j < kWords; ++j) {
      const std::uint32_t w = c0 + (i * kWords) + j;
      for (std::uint32_t b = bits[j]; b != 0; b &= b - 1) {
        if (slot < capacity) {
          out[slot] = (w * 32) + static_cast<std::uint32_t>(__builtin_ctz(b));
        }
        ++slot;
      }
    }
    written += total;
    __syncthreads();  // wave_total is rewritten by the next pass
  }
  if (i == 0) {
    counts[t] = min(written, capacity);
  }
}

/// grid (heads, queries), block 256 = head dim. Keys stream in tiles of 256
/// positions: lane j scores key j of the tile, then lane d accumulates
/// value column d with online softmax rescaling. In the sparse window the
/// tiles are gathered from the query's selected blocks (compacted through
/// LDS in windows of 1024 blocks, tiles cut across windows), so the key
/// work costs the budget rather than the context.
/// grid.z splits the key tiles round-robin across `gridDim.z` blocks; with
/// more than one split each block writes (max, sum, unnormalized acc) to
/// `partials[(t * heads + h) * splits + z]` and AttentionMergeKernel
/// combines them, otherwise the normalized row goes straight to `out`.
/// With `lists`/`counts` (CompactSelectionKernel's output, ratio 4 only) the
/// selected blocks are read from the ready-made list instead of being
/// compacted from the mask. The tiles are the same 64-block tiles dealt to the
/// splits in the same order, so the output is bitwise identical.
template<std::uint32_t kGroup, int kKvarn>
__global__ void AttentionKernel(
    const float* q, const __half* k_cache, const __half* v_cache,
    const std::uint32_t* mask, std::uint32_t mask_words, float* out,
    float* partials, const std::uint32_t* start_pos, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t d, std::uint32_t ratio,
    const std::uint32_t* lists, const std::uint32_t* counts,
    std::uint32_t capacity, const std::uint32_t* kv_blocks, KvarnReadView kvarn,
    std::uint32_t dense_prefix) {
  constexpr std::uint32_t kTile = 256;
  constexpr std::uint32_t kWindow = 1024;  // blocks compacted per pass
  const std::uint32_t split = blockIdx.z;
  const std::uint32_t splits = gridDim.z;
  __shared__ float qs[kGroup][256];
  __shared__ float p[kGroup][kTile];
  __shared__ float shared[kGroup][8];
  __shared__ std::uint32_t keys[kTile];
  // Room for one carried partial tile at the model's ratio of 4 (under a
  // tile's worth of blocks). A whole tile of keys would push the kernel past
  // 16 KiB of LDS and cost a resident block per CU on multi-row calls; below
  // ratio 4 a partial tile too big to carry is attended in its window.
  constexpr std::uint32_t kCarry = kTile / 4;
  __shared__ std::uint32_t list[kWindow + kCarry];
  __shared__ std::uint32_t wave_total[8];
  // A block serves kGroup query heads of one KV head: every K and V row it
  // loads feeds each head's unchanged arithmetic.
  const std::uint32_t h0 = blockIdx.x * kGroup;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t pos = *start_pos + t;
  const std::uint32_t kvh = h0 / (heads / kv_heads);
  const std::uint32_t n_kv = pos + 1;
  const std::uint32_t tail_start = (n_kv / ratio) * ratio;
  const std::uint32_t* words =
      mask == nullptr ? nullptr
                      : mask + static_cast<std::size_t>(t) * mask_words;
  const float scale = rsqrtf(static_cast<float>(d));
  const std::uint32_t i = threadIdx.x;
  const std::uint32_t lane = i & 31u;
  const std::uint32_t wave = i >> 5u;
#pragma unroll
  for (std::uint32_t g = 0; g < kGroup; ++g) {
    qs[g][i] = i < d ? q[(static_cast<std::size_t>(t) * heads + h0 + g) * d + i]
                     : 0.0f;
  }
  __syncthreads();
  float m[kGroup];
  float l[kGroup];
#pragma unroll
  for (std::uint32_t g = 0; g < kGroup; ++g) {
    m[g] = -INFINITY;
    l[g] = 0.0f;
  }
  // Every wave accumulates its own 32 keys of each tile into a partial
  // context row (eight dims per lane); the partials are summed at the end.
  float acc[kGroup][8] = {};
  const std::size_t kv_stride = static_cast<std::size_t>(kv_heads) * d;
  const __half* k_head = k_cache + kvh * d;
  const __half* v_head = v_cache + kvh * d;

  // Rows are read eight contiguous halves per lane, one 16-byte load, so a
  // wave reads a whole 256-wide row at once. A wave issues the rows of
  // `kBatch` consecutive keys of its 32 before using any, so their loads
  // overlap instead of each waiting out its own latency. Empty lanes read
  // the last key's row and are never used; a batch with no key at all is
  // skipped (returns false).
  constexpr std::uint32_t kBatch = 8;
  // KVarN rows come from the sink, the records or the ring (KvarnLoad8),
  // as the query at `pos` reads them.
  const auto load_batch = [&](const __half* head, std::uint32_t base,
                              uint4* rows) {
    const bool key = head == k_head;
    (void)key;
    bool any = false;
#pragma unroll
    for (std::uint32_t b = 0; b < kBatch; ++b) {
      any = any || keys[base + b] < n_kv;
    }
    if (!any) {
      return false;
    }
#pragma unroll
    for (std::uint32_t b = 0; b < kBatch; ++b) {
      const std::uint32_t j = min(keys[base + b], n_kv - 1);
      if constexpr (kKvarn)
        rows[b] = KvarnLoad8<kKvarn>(kvarn, key, pos, j, kvh, lane * 8);
      else
        rows[b] = *reinterpret_cast<const uint4*>(
            head + (KvRow(kv_blocks, j) * kv_stride) + (lane * 8));
    }
    return true;
  };
  const auto unpack8 = [](const uint4& packed, float* out8) {
    const auto* h2 = reinterpret_cast<const __half2*>(&packed);
#pragma unroll
    for (std::uint32_t j = 0; j < 4; ++j) {
      const float2 f = __half22float2(h2[j]);
      out8[2 * j] = f.x;
      out8[2 * j + 1] = f.y;
    }
  };

  // One tile: lane i holds key position keys[i] (n_kv marks an empty lane).
  // Scores: wave w takes keys 32w..32w+31, a wave-wide dot product per key.
  // Values: the same split, eight dims per lane.
  // The block reductions of every head in one pass: each head keeps
  // BlockMax/BlockSum's order (lane tree, then waves in order).
  const auto group_max = [&](float* v) {
#pragma unroll
    for (std::uint32_t g = 0; g < kGroup; ++g) {
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        v[g] = fmaxf(v[g], XorLane(v[g], offset));
      }
    }
    __syncthreads();
    if (lane == 0) {
#pragma unroll
      for (std::uint32_t g = 0; g < kGroup; ++g)
        shared[g][wave] = v[g];
    }
    __syncthreads();
#pragma unroll
    for (std::uint32_t g = 0; g < kGroup; ++g) {
      float total = -INFINITY;
      for (std::uint32_t w = 0; w < 8; ++w) {
        total = fmaxf(total, shared[g][w]);
      }
      v[g] = total;
    }
  };
  const auto group_sum = [&](float* v) {
#pragma unroll
    for (std::uint32_t g = 0; g < kGroup; ++g) {
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        v[g] += XorLane(v[g], offset);
      }
    }
    __syncthreads();
    if (lane == 0) {
#pragma unroll
      for (std::uint32_t g = 0; g < kGroup; ++g)
        shared[g][wave] = v[g];
    }
    __syncthreads();
#pragma unroll
    for (std::uint32_t g = 0; g < kGroup; ++g) {
      float total = 0.0f;
      for (std::uint32_t w = 0; w < 8; ++w) {
        total += shared[g][w];
      }
      v[g] = total;
    }
  };

  // One tile: lane i holds key position keys[i] (n_kv marks an empty lane).
  // Scores: wave w takes keys 32w..32w+31, a wave-wide dot product per key.
  // Values: the same split, eight dims per lane.
  const auto tile_step = [&]() {
    float qv[kGroup][8];
#pragma unroll
    for (std::uint32_t g = 0; g < kGroup; ++g) {
#pragma unroll
      for (std::uint32_t j = 0; j < 8; ++j) {
        qv[g][j] = qs[g][(lane * 8) + j];
      }
    }
    float s_mine[kGroup];
#pragma unroll
    for (std::uint32_t g = 0; g < kGroup; ++g)
      s_mine[g] = -INFINITY;
    for (std::uint32_t k0 = 0; k0 < 32; k0 += kBatch) {
      uint4 rows[kBatch];
      if (!load_batch(k_head, (wave * 32) + k0, rows)) {
        continue;
      }
#pragma unroll
      for (std::uint32_t b = 0; b < kBatch; ++b) {
        const std::uint32_t j = keys[(wave * 32) + k0 + b];
        float kv[8];
        unpack8(rows[b], kv);
#pragma unroll
        for (std::uint32_t g = 0; g < kGroup; ++g) {
          // The lane's eight products in the order fast-math has always
          // summed them, spelled out so the scores stay bitwise.
          float dot = qv[g][7] * kv[7];
          dot = fmaf(qv[g][3], kv[3], dot);
          dot = fmaf(qv[g][5], kv[5], dot);
          dot = fmaf(qv[g][1], kv[1], dot);
          dot = fmaf(qv[g][6], kv[6], dot);
          dot = fmaf(qv[g][2], kv[2], dot);
          dot = fmaf(qv[g][4], kv[4], dot);
          dot = fmaf(qv[g][0], kv[0], dot);
          dot = WaveSum(dot);
          if (lane == k0 + b) {
            s_mine[g] = j < n_kv ? dot * scale : -INFINITY;
          }
        }
      }
    }
    float tile_max[kGroup];
#pragma unroll
    for (std::uint32_t g = 0; g < kGroup; ++g)
      tile_max[g] = s_mine[g];
    group_max(tile_max);
    float rescale[kGroup];
    float tile_sum[kGroup];
#pragma unroll
    for (std::uint32_t g = 0; g < kGroup; ++g) {
      const float m_new = fmaxf(m[g], tile_max[g]);
      rescale[g] = m[g] == -INFINITY ? 0.0f : __expf(m[g] - m_new);
      const float pj =
          s_mine[g] == -INFINITY ? 0.0f : __expf(s_mine[g] - m_new);
      p[g][i] = pj;
      tile_sum[g] = pj;
      m[g] = m_new;
    }
    group_sum(tile_sum);
#pragma unroll
    for (std::uint32_t g = 0; g < kGroup; ++g) {
      l[g] = l[g] * rescale[g] + tile_sum[g];
#pragma unroll
      for (std::uint32_t x = 0; x < 8; ++x) {
        acc[g][x] *= rescale[g];
      }
    }
    __syncthreads();
    for (std::uint32_t k0 = 0; k0 < 32; k0 += kBatch) {
      uint4 rows[kBatch];
      if (!load_batch(v_head, (wave * 32) + k0, rows)) {
        continue;
      }
#pragma unroll
      for (std::uint32_t b = 0; b < kBatch; ++b) {
        float vv[8];
        unpack8(rows[b], vv);
#pragma unroll
        for (std::uint32_t g = 0; g < kGroup; ++g) {
          const float w = p[g][(wave * 32) + k0 + b];
          if (w != 0.0f) {
#pragma unroll
            for (std::uint32_t x = 0; x < 8; ++x) {
              acc[g][x] = fmaf(w, vv[x], acc[g][x]);
            }
          }
        }
      }
    }
    __syncthreads();
  };

  if (words == nullptr || pos < dense_prefix) {
    for (std::uint32_t tile = split * kTile; tile < n_kv;
         tile += kTile * splits) {
      keys[i] = tile + i < n_kv ? tile + i : n_kv;
      __syncthreads();
      tile_step();
    }
  } else {
    const std::uint32_t n_complete = tail_start / ratio;
    const std::uint32_t lane = i & 31u;
    const std::uint32_t wave = i >> 5u;
    // Tiles are cut from the running compacted list, not per window: the
    // budget's ~512 selected blocks make the same eight full tiles at any
    // depth, dealt round-robin over the splits. Cut per window, a deep
    // context scattered them into dozens of mostly empty tiles, and every
    // window holding fewer than one tile's blocks landed on split 0 alone,
    // so decode attention grew with the context instead of the budget.
    const std::uint32_t per_tile = kTile / ratio;
    std::uint32_t carry = 0;    // compacted blocks not yet cut into a tile
    std::uint32_t tile_no = 0;  // tiles cut so far (the same in every split)
    if (lists != nullptr) {
      // The whole row's list is ready (ratio 4: a tile is 64 blocks, exactly
      // the carry room, so the window loop below never attends a partial tile
      // in its window; it cuts full tiles and leaves one trailing partial
      // tile, which is the last tile here).
      const std::uint32_t n = counts[t];
      const std::uint32_t* row = lists + static_cast<std::size_t>(t) * capacity;
      const std::uint32_t entry = i / ratio;   // this thread's block in a tile
      const std::uint32_t offset = i % ratio;  // and its key within the block
      for (std::uint32_t t0 = 0; t0 < n; t0 += per_tile, ++tile_no) {
        if (tile_no % splits != split) {
          continue;
        }
        const std::uint32_t len = min(per_tile, n - t0);
        keys[i] = entry < len ? (row[t0 + entry] * ratio) + offset : n_kv;
        __syncthreads();
        tile_step();
      }
    }
    for (std::uint32_t w0 = 0; lists == nullptr && w0 < n_complete;
         w0 += kWindow) {
      // Thread i owns blocks w0 + 4i .. +3 of the window: flag the selected
      // ones and compact their indices with a block-wide exclusive scan,
      // appending after the carried-over blocks.
      // The four blocks share one mask word: read it once.
      std::uint32_t flags = 0;
      std::uint32_t count = 0;
      const std::uint32_t b0 = w0 + (i * 4);
      // Blocks past the mask row are unselected (see CompactSelectionKernel).
      const std::uint32_t word =
          b0 < n_complete && b0 / 32 < mask_words ? words[b0 / 32] : 0u;
      for (std::uint32_t k = 0; k < 4; ++k) {
        const std::uint32_t b = b0 + k;
        const bool set = b < n_complete && ((word >> (b % 32)) & 1u);
        flags |= (set ? 1u : 0u) << k;
        count += set ? 1u : 0u;
      }
      std::uint32_t incl = count;
      for (std::uint32_t off = 1; off < 32; off <<= 1) {
        const std::uint32_t v = __shfl_up(incl, off);
        incl += lane >= off ? v : 0u;
      }
      if (lane == 31) {
        wave_total[wave] = incl;
      }
      __syncthreads();
      std::uint32_t base = 0;
      for (std::uint32_t w = 0; w < wave; ++w) {
        base += wave_total[w];
      }
      std::uint32_t total = 0;
      for (std::uint32_t w = 0; w < 8; ++w) {
        total += wave_total[w];
      }
      std::uint32_t slot = carry + base + incl - count;
      for (std::uint32_t k = 0; k < 4; ++k) {
        if ((flags >> k) & 1u) {
          list[slot++] = w0 + (i * 4) + k;
        }
      }
      __syncthreads();
      const std::uint32_t n = carry + total;
      std::uint32_t t0 = 0;
      for (; t0 < n; t0 += per_tile, ++tile_no) {
        const std::uint32_t len = min(per_tile, n - t0);
        if (len < per_tile && len <= kCarry) {
          break;  // carried into the next window
        }
        if (tile_no % splits != split) {
          continue;
        }
        const std::uint32_t entry = i / ratio;
        keys[i] = entry < len ? (list[t0 + entry] * ratio) + (i % ratio) : n_kv;
        __syncthreads();
        tile_step();
      }
      // Move the partial tile's blocks to the front for the next window.
      carry = t0 < n ? n - t0 : 0;
      if (t0 > 0) {
        const std::uint32_t moved = i < carry ? list[t0 + i] : 0u;
        __syncthreads();
        if (i < carry) {
          list[i] = moved;
        }
      }
      __syncthreads();
    }
    if (carry > 0 && tile_no % splits == split) {
      const std::uint32_t entry = i / ratio;
      keys[i] = entry < carry ? (list[entry] * ratio) + (i % ratio) : n_kv;
      __syncthreads();
      tile_step();
    }
    // The incomplete tail block is always visible.
    if (tail_start < n_kv && split == 0) {
      keys[i] = tail_start + i < n_kv ? tail_start + i : n_kv;
      __syncthreads();
      tile_step();
    }
  }
  // Sum the eight wave partials: wave w's lane holds dims 8*lane..+7.
  __shared__ float red[8][256];
#pragma unroll
  for (std::uint32_t g = 0; g < kGroup; ++g) {
    const std::uint32_t h = h0 + g;
#pragma unroll
    for (std::uint32_t x = 0; x < 8; ++x) {
      red[wave][(lane * 8) + x] = acc[g][x];
    }
    __syncthreads();
    float total = 0.0f;
    for (std::uint32_t w = 0; w < 8; ++w) {
      total += red[w][i];
    }
    if (splits == 1) {
      if (i < d) {
        out[(static_cast<std::size_t>(t) * heads + h) * d + i] = total / l[g];
      }
    } else {
      float* part =
          partials +
          ((static_cast<std::size_t>(t) * heads + h) * splits + split) *
              (d + 2);
      if (i < d) {
        part[2 + i] = total;
      }
      if (i == 0) {
        part[0] = m[g];
        part[1] = l[g];
      }
    }
    __syncthreads();  // red is rewritten by the next head
  }
}

/// grid (heads, queries): merges the split partials of one row with the
/// usual log-sum-exp rescaling.
__global__ void AttentionMergeKernel(const float* partials, float* out,
                                     std::uint32_t heads, std::uint32_t d,
                                     std::uint32_t splits) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t i = threadIdx.x;
  const float* base =
      partials + (static_cast<std::size_t>(t) * heads + h) * splits * (d + 2);
  float m = -INFINITY;
  for (std::uint32_t s = 0; s < splits; ++s) {
    m = fmaxf(m, base[s * (d + 2)]);
  }
  float l = 0.0f;
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < splits; ++s) {
    const float* part = base + s * (d + 2);
    const float ms = part[0];
    const float scale = ms == -INFINITY ? 0.0f : __expf(ms - m);
    l += part[1] * scale;
    if (i < d) {
      acc += part[2 + i] * scale;
    }
  }
  if (i < d) {
    out[(static_cast<std::size_t>(t) * heads + h) * d + i] = acc / l;
  }
}

// Masked prefill attention on the WMMA matrix cores, ported from the Qwen
// 27B route (src/models/qwen/hip/kernels/attention_wmma.hip) to this
// model's 24 x 256 query heads over two KV heads. Wave32 fragment layout: A
// holds row L%16 and 16 contiguous k, B holds column L%16 and 16 contiguous
// k, and C element i is row 2i + L/16, column L%16. Every block owns 32
// queries x 2 heads = 64 rows against one 16-key tile at a time; eight waves
// split as four S tiles x two halves of the head dimension, and the next
// tile's K/V is prefetched into registers behind the current tile's S,
// softmax and PV phases. V reaches LDS transposed with one key per lane so
// the writes stay conflict-free. The optional block mask follows the model's
// sparse selection: keys at or past the query's incomplete tail block are
// always visible, earlier blocks only when their bit is set.
constexpr std::uint32_t kWmmaHeadDim = 256;
constexpr std::uint32_t kWmmaQueryHeads = 24;
constexpr std::uint32_t kWmmaKvHeads = 2;
constexpr std::uint32_t kWmmaGqa = kWmmaQueryHeads / kWmmaKvHeads;
constexpr std::uint32_t kWmmaAttnWidth = kWmmaQueryHeads * kWmmaHeadDim;
constexpr std::uint32_t kWmmaKvWidth = kWmmaKvHeads * kWmmaHeadDim;
constexpr std::uint32_t kWmmaHeads = 2;  // query heads per block, divides GQA
// Sixteen queries keep the causal-tail accumulator in registers.
constexpr std::uint32_t kWmmaQueryRows = 16;
constexpr std::uint32_t kWmmaKeys = 16;
constexpr std::uint32_t kWmmaRatio = 4;
constexpr std::uint32_t kWmmaKSteps = kWmmaHeadDim / 16;
// Row padding keeps a 16-byte-per-lane fragment read off a single bank group.
constexpr std::uint32_t kWmmaKStride = kWmmaHeadDim + 8;
// Union-mask capacity of the live visible extent: one bit per 4-token block,
// 262,144 visible tokens. Rows may be strided wider.
constexpr std::uint32_t kWmmaMaxMaskWords = 2048;

/// Two row layouts share the kernel. The dense window packs 16 queries x 2
/// heads (a row block is 16 queries of one head, grid.y over head pairs).
/// The sparse window packs four queries x twelve heads into three row
/// blocks without padding, with adjacent blocks covering both KV heads: the key
/// tiles are gathered from the union of the block's query selections and
/// four selections overlap far less than 32 (measured at 16k depth: 846
/// versus 2,478 selected blocks against 512 per query).
template<std::uint32_t kQueryRows, std::uint32_t kKeys, bool kPackHeads,
         int kKvarn = 0>
__launch_bounds__(256, 2) __global__ void WmmaCausalAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    const std::uint32_t* __restrict__ mask, std::uint32_t mask_words,
    float* __restrict__ out, std::uint32_t start_pos, std::uint32_t n_tokens,
    std::uint32_t first_query_group, const std::uint32_t* kv_blocks,
    KvarnReadView kvarn = {}, std::uint32_t gate_stride = 0) {
  // The launcher accepts only four-token selection blocks. Keep that
  // geometry constant throughout the key sweep.
  constexpr std::uint32_t ratio = kWmmaRatio;
  constexpr std::uint32_t kHeadDim = kWmmaHeadDim;
  constexpr std::uint32_t kRowBlocks = kPackHeads
                                           ? (kQueryRows * kWmmaGqa + 15) / 16
                                           : (kQueryRows / 16) * kWmmaHeads;
  static_assert(!kPackHeads || kQueryRows * kWmmaGqa == 48,
                "four queries pack twelve heads into three tiles");
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / 2;
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kOTilesPerWave = (kRowBlocks * kWmmaKSteps) / 8;
  constexpr std::uint32_t kSoftmaxLanes = 4;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  static_assert(kSTiles == (kPackHeads ? 3 : 2), "two waves per S tile");
  static_assert(kOTilesPerWave == 2 * kRowBlocks, "O tiles per wave");
  static_assert(kKeys % 16 == 0 && (kPackHeads || kQueryRows % 16 == 0),
                "16-row WMMA tiles");
  static_assert(kKStepsPerWave * 2 == kWmmaKSteps, "k split");
  static_assert(kWmmaKStride % 8 == 0 && kVtStride % 8 == 0,
                "fragment rows must start on a 16-byte boundary");
  static_assert(kSoftmaxLanes * (kKeys / kSoftmaxLanes) == kKeys, "softmax");

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  // Keep the two KV heads of nearby queries together in the launch order.
  // This improves cache reuse without changing any query's key sweep.
  const std::uint32_t linear = blockIdx.y * gridDim.x + blockIdx.x;
  const std::uint32_t query_group =
      first_query_group + (kPackHeads ? linear / kWmmaKvHeads : blockIdx.x);
  const std::uint32_t query_start = query_group * kQueryRows;
  const std::uint32_t kv_head =
      kPackHeads ? linear % kWmmaKvHeads : blockIdx.y / (kWmmaGqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      kPackHeads ? kv_head * kWmmaGqa
                 : (kv_head * kWmmaGqa) +
                       ((blockIdx.y % (kWmmaGqa / kWmmaHeads)) * kWmmaHeads);
  constexpr float attention_scale = 1.0F / 16.0F;

  // Row (row block rb, row r) -> (local query, query head, live).
  const auto row_query = [&](std::uint32_t rb, std::uint32_t r) {
    return kPackHeads ? query_start + (rb * 16 + r) / kWmmaGqa
                      : query_start + ((rb / kWmmaHeads) * 16) + r;
  };
  const auto row_head = [&](std::uint32_t rb, std::uint32_t r) {
    return kPackHeads ? first_query_head + (rb * 16 + r) % kWmmaGqa
                      : first_query_head + (rb % kWmmaHeads);
  };
  const auto row_live = [&](std::uint32_t rb, std::uint32_t r) {
    return row_query(rb, r) < n_tokens &&
           (!kPackHeads || rb * 16 + r < kQueryRows * kWmmaGqa);
  };

  // K and V^T have their own buffers so the next tile's S overlaps this
  // tile's PV (see the key sweep below).
  __shared__ __half k_lds[kKeys * kWmmaKStride];
  __shared__ __attribute__((aligned(16))) __half v_lds[kHeadDim * kVtStride];
  // Pad score rows for the four-lane softmax reads and probability rows
  // for the WMMA fragments. Both transposes otherwise repeat LDS banks.
  // Dense score rows are padded to 20 floats: the WMMA scatter stays nearly
  // conflict-free and each softmax lane reads its four scores as a float4.
  // The sparse window keeps 17-float rows and scalar reads (faster there).
  constexpr bool kVectorSoftmax = !kPackHeads;
  constexpr std::uint32_t kScoreStride = kVectorSoftmax ? 20 : 17;
  static_assert(kKeys / kSoftmaxLanes == 4, "one float4 of scores per lane");
  __shared__
      __attribute__((aligned(16))) float s_lds[2][kSTiles][16][kScoreStride];
  __shared__ __attribute__((aligned(16))) __half p_lds[kRows][kKeys + 8];
  __shared__ float row_sum[kRows];
  // Rescale factors grouped as [row block][row parity][row / 2]: a lane of
  // the PV phase needs the eight rows of its parity, read as two float4.
  __shared__ __attribute__((aligned(16))) float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  if (wave < 2 * kSTiles) {
    const std::uint32_t local_query = row_query(s_rb, sub);
    const bool live = row_live(s_rb, sub);
    const float* q_row =
        q +
        (static_cast<std::size_t>(live ? local_query : 0) * kWmmaAttnWidth) +
        (static_cast<std::size_t>(live ? row_head(s_rb, sub) : 0) * kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] = static_cast<_Float16>(f.x * attention_scale);
        q_frag[ks][(v * 4) + 1] = static_cast<_Float16>(f.y * attention_scale);
        q_frag[ks][(v * 4) + 2] = static_cast<_Float16>(f.z * attention_scale);
        q_frag[ks][(v * 4) + 3] = static_cast<_Float16>(f.w * attention_scale);
      }
    }
  }

  // This wave owns dim tiles `wave` and `wave + 8` for every row block.
  v8f o_acc[kRowBlocks][2] = {};
  // The same four threads own each softmax row throughout the key sweep.
  // Keep its running statistics in registers; only the final denominator
  // and each tile's rescale factor need to cross waves.
  float running_max = -INFINITY;
  float running_sum = 0.0F;

  const std::uint32_t context_end = start_pos + n_tokens;
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kQueryRows);

  // Key tiles are gathered from `kBlocksPerTile` selection blocks (ratio
  // keys each). In the sparse window the blocks come from the union of this
  // block's query masks (every row still applies its own mask below), then
  // every block from the earliest row's tail on; the dense window walks the
  // blocks in order. The sweep therefore costs the union of the selected
  // windows, not the context.
  constexpr std::uint32_t kBlocksPerTile = kKeys / 4;
  static_assert(kBlocksPerTile == 4, "a tile is four selection blocks");
  const std::uint32_t n_blocks = (max_visible + ratio - 1) / ratio;
  const std::uint32_t tail_block =
      ((start_pos + query_start + 1) / ratio);  // first always-visible block
  constexpr bool sparse = kPackHeads;
  // Four 512-block selections plus their incomplete tail. The same LDS
  // stores a mask for arbitrary wider selections used by operator callers.
  constexpr unsigned kListCapacity = 4 * 512 + 4;
  // A compact entry also carries which of the packed queries selected the
  // block, so the softmax reads its mask bits from the tile, not global.
  constexpr unsigned kMaskRows = kPackHeads ? kQueryRows : 1;
  constexpr unsigned kMemberShift = 28;
  static_assert(kMaskRows <= 32 - kMemberShift, "membership bits fit");
  __shared__ unsigned union_words[kListCapacity];
  __shared__ unsigned wave_counts[8];
  bool compact = false;
  unsigned selected = 0;
  if (sparse) {
    // Consecutive word ranges per thread make the prefix scan preserve
    // increasing block order, including ties and the always-visible tail.
    const unsigned live_rows = min(kQueryRows, n_tokens - query_start);
    const unsigned word_count = (n_blocks + 31) / 32;
    const unsigned words_per_thread = (word_count + 255) / 256;
    unsigned local_words[8];
    unsigned row_words[8][kMaskRows];
    unsigned count = 0;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      const unsigned w = tid * words_per_thread + j;
      unsigned bits = 0;
#pragma unroll
      for (unsigned r = 0; r < kMaskRows; ++r)
        row_words[j][r] = 0;
      if (j < words_per_thread && w < word_count) {
#pragma unroll
        for (unsigned r = 0; r < kMaskRows; ++r) {
          if (r < live_rows) {
            row_words[j][r] = mask[size_t(query_start + r) * mask_words + w];
            bits |= row_words[j][r];
          }
        }
        if (w * 32 >= tail_block)
          bits = ~0u;
        else if ((w + 1) * 32 > tail_block)
          bits |= ~0u << (tail_block % 32);
        if ((w + 1) * 32 > n_blocks)
          bits &= (1u << (n_blocks % 32)) - 1u;
      }
      local_words[j] = bits;
      count += __popc(bits);
    }
    unsigned scan = count;
#pragma unroll
    for (unsigned offset = 1; offset < 32; offset *= 2) {
      unsigned before = __shfl_up(scan, offset, 32);
      if (lane >= offset)
        scan += before;
    }
    if (lane == 31)
      wave_counts[wave] = scan;
    __syncthreads();
    unsigned prefix = scan - count;
#pragma unroll
    for (unsigned w = 0; w < 8; ++w) {
      if (w < wave)
        prefix += wave_counts[w];
      selected += wave_counts[w];
    }
    compact = selected <= kListCapacity;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      const unsigned word = tid * words_per_thread + j;
      unsigned bits = local_words[j];
      if (compact) {
        while (bits) {
          unsigned bit = __builtin_ctz(bits);
          unsigned members = 0;
#pragma unroll
          for (unsigned r = 0; r < kMaskRows; ++r)
            members |= ((row_words[j][r] >> bit) & 1u) << r;
          union_words[prefix++] = (word * 32 + bit) | (members << kMemberShift);
          bits &= bits - 1;
        }
      } else if (j < words_per_thread && word < word_count) {
        union_words[word] = bits;
      }
    }
    __syncthreads();
  }
  // First visible block at or after `b` (n_blocks when none). Uniform across
  // the block: every thread walks the same words.
  const auto next_block = [&](std::uint32_t b) -> std::uint32_t {
    if (!sparse) {
      return b;
    }
    while (b < tail_block && b < n_blocks) {
      const std::uint32_t bits = union_words[b / 32] >> (b % 32);
      // Never jump past the tail: from tail_block on every block is visible.
      if (bits != 0) {
        return min(b + static_cast<std::uint32_t>(__builtin_ctz(bits)),
                   tail_block);
      }
      b = min(((b / 32) + 1) * 32u, tail_block);
    }
    return b;
  };
  struct Tile {
    std::uint32_t block[kBlocksPerTile];
    std::uint32_t count;
    std::uint32_t members;  // kMaskRows bits per block, compact lists only
  };
  // Gathers the next tile starting the search at block `b`; returns the
  // block to continue from.
  const auto gather_tile = [&](std::uint32_t b, Tile& tile) {
    if constexpr (!sparse) {
      // The dense window's tiles are four consecutive blocks.
      tile.count = min(kBlocksPerTile, n_blocks - min(b, n_blocks));
      tile.members = 0;
#pragma unroll
      for (std::uint32_t i = 0; i < kBlocksPerTile; ++i)
        tile.block[i] = b + i < n_blocks ? b + i : n_blocks;
      return b + tile.count;
    }
    if (compact) {
      tile.count = min(kBlocksPerTile, selected - min(b, selected));
      tile.members = 0;
#pragma unroll
      for (unsigned i = 0; i < kBlocksPerTile; ++i) {
        const unsigned entry = b + i < selected ? union_words[b + i] : 0u;
        tile.block[i] =
            b + i < selected ? entry & ((1u << kMemberShift) - 1u) : n_blocks;
        tile.members |= (entry >> kMemberShift) << (i * kMaskRows);
      }
      return b + tile.count;
    }
    tile.count = 0;
    tile.members = 0;
#pragma unroll
    for (std::uint32_t i = 0; i < kBlocksPerTile; ++i) {
      b = next_block(b);
      const bool have = b < n_blocks;
      tile.block[i] = have ? b : n_blocks;
      tile.count += have ? 1u : 0u;
      b = have ? b + 1 : b;
    }
    return b;
  };
  // Key position of tile row `r` (0..15), or context_end past the tile.
  const auto tile_key = [&](const Tile& tile, std::uint32_t r) {
    if constexpr (!sparse) {
      const std::uint32_t blk = tile.block[0] + (r / ratio);
      return blk < n_blocks ? (blk * ratio) + (r % ratio) : context_end;
    }
    const std::uint32_t i = r / ratio;
    std::uint32_t blk = n_blocks;
#pragma unroll
    for (std::uint32_t j = 0; j < kBlocksPerTile; ++j) {
      if (i == j) {
        blk = tile.block[j];
      }
    }
    return blk < n_blocks ? (blk * ratio) + (r % ratio) : context_end;
  };

  // One key per lane for V, a 16-dim slice per thread: a strided global read
  // in exchange for conflict-free transpose writes.
  constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (256 * 8);
  constexpr std::uint32_t kKRegs = (kKeys * (kHeadDim / 8)) / 256;
  static_assert(kKRegs * 256 == kKeys * (kHeadDim / 8), "K stages evenly");
  const std::uint32_t v_key = lane % kKeys;
  const std::uint32_t v_slice = (tid / kKeys) * (kVRegs * 8);
  const auto* v_base =
      v_cache + (static_cast<std::size_t>(kv_head) * kHeadDim) + v_slice;
  const auto* k_base = k_cache + (static_cast<std::size_t>(kv_head) * kHeadDim);

  const auto load_v = [&](const Tile& tile, uint4* dst) {
    const std::uint32_t key_position = tile_key(tile, v_key);
    const bool live = key_position < context_end;
    if constexpr (kKvarn) {
      // A prompt chunk reads one fixed record range (quant_end), so every
      // query of the tile reads the same rows.
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j)
        dst[j] = live ? KvarnLoad8<kKvarn>(kvarn, false, 0, key_position,
                                           kv_head, v_slice + (j * 8))
                      : make_uint4(0u, 0u, 0u, 0u);
      return;
    }
    const auto* src =
        v_base + (KvRow(kv_blocks, live ? key_position : 0) * kWmmaKvWidth);
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      dst[j] = live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                    : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  // Coalesced: a wave reads one key row's 512 contiguous bytes.
  const auto load_k = [&](const Tile& tile, uint4* dst) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_position = tile_key(tile, idx / (kHeadDim / 8));
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      const bool live = key_position < context_end;
      if constexpr (kKvarn) {
        dst[n] =
            live ? KvarnLoad8<kKvarn>(kvarn, true, 0, key_position, kv_head, d8)
                 : make_uint4(0u, 0u, 0u, 0u);
        continue;
      }
      dst[n] = live ? *reinterpret_cast<const uint4*>(
                          k_base +
                          (KvRow(kv_blocks, key_position) * kWmmaKvWidth) + d8)
                    : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  const auto stage_k = [&](const uint4* regs) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      *reinterpret_cast<uint4*>(&k_lds[(key_row * kWmmaKStride) + d8]) =
          regs[n];
    }
  };
  const auto stage_v = [&](const uint4* regs) {
    // Neighbouring lanes hold neighbouring keys of the same dims. Swapping
    // one half with the partner lets each lane store a key pair of one dim
    // as a word: the even lane the even dim, the odd lane the odd dim.
    const bool odd = (v_key & 1u) != 0u;
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      const std::uint32_t words[4] = {regs[j].x, regs[j].y, regs[j].z,
                                      regs[j].w};
#pragma unroll
      for (std::uint32_t w = 0; w < 4; ++w) {
        const std::uint32_t own = words[w];
        const auto partner =
            static_cast<std::uint32_t>(__builtin_amdgcn_mov_dpp(
                static_cast<int>(own), 0xB1, 0xF, 0xF, false));
        const std::uint32_t pair =
            odd ? __builtin_amdgcn_perm(partner, own, 0x03020706U)
                : __builtin_amdgcn_perm(partner, own, 0x05040100U);
        const std::uint32_t dim = v_slice + (j * 8) + (2 * w) + (odd ? 1 : 0);
        *reinterpret_cast<std::uint32_t*>(
            &v_lds[(dim * kVtStride) + (v_key & ~1u)]) = pair;
      }
    }
  };

  // --- S = Q K^T of the tile in k_lds. Each half writes its own slot; the
  // reader sums them.
  const auto scores = [&] {
    if (wave < 2 * kSTiles) {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const v16h k_frag =
            LoadFrag(&k_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
        s_acc = Wmma(q_frag[ks], k_frag, s_acc);
        if constexpr (kPackHeads) {
          // Bound temporary K fragments without changing the dot-product
          // sequence. Hoisting all eight LDS fragments raises register use.
          if ((ks + 1) % 2 == 0)
            __builtin_amdgcn_sched_barrier(0);
        }
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
  };

  // --- online softmax of the scores in s_lds: kSoftmaxLanes threads per
  // query row ---
  const auto softmax = [&](const Tile& tile) {
    const std::uint32_t rg = tid / kSoftmaxLanes;
    if (rg < kRows) {
      const std::uint32_t seg = tid % kSoftmaxLanes;
      constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const std::uint32_t local_query = row_query(rb, row);
      const bool live_row = row_live(rb, row);
      const std::uint32_t absolute_query = start_pos + local_query;
      // Keys in the query's own incomplete block are always visible; earlier
      // blocks follow the selection mask.
      const std::uint32_t tail_start = ((absolute_query + 1) / ratio) * ratio;
      // The dense window runs without a selection mask.
      const std::uint32_t* words =
          !sparse || mask == nullptr || !live_row
              ? nullptr
              : mask + static_cast<std::size_t>(local_query) * mask_words;
      float part_max = -INFINITY;
      float vals[kPerLane];
      float s0v[kPerLane];
      float s1v[kPerLane];
      // A lane's four keys form one selection block, so the sparse window
      // decides its mask bit once per lane instead of once per key.
      static_assert(kPerLane == ratio, "one selection block per lane");
      bool block_selected = true;
      std::uint32_t blk = 0;
      if constexpr (sparse) {
        blk = n_blocks;
#pragma unroll
        for (std::uint32_t j = 0; j < kBlocksPerTile; ++j) {
          if (seg == j) {
            blk = tile.block[j];
          }
        }
        if (words != nullptr && blk < n_blocks && blk * ratio < tail_start) {
          const std::uint32_t member =
              (seg * kMaskRows) + (local_query - query_start);
          block_selected = compact
                               ? ((tile.members >> member) & 1u) != 0u
                               : ((words[blk / 32] >> (blk % 32)) & 1u) != 0u;
        }
      }
      if constexpr (kVectorSoftmax) {
        const float4 s0 = *reinterpret_cast<const float4*>(
            &s_lds[0][rb][row][seg * kPerLane]);
        const float4 s1 = *reinterpret_cast<const float4*>(
            &s_lds[1][rb][row][seg * kPerLane]);
        s0v[0] = s0.x, s0v[1] = s0.y, s0v[2] = s0.z, s0v[3] = s0.w;
        s1v[0] = s1.x, s1v[1] = s1.y, s1v[2] = s1.z, s1v[3] = s1.w;
      } else {
        // Only lanes with a visible block read their scores, which saves
        // LDS bandwidth, but under one condition so the reads share a wait.
#pragma unroll
        for (std::uint32_t m = 0; m < kPerLane; ++m)
          s0v[m] = s1v[m] = 0.0F;
        if (live_row && block_selected && blk < n_blocks) {
#pragma unroll
          for (std::uint32_t m = 0; m < kPerLane; ++m) {
            const std::uint32_t col = (seg * kPerLane) + m;
            const std::uint32_t s_index = ((col / 16) * kRowBlocks) + rb;
            s0v[m] = s_lds[0][s_index][row][col % 16];
            s1v[m] = s_lds[1][s_index][row][col % 16];
          }
        }
      }
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = (seg * kPerLane) + m;
        const std::uint32_t key_position = tile_key(tile, col);
        const bool valid = live_row && key_position <= absolute_query &&
                           key_position < context_end && block_selected;
        vals[m] = valid ? (s0v[m] + s1v[m]) : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_max = fmaxf(part_max, XorLane(part_max, off));
      }
      const float prev_max = running_max;
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
      __half p[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p[m] = static_cast<__half>(w);
        if constexpr (!kVectorSoftmax)
          p_lds[rg][(seg * kPerLane) + m] = p[m];
      }
      if constexpr (kVectorSoftmax)
        __builtin_memcpy(&p_lds[rg][seg * kPerLane], p, sizeof(p));
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_sum += XorLane(part_sum, off);
      }
      running_max = next_max;
      // The completed partial sum, then one fused update: written out so
      // fast-math cannot reassociate the reduction into the update.
      running_sum = __fmaf_rn(prior_scale, running_sum, part_sum);
      if (seg == 0) {
        row_scale[kVectorSoftmax ? (rb * 16) + ((row % 2) * 8) + (row / 2)
                                 : rg] = prior_scale;
      }
    }
  };

  // --- rescale the running O by the tile's new maximum, then O += P V. A
  // lane touches only rows 2i + half_id, so the factors are read once per
  // row block.
  const auto accumulate = [&](const Tile& tile) {
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
      float scale[8];
      if constexpr (kVectorSoftmax) {
        const auto* factors = reinterpret_cast<const float4*>(
            &row_scale[(rb * 16) + (half_id * 8)]);
        const float4 f0 = factors[0];
        const float4 f1 = factors[1];
        scale[0] = f0.x, scale[1] = f0.y, scale[2] = f0.z, scale[3] = f0.w;
        scale[4] = f1.x, scale[5] = f1.y, scale[6] = f1.z, scale[7] = f1.w;
      } else {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i)
          scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[rb][t][i] *= scale[i];
        }
      }
    }
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
      v16h v_frag[kKeyBlocks];
#pragma unroll
      for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
        v_frag[kb] =
            LoadFrag(&v_lds[(((dim_tile * 16) + sub) * kVtStride) + (kb * 16)]);
      }
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          v8f next = Wmma(p_frag, v_frag[kb], o_acc[rb][t]);
          if constexpr (!kPackHeads) {
            const std::uint32_t first_key = tile.block[0] * ratio;
            // WMMA's dot accumulation can change its last bits when zero
            // probabilities multiply populated future values instead of the
            // zero padding at a chunk boundary. Accumulate each query's final
            // partial tile over its visible keys only. Earlier complete tiles
            // retain the matrix-core path.
            if (first_key + kKeys - 1 > start_pos + query_start) {
#pragma unroll
              for (std::uint32_t i = 0; i < 8; ++i) {
                const std::uint32_t row = (2 * i) + half_id;
                const std::uint32_t absolute_query =
                    start_pos + row_query(rb, row);
                if (absolute_query < first_key + kKeys - 1) {
                  float acc = o_acc[rb][t][i];
                  for (std::uint32_t key = 0;
                       key < kKeys && first_key + key <= absolute_query;
                       ++key) {
                    acc = fmaf(
                        __half2float(p_lds[(rb * 16) + row][key]),
                        __half2float(
                            v_lds[((dim_tile * 16) + sub) * kVtStride + key]),
                        acc);
                  }
                  next[i] = acc;
                }
              }
            }
          }
          o_acc[rb][t] = next;
        }
      }
    }
  };

  // Software-pipelined key sweep. Phase A of step j accumulates tile j's PV
  // while the score waves compute tile j + 1's S; phase B runs tile j + 1's
  // softmax while staging K for tile j + 2 and V for tile j + 1, whose
  // registers were loaded one step earlier. Every row sees the same S, P,
  // rescale and accumulation sequence as a sequential sweep.
  uint4 k_regs[kKRegs] = {};
  uint4 v_regs[kVRegs] = {};
  Tile cur;     // tile j: P in p_lds, V in v_lds
  Tile next;    // tile j + 1: K in k_lds
  Tile k_tile;  // tile j + 2: K in k_regs
  Tile v_tile;  // tile j + 1 V in v_regs
  std::uint32_t cursor = gather_tile(0, cur);
  if (cur.count != 0) {
    load_k(cur, k_regs);
    load_v(cur, v_regs);
  }
  stage_k(k_regs);
  stage_v(v_regs);
  cursor = gather_tile(cursor, next);
  if (next.count != 0) {
    load_k(next, k_regs);
    load_v(next, v_regs);
  }
  SyncLds();
  if (cur.count != 0)
    scores();
  SyncLds();
  if (cur.count != 0)
    softmax(cur);
  stage_k(k_regs);
  v_tile = next;
  cursor = gather_tile(cursor, k_tile);
  if (k_tile.count != 0)
    load_k(k_tile, k_regs);
  SyncLds();

  while (cur.count != 0) {
    // Phase A: k_lds = K(j + 1), v_lds = V(j), p_lds = P(j).
    if (next.count != 0)
      scores();
    accumulate(cur);
    // Finish PV before the barrier. Left alone, the compiler sinks these
    // WMMAs into phase B behind the softmax, which is the critical path.
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb)
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t)
        asm volatile("" : "+v"(o_acc[rb][t]));
    SyncLds();
    // Phase B.
    if (next.count != 0)
      softmax(next);
    stage_v(v_regs);
    stage_k(k_regs);
    cur = next;
    next = k_tile;
    v_tile = next;
    cursor = gather_tile(cursor, k_tile);
    if (k_tile.count != 0)
      load_k(k_tile, k_regs);
    if (v_tile.count != 0)
      load_v(v_tile, v_regs);
    SyncLds();
  }
  if (tid < kRows * kSoftmaxLanes && tid % kSoftmaxLanes == 0)
    row_sum[tid / kSoftmaxLanes] = running_sum;
  SyncLds();

  // --- epilogue: normalize and apply the sigmoid output gate ---
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        if (!row_live(rb, row)) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(row_query(rb, row)) * kWmmaAttnWidth) +
            (static_cast<std::size_t>(row_head(rb, row)) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        if (gate != nullptr) {
          // gate_stride != 0: the packed [q | gate] projection rows, the
          // gate half of each head (gate points at the first one).
          const std::size_t gate_offset =
              gate_stride == 0
                  ? offset
                  : (static_cast<std::size_t>(row_query(rb, row)) *
                     gate_stride) +
                        (static_cast<std::size_t>(row_head(rb, row)) * 2 *
                         kHeadDim) +
                        (dim_tile * 16) + sub;
          value *= SigmoidF(gate[gate_offset]);
        }
        out[offset] = value;
      }
    }
  }
}

}  // namespace

void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const std::uint32_t* mask, std::uint32_t mask_words, float* out,
               float* partials, std::uint32_t splits, std::uint32_t n_tokens,
               const std::uint32_t* start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t d, std::uint32_t ratio,
               hipStream_t stream, std::uint32_t* lists, std::uint32_t* counts,
               std::uint32_t list_capacity, const std::uint32_t* kv_blocks,
               const KvarnReadView* kvarn, std::uint32_t dense_prefix) {
  if (kvarn != nullptr && kvarn->bits != 4 && kvarn->bits != 6)
    throw std::invalid_argument("KVarN bits must be 4 or 6");

  if (d != 256) {
    return;  // the kernel is written for the model's 256-wide heads
  }
  const std::uint32_t z = partials != nullptr ? std::max(splits, 1u) : 1u;
  if (ratio != 4 || mask == nullptr || lists == nullptr || counts == nullptr ||
      list_capacity == 0) {
    lists = nullptr;
    counts = nullptr;
    list_capacity = 0;
  }
  if (lists != nullptr) {
    // Compact each row's selection once; the attention blocks of every head
    // and split then read the list. Both launches read *start_pos on device.
    hipLaunchKernelGGL(CompactSelectionKernel, dim3(n_tokens), dim3(kThreads),
                       0, stream, mask, mask_words, start_pos, ratio, lists,
                       counts, list_capacity);
  }
  // Query heads sharing a KV head share each block's K/V loads. Two per
  // block keep enough blocks in flight for one or two rows; four suit wider
  // verification batches (measured at d32K/d128K).
  if ((heads / kv_heads) % 4 != 0) {
    return;
  }
  // KVarN reads through the view; its FP16 planes stand in for the caches
  // so the kernel tells K from V.
  const KvarnReadView view = kvarn != nullptr ? *kvarn : KvarnReadView{};
  if (kvarn != nullptr) {
    k_cache = kvarn->k_fp16;
    v_cache = kvarn->v_fp16;
  }
  const auto launch = [&](auto kernel, std::uint32_t group) {
    hipLaunchKernelGGL(kernel, dim3(heads / group, n_tokens, z), dim3(kThreads),
                       0, stream, q, k_cache, v_cache, mask, mask_words, out,
                       partials, start_pos, heads, kv_heads, d, ratio, lists,
                       counts, list_capacity, kv_blocks, view, dense_prefix);
  };
  if (kvarn != nullptr && kvarn->bits == 6) {
    if (n_tokens <= 2)
      launch(AttentionKernel<2, 6>, 2);
    else
      launch(AttentionKernel<4, 6>, 4);
  } else if (kvarn != nullptr) {
    if (n_tokens <= 2)
      launch(AttentionKernel<2, 4>, 2);
    else
      launch(AttentionKernel<4, 4>, 4);
  } else if (n_tokens <= 2) {
    launch(AttentionKernel<2, 0>, 2);
  } else {
    launch(AttentionKernel<4, 0>, 4);
  }
  if (z > 1) {
    hipLaunchKernelGGL(AttentionMergeKernel, dim3(heads, n_tokens),
                       dim3(kThreads), 0, stream, partials, out, heads, d, z);
  }
}

std::uint32_t WmmaCausalAttentionContextLimit(std::uint32_t heads,
                                              std::uint32_t kv_heads,
                                              std::uint32_t d,
                                              std::uint32_t ratio) {
  if (heads != kWmmaQueryHeads || kv_heads != kWmmaKvHeads ||
      d != kWmmaHeadDim || ratio != kWmmaRatio)
    return 0;
  return kWmmaMaxMaskWords * kWmmaRatio * 32;
}

bool WmmaCausalAttention(const float* q, const float* gate,
                         const __half* k_cache, const __half* v_cache,
                         const std::uint32_t* mask, std::uint32_t mask_words,
                         float* out, std::uint32_t n_tokens,
                         std::uint32_t start_pos, std::uint32_t heads,
                         std::uint32_t kv_heads, std::uint32_t d,
                         std::uint32_t ratio, hipStream_t stream,
                         bool last_only, const std::uint32_t* kv_blocks,
                         const KvarnReadView* kvarn,
                         std::uint32_t gate_stride) {
  if (kvarn != nullptr && kvarn->bits != 4 && kvarn->bits != 6)
    throw std::invalid_argument("KVarN bits must be 4 or 6");

  if (WmmaCausalAttentionContextLimit(heads, kv_heads, d, ratio) == 0 ||
      n_tokens == 0) {
    return false;
  }
  if (mask != nullptr) {
    // The kernel's union mask covers the LIVE visible extent; mask_words is
    // only the row stride and may be wider when the session capacity
    // exceeds 262,144 tokens.
    const std::uint64_t live_blocks =
        (std::uint64_t{start_pos} + n_tokens + kWmmaRatio - 1) / kWmmaRatio;
    const std::uint64_t live_words = (live_blocks + 31) / 32;
    if (live_words > kWmmaMaxMaskWords || live_words > mask_words)
      return false;
  }
  if (mask != nullptr) {
    constexpr std::uint32_t kPackedQueries = 4;
    const std::uint32_t first_group =
        last_only ? (n_tokens - 1) / kPackedQueries : 0;
    const dim3 grid(
        (n_tokens + kPackedQueries - 1) / kPackedQueries - first_group,
        kWmmaKvHeads);
    if (kvarn != nullptr && kvarn->bits == 6)
      hipLaunchKernelGGL(
          (WmmaCausalAttentionKernel<kPackedQueries, kWmmaKeys, true, 6>), grid,
          dim3(kThreads), 0, stream, q, gate, k_cache, v_cache, mask,
          mask_words, out, start_pos, n_tokens, first_group, kv_blocks, *kvarn,
          gate_stride);
    else if (kvarn != nullptr)
      hipLaunchKernelGGL(
          (WmmaCausalAttentionKernel<kPackedQueries, kWmmaKeys, true, 4>), grid,
          dim3(kThreads), 0, stream, q, gate, k_cache, v_cache, mask,
          mask_words, out, start_pos, n_tokens, first_group, kv_blocks, *kvarn,
          gate_stride);
    else
      hipLaunchKernelGGL(
          (WmmaCausalAttentionKernel<kPackedQueries, kWmmaKeys, true>), grid,
          dim3(kThreads), 0, stream, q, gate, k_cache, v_cache, mask,
          mask_words, out, start_pos, n_tokens, first_group, kv_blocks,
          KvarnReadView{}, gate_stride);
    return true;
  }
  const std::uint32_t first_group =
      last_only ? (n_tokens - 1) / kWmmaQueryRows : 0;
  const dim3 grid(
      (n_tokens + kWmmaQueryRows - 1) / kWmmaQueryRows - first_group,
      kWmmaKvHeads * (kWmmaGqa / kWmmaHeads));
  if (kvarn != nullptr && kvarn->bits == 6)
    hipLaunchKernelGGL(
        (WmmaCausalAttentionKernel<kWmmaQueryRows, kWmmaKeys, false, 6>), grid,
        dim3(kThreads), 0, stream, q, gate, k_cache, v_cache, mask, mask_words,
        out, start_pos, n_tokens, first_group, kv_blocks, *kvarn, gate_stride);
  else if (kvarn != nullptr)
    hipLaunchKernelGGL(
        (WmmaCausalAttentionKernel<kWmmaQueryRows, kWmmaKeys, false, 4>), grid,
        dim3(kThreads), 0, stream, q, gate, k_cache, v_cache, mask, mask_words,
        out, start_pos, n_tokens, first_group, kv_blocks, *kvarn, gate_stride);
  else
    hipLaunchKernelGGL(
        (WmmaCausalAttentionKernel<kWmmaQueryRows, kWmmaKeys, false>), grid,
        dim3(kThreads), 0, stream, q, gate, k_cache, v_cache, mask, mask_words,
        out, start_pos, n_tokens, first_group, kv_blocks, KvarnReadView{},
        gate_stride);
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
