// Per-layer embeddings: the PLE row fetch and prefetch, and the PLE block.

#include "src/models/qwen38_flash_next/runtime/executor_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

bool Executor::PleFetch(Session& session, std::span<const std::int32_t> tokens,
                        bool speculative, std::string* error_msg) const {
  if (ple_pending_ && !WaitPle(error_msg)) {
    return false;
  }
  const Config& c = config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if ((prefetch_.pending || prefetch_.ready) && !speculative &&
      session.ngram_ == prefetch_.before) {
    // A cache checkpoint or another step budget can end this chunk before
    // or after the hinted one. Rows are per token, so a common prefix still
    // serves every token both cover.
    const auto shared = static_cast<std::size_t>(
        std::ranges::mismatch(tokens, prefetch_.tokens).in1 - tokens.begin());
    if (shared == tokens.size() || shared == prefetch_.tokens.size()) {
      prefetch_.claimed = shared;
      ple_pending_ = true;
      if (shared == prefetch_.tokens.size()) {
        session.ngram_ = prefetch_.after;
      } else {
        HashNgramRows(
            c, session.ngram_, tokens,
            std::span<std::uint32_t>(host_rows_.data(), n * c.ple_heads));
      }
      if (shared == tokens.size())
        return true;
      // Only one table read runs at a time: finish the prefetch, which had a
      // whole chunk to complete, then read the rows past it.
      FinishPrefetch();
      const auto rest = tokens.subspan(shared);
      const auto rows = rest.size() * c.ple_heads;
      HashNgramRows(c, session.ngram_, rest,
                    std::span<std::uint32_t>(host_rows_.data(), rows));
      const NgramGatherOwner owner{rows, &session.ngram_counters_};
      prefetch_.tail = ngram_->StartRead(
          std::span<const std::uint32_t>(host_rows_.data(), rows),
          std::span<float>(host_emb_ + shared * c.PleEmbeddingDim(),
                           rest.size() * c.PleEmbeddingDim()),
          std::span(&owner, 1));
      if (!prefetch_.tail) {
        prefetch_.claimed = 0;
        prefetch_.ready = false;
        ple_pending_ = false;
        AssignError(error_msg, "n-gram table read could not start");
      }
      return prefetch_.tail;
    }
  }
  FinishPrefetch();
  // Only proper prefixes need snapshots; the full batch keeps its live state.
  if (speculative) {
    for (std::uint32_t i = 0; i < n; ++i) {
      HashNgramRows(c, session.ngram_, tokens.subspan(i, 1),
                    std::span<std::uint32_t>(
                        host_rows_.data() + i * c.ple_heads, c.ple_heads));
      if (i + 1 < n) {
        session.ngram_snapshots_[i] = session.ngram_;
      }
    }
  } else {
    HashNgramRows(c, session.ngram_, tokens,
                  std::span<std::uint32_t>(host_rows_.data(), n * c.ple_heads));
  }
  const NgramGatherOwner owner{n * c.ple_heads, &session.ngram_counters_};
  ple_pending_ = ngram_->StartRead(
      std::span<const std::uint32_t>(host_rows_.data(), n * c.ple_heads),
      std::span<float>(host_emb_,
                       static_cast<std::size_t>(n) * c.PleEmbeddingDim()),
      std::span(&owner, 1));
  if (!ple_pending_) {
    AssignError(error_msg, "n-gram table read could not start");
  }
  return ple_pending_;
}

bool Executor::WaitPle(std::string* error_msg) const {
  bool ok = ple_pending_;
  if (ok && prefetch_.claimed != 0) {
    FinishPrefetch();
    ok = prefetch_.ready;
    if (ok) {
      std::copy_n(prefetch_.rows.data(),
                  prefetch_.claimed * config().PleEmbeddingDim(), host_emb_);
    }
    if (prefetch_.tail)
      ok = ngram_->WaitRead() && ok;
    prefetch_.ready = false;
    prefetch_.claimed = 0;
    prefetch_.tail = false;
  } else if (ok) {
    ok = ngram_->WaitRead();
  }
  ple_pending_ = false;
  if (!ok) {
    AssignError(error_msg, "n-gram table read failed");
  }
  return ok;
}

void Executor::PrefetchPle(const Session& session,
                           std::span<const std::int32_t> next) const {
  const Config& c = config();
  if (ple_pending_ || prefetch_.pending || next.empty() ||
      next.size() > options_.max_batch) {
    return;
  }
  // Hashed from the history after this batch, which PleFetch already
  // advanced; the session keeps it until the next batch claims the rows.
  prefetch_.ready = false;
  prefetch_.before = session.ngram_;
  prefetch_.after = session.ngram_;
  const std::span<std::uint32_t> rows(host_rows_.data(),
                                      next.size() * c.ple_heads);
  HashNgramRows(c, prefetch_.after, next, rows);
  prefetch_.tokens.assign(next.begin(), next.end());
  const std::size_t count = next.size() * c.PleEmbeddingDim();
  if (prefetch_.rows.size() < count) {
    prefetch_.rows.resize(count);
  }
  const NgramGatherOwner owner{rows.size(), &session.ngram_counters_};
  prefetch_.pending =
      ngram_->StartRead(rows, std::span<float>(prefetch_.rows.data(), count),
                        std::span(&owner, 1));
}

void Executor::FinishPrefetch() const {
  if (prefetch_.pending) {
    prefetch_.ready = ngram_->WaitRead();
    prefetch_.pending = false;
  }
}

bool Executor::Ple(const DeviceLayer& l, Session& session, std::uint32_t n,
                   float* res, bool speculative, std::string* error_msg,
                   bool embeddings_ready, PrefillCheckpoint* checkpoint) const {
  const Config& c = config();
  if (!embeddings_ready && !WaitPle(error_msg)) {
    return false;
  }
  if (!embeddings_ready && !prefetch_next_.empty()) {
    PrefetchPle(session, prefetch_next_);
    prefetch_next_ = {};
  }
  const std::uint32_t hc_dim = c.HcDim();
  Q8Input emb;
  if (!Quantize(s_.ple_emb, n, c.PleEmbeddingDim(), &emb, error_msg) ||
      !Dense(l.ple_key, emb, s_.ple_key, error_msg) ||
      !Dense(l.ple_value, emb, s_.ple_value, error_msg)) {
    return false;
  }
  RmsNormRows(s_.ple_key, l.ple_norm_key.f32(), s_.ple_key, n, hc_dim,
              c.hc_count, c.rms_eps, stream_);
  RmsNormRows(res, l.ple_norm_query.f32(), s_.ple_query, n, hc_dim, c.hc_count,
              c.rms_eps, stream_);
  PleGate(s_.ple_key, s_.ple_query, s_.ple_value, s_.ple_gated, n,
          c.hidden_size, c.hc_count, stream_);
  RmsNormRows(s_.ple_gated, l.ple_norm_conv.f32(), s_.ple_norm, n, hc_dim,
              c.hc_count, c.rms_eps, stream_);
  if (checkpoint)
    HistoryPrefix(s_.ple_norm, hc_dim, session.ple_history_,
                  checkpoint->state->ple_history_, checkpoint->tokens, hc_dim,
                  c.PleConvHistory(), stream_);
  PleConv(s_.ple_norm, l.ple_conv1d.f32(), session.ple_history_,
          s_.ple_history_scratch, s_.ple_conv,
          speculative ? session.ple_snapshots_ : RollbackRows{}, n, hc_dim,
          c.ple_conv_kernel, c.ple_ngram_size, stream_);
  PleInject(res, s_.ple_gated, s_.ple_conv,
            static_cast<std::size_t>(n) * hc_dim, stream_);
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
