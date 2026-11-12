#include "src/models/qwen38_flash_next/engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <optional>
#include <type_traits>

#include "src/core/gguf_identity.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/quant/exl3/layout.hpp"
#include "src/models/qwen/native_pack.hpp"
#include "src/models/qwen38_flash_next/continuation_hooks.hpp"
#include "src/models/qwen38_flash_next/draft_vocab.hpp"
#include "src/models/qwen38_flash_next/exl3_weights.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "src/models/qwen38_flash_next/native_metadata.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/runtime/device_model.hpp"
#include "src/models/qwen38_flash_next/runtime/executor.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

// gfx1151, served repository prompts: 4096-token chunks prefill 24K-102K
// tokens 6-11% faster than 2048 (pp4096 +5% at d0, +6% at d32K). Each
// chunk's n-gram gather doubles; next-chunk prefetch overlaps that read.
// Prefill logits do not depend on the chunk size
// (pre-budget queries always take the dense attention tiles).
constexpr std::uint32_t kPrefillChunkTokens = 4096;
// A prompt ending this close past a chunk finishes in that chunk: a separate
// tail pass costs about as much as this many more chunk rows.
constexpr std::uint32_t kPrefillTailTokens = 128;

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

constexpr std::array<char, 8> kSessionSnapshotMagic{'Q', 'F', 'N', 'S',
                                                    'E', 'S', 'S', '1'};

/// Host-side session fields ahead of the executor payload: the token
/// history and the logits of the last token.
struct SessionSnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t version;
  std::uint32_t vocab_size;
  std::uint32_t token_count;
  std::uint32_t hidden_rows;
  std::uint64_t executor_bytes;
  MtpLengthState draft_policy;
  std::uint32_t image_identity_bytes;
  std::uint32_t policy_concurrency;
  std::uint32_t reserved{0};
  speculative::PromptLookup<>::State lookup;
};
static_assert(std::is_trivially_copyable_v<SessionSnapshotHeader>);

std::uint64_t SessionSnapshotHostBytes(std::uint32_t token_count,
                                       std::uint32_t vocab_size,
                                       std::size_t image_bytes) {
  return sizeof(SessionSnapshotHeader) + image_bytes +
         std::uint64_t{token_count} * sizeof(std::int32_t) +
         std::uint64_t{vocab_size} * sizeof(float);
}

// Brackets one public execution with the continuation hooks. AfterExecution
// also runs when the execution fails or throws: it only marks the device
// work submitted so far.
class ExecutionScope {
public:
  explicit ExecutionScope(ContinuationHooks* hooks) : hooks_(hooks) {
    if (hooks_)
      hooks_->BeforeExecution();
  }
  ~ExecutionScope() {
    if (hooks_)
      hooks_->AfterExecution();
  }
  ExecutionScope(const ExecutionScope&) = delete;
  ExecutionScope& operator=(const ExecutionScope&) = delete;

private:
  ContinuationHooks* hooks_;
};

/// sampling::CollectLogprobs of `rows` consecutive `vocab`-wide rows of
/// `raw`, row i selecting target(i) with `count` alternatives. Each row runs
/// the same serial full-vocabulary reduction wherever it runs, so the values
/// stay bit-identical. Running the rows concurrently keeps a long chain from
/// holding the executor once per row. A fixed number of workers takes
/// interleaved rows: a 63-row wide chain with one thread per row added 62
/// threads and about 95 MiB of RSS. A non-finite row throws.
template<class Target>
std::vector<std::optional<sampling::TokenLogprobs>> CollectRowLogprobs(
    std::span<const float> raw, std::size_t vocab, std::uint32_t rows,
    const Target& target, std::optional<std::uint32_t> count) {
  constexpr std::uint32_t kLogprobWorkers = 8;
  std::vector<std::optional<sampling::TokenLogprobs>> values(rows);
  const auto workers = std::min(rows, kLogprobWorkers);
  const auto work = [&](std::uint32_t first) {
    for (auto i = first; i < rows; i += workers)
      values[i] = sampling::CollectLogprobs(
          raw.subspan(std::size_t{i} * vocab, vocab),
          static_cast<sampling::TokenId>(target(i)), count);
  };
  std::vector<std::future<void>> others;
  others.reserve(workers ? workers - 1 : 0);
  for (std::uint32_t w = 1; w < workers; ++w)
    others.push_back(
        std::async(std::launch::async | std::launch::deferred, work, w));
  // If a row throws, the remaining futures join as they are destroyed.
  if (workers)
    work(0);
  for (auto& other : others)
    other.get();
  return values;
}

/// Appends rows the device scored (Executor::ForwardScoreRows/ScorePack) for
/// `targets`, refusing a non-finite row as sampling::CollectLogprobs does.
void AppendScored(std::span<const rocm::ScoredLogprob> rows,
                  std::span<const std::int32_t> targets,
                  std::vector<sampling::TokenLogprobs>* scored) {
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (rows[i].nonfinite != 0)
      throw std::runtime_error("non-finite target logits");
    scored->push_back({.selected = {static_cast<sampling::TokenId>(targets[i]),
                                    rows[i].logprob}});
  }
}

}  // namespace

Model::~Model() = default;

bool Model::IsNativeExl3() const noexcept {
  return reader_ &&
         reader_->GetMetadataBool(quant::exl3::kNativeKey).value_or(false);
}

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  if (options.kvarn_bits != 4 && options.kvarn_bits != 6) {
    if (error_msg)
      *error_msg = "KVarN bits must be 4 or 6";
    return nullptr;
  }
  if (options.decode_concurrency == 0 || options.decode_concurrency > 8) {
    AssignError(error_msg, "decode concurrency must be between one and eight");
    return nullptr;
  }
  if (options.max_draft_tokens == 0) {
    AssignError(error_msg, "draft token limit must be positive");
    return nullptr;
  }
  if (!options.mtp_model_path.empty() &&
      options.max_draft_tokens > kMaxMtpDraftTokens) {
    AssignError(error_msg,
                "Flash-Next MTP supports at most seven draft tokens");
    return nullptr;
  }
  m->options_ = options;
  m->reader_ = OpenArtifactMetadata(model_path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  const bool exl3 =
      m->reader_->FindMetadata(quant::exl3::kCheckpointKey) != nullptr;
  // The calibrated sidecar exists only beside a native EXL3 pack.
  if (!exl3 && options.hc_mode == HcMode::kQ8Calibrated) {
    AssignError(error_msg,
                "--hc-mode q8-calibrated requires a native EXL3 pack with its "
                "calibrated hc_q8/ sidecar");
    return nullptr;
  }
  auto weights =
      exl3 ? LoadExl3Model(model_path, *m->reader_, options.hc_mode, error_msg)
           : ModelWeights::Bind(*m->reader_, error_msg);
  if (!weights) {
    return nullptr;
  }
  weights->hc_mode = options.hc_mode;
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  if (options.rope_scaling.has_value()) {
    if (!ValidateRopeScaling(*options.rope_scaling,
                             m->weights_->config.context_length, error_msg)) {
      return nullptr;
    }
    m->weights_->config.rope_scaling = *options.rope_scaling;
  }
  const Config& c = m->weights_->config;
  if (options.max_context == 0 || options.max_context > c.MaxContextLength()) {
    AssignError(error_msg, "context exceeds the model's " +
                               std::to_string(c.MaxContextLength()) +
                               " tokens");
    return nullptr;
  }
  try {
    m->vision_ = qwen::vision::Encoder::Open(
        model_path, options.vision_model_path, c.hidden_size);
  } catch (const std::exception& e) {
    AssignError(error_msg, e.what());
    return nullptr;
  }
  m->tokenizer_ =
      tokenization::QwenTokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  if (c.ple_layer >= 0) {
    m->ngram_ =
        NgramTable::ForModel(model_path, *m->reader_, *m->weights_, error_msg);
    if (!m->ngram_) {
      return nullptr;
    }
  }
  if (!options.mtp_model_path.empty()) {
    m->mtp_reader_ = OpenArtifactMetadata(options.mtp_model_path, error_msg);
    if (!m->mtp_reader_) {
      return nullptr;
    }
    const bool mtp_exl3 =
        m->mtp_reader_->FindMetadata(quant::exl3::kCheckpointKey) != nullptr;
    if (mtp_exl3 != exl3) {
      AssignError(error_msg, "trunk and MTP weight formats must match");
      return nullptr;
    }
    auto mtp = mtp_exl3 ? LoadExl3Mtp(options.mtp_model_path, *m->mtp_reader_,
                                      *m->weights_, error_msg)
                        : MtpWeights::Bind(*m->mtp_reader_, c, error_msg);
    if (!mtp) {
      return nullptr;
    }
    m->mtp_weights_ = std::make_unique<MtpWeights>(std::move(*mtp));
  }
  m->device_ = rocm::DeviceModel::Upload(*m->weights_, *m->reader_,
                                         m->mtp_weights_.get(),
                                         m->mtp_reader_.get(), error_msg);
  if (!m->device_) {
    return nullptr;
  }
  if (m->mtp_weights_ && m->device_->output().rows > kMtpDraftVocab) {
    std::vector<std::int32_t> specials;
    for (std::uint32_t id = kMtpDraftVocab; id < m->device_->output().rows;
         ++id)
      if (m->tokenizer_->IsSpecialToken(id))
        specials.push_back(static_cast<std::int32_t>(id));
    // A vocabulary with more special tokens than table rows drafts over the
    // full head.
    if (specials.size() < kMtpDraftExtraRows) {
      m->draft_specials_ = static_cast<std::uint32_t>(specials.size());
      if (!m->device_->SetDraftVocab(kMtpDraftVocab, std::move(specials),
                                     error_msg))
        return nullptr;
      m->draft_subset_ = true;
    }
  }
  rocm::Executor::Options exec;
  exec.max_batch = m->PrefillThroughCapacity();
  exec.max_logit_rows =
      m->mtp_weights_
          ? static_cast<std::uint32_t>(std::min<std::uint64_t>(
                exec.max_batch, std::uint64_t{options.max_draft_tokens} + 1))
          : 1;
  exec.max_speculative = exec.max_logit_rows;
  exec.max_context = options.max_context;
  exec.kvarn = options.kvarn;
  exec.kvarn_bits = options.kvarn_bits;
  m->executor_ =
      rocm::Executor::Create(*m->device_, m->ngram_.get(), exec, error_msg);
  if (!m->executor_) {
    return nullptr;
  }
  try {
    const auto& c = m->config();
    if (auto profiles = LoadAbliterationProfiles(options.abliteration_profiles,
                                                 c.hidden_size, c.num_layers,
                                                 m->ArtifactIdentity());
        !profiles.empty() &&
        !m->executor_->LoadAbliteration(std::move(profiles), error_msg))
      return nullptr;
    // Each dynamic profile turns ON with its defaults.
    for (const auto& profile : m->executor_->AbliterationProfiles())
      if (profile->trigger)
        m->dynamic_candidates_.push_back(
            {profile->name, *m->executor_->DetectorColumn(profile->name),
             profile->trigger->threshold,
             ResolveAbliterationSettings(*profile)});
    qwen::ValidateNativeSources(*m->reader_);
    if (m->mtp_reader_)
      qwen::ValidateNativeSources(*m->mtp_reader_);
  } catch (const std::exception& e) {
    AssignError(error_msg, e.what());
    return nullptr;
  }
  return m;
}

std::unique_ptr<Session> Model::CreateSession(
    core::SessionMode mode, std::uint32_t max_context, std::string* error_msg,
    std::shared_ptr<hip::KvBlockPool> pool) {
  if (max_context == 0 || max_context > options_.max_context) {
    AssignError(error_msg, "session context is outside the model limits");
    return nullptr;
  }
  auto native =
      executor_->CreateSession(mode, max_context, error_msg, std::move(pool));
  if (!native) {
    return nullptr;
  }
  if (vision_) {
    // Image-capable slots can first serve text. Prepare the stable RoPE
    // descriptor before any graphs capture a null pointer: replacing it
    // later destroys those graphs during image admission (5–12 ms here).
    try {
      native->PrepareVision(executor_->stream());
    } catch (const std::exception& exception) {
      AssignError(error_msg, exception.what());
      return nullptr;
    }
  }
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(native)));
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  std::vector<std::int32_t> out;
  for (auto id : tokenizer_->Encode(text)) {
    out.push_back(static_cast<std::int32_t>(id));
  }
  return out;
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<tokenization::TokenId> ids(tokens.begin(), tokens.end());
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(static_cast<tokenization::TokenId>(token));
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return tokenizer_->IsStopToken(static_cast<tokenization::TokenId>(token));
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

std::uint32_t Model::PrefillCapacity() const noexcept {
  return std::min(kPrefillChunkTokens, options_.max_context);
}

std::uint32_t Model::PrefillThroughCapacity() const noexcept {
  return std::min(kPrefillChunkTokens + kPrefillTailTokens,
                  options_.max_context);
}

std::uint32_t Model::PrefillCheckpointTail() const noexcept {
  return executor_->PrefillCheckpointTail();
}

bool Model::HasMtp() const noexcept {
  return device_->has_mtp();
}

std::uint64_t Model::MixedKvOnlyCohorts() const noexcept {
  return executor_->MixedKvOnlyCohorts();
}

bool Model::InDraftSubset(std::int32_t token) const {
  return token >= 0 && (static_cast<std::uint32_t>(token) < kMtpDraftVocab ||
                        tokenizer_->IsSpecialToken(
                            static_cast<tokenization::TokenId>(token)));
}

std::string Model::ModelName() const {
  return std::string(reader_->GetMetadataString("general.name")
                         .value_or("Qwen3.8-Flash-Next"));
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

hip::KvBlockGeometry Model::KvGeometry(core::SessionMode mode,
                                       std::uint32_t context) const {
  return executor_->KvGeometry(mode, context);
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes() + (vision_ ? vision_->ResidentBytes() : 0);
}

std::size_t Model::SessionBytes(core::SessionMode mode,
                                std::uint32_t context) const noexcept {
  const std::size_t vision =
      vision_ ? std::size_t{context} * (config().hidden_size * sizeof(float) +
                                        3 * sizeof(std::int32_t)) +
                    64
              : 0;
  return BaseSessionBytes(mode, context) + vision;
}

std::size_t Model::BaseSessionBytes(core::SessionMode mode,
                                    std::uint32_t context) const noexcept {
  // Request-specific images are budgeted from their prepared dimensions by
  // the serving runner, not as max_context image tokens for every slot.
  return executor_->SessionBytes(mode, context,
                                 mode == core::SessionMode::kSpeculative
                                     ? executor_->max_speculative() - 1
                                     : 0);
}

std::size_t Model::DeferredScratchBytes() const {
  return executor_->DeferredScratchBytes();
}

static_assert(Model::kWideCopyRows == rocm::Executor::kWideRows);

void Model::LimitCopyRows(std::uint32_t rows) noexcept {
  executor_->LimitCopyRows(rows);
}

bool Model::ReserveDecodeScratch(std::string* error_msg) {
  return executor_->ReserveDecodeScratch(error_msg);
}

std::size_t Session::AllocatedBytes() const noexcept {
  return session_->AllocatedBytes();
}

Session::MemoryUsage Session::Memory() const noexcept {
  return {
      session_->AllocatedBytes(), session_->RollbackBytes(),
      session_->VisionBytes(), session_->GraphCount(),
      (tokens_.capacity() + draft_added_.capacity()) * sizeof(std::int32_t) +
          (logits_.capacity() + verify_logits_.capacity()) * sizeof(float)};
}

NgramTotals Session::NgramRowTotals() const noexcept {
  return session_->NgramRowTotals();
}

bool Session::MtpEnabled() const noexcept {
  return session_->mtp_enabled();
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> session)
    : model_(std::move(model)),
      session_(std::move(session)),
      draft_length_(model_->options_.max_draft_tokens,
                    model_->DecodeConcurrency()),
      draft_vocab_(kMtpDraftExtraRows - model_->draft_specials_) {
  logits_.resize(model_->VocabSize());
}

bool Session::SyncDraftVocab(std::string* error_msg) {
  if (!model_->draft_subset_ || !MtpEnabled())
    return true;
  auto& exec = *model_->executor_;
  const auto first = model_->draft_specials_;
  if (draft_vocab_stale_ || draft_vocab_.scanned() > tokens_.size()) {
    if (!exec.ClearDraftRows(*session_, first, error_msg))
      return false;
    draft_vocab_.Reset();
    draft_vocab_stale_ = false;
  }
  const auto tabled = static_cast<std::uint32_t>(draft_vocab_.rare().size());
  draft_added_.clear();
  draft_vocab_.CatchUp(
      tokens_, [&](std::int32_t token) { return model_->InDraftSubset(token); },
      &draft_added_);
  return exec.AppendDraftRows(*session_, first + tabled, draft_added_,
                              error_msg);
}

Session::~Session() = default;

void Session::ResetDraftPolicy() {
  const ExecutionScope execution(continuation_hooks_);
  draft_length_.Reset();
  // A fully reused prompt can start decoding without SyncImpl. It still
  // starts a new request: do not inherit a copy cursor or widen its first step.
  lookup_.BeginRequest();
  copy_rows_ = 0;
  copy_streak_ = 0;
}

std::uint32_t Session::Position() const noexcept {
  return session_->position();
}
std::uint32_t Session::ContextSize() const noexcept {
  return session_->max_context();
}

void Session::Reset() {
  ResetImpl(false);
}

bool Session::ScratchReusable() const noexcept {
  // Only a valid session keeps its MTP scratch for the next request. Keeping
  // it after a stopped or cancelled request as well (0e0ebad8) left every
  // later short prefill about 7 ms (5 %) slower on gfx1151 and slowed
  // concurrent short replies 6-7 % (S3, docs/GFX1151-KNOWHOW.md). The
  // cancelled-peer stall it targeted did not recur without it (four FP16
  // runs and one KVarN K4V4 run, 2026-11-10).
  return IsValid();
}

void Session::ResetForReuse() {
  ResetImpl(ScratchReusable());
}

void Session::ResetImpl(bool reuse_mtp) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeReset();
  valid_ = false;
  session_->Reset(reuse_mtp);
  tokens_.clear();
  ForgetDraftVocab();
  lookup_.Clear();
  hidden_base_ = 0;
  draft_length_.Reset();
  model_->executor_->MtpRewind(*session_, 0);
  if (continuation_hooks_)
    continuation_hooks_->AfterReset();
  valid_ = true;
}

void Session::SetCancellationCheck(std::function<bool()> check) {
  session_->SetCancellationCheck(std::move(check));
}

std::string Model::ArtifactIdentity() const {
  return reader_->GetMetadataBool(quant::exl3::kNativeKey).value_or(false)
             ? qwen::NativeArtifactIdentity(*reader_)
             : core::GgufIdentityHex(*reader_);
}

std::span<const std::shared_ptr<const qwen::AbliterationProfile>>
Model::AbliterationProfiles() const {
  return executor_->AbliterationProfiles();
}

std::shared_ptr<DynamicAbliteration> Model::StartDynamicAbliteration() const {
  return dynamic_candidates_.empty()
             ? nullptr
             : std::make_shared<DynamicAbliteration>(dynamic_candidates_);
}

void Session::SetActivationObserver(ActivationObserver observer) {
  session_->SetActivationObserver(std::move(observer));
}

void Session::ConfigureAbliteration(
    std::shared_ptr<const AbliterationSettings> settings) {
  const ExecutionScope execution(continuation_hooks_);
  const auto* active = session_->Abliteration();
  if (settings ? active && active->identity == settings->identity : !active)
    return;
  if (settings && !model_->executor_->FindAbliteration(settings->profile))
    throw std::invalid_argument("abliteration profile is not loaded");
  if (!tokens_.empty())
    Reset();
  session_->ConfigureAbliteration(std::move(settings));
}

void Session::ConfigureVision(
    std::shared_ptr<const qwen::vision::Prompt> prompt) {
  const ExecutionScope execution(continuation_hooks_);
  const auto identity = prompt ? prompt->IdentityForPrefix(tokens_.size())
                               : std::span<const std::uint8_t>{};
  if (!tokens_.empty() &&
      !std::ranges::equal(identity, ImageIdentity(tokens_.size())))
    Reset();
  const bool was_valid = valid_;
  valid_ = false;
  session_->ConfigureVision(prompt, model_->vision_,
                            model_->executor_->stream());
  image_prompt_ = std::move(prompt);
  valid_ = was_valid;
}

std::span<const std::uint8_t> Session::ImageIdentity(
    std::size_t token_count) const {
  return image_prompt_ ? image_prompt_->IdentityForPrefix(token_count)
                       : std::span<const std::uint8_t>{};
}

const AbliterationSettings* Session::ActiveAbliteration() const {
  return session_->Abliteration();
}

void Session::ConfigureDynamicAbliteration(
    std::shared_ptr<DynamicAbliteration> state) {
  if (state && session_->Abliteration())
    throw std::logic_error(
        "dynamic abliteration starts on a session that is OFF");
  dynamic_ = std::move(state);
  session_->SetDetection(dynamic_ != nullptr);
}

void Session::ApplyDynamicAbliteration() {
  if (!dynamic_ || dynamic_->engaged())
    return;
  const auto committed = static_cast<std::uint32_t>(tokens_.size());
  dynamic_->Observe(session_->DetectorScores(), committed);
  auto settings = dynamic_->Engage(committed);
  if (!settings)
    return;
  // A1: the committed tokens keep their OFF state (no reset); every later
  // trunk pass projects. Detection stops: one profile at a time, sticky.
  session_->SetDetection(false);
  session_->ConfigureAbliteration(std::move(settings));
}

std::optional<DynamicAbliterationReport> Session::DynamicReport() {
  if (!dynamic_)
    return std::nullopt;
  if (!dynamic_->engaged())
    dynamic_->Observe(session_->DetectorScores(),
                      static_cast<std::uint32_t>(tokens_.size()));
  return dynamic_->Report();
}

std::vector<std::uint8_t> Session::ContinuationIdentity(
    std::size_t tokens) const {
  const auto image = ImageIdentity(tokens);
  // Up to a dynamic profile's switch point the tokens were computed OFF.
  const auto* settings = session_->AbliterationFor(tokens);
  return settings ? AbliterationInputIdentity(image, settings->identity)
                  : std::vector<std::uint8_t>(image.begin(), image.end());
}

std::uint32_t Session::KeptHiddenRows() const noexcept {
  return MtpEnabled()
             ? static_cast<std::uint32_t>(tokens_.size() - hidden_base_)
             : 0;
}

std::uint64_t Session::SnapshotBytes() const {
  if (!valid_)
    return 0;
  return SessionSnapshotHostBytes(static_cast<std::uint32_t>(tokens_.size()),
                                  model_->VocabSize(),
                                  ContinuationIdentity(tokens_.size()).size()) +
         model_->executor_->SnapshotBytes(*session_, KeptHiddenRows());
}

std::uint64_t Session::PrefillCheckpointBytes(std::uint32_t position) const {
  if (!session_->VisionLayout().images.empty())
    return 0;
  const auto payload =
      model_->executor_->PrefillCheckpointBytes(*session_, position);
  return payload
             ? SessionSnapshotHostBytes(position, model_->VocabSize(),
                                        ContinuationIdentity(position).size()) +
                   payload
             : 0;
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg, SnapshotMode mode) const {
  if (!valid_ || tokens_.empty() || tokens_.size() != session_->position() ||
      tokens_.size() > std::numeric_limits<std::uint32_t>::max()) {
    AssignError(error_msg, "snapshot needs a synced, non-empty context");
    return nullptr;
  }
  if (!session_->CheckCancellation(error_msg))
    return nullptr;
  return SaveSnapshotImpl(*session_, tokens_, logits_, KeptHiddenRows(),
                          error_msg, mode);
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshotImpl(
    const rocm::Session& state, std::span<const std::int32_t> tokens,
    std::span<const float> logits, std::uint32_t hidden_rows,
    std::string* error_msg, SnapshotMode mode) const {
  const auto token_count = static_cast<std::uint32_t>(tokens.size());
  const auto identity = ContinuationIdentity(token_count);
  const std::uint64_t executor_bytes =
      model_->executor_->SnapshotBytes(state, hidden_rows);
  const std::uint64_t host_bytes = SessionSnapshotHostBytes(
      token_count, model_->VocabSize(), identity.size());
  std::unique_ptr<SessionSnapshot> snapshot(
      new SessionSnapshot(host_bytes + executor_bytes));
  std::uint8_t* out = snapshot->data_.get();
  const SessionSnapshotHeader header{
      .magic = kSessionSnapshotMagic,
      .version = kSnapshotPayloadVersion,
      .vocab_size = model_->VocabSize(),
      .token_count = token_count,
      .hidden_rows = hidden_rows,
      .executor_bytes = executor_bytes,
      .draft_policy = draft_length_.State(),
      .image_identity_bytes = static_cast<std::uint32_t>(identity.size()),
      .policy_concurrency = model_->DecodeConcurrency(),
      .reserved = state.AbliterationFor(token_count) ? 1u : 0u,
      .lookup = lookup_.Save(),
  };
  std::memcpy(out, &header, sizeof(header));
  out += sizeof(header);
  if (!identity.empty())
    std::memcpy(out, identity.data(), identity.size());
  out += identity.size();
  std::memcpy(out, tokens.data(), tokens.size_bytes());
  out += tokens.size_bytes();
  std::memcpy(out, logits.data(), logits.size_bytes());
  out += logits.size_bytes();
  if (!model_->executor_->SaveSnapshot(
          state, hidden_rows,
          std::span<std::uint8_t>(out,
                                  static_cast<std::size_t>(executor_bytes)),
          error_msg,
          mode == SnapshotMode::kBorrowed ? &snapshot->deferred_ : nullptr,
          snapshot->data_)) {
    return nullptr;
  }
  return snapshot;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeRestore();
  return RestoreSnapshotImpl({snapshot.data_.get(), snapshot.size_},
                             snapshot.deferred_.get(), error_msg);
}

bool Session::OwnsSnapshot(const SessionSnapshot& snapshot) const noexcept {
  return snapshot.deferred_ && session_->OwnsSnapshot(*snapshot.deferred_);
}

std::uint64_t SessionSnapshot::DeviceBytes() const {
  return deferred_ ? deferred_->DeviceBytes() : 0;
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeRestore();
  return RestoreSnapshotImpl(payload, nullptr, error_msg);
}

bool Session::RestoreSnapshotImpl(std::span<const std::uint8_t> payload,
                                  const rocm::SnapshotState* deferred,
                                  std::string* error_msg) {
  SessionSnapshotHeader header{};
  if (payload.size() < sizeof(header)) {
    AssignError(error_msg, "session snapshot is truncated");
    return false;
  }
  std::memcpy(&header, payload.data(), sizeof(header));
  if (header.magic != kSessionSnapshotMagic ||
      header.version != kSnapshotPayloadVersion) {
    AssignError(error_msg, "session snapshot format is not supported");
    return false;
  }
  if (header.vocab_size != model_->VocabSize() || header.token_count == 0 ||
      header.policy_concurrency != model_->DecodeConcurrency() ||
      header.reserved != (session_->AbliterationEnabled() ? 1u : 0u) ||
      header.token_count > ContextSize() ||
      (header.image_identity_bytes != 0 && header.image_identity_bytes != 32) ||
      payload.size() != SessionSnapshotHostBytes(header.token_count,
                                                 header.vocab_size,
                                                 header.image_identity_bytes) +
                            header.executor_bytes) {
    AssignError(error_msg, "session snapshot does not fit this session");
    return false;
  }
  auto restored_policy = draft_length_;
  if (!restored_policy.Restore(header.draft_policy)) {
    AssignError(error_msg, "session snapshot draft policy is invalid");
    return false;
  }
  if (!speculative::PromptLookup<>::Valid(header.lookup, header.token_count)) {
    AssignError(error_msg, "session snapshot prompt-lookup state is invalid");
    return false;
  }
  const std::uint8_t* in = payload.data() + sizeof(header);
  std::vector<std::uint8_t> image_identity(in,
                                           in + header.image_identity_bytes);
  if (!image_identity.empty() &&
      !std::ranges::equal(image_identity,
                          ContinuationIdentity(header.token_count))) {
    AssignError(error_msg,
                "image snapshot requires its matching prompt attachment");
    return false;
  }
  in += header.image_identity_bytes;
  std::vector<std::int32_t> tokens(header.token_count);
  std::memcpy(tokens.data(), in, tokens.size() * sizeof(std::int32_t));
  in += tokens.size() * sizeof(std::int32_t);
  std::vector<float> logits(header.vocab_size);
  std::memcpy(logits.data(), in, logits.size() * sizeof(float));
  in += logits.size() * sizeof(float);

  if (image_identity.empty())
    ConfigureVision(nullptr);
  valid_ = false;
  rocm::Executor::SnapshotInfo info;
  const auto remaining = ContextSize() - header.token_count;
  auto next_drafts = MtpEnabled()
                         ? restored_policy.Choose(remaining ? remaining - 1 : 0,
                                                  header.token_count)
                         : 0;
  if (MtpEnabled() && model_->draft_subset_) {
    // The next request's sampler/head is not known yet. Avoid discarding
    // existing scratch that either eligible policy can use after restore.
    next_drafts = std::max(next_drafts,
                           restored_policy.Choose(remaining ? remaining - 1 : 0,
                                                  header.token_count, 0, true));
  }
  if (!model_->executor_->RestoreSnapshot(
          *session_,
          std::span<const std::uint8_t>(
              in, static_cast<std::size_t>(header.executor_bytes)),
          &info, error_msg, next_drafts, deferred)) {
    Reset();
    return false;
  }
  if (info.position != header.token_count ||
      info.hidden_rows != header.hidden_rows) {
    Reset();
    AssignError(error_msg, "session snapshot positions are inconsistent");
    return false;
  }
  tokens_ = std::move(tokens);
  ForgetDraftVocab();
  lookup_.Restore(header.lookup);
  logits_ = std::move(logits);
  hidden_base_ = info.position - info.hidden_rows;
  draft_token_ = 0;
  draft_length_ = restored_policy;
  stats_ = {};
  valid_ = true;
  return true;
}

SessionSnapshot::SessionSnapshot(std::uint64_t size)
    : data_(new std::uint8_t[size]), size_(size) {
  // Leave append-only sections untouched: a warm checkpoint borrows those
  // rows from the live session, so faulting their pages here is wasted work.
}

SessionSnapshot::~SessionSnapshot() = default;

std::span<const std::uint8_t> SessionSnapshot::bytes() const {
  if (deferred_)
    deferred_->Materialize();
  return {data_.get(), size_};
}

bool SessionSnapshot::CopyTo(
    std::span<std::uint8_t> destination) const noexcept {
  if (destination.size() != size_)
    return false;
  try {
    const auto payload = bytes();
    std::memcpy(destination.data(), payload.data(), payload.size());
    return true;
  } catch (...) {
    return false;
  }
}

bool Session::DraftReplay(std::int32_t next_token,
                          std::vector<std::int32_t>* replay,
                          std::int32_t* hidden_row,
                          std::string* error_msg) const {
  // The draft block trails the trunk: MTP position i consumes token i+1 and
  // the trunk's hidden of position i, so positions up to the current one are
  // replayed once their successor token is known. The session keeps hidden
  // rows of positions [hidden_base_, tokens_.size()).
  rocm::Executor& exec = *model_->executor_;
  const auto size = static_cast<std::uint32_t>(tokens_.size());
  const std::uint32_t mp = exec.MtpPosition(*session_);
  if (mp >= size) {
    return true;
  }
  if (mp < hidden_base_) {
    AssignError(error_msg, "draft block fell behind the kept hidden rows");
    return false;
  }
  replay->assign(tokens_.begin() + mp + 1, tokens_.end());
  replay->push_back(next_token);
  *hidden_row = static_cast<std::int32_t>(mp - hidden_base_);
  return true;
}

bool Session::DraftCatchUp(std::int32_t next_token, bool propose,
                           std::string* error_msg,
                           MtpCandidateLogits* candidates, bool draft_subset,
                           bool token_with_candidates) {
  std::vector<std::int32_t> replay;
  std::int32_t hidden_row = 0;
  if (!DraftReplay(next_token, &replay, &hidden_row, error_msg))
    return false;
  if (replay.empty())
    return true;
  auto& exec = *model_->executor_;
  if (!exec.MtpForward(
          *session_, replay, hidden_row,
          {.token = propose && (candidates == nullptr || token_with_candidates)
                        ? &draft_token_
                        : nullptr,
           .candidates = candidates,
           .kv_only = !propose,
           .draft_subset = draft_subset},
          error_msg)) {
    return false;
  }
  return true;
}

bool Session::ProbeDraftChain(std::int32_t anchor, std::uint32_t depth,
                              DraftProbe* probe, std::string* error_msg) {
  const ExecutionScope execution(continuation_hooks_);
  if (!valid_ || probe == nullptr || !MtpEnabled() || tokens_.empty() ||
      depth == 0 || depth > kMaxMtpDraftTokens) {
    AssignError(error_msg,
                "draft probe needs a synced MTP session and 1..7 drafts");
    return false;
  }
  *probe = {};
  rocm::Executor& exec = *model_->executor_;
  const auto base = static_cast<std::uint32_t>(tokens_.size());
  valid_ = false;
  // The head choice of a greedy single-session cycle (PrepareDecode with
  // MtpDraftSubsetAllowed true and no lookup bypass).
  if (!SyncDraftVocab(error_msg))
    return false;
  probe->small_head = model_->draft_subset_ &&
                      draft_vocab_.Fits(anchor, [&](std::int32_t token) {
                        return model_->InDraftSubset(token);
                      });
  // DraftCatchUp(anchor, true, ...), also reporting candidates.
  std::vector<std::int32_t> replay;
  std::int32_t hidden_row = 0;
  if (!DraftReplay(anchor, &replay, &hidden_row, error_msg))
    return false;
  if (replay.empty()) {
    AssignError(error_msg, "draft probe has nothing to catch up");
    return false;
  }
  for (std::uint32_t d = 0; d < depth; ++d) {
    std::int32_t token = 0;
    const bool first = d == 0;
    const std::span<const std::int32_t> input =
        first ? std::span<const std::int32_t>(replay)
              : std::span<const std::int32_t>(&probe->tokens[d - 1], 1);
    if (!exec.MtpForward(*session_, input, first ? hidden_row : -1,
                         {.token = &token,
                          .candidates = &probe->candidates[d],
                          .draft_subset = probe->small_head},
                         error_msg)) {
      return false;
    }
    probe->tokens[d] = token;
  }
  probe->depth = depth;
  // FinishDecode's rewind: the committed catch-up rows stay, drafts go.
  exec.MtpRewind(*session_, base);
  valid_ = true;
  return true;
}

bool Session::DraftCatchUpBatch(std::span<const AdvanceRequest> requests,
                                std::string* error_msg,
                                std::span<const std::uint8_t> propose) {
  if (requests.empty())
    return true;
  if (!propose.empty() && propose.size() != requests.size()) {
    AssignError(error_msg, "draft catch-up head count differs from requests");
    return false;
  }
  auto& exec = *requests.front().session->model_->executor_;
  std::vector<std::vector<std::int32_t>> replays(requests.size());
  std::vector<rocm::Executor::MtpBatchItem> items;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& session = *requests[i].session;
    if (!session.MtpEnabled())
      continue;
    std::int32_t hidden_row = 0;
    if (!session.DraftReplay(requests[i].token, &replays[i], &hidden_row,
                             error_msg))
      return false;
    if (replays[i].empty())
      continue;
    items.push_back({session.session_.get(), replays[i], hidden_row,
                     propose.empty() || !propose[i]});
  }
  return items.empty() || exec.MtpForwardBatch(items, error_msg);
}

bool Session::Feed(std::span<const std::int32_t> tokens, std::string* error_msg,
                   bool prefill, std::uint32_t boundary,
                   std::unique_ptr<SessionSnapshot>* checkpoint,
                   double* capture_ms, std::span<const std::int32_t> after,
                   const PrefillCapture& consume) {
  rocm::Executor& exec = *model_->executor_;
  const std::size_t ratio =
      std::max<std::uint32_t>(1, model_->config().compress_ratio);
  for (std::size_t off = 0; off < tokens.size();) {
    const auto remaining = tokens.size() - off;
    const auto capacity = remaining <= exec.max_batch()
                              ? exec.max_batch()
                              : model_->PrefillCapacity();
    std::size_t n = std::min<std::size_t>(capacity, remaining);
    // A following pass starts on a selection block (WmmaCausalAttention).
    if (const auto end = tokens_.size() + n;
        prefill && off + n < tokens.size() && end % ratio != 0 &&
        n > end % ratio)
      n -= end % ratio;
    // Extended-context attention can fall back from WMMA to the per-token
    // kernel. Its checkpoint must end a pass: the fused in-pass path requires
    // WMMA for both query groups. Preserve ordinary split-pass arithmetic on
    // either side of the boundary. Use the actual launch capability rather
    // than the model's independently configured native context length.
    if ((checkpoint || consume) && boundary > tokens_.size() &&
        boundary < tokens_.size() + n &&
        tokens_.size() + n >
            rocm::WmmaCausalAttentionContextLimit(
                model_->config().num_heads, model_->config().num_kv_heads,
                model_->config().head_dim, model_->config().compress_ratio)) {
      n = boundary - tokens_.size();
    }
    const auto chunk = tokens.subspan(off, n);
    if (MtpEnabled() && !tokens_.empty() &&
        !DraftCatchUp(chunk[0], false, error_msg)) {
      return false;
    }
    const auto mode = prefill ? rocm::Executor::ForwardMode::kPrefill
                              : rocm::Executor::ForwardMode::kDecode;
    std::unique_ptr<rocm::PrefillCheckpoint> capture;
    if ((checkpoint || consume) && boundary > tokens_.size() &&
        boundary <= tokens_.size() + n) {
      capture = exec.MakePrefillCheckpoint(
          *session_, chunk,
          static_cast<std::uint32_t>(boundary - tokens_.size()), error_msg);
      if (!capture)
        return false;
    }
    // The chunk this loop, or the following Sync, takes next.
    const auto rest = off + n < tokens.size() ? tokens.subspan(off + n) : after;
    const auto next = rest.first(std::min<std::size_t>(
        rest.size() <= exec.max_batch() ? exec.max_batch()
                                        : model_->PrefillCapacity(),
        rest.size()));
    if (!exec.Forward(*session_, chunk, 1, logits_.data(), mode, error_msg,
                      capture.get(), next)) {
      return false;
    }
    const auto kept = capture && capture->tokens < n ? n - capture->tokens : n;
    hidden_base_ = static_cast<std::uint32_t>(
        tokens_.size() + n -
        std::min<std::size_t>(kept, exec.max_speculative()));
    tokens_.insert(tokens_.end(), chunk.begin(), chunk.end());
    if (capture) {
      const auto started = std::chrono::steady_clock::now();
      if (consume) {
        consume(*capture);
      } else {
        *checkpoint = SaveSnapshotImpl(
            *capture->state,
            std::span<const std::int32_t>(tokens_).first(boundary),
            capture->logits, capture->hidden_rows, error_msg,
            SnapshotMode::kBorrowed);
        if (capture_ms)
          *capture_ms += std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count();
        if (!*checkpoint)
          return false;
      }
    }
    off += n;
  }
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt, std::string* error_msg,
                   std::span<const std::int32_t> next) {
  const ExecutionScope execution(continuation_hooks_);
  return SyncImpl(prompt, error_msg, 0, nullptr, nullptr, next);
}

bool Session::SyncThrough(std::span<const std::int32_t> prompt,
                          std::uint32_t boundary,
                          std::unique_ptr<SessionSnapshot>* checkpoint,
                          std::string* error_msg, double* capture_ms) {
  const ExecutionScope execution(continuation_hooks_);
  if (!checkpoint || boundary == 0 || boundary > prompt.size() ||
      prompt.size() - boundary > model_->PrefillCheckpointTail() ||
      boundary <= tokens_.size()) {
    AssignError(error_msg,
                "checkpoint must extend the frontier in the text tail");
    return false;
  }
  checkpoint->reset();
  if (capture_ms)
    *capture_ms = 0;
  return SyncImpl(prompt, error_msg, boundary, checkpoint, capture_ms);
}

bool Session::SyncThroughComponents(std::span<const std::int32_t> prompt,
                                    std::uint32_t boundary,
                                    const PrefillCapture& capture,
                                    std::string* error_msg) {
  const ExecutionScope execution(continuation_hooks_);
  if (!capture || boundary == 0 || boundary > prompt.size() ||
      prompt.size() - boundary > model_->PrefillCheckpointTail() ||
      boundary <= tokens_.size()) {
    AssignError(
        error_msg,
        "component checkpoint must extend the frontier in the text tail");
    return false;
  }
  return SyncImpl(prompt, error_msg, boundary, nullptr, nullptr, {}, capture);
}

bool Session::SyncImpl(std::span<const std::int32_t> prompt,
                       std::string* error_msg, std::uint32_t boundary,
                       std::unique_ptr<SessionSnapshot>* checkpoint,
                       double* capture_ms, std::span<const std::int32_t> next,
                       const PrefillCapture& capture) {
  if (prompt.empty()) {
    AssignError(error_msg, "prompt is empty");
    return false;
  }
  if (prompt.size() > ContextSize()) {
    AssignError(error_msg, "prompt exceeds the session context");
    return false;
  }
  lookup_.BeginRequest();
  copy_rows_ = 0;
  copy_streak_ = 0;
  if (!valid_)
    Reset();
  // Recurrent state cannot be rewound, so any divergence restarts the
  // session; an extension only feeds the new tail.
  std::size_t common = 0;
  while (common < tokens_.size() && common < prompt.size() &&
         tokens_[common] == prompt[common]) {
    ++common;
  }
  if (common == prompt.size() && common == tokens_.size()) {
    return true;
  }
  if (common != tokens_.size()) {
    Reset();
    common = 0;
  }
  valid_ = false;
  const bool ok = Feed(prompt.subspan(common), error_msg, true, boundary,
                       checkpoint, capture_ms, next, capture);
  valid_ = ok;
  return ok;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  const ExecutionScope execution(continuation_hooks_);
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  ApplyDynamicAbliteration();
  if (tokens_.size() >= ContextSize()) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  valid_ = false;
  const bool ok = Feed(std::span<const std::int32_t>(&token, 1), error_msg);
  valid_ = ok;
  return ok;
}

bool Session::ScoreTokens(std::span<const std::int32_t> tokens,
                          std::vector<sampling::TokenLogprobs>* scored,
                          std::string* error_msg) {
  if (!valid_ || tokens_.empty()) {
    AssignError(error_msg, "scoring needs a prefilled session");
    return false;
  }
  const std::vector<std::int32_t> prompt = tokens_;
  return ScorePrompt(prompt, tokens, scored, error_msg);
}

bool Session::ScorePrompt(std::span<const std::int32_t> prompt,
                          std::span<const std::int32_t> tokens,
                          std::vector<sampling::TokenLogprobs>* scored,
                          std::string* error_msg) {
  try {
    // A hook refusing execution (an invalid continuation slot) fails this
    // request like any other error.
    const ExecutionScope execution(continuation_hooks_);
    return ScorePromptImpl(prompt, tokens, scored, error_msg);
  } catch (const std::exception& error) {
    // CollectLogprobs refuses non-finite logits.
    valid_ = false;
    AssignError(error_msg, error.what());
    return false;
  }
}

bool Session::BeginScore(std::span<const std::int32_t> prompt,
                         std::span<const std::int32_t> tokens,
                         std::vector<sampling::TokenLogprobs>* scored,
                         ScorePlan* plan, std::string* error_msg) {
  if (prompt.empty()) {
    AssignError(error_msg, "scoring needs a prompt");
    return false;
  }
  if (tokens.empty()) {
    AssignError(error_msg, "no tokens to score");
    return false;
  }
  const auto vocab = model_->VocabSize();
  for (const auto token : tokens) {
    if (token < 0 || static_cast<std::uint32_t>(token) >= vocab) {
      AssignError(error_msg, "scored token out of range");
      return false;
    }
  }
  if (prompt.size() + tokens.size() - 1 > ContextSize()) {
    AssignError(error_msg, "scored tokens exceed the session context");
    return false;
  }
  // As SyncImpl: keep the longest common prefix; recurrent state cannot be
  // rewound, so any divergence restarts the session.
  lookup_.BeginRequest();
  copy_rows_ = 0;
  copy_streak_ = 0;
  if (!valid_)
    Reset();
  std::size_t common = 0;
  while (common < tokens_.size() && common < prompt.size() &&
         tokens_[common] == prompt[common])
    ++common;
  if (common != tokens_.size()) {
    Reset();
    common = 0;
  }
  scored->clear();
  scored->reserve(tokens.size());
  // Global row `first` is the first whose logits score a token: the
  // prompt's last row, or after it when the session already holds the whole
  // prompt and its frontier logits.
  plan->first = prompt.size() - 1;
  if (common == prompt.size()) {
    // count 0: the selected token only, as `logprobs` without top_logprobs.
    scored->push_back(*sampling::CollectLogprobs(
        logits_, static_cast<sampling::TokenId>(tokens[0]), 0U));
    ++plan->first;
  }
  // The unfed prompt and every scored token but the last.
  plan->fed.assign(prompt.begin() + common, prompt.end());
  plan->fed.insert(plan->fed.end(), tokens.begin(), tokens.end() - 1);
  return true;
}

bool Session::ScorePromptImpl(std::span<const std::int32_t> prompt,
                              std::span<const std::int32_t> tokens,
                              std::vector<sampling::TokenLogprobs>* scored,
                              std::string* error_msg) {
  ScorePlan plan;
  if (!BeginScore(prompt, tokens, scored, &plan, error_msg))
    return false;
  const auto first = plan.first;
  const auto& fed = plan.fed;
  // The fed rows take prompt (kPrefill) passes whose rows from `first` on
  // score their successor (Executor::ForwardScoreRows), so a short prompt
  // and its answer share one pass. A prefill pass streams the routed experts
  // once for all its rows: about a fixed 0.2 s plus 0.8 ms per row, where
  // 8-row verification costs about 4.5 ms per row at any width
  // (GFX1151-KNOWHOW "Scoring speed"). Rows carry prompt arithmetic: the
  // first token is scored on a prefill row as generation samples it, and the
  // rest agree with generated (verification) logprobs to numerical error.
  // The device reduces each row to its target's log-softmax (ScoreLogprobs,
  // within about 1e-5 of sampling::CollectLogprobs' FP64 value), so only
  // those values and the frontier row come to the host.
  // Global row g scores sequence token g + 1: prompt, then `tokens`.
  const auto target = [&](std::size_t g) {
    return g + 1 < prompt.size() ? prompt[g + 1]
                                 : tokens[g + 1 - prompt.size()];
  };
  auto& exec = *model_->executor_;
  const std::size_t ratio =
      std::max<std::uint32_t>(1, model_->config().compress_ratio);
  valid_ = false;
  for (std::size_t off = 0; off < fed.size();) {
    const std::size_t start = tokens_.size();
    const std::size_t logit_from = std::max(start, first);
    const auto remaining = fed.size() - off;
    // Feed's chunking, bounded so the logit rows fit the scoring head.
    std::size_t n = std::min<std::size_t>(
        {remaining,
         remaining <= exec.max_batch() ? exec.max_batch()
                                       : model_->PrefillCapacity(),
         logit_from - start + exec.max_score_rows()});
    // As in Feed: a following pass starts on a selection block.
    if (const auto end = start + n;
        off + n < fed.size() && end % ratio != 0 && n > end % ratio)
      n -= end % ratio;
    const auto chunk = std::span<const std::int32_t>(fed).subspan(off, n);
    const auto logit_rows =
        start + n > logit_from ? start + n - logit_from : std::size_t{0};
    // Prefill needs the draft block caught up to the frontier (Feed).
    if (MtpEnabled() && !tokens_.empty() &&
        !DraftCatchUp(chunk[0], false, error_msg))
      return false;
    std::vector<std::int32_t> targets(logit_rows);
    for (std::size_t r = 0; r < logit_rows; ++r)
      targets[r] = target(logit_from + r);
    std::span<const rocm::ScoredLogprob> rows;
    // The last pass also keeps its frontier row, as Feed does.
    if (!exec.ForwardScoreRows(
            *session_, chunk, targets, &rows,
            off + n == fed.size() && logit_rows != 0 ? logits_.data() : nullptr,
            error_msg))
      return false;
    AppendScored(rows, targets, scored);
    hidden_base_ = static_cast<std::uint32_t>(
        start + n - std::min<std::size_t>(n, exec.max_speculative()));
    tokens_.insert(tokens_.end(), chunk.begin(), chunk.end());
    off += n;
  }
  if (scored->size() != tokens.size()) {
    AssignError(error_msg, "scoring returned an incomplete row set");
    return false;
  }
  valid_ = true;
  return true;
}

std::size_t Session::ScorePackRows(const Model& model) noexcept {
  return model.executor_->max_batch();
}

void Session::ScoreBatch(std::span<const ScoreRequest> requests) {
  if (requests.empty())
    return;
  const auto fail = [](const ScoreRequest& r, const std::string& error) {
    r.scored->clear();
    if (r.error)
      *r.error = error;
    *r.ok = false;
  };
  struct Packed {
    const ScoreRequest* request;
    ScorePlan plan;
  };
  std::vector<Packed> packable;
  // The model packs share: the first request's that names a session.
  const Model* model = nullptr;
  for (const auto& r : requests)
    if (r.session != nullptr) {
      model = r.session->model_.get();
      break;
    }
  for (const auto& r : requests) {
    *r.ok = false;
    if (r.session == nullptr || r.scored == nullptr) {
      if (r.scored)
        fail(r, "invalid scoring request");
      continue;
    }
    auto& session = *r.session;
    const bool distinct =
        std::ranges::count_if(requests, [&](const ScoreRequest& other) {
          return other.session == r.session;
        }) == 1;
    // KVarN and image sessions, a model whose sessions differ, and a
    // sequence longer than one pass score on their own.
    auto& exec = *session.model_->executor_;
    const bool alone = !distinct || session.model_->Kvarn() ||
                       session.session_->VisionLayout().PrefixLength() != 0 ||
                       session.model_.get() != model;
    if (alone) {
      *r.ok = distinct &&
              session.ScorePrompt(r.prompt, r.tokens, r.scored, r.error);
      if (!distinct)
        fail(r, "a scoring batch names a session twice");
      continue;
    }
    Packed item{&r, {}};
    try {
      const ExecutionScope execution(session.continuation_hooks_);
      if (!session.BeginScore(r.prompt, r.tokens, r.scored, &item.plan,
                              r.error)) {
        *r.ok = false;
        continue;
      }
    } catch (const std::exception& error) {
      session.valid_ = false;
      fail(r, error.what());
      continue;
    }
    if (item.plan.fed.empty()) {
      // The frontier already scored the only token.
      session.valid_ = true;
      *r.ok = true;
      continue;
    }
    if (item.plan.fed.size() > exec.max_batch()) {
      *r.ok = session.ScorePrompt(r.prompt, r.tokens, r.scored, r.error);
      continue;
    }
    packable.push_back(std::move(item));
  }
  if (packable.empty())
    return;
  auto& exec = *packable.front().request->session->model_->executor_;
  // Greedy packs of at most max_batch() rows and kScorePackSessions sessions.
  for (std::size_t begin = 0; begin < packable.size();) {
    std::size_t end = begin;
    std::size_t rows = 0;
    while (end < packable.size() &&
           end - begin < rocm::Executor::kScorePackSessions &&
           rows + packable[end].plan.fed.size() <= exec.max_batch())
      rows += packable[end++].plan.fed.size();
    const auto pack = std::span(packable).subspan(begin, end - begin);
    begin = end;
    std::vector<rocm::Executor::ScorePackItem> items;
    std::deque<ExecutionScope> scopes;
    std::string pack_error;
    bool ok = true;
    try {
      for (auto& p : pack) {
        auto& session = *p.request->session;
        scopes.emplace_back(session.continuation_hooks_);
        session.valid_ = false;
        // Its rows score the tokens BeginScore left, in order.
        items.push_back(
            {session.session_.get(), p.plan.fed,
             std::span(p.request->tokens).subspan(p.request->scored->size())});
      }
      std::span<const rocm::ScoredLogprob> rows;
      ok = exec.ForwardScorePack(items, &rows, &pack_error);
      for (std::size_t i = 0, q = 0; ok && i < pack.size(); ++i) {
        const auto count = items[i].targets.size();
        AppendScored(rows.subspan(q, count), items[i].targets,
                     pack[i].request->scored);
        q += count;
      }
    } catch (const std::exception& error) {
      ok = false;
      pack_error = error.what();
    }
    scopes.clear();
    std::vector<std::size_t> retry;
    for (std::size_t i = 0; i < pack.size(); ++i) {
      const auto& p = pack[i];
      const auto& r = *p.request;
      auto& session = *r.session;
      // The draft block did not run, and a failed pack may have advanced any
      // session: either way the session resets before its next use.
      session.valid_ = false;
      if (ok && r.scored->size() == r.tokens.size()) {
        session.tokens_.insert(session.tokens_.end(), p.plan.fed.begin(),
                               p.plan.fed.end());
        *r.ok = true;
        continue;
      }
      retry.push_back(i);
    }
    // One request's failure (cancellation, KV rows, a non-finite row) fails
    // the pack: score each of its requests on its own so that only the
    // offending one fails. Reset every one first, so the rows the pack wrote
    // return to a shared KV pool before any retry claims rows, and each
    // retry starts from a valid, empty session.
    for (const auto i : retry) {
      const auto& r = *pack[i].request;
      try {
        r.session->Reset();
      } catch (const std::exception& error) {
        fail(r, error.what());
      }
    }
    for (const auto i : retry) {
      const auto& r = *pack[i].request;
      if (!r.session->valid_)
        continue;  // its reset failed above
      *r.ok = r.session->ScorePrompt(r.prompt, r.tokens, r.scored, r.error);
      if (!*r.ok && r.error && r.error->empty())
        *r.error = pack_error.empty() ? "scoring failed" : pack_error;
    }
  }
}

struct Session::PendingDecode {
  std::vector<std::int32_t> chain;
  std::vector<MtpProposal> proposals;
  std::uint32_t base{0};
  bool speculative{false};
  bool sampled{false};
  /// A greedy chain under an active Han scale: its heads also return
  /// candidates and each draft is their adjusted argmax (AppendDraft).
  bool scaled_greedy{false};
  bool gpu_greedy{false};
  bool gpu_verification{false};
  std::size_t width{0};
  std::optional<sampling::SamplerState> draft_sampler;
  std::uint64_t draft_rng{0};
  MtpCandidateLogits candidates;
  std::int32_t draft{0};
  bool draft_subset{false};  ///< this cycle drafts through the subset head
  bool sampled_subset_cost{false};
  /// Trailing chain tokens copied by prompt lookup (after the MTP drafts).
  std::uint32_t lookup{0};
  /// Committed index the copied tokens were taken from.
  std::size_t lookup_start{0};
  /// Identical copy budget in isolated and batched execution; zero on the
  /// request's first speculative step.
  std::size_t lookup_cap{0};
  std::uint32_t policy_concurrency{0};
  /// A copy-only cycle here may verify more than max_speculative() rows.
  bool wide{false};
};

void Session::SelectScaledDraft(PendingDecode& pending) {
  // The device head takes the raw argmax. Rank the head's top candidates as
  // the target does, so its Han offset (and penalties) cannot reject every
  // Han draft. Idempotent until the next head overwrites the candidates.
  const auto size = pending.candidates.size;
  const auto best =
      pending.draft_sampler
          ->Distribution(std::span(pending.candidates.logits).first(size),
                         std::span(pending.candidates.ids).first(size))
          .best_token();
  pending.draft = static_cast<std::int32_t>(pending.candidates.ids[best]);
}

void Session::AppendDraft(PendingDecode& pending) {
  if (pending.scaled_greedy) {
    SelectScaledDraft(pending);
    pending.draft_sampler->Accept(
        static_cast<sampling::TokenId>(pending.draft));
  }
  if (pending.sampled) {
    pending.proposals.push_back(SampleMtpProposal(
        pending.candidates, *pending.draft_sampler, &pending.draft_rng));
    pending.draft = static_cast<std::int32_t>(pending.proposals.back().token);
    pending.draft_sampler->Accept(pending.proposals.back().token);
  }
  pending.chain.push_back(pending.draft);
}

bool Session::AppendLookup(PendingDecode& pending, bool stop_at_eos,
                           std::size_t minimum, bool require_resume) {
  const auto cap = pending.lookup_cap;
  if (pending.chain.size() >= cap) {
    return false;
  }
  if (stop_at_eos && model_->IsStopToken(pending.chain.back())) {
    return false;
  }
  const auto match = lookup_.Find(
      tokens_, pending.chain,
      pending.sampled ? speculative::PromptLookup<>::kSampledMinMatch
                      : speculative::PromptLookup<>::kMinMatch,
      require_resume);
  if (match.length == 0) {
    return false;
  }
  const std::size_t available = std::min(cap - pending.chain.size(),
                                         lookup_.IndexedTokens() - match.start);
  if (available < minimum)
    return false;
  std::size_t count = 0;
  for (; count < available; ++count) {
    const std::int32_t token = tokens_[match.start + count];
    pending.chain.push_back(token);
    if (pending.sampled) {
      // A point-mass proposal: accepted with the target's p(token), and a
      // rejection resamples from the target with the token removed.
      MtpProposal proposal;
      proposal.ids[0] = static_cast<sampling::TokenId>(token);
      proposal.probabilities[0] = 1.0F;
      proposal.size = 1;
      proposal.token = proposal.ids[0];
      proposal.probability = 1.0F;
      pending.proposals.push_back(proposal);
    }
    if (stop_at_eos && model_->IsStopToken(token)) {
      ++count;
      break;
    }
  }
  pending.lookup = static_cast<std::uint32_t>(count);
  pending.lookup_start = match.start;
  pending.width = pending.chain.size();
  return count != 0;
}

bool Session::PrepareDecode(const DecodeRequest& request,
                            PendingDecode* pending, std::string* error_msg,
                            bool defer_head,
                            std::optional<std::uint32_t> batch_drafts,
                            std::uint32_t occupancy, std::uint32_t residents) {
  const auto max_tokens = request.max_tokens;
  auto& sampler = *request.sampler;
  auto* result = request.result;
  const bool stop_at_eos = request.stop_at_eos;
  if (result == nullptr || max_tokens == 0 || tokens_.empty()) {
    AssignError(
        error_msg,
        "decode needs an output, a positive budget and a synced prompt");
    return false;
  }
  *result = {};
  const auto is_stop = [&](std::int32_t token) {
    return stop_at_eos && model_->IsStopToken(token);
  };
  rocm::Executor& exec = *model_->executor_;
  const std::size_t room = ContextSize() - tokens_.size();
  const std::size_t cap =
      std::min<std::size_t>({max_tokens, room, exec.max_speculative()});
  // Head eligibility is MtpDraftSubsetAllowed. Sampled cohorts are never
  // jointly planned (DecodeBatchImpl), so a sampled request decides the same
  // way alone and in any batch. A greedy request decoding alone has no
  // peers to delay, so it drafts as on one session. Every request keeps the
  // established length controller and AR backoff.
  const bool allow_subset = MtpDraftSubsetAllowed(
      sampler.config().uses_random_sampling(), model_->DecodeConcurrency(),
      occupancy, batch_drafts.has_value());
  pending->policy_concurrency = MtpPolicyConcurrency(
      sampler.config().uses_random_sampling(), sampler.config().seed >= 0,
      batch_drafts.has_value(), occupancy);
  // A lone greedy request on a single-session model stops its chain by draft
  // confidence (MtpConfidenceRule) once the request has observed acceptance.
  // Until then the length controller's conservative first width applies, as
  // for every other request: widening a short reply's first cycle slowed it.
  // The controller observes every cycle and decides its AR backoff; after the
  // first observation its chosen width is not used.
  const bool confidence_rule = !sampler.config().uses_random_sampling() &&
                               !batch_drafts && !defer_head && occupancy == 1 &&
                               model_->DecodeConcurrency() == 1 &&
                               draft_length_.State().explored_depth != 0;
  const auto choose = [&] {
    const auto chosen =
        draft_length_.Choose(static_cast<std::uint32_t>(cap - 1),
                             static_cast<std::uint32_t>(tokens_.size()),
                             pending->policy_concurrency);
    return confidence_rule && chosen != 0
               ? std::min<std::uint32_t>(static_cast<std::uint32_t>(cap - 1),
                                         draft_length_.Limit())
               : chosen;
  };
  const std::size_t width =
      MtpEnabled() && cap > 1
          ? 1 + (batch_drafts ? std::min<std::uint32_t>(*batch_drafts, cap - 1)
                              : choose())
          : cap;
  if (width == 0) {
    result->stop = true;
    return true;
  }
  const auto anchor = static_cast<std::int32_t>(sampler.Sample(logits_));
  auto anchor_logprobs =
      sampling::CollectLogprobs(logits_, anchor, sampler.config().logprobs);
  if (is_stop(anchor)) {
    result->terminal_logprobs = std::move(anchor_logprobs);
    result->stop = true;
    return true;
  }
  if (anchor_logprobs)
    result->logprobs.push_back(std::move(*anchor_logprobs));
  if (!MtpEnabled() || width < 2) {
    if (MtpEnabled() && !defer_head &&
        !DraftCatchUp(anchor, false, error_msg)) {
      return false;
    }
    pending->chain = {anchor};
    pending->base = static_cast<std::uint32_t>(tokens_.size());
    return true;
  }

  const std::uint32_t base = static_cast<std::uint32_t>(tokens_.size());
  const bool sampled = sampler.config().uses_random_sampling();
  const bool gpu_greedy = sampler.config().temperature == 0.0F;
  const bool gpu_verification =
      gpu_greedy || rocm::Executor::CanCompactMtpVerification(sampler);
  // A cycle-local proposal stream needs no pending RNG state in snapshots.
  // Target verification keeps its own draws after this independent seed.
  pending->draft_rng =
      sampled ? sampling::NextRandom(sampler.mutable_rng_state()) : 0;
  pending->draft_sampler = sampler.WithoutConstraint();
  pending->draft_sampler->Accept(static_cast<sampling::TokenId>(anchor));
  pending->chain = {anchor};
  pending->draft = draft_token_;
  pending->width = width;
  pending->base = base;
  pending->speculative = true;
  pending->sampled = sampled;
  pending->scaled_greedy = !sampled && sampler.config().han_bias() != 0;
  pending->gpu_greedy = gpu_greedy;
  pending->gpu_verification = gpu_verification;
  if (!gpu_verification &&
      verify_logits_.size() < exec.max_speculative() * model_->VocabSize()) {
    verify_logits_.resize(exec.max_speculative() * model_->VocabSize());
  }
  // Both execution paths use the same request-local copy rule. Scheduling
  // a peer must not change a sampled request's proposals or RNG sequence.
  // A request's first speculative step copies nothing (see
  // speculative::PromptLookup<>).
  pending->lookup_cap = lookup_.Step(base) ? cap : 0;
  if (pending->lookup_cap != 0) {
    lookup_.Extend(tokens_, sampled);
  }
  // A session decoding alone may verify a copy wider than the MTP chain
  // (Executor::ForwardWide): the shared projections read their weights
  // once for every row. The width grows only while whole copies are
  // accepted, so a copy that diverges costs at most the last width.
  // Sampled requests qualify when nothing replays them (no seed) and their
  // rows verify compactly on the device. Batched cycles (defer_head) and
  // seeded requests keep the ordinary width, so their proposals and draws
  // do not depend on scheduling. A serial step priced for a cohort
  // (occupancy > 1) keeps it too, matching its batched execution. So does
  // a step taken while the server holds another request (residents > 1):
  // a peer that is queued or prefilling would wait for the whole wide pass.
  pending->wide = !defer_head && occupancy <= 1 && residents <= 1 &&
                  pending->lookup_cap != 0 &&
                  (gpu_greedy || (sampled && sampler.config().seed < 0 &&
                                  rocm::Executor::CanCompactWideVerification(
                                      sampler, exec.max_copy_rows())));
  const std::size_t wide_rows =
      pending->wide
          ? std::min<std::size_t>(
                {max_tokens, room, exec.max_copy_rows(),
                 std::max<std::size_t>(copy_rows_, exec.max_speculative())})
          : 0;
  if (wide_rows > pending->lookup_cap)
    pending->lookup_cap = wide_rows;
  // A sampled copy can skip the neural head only after an accepted copy
  // established its source and a full sampled match still holds. New or
  // short matches retain the neural proposal: replacing it based only on
  // shared syntax previously regressed an identical Pi code write by 9%.
  if (AppendLookup(*pending, stop_at_eos, 3, sampled)) {
    return defer_head || DraftCatchUp(anchor, false, error_msg);
  }
  // Only a copy-only cycle verifies wide: MTP drafts keep the ordinary width.
  pending->lookup_cap = std::min(pending->lookup_cap, cap);
  if (allow_subset && !SyncDraftVocab(error_msg))
    return false;
  pending->draft_subset = allow_subset && model_->draft_subset_ &&
                          draft_vocab_.Fits(anchor, [&](std::int32_t token) {
                            return model_->InDraftSubset(token);
                          });
  // Wider verification slowed short resumed replies in matched controls.
  // Keep the established costs within two blocks of the remaining budget.
  // Full target rows (for example, active tool constraints) also keep their
  // original costs; the smaller-head curve does not price their transfers.
  pending->sampled_subset_cost =
      sampled && gpu_verification && pending->draft_subset &&
      std::min(max_tokens, room) > 2 * exec.max_speculative();
  if (pending->sampled_subset_cost && !batch_drafts) {
    // Price the head actually used, after vocabulary fallback and the copy
    // bypass. The policy depends only on committed history, not timings.
    pending->width =
        1 + draft_length_.Choose(static_cast<std::uint32_t>(cap - 1), base,
                                 pending->policy_concurrency, true);
  }
  const bool candidates = sampled || confidence_rule || pending->scaled_greedy;
  if (!defer_head && !DraftCatchUp(anchor, true, error_msg,
                                   candidates ? &pending->candidates : nullptr,
                                   pending->draft_subset,
                                   confidence_rule || pending->scaled_greedy)) {
    return false;
  }
  pending->draft = draft_token_;
  if (!defer_head) {
    MtpConfidenceRule rule;
    while (pending->chain.size() < pending->width) {
      bool more = true;
      if (confidence_rule) {
        // Judge the token AppendDraft will append.
        if (pending->scaled_greedy)
          SelectScaledDraft(*pending);
        const auto step =
            rule.Step(static_cast<std::uint32_t>(pending->chain.size() - 1),
                      DraftConfidence(pending->candidates, pending->draft));
        if (!step.keep) {
          pending->width = pending->chain.size();
          break;
        }
        more = step.more;
      }
      AppendDraft(*pending);
      if (stop_at_eos && model_->IsStopToken(pending->chain.back())) {
        pending->width = pending->chain.size();
        break;
      }
      if (AppendLookup(*pending, stop_at_eos)) {
        break;
      }
      if (!more) {
        pending->width = pending->chain.size();
        break;
      }
      if (pending->chain.size() < pending->width &&
          !exec.MtpForward(
              *session_, std::span<const std::int32_t>(&pending->draft, 1), -1,
              {.token = sampled ? nullptr : &pending->draft,
               .candidates = candidates ? &pending->candidates : nullptr,
               .draft_subset = pending->draft_subset},
              error_msg)) {
        return false;
      }
    }
  }
  return true;
}

bool Session::FinishDecode(const DecodeRequest& request,
                           const PendingDecode& pending,
                           std::string* error_msg) {
  auto& sampler = *request.sampler;
  auto* result = request.result;
  auto& exec = *model_->executor_;
  const auto& chain = pending.chain;
  const auto& proposals = pending.proposals;
  const auto base = pending.base;
  const bool sampled = pending.sampled;
  const bool gpu_greedy = pending.gpu_greedy;
  const bool gpu_verification = pending.gpu_verification;
  const auto anchor = chain.front();
  const auto k = static_cast<std::uint32_t>(chain.size());
  const auto vocab = model_->VocabSize();
  const auto is_stop = [&](std::int32_t token) {
    return request.stop_at_eos && model_->IsStopToken(token);
  };
  if (!pending.speculative) {
    draft_length_.ObserveArToken();
    hidden_base_ = base;
    tokens_.push_back(anchor);
    sampler.Accept(static_cast<sampling::TokenId>(anchor));
    result->tokens.push_back(anchor);
    return true;
  }
  sampler.Accept(static_cast<sampling::TokenId>(anchor));
  std::array<rocm::ArgmaxCandidate, rocm::Executor::kWideRows - 1> greedy{};
  if (gpu_greedy &&
      !exec.GreedyMtpPredictions(std::span(greedy).first(k - 1), sampler,
                                 std::span(chain).subspan(1), error_msg)) {
    return false;
  }
  std::uint32_t keep = 1;
  std::optional<std::int32_t> correction;
  std::optional<std::int32_t> stop_token;
  bool cpu_rows = false;
  const auto download_remaining = [&](std::uint32_t first) {
    verify_logits_.resize(std::max<std::size_t>(exec.max_speculative(), k) *
                          vocab);
    auto rows =
        std::span(verify_logits_).subspan(first * vocab, (k - first) * vocab);
    if (!exec.ReadVerificationRows(first, rows, error_msg))
      return false;
    cpu_rows = true;
    return true;
  };
  std::array<std::span<const sampling::LogitCandidate>,
             rocm::Executor::kWideRows - 1>
      candidates{};
  if (gpu_verification && !gpu_greedy) {
    // Accepting the anchor may activate a tool grammar, reasoning budget,
    // or exceed the bounded penalty workspace. Keep the complete-row path.
    if (!rocm::Executor::CanCompactMtpVerification(sampler) ||
        (k > exec.max_speculative() &&
         !rocm::Executor::CanCompactWideVerification(sampler, k))) {
      if (!download_remaining(0))
        return false;
    } else if (!exec.SampledMtpCandidates(std::span(candidates).first(k - 1),
                                          sampler, std::span(chain).subspan(1),
                                          error_msg)) {
      return false;
    }
  }
  while (keep < k) {
    if (gpu_greedy && !cpu_rows) {
      const auto& prediction = greedy[keep - 1];
      if (!std::isfinite(prediction.value)) {
        AssignError(error_msg, "logit distribution contains no finite values");
        return false;
      }
      if (!sampler.CanSelectArgmax(prediction.index,
                                   /*penalties_applied=*/true)) {
        // Most native tool tokens already obey the grammar. On the first
        // forbidden argmax, download the remaining rows once and use exact
        // masked selection. Avoid one synchronization per rejected candidate.
        // SelectBatchLogits has already installed this session's row offset.
        if (!download_remaining(keep - 1))
          return false;
        continue;
      }
      if (is_stop(prediction.index)) {
        stop_token = prediction.index;
        result->stop = true;
        break;
      }
      if (prediction.index != chain[keep]) {
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(prediction.index));
      ++keep;
      continue;
    }
    if (sampled) {
      if (gpu_verification && !cpu_rows && sampler.NeedsConstraintMask() &&
          !download_remaining(keep - 1))
        return false;
      const auto verified =
          gpu_verification && !cpu_rows
              ? VerifyMtpProposal(candidates[keep - 1], vocab,
                                  proposals[keep - 1], sampler)
              : VerifyMtpProposal(std::span<const float>(verify_logits_)
                                      .subspan((keep - 1) * vocab, vocab),
                                  proposals[keep - 1], sampler);
      const auto token = static_cast<std::int32_t>(verified.token);
      const bool accepted = verified.accepted;
      if (is_stop(token)) {
        stop_token = token;
        result->stop = true;
        break;
      }
      if (!accepted) {
        correction = token;
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(token));
      ++keep;
      continue;
    }
    const auto decision = VerifyDraft(std::span<const float>(verify_logits_)
                                          .subspan((keep - 1) * vocab, vocab),
                                      chain[keep], sampler, is_stop);
    if (decision != DraftDecision::kAccept) {
      result->stop = decision == DraftDecision::kStop;
      if (result->stop && sampler.config().logprobs) {
        auto copy = sampler;
        stop_token = static_cast<std::int32_t>(
            copy.Sample(std::span<const float>(verify_logits_)
                            .subspan((keep - 1) * vocab, vocab)));
      }
      break;
    }
    ++keep;
  }
  if (sampler.config().logprobs) {
    const auto rows = keep - 1 + (stop_token ? 1 : 0);
    std::vector<float> raw(rows * vocab);
    if (rows && !exec.ReadVerificationRows(0, raw, error_msg))
      return false;
    auto values = CollectRowLogprobs(
        raw, vocab, rows,
        [&](std::uint32_t i) {
          return i + 1 < keep ? chain[i + 1] : *stop_token;
        },
        sampler.config().logprobs);
    for (std::uint32_t i = 0; i + 1 < keep; ++i)
      result->logprobs.push_back(std::move(*values[i]));
    if (stop_token)
      result->terminal_logprobs = std::move(values[keep - 1]);
  }
  if (!exec.Rollback(
          *session_, keep, error_msg,
          gpu_verification && !cpu_rows ? logits_.data() : nullptr)) {
    return false;
  }
  if (!gpu_verification || cpu_rows) {
    std::copy_n(verify_logits_.data() + (keep - 1) * vocab, vocab,
                logits_.begin());
  }
  hidden_base_ = base;
  tokens_.insert(tokens_.end(), chain.begin(), chain.begin() + keep);
  result->tokens.assign(chain.begin(), chain.begin() + keep);
  stats_.cycles += 1;
  stats_.drafted += k - 1;
  stats_.accepted += keep - 1;
  stats_.small_head_cycles += pending.draft_subset;
  // Copied proposals trail the MTP drafts. After a fully accepted MTP part,
  // the length controller also counts the accepted copied tokens as accepted
  // drafts: a copy that ended drafting early would otherwise make a perfect
  // chain look short, and the controller would probe shallower afterwards. A
  // rejected copy says nothing about the MTP head, so it is not a failure.
  const std::uint32_t mtp_drafts = k - 1 - pending.lookup;
  const std::uint32_t mtp_accepted = std::min(keep - 1, mtp_drafts);
  stats_.lookup += pending.lookup;
  stats_.lookup_accepted += keep - 1 - mtp_accepted;
  if (pending.lookup != 0) {
    lookup_.Observe(pending.lookup_start, pending.lookup,
                    keep - 1 - mtp_accepted);
    // Only a copy-only cycle that may verify wide sets the width
    // (PrepareDecode): two whole copies in a row at the current width
    // double it, and any rejected copy returns it to the ordinary width.
    if (mtp_drafts == 0 && pending.wide) {
      const auto width = std::max(copy_rows_, exec.max_speculative());
      if (keep < k) {
        copy_rows_ = exec.max_speculative();
        copy_streak_ = 0;
      } else if (k >= width && ++copy_streak_ >= 2) {
        copy_rows_ = std::min(
            2 * width, std::max(exec.max_copy_rows(), exec.max_speculative()));
        copy_streak_ = 0;
      }
    }
  }
  const bool mtp_all = mtp_accepted == mtp_drafts;
  const std::uint32_t observed = mtp_all ? keep - 1 : mtp_accepted;
  // A target stop ends the request; it does not classify the remaining
  // proposals as failed predictions.
  draft_length_.Observe(
      observed, result->stop || mtp_all ? observed : mtp_drafts, base,
      pending.policy_concurrency, pending.sampled_subset_cost);

  // The next call knows the next sampled anchor. Defer draft catch-up until
  // then, retaining this session's target hidden rows across interleaving.
  exec.MtpRewind(*session_, base);
  if (keep > exec.max_speculative()) {
    // Snapshots and batched cycles keep at most max_speculative() hidden
    // rows. Catch the draft block up to the last kept row now (K/V only,
    // as for prompt rows) and keep that row's hidden.
    const std::vector<std::int32_t> replay(tokens_.begin() + base + 1,
                                           tokens_.begin() + base + keep);
    if (!exec.MtpForward(*session_, replay, 0, {.kv_only = true}, error_msg) ||
        !exec.KeepHiddenRow(*session_, keep - 1, error_msg))
      return false;
    hidden_base_ = base + keep - 1;
  }
  if (correction) {
    // Evaluate the residual as the next cycle's anchor, avoiding a separate
    // target pass. Preserve the actual draw: resampling p would be biased.
    sampler.DeferSample(static_cast<sampling::TokenId>(*correction));
  }
  return true;
}

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos,
                         std::uint32_t occupancy, std::uint32_t residents) {
  const ExecutionScope execution(continuation_hooks_);
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  ApplyDynamicAbliteration();
  valid_ = false;
  const DecodeRequest request{this, max_tokens, &sampler, result, stop_at_eos};
  PendingDecode pending;
  if (!PrepareDecode(request, &pending, error_msg, false, std::nullopt,
                     occupancy, residents)) {
    return false;
  }
  if (pending.chain.empty()) {
    valid_ = true;
    return true;
  }
  if (pending.lookup != 0)
    model_->batch_policy_.SkipCycle();
  float* logits = !pending.speculative       ? logits_.data()
                  : pending.gpu_verification ? nullptr
                                             : verify_logits_.data();
  auto& exec = *model_->executor_;
  if (pending.chain.size() > exec.max_speculative()) {
    // A wide copy (PrepareDecode).
    if (!pending.gpu_verification)
      verify_logits_.resize(std::max(
          verify_logits_.size(), pending.chain.size() * model_->VocabSize()));
    if (!exec.ForwardWide(*session_, pending.chain, error_msg) ||
        !exec.SelectBatchLogits(
            0, static_cast<std::uint32_t>(pending.chain.size()),
            pending.gpu_verification ? nullptr : verify_logits_.data(),
            error_msg))
      return false;
  } else if (!exec.Forward(
                 *session_, pending.chain, pending.chain.size(), logits,
                 pending.speculative ? rocm::Executor::ForwardMode::kVerify
                                     : rocm::Executor::ForwardMode::kDecode,
                 error_msg)) {
    return false;
  }
  const bool ok = FinishDecode(request, pending, error_msg);
  valid_ = ok;
  return ok;
}

template<class Request>
bool Session::RunIsolatedBatch(std::span<const Request> requests,
                               std::string* error_msg) {
  if (requests.empty() || requests.size() > 8) {
    AssignError(error_msg, "batch must contain 1..8 sessions");
    return false;
  }
  std::array<BatchOutcome, 8> outcomes{};
  std::array<std::uint64_t, 8> epochs{};
  std::array<sampling::SamplerState::DrawState, 8> draws{};
  std::array<speculative::PromptLookup<>::State, 8> lookup_states{};
  std::vector<Request> active;
  active.reserve(requests.size());
  // Every session whose BeforeExecution succeeded may have batch work
  // submitted; mark the end of it once the batch and its retries return.
  struct AfterBatch {
    std::span<const Request> requests;
    std::array<bool, 8> executed{};
    ~AfterBatch() {
      for (std::size_t i = 0; i < requests.size(); ++i)
        if (executed[i])
          requests[i].session->continuation_hooks_->AfterExecution();
    }
  } after_batch{requests};
  Model* model = nullptr;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    auto& outcome = outcomes[i];
    if (r.session)
      epochs[i] = r.session->session_->MutationEpoch();
    bool valid = r.session && r.session->valid_;
    if constexpr (std::is_same_v<Request, DecodeRequest>) {
      valid = valid && r.sampler && r.result && r.max_tokens > 0 &&
              !r.session->tokens_.empty();
    } else {
      valid = valid && r.token >= 0 &&
              static_cast<std::uint32_t>(r.token) <
                  r.session->model_->VocabSize() &&
              r.session->Position() < r.session->ContextSize();
    }
    for (std::size_t j = 0; j < requests.size(); ++j) {
      if (i == j)
        continue;
      valid = valid && r.session != requests[j].session;
      if constexpr (std::is_same_v<Request, DecodeRequest>)
        valid = valid && r.sampler != requests[j].sampler &&
                r.result != requests[j].result;
    }
    if (valid && model && r.session->model_.get() != model)
      valid = false;
    if (!valid) {
      outcome.error = "invalid or non-independent batch request";
      continue;
    }
    try {
      if (r.session->continuation_hooks_) {
        r.session->continuation_hooks_->BeforeExecution();
        after_batch.executed[i] = true;
      }
    } catch (const std::exception& exception) {
      outcome.error = exception.what();
      continue;
    }
    if (!r.session->session_->CheckCancellation(&outcome.error))
      continue;
    model = r.session->model_.get();
    if constexpr (std::is_same_v<Request, DecodeRequest>) {
      draws[i] = r.sampler->SaveDrawState();
      lookup_states[i] = r.session->lookup_.Save();
    }
    auto copy = r;
    copy.outcome = &outcome;
    active.push_back(copy);
  }
  std::string shared_error;
  try {
    if (!active.empty()) {
      if constexpr (std::is_same_v<Request, DecodeRequest>)
        (void)DecodeBatchImpl(active, &shared_error,
                              static_cast<std::uint32_t>(requests.size()));
      else
        (void)EvaluateBatchImpl(active, &shared_error);
    }
  } catch (const std::exception& exception) {
    shared_error = exception.what();
  }
  bool success = true;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    auto& outcome = outcomes[i];
    if (!outcome.completed && outcome.error.empty()) {
      auto& session = *r.session;
      if (session.session_->Cancelled()) {
        outcome.error = "generation cancelled";
      } else if (session.session_->MutationEpoch() == epochs[i]) {
        // A preparation/batch-allocation failure did not touch this peer.
        // Retry it independently, without changing its cached frontier.
        session.valid_ = true;
        if constexpr (std::is_same_v<Request, DecodeRequest>) {
          r.sampler->RestoreDrawState(draws[i]);
          session.lookup_.Restore(lookup_states[i]);
        }
        try {
          if constexpr (std::is_same_v<Request, DecodeRequest>)
            outcome.completed = session.DecodeStep(
                r.max_tokens, *r.sampler, r.result, &outcome.error,
                r.stop_at_eos, 1, static_cast<std::uint32_t>(requests.size()));
          else
            outcome.completed = session.Evaluate(r.token, &outcome.error);
        } catch (const std::exception& exception) {
          outcome.error = exception.what();
        }
      } else {
        outcome.error =
            shared_error.empty() ? "batch execution failed" : shared_error;
      }
    }
    if (!outcome.completed) {
      if (outcome.error.empty())
        outcome.error = "batch request failed";
      if (r.session && r.session->session_->MutationEpoch() != epochs[i])
        r.session->valid_ = false;
      if (success)
        AssignError(error_msg, outcome.error);
      success = false;
    } else {
      r.session->valid_ = true;
    }
    if (r.outcome)
      *r.outcome = std::move(outcome);
  }
  return success;
}

bool Session::DecodeBatch(std::span<const DecodeRequest> requests,
                          std::string* error_msg) {
  return RunIsolatedBatch(requests, error_msg);
}

bool Session::EvaluateBatch(std::span<const AdvanceRequest> requests,
                            std::string* error_msg) {
  return RunIsolatedBatch(requests, error_msg);
}

bool Session::DecodeBatchImpl(std::span<const DecodeRequest> requests,
                              std::string* error_msg, std::uint32_t residents) {
  if (requests.size() == 1) {
    // A cohort that lost its peers keeps their count: they are still held.
    const auto& r = requests.front();
    r.outcome->completed =
        r.session->DecodeStep(r.max_tokens, *r.sampler, r.result,
                              &r.outcome->error, r.stop_at_eos, 1, residents);
    return r.outcome->completed;
  }
  auto& exec = *requests.front().session->model_->executor_;
  // Before any of this cycle's trunk work, as DecodeStep does.
  for (const auto& r : requests)
    r.session->ApplyDynamicAbliteration();
  std::optional<std::uint32_t> batch_drafts;
  std::uint32_t batch_context = 0;
  auto& policy = requests.front().session->model_->batch_policy_;
  if (std::ranges::all_of(
          requests, [](const auto& r) { return r.session->MtpEnabled(); }) &&
      std::ranges::none_of(requests, [](const auto& request) {
        return request.sampler->config().uses_random_sampling();
      })) {
    std::array<MtpBatchController::Row, 8> rows{};
    for (std::size_t i = 0; i < requests.size(); ++i) {
      const auto& r = requests[i];
      const auto cap = std::min<std::size_t>(
          r.max_tokens, r.session->ContextSize() - r.session->Position());
      rows[i] = {&r.session->draft_length_,
                 static_cast<std::uint32_t>(
                     std::min<std::size_t>(cap, exec.max_speculative())) -
                     (cap != 0)};
      batch_context = std::max(batch_context, r.session->Position());
    }
    batch_drafts =
        policy.Choose(std::span(rows).first(requests.size()), batch_context);
  }
  const auto cycle_start = std::chrono::steady_clock::now();
  std::vector<PendingDecode> pending(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    try {
      if (!r.session->PrepareDecode(
              r, &pending[i], &r.outcome->error, true, batch_drafts,
              static_cast<std::uint32_t>(requests.size())))
        pending[i] = {};
    } catch (const std::exception& exception) {
      r.outcome->error = exception.what();
      pending[i] = {};
    }
  }
  std::vector<AdvanceRequest> catchup;
  std::vector<std::uint8_t> propose;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    if (!pending[i].chain.empty()) {
      catchup.push_back({requests[i].session, pending[i].chain.front()});
      propose.push_back(pending[i].speculative &&
                        pending[i].chain.size() < pending[i].width);
    }
  }
  if (!DraftCatchUpBatch(catchup, error_msg, propose))
    return false;
  // Each round shares predictor projections across ready sessions.
  // Attention state, proposal distributions and RNG streams stay private.
  for (;;) {
    std::vector<rocm::Executor::MtpHeadItem> heads;
    std::vector<rocm::Executor::MtpBatchItem> bodies;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (!requests[i].session->session_->Cancelled() && p.speculative &&
          p.chain.size() < p.width) {
        heads.push_back(
            {requests[i].session->session_.get(),
             {.token = p.sampled ? nullptr : &p.draft,
              .candidates =
                  p.sampled || p.scaled_greedy ? &p.candidates : nullptr,
              .draft_subset = p.draft_subset}});
      }
    }
    if (heads.empty())
      break;
    if (!exec.MtpHeads(heads, error_msg))
      return false;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (requests[i].session->session_->Cancelled() || !p.speculative ||
          p.chain.size() >= p.width)
        continue;
      AppendDraft(p);
      auto& session = *requests[i].session;
      if (requests[i].stop_at_eos &&
          session.model_->IsStopToken(p.chain.back())) {
        p.width = p.chain.size();
        continue;
      }
      if (session.AppendLookup(p, requests[i].stop_at_eos))
        continue;
      if (p.chain.size() < p.width)
        bodies.push_back({requests[i].session->session_.get(),
                          std::span<const std::int32_t>(&p.draft, 1), -1});
    }
    if (!bodies.empty() && !exec.MtpForwardBatch(bodies, error_msg))
      return false;
  }
  // In an all-greedy cohort, full copies can widen verification far beyond
  // the neural planner's chosen shape and delay peers producing new text.
  // Keep full width when every active row copies. In a mixed cohort, bound
  // copies to half the verifier, or the already planned width if larger.
  // Only copied suffixes are trimmed; sampled proposal/RNG streams are fixed.
  const bool mixed_lookup =
      batch_drafts && std::ranges::any_of(pending, [](const auto& p) {
        return !p.chain.empty() && p.lookup == 0;
      });
  if (mixed_lookup) {
    for (auto& p : pending) {
      if (p.lookup == 0)
        continue;
      const auto cap =
          std::max<std::size_t>(1 + *batch_drafts, (p.lookup_cap + 1) / 2);
      if (p.chain.size() > cap) {
        p.lookup -= static_cast<std::uint32_t>(p.chain.size() - cap);
        p.chain.resize(cap);
        p.width = cap;
      }
    }
  }
  std::vector<rocm::Executor::BatchItem> items;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    if (!r.session->session_->Cancelled() && !pending[i].chain.empty()) {
      items.push_back({r.session->session_.get(), pending[i].chain,
                       pending[i].speculative});
    }
  }
  if (!items.empty() && !exec.ForwardBatch(items, error_msg))
    return false;
  std::uint32_t offset = 0;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& p = pending[i];
    const auto& r = requests[i];
    auto& session = *r.session;
    const bool included =
        std::any_of(items.begin(), items.end(), [&](const auto& item) {
          return item.session == session.session_.get();
        });
    const auto row_offset = offset;
    if (included)
      offset += p.chain.size();
    if (!r.outcome->error.empty())
      continue;
    if (session.session_->Cancelled()) {
      r.outcome->error = "generation cancelled";
      continue;
    }
    if (p.chain.empty()) {
      r.outcome->completed = true;
      continue;
    }
    float* logits = !p.speculative       ? session.logits_.data()
                    : p.gpu_verification ? nullptr
                                         : session.verify_logits_.data();
    try {
      r.outcome->completed =
          exec.SelectBatchLogits(row_offset, p.chain.size(), logits,
                                 &r.outcome->error) &&
          session.FinishDecode(r, p, &r.outcome->error);
    } catch (const std::exception& exception) {
      r.outcome->error = exception.what();
    }
  }
  const bool used_lookup =
      std::ranges::any_of(pending, [](const auto& p) { return p.lookup != 0; });
  if (used_lookup)
    policy.SkipCycle();
  if (!used_lookup && batch_drafts && items.size() == requests.size() &&
      std::ranges::all_of(requests,
                          [](const auto& r) { return r.outcome->completed; })) {
    const auto ms = std::chrono::duration<float, std::milli>(
                        std::chrono::steady_clock::now() - cycle_start)
                        .count();
    policy.Observe(requests.size(), batch_context, *batch_drafts, ms);
  }
  return true;
}

bool Session::EvaluateBatchImpl(std::span<const AdvanceRequest> requests,
                                std::string* error_msg) {
  if (requests.size() == 1) {
    const auto& r = requests.front();
    r.outcome->completed = r.session->Evaluate(r.token, &r.outcome->error);
    return r.outcome->completed;
  }
  auto& exec = *requests.front().session->model_->executor_;
  for (const auto& r : requests)
    r.session->ApplyDynamicAbliteration();
  if (!DraftCatchUpBatch(requests, error_msg))
    return false;
  std::vector<rocm::Executor::BatchItem> items;
  for (const auto& r : requests) {
    items.push_back({r.session->session_.get(), {&r.token, 1}, false});
  }
  if (!exec.ForwardBatch(items, error_msg)) {
    return false;
  }
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& session = *requests[i].session;
    auto& outcome = *requests[i].outcome;
    if (session.session_->Cancelled()) {
      outcome.error = "generation cancelled";
      continue;
    }
    try {
      if (!exec.SelectBatchLogits(i, 1, session.logits_.data(), &outcome.error))
        continue;
      session.hidden_base_ = static_cast<std::uint32_t>(session.tokens_.size());
      session.tokens_.push_back(requests[i].token);
      outcome.completed = true;
    } catch (const std::exception& exception) {
      outcome.error = exception.what();
    }
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next
