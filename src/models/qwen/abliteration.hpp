#ifndef GUFO_MODELS_QWEN_ABLITERATION_HPP_
#define GUFO_MODELS_QWEN_ABLITERATION_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::qwen {

/// Every runtime projection profile carries a schema. The placement names
/// where a model applies it, so a profile prepared for one architecture never
/// loads into another even before the model identity check.
/// v1: one `direction`; its source layer is unknown, so requests cannot pick
/// another one.
inline constexpr std::string_view kAbliterationSchemaV1 =
    "gufo-abliteration-v1";
/// v2: `directions` holds one unit direction per layer ([layers][hidden]);
/// `direction_layer`, `attention`, `mlp` and the optional `residual` (absent:
/// all zero) are the request defaults. Optional D18 members add a projected
/// direction kind, a rank-k subspace per layer and kind, and their defaults
/// (`direction_kind`, `direction_count`, `preserve_norm`); a v2 profile
/// without them is rank 1, mean-difference, no norm preservation.
inline constexpr std::string_view kAbliterationSchemaV2 =
    "gufo-abliteration-v2";
/// Qwen3.8-Flash-Next: attention/GDN and aggregated MoE outputs, immediately
/// before hyper-connection injection; `residual` strengths act on each of the
/// HC residual streams after the layer. The name predates residual strengths.
inline constexpr std::string_view kAbliterationWriterBeforeHc =
    "writer-before-hc";
/// Qwen3.8-27B (plain residual): attention/GDN output projection and MLP down
/// projection outputs, immediately before their residual add; `residual`
/// strengths act on the residual after the layer's MLP add. The name predates
/// residual strengths.
inline constexpr std::string_view kAbliterationWriterBeforeResidual =
    "writer-before-residual";
/// Request and v2 profile strengths are in [0, kAbliterationMaxWeight].
inline constexpr float kAbliterationMaxWeight = 4.0F;
/// A v2 27B profile (64 x 5120 directions) is about 4.5 MB of JSON when
/// written with float32 round-trip precision (prepare.py); the D18 members
/// are base64 float32, so rank 8 with both kinds adds about 26 MB.
inline constexpr std::size_t kAbliterationMaxProfileBytes = 48 << 20;
/// The largest subspace rank (orthonormal directions per layer and kind).
/// The projection kernel keeps one FP32 dot accumulator per direction in
/// registers and reduces all of them in one 8 x 256-float shared block, and a
/// 27B rank-8 profile with both kinds already holds 64 x 2 x 8 x 5120 floats
/// (21 MB on the device). Refusal subspaces reported in the literature are
/// low-dimensional; beyond a few directions ablation approaches removing a
/// generic subspace and the KL to OFF grows.
inline constexpr std::uint32_t kAbliterationMaxDirections = 8;

/// The most profiles one server loads (`--abliteration-profiles`).
inline constexpr std::size_t kAbliterationMaxProfiles = 8;
/// A dynamic profile's detector schema (fit_trigger.py).
inline constexpr std::string_view kAbliterationTriggerSchema =
    "gufo-abliteration-trigger-v1";
/// The collector readout a placement's detector must be fitted on: the
/// post-layer residual of the last prompt token, averaged over the HC
/// streams on Flash-Next.
[[nodiscard]] std::string_view AbliterationReadout(
    std::string_view placement) noexcept;

/// A profile's detector (R2 dynamic abliteration): the server scores
/// <h, direction> on the post-layer residual of `layer`, read exactly where
/// the collector reads it, and turns the profile ON for the rest of a request
/// whose score exceeds `threshold`.
struct AbliterationTrigger {
  std::uint32_t layer{0};
  /// Unit vector of the profile's hidden size.
  std::vector<float> direction;
  float threshold{0};
};

/// Which direction family a request projects (D18).
enum class AbliterationDirectionKind : std::uint32_t {
  /// refusal mean minus ordinary mean (the D14/D16 direction)
  kMeanDiff = 0,
  /// the same with its component along the normalized ordinary mean removed
  /// (projected abliteration)
  kProjected = 1,
};
/// "mean_diff" or "projected".
[[nodiscard]] std::string_view AbliterationDirectionKindName(
    AbliterationDirectionKind kind) noexcept;
/// The kind named `name`; empty for an unknown name.
[[nodiscard]] std::optional<AbliterationDirectionKind>
ParseAbliterationDirectionKind(std::string_view name) noexcept;

/// Device word 0 of a session's settings: the first direction row (bits
/// 0-23), the number of directions minus one (bits 24-26) and the
/// norm-preservation flag (bit 27). One direction without norm preservation
/// is the bare row index, exactly the D16 word.
inline constexpr std::uint32_t kAbliterationRowMask = (1U << 24) - 1;
inline constexpr std::uint32_t kAbliterationCountShift = 24;
inline constexpr std::uint32_t kAbliterationPreserveNormBit = 1U << 27;

/// Immutable, model-bound runtime projection. No model weights are edited.
/// JSON keeps this sidecar independent of the weight shard loader.
struct AbliterationProfile {
  /// 1 or 2 (kAbliterationSchemaV1/V2).
  std::uint32_t schema{0};
  std::uint32_t hidden{0};
  std::uint32_t layers{0};
  /// Orthonormal directions per layer and kind (1 without D18 members).
  std::uint32_t rank{1};
  /// Direction kinds stored: 1 (mean difference) or 2 (also projected).
  std::uint32_t kinds{1};
  /// Unit directions, row-major [rows][hidden]: DirectionRow(kind, layer)
  /// starts `rank` orthonormal rows; the first is the kind's single
  /// direction. v2 rank 1 with one kind: `layers` rows, row l measured at
  /// layer l. v1: one row of unknown source layer.
  std::vector<float> directions;
  /// The default direction row's source layer; empty for v1.
  std::optional<std::uint32_t> direction_layer;
  /// Request defaults of the D18 options.
  AbliterationDirectionKind direction_kind{
      AbliterationDirectionKind::kMeanDiff};
  std::uint32_t direction_count{1};
  bool preserve_norm{false};
  /// How the subspace was fitted ("pca", "som"); empty when not recorded.
  std::string subspace_method;
  /// Default per-layer strengths. `residual` is all zero for v1 and for a v2
  /// profile without that member.
  std::vector<float> attention;
  std::vector<float> mlp;
  std::vector<float> residual;
  /// SHA-256 of the exact profile file bytes.
  std::string identity;
  std::string model_identity;
  std::string placement;
  /// The file name without `.json`; requests select the profile by it.
  std::string name;
  /// Optional v2 member `trigger`: only such a profile turns ON by itself.
  std::optional<AbliterationTrigger> trigger;

  [[nodiscard]] std::uint32_t direction_rows() const noexcept {
    return hidden ? static_cast<std::uint32_t>(directions.size() / hidden) : 0;
  }
  /// Whether a request may choose its direction layer (v2).
  [[nodiscard]] bool selectable_direction() const noexcept {
    return direction_layer.has_value();
  }
  [[nodiscard]] bool has_kind(AbliterationDirectionKind kind) const noexcept {
    return static_cast<std::uint32_t>(kind) < kinds;
  }
  /// The first of `rank` rows of `kind` measured at `layer` (0 for v1).
  [[nodiscard]] std::uint32_t DirectionRow(AbliterationDirectionKind kind,
                                           std::uint32_t layer) const noexcept {
    return direction_layer
               ? (static_cast<std::uint32_t>(kind) * layers + layer) * rank
               : 0;
  }

  static AbliterationProfile Load(const std::filesystem::path& path,
                                  std::uint32_t hidden, std::uint32_t layers,
                                  std::string_view model_identity,
                                  std::string_view placement);
};

/// One session's effective projection: the direction and every layer's
/// strengths, with an identity that salts its cache state. A null settings
/// pointer everywhere means OFF.
struct AbliterationSettings {
  /// AbliterationProfile::name of the profile these settings resolve; not
  /// part of `identity`, which already covers the profile's bytes.
  std::string profile;
  /// The chosen direction's source layer; empty for a v1 profile.
  std::optional<std::uint32_t> direction_layer;
  /// First row of AbliterationProfile::directions; the projection uses
  /// `direction_count` consecutive orthonormal rows from it.
  std::uint32_t direction_row{0};
  AbliterationDirectionKind direction_kind{
      AbliterationDirectionKind::kMeanDiff};
  /// In [1, profile rank]: y -= s * sum_j d_j * dot(d_j, y).
  std::uint32_t direction_count{1};
  /// After the projection, rescale each projected row (writer output row,
  /// residual row or HC stream) to its L2 norm before the projection.
  bool preserve_norm{false};
  std::vector<float> attention;
  std::vector<float> mlp;
  /// Post-layer residual strengths: r -= s * d * dot(d, r) at the point the
  /// collectors read the directions, so s = 1 zeroes that component there.
  std::vector<float> residual;
  /// Set on settings a dynamic profile turned ON in the middle of a request
  /// (DynamicAbliterationSettings): the session's first this-many tokens were
  /// computed OFF, so a state of at most this many tokens is an OFF state.
  std::optional<std::uint32_t> dynamic_from;
  /// 64 lowercase hex characters: SHA-256 over the profile identity, the
  /// direction layer, the exact float bytes of `attention`, `mlp` and
  /// `residual`, the direction kind, count and norm preservation
  /// (gufo-abliteration-settings-v3).
  std::string identity;

  /// The device copy each ON session owns: AbliterationDeviceWords(layers)
  /// 32-bit words, read by the projection kernels at launch time. Word 0
  /// packs `direction_row`, `direction_count` and `preserve_norm` as integer
  /// bits (kAbliterationRowMask and below; exactly `direction_row` for one
  /// direction without norm preservation); word AbliterationWeightIndex(...)
  /// is one site's strength.
  /// `row_offset` is added to the direction row: a model holding several
  /// profiles in one direction table passes the profile's first row.
  [[nodiscard]] std::vector<float> DeviceWords(
      std::uint32_t row_offset = 0) const;
  /// Whether any residual strength is nonzero. Only then do the models launch
  /// the residual projection (at every layer, zero strengths included) and
  /// leave fusions that derive values from the post-layer residual. Like
  /// ON/OFF, a change of this bit changes which kernels a pass launches.
  [[nodiscard]] bool ResidualActive() const noexcept;
};

/// Where a strength applies; also its block of the device words.
enum class AbliterationPoint : std::uint32_t {
  kAttention = 0,  ///< attention/GDN writer output
  kMlp = 1,        ///< MLP/MoE writer output
  kResidual = 2,   ///< post-layer residual (each HC stream on Flash-Next)
};
[[nodiscard]] constexpr std::size_t AbliterationDeviceWords(
    std::uint32_t layers) noexcept {
  return 1 + 3 * std::size_t{layers};
}
[[nodiscard]] constexpr std::uint32_t AbliterationWeightIndex(
    std::uint32_t layers, std::uint32_t layer,
    AbliterationPoint point) noexcept {
  return 1 + static_cast<std::uint32_t>(point) * layers + layer;
}
[[nodiscard]] constexpr std::uint32_t AbliterationWeightIndex(
    std::uint32_t layers, std::uint32_t layer, bool mlp) noexcept {
  return AbliterationWeightIndex(
      layers, layer,
      mlp ? AbliterationPoint::kMlp : AbliterationPoint::kAttention);
}

/// The request's members of `{"abliteration": {...}}`. Empty strength vectors
/// keep the profile defaults; one value applies to every layer; otherwise
/// exactly `layers` values.
struct AbliterationOverrides {
  /// AbliterationProfile::name; empty selects the only loaded profile.
  std::string profile;
  std::optional<std::uint32_t> direction_layer;
  std::vector<float> attention;
  std::vector<float> mlp;
  std::vector<float> residual;
  /// D18: `direction_kind`, `directions` (count) and `preserve_norm`.
  std::optional<AbliterationDirectionKind> direction_kind;
  std::optional<std::uint32_t> direction_count;
  std::optional<bool> preserve_norm;
};

/// The effective settings of `overrides` on `profile`'s defaults. Throws
/// std::invalid_argument on an out-of-range layer, a direction layer on a v1
/// profile, a wrong strength count, a strength outside [0, 4], a kind the
/// profile does not store, or a direction count outside [1, rank].
std::shared_ptr<const AbliterationSettings> ResolveAbliterationSettings(
    const AbliterationProfile& profile,
    const AbliterationOverrides& overrides = {});

/// `base` turned ON by a detector at `position` (R2 A1): the same
/// projection, `dynamic_from` = position and an identity salted with the
/// position (gufo-abliteration-dynamic-v1), so a state that mixes OFF and ON
/// tokens never matches an explicit ON request, another switch point or
/// OFF. States of at most `position` tokens are OFF states
/// (AbliterationForState).
std::shared_ptr<const AbliterationSettings> DynamicAbliterationSettings(
    const AbliterationSettings& base, std::uint32_t position);
/// The settings that computed a state of `tokens` tokens of a session whose
/// active settings are `active`: null for OFF, and for a dynamic switch at
/// or after `tokens`.
[[nodiscard]] const AbliterationSettings* AbliterationForState(
    const AbliterationSettings* active, std::size_t tokens) noexcept;

/// Whether `settings` index only `profile`'s layers and direction rows.
[[nodiscard]] bool AbliterationSettingsFit(
    const AbliterationProfile& profile,
    const AbliterationSettings& settings) noexcept;

/// Domain-separated input identity; OFF callers keep their original bytes.
/// `settings_identity` is AbliterationSettings::identity.
std::vector<std::uint8_t> AbliterationInputIdentity(
    std::span<const std::uint8_t> input, std::string_view settings_identity);

/// Loads one profile file. An empty path loads none; no file beside the
/// model is ever picked up implicitly.
std::shared_ptr<const AbliterationProfile> LoadAbliterationOption(
    const std::string& option, std::uint32_t hidden, std::uint32_t layers,
    std::string_view model_identity, std::string_view placement);

/// The profile files of `--abliteration-profiles DIR`: every regular
/// `*.json` file directly in DIR, sorted by name. An empty option lists none.
/// Throws std::invalid_argument for a missing directory, one without
/// profiles, or more than kAbliterationMaxProfiles.
std::vector<std::string> ListAbliterationProfiles(const std::string& option);

/// The profile `name` selects among `profiles` (empty: the only one).
/// Throws std::invalid_argument naming the loaded profiles otherwise.
const AbliterationProfile& SelectAbliterationProfile(
    std::span<const std::shared_ptr<const AbliterationProfile>> profiles,
    std::string_view name);

/// Mode and settings recorded in a component-cache continuation: a guard
/// independent of the salted context identity. OFF (`active` null) records
/// mode 0 and the raw image identity, exactly the bytes continuations carried
/// before abliteration existed. ON records mode 1 and the image identity
/// salted with the full settings identity.
struct AbliterationContinuationRecord {
  static constexpr std::uint32_t kOff = 0;
  static constexpr std::uint32_t kOn = 1;
  std::uint32_t mode{kOff};
  std::array<std::uint8_t, 32> image{};
};
/// `image` is the session's image identity at the captured position (empty
/// or 32 bytes).
AbliterationContinuationRecord RecordAbliterationContinuation(
    std::span<const std::uint8_t> image, const AbliterationSettings* active);
/// Whether a recorded continuation may restore into a session whose image
/// identity is `image` and whose active settings are `active` (null for OFF).
/// `image_bytes` is the recorded raw image identity length.
bool AbliterationContinuationMatches(std::uint32_t mode,
                                     std::uint32_t image_bytes,
                                     std::span<const std::uint8_t, 32> recorded,
                                     std::span<const std::uint8_t> image,
                                     const AbliterationSettings* active);

}  // namespace gufo::models::qwen
#endif
