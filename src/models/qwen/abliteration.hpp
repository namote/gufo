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
/// all zero) are the request defaults.
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
/// written with float32 round-trip precision (prepare.py).
inline constexpr std::size_t kAbliterationMaxProfileBytes = 16 << 20;

/// Immutable, model-bound runtime projection. No model weights are edited.
/// JSON keeps this sidecar independent of the weight shard loader.
struct AbliterationProfile {
  /// 1 or 2 (kAbliterationSchemaV1/V2).
  std::uint32_t schema{0};
  std::uint32_t hidden{0};
  std::uint32_t layers{0};
  /// Unit directions, row-major [rows][hidden]. v2: `layers` rows, row l
  /// measured at layer l. v1: one row of unknown source layer.
  std::vector<float> directions;
  /// The default direction row's source layer; empty for v1.
  std::optional<std::uint32_t> direction_layer;
  /// Default per-layer strengths. `residual` is all zero for v1 and for a v2
  /// profile without that member.
  std::vector<float> attention;
  std::vector<float> mlp;
  std::vector<float> residual;
  /// SHA-256 of the exact profile file bytes.
  std::string identity;
  std::string model_identity;
  std::string placement;

  [[nodiscard]] std::uint32_t direction_rows() const noexcept {
    return hidden ? static_cast<std::uint32_t>(directions.size() / hidden) : 0;
  }
  /// Whether a request may choose its direction layer (v2).
  [[nodiscard]] bool selectable_direction() const noexcept {
    return direction_layer.has_value();
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
  /// The chosen direction's source layer; empty for a v1 profile.
  std::optional<std::uint32_t> direction_layer;
  /// Row of AbliterationProfile::directions.
  std::uint32_t direction_row{0};
  std::vector<float> attention;
  std::vector<float> mlp;
  /// Post-layer residual strengths: r -= s * d * dot(d, r) at the point the
  /// collectors read the directions, so s = 1 zeroes that component there.
  std::vector<float> residual;
  /// 64 lowercase hex characters: SHA-256 over the profile identity, the
  /// direction layer and the exact float bytes of `attention`, `mlp` and
  /// `residual`.
  std::string identity;

  /// The device copy each ON session owns: AbliterationDeviceWords(layers)
  /// 32-bit words, read by the projection kernels at launch time. Word 0 is
  /// `direction_row` (as integer bits); word AbliterationWeightIndex(...) is
  /// one site's strength.
  [[nodiscard]] std::vector<float> DeviceWords() const;
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
  std::optional<std::uint32_t> direction_layer;
  std::vector<float> attention;
  std::vector<float> mlp;
  std::vector<float> residual;
};

/// The effective settings of `overrides` on `profile`'s defaults. Throws
/// std::invalid_argument on an out-of-range layer, a direction layer on a v1
/// profile, a wrong strength count, or a strength outside [0, 4].
std::shared_ptr<const AbliterationSettings> ResolveAbliterationSettings(
    const AbliterationProfile& profile,
    const AbliterationOverrides& overrides = {});

/// Domain-separated input identity; OFF callers keep their original bytes.
/// `settings_identity` is AbliterationSettings::identity.
std::vector<std::uint8_t> AbliterationInputIdentity(
    std::span<const std::uint8_t> input, std::string_view settings_identity);

/// Loads the explicit `--abliteration-profile` file. An empty option loads
/// none; no file beside the model is ever picked up implicitly.
std::shared_ptr<const AbliterationProfile> LoadAbliterationOption(
    const std::string& option, std::uint32_t hidden, std::uint32_t layers,
    std::string_view model_identity, std::string_view placement);

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
