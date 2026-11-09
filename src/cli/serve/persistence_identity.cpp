#include "src/cli/serve/persistence_identity.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

#include "src/core/gguf_identity.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::server {

bool IsSha256Hex(std::string_view value) noexcept {
  return value.size() == 64 && std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::vector<std::uint8_t> QwenFlashNextCompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view mtp_fingerprint,
    bool has_mtp, std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::uint32_t decode_concurrency, std::string_view position_policy,
    bool native, std::uint32_t payload_version,
    std::string_view device_code_sha256) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen3.8-Flash-Next disk cache requires an artifact fingerprint");
  }
  if (has_mtp && !IsSha256Hex(mtp_fingerprint)) {
    throw std::invalid_argument(
        "Qwen3.8-Flash-Next MTP disk cache requires a draft artifact "
        "fingerprint");
  }
  const auto scheme =
      native ? core::kNativeIdentityScheme : core::kGgufIdentityScheme;
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v2\n"
           << "model_kind=qwen38-flash-next\n"
           << "artifact_id=" << scheme << ':' << artifact_fingerprint << '\n'
           << "tokenizer=embedded-in-artifact\n"
           << "chat_template=qwen38-reasoning-compiled-v3\n"
           << "chat_template_reference_sha256="
           << tokenization::QwenChatTemplate::OfficialTemplateSha256() << '\n'
           << "state_abi=qwen38-flash-next-rocm-session-v1\n"
           << "payload_layout=qfn-rocm-session-snapshot-v" << payload_version
           << '\n'
           << "device_code="
           << (device_code_sha256.empty() ? std::string_view("unknown")
                                          : std::string_view("sha256:"))
           << device_code_sha256 << '\n'
           << "context_tokens=" << max_context << '\n'
           << "position_policy=" << position_policy << '\n'
           << "adapters=none\n";
  if (has_mtp) {
    identity << "draft_backend=qfn-mtp-v1\n"
             << "draft_artifact_id=" << scheme << ':' << mtp_fingerprint << '\n'
             << "draft_max_tokens=" << max_draft_tokens << '\n'
             << "draft_cost_concurrency=" << decode_concurrency << '\n';
  }
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

}  // namespace gufo::server
