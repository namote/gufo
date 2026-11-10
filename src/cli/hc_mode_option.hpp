#ifndef GUFO_CLI_HC_MODE_OPTION_HPP_
#define GUFO_CLI_HC_MODE_OPTION_HPP_

#include <optional>
#include <string>
#include <string_view>

#include "src/cli/arg_parser.hpp"
#include "src/models/qwen38_flash_next/hc_mode.hpp"

namespace gufo::cli {

/// Registers Flash-Next's `--hc-mode`; an absent option leaves `*target`
/// empty, which loads fp16. Other models reject an explicit value.
inline void RegisterHcModeOption(
    ArgParser& parser, std::optional<models::qwen38_flash_next::HcMode>* target,
    std::string_view group) {
  parser.AddCustomOption(
      "", "--hc-mode", "MODE",
      "Flash-Next HyperConnection mixer weights: fp16 (stored F16), q8 "
      "(Q8_0 rounded at load) or q8-calibrated (the pack's hc_q8/ sidecar, "
      "required) (default: fp16)",
      group,
      [target](std::string_view, std::string_view value, std::string* error) {
        const auto mode = models::qwen38_flash_next::ParseHcMode(value);
        if (!mode) {
          if (error != nullptr)
            *error = "--hc-mode must be fp16, q8 or q8-calibrated";
          return false;
        }
        *target = mode;
        return true;
      });
}

}  // namespace gufo::cli

#endif  // GUFO_CLI_HC_MODE_OPTION_HPP_
