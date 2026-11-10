#include "src/models/qwen38_flash_next/hc_mode.hpp"

#include <iostream>
#include <stdexcept>

namespace q = gufo::models::qwen38_flash_next;

namespace {
void Require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}

void ParsesCanonicalNames() {
  Require(q::kDefaultHcMode == q::HcMode::kFp16, "default is not fp16");
  for (const auto mode :
       {q::HcMode::kFp16, q::HcMode::kQ8, q::HcMode::kQ8Calibrated})
    Require(q::ParseHcMode(q::HcModeName(mode)) == mode, "name round trip");
  Require(q::HcModeName(q::HcMode::kFp16) == "fp16", "fp16 name");
  Require(q::HcModeName(q::HcMode::kQ8) == "q8", "q8 name");
  Require(q::HcModeName(q::HcMode::kQ8Calibrated) == "q8-calibrated",
          "q8-calibrated name");
  // One canonical spelling per mode: no aliases.
  for (const char* other : {"", "f16", "FP16", "q8_0", "Q8", "calibrated",
                            "q8_calibrated", "q8-calibrated "})
    Require(!q::ParseHcMode(other), "accepted a non-canonical mode");
}

void PlansMixerUploads() {
  // fp16 uploads the stored weights, never a Q8 copy.
  for (const bool exl3 : {false, true}) {
    const auto plan = q::PlanHcMixer(q::HcMode::kFp16, false, true, true, exl3);
    Require(plan.valid && !plan.calibrated && !plan.convert_down &&
                !plan.convert_up,
            "fp16 converted the mixer");
  }
  // q8 is the load-time conversion on native EXL3 packs only.
  auto plan = q::PlanHcMixer(q::HcMode::kQ8, false, true, true, true);
  Require(
      plan.valid && !plan.calibrated && plan.convert_down && plan.convert_up,
      "q8 did not convert an EXL3 mixer");
  plan = q::PlanHcMixer(q::HcMode::kQ8, false, true, true, false);
  Require(plan.valid && !plan.convert_down && !plan.convert_up,
          "q8 converted a GGUF mixer");
  plan = q::PlanHcMixer(q::HcMode::kQ8, false, false, true, true);
  Require(plan.valid && !plan.convert_down && !plan.convert_up,
          "q8 converted without an F16 down projection");
  // q8-calibrated uploads the sidecar pair and converts nothing.
  plan = q::PlanHcMixer(q::HcMode::kQ8Calibrated, true, true, true, true);
  Require(
      plan.valid && plan.calibrated && !plan.convert_down && !plan.convert_up,
      "q8-calibrated did not take the sidecar");
  // Sidecar presence must match the mode: no silent switch.
  Require(
      !q::PlanHcMixer(q::HcMode::kQ8Calibrated, false, true, true, true).valid,
      "q8-calibrated accepted a mixer without the sidecar");
  Require(!q::PlanHcMixer(q::HcMode::kFp16, true, true, true, true).valid,
          "fp16 accepted calibrated weights");
  Require(!q::PlanHcMixer(q::HcMode::kQ8, true, true, true, true).valid,
          "q8 accepted calibrated weights");
}
}  // namespace

int main() {
  try {
    ParsesCanonicalNames();
    PlansMixerUploads();
    std::cout << "Flash-Next HC mode parsing and mixer plans passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
