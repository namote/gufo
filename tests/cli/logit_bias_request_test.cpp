#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/sampling_request.hpp"
#include "src/core/json.hpp"
#include "src/core/logit_bias.hpp"

namespace {

using gufo::sampling::LogitBiasCatalog;
using gufo::sampling::LogitBiasProfile;
using gufo::sampling::LogitBiasScope;
using gufo::sampling::SamplingConfig;
using gufo::server::ParseSamplingConfig;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

LogitBiasCatalog Catalog() {
  std::vector<LogitBiasProfile> profiles(2);
  profiles[0].name = "wait";
  profiles[0].scope = LogitBiasScope::kReasoning;
  profiles[0].bias = -4.0F;
  profiles[0].tokens = {1, 2};
  profiles[1].name = "hmm";
  profiles[1].scope = LogitBiasScope::kAll;
  profiles[1].bias = -2.0F;
  profiles[1].tokens = {2, 3};
  return LogitBiasCatalog(16, std::move(profiles));
}

struct Parsed {
  SamplingConfig config;
  std::string code;  ///< empty when accepted
};

Parsed Parse(std::string_view body, const LogitBiasCatalog* catalog) {
  Parsed result;
  if (const auto error = ParseSamplingConfig(gufo::json::parse(body), {},
                                             &result.config, catalog))
    result.code = error->code;
  return result;
}

void TestSelection() {
  const auto catalog = Catalog();
  const auto accepted =
      Parse(R"({"logit_bias_profiles":["wait","hmm"]})", &catalog);
  Expect(accepted.code.empty() && accepted.config.logit_bias_active(),
         "several profiles combine");
  const auto& bias = *accepted.config.logit_bias;
  Expect(bias.phase_dependent() && (*bias.reasoning)[1] == -4.0F &&
             (*bias.reasoning)[2] == -6.0F && (*bias.answer)[2] == -2.0F &&
             (*bias.answer)[1] == 0.0F,
         "overlapping sets add per phase");

  const auto overridden = Parse(R"({"logit_bias_profiles":["wait","hmm"],
                "logit_bias_overrides":{"wait":-6}})",
                                &catalog);
  Expect(overridden.code.empty() &&
             (*overridden.config.logit_bias->reasoning)[1] == -6.0F &&
             (*overridden.config.logit_bias->reasoning)[3] == -2.0F,
         "an override replaces one profile's bias");

  Expect(Parse(R"({"logit_bias_profiles":["nope"]})", &catalog).code ==
             "unknown_logit_bias_profile",
         "unknown names are rejected");
  Expect(Parse(R"({"logit_bias_profiles":["wait","wait"]})", &catalog).code ==
             "invalid_logit_bias_profiles",
         "a name selected twice is rejected");
  Expect(Parse(R"({"logit_bias_profiles":"wait"})", &catalog).code ==
             "invalid_logit_bias_profiles",
         "the selection is an array");
  Expect(Parse(R"({"logit_bias_profiles":[1]})", &catalog).code ==
             "invalid_logit_bias_profiles",
         "the selection holds names");
  Expect(Parse(R"({"logit_bias_profiles":["wait","hmm","a","b","c","d","e",
                   "f","g"]})",
               &catalog)
                 .code == "invalid_logit_bias_profiles",
         "at most eight profiles");
  Expect(Parse(R"({"logit_bias_profiles":["wait"],
                   "logit_bias_overrides":{"hmm":-1}})",
               &catalog)
                 .code == "invalid_logit_bias_overrides",
         "an override must name a selected profile");
  Expect(Parse(R"({"logit_bias_profiles":["wait"],
                   "logit_bias_overrides":{"wait":-101}})",
               &catalog)
                 .code == "invalid_logit_bias_overrides",
         "overrides are bounded");
  Expect(Parse(R"({"logit_bias_profiles":["wait"],
                   "logit_bias_overrides":[-1]})",
               &catalog)
                 .code == "invalid_logit_bias_overrides",
         "overrides are an object");
  const auto nulls = Parse(R"({"logit_bias_profiles":null,
                               "logit_bias_overrides":null,
                               "logit_bias":null})",
                           &catalog);
  Expect(nulls.code.empty() && !nulls.config.logit_bias,
         "null fields select nothing");
  Expect(
      Parse(R"({"logit_bias_profiles":[]})", &catalog).code.empty() &&
          !Parse(R"({"logit_bias_profiles":[]})", &catalog).config.logit_bias,
      "an empty selection adds nothing");
  Expect(Parse(R"({"logit_bias_other":1})", &catalog).code ==
             "unsupported_sampling",
         "other logit_bias spellings stay unsupported");
}

void TestOpenAiMap() {
  const auto catalog = Catalog();
  const auto map = Parse(R"({"logit_bias":{"5":-100,"0":2.5}})", &catalog);
  Expect(map.code.empty() && !map.config.logit_bias->phase_dependent() &&
             (*map.config.logit_bias->answer)[5] == -100.0F &&
             (*map.config.logit_bias->answer)[0] == 2.5F &&
             map.config.logit_bias->answer_max == 2.5F,
         "the OpenAI map is an ad hoc set with scope all");
  const auto both = Parse(R"({"logit_bias":{"1":1},
                              "logit_bias_profiles":["wait"]})",
                          &catalog);
  Expect(both.code.empty() &&
             (*both.config.logit_bias->reasoning)[1] == -3.0F &&
             (*both.config.logit_bias->answer)[1] == 1.0F,
         "the map adds to the selected profiles");
  for (const auto* body :
       {R"({"logit_bias":{"16":1}})", R"({"logit_bias":{"-1":1}})",
        R"({"logit_bias":{"01":1}})", R"({"logit_bias":{"a":1}})",
        R"({"logit_bias":{" 1":1}})", R"({"logit_bias":{"1":101}})",
        R"({"logit_bias":{"1":"1"}})", R"({"logit_bias":0.8})",
        R"({"logit_bias":[1]})"})
    Expect(Parse(body, &catalog).code == "invalid_logit_bias",
           std::string("malformed logit_bias is rejected: ") + body);
  Expect(Parse(R"({"logit_bias":{}})", &catalog).code.empty() &&
             !Parse(R"({"logit_bias":{}})", &catalog).config.logit_bias,
         "an empty map adds nothing");
}

void TestUnsupportedBackend() {
  // Without a catalog the fields stay unsupported, as before.
  for (const auto* body : {R"({"logit_bias":{"1":1}})", R"({"logit_bias":0.8})",
                           R"({"logit_bias_profiles":["wait"]})",
                           R"({"logit_bias_overrides":{"wait":1}})"})
    Expect(Parse(body, nullptr).code == "unsupported_sampling",
           std::string("unsupported without profiles: ") + body);
  for (const auto* body : {R"({"logit_bias":{}})", R"({"logit_bias":null})",
                           R"({"logit_bias_profiles":null})"})
    Expect(Parse(body, nullptr).code.empty(),
           std::string("neutral values stay accepted: ") + body);
}

}  // namespace

int main() {
  TestSelection();
  TestOpenAiMap();
  TestUnsupportedBackend();
  std::cout << "logit_bias_request_test passed\n";
  return 0;
}
