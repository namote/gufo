// In-server activation collection jobs (`--collect-dir`) over a fake runner:
// validation, lifecycle, `since` paging, collect.cpp's output layout, resume
// and cancellation.
#include "src/cli/serve/collection_jobs.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"

namespace fs = std::filesystem;
namespace json = gufo::json;
namespace q = gufo::models::qwen38_flash_next;
using gufo::server::CollectionJobError;
using gufo::server::CollectionJobOptions;
using gufo::server::CollectionJobService;
using gufo::server::CollectionModelIdentity;
using gufo::server::CollectionRunner;

namespace {
void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAILED: " << what << '\n';
    std::exit(1);
  }
}

constexpr std::uint32_t kLayers = 3;
constexpr std::uint32_t kHidden = 2;

/// Tokens are the text's bytes; a record's readout is its id's first byte
/// plus the element index. `hold` blocks Run until released (or cancelled).
class FakeRunner final : public CollectionRunner {
public:
  CollectionModelIdentity Identity() const override {
    return {.model_identity = "model",
            .device_code = "device",
            .executable_sha256 = "exe",
            .hidden = kHidden,
            .layers = kLayers,
            .deviations = {"hc-mode q8-calibrated"}};
  }
  q::CollectionPlan Plan(q::CollectionMode mode,
                         const json::Value& record) const override {
    const auto text = record.member_str("text");
    if (text == "too long")
      throw std::invalid_argument("prompt invalid/too long");
    q::CollectionPlan plan;
    plan.prompt.assign(text.begin(), text.end());
    plan.prompt.push_back(0);
    if (mode == q::CollectionMode::kAnswers) {
      const auto answer = record.member_str("answer");
      plan.answer.assign(answer.begin(), answer.end());
    }
    if (mode == q::CollectionMode::kSegments) {
      // One token per segment; segment k's end row predicts segment k+1.
      for (const auto& part : record.find("segments")->items())
        plan.answer.push_back(part.str()[0]);
      plan.selected.assign(plan.answer.size(), false);
      for (const auto& k : record.find("select")->items())
        if (!plan.selected[k.as_size() + 1]) {
          plan.selected[k.as_size() + 1] = true;
          ++plan.selected_rows;
        }
    }
    return plan;
  }
  std::optional<std::vector<float>> Run(
      q::CollectionMode, const q::CollectionPlan&,
      const std::function<bool()>& cancelled,
      const std::function<void(std::string)>& step) override {
    step("waiting for an idle scheduler");
    std::unique_lock<std::mutex> lock(mutex);
    ++running;
    changed.notify_all();
    while (hold) {
      if (cancelled()) {
        --running;
        return std::nullopt;
      }
      changed.wait_for(lock, std::chrono::milliseconds(5));
    }
    --running;
    ++runs;
    if (fail_after && runs > fail_after)
      throw std::runtime_error("injected pass failure");
    std::vector<float> values(std::size_t{kLayers} * kHidden);
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = float(runs * 100 + i) + 0.25F;
    return values;
  }
  void Release() override { ++releases; }

  void Hold(bool value) {
    const std::lock_guard<std::mutex> lock(mutex);
    hold = value;
    changed.notify_all();
  }
  void WaitRunning() {
    std::unique_lock<std::mutex> lock(mutex);
    changed.wait(lock, [&] { return running > 0; });
  }

  std::mutex mutex;
  std::condition_variable changed;
  bool hold{false};
  int running{0};
  int runs{0};
  int fail_after{0};
  std::atomic<int> releases{0};
};

struct Fixture {
  explicit Fixture(std::size_t max_jobs = gufo::server::kMaxCollectionJobs)
      : root(fs::temp_directory_path() /
             ("gufo-collection-jobs-" +
              std::to_string(
                  std::chrono::steady_clock::now().time_since_epoch().count()) +
              "-" + std::to_string(counter++))),
        runner(std::make_shared<FakeRunner>()) {
    fs::remove_all(root);
    fs::create_directories(root);
    service = std::make_unique<CollectionJobService>(CollectionJobOptions{
        .root = root, .runner = runner, .max_jobs = max_jobs});
  }
  ~Fixture() {
    runner->Hold(false);
    service.reset();
    fs::remove_all(root);
  }
  std::string Start(const std::string& body) {
    return service->Create(body).member_str("job");
  }
  json::Value WaitFinished(const std::string& key) {
    for (int i = 0; i < 2000; ++i) {
      auto job = service->Get(key, 0);
      const auto status = job.member_str("status");
      if (status == "done" || status == "failed" || status == "cancelled")
        return job;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    Expect(false, "job finishes");
    return {};
  }

  static inline int counter = 0;
  fs::path root;
  std::shared_ptr<FakeRunner> runner;
  std::unique_ptr<CollectionJobService> service;
};

std::string CollectRecord(const std::string& id,
                          const std::string& group = "refusal") {
  return R"({"id":")" + id + R"(","split":"collection","group":")" + group +
         R"(","thinking":false,"text":"hi )" + id + R"("})";
}

std::string Body(const std::string& name,
                 const std::vector<std::string>& records,
                 const std::string& mode = "collect", bool resume = false) {
  std::string body = R"({"operation":"collect","mode":")" + mode +
                     R"(","name":")" + name + R"(",)";
  if (resume)
    body += R"("resume":true,)";
  body += R"("records":[)";
  for (std::size_t i = 0; i < records.size(); ++i)
    body += (i ? "," : "") + records[i];
  return body + "]}";
}

std::string ErrorCode(Fixture& fixture, const std::string& body,
                      int* status = nullptr) {
  try {
    (void)fixture.service->Create(body);
  } catch (const CollectionJobError& e) {
    if (status)
      *status = e.status();
    return e.code();
  }
  return {};
}

std::string ReadFile(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), {}};
}

void TestValidation() {
  Fixture f;
  int status = 0;
  Expect(ErrorCode(f, "{", &status) == "invalid_json" && status == 400,
         "invalid JSON is 400");
  Expect(ErrorCode(f, R"({"operation":"fit","mode":"collect"})") ==
             "unsupported_operation",
         "only the collect operation");
  Expect(ErrorCode(f, R"({"operation":"collect","mode":"evaluate"})") ==
             "invalid_mode",
         "evaluate is not a collection mode");
  for (const auto* name : {"", "..", ".", "a/b", "a b"})
    Expect(ErrorCode(f, Body(name, {CollectRecord("a1")})) == "invalid_name",
           std::string("name refused: ") + name);
  Expect(ErrorCode(f, Body("x", {})) == "invalid_records",
         "at least one record");
  Expect(
      ErrorCode(f, Body("x", {CollectRecord("a1", "own")})) == "invalid_record",
      "collect groups are collect.cpp's");
  Expect(ErrorCode(f, Body("x",
                           {R"({"id":"b1","group":"own","pair":"p",)"
                            R"("text":"t"})"},
                           "answers")) == "invalid_record",
         "answers need an answer");
  Expect(ErrorCode(f, Body("x",
                           {R"({"id":"c1","group":"own","pair":"p",)"
                            R"("text":"t","segments":["a"],"select":[0]})"},
                           "segments")) == "invalid_record",
         "segments need two segments");
  Expect(ErrorCode(f, Body("x", {CollectRecord("a1"), CollectRecord("a1")})) ==
             "invalid_record",
         "duplicate IDs are refused");
  Expect(ErrorCode(f, Body("x", {R"({"id":"a1","group":"refusal",)"
                                 R"("thinking":true,"text":"too long"})"})) ==
             "invalid_record",
         "a record the runner cannot plan is refused at validation");
  Expect(ErrorCode(f, Body("x", {CollectRecord("a1"),
                                 R"({"id":"a2","split":"other",)"
                                 R"("group":"refusal","thinking":true})"})) ==
             "invalid_record",
         "one split per job");
  Expect(!fs::exists(f.root / "x"), "a refused job writes nothing");
  bool unknown = false;
  try {
    (void)f.service->Get("nope", 0);
  } catch (const CollectionJobError& e) {
    unknown = e.status() == 404;
  }
  Expect(unknown, "unknown keys are 404");
}

void TestLifecycleAndLayout() {
  Fixture f;
  f.runner->Hold(true);
  const auto key = f.Start(
      Body("first", {CollectRecord("a1"), CollectRecord("a2", "ordinary"),
                     CollectRecord("a3")}));
  Expect(key.size() == 32 &&
             key.find_first_not_of("0123456789abcdef") == std::string::npos,
         "keys are 128 random bits in hex");
  const auto second = f.Start(Body("second", {CollectRecord("b1")}));
  Expect(second != key, "keys differ");
  f.runner->WaitRunning();
  auto running = f.service->Get(key, 0);
  Expect(running.member_str("status") == "running", "the first job runs");
  Expect(running.find("progress")->member_size("total") == 3 &&
             running.find("progress")->member_size("done") == 0,
         "progress counts records");
  Expect(
      running.find("progress")->member_str("step").find("record 1/3 a1") == 0,
      "the step names the record: " +
          running.find("progress")->member_str("step"));
  Expect(running.find("numerics_deviations")->size() == 1,
         "deviations from collect.cpp are reported");
  Expect(f.service->Get(second, 0).member_str("status") == "queued",
         "one job runs at a time; the next queues");
  Expect(
      ErrorCode(f, Body("first", {CollectRecord("c1")})) == "collection_busy",
      "a running job's name is busy");
  const auto list = f.service->List();
  Expect(list.find("data")->size() == 2 &&
             list.find("data")->items()[0].member_str("job") == second &&
             !list.find("data")->items()[0].contains("results"),
         "the list is newest first, without results");
  f.runner->Hold(false);
  const auto done = f.WaitFinished(key);
  Expect(done.member_str("status") == "done", "the job completes");
  Expect(done.find("results")->size() == 3 && done.member_size("last_seq") == 3,
         "every record is a result");
  const auto& first = done.find("results")->items()[0];
  Expect(first.member_size("seq") == 1 && first.member_str("id") == "a1" &&
             first.member_str("group") == "refusal" &&
             first.member_size("rows") == 1 &&
             first.member_str("file") == "a1.f32" &&
             !first.find("resumed")->as_bool(),
         "result entries");
  const auto paged = f.service->Get(key, 2);
  Expect(paged.find("results")->size() == 1 &&
             paged.find("results")->items()[0].member_str("id") == "a3",
         "since=2 returns the third result only");
  Expect(f.service->Get(key, 9).find("results")->empty(),
         "since past the end is empty");
  Expect(f.WaitFinished(second).member_str("status") == "done",
         "the queued job runs next");
  Expect(f.runner->releases.load() == 2, "each job releases its session");

  // collect.cpp's layout: manifest.json, <id>.json and <id>.f32.
  const auto dir = f.root / "first";
  const auto manifest = json::parse(ReadFile(dir / "manifest.json"));
  std::vector<std::string> keys;
  for (const auto& [name, value] : manifest.members())
    keys.push_back(name);
  Expect(keys == std::vector<std::string>{"schema", "model_identity",
                                          "device_code", "executable_sha256",
                                          "context", "max_new_tokens",
                                          "template", "readout", "data_sha256",
                                          "hidden", "layers", "split", "mode",
                                          "profile", "records"},
         "the manifest has collect.cpp's members in its order");
  Expect(manifest.member_str("schema") == "gufo-abliteration-capture-v1" &&
             manifest.member_size("context") == 2048 &&
             manifest.member_size("max_new_tokens") == 0 &&
             manifest.member_str("template") == "gufo-qwen-native-medium-v1" &&
             manifest.member_str("readout") ==
                 "last-prompt-token-post-layer-hc-mean-fp32-v1" &&
             manifest.member_size("hidden") == kHidden &&
             manifest.member_size("layers") == kLayers &&
             manifest.member_str("split") == "collection" &&
             manifest.member_str("mode") == "collect" &&
             manifest.member_str("profile") == "off" &&
             manifest.find("records")->size() == 3,
         "manifest values");
  std::string lines;
  for (const auto& record : manifest.find("records")->items())
    lines += record.dump() + "\n";
  Expect(manifest.member_str("data_sha256") ==
             gufo::crypto::Sha256Hex(
                 std::span(reinterpret_cast<const std::uint8_t*>(lines.data()),
                           lines.size())),
         "data_sha256 hashes the records as JSON lines");
  const auto receipt = json::parse(ReadFile(dir / "a2.json"));
  keys.clear();
  for (const auto& [name, value] : receipt.members())
    keys.push_back(name);
  Expect(keys == std::vector<std::string>{"id", "prompt_tokens",
                                          "activation_sha256"},
         "collect receipts have collect.cpp's members");
  Expect(
      receipt.member_size("prompt_tokens") == std::string("hi a2").size() + 1,
      "prompt_tokens counts the plan's prompt");
  const auto bytes = ReadFile(dir / "a2.f32");
  Expect(bytes.size() == std::size_t{kLayers} * kHidden * 4,
         "activations are [layers][hidden] FP32");
  Expect(receipt.member_str("activation_sha256") ==
             gufo::crypto::Sha256FileHex(dir / "a2.f32"),
         "the receipt hashes the activation file");
  float value = 0;
  std::memcpy(&value, bytes.data() + 4, 4);
  Expect(value == 201.25F, "the runner's readout is written as is");
  // Finished jobs are forgotten on DELETE; their files stay.
  const auto deleted = f.service->Delete(key);
  Expect(deleted.find("deleted")->as_bool(), "a finished job is deleted");
  Expect(fs::exists(dir / "manifest.json"), "files stay");
  bool gone = false;
  try {
    (void)f.service->Get(key, 0);
  } catch (const CollectionJobError& e) {
    gone = e.status() == 404;
  }
  Expect(gone, "a deleted job is unknown");
}

void TestAnswersReceipts() {
  Fixture f;
  const auto key = f.Start(
      Body("ans",
           {R"({"id":"e1","split":"collection","group":"exemplar","pair":"p1",)"
            R"("text":"q","answer":"abc"})"},
           "answers"));
  const auto job = f.WaitFinished(key);
  Expect(job.member_str("status") == "done", "answers job completes");
  const auto& entry = job.find("results")->items()[0];
  Expect(entry.member_str("pair") == "p1" && entry.member_size("rows") == 3,
         "answers results carry the pair and the answer rows");
  const auto receipt = json::parse(ReadFile(f.root / "ans" / "e1.json"));
  std::vector<std::string> keys;
  for (const auto& [name, value] : receipt.members())
    keys.push_back(name);
  Expect(
      keys == std::vector<std::string>{"id", "prompt_tokens", "answer_tokens",
                                       "pair", "group", "activation_sha256"},
      "answers receipts have collect.cpp's members");
  const auto manifest = json::parse(ReadFile(f.root / "ans" / "manifest.json"));
  Expect(manifest.member_size("context") == 8192 &&
             manifest.member_str("template") == "gufo-qwen-native-scoring-v1",
         "answers manifest");
}

void TestSegmentsReceipts() {
  Fixture f;
  const auto key = f.Start(
      Body("seg",
           {R"({"id":"e2","split":"collection","group":"own","pair":"p2",)"
            R"("text":"q","segments":["a","b","c"],"select":[0,1,0]})"},
           "segments"));
  const auto job = f.WaitFinished(key);
  Expect(job.member_str("status") == "done", "segments job completes");
  Expect(job.find("results")->items()[0].member_size("rows") == 2,
         "segments results count the selected rows");
  const auto receipt = json::parse(ReadFile(f.root / "seg" / "e2.json"));
  std::vector<std::string> keys;
  for (const auto& [name, value] : receipt.members())
    keys.push_back(name);
  Expect(keys == std::vector<std::string>{"id", "prompt_tokens",
                                          "answer_tokens", "selected_rows",
                                          "pair", "group", "activation_sha256"},
         "segments receipts have collect.cpp's members");
  Expect(receipt.member_size("selected_rows") == 2 &&
             receipt.member_size("answer_tokens") == 3,
         "segments receipt values");
  const auto manifest = json::parse(ReadFile(f.root / "seg" / "manifest.json"));
  Expect(manifest.member_size("context") == 24576 &&
             manifest.member_str("template") ==
                 "gufo-qwen-native-thinking-medium-v1" &&
             manifest.member_str("readout") ==
                 "selected-segment-end-rows-mean-post-layer-hc-mean-fp32-v1",
         "segments manifest");
}

void TestResume() {
  Fixture f;
  const std::vector<std::string> records = {
      CollectRecord("a1"), CollectRecord("a2"), CollectRecord("a3")};
  f.runner->fail_after = 1;
  const auto failed = f.WaitFinished(f.Start(Body("run", records)));
  Expect(failed.member_str("status") == "failed" &&
             failed.member_str("error") == "injected pass failure",
         "a failing record fails the job");
  Expect(fs::exists(f.root / "run" / "a1.json") &&
             !fs::exists(f.root / "run" / "a2.json"),
         "completed records keep their receipts");
  int status = 0;
  Expect(ErrorCode(f, Body("run", records), &status) == "collection_exists" &&
             status == 409,
         "an existing collection is not overwritten");
  Expect(ErrorCode(f, Body("run", {CollectRecord("a1")}, "collect", true)) ==
             "collection_differs",
         "resume needs the same manifest");
  f.runner->fail_after = 0;
  const int before = f.runner->runs;
  const auto resumed =
      f.WaitFinished(f.Start(Body("run", records, "collect", true)));
  Expect(resumed.member_str("status") == "done", "the resumed job completes");
  const auto& results = resumed.find("results")->items();
  Expect(results.size() == 3 && results[0].find("resumed")->as_bool() &&
             !results[1].find("resumed")->as_bool(),
         "the completed record is reused");
  Expect(f.runner->runs - before == 2, "only the missing records run");
  // A corrupt activation file fails the resume as collect.cpp does.
  {
    std::ofstream corrupt(f.root / "run" / "a1.f32", std::ios::binary);
    corrupt << "x";
  }
  const auto broken =
      f.WaitFinished(f.Start(Body("run", records, "collect", true)));
  Expect(broken.member_str("status") == "failed" &&
             broken.member_str("error").find("integrity") != std::string::npos,
         "a corrupt saved record fails the job");
}

void TestCancel() {
  Fixture f;
  f.runner->Hold(true);
  const auto running =
      f.Start(Body("one", {CollectRecord("a1"), CollectRecord("a2")}));
  const auto queued = f.Start(Body("two", {CollectRecord("b1")}));
  f.runner->WaitRunning();
  const auto dequeued = f.service->Delete(queued);
  Expect(dequeued.member_str("status") == "cancelled",
         "a queued job is cancelled at once");
  const auto cancelling = f.service->Delete(running);
  Expect(cancelling.find("cancel_requested")->as_bool(),
         "a running job is asked to stop");
  const auto job = f.WaitFinished(running);
  Expect(
      job.member_str("status") == "cancelled" && job.find("results")->empty(),
      "the running job stops at its record");
  Expect(f.service->Get(queued, 0).member_str("status") == "cancelled",
         "the cancelled queued job never runs");
  Expect(!fs::exists(f.root / "two"), "nothing was written for it");
  // A cancelled job is finished: DELETE forgets it.
  Expect(f.service->Delete(running).find("deleted")->as_bool(),
         "a cancelled job is forgotten on DELETE");
}

void TestJobCap() {
  Fixture f(2);
  const auto a = f.Start(Body("a", {CollectRecord("a1")}));
  f.WaitFinished(a);
  f.runner->Hold(true);
  const auto b = f.Start(Body("b", {CollectRecord("b1")}));
  const auto c = f.Start(Body("c", {CollectRecord("c1")}));
  bool forgotten = false;
  try {
    (void)f.service->Get(a, 0);
  } catch (const CollectionJobError&) {
    forgotten = true;
  }
  Expect(forgotten, "the oldest finished job makes room");
  int status = 0;
  Expect(ErrorCode(f, Body("d", {CollectRecord("d1")}), &status) ==
                 "too_many_jobs" &&
             status == 429,
         "unfinished jobs are never dropped");
  f.runner->Hold(false);
  f.WaitFinished(b);
  f.WaitFinished(c);
}
}  // namespace

int main() {
  TestValidation();
  TestLifecycleAndLayout();
  TestAnswersReceipts();
  TestSegmentsReceipts();
  TestResume();
  TestCancel();
  TestJobCap();
  std::cout << "collection jobs: ok\n";
  return 0;
}
