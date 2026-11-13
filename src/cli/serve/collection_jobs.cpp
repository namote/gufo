#include "src/cli/serve/collection_jobs.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <utility>

#include "src/cli/serve/logging.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/platform/os_info.hpp"
#include "src/models/qwen/abliteration.hpp"

namespace gufo::server {
namespace {
namespace fs = std::filesystem;
namespace qfn = models::qwen38_flash_next;

constexpr std::string_view kJobsPath = "/v1/abliteration/jobs";

std::int64_t NowSeconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string RandomKey() {
  std::array<std::uint8_t, 16> bytes{};
  if (!platform::FillRandomBytes(bytes))
    throw CollectionJobError(500, "internal_error",
                             "no random bytes for a job key");
  static constexpr char kHex[] = "0123456789abcdef";
  std::string key;
  for (const auto byte : bytes) {
    key.push_back(kHex[byte >> 4]);
    key.push_back(kHex[byte & 15]);
  }
  return key;
}

// collect.cpp's file writes: a temporary beside the target, then a rename.
std::string Read(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open " + path.string());
  return {std::istreambuf_iterator<char>(input), {}};
}
void Save(const fs::path& path, const json::Value& value) {
  const fs::path temporary = path.string() + ".tmp";
  {
    std::ofstream out(temporary);
    out << value.dump() << '\n';
    if (!out)
      throw std::runtime_error("write failed");
  }
  fs::rename(temporary, path);
}
void Floats(const fs::path& path, std::span<const float> values) {
  const fs::path temporary = path.string() + ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size_bytes()));
    if (!out)
      throw std::runtime_error("activation write failed");
  }
  fs::rename(temporary, path);
}

bool ValidCollectionName(std::string_view name) {
  return models::qwen::ValidAbliterationProfileName(name) && name != "." &&
         name != "..";
}

enum class Status : std::uint8_t {
  kQueued,
  kRunning,
  kDone,
  kFailed,
  kCancelled
};

std::string_view StatusName(Status status) {
  switch (status) {
    case Status::kQueued:
      return "queued";
    case Status::kRunning:
      return "running";
    case Status::kDone:
      return "done";
    case Status::kFailed:
      return "failed";
    case Status::kCancelled:
      return "cancelled";
  }
  return "failed";
}

bool Finished(Status status) {
  return status == Status::kDone || status == Status::kFailed ||
         status == Status::kCancelled;
}

struct Job {
  std::string key;
  qfn::CollectionMode mode{};
  std::string name;
  bool resume{false};
  /// The records and the manifest, dropped once the job has finished.
  json::Value records;
  json::Value manifest;
  std::size_t total{0};
  std::vector<std::string> deviations;
  // Guarded by Impl::mutex.
  Status status{Status::kQueued};
  std::size_t done{0};
  std::string step{"queued"};
  std::int64_t created_at{0};
  std::int64_t started_at{0};
  std::int64_t updated_at{0};
  std::string error;
  std::vector<json::Value> results;
  std::atomic<bool> cancel{false};
};
}  // namespace

struct CollectionJobService::Impl {
  CollectionJobOptions options;
  mutable std::mutex mutex;
  std::condition_variable wake;
  /// Creation order.
  std::vector<std::shared_ptr<Job>> jobs;
  std::deque<std::shared_ptr<Job>> queue;
  bool stopping{false};
  /// Hashing the executable takes a moment: once, on the first job, under
  /// its own lock so polls never wait for it.
  std::mutex identity_mutex;
  std::optional<CollectionModelIdentity> identity;
  std::thread worker;

  CollectionModelIdentity Identity() {
    const std::lock_guard<std::mutex> lock(identity_mutex);
    if (!identity)
      identity = options.runner->Identity();
    return *identity;
  }

  std::shared_ptr<Job> Find(std::string_view key) const {
    for (const auto& job : jobs)
      if (job->key == key)
        return job;
    return nullptr;
  }

  void Update(Job& job, auto&& change) {
    const std::lock_guard<std::mutex> lock(mutex);
    change();
    job.updated_at = NowSeconds();
  }

  json::Value Snapshot(const Job& job, std::optional<std::size_t> since) const {
    json::Value root = json::Value::object();
    root["job"] = job.key;
    root["object"] = "abliteration.job";
    root["operation"] = "collect";
    root["mode"] = std::string(qfn::CollectionModeName(job.mode));
    root["name"] = job.name;
    root["status"] = std::string(StatusName(job.status));
    json::Value progress = json::Value::object();
    progress["done"] = job.done;
    progress["total"] = job.total;
    progress["step"] = job.step;
    root["progress"] = std::move(progress);
    root["created_at"] = static_cast<long long>(job.created_at);
    root["started_at"] =
        job.started_at ? json::Value(static_cast<long long>(job.started_at))
                       : json::Value(nullptr);
    root["updated_at"] = static_cast<long long>(job.updated_at);
    root["error"] =
        job.error.empty() ? json::Value(nullptr) : json::Value(job.error);
    root["cancel_requested"] = job.cancel.load();
    json::Value deviations = json::Value::array();
    for (const auto& deviation : job.deviations)
      deviations.push_back(deviation);
    root["numerics_deviations"] = std::move(deviations);
    if (since) {
      json::Value results = json::Value::array();
      for (std::size_t i = std::min(*since, job.results.size());
           i < job.results.size(); ++i)
        results.push_back(job.results[i]);
      root["results"] = std::move(results);
      root["last_seq"] = job.results.size();
    }
    return root;
  }

  void Work() {
    for (;;) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock, [&] { return stopping || !queue.empty(); });
        if (stopping)
          return;
        job = std::move(queue.front());
        queue.pop_front();
        job->status = Status::kRunning;
        job->step = "starting";
        job->started_at = job->updated_at = NowSeconds();
      }
      Logger::Info(
          "collect",
          "event=collection_job_start job=" + job->key + " name=" + job->name +
              " mode=" + std::string(qfn::CollectionModeName(job->mode)) +
              " records=" + std::to_string(job->records.size()));
      Status outcome = Status::kDone;
      std::string error;
      try {
        outcome = RunJob(*job);
      } catch (const std::exception& e) {
        outcome = Status::kFailed;
        error = e.what();
      }
      try {
        options.runner->Release();
      } catch (const std::exception& e) {
        Logger::Warn("collect", "event=collection_release_failed error=" +
                                    std::string(e.what()));
      }
      Update(*job, [&] {
        job->status = outcome;
        job->error = error;
        job->step = std::string(StatusName(outcome));
        job->records = json::Value();
        job->manifest = json::Value();
      });
      Logger::Info("collect",
                   "event=collection_job_end job=" + job->key +
                       " status=" + std::string(StatusName(outcome)) +
                       (error.empty() ? "" : " error=" + error));
    }
  }

  Status RunJob(Job& job) {
    const fs::path output = options.root / job.name;
    fs::create_directories(output);
    if (fs::exists(output / "manifest.json")) {
      if (json::parse(Read(output / "manifest.json")).dump() !=
          job.manifest.dump())
        throw std::runtime_error(
            "existing collection identity differs; use a new output "
            "directory");
    } else {
      Save(output / "manifest.json", job.manifest);
    }
    const bool answers = job.mode != qfn::CollectionMode::kCollect;
    const bool segments = job.mode == qfn::CollectionMode::kSegments;
    const auto total = job.records.size();
    std::size_t index = 0;
    for (const auto& record : job.records.items()) {
      if (job.cancel.load() || Stopping())
        return Status::kCancelled;
      ++index;
      const auto id = record.member_str("id");
      const auto receipt = output / (id + ".json");
      const auto data = output / (id + ".f32");
      json::Value entry = json::Value::object();
      entry["id"] = id;
      entry["group"] = record.member_str("group");
      entry["pair"] = answers ? json::Value(record.member_str("pair"))
                              : json::Value(nullptr);
      entry["file"] = id + ".f32";
      if (fs::exists(receipt)) {
        const auto saved = json::parse(Read(receipt));
        if (!fs::exists(data) || saved.member_str("activation_sha256") !=
                                     crypto::Sha256FileHex(data))
          throw std::runtime_error("saved prompt data integrity failure: " +
                                   id);
        entry["rows"] = segments  ? saved.member_size("selected_rows")
                        : answers ? saved.member_size("answer_tokens")
                                  : std::size_t{1};
        entry["activation_sha256"] = saved.member_str("activation_sha256");
        entry["resumed"] = true;
        AddResult(job, std::move(entry));
        continue;
      }
      const std::string where = "record " + std::to_string(index) + "/" +
                                std::to_string(total) + " " + id;
      Update(job, [&] { job.step = where; });
      const auto plan = options.runner->Plan(job.mode, record);
      const auto activations = options.runner->Run(
          job.mode, plan, [&] { return job.cancel.load() || Stopping(); },
          [&](std::string step) {
            Update(job, [&] { job.step = where + ": " + step; });
          });
      if (!activations)
        return Status::kCancelled;
      if (activations->size() != job.manifest.member_size("layers") *
                                     job.manifest.member_size("hidden"))
        throw std::runtime_error("collector readout has the wrong size: " + id);
      Floats(data, *activations);
      auto result = json::Value::object();
      result["id"] = id;
      result["prompt_tokens"] = plan.prompt.size();
      if (answers) {
        result["answer_tokens"] = plan.answer.size();
        if (segments)
          result["selected_rows"] =
              static_cast<unsigned long long>(plan.selected_rows);
        result["pair"] = record.member_str("pair");
        result["group"] = record.member_str("group");
      }
      const auto sha = crypto::Sha256FileHex(data);
      result["activation_sha256"] = sha;
      Save(receipt, result);
      entry["rows"] = segments  ? plan.selected_rows
                      : answers ? plan.answer.size()
                                : std::size_t{1};
      entry["activation_sha256"] = sha;
      entry["resumed"] = false;
      AddResult(job, std::move(entry));
    }
    return Status::kDone;
  }

  void AddResult(Job& job, json::Value entry) {
    Update(job, [&] {
      entry["seq"] = job.results.size() + 1;
      job.results.push_back(std::move(entry));
      ++job.done;
    });
  }

  bool Stopping() const {
    const std::lock_guard<std::mutex> lock(mutex);
    return stopping;
  }
};

CollectionJobService::CollectionJobService(CollectionJobOptions options)
    : impl_(std::make_unique<Impl>()) {
  if (!options.runner)
    throw std::invalid_argument("collection jobs need a runner");
  if (!options.key_factory)
    options.key_factory = RandomKey;
  impl_->options = std::move(options);
  impl_->worker = std::thread([impl = impl_.get()] { impl->Work(); });
}

CollectionJobService::~CollectionJobService() {
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->stopping = true;
    for (const auto& job : impl_->jobs)
      job->cancel = true;
  }
  impl_->wake.notify_all();
  impl_->worker.join();
}

json::Value CollectionJobService::Create(std::string_view body) {
  const auto invalid = [](std::string code, const std::string& message) {
    return CollectionJobError(400, std::move(code), message);
  };
  json::Value request;
  try {
    request = json::parse(body);
  } catch (const std::exception& e) {
    throw invalid("invalid_json", std::string("invalid JSON: ") + e.what());
  }
  if (!request.is_object())
    throw invalid("invalid_request", "the body is a JSON object");
  if (request.member_str("operation") != "collect")
    throw invalid("unsupported_operation", "'operation' must be \"collect\"");
  const auto mode = qfn::ParseCollectionMode(request.member_str("mode"));
  if (!mode)
    throw invalid("invalid_mode",
                  "'mode' must be \"collect\", \"answers\" or \"segments\"");
  const auto name = request.member_str("name");
  if (!ValidCollectionName(name))
    throw invalid("invalid_name",
                  "'name' is 1-64 letters, digits, '_', '-' or '.', not '.' "
                  "or '..'");
  bool resume = false;
  if (const auto* value = request.find("resume")) {
    if (!value->is_bool())
      throw invalid("invalid_request", "'resume' must be a boolean");
    resume = value->as_bool();
  }
  const auto* records = request.find("records");
  if (records == nullptr || !records->is_array() || records->empty())
    throw invalid("invalid_records", "'records' must be a non-empty array");
  std::string split;
  std::set<std::string> ids;
  std::string data;
  std::size_t index = 0;
  for (const auto& record : records->items()) {
    const auto where = "record " + std::to_string(index++);
    try {
      qfn::ValidateCollectionRecord(*mode, record, &split);
      const auto id = record.member_str("id");
      if (!ids.insert(id).second)
        throw std::invalid_argument("duplicate record ID " + id);
      (void)impl_->options.runner->Plan(*mode, record);
    } catch (const std::exception& e) {
      throw invalid("invalid_record", where + ": " + e.what());
    }
    data += record.dump();
    data += '\n';
  }
  const auto identity = impl_->Identity();
  // collect.cpp's manifest, member for member. data_sha256 hashes the
  // records as compact JSON lines (the tool hashes its input file).
  auto manifest = json::Value::object();
  manifest["schema"] = "gufo-abliteration-capture-v1";
  manifest["model_identity"] = identity.model_identity;
  manifest["device_code"] = identity.device_code;
  manifest["executable_sha256"] = identity.executable_sha256;
  manifest["context"] =
      static_cast<unsigned long long>(qfn::CollectionContext(*mode));
  manifest["max_new_tokens"] = static_cast<unsigned long long>(0);
  manifest["template"] = std::string(qfn::CollectionTemplate(*mode));
  manifest["readout"] = std::string(qfn::CollectionReadoutName(*mode));
  manifest["data_sha256"] = crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(data.data()), data.size()));
  manifest["hidden"] = static_cast<unsigned long long>(identity.hidden);
  manifest["layers"] = static_cast<unsigned long long>(identity.layers);
  manifest["split"] = split;
  manifest["mode"] = std::string(qfn::CollectionModeName(*mode));
  manifest["profile"] = "off";
  manifest["records"] = *records;

  auto job = std::make_shared<Job>();
  job->mode = *mode;
  job->name = name;
  job->resume = resume;
  job->records = *records;
  job->manifest = std::move(manifest);
  job->total = records->size();
  job->deviations = identity.deviations;
  // A shorter session refuses records collect.cpp would take; the records
  // it accepts read out the same.
  if (identity.max_context != 0 &&
      identity.max_context < qfn::CollectionContext(*mode))
    job->deviations.push_back(
        "context " + std::to_string(identity.max_context) + " below the " +
        std::string(qfn::CollectionModeName(*mode)) + " mode's " +
        std::to_string(qfn::CollectionContext(*mode)));
  job->created_at = job->updated_at = NowSeconds();
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->stopping)
      throw CollectionJobError(503, "server_stopping",
                               "the server is stopping");
    for (const auto& other : impl_->jobs)
      if (other->name == name && !Finished(other->status))
        throw CollectionJobError(
            409, "collection_busy",
            "another queued or running job writes '" + name + "'");
    const fs::path output = impl_->options.root / name;
    std::error_code error;
    if (fs::exists(output, error)) {
      if (!resume)
        throw CollectionJobError(
            409, "collection_exists",
            "'" + name + "' exists; pass \"resume\": true to continue it");
      if (fs::exists(output / "manifest.json", error)) {
        bool same = false;
        try {
          same = json::parse(Read(output / "manifest.json")).dump() ==
                 job->manifest.dump();
        } catch (const std::exception&) {
        }
        if (!same)
          throw CollectionJobError(
              409, "collection_differs",
              "existing collection identity differs; use a new output "
              "directory");
      }
    }
    if (impl_->jobs.size() >= impl_->options.max_jobs) {
      const auto oldest = std::ranges::find_if(
          impl_->jobs, [](const auto& j) { return Finished(j->status); });
      if (oldest == impl_->jobs.end())
        throw CollectionJobError(429, "too_many_jobs",
                                 "at most " +
                                     std::to_string(impl_->options.max_jobs) +
                                     " jobs are kept and none has finished");
      impl_->jobs.erase(oldest);
    }
    for (;;) {
      job->key = impl_->options.key_factory();
      if (!impl_->Find(job->key))
        break;
    }
    impl_->jobs.push_back(job);
    impl_->queue.push_back(job);
  }
  impl_->wake.notify_all();
  json::Value reply = json::Value::object();
  reply["job"] = job->key;
  reply["status"] = "queued";
  return reply;
}

json::Value CollectionJobService::Get(std::string_view key,
                                      std::size_t since) const {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto job = impl_->Find(key);
  if (!job)
    throw CollectionJobError(404, "unknown_job", "no such collection job");
  return impl_->Snapshot(*job, since);
}

json::Value CollectionJobService::List() const {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  json::Value data = json::Value::array();
  for (auto it = impl_->jobs.rbegin(); it != impl_->jobs.rend(); ++it)
    data.push_back(impl_->Snapshot(**it, std::nullopt));
  json::Value root = json::Value::object();
  root["object"] = "list";
  root["data"] = std::move(data);
  return root;
}

json::Value CollectionJobService::Delete(std::string_view key) {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto job = impl_->Find(key);
  if (!job)
    throw CollectionJobError(404, "unknown_job", "no such collection job");
  if (Finished(job->status)) {
    std::erase(impl_->jobs, job);
    json::Value reply = json::Value::object();
    reply["job"] = job->key;
    reply["deleted"] = true;
    return reply;
  }
  job->cancel = true;
  if (job->status == Status::kQueued) {
    std::erase(impl_->queue, job);
    job->status = Status::kCancelled;
    job->step = "cancelled";
    job->records = json::Value();
    job->manifest = json::Value();
  }
  job->updated_at = NowSeconds();
  return impl_->Snapshot(*job, std::nullopt);
}

bool IsCollectionJobsPath(std::string_view path) noexcept {
  return path == kJobsPath ||
         (path.starts_with(kJobsPath) && path.size() > kJobsPath.size() &&
          path[kJobsPath.size()] == '/');
}

namespace {
HttpResponse JobError(int status, std::string_view code,
                      const std::string& message) {
  const char* reason = status == 400   ? "Bad Request"
                       : status == 404 ? "Not Found"
                       : status == 405 ? "Method Not Allowed"
                       : status == 409 ? "Conflict"
                       : status == 429 ? "Too Many Requests"
                       : status == 503 ? "Service Unavailable"
                                       : "Internal Server Error";
  json::Value error = json::Value::object();
  error["message"] = message;
  error["type"] =
      status >= 500 ? "server_error" : std::string("invalid_request_error");
  error["code"] = std::string(code);
  json::Value root = json::Value::object();
  root["error"] = std::move(error);
  return {.status = status,
          .reason = reason,
          .body = root.dump(),
          .headers = {{"Content-Type", "application/json"}}};
}

HttpResponse JobReply(const json::Value& body) {
  return {.status = 200,
          .reason = "OK",
          .body = body.dump(),
          .headers = {{"Content-Type", "application/json"}}};
}
}  // namespace

HttpResponse HandleCollectionJobsRequest(const HttpRequest& request,
                                         CollectionJobService& service) {
  try {
    if (request.path == kJobsPath) {
      if (request.method == "POST")
        return JobReply(service.Create(request.body));
      if (request.method == "GET")
        return JobReply(service.List());
      return JobError(405, "method_not_allowed",
                      "use GET or POST on " + std::string(kJobsPath));
    }
    const auto key =
        std::string_view(request.path).substr(kJobsPath.size() + 1);
    if (key.empty() || key.find('/') != std::string_view::npos)
      return JobError(404, "unknown_job", "no such collection job");
    if (request.method == "DELETE")
      return JobReply(service.Delete(key));
    if (request.method != "GET")
      return JobError(405, "method_not_allowed",
                      "use GET or DELETE on a collection job");
    std::size_t since = 0;
    if (const auto text = request.query_param("since"); !text.empty()) {
      const auto [end, error] =
          std::from_chars(text.data(), text.data() + text.size(), since);
      if (error != std::errc{} || end != text.data() + text.size())
        return JobError(400, "invalid_since",
                        "'since' must be a non-negative integer");
    }
    return JobReply(service.Get(key, since));
  } catch (const CollectionJobError& e) {
    return JobError(e.status(), e.code(), e.what());
  } catch (const std::exception& e) {
    return JobError(500, "internal_error", e.what());
  }
}

}  // namespace gufo::server
