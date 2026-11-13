#ifndef GUFO_SERVER_COLLECTION_JOBS_HPP_
#define GUFO_SERVER_COLLECTION_JOBS_HPP_

// In-server activation collection (`gufo serve --collect-dir DIR`):
// `POST /v1/abliteration/jobs` starts a keyed job that runs the records of
// tools/qwen38_flash_next/abliteration/collect.cpp's collect, answers or
// segments mode on the loaded model as low-priority scheduler work and writes
// that tool's collection layout (manifest.json, <id>.json, <id>.f32) under
// DIR/<name>; `GET /v1/abliteration/jobs/{key}?since=n` polls it.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen38_flash_next/activation_collection.hpp"

namespace gufo::server {

/// Jobs kept in memory, finished ones included; the oldest finished job is
/// forgotten to admit a new one. Files on disk stay.
inline constexpr std::size_t kMaxCollectionJobs = 64;
/// The largest job body an authorized client may send.
inline constexpr std::size_t kMaxCollectionJobBytes = std::size_t{64} << 20;

/// What the manifest records about the model and binary that collected.
struct CollectionModelIdentity {
  std::string model_identity;
  std::string device_code;
  std::string executable_sha256;
  std::uint32_t hidden{0};
  std::uint32_t layers{0};
  /// The longest session the server can make (its context); zero leaves
  /// every mode's collect.cpp context.
  std::uint32_t max_context{0};
  /// Server settings whose arithmetic differs from collect.cpp's (KVarN K/V,
  /// a non-default HC mode, YaRN, a short context); empty when none does.
  std::vector<std::string> deviations;
};

/// The loaded model's side of a collection job (Flash-Next's runner in
/// inference_backend.cpp; tests fake it).
class CollectionRunner {
public:
  virtual ~CollectionRunner() = default;
  [[nodiscard]] virtual CollectionModelIdentity Identity() const = 0;
  /// Frames and tokenizes a validated record
  /// (models::qwen38_flash_next::PlanCollectionRecord). Throws
  /// std::invalid_argument when the record cannot run.
  [[nodiscard]] virtual models::qwen38_flash_next::CollectionPlan Plan(
      models::qwen38_flash_next::CollectionMode mode,
      const json::Value& record) const = 0;
  /// Runs one record's passes as low-priority work and returns its
  /// [layers][hidden] readout; nullopt once `cancelled` returns true.
  /// `step` reports what the record waits for. Throws on failure.
  [[nodiscard]] virtual std::optional<std::vector<float>> Run(
      models::qwen38_flash_next::CollectionMode mode,
      const models::qwen38_flash_next::CollectionPlan& plan,
      const std::function<bool()>& cancelled,
      const std::function<void(std::string)>& step) = 0;
  /// Frees what a job held (its session) after its last record.
  virtual void Release() {}
};

/// A refused job request: the HTTP status and error code.
class CollectionJobError : public std::runtime_error {
public:
  CollectionJobError(int status, std::string code, const std::string& message)
      : std::runtime_error(message), status_(status), code_(std::move(code)) {}
  [[nodiscard]] int status() const noexcept { return status_; }
  [[nodiscard]] const std::string& code() const noexcept { return code_; }

private:
  int status_;
  std::string code_;
};

struct CollectionJobOptions {
  /// `--collect-dir`: collections are written to root/<name>.
  std::filesystem::path root;
  std::shared_ptr<CollectionRunner> runner;
  std::size_t max_jobs{kMaxCollectionJobs};
  /// Tests: deterministic keys. Default: 128 random bits as hex.
  std::function<std::string()> key_factory;
};

/// Keyed collection jobs, run one at a time on a worker thread in arrival
/// order. Thread-safe.
class CollectionJobService {
public:
  explicit CollectionJobService(CollectionJobOptions options);
  ~CollectionJobService();
  CollectionJobService(const CollectionJobService&) = delete;
  CollectionJobService& operator=(const CollectionJobService&) = delete;

  /// Validates a `POST /v1/abliteration/jobs` body and queues the job:
  /// {"job": key, "status": "queued"}. Throws CollectionJobError.
  [[nodiscard]] json::Value Create(std::string_view body);
  /// The job with the results after sequence `since`. Throws
  /// CollectionJobError (404) for an unknown key.
  [[nodiscard]] json::Value Get(std::string_view key, std::size_t since) const;
  /// Every kept job without its results, newest first.
  [[nodiscard]] json::Value List() const;
  /// Cancels a queued or running job (its snapshot), or forgets a finished
  /// one ({"job": key, "deleted": true}). Throws CollectionJobError (404).
  [[nodiscard]] json::Value Delete(std::string_view key);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] bool IsCollectionJobsPath(std::string_view path) noexcept;
[[nodiscard]] HttpResponse HandleCollectionJobsRequest(
    const HttpRequest& request, CollectionJobService& service);

}  // namespace gufo::server

#endif  // GUFO_SERVER_COLLECTION_JOBS_HPP_
