# Image requests refused at 4 sessions (brief, 2026-11-17)

## Symptom (production runtime-20261117a, 1M context, 4 sessions, K4V4, MTP 7)

Production went from 2 to 4 sessions today. Every request with this image now
fails at once (59 ms), also when no other slot is busy:

```
request=r23 event=received body_bytes=257606
request=r23 event=server_exception reason=image request exceeds available memory headroom
request=r23 event=completed status=500 duration_ms=59.1 rss_mib=2367 host_available_mib=8454
memory ... deferred_scratch_budget_bytes=354443264 pending_image_and_cache_budget_bytes=0 device_free_bytes=6311145472
```
Log: the newest `~/.local/share/gufo/logs/*/server.log` (read only). At 2
sessions device free was ~18 GB higher and images worked.

## Code

`src/cli/serve/inference_backend.cpp` `ImageClaim` (~3047) and `ImageHeadroomFor`
(~3068): headroom = min(host available - 4 GiB - deferred, hipMemGetInfo free
- 4 GiB - deferred), reusing `kHostSnapshotHeadroomBytes` (4 GiB,
`text_model_runner.hpp:121`). `RequestMemoryBudget` (`request_memory_budget.hpp`)
throws when one claim alone exceeds headroom. Estimate:
`models::qwen::vision::EstimateImageMemory`.

## Goal

Image requests must work in production at 4 sessions, generically (all
vision models), without risking an out-of-memory crash. Strix Halo is
unified memory (121 GiB total, no disk swap, 8 GiB zram).

1. Log the claim (persistent, scratch), the headroom and its inputs on every
   refusal, so a refusal is explainable.
2. Check whether `EstimateImageMemory` matches what the encoder really
   allocates (compute it for typical sizes, e.g. 1600x900 and 4K screenshots,
   and compare with the allocation path); fix if it overestimates.
3. Revisit the reserve: is subtracting the full host-snapshot headroom (4 GiB)
   from device free right for a short-lived encoder scratch on unified memory?
   Pick a justified reserve.
4. Bound the encoder scratch so a large image never needs more than a fixed
   budget: encode in tiles / chunks of patches (or windowed blocks, whatever
   the encoder's attention structure allows exactly), so with little memory an
   image is slower, never refused. Output must stay numerically equivalent
   within the existing vision tests' tolerance.
5. If an image still cannot fit, wait (queue) for memory instead of throwing
   when other requests hold claims; only throw when even an idle server could
   never fit it, with a clear message.

## Rules

- Do not touch production (`~/.local/share/gufo/`), the running server, or
  `~/projects/_ai.tools`. Never modify `tools/exl3_quant/hc/`.
- No GPU/model/server runs. Build affected targets and run CPU/unit tests;
  add tests for the claim/headroom logic and for chunked-encoding bookkeeping
  if CPU-testable. Say which GPU check is needed to validate numerics.
- Keep code style; run the format check. Do not commit. Report in English:
  numbers for (1)-(3), the change, tests and results, the GPU check needed,
  open risks.
