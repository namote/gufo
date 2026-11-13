# Brief B: collect activations inside the server, as polled jobs

Read `plan/research/STEERING-API-DESIGN.md` (sections B and "Jobs"), `docs/SERVER.md` (abliteration, runtime profile updates from bc35eaf3, scoring), `tools/qwen38_flash_next/abliteration/collect.cpp` (the reference: modes `collect`, `answers`, `segments`, the readout, framing, and the manifest/receipt format), and the repo's CLAUDE.md/AGENTS.md. Flash-Next only.

## Build

1. **Start a job:** `POST /v1/abliteration/jobs` with `{"operation": "collect", "mode": "collect|answers|segments", "records": [...], "name": "<output dir name>"}`.
   - Records use the same JSON as collect.cpp's input lines for that mode.
   - Validate all records first (as collect.cpp does) and reject the job with 400 if any is bad.
   - Reply at once: `{"job": "<key>", "status": "queued"}`. The key must be unguessable (random, at least 128 bits).
2. **Poll:** `GET /v1/abliteration/jobs/{key}[?since=n]` returns:
   - `status` (queued, running, done, failed, cancelled), `progress` {`done`, `total`, `step`}, `created_at`/`updated_at`, `error`;
   - `results`: the record entries finished after sequence `n` (each: seq, id, group, pair, rows read, `activation_sha256`, file name).
   
   `GET /v1/abliteration/jobs` lists jobs. `DELETE /v1/abliteration/jobs/{key}` cancels a queued or running job; on a finished job it forgets it (files stay).
3. **Output:** write exactly collect.cpp's collection layout (`manifest.json`, per-record `.json` receipt and `.f32` activation) under `<--collect-dir>/<name>/`.
   - `fit_steering.py` and `prepare.py` must read it unchanged.
   - `name` follows the same rule as profile names. Refuse to overwrite an existing directory unless the job resumes it: reuse completed receipts like collect.cpp does.
4. **Numerics:** the same readout as collect.cpp (post-layer residual, HC mean, FP32; the selected rows per mode). The acceptance test is equality with collect.cpp on the same records, within the tolerance batching allows; state what you can and cannot guarantee.
   - Abliteration and logit bias must be OFF for collection passes.
   - Collection must never write into the prompt/continuation caches (RAM or disk).
5. **Scheduling:** collection uses the loaded model through the scheduler as low-priority work.
   - Chat requests always go first; a collection pass runs only when a session is free, one record at a time, between steps (see `RunBetweenSteps` from bc35eaf3 and how scoring requests are scheduled).
   - Record length is capped (as collect.cpp: 2048/8192/24576 tokens by mode); reject longer records at validation.
   - One running collection job at a time; the others queue.
6. **Gating:** enabled only with a server option naming the output root (`--collect-dir DIR`; one canonical name). Without it these paths return 404. With `--api-key` they need the key.
7. **Lifetime:** job state in memory with a cap (e.g. the last 64 jobs; document it); files on disk stay.
8. **Tests (CPU):** gating and auth, validation errors per mode, job lifecycle (queued, running, done, cancel), `since` paging, the output layout matching collect.cpp's manifest/receipt format, resume, and the low-priority interleaving with a fake runner if the scheduler tests allow. GPU tests may be built but not run.
9. **Docs:** docs/SERVER.md section for the jobs API.

## Build and test

`nice -n 19 python3 -I tools/docker/pinned_rocm101.py run --build-dir build/lbias --source . -- sh -c 'cmake --build /build/obj -j2 --target <targets> && cd /build/obj && ctest -R "^(<tests>)$" --output-on-failure'`. The build container is allowed; keep -j2 (a model server is running). clang-format: `/nix/store/g2ycks7l2dz7rr4l6cyxf4in7qrfjk8j-clang-tools-21.1.8/bin/clang-format` inside that container.

## Do not

- Use the GPU, start/stop `llm` or any `gufo-*` container, push, or use `pgrep -f`/`pkill -f`.
- Modify `~/.local/share/gufo/`, `build/b1/`, `tools/exl3_quant/hc/`, `plan/exl3/HC-Q8-CALIBRATION.md`.

## Commit and review

Single-line Conventional Commit messages ending with:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016Kh8svoeuCWVkepKUHx2XF
```
Stage only your files. Before finishing, have a second reviewer agent review the diff, fix what it finds, and have it re-review.

## Report

Endpoints and options, how scheduling keeps chat first, numeric equivalence with collect.cpp (what is guaranteed), files changed, tests and results, review findings and fixes, anything left undone.
