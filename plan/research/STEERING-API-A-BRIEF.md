# Brief A: load abliteration and logit-bias profiles at runtime

Read `plan/research/STEERING-API-DESIGN.md` (section A), `docs/SERVER.md` (abliteration, several profiles, additive profiles, dynamic abliteration, logit-bias profiles), and the repo's CLAUDE.md/AGENTS.md.

## Build

1. `POST /v1/abliteration/profiles/{name}` with a profile JSON body: validate exactly as the startup loader does for a file in `--abliteration-profiles DIR` (including triggers, additive mode, model identity binding), then add or replace the profile under `name`. `DELETE /v1/abliteration/profiles/{name}` removes it. Respond with the profile summary that `/props` lists.
2. The same for logit-bias profiles: `POST/DELETE /v1/logit-bias/profiles/{name}` (`src/core/logit_bias.*`; the catalog is rebuilt; the combination cache must not return arrays from the old profile).
3. **Off by default.** Enable with one server option (one canonical name, e.g. `--allow-profile-updates`). When the server has an API key, these endpoints require it. Without the option they return 404 or 403 consistently with similar endpoints.
4. **Consistency.** A request snapshots the profile set when admitted; a replace or delete never changes a running request. Cache identity: a replaced profile must not reuse state cached under the old one (check how settings identities are derived; include whatever makes them distinct). Detectors: adding or removing a trigger profile updates the set scored by later passes; a request in flight keeps its own. If a session's device state holds profile data (strength arrays, direction rows), make sure the update path reaches it safely between steps.
5. **Limits.** Cap the body size and the number of profiles (document both). Persisting uploads to disk is not required; document that they live until restart.
6. **Tests (CPU):** endpoint gating, validation errors, add, replace, delete, `/props` reflects changes, a request snapshot not affected by a concurrent replace (where testable), logit-bias catalog replacement and cache invalidation. GPU tests may be built, not run.
7. **Docs:** docs/SERVER.md sections for both endpoints.

## Build and test

`nice -n 19 python3 -I tools/docker/pinned_rocm101.py run --build-dir build/lbias --source . -- sh -c 'cmake --build /build/obj -j2 --target <targets> && cd /build/obj && ctest -R "^(<tests>)$" --output-on-failure'`. The build container is allowed. Memory is tight (a model server is running): keep -j2. Run clang-format via the container's nix store if `nix` is missing on the host, as the previous agent did.

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

Endpoints and option names, how snapshots and cache identities stay consistent, files changed, tests run and results, review findings and fixes, and anything left undone.
