# V1 brief: small upstream bug fixes (for the implementing agent)

Read `plan/upstream-sync2/PLAN.md` (V1 table) and the matching sections of
`plan/upstream-sync2/survey-upstream.md` first. Read `docs/GFX1151-KNOWHOW.md` only as needed.

## Rules

- **No GPU, model or server runs. No production changes.** Do not touch `~/.local/share/gufo`, `llm`, or docker containers other than the build.
- Build only with `build/b1/build.sh <targets>`. Run CPU tests with `build/b1/build.sh --run /build/obj/<path-to-test-binary>`. The source tree is mounted at `/build/source`. Format with `build/b1/fmt.sh -i <files>`, then check with `build/b1/checkfmt.sh`.
- Never run anything from `~/projects/_ai.tools`, including `ai-tools/*` docker images.
- Do not touch `tools/exl3_quant/hc/` or `plan/exl3/HC-Q8-CALIBRATION.md`. Leave the untracked `plan/research/*.md` alone.
- `git fetch upstream` is already done. Use `git show <hash>` for the upstream diffs.
- **One commit per item.**
  - Message: single-line Conventional Commit with the upstream PR number, e.g. `fix(qwen38-flash-next): keep compacted MTP inject rows out of the norm buffer (upstream #1230)`.
  - Clean cherry-picks use `git cherry-pick -x`. Then amend to add the trailers if a trailer block is needed.
  - Every commit ends with these two trailer lines:
    ```
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
    Claude-Session: https://claude.ai/code/session_016Kh8svPWb3TTEKxGwdHJTN
    ```
  - Do not push.
- Keep our fork's structure. Our Flash-Next runtime lives in `src/models/qwen38_flash_next/runtime/`; upstream's `kernels/rocm/batch.cpp` and friends map onto it. Port by content, not by file.

## Items (in order)

1. **V1-1 #1230 (800ce328):** hand port into `runtime/executor_batch.cpp` (the compact tail sets `tail.inject = base.mtp_h`). Build `gufo` and the Flash-Next session/mtp CPU-buildable tests that compile.
2. **V1-3 #1146 (4e95aecb):** cherry-pick. Run the json and openai_chat tests.
3. **V1-5 #1168 (53193819):** cherry-pick; hand-merge the fixture if needed. Run json_constraint tests, including llama parity if present.
4. **V1-6 #1102 (72442dce):** rewrite the path `src/families/qwen/text/` to `src/models/qwen/`. Run the chat_template tests.
5. **V1-7 #912 (49b5df18):** hand port into `mtp_policy.hpp` and `engine.cpp`. Run `qwen38_flash_next_mtp_sampling_test`. It must still pass with the 2026-11-20 cost tables committed in `0add8030`.
6. **V1-8 #1104 (fe2694bc):** cherry-pick. Run the cache retention tests.
7. **V1-9 #1181 (77621b27):** hand port into the capture-resume path of `text_generation_scheduler.cpp` (around `:2631`, `push_back`). Run the scheduler tests.
8. **V1-10 #1117 (3016d652):** cherry-pick into `src/core/hip/snapshot_copy.hip` and its callers. Build and run any CPU test covering it.
9. **V1-4 #1273 (390639b3):** port the formula, not the file move. The host reserve becomes `max(CmaFree, 4 GiB)` instead of CmaFree plus 4 GiB, in `src/cli/serve/text_model_runner.cpp` (~108-129, ~211). Keep or adapt upstream's tests. Run the text_model_runner tests.
10. **V1-2 #1249 (6ecfa504):** cherry-pick `src/core/hip/host_allocations.{cpp,hpp}` and the fake_hip header.
    - Convert our raw GPU-visible `hipHostMalloc` sites to `HostAllocations`, or to the same pin-then-register path: `src/cli/serve/inference_backend.cpp:~1155` (snapshot arenas) and `src/models/qwen/hip/arena.cpp:~1275` (27B KV block table). Leave load-only staging (`qwen/hip/model_loader.cpp:~99`) unless trivial.
    - Add one INFO log at startup (or first allocation) with the bytes pinned long-term against the bytes that fell back to hipHostMalloc. The GPU pass must be able to see whether the pin worked inside docker.
    - Write the exact docker flags the pin needs into `plan/upstream-sync2/V1-BRIEF.md` under "Container notes": `--ulimit memlock=-1` and a seccomp profile allowing `io_uring_setup`/`io_uring_enter`/`io_uring_register`. If a minimal seccomp JSON is needed, put it at `tools/serving/seccomp-io-uring.json`, derived from Docker's default profile. **Do not edit the supervisors** (`tools/serving/*/gufo-server.py`).
    - Run the host_allocations/cache tests that use fake_hip.

When an item cannot be ported cleanly or turns out not to apply, skip it. Say why in your report; do not force it.

## Done

- Each item is committed, and the targets you touched build.
- The listed CPU tests pass, and `build/b1/checkfmt.sh` passes.
- Report in English: per item, the commit hash, what changed, the tests run with their results, and anything skipped or uncertain. List the GPU checks each item needs in the combined pass.

## Container notes

(to be filled by V1-2)
