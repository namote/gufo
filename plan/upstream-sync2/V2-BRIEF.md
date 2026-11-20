# V2 brief: medium upstream items (for the implementing agent)

Read `plan/upstream-sync2/PLAN.md` (V2 table), the matching sections of `plan/upstream-sync2/survey-upstream.md`
(§7, §12-16, §18 #1205, §20) and `survey-kyojin-exl3.md` §3. Read `docs/GFX1151-KNOWHOW.md` and `docs/KV-CACHE.md` as needed.
Branch `ai-tools/port-u4f`, HEAD 942bf0b2 (V1 and its follow-up are in).

## Rules

- **No GPU, model or server runs. No production changes.** Do not touch `~/.local/share/gufo`, `llm`, or docker containers other than the build.
- Build only with `build/b1/build.sh <targets>`. Run CPU tests with `build/b1/build.sh --run /build/obj/<path-to-test-binary>`. The source tree is mounted at `/build/source`. Format with `build/b1/fmt.sh -i <files>`, then check with `build/b1/checkfmt.sh`. Python functional self-tests: `python3 tests/functional/test_runner.py`.
- Never run anything from `~/projects/_ai.tools`, including any `ai-tools/*` docker image (also not `ai-tools/gufo-eval`, not even for objdump).
- Do not touch `tools/exl3_quant/hc/` or `plan/exl3/HC-Q8-CALIBRATION.md`. Leave the untracked `plan/research/*.md` alone.
- `git fetch upstream` is done. Use `git show <hash>` for upstream diffs. exllamav3 is not fetched: port that idea from the survey description, not code.
- **One commit per item.** Single-line Conventional Commit with the upstream PR number when there is one, e.g.
  `perf(cache): sync new disk checkpoint payloads concurrently (upstream #1205)`. Every commit ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_016Kh8svPWb3TTEKxGwdHJTN
  ```
  Do not push.
- Keep our fork's structure: Flash-Next runtime lives in `src/models/qwen38_flash_next/runtime/` (upstream `kernels/rocm/*` maps onto it); our cache has **no RAM checkpoint tier** since c597eed1 (live slot rows, then disk through host snapshot arenas). Port by content, not by file.
- If an item does not apply to our design (for example it only matters for a RAM tier we no longer have), or cannot be ported cleanly, **skip it and say why**. Do not force it.
- Speed items cannot be timed here. Make them the default when they are exact by construction (proposal-only changes, I/O ordering), and keep each in its own commit so the combined GPU pass can revert one cleanly.

## Items (in this order)

1. **V2-6 #1205 (70bd0fda):** concurrent fsync of new disk payloads in `src/cache/disk.cpp` (ours syncs one at a time near `:815`). Port with its tests (`tests/cache/disk_publication_test.cpp`, `streaming_test.cpp`).
2. **V2-2 #1128 (bd6b0fdd):** nested tool objects that declare `properties` without `additionalProperties` become closed; accept llama.cpp's required-first member order. Hand-merge into `src/core/json_constraint.cpp` (our U1-2 port differs), with the json_constraint tests and the `tests/fixtures/llama_cpp_tool_grammar.json` fixture. Restore the `digits/nested` fixture row dropped in V1-5 (2d5d81c2) if it now passes.
3. **V2-8 #1270 (fdbc654e):** learn a shared branch point after 192 common tokens (waiting for a peer still needs 512). `text_model_runner.hpp:~794`, `.cpp`. Include its CPU test and the functional threshold edits.
4. **V2-3 #1288 + #1290 (0a473ebf, cc405337):** retention grace for a new cheap conversation and keep edit history on an unchanged retry, in `src/cache/retention.cpp`. First decide whether our retention policy (which now governs disk entries and live captures, not a RAM tier) has the same failure. Port only what applies, with `tests/cache/retention_test.cpp`.
5. **V2-5 #1285 (11a51d2f):** commit backing for a disk restore's capture before the restore (avoid a serving-thread stall of ~57 ms/GiB). Check whether our host snapshot arenas still commit backing on the serving thread after a disk restore; port only the hook if so.
6. **V2-1 #1158 (850b5271):** `AlignToLiveFrontier`: continue the live frontier when the client's re-tokenized history splits our previous answer differently but the text matches (including template-trimmed whitespace at reply edges). ~250 lines into `component_text_cache.cpp`, `inference_backend.cpp`, `text_model_runner.*`, scheduler. Port upstream's 147 test lines. It must coexist with our `continue_final_message` work (28e3873a) and U2-2 alignment; a wrong alignment means a wrong continuation, so be conservative.
7. **V2-9 franzmoca e2ecbf12 + 41d7d621** (`git show` them; they are on `origin`): greedy prompt-lookup copies stop where other matching occurrences disagree. `src/core/speculative/prompt_lookup.hpp`, and our Flash-Next lookup append in `engine.cpp`. Proposal-only, so exact; CPU tests for the lookup.
8. **V2-4 #1209 (a33323f2):** draft MTP proposals with a 4-bit copy of the head prefix (98,304-row subset, `draft_vocab.hpp`, `runtime/executor_mtp.cpp:~497`). Our head is EXL3 K8 (`exl3_weights.cpp:~448`). Build the 4-bit copy at load (EXL3 K8 → F16 → Q4_K, using our existing Q4_K GEMV if there is one, or an EXL3 K4 path), used for proposals only; the target verify stays full precision. Report the extra VRAM. Add a CPU or reference test for the quantized copy if feasible; the kernel needs a GPU check in the combined pass.
9. **V2-7 exllamav3 47d42fe idea:** frequency-ranked dynamic draft vocabulary (survey-kyojin-exl3 §3). Rank tokens by decayed counts (prompt tokens weighted 0.2), rebuild every 128 tokens with 25% hysteresis, use the subset only while the attributable miss rate is below break-even, probe the full head every 128 rounds. It interacts with V2-4 (the 4-bit copy should cover whatever subset is used). If this is too large to land safely in one commit, implement the ranking/miss-rate machinery with CPU tests and leave it disabled, and say so.

## Done

- Each applied item is committed; targets you touched build; listed CPU tests pass; `build/b1/checkfmt.sh` passes.
- Report in English: per item, the commit hash (or skipped + why), what changed, tests run with results, risks, and **the exact GPU checks the combined pass needs** (functional suites, `qwen38_flash_next_session_test`/`mtp_sampling` cases, cost-audit or speed cells, acceptance measurements, pi_agent).
