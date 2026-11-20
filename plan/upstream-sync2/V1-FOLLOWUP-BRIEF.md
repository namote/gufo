# V1 follow-up brief: short-replay cache miss, session test, recovered-call numbers

Read `plan/upstream-sync2/PLAN.md` (the "V1 후속" and "V1 패스에서 새로 찾은 기존 문제" notes), and `docs/KV-CACHE.md`.
Read `docs/GFX1151-KNOWHOW.md` only as needed. Work on branch `ai-tools/port-u4f`; HEAD is the V1 work.

## Rules

- **No GPU, model or server runs. No production changes.** Do not touch `~/.local/share/gufo`, `llm`, or docker containers other than the build.
- Build only with `build/b1/build.sh <targets>`. Run CPU tests with `build/b1/build.sh --run /build/obj/<path-to-test-binary>`. The source tree is mounted at `/build/source`. Format with `build/b1/fmt.sh -i <files>`, then check with `build/b1/checkfmt.sh`.
- Never run anything from `~/projects/_ai.tools`, including `ai-tools/*` docker images (no `ai-tools/gufo-eval` either).
- Do not touch `tools/exl3_quant/hc/` or `plan/exl3/HC-Q8-CALIBRATION.md`. Leave the untracked `plan/research/*.md` alone.
- **Do not commit.** Leave the diff in the working tree. Main session reviews and commits.
- Fix behavior, not tests, unless the test's expectation is genuinely obsolete for the current design (the RAM checkpoint tier was removed in c597eed1 on purpose and must not come back). When you change a test expectation, say why in the report.

## Item A: short conversations lose their cache (main item)

Evidence (read-only, already collected on GPU): `build/b1/pass-20261120v1/func2/{base-tools,cand-tools}/`.
That run had `--cache-disk` enabled (the `cache` suite was selected; `server.log` shows `disk_cache=enabled`), on Flash-Next
EXL3 with `--sessions 4 --kv-cache-type kvarn-k4v4 --speculative mtp`. Base (production build before V1) and candidate fail identically:

- `tools` (`openai_sdk.py:2084 check_auto_tools`, and `:1546 check_strict_tools`): an unchanged ~290-token replay gets `cached_tokens=0`,
  `reuse=none`, `cache_miss_reason=no_compatible_boundary`, `deepest_live_tokens=0`, `deepest_disk_tokens=0`.
- `structured` (`:1310`, `:1343`), `structured-limits` (`:2280`), `tool-schema-edges` (`cache_growth.py:44 check_unchanged_retry`).
- `text-cancel`, `thinking-tool-cancel`, `legacy-tool-cancel` (`continuation.py:248`): "interrupted conversation lost its prefix",
  e.g. `{'cached': 0, 'prefill': 186, 'disk': False}`.
- An earlier run without disk (`build/b1/pass-20261120v1/func/cand-tools/server.log`) logs
  `event=live_frontier_capture_refused reason=reassignment ... cause=no_disk_cache` 204 times. The disk run logs only 3 `live_frontier_captured`.
- Production serving (4 sessions, disk on) mostly hits: 360 `reuse=live`, 106 `reuse=disk`, 4 `reuse=none`. So the loss is specific to
  short prompts (below the 2,048-token capture grid) and/or cancelled requests, plus slot reassignment.

Find why a short conversation's state is neither kept live nor written back on reassignment/cancel when disk is enabled, and fix it
so that an unchanged or continued short conversation reuses its prefix (from the live slot, or from the live-frontier disk write-back
documented in `docs/KV-CACHE.md` "A cold assignment, or a pool reclaim..."). Candidates to check: slot choice for a new request
(does it evict the idle slot that holds the matching prefix?), the cancel path (does a cancelled request reset its slot without a
frontier capture?), size thresholds that skip short frontiers, and whether a just-written frontier is visible to the very next lookup.
Keep the 2,048-token disk spacing for ordinary checkpoints.

Also judge the `disk-spacing` "short turns were not skipped: min_step_skips 0" failure in `func/base-bc` (it passed in `func2`). If the
check depends on log lines Flash-Next no longer emits, or on timing, make it robust; otherwise fix the cause.

Add CPU tests (e.g. `tests/cli/component_text_cache_test.cpp`, scheduler tests) that reproduce the lost prefix and pass after the fix.

## Item B: `qwen38_flash_next_session_test` `CheckServingEos`

It fails with "scalar EOS checkpoint did not replay exactly without prefill" on a backend without a cache tier (since c597eed1).
Make the expectation match the current design (or the fix from item A). This is a GPU test; only make sure it builds.

## Item C: recovered-call number edge cases (`src/cli/serve/openai_chat.cpp`, recovered-call schema check, commit d7502019)

1. A call mixing an integer-valued `1.0` (integer schema) and `1e-7` passes with neither spelling. Fix the canonical spelling so it
   does not emit a leading zero in the exponent (or otherwise make both constraints satisfiable together), checked per number.
2. Integers above 2^53 are bound-checked on the canonical double. Substitute the canonical spelling per number leaf only when its
   decimal value is preserved exactly.
Add tests in `tests/cli/openai_chat_test.cpp` (and `tests/core/json*` if you touch `src/core/json.*`).

## Done

- Targets you touched build; the related CPU tests pass; `build/b1/checkfmt.sh` passes.
- Report in English: root cause of A with file:line evidence, what changed per item, tests run with results, and the exact
  functional suites the GPU pass must rerun.
