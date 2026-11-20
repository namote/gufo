# V3-2 + V3-3 brief: typical acceptance with loop guards, and streaming loop detection

Read `plan/upstream-sync2/PLAN.md` (V3-2, V3-3), `plan/upstream-sync2/survey-kyojin-exl3.md` §1 and §2, and in `docs/GFX1151-KNOWHOW.md` the sections "Flash-Next quality versus Kyojin: typical acceptance and HC Q8, November 10, 2026" and "U4-4 typical acceptance (Kyojin d8dbb75), opt-in". Branch `ai-tools/port-u4f`. Use `src/models/qwen/AGENTS.md` for control tokens.

## Background

- `da021c03` ported typical acceptance for sampled MTP verification (Kyojin d8dbb75). U5 measured all 18 sampled requests faster, +2.5…+27.6 %.
- On 2026-11-10 it caused thinking loops (xhigh, misspelled Korean lodging-review image, seeds 1–4: 3/4 looped; off: 4/4 finished), so it was removed in `7f54d782` (D10).
- Kyojin since added fixes (`9f1b368` #177, `2484e31` #183; the Kyojin repo is a partial reference only, read it from the survey; do not run it):
  - copied (prompt-lookup) drafts stay exact;
  - refuse a relaxed acceptance that would complete a 12-gram seen ≥2 times in the last 2048 generated tokens; after such a signal use exact acceptance for an adaptive window (512 tokens, doubled on retrigger, cap 16384);
  - phase gate: relaxed only after `</think>` closes (`answer`), exact inside thinking.

## V3-2: bring typical acceptance back, guarded and opt-in

- Restore the D10 removal (`git show 7f54d782`, `da021c03`) onto the current tree (MTP sampled verify in `src/models/qwen38_flash_next/runtime/executor_mtp.cpp` / `mtp_sampling.hpp`, `src/cli/serve/sampling_request.hpp`, shared sampling option and `docs/SERVER.md`). One canonical option name; request field and server flag as before.
- **Default off.** Greedy and the exact path must stay bit-identical to today when off (the session test and `mtp_sampling_test` check this).
- Add the three guards above as defaults when on: MTP drafts only (lookup copies exact), answer-phase only (use the shared Qwen control-token constants), 12-gram repetition refusal with the adaptive exact window. Keep them per-request host state with CPU tests.
- Must coexist with V2-4: sampled drafts read the 4-bit head copy; verification semantics are unchanged by that.

## V3-3: streaming loop detection (model independent)

- Port the exllamav3 `805caee` idea (rolling k-gram hashes, k=6; candidate periods with ±5% drift; a loop is 8 repetitions over 8192 tokens or 4 exact repetitions over 1024 tokens; exempt text copying the prompt; handle low-diversity text) as ~300 lines of C++ in the shared scheduler/generation path (`src/cli/serve/text_generation_scheduler.cpp` or the per-request loop), so Flash-Next and 27B share it. O(1) per token.
- Plus Kyojin's thinking rule: inside an open think block, a non-fenced line of ≥40 characters seen 3 times is a loop.
- Action when a loop is detected, **opt-in** per request and server option (default off, because code/JSON repetition can be legitimate):
  - inside thinking: close the think block (emit the shared `</think>` control token as the next input) and continue to the answer;
  - in the answer: stop with finish reason `loop_detected` (document it; map to a valid OpenAI/Anthropic finish reason where the schema requires one).
- Always log a detection at INFO (`event=loop_detected phase=... period=... repeats=...`) even when the action is off, so production logs show how often it would fire.

## Rules

- **No GPU, model or server runs. No production changes.** Do not touch `~/.local/share/gufo`, `llm`, or docker containers other than the build.
- Build only with `build/b1/build.sh <targets>`; CPU tests via `build/b1/build.sh --run /build/obj/<path>`; `build/b1/fmt.sh -i`, `build/b1/checkfmt.sh`; `python3 tests/functional/test_runner.py`.
- Never run anything from `~/projects/_ai.tools` or any `ai-tools/*` docker image, and never run Kyojin. No `pgrep -f`/`pkill -f`.
- Do not touch `tools/exl3_quant/hc/`, `plan/exl3/HC-Q8-CALIBRATION.md`, or the untracked `plan/research/*.md`.
- **Stage and commit only your own files** (`git add <paths>`; other sessions commit on this branch). One commit per item (V3-2, V3-3), single-line Conventional Commits, do not push. Trailers:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_016Kh8svPWb3TTEKxGwdHJTN
  ```
- Have a subagent review the design and the diff; fix and re-review.

## Done

- Builds; `mtp_sampling_test`, sampling/serve/scheduler CPU tests and new guard/detector tests pass; checkfmt passes.
- Report in English: commits, option names and defaults, what each guard does, false-positive handling, risks, and the exact GPU checks: off = bit-identical (`qwen38_flash_next_session_test`, functional `batch`/`tools`/`structured`), on = the 11-10 lodging-review repro (`build/b1/quality-20261110/` request scripts; xhigh, seeds 1–4, plus text-only xhigh) must not loop, sampled speed cells C1/C4 on vs off, and loop-detector logs on a pi_agent run.
