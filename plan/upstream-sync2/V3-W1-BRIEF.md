# V3-W1 brief: release idle rewind points once disk covers them

Read `plan/upstream-sync2/PLAN.md` ("V3 마무리"), `docs/KV-CACHE.md` ("Idle and stopping sessions are written back", "Rewind points keep a rewritten reply's turn reusable") and the KNOWHOW entry "V3-0: in-place rewind point". Branch `ai-tools/port-u4f`. Code: `src/cli/serve/component_text_cache.cpp` (rewind points, `RewindArenas`, `ArenaCharge`), `src/cli/serve/text_generation_scheduler.cpp` (idle write-back around the `idle_write_back_delay` loop).

## Problem

The user worries that cache memory becomes an outage cause (11-18: slowdowns and failures under memory pressure) and that it cannot grow without bound. A rewind point holds host copies (~135 MB per slot, budget = a quarter of the snapshot arenas) for as long as its slot stays assigned, even when the conversation has been idle for hours and idle write-back already put that same boundary on disk. Snapshot arenas trim after 60 s idle, but arenas a rewind point holds are never trimmed.

## Goal

A rewind point whose conversation has been idle for a release delay (default: a few minutes; pick a constant, e.g. 5 minutes, tied to or a small multiple of `idle_write_back_delay`; no new server option unless needed) and whose boundary is covered by a **published** disk checkpoint (not merely queued) is dropped: its arenas return (and can then be trimmed by the existing idle trim), its retained KV rows stay in the slot as ordinary rows. The next turn of that conversation then restores from disk (or continues the live frontier if it does), exactly as when no point exists. If the boundary is not covered, write it first (idle write-back already does this for marked conversations) and release only after publication; a failed write keeps the point.

- Never drop a point a running or queued request might use, or one with a queued disk write still reading its arenas.
- Log `event=rewind_released reason=idle host_bytes=... idle_s=...`.
- No change to behaviour while requests are active; no extra device work on the decode path (host-side check in the idle loop only).
- Keep greedy/seeded exactness: a turn after release must produce the same tokens as one served from the disk checkpoint of that boundary.

## Rules

- **No GPU, model or server runs. No production changes.** Do not touch `~/.local/share/gufo`, `llm`, or docker containers other than the build.
- Build only with `build/b1/build.sh <targets>`; CPU tests via `build/b1/build.sh --run /build/obj/<path>`; `build/b1/fmt.sh -i`, `build/b1/checkfmt.sh`; `python3 tests/functional/test_runner.py`.
- Never run anything from `~/projects/_ai.tools`, any `ai-tools/*` docker image, or Kyojin. No `pgrep -f`/`pkill -f`.
- Do not touch `tools/exl3_quant/hc/`, `plan/exl3/HC-Q8-CALIBRATION.md`, or the untracked `plan/research/*.md`.
- Leave the diff uncommitted in the working tree (the main session commits it).
- Update `docs/KV-CACHE.md` (the rewind paragraph and the log table).

## Done

- Builds (`gufo` plus the affected CPU tests); CPU tests for the component cache / scheduler pass, with new tests: idle point covered → released after the delay; uncovered → written then released; queued write or active request → kept; next turn after release restores from disk. checkfmt passes.
- Report in English: design, constant chosen, files, tests, risks, and how the soak (`build/b1/soak/`) will show it (`rewind_released` count, `host_available_mib` trend).
