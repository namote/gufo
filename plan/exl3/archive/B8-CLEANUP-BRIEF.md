# B8 brief: Flash-Next optimization wrap-up cleanup (2026-11-08)

The user ended the optimization work (B1–B7) and asked for a clean, tidy final
state. Scope: everything changed on `ai-tools/port-u4f` since `7c2e79c9`
(`git diff --stat 7c2e79c9..HEAD -- src tests tools`). Main checkout only, no
worktrees, no GPU/model/timing runs, no push. Second-reviewer subagent review of
the whole cleanup diff before committing; Conventional single-line commits with
the session attribution lines.

**Production behaviour must not change.** Acceptance: after the cleanup, the
stripped production code of `gufo` (`llvm-objcopy --strip-debug`, then compare
`.text` and the embedded HIP code objects, e.g. via
`clang-offload-bundler`/`--offload-arch` extraction or the `.hip_fatbin`
section) is byte-identical to the pre-cleanup HEAD build, OR every difference is
explained as non-functional (line tables, symbol names) in the report. If a
cleanup would change production code, do it only if it is a pure dead-code
removal and say so; otherwise leave it.

Read KNOWHOW entries "Flash-Next B1 … B7" for what was kept and reverted.

## Tasks

1. **Leftovers of reverted work.** Search for dead code, unused buffers,
   counters, parameters, template flags, includes, enum values, CMake entries
   and comments that refer to reverted items: B1 kPrep, B3 G1 HcUpMix/
   HcDownSilu tail path, G3/G4a/G4b/G4d, G6 kernarg default, B4 H4b ranges,
   B5 J1 dense prefetch, B6 P7 side-stream policy. Remove what is unused.
2. **Benches and reference copies.** Decide for each of
   `tools/bench/{flash_next_dense_gemv,flash_next_hc_decode,flash_next_moe_glue,
   flash_next_hc_prefill,gfx1151_launch_gap,gfx1151_exl3_wmma_ceiling}.hip`,
   `tests/models/qwen38_flash_next/{exl3_gemv_reference,exl3_routed_reference,
   hc_combine_reference}.hpp` and the bench CMake targets: keep if it guards an
   exactness contract of kept code or records a measurement cited in KNOWHOW;
   remove if it only compares two identical kernels now (e.g. dense GEMV bench
   after the J1 revert) — then also fix the KNOWHOW mention in the same commit
   (docs edits allowed only for that). Keep the frozen reference headers that
   tests rely on, but make sure their header comments say what they freeze and
   why.
3. **Comments and naming.** Comments in kept code should describe the current
   design, not the batch history ("B4 H4a", "G1 …") — rewrite such comments to
   say what and why. Check names are consistent with neighbouring code.
4. **Duplication.** Where kept fused paths duplicated helpers (e.g. SiLU/Q8_1
   quantize, restore scaling, rotation slabs), factor shared device helpers if
   it is a pure refactor that keeps generated code identical.
5. **Tests.** Make sure every test added in B1–B7 is registered in CTest with a
   sensible name and not left as an orphan target; remove debug-only prints;
   make the known pre-existing gdn_ops failure explicit (find whether it is a
   stale expectation and, if fixing the test expectation is clearly right from
   the code, fix it; otherwise leave and report).
6. **Formatting and CI.** Run `tools/ci/check-format.py` with the pinned
   clang-format and the Python/repo checks that run on the host
   (`nix build .#checks.x86_64-linux.pr` is not available; run what the CPU
   container can: `cmake --build --preset pr` equivalent if possible, or the
   CPU unit tests built in `build/b1`).
7. **Plan files.** `plan/exl3/B*-BRIEF.md`, `B*-GPU-PASS.md`,
   `B3-PERSISTENT-DECODE.md`, `B7-CEILING-*.md`: add a short
   `plan/exl3/B-SERIES-SUMMARY.md` index (one line per batch: goal, kept,
   reverted, measured result, KNOWHOW entry) and mark `B3-PERSISTENT-DECODE.md`
   as not pursued with the reason (B3 launch-gap and B7 results).

Report: commits, what was removed/kept and why, the binary comparison result,
the test/check results, and anything left for later.
