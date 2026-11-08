# B9 brief: Flash-Next ROCm structure cleanup and serving ops in-repo (2026-11-08)

gufo is our own fork; no upstream PRs (restructure freely, keep the repo's
CLAUDE.md conventions). Main checkout `/home/m2on/projects/gufo`, branch
`ai-tools/port-u4f`, no worktrees, no push, no GPU runs. Second-reviewer
subagent review per item, fix, re-review. Conventional single-line commits with
the session attribution lines. Backups only under `~/projects/_backup/`.

## S1 split the monolith files (pure relocation)

`src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp` (9,905 lines) and
`executor.cpp` (6,672 lines) mix HC, GDN, attention, MoE, PLE, prefill, decode
and MTP code. Split them by stage into cohesive translation units, e.g.
`kernels/rocm/{hc,gdn,attention,moe_glue,norm_rope,ple,…}.hip.cpp` and
`executor_{hc,gdn,attention,moe,prefill,decode,mtp}.cpp` (+ a private
`executor_internal.hpp` for shared helpers), registered in the model's
CMakeLists. Rules:
- Pure moves: no logic, signature or numeric change. Keep anonymous-namespace
  kernels anonymous in their new TU; shared device helpers go to a private
  `.hpp` included where needed. Keep `kernels.hpp` as the public interface.
- Watch fast-math and contraction: a function moved into another TU or inlined
  differently can change codegen (B3 SsmConv4 lesson). Acceptance is the
  device code: every gfx1151 kernel's code object must be byte-identical to
  the pre-split build after strip-debug (kernel symbol set identical; compare
  per-kernel `.text` from the extracted code objects, since TU boundaries
  change the bundling). Host code: same behaviour; differences only from
  relocation (report). Reuse `build/b1/cmp/` scripts from B8.
- Build time: the split should also cut incremental rebuild time; report the
  per-TU compile times before/after.
- Do it in a few commits (kernels split, executor split), each passing the
  comparison.

## S2 serving ops into the repo

`~/.local/share/gufo/ops/` (copied verbatim from retired ai.tools, minified
style) and `~/.local/share/gufo/gufo-server.py` run the live server outside
git. Move them into `tools/serving/flash_next/` (or the existing
`tools/serving/` layout if one fits): readable formatting, same behaviour, no
new features, standard library only, a README (install, `llm` wrapper,
rollback via `~/projects/_backup/`), and CPU tests for the pure parts (fence
parsing, host classification with recorded fixtures). Then install: the live
`~/.local/share/gufo/gufo-server.py` and `ops/` become thin copies or a
symlink to the repo files (decide what is safest; the server must not depend
on the working tree mid-edit — prefer an install script that copies a tagged
version into `~/.local/share/gufo/` and records its git sha). Back up the
current live files under `~/projects/_backup/` first. Do NOT restart the live
server; verify with `llm status` and a dry `python3 -m py_compile`, and report
that a restart is needed to exercise the new copy (the parent will ask the
user).

## S3 report

Commits, the file map (old → new), the device-code comparison result, compile
times, the serving move, tests/checks, and remaining structural debt.

## Additions (2026-11-08, after the tooling import `a059fc13..ced06524`)

- **S0 checks first.** `tools/ci/check-format.py` fails on
  `tests/models/qwen38_flash_next/exl3_routed_reference.hpp` (comment edit in
  B8); fix it. `check-kernel-resources` reports 14 kernels over budget on the
  current binary: determine whether that is pre-existing (compare with the
  budget file history / the B5 deployed binary) and either update the budget
  with evidence or report.
- Evaluation inputs moved out of `_ai.tools` to
  `/home/m2on/projects/models/eval/flash-next/` (README there); point any
  tracked docs/tools defaults that name the old `_ai.tools` evaluation paths at
  the new ones (local `build/b1/gpu/gpu.sh` already repointed).
- **S4 structure review loop (the user's goal: "folders cleanly split by
  structure").** After S1/S2, have a fresh reviewer subagent (not anchored on
  your reasoning; give it the tree, not your conclusions) audit the
  Flash-Next-related layout — `src/models/qwen38_flash_next/**`,
  `tests/models/qwen38_flash_next/**`, `tools/{exl3_quant,docker,serving,
  bench,qwen38_flash_next}/**`, `plan/` — for: files mixing unrelated stages,
  >2,500-line files, misplaced tests/benches/tools (model-specific code outside
  the model's folders, or generic code inside them), duplicated helpers,
  dead targets, naming inconsistencies, READMEs missing for non-obvious
  folders. Fix what it finds (same exactness rules: device code identical,
  behaviour unchanged), then audit again with a new reviewer. Repeat until a
  round finds nothing material (max 4 rounds; report what remains if any).
