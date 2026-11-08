# B10 brief: combined GPU pass after B9 and server cutover (2026-11-08)

The user allowed the live service to be taken down for this pass ("지금
서비스 내려도되고 아무거나 다해도되"). Main checkout
`/home/m2on/projects/gufo`, branch `ai-tools/port-u4f`, HEAD `b56b9f54`, no
worktrees, no push. Never import or run code from `~/projects/_ai.tools`.
Backups go only under `~/projects/_backup/` (add a README line). Do not build
anything while a GPU timing run is in progress (shared LPDDR).

Inputs: candidate `build/b1/b9/b9-final-gufo` (device code = HEAD, 7,059/7,059
symbols identical to pre-B9); deployed baseline
`~/.local/share/gufo/runtime-b5-20261108/gufo` (source eb789c77). The GPU
wrapper is `build/b1/gpu/gpu.sh` (over `tools/docker/gpu-run.sh`); earlier
runs and `perf.sh`/`parse.py`/`RESULT.json` are in `build/b1/gpu/`. Read
`docs/GFX1151-KNOWHOW.md` and `plan/exl3/HANDOFF.md` first; do not re-measure
what KNOWHOW already records.

## G0 stop the server
`llm stop`; confirm the GPU is idle (no gufo container; searxng may stay).

## G1 correctness
- GPU CTests for Flash-Next (`qwen38_flash_next.*` with the GPU label,
  including the `exl3_ops` modes, `exl3_moe_kernel`), `tests/core`
  `snapshot_copy_test`, `qwen38_flash_next_model_tests` (session, snapshot,
  adapter). Use the gpu-test preset build inside the pinned container.
- Prefill logits sha must still be `17dc8f9f…` (as since B3) on the candidate.
- `gdn_stable_route_equality` / "fused stable GDN differs from the separate
  convolution" fails at baseline too. Diagnose and fix it properly (either the
  kernel or a test whose contract is wrong, with evidence). If the fix changes
  prefill/decode bits, run the KL256/top1/NLL quality check against the c03
  reference (see `build/b1/gpu/quality/`) and accept only if quality is
  unchanged within noise. Implement → second-reviewer review → fix → re-review.

## G2 performance (one matched run)
Candidate vs deployed B5 with `perf.sh`-style alternation: PP2048/PP8192,
AR TG128 d0/d32K, MTP TG128 d0/d32K. Expect B6 P1/P3/P5 ≈ +0.5% PP, decode
unchanged. Per-cell 5% / 10 ms flags; investigate flags once.

## G3 cutover
If G1 and G2 pass: install the candidate as
`~/.local/share/gufo/runtime-b9-20261108/` (binary + whatever the runtime dir
needs, mirroring runtime-b5), point the installed supervisor at it via the
repo (`tools/serving/flash_next/`, commit, then `install.py`), and move
`runtime-b5-20261108` into `~/projects/_backup/` as the rollback runtime with a
README restore recipe. If they fail, keep B5 and report. Then `llm start`,
verify `llm status` ready, one text and one image smoke request, and fence
readings at the next stop are not needed — just leave it running and healthy.

## G4 record and report
Record every measurement in `docs/GFX1151-KNOWHOW.md` (and BENCHMARKS.md if a
retained number changes), update `plan/exl3/HANDOFF.md` and
`plan/README.md`, archive this brief, commit (Conventional single-line,
session attribution lines), no push. Report: tests, sha, GDN fix, perf table,
deployed binary sha, rollback location.
