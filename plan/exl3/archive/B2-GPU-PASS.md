# B2 combined GPU pass (2026-11-07)

Same procedure as the B1 pass below, with these B2 changes (they override):

- Commits under test: `c49ea976..HEAD` (`plan/exl3/archive/B2-BRIEF.md`, items E1, E2,
  E4, E6, E7 + fixes). Baseline = B1 candidate: build `cc86343e` in a
  temporary git worktree with its own copy of `build/b1/obj` (as the B1 pass
  did with `build/b1/base-src`), save the binary as `build/b1/b2-baseline-gufo`,
  then remove the worktree. Results go to `build/b1/gpu-b2/`.
- Unit tests: the CTests `qwen38_flash_next.exl3_ops`, `exl3_producer_rotation`,
  `exl3_moe_kernel`, `gdn_ops`, `hc_mix_ops`, `snapshot_copy`,
  `exl3_prefill_executor`, `exl3_gdn_prefill`, `exl3_prefill_overlap`,
  `mtp_kv_executor`, plus all B1 ones. Known pre-existing failure: gdn_ops
  "fused stable GDN differs from the separate convolution" (fails at f24b10ea).
- E1/E4/E6/E7 are intended exact: candidate greedy AR and MTP bench token
  hashes should equal the B1 candidate (`6d067a1254ff…` d0, `d856f78e53b9…`
  d32768). A difference is a finding to diagnose (E2 explicit FMA is the only
  allowed prefill bit change; check with the quality collector).
- Quality: collect once; compare against B1 candidate
  (`build/b1/gpu/quality/`) and the saved c03 reference.
- Perf: B C B C as before. Additionally isolate E7 (side-stream shared MLP)
  by a candidate variant with E7 reverted (`2e8a81fb`, `c4f054d4`, `51044775`),
  TG rows only, and report its delta; also report MTP draft/accept/cycle counts.
- Decode profile (step 7) is required this time: candidate AR TG d0, report
  kernel-active %, launches/nodes per token and the top kernel categories by
  time (ms/token), using `tools/prof/prof.py`.

---

# (B1 procedure, reused)

One combined correctness + quality + performance window for commits
`7c2e79c9..HEAD` on `ai-tools/port-u4f` (see `plan/exl3/archive/B1-BRIEF.md` for the items
and the numerical policy: bits may change; lossy C2/C3/C4 are kept only if
quality holds). The user approved stopping the live server for this window.

## Fixed inputs

- Repo `/home/m2on/projects/gufo`; incremental CPU build `build/b1/build.sh`
  (pinned Nix ROCm10.1 container, `/build/obj` = `build/b1/obj`).
- Baseline binary: `build/b1/baseline-gufo` (source `f24b10ea`, same toolchain
  and flags). Candidate: `build/b1/obj/gufo` at HEAD.
- Pack: `/home/m2on/projects/ai.tools/storage/models/llm/flash-next-exl3-gufo-c03`
  (mount read-only at `/models/fnx`).
- Existing tooling to reuse/adapt (do not edit it in place; copy into
  `build/b1/gpu/` if you need changes):
  `/home/m2on/projects/ai.tools/test/experiments/flash_next_quant_20261104/`
  `final_current_quality.py` (rq_logits 32,760-position collector + BF16-teacher
  scoring), `final_throughput_current_job.py` (bench args), `run_stage.py`
  (GPU docker flags: `--device /dev/kfd --device /dev/dri --ipc host`,
  `--group-add` the kfd/render gids, image
  `sha256:e7a761a40187efb13515a0275362ab09d084a129f4544aa4300ec5cccb8c7468`,
  volume `gufo-nix` at `/nix`, runtime env from `final_current_build.environment()`).
- Saved c03 FP16 quality reference result:
  `/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104/final-validation/quality-current-001/`
  (KL256 0.04152218, top-1 90.8394%, NLL 2.1626503 per KNOWHOW "c03 KVarN K4V4
  versus retained FP16"). Reuse it; do not recollect the baseline quality.

## Procedure

1. Build HEAD targets: `gufo`, `qwen38_flash_next_rq_logits` and the GPU unit
   tests `qwen38_flash_next_{gdn_ops,exl3_moe_kernel,projection_ops,hc_mix_ops,exl3_ops}_test`
   (check exact target names in `tests/models/qwen38_flash_next/CMakeLists.txt`).
   Finish ALL builds before any GPU timing (shared LPDDR; builds corrupt timing).
2. `llm stop`; confirm the service container is gone, no other KFD compute
   owner, and host MemAvailable is enough (model ≈64 GiB).
3. GPU unit tests (container with GPU devices, mount `build/b1` at `/build` and
   the repo at `/build/source`). Any failure: diagnose and fix in the source
   (commit separately), rebuild only affected targets, rerun only failing tests.
4. Quality: candidate rq_logits collection on the same 32,760 positions,
   FP16 KV, then score vs BF16 teacher and compare with the saved reference
   (KL256, top-1, NLL, paired window/bootstrap intervals as the existing scorer
   reports). If the candidate is clearly worse than the saved reference
   (e.g. ΔKL256 > +0.002 or Δtop-1 < −0.2 pp, or paired NLL interval excludes
   zero upward), isolate by building variants that revert C4 (`91ee3ca7`,`3ef9ffcc`),
   C3 (`1da02670`) and C2 (`9b2152f9`) individually, and report each. Do not
   widen thresholds; report the numbers and let the parent decide.
5. Performance, same window, alternating baseline/candidate (B C B C), r2 each,
   with the existing bench args: `bench -m /models/fnx -r 2 --kv-cache-type fp16
   --temperature 0 --top-p 1 --top-k 0 --min-p 0 --repeat-penalty 1 --seed 314159
   --verbose` + `-p 2048,8192 -d 0` (PP) or `-n 128 -d 0,32768` (TG), each with
   `--speculative off` and `--speculative mtp --mtp-model /models/fnx --draft-tokens 7`.
   Judge each row at 5% and 10 ms; keep all raw outputs.
6. Output sanity: candidate greedy AR and MTP emitted tokens must agree with
   each other (MTP verification keeps greedy output equal to AR); inspect that
   the text is coherent. Baseline vs candidate token hashes may differ.
7. Optional if time permits: a short decode profile (`tools/prof/prof.py`) of
   candidate AR to report kernel-active % and nodes/token.
8. Restart the service: `llm start`, wait for ready, check `llm status` shows
   ready and a short chat request answers. This step is mandatory even if
   earlier steps fail.

## Report

Write `build/b1/gpu/RESULT.json` and report: unit test results, quality table
(candidate vs saved reference, plus any isolation variants), perf table
(baseline vs candidate per row, % change, verdict), AR/MTP equality, server
restored state, fixes committed. Do not edit `docs/` — the parent records.
