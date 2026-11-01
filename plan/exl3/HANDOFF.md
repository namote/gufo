# EXL3 / Flash-Next F3 handoff (2026-11-01; qualified batch integrated locally)

Latest status: the qualified F3 batch is committed and locally integrated.
Code checkpoint: `2c159a12` on `ai-tools/f3-integ`.
Port integration merge: `9fcaffea`;
integ merge: `ffc0738a`. See
[F3-INTEGRATION.json](F3-INTEGRATION.json) for the compiled-input identity proof.
All tracked production/test/toolchain inputs match the qualified checkpoint;
only documentation/guidance differ. No new build, GPU window or baseline
inference was needed for integration. No remote push was performed.

Retained: direct routing-ID combine, attention mask-pruning eligibility, and
grouped dense X3 scheduling. `fn-prefill-batch-nix101` is COMPLETE/health_PASS:
32,760 collected records and 64 head checks match saved Nix q6 exactly.
Prefill versus saved pre-batch Nix T1: pp512 +2.14%, pp2048 +2.04%, pp8192
+2.18%, d32K pp2048 +1.86% (one candidate r2 window). These are batch-level
observations, with prior hc7-relative quality drift still recorded.

The tested image remains `ai-tools/gufo-eval:f3-prefill-batch-f712b2d0-rocm101`
(image `93197735...`, as-built source patch `f712b2d0...`). Integration commits
attest source identity; they are not newly built image revisions. Preserve
`~/projects/gufo-f3-integ/build/f3-nix-incremental/{release,units}`, the operator
objects and all baseline/candidate artifacts for incremental work.

Performance-neutral simplifications/correctness remain eligible for retention.
Wider and persistent matrix prototypes added complexity without a qualified
benefit; their source/binaries are preserved. Previously completed TG/MTP work
is kept. Further prefill work remains the priority; the >=2,000 tok/s goal is
not met. TG/MTP expansion, deferred serving/27B qualification and deployment
are separate remaining work, not implied complete by this integration.
gufo-00 has ended; this chat coordinates machine use. Long validations stay
batched, and stored matching baselines are reused.

## Historical resume record (superseded by the latest status above)

Resume update (2026-11-01): T1 probe completed; shared indexer Q/K projection
reuse is implemented and independently reviewed in the uncommitted f3-integ
worktree. FP16-KV/KVarN collector bytes match hc7; focused Nix ROCm 10.1 unit
passes. Candidate images are `ai-tools/gufo-eval:f3-indexer-b03a1f1e-rocm101`
and `-tests-rocm101`. Full evidence and runtime caveats are in
`~/projects/gufo-f3-integ/plan/exl3/F3-RESUME-RESULTS.md`.
The user requires Nix ROCm 10.1, one combined final model/timing pass, and
reuse of stored matching baselines. No merge or final model qualification yet.
The first q3 batch item's Thai rare-vocabulary fixture failure is resolved:
raw repetition produced identical AR/MTP English reasoning, so the fixture
now explicitly continues a non-thinking assistant's requested repetitions.
All original coverage/exactness gates remain strict. Focused lifecycle and
sampled-head checks PASS on EXL3 and GGUF (`fn-draft-fixture-nix101`, health_PASS).
The q3 resume scope is now complete: reuse the original passed batch prefix
plus the repaired fixture tail; sampling, prefill and execution32k passed in
`fn-final-q3-remaining-nix101`. The EXL3 MTP oracle needed storage-boundary
and full-head repairs; independently reviewed, original tolerances unchanged.
Its final matched-input audit passed in `fn-final-q3-mtpinput-nix101`
(maximum carry relative RMS 4.71e-5, limit 5e-5). That window also completed
the three-token CPU/GPU floor diagnostic (604.27 s, top-1 3/3, finite KL up
to 0.00175); this is not a full-quality gate. health_PASS, actual loaded
libraries all Nix ROCm 10.1. Do not repeat this expensive CPU probe routinely.
Final test overlay: `ai-tools/gufo-eval:f3-indexer-mtpinput-rocm101`.
Evidence index: `fn-final-q3-mtpinput-nix101/q3-summary.json`; details and
hashes are in the F3 resume results. q6, Nix FN/27B timing, performance goals
and formal integration remain. Next q6 must use the Nix collector and reuse
saved BF16/UD results with their recorded provenance (the saved UD quality
metadata says 7.2.4 despite an old plan note calling it Nix; do not conflate
that with the separate Nix timing baselines).
q6 resume update: `fn-final-q6-nix101` completed32760 rows/64 head checks,
health_PASS, one candidate collection with stored BF16/UD/hc7 data reused.
KL2560.026481 and top1 92.0543% remain better than historical UD; versus hc7,
NLL increases0.006600 (window95%CI [0.001729,0.013902]), about0.66% perplexity.
This warning remains recorded; q6 is measured with documented drift.
The user directs prioritizing performance over eliminating small quality
changes. No precision patch was applied. Stop further quantizer diagnosis
and proceed to current-candidate Nix timing; see Q6-SUMMARY.json and the F3
resume results for tails, provenance and the isolated compiler diagnostic.
Latest user direction: **prefill is the highest priority**. Defer TG/MTP
optimization and T4. Completed pure-Nix T1 AR prefill baselines are pp512
1039.61, pp2048 1372.10, pp8192 1411.52 tok/s; d32K pp2048 1257.46.
Reuse `fn-final-t1-x-ar-nix101`; do not repeat it. T2 is also saved.
T3 worktree wrapper completed K17, then D was intentionally stopped;
`fn-final-t3-worktree-nix101/USER-PRIORITY-STOP.json`, health_PASS. Do not
interpret that cancellation as a failing candidate. Historical UD timing
images also inherited /opt/rocm/lib, so they are not proven matched controls.
An unqualified prefill candidate now replaces repeated MoE bucket searches
with retained original routing IDs; focused exactness tests and one combined
candidate validation/prefill timing pass are next. See F3 resume results.
The original hold-point inventory below remains historical context.

This lists what has **not** been done, what is **still unexplained**, and the
order to resume. Read [docs/GFX1151-KNOWHOW.md](../../docs/GFX1151-KNOWHOW.md)
first. It records what was already measured, so do not re-run that. It is a
living document: update it with every new result. History is in
[F3-PLAN.md](F3-PLAN.md) under "Status at hold", in
[F1-RESULTS.md](F1-RESULTS.md) and in [F3-SESSION-FINDINGS.md](F3-SESSION-FINDINGS.md).

## Where things are

| What | Where |
|---|---|
| Newest code (all F3 work, Flash-Next EXL3 prefill) | `ai-tools/f3-integ` @ `966f84b8`, worktree `~/projects/gufo-f3-integ`. Not merged into `ai-tools/port-u4f` or `ai-tools/integ`. |
| Last final images | `ai-tools/gufo-eval:final-68d4e3c8-rocm101` and `-tests-rocm101`. They predate all F3 speed work. |
| Plans and LOCKS | `~/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/` (PLAN-FINAL-FN2*.json, PLAN-FN-*.json, PLAN-FINAL-27B.json) |
| Receipts | `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/` (final-fn2/, fn-713c, fn-434f, fn-hc2…hc9, fn-att1/2, gdn/) |
| UD-Q4 timing baselines | Measured once on 68d4e3c8 with nix: T1 probe AR, T2-q MTP, T3-q card. Reuse them while the GGUF path stays byte-identical. |
| Packs | 5.05: internal `ai.tools/storage/models/llm/flash-next-exl3-5.05-h6-ng6/`. 4.05 and Kyojin (≈4.0 bpw): T5 EVO USB, about 270 s load each. |
| Dropped branch, kept as record | `ai-tools/fn-hc-int8` (HC int8 decode: +3.5 % tg, failed the gate) |

## Current numbers (docker 7.2.4 RelWithDebInfo probes, 5.05 + KVarN)

| | Start (68d4) | Now (hc7/att2) | UD-Q4 (nix) | Target |
|---|---:|---:|---:|---:|
| pp2048 | 454 | ≈1,255–1,260 | 1,762 | ≥2,000 |
| pp8192 | 466 | ≈1,290–1,324 | 1,796 | ≥2,000 |
| tg128 | 24.7 | ≈24.7 | 29.2 | at least Kyojin |
| Quality KL vs BF16 | 0.0269 | 0.0264 | 0.0392 | better than UD (met) |

The docker build runs about 10 % below nix: UD measured 1,567 under the
profiler vs 1,762 on nix. So EXL3 on nix is probably about 1,390–1,450 at
pp2048. That is an estimate and has not been measured.

## Not done yet

1. **No end-to-end measurement of `966f84b8`.** These have never been measured
   together: the 8x1 bucket tile (df1f7409), the hot-expert 128-row buckets
   (56202401), the PrefillCombine wrapper (8c9fb296) and the attention fusion
   (2e054760).
2. **Branches not merged.** `f3-integ` is not merged into port-u4f/integ, so no
   final nix images exist for it.
3. **Combined final pass never run** (one pass, with integration and timing
   together, per AGENTS.md "Iteration workflow"):
   - **q3**, the EXL3 session parity run: `--batch-only`, `--prefill-only`,
     execution-32k, MTP audit and engine floor. The parity bugs were fixed and
     pass the unit tests, but q3 itself was never rerun at model level.
   - **q6** (full-vocabulary prefill vs HF BF16, 32,760 rows) on the final
     build. Last pass: hc7 8737db4e, KL 0.02643.
   - **Flash-Next timing on nix:** EXL3 arms only, against the stored UD
     baselines (T1-x AR, T2-x MTP, T3-x card, T4 KVarN cost). There is still
     no measured MTP/card comparison against Kyojin's card.
   - **27B final timing:** the PLAN-FINAL-27B final-t-* windows have never run.
     Run them on the final image after a 27B fingerprint check, because W3 also
     changed shared EXL3 kernels.
4. **Decision pending: the default Flash-Next pack.** 5.05 is recommended
   (KL 0.0264 vs UD 0.0392). 4.05 is about UD quality (KL 0.0424) and +7 % tg.
   Kyojin's pack is worse (KL 0.0598).
5. **Decode not worked on since D2.** tg is about 15 % below UD. Routed
   experts run at 141–158 GB/s and HC plain F16 at 126–165 GB/s, against
   205–221 GB/s for the trunk GEMV.
6. **KVarN serving has no prefix cache.** The cache-reuse asserts are
   expected-fail (q5/q5b); adding the cache is an F3 follow-up.
7. **Know-how doc commit:** docs/GFX1151-KNOWHOW.md, F3-SESSION-FINDINGS.md,
   the SKILL.md pointer and the AGENTS.md line are waiting for the cross-review
   (Codex thread `knowhow-review`) and a commit.

## Still unexplained (no root cause found)

| Symptom | Numbers | What is known / why it stopped |
|---|---|---|
| Routed bucket GEMM issue stalls | 3.2–3.4 ms/call vs ≈2.05 ms floor (≈1.2 ms stalls) | Ping-pong, stagger, sync removal and hoisting all failed. This ROCm build has no SQ stall counters. |
| Dense X3 efficiency | 31–36 TFLOPS, about 61 % of peak | Not fragment reads (a 0.5-read tile gives no gain). Suspects: LDS staging and syncs in 32-wave blocks, plus serialized decode. Not broken down. |
| kWideHalf 128×256 tile (n % 256 ≠ 0) | 20–29 vs 31–36 TFLOPS | Used by Flash-Next shared-expert gate/up and the combined indexer (n=640, rows > 64); no 27B projection uses it. The cause is not measured. The estimated impact is small (≈5–10 ms/pass); confirm call counts first. |
| HC down in situ | 1.78 ms/call vs 0.77 ms warm bench | The dirty-cache effect is confirmed. The gap left after non-temporal producer stores is not explained. |
| HC int8 decode gain | +3.5 % tg vs ≈7–10 % expected | Not investigated; the experiment was dropped at the user's 10 % bar. |
| Fused MoE combine (HcCombine kMoe) | 2.3 ms/call | Suspected slot-gather latency (2.5 segments per wave); not measured. |
| First-request kernel gaps | 27/11/11 ms | Suspected lazy code-object loading; affects TTFT only. Not measured. |
| MTP audit tolerance (q3, first run) | rms ≈2e-4, over the oracle tolerance | It failed together with the parity bugs and was never rerun after the fixes. |

## Open leads not tried (from the code sweep; see the doc's open-leads section)

- Sparse prefill computes the combined indexer projection twice; reuse it. This
  is a cheap, sure prefill win.
- Combine repeats expert-ID searches.
- A tested scheduling hint (iglp) is excluded from grouped X3.
- The 27B crossover threshold is still provisional.
- Check the remaining GGUF dense-core assumptions; revisit the EXL3-specific
  MTP costs; inventory the standalone stable fallbacks still reachable.
- A masked 256×256 tile in place of kWideHalf (see above).
- A persistent/stream-K bucket schedule and larger prefill chunks. Both pay
  only once routed is weight-bound, which it is not yet.
- Code cleanups flagged: stale dispatch comments, the ISA help path, bench vs
  production no-SLP flags, and profiler classification of EXL3 kernels.

## Resume order

1. T1 probe on `966f84b8` (EXL3 only, docker binary; label it as a probe).
2. Fix the double indexer projection; gate it with the collector byte check.
3. Get the default pack decided, then merge `f3-integ` into port-u4f/integ and
   build the nix final images (X-A's recipe, label = merge commit).
4. Run the combined final pass once (q3 incl. the MTP audit, q6, FN timing,
   27B timing). Diagnose failures, fix, and rerun only the affected checks.
5. Then decode (routed and HC GB/s), and the unexplained items above if
   prefill is still short of the target.

## Rules that still apply

- GPU windows one at a time under `ABBA-OWNER.lock`. gufo-00 has ended; this chat coordinates builds,
  hashing and large writes. No builds during timing windows.
- Root free space must be ≥150 GiB.
- Commit your own files only, as one-line Conventional Commits, and do not push.
- Bit changes need the q6 gate; chunk invariance is mandatory. The collector
  `.bin` byte check is authoritative for "bit-exact".
- Timing: measure each baseline once and reuse it; one run per arm; repeat
  only results inside 5 %·10 ms.
- Iterate as implement → second-reviewer review → fix. No per-item model
  windows; one combined pass at the end.
