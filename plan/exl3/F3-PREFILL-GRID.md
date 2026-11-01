# F3 dense traversal follow-up, 2026-11-01

Qualified result: **pp2048 1416.51 ±.82 tok/s**, +.80% versus batch2 and
+1.17% versus the saved starting batch. The actual release object passed exact
saved-output/guard checks. The >=2000 tok/s objective remains unmet.

## Mechanism and complete shape coverage

For the singleton K7 6144→2560 projection at 2048 rows, traversing token blocks
before changing output-channel tiles reduced the operator from 1.930083 to
1.639585 ms (−15.05%). This changes output-tile order, not K16 accumulation,
rotation, scales, or storage. Production eligibility is restricted to the
256×256 kStore tile, grouped descriptor mode1, one matrix, and those exact
K/N/row dimensions. Other paths retain their prior traversal.

Here the row predicate uses the grouped launch's `max_rows`, before loading
the descriptor's row count. Flash-Next sets every descriptor's rows to that
same value. A generic padded launch still has a bijective tile mapping and
masked stores; its performance is outside this measured scope.

Actual trunk pack metadata contains 300 K7 parents across seven shapes:

| K→N | Parents | Projection | Groups |
|---|---:|---|---:|
| 640→2560 | 48 | Shared down | 1 |
| 2560→512 | 24 | Attention K/V | 2 |
| 2560→640 | 96 | Shared gate/up | 2 |
| 2560→6144 | 36 | GDN Z | 1 |
| 2560→10240 | 36 | GDN QKV | 1 |
| 2560→12288 | 12 | Attention Q | 1 |
| 6144→2560 | 48 | GDN/attention out | 1 |

All seven shapes were screened in two short operator windows. Existing controls
were reused; the four previously unmeasured shape/group combinations received
one saved control each. Equivalent full-group traversal sizes were omitted in
the remaining-shape batch. Mapping was independently checked as a bijection,
including partial row groups. All candidates passed exact outputs and guards.

Best other packed alternatives: 2560→6144 +2.37%, 640→2560 +1.49%,
2560→10240 +4.23%, 2560→12288 −.48%, grouped512 +.57%, grouped640 −1.74%.
Only the large, distinct 6144→2560 signal is promoted; the others do not justify
additional shape-specific dispatch branches. Saved whole-model baselines were
never rerun. [Every measurement and provenance](F3-PREFILL-BATCH2-SCREENS.json).

## Cached-weight branch rejected

The previous reconstruction benchmark used an older FP16 consumer. A separate
screen fed raw decoded FP16 W′ into the current X3 consumer, keeping K16 order
and H128/svh exactly. Original traversal lost 57.30%/7.31% on the two large
shapes. Even after traversal tuning, their best cached alternatives still lost
17.82%/2.70%. The small 640→2560 shape improved at most 4.55%, before cache
initialization. Expanding all 300 trunk K7 parents would add 5.42 GiB and multiply
weight-fetch bytes by 16/7. No model cache was added.

## Qualification scope

- Batch2's one 32,760-record collector with 64 head replays remains exact and is
  reused. Its explicit batch 512 cannot execute the new 2048-row predicate.
- The final release `prefill_fp16.hip.o` is linked directly into a small
  harness. It calls the public grouped API once per saved K7 shape, untimed,
  and checks finite/nonzero exact bytes plus output guards. This is the new
  2048-path correctness evidence, separate from the batch 512 collector.
- The final window combines that check with production d0 and d32K prefill
  benchmarks at r2. No new full collector or baseline inference runs.
- The harness's Ninja graph tracks the production object as a dependency;
  runtime maps and installed ELF hashes identify the exact tested artifacts.

As-built source `0b5d58a3-prefill3-67a50024`, patch SHA256
`67a50024ddc3b6b00ad99809753899f68934065f6d982864059a55f66b47664e`.
Image `ai-tools/gufo-eval:f3-prefill3-67a50024-rocm101`, ID
`sha256:31fd777e84c185efbe8c55926cf46a48085d4d2a6bb3493bbf0cb3451f693bb3`.
Persistent release/units rebuilt only 3/4 targets. All 758 formatted C++ files
passed pinned clang-format21.1.8. Prior objects, images, source patches and
receipts remain preserved. No new model profiling or TG/MTP tuning occurred.

Artifacts: `build/f3-prefill-{predecode,grid,grid-rest}`,
`build/f3-prod-gridcheck`, `build/f3-nix-incremental/batch3` in the F3 worktree.
Receipts: `fn-prefill-{predecode,grid,grid-rest}-screen-nix101` and
`fn-prefill-batch3-nix101` under the usual test-results root.

## Final model measurements

Nix ROCm10.1, KVarN k4v4, speculative off, temperature0, seed314159, r2.
[Qualification identities and exact scopes](F3-PREFILL-GRID-QUALIFICATION.json).

| Case | Saved starting batch | Batch2 | Final batch3 | Total change |
|---|---:|---:|---:|---:|
| pp512 | 1061.85 ±.37 | 1062.82 ±.08 | 1063.05 ±.22 | +.11% |
| pp2048 | 1400.15 ±.83 | 1405.25 ±.83 | 1416.51 ±.82 | +1.17% |
| pp8192 | 1442.28 ±1.13 | 1446.11 ±1.33 | 1447.47 ±1.36 | +.36% |
| pp2048 at d32768 | 1280.80 ±3.23 | 1282.89 ±2.18 | 1291.03 ±1.09 | +.80% |

`fn-prefill-batch3-nix101` is COMPLETE/health_PASS. Production-object correctness
check took less than one second; the two performance cases ran once at r2.
The reused collector retains prior hc7-relative NLL drift +.006600; no BF16
parity claim is made. No additional TG/MTP tuning or validation was performed;
prior work and results are retained.
