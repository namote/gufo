# F3 narrow dense and output-slice batch, 2026-11-01

The batch is qualified: **pp2048 1405.25 ±.83 tok/s**, versus saved1400.15 ±.83
(+.36%). The exact collector payload is unchanged. The >=2000 tok/s objective
remains open; these local/operator gains did not produce a large model gain.

## Changes selected for qualification

- K5 singleton N640/K2560 dense output uses the 128×128, 4×2-wave, K32 tile
  for 65–2048 token rows. Other bitrates, grouped calls, residual epilogues,
  and larger rows retain their prior dispatch. Accumulation still advances
  through K16 WMMA operations in the same order.
- A singleton output slice aligned to 128 channels computes only that slice.
  Its trellis K-stride remains the parent width; the former reserved descriptor
  word carries that width without changing the 40-byte descriptor ABI.
  Input rotation remains the parent's. Paired q/k outputs still reuse the
  complete parent projection, and unaligned slices retain the existing path.
- Assertion-enabled tests compare old 64-row GPU chunks with singleton rows
  64/65/2048/2049, check allocation guards, and cover prefix, suffix, unaligned
  and paired output slices. No large CPU matrix multiplication was added.

The reviewer approved the dispatch, descriptor offsets, parent K-stride,
Hadamard boundaries, and formatted production/test patch.

## Small operator screening

All three windows below are COMPLETE/health_PASS. Every candidate checked
finite exact outputs and initialized allocation guards before and after timing.
Saved controls were reused; newly needed shapes were measured and frozen once.
[Full values and provenance](F3-PREFILL-BATCH2-SCREENS.json).

| Mechanism | Matched operator result | Decision |
|---|---|---|
| K5 singleton narrow tile | R512 .104957→.074289 ms (−29.22%); R2048 .255579→.216450 (−15.31%); R4096 .428508→.436127 (+1.78%) | Qualify only through 2048 rows |
| Routed operand transpose and contiguous vector stores | Gate +2.18%; down +.01%; removing zero scaling did not improve the aggregate | Preserve prototype, exclude from production |
| Reuse full 64 KiB LDS for K7 dense epilogue | Down/up shapes +.08%/+3.03%; XOR swizzle and combinations were also slower | Preserve prototype, exclude |
| Wave64 routed WMMA core | Gate +1.08%; down −.21%; wider hot workgroups +8.92%/+9.45% | Preserve prototype, exclude |

The routed references are the previously measured **1172-slot** current-capacity
controls: gate2.979682 and down3.198049 ms. Some screen logs also print the older
832-slot control timings alongside cached output identities; those are not the
denominators of the reported performance changes. Mixed negative/zero/positive
down exponents were additionally checked untimed in the store experiment.

The epilogue K7 controls were first measured once in that window using a
verbatim copy of the original production template. Wave64 only exercised raw
routed Mode3; its dense Hadamard epilogue is not qualified. Wave64 restored
32-lane subgroup trellis decoding and used four accumulators per lane, and
exact saved-output comparisons passed. None of these timings establishes
whole-model speedup or quality by itself.

Artifacts remain under `build/f3-prefill-{store,epilogue,wave64}` in the F3
worktree. Receipts are `fn-prefill-{store,epilogue,wave64}-screen-nix101` under
the usual `ai.tools/tmp/test-results/gufo-upstream-port-20261023` root.

## Build and validation provenance

Pinned Nix ROCm10.1, Clang24/GCC16.2, gfx1151. Persistent release and unit trees
rebuilt 19 and 22 targets; the prior 13 affected objects, old images and binaries
were preserved. Pinned clang-format21.1.8 passed all 758 C++ files.

- As-built revision: `0b5d58a3-prefill2-f497677d`.
- Source patch SHA256: `f497677d913202d10f765773fb6d8339108243e907c699c07736593955b492d3`.
- Image: `ai-tools/gufo-eval:f3-prefill2-f497677d-rocm101-r1`.
- Image ID: `sha256:d256d2ab05575cf66f9dc771d188cfdb7020e313c6bd53f1ca0259d752e0e356`.
- Installed ELF SHA256: `a25da47618d3315be124004f8360f408ce0cb928d49f377f6e7b451fce30bb3c`.
- Evidence: `build/f3-nix-incremental/{BATCH2-SOURCE.*,batch2/*}`.
- Quality receipt: `fn-prefill-batch2-nix101`; remaining benchmarks:
  `fn-prefill-batch2-bench-nix101`.

The window runs units, one 32,760-record collector with 64 head replays, saved
payload verification, then pp512/2048/8192 and d32K pp2048 at r2. No old baseline
inference, new model profiling, TG/MTP expansion, or overlapping build occurs.
The unit child maps confirm patched Nix ROCr10.1 and CLR10.1; installed binary
hashes match the staged image. Completed TG/MTP code stays intact.

Unused unshared Docker packaging-cache records reclaimed4,719,734,784 bytes;
the manifest is `build/f3-nix-incremental/UNUSED-CACHE-CLEANUP-20261101.json`.
No Nix volume, incremental object tree, baseline receipt or image was deleted.

## Qualified model results

KVarN k4v4, speculative off, temperature0, seed314159, r2. Stored matched
baseline reused. [Machine-readable qualification](F3-PREFILL-BATCH2-QUALIFICATION.json).

| Case | Saved baseline tok/s | Candidate tok/s | Change |
|---|---:|---:|---:|
| pp512 | 1061.85 ±.37 | 1062.82 ±.08 | +.09% |
| pp2048 | 1400.15 ±.83 | 1405.25 ±.83 | +.36% |
| pp8192 | 1442.28 ±1.13 | 1446.11 ±1.33 | +.27% |
| pp2048 at d32768 | 1280.80 ±3.23 | 1282.89 ±2.18 | +.16% |

32,760 records, 64 head replay checks, and 101,424,960 payload bytes match saved
q6 SHA256 `789c46a2a6939dc980e23b2cbf78bac71ccbcd9b1472899be846a1d56584f2a0`.
The prior hc7-relative NLL drift +.006600 remains; this is not BF16 parity.

The first window passed units, collector and exact verification, then stopped
before bench because the packaging wrapper was 0644. Only its mode was changed
to 0755; all installed ELF hashes remained identical. CLI smoke passed, and
only the two unexecuted bench cases resumed in the second healthy window.
No completed validation was repeated. The first receipt also read an old ignored
host `.gufo-revision` marker; installed hashes and its image source manifest
identify the actual batch2 binaries. The marker was corrected before benchmarks.
The original failed-window receipt/image remains preserved, not relabeled.
