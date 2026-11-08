# F3 prefill search status, 2026-11-02

User priority moved to TG/MTP after freezing this prefill phase on November2.
The qualified follow-up removes routed verifier LDS and vectorizes exact HC320
loads. See [TG/MTP qualification](F3-TGMTP-QUALIFICATION.json) and
[full execution-flow review](F3-TGMTP-FLOW.json). The prefill figures and rejected
research below remain preserved history; do not repeat them as a new TG task.

The qualified result is **1435.59 ±.07 tok/s at pp2048**. The target of
2000 tok/s is **not reached**: it would still require about 39% more throughput,
or 28.2% lower latency. No claim of hardware impossibility is made.

The current continuation has screened the concrete matrix mechanisms supported
by the existing profiles and retained the qualified changes. The remaining
ideas below are research leads, not known improvements waiting to be enabled.
Do not repeat rejected experiments unchanged. A user-requested profile of the
pre-batch6 shared5 binary now replaces the older profile as the current bottleneck
map: main-stream MoE matrices32.31%, HC-family mixing/projections (including fused MoE combine)22.74%, dense EXL3
matrices20.62%, with99.676% kernel-active time (not ALU utilization). The shared side stream is already
mostly overlapped. Reuse [this profile](F3-CURRENT-PREFILL-PROFILE.json); no
extra baseline or deep-context profile was collected.

## HC/MoE combined qualification, November 2, 2026

HC operand-load grouping and K5 Mode3 dead-subtile/K-tail staging are now
qualified together on pinned Nix ROCm10.1. Dense K7 K-tail removal is not
selected. Current image: `ai-tools/gufo-eval:f3-prefill6-f15434b2-rocm101`,
as-built `1dbee4d6-prefill6-f15434b2`.

| Case | Saved shared5 tok/s | Qualified tok/s | Change |
|---|---:|---:|---:|
| pp512 | 1068.58 ±3.09 | 1078.15 ±.40 | +.90% |
| pp2048 | 1429.13 ±1.13 | 1435.59 ±.07 | +.45% |
| pp8192 | 1460.97 ±.73 | 1466.18 ±1.12 | +.36% |
| pp2048 at d32768 | 1301.99 ±1.41 | 1311.98 ±1.03 | +.77% |

One fresh32760-record/64-head-replay FP16-KV collector is byte-exact to the
saved Nix baseline; affected full-FFN1025/shared-state and expert/final-tail
checks pass. Performance uses k4v4/temp0/seed314159/r2. The initial pp2048
1342.22±103.86 was inconclusive. A shortened-list replay inadvertently changed
context8193→4096 and is diagnostic only; the final d0 replay restored the
original512,2048,8192 list and8193 capacity. All attempts are retained; no
baseline, correctness or d32K rerun. Target2000 is still unmet.
[Qualification, runtime/code identities and all attempts](F3-PREFILL-BATCH6-QUALIFICATION.json).

Dense K7 K-tail removal is the last screened candidate and is not selected;
its largest QKV shape is+5.98% against the saved control. All six outputs pass.
[Audit/measurements](F3-DENSE-KTAIL-SCREEN.json). HC DPP and gate-load hoisting
are closed leads (neutral/already present). MoE staging and HC load grouping
are now applied and qualified; no pending full-model pass remains for them.

## Retained and qualified

1. K5 singleton N640/K2560 narrow tile through 2048 rows.
2. Aligned singleton parent-output slicing with the original trellis K-stride.
3. K7 singleton 6144→2560/R2048 token-first tile traversal.

4. Early shared EXL3 MLP beside router/routed MoE, with dedicated operands.
5. KVarN register selection removes80B private arrays without changing arithmetic.
6. HC operand loads grouped with original restore rounding.
7. K5 Mode3 dead B staging and invariant K-tail removal.

Prior to batch6, saved batch1400.15→1429.13 was+2.07%; the stream/private-memory
batch alone improves1416.51→1429.13(+.89%). Other final cases: pp5121068.58,
pp81921460.97 and d32Kpp20481301.99. TG/MTP work remains.
[Final selection and exactness scopes](F3-STREAM-QUALIFICATION.json).

## Evidence against the tested alternatives

| Mechanism | Evidence | Disposition |
|---|---|---|
| K32 stages, smaller waves, masked wide tiles | Most dense/routed cases lost; narrow singleton exception qualified | Keep exception only |
| Tight MoE capacity and cold/hot descriptor split | Weighted operator estimate +.32%/−.12%, before builder cost | No extra dispatch |
| Routed operand transpose/vector stores; scale removal | No useful aggregate gain, exact outputs | Preserve prototypes |
| Wave64 routed core | Gate +1.08%, down −.21%; wider hot groups worse | No promotion |
| Dense full-LDS output staging/XOR swizzle | No gain on large shapes, exact outputs | No promotion |
| Cached raw FP16 weights in current X3 | Even tuned traversal lost 17.82%/2.70% on large shapes; would add 5.42 GiB | No model cache |
| Dense traversal across all seven actual K7 shape/group cases | One clear 15.05% local exception; others slower or small | Keep qualified exception |
| Compact 32/64/128 and16/32/64/128 MoE classes | Weighted 2×gate+down +.89%/+3.21%, including launches but excluding builder | No promotion |

All latest operator measurements, control identities, exactness checks and
runtime fingerprints are in [the screen ledger](F3-PREFILL-BATCH2-SCREENS.json).
Earlier persistent scheduling, wider routed tiles, deeper stages, prefetch,
double buffering, fragment hoisting, direct-global B and scheduling hints are
already recorded in [session findings](F3-SESSION-FINDINGS.md) and
[hardware know-how](../../../docs/GFX1151-KNOWHOW.md). They are not new leads.

The final cold-class screen was COMPLETE/health_PASS and used the unchanged
frozen routing. Live bucket counts were 127/56/62/155 for 1–16/17–32/33–64/65–128
rows. Launch bounds were 512 per cold class and 315 hot. Partitioning retained
each descriptor's expert, first row and count; every launch was timed. Exact
finite outputs and guards passed. Three-class times were 3.027125/3.184357 ms
for gate/down; four-class 3.115747/3.220175, versus saved 2.979682/3.198049.
Neither can become a win by adding the omitted device partition-builder cost.

## Open research boundaries

- **EXL3 stream schedules have whole-model qualification.**
  Shared/routed MoE is retained. Q/KV alone is neutral and its combination
  with shared overlap regressed; dedicated events did not fix it. GDN Z alone,
  smaller-LDS Z, heterogeneous grouping and routed gate/up have weak or negative
  evidence. See the [stream ledger](F3-STREAM-OVERLAP.md). GDN alpha/beta and
  broader dependency schedules remain separate research leads.
- Stream-K remains unimplemented. It changes reduction structure and needs a
  fresh numerical/quality argument; existing full-grid/persistent results do
  not establish a useful scheduling tail to recover. It is not a validated
  route to the remaining 28.2% latency reduction.
- Single-launch grouped routed gate/up was screened on Nix10.1. Launches
  fell from four to two, but the shared-side operator span rose 0.90% at2048
  and1.15% at512; exact output checks passed. No promotion; retain prototypes.
  See [the measured screen](F3-GROUPED-MOE-SCREEN.json). HC combine/mix gather
  latency remains incompletely explained; no demonstrated fix covers409ms.
- A new matrix algorithm or instruction schedule could still improve things.
  Existing measurements show WMMA/VALU competition and LDS/core limits; neither
  a new kernel nor a universal hardware limit has been proved here.
- Precision-changing designs need their own quality evidence. The earlier
  int8 HC attempt failed its accepted quality bound; the earlier GDN scan
  regressed greedy/tool behavior. Neither is silently reintroduced.
- Higher-memory cache designs are not prohibited in principle; the tested
  exact FP16 cache is rejected on measured performance, not on an invented
  user memory constraint. Likewise, no claim covers every possible library
  version, future compiler or different hardware.

## Why routed throughput is below the55.3 WMMA reference

[Offline calculations and routing provenance](F3-MOE-CEILING-DIAGNOSIS.json).
No new GPU run or kernel change was made for this diagnosis.

- **Different denominators:**55.34TFLOPS is a register-fed FP16 probe on
  ROCm7.2.4. Current Nix10.1 gate/up spends303.137ms on6.31662e12 useful FLOPs,
  giving20.838TFLOPS. The earlier21.253 estimate assumed48 full2048-row FFNs;
  the final FFN actually retains128 rows, as executor pruning and the profile
  confirm. Count47 full layers plus128 final rows. Its duration includes decode, data movement, padded work,
  scheduling and shared-side contention. The quotient37.7% is not ALU usage.
- **Less reuse in routed tiles:** the cold128x64 tile has one A and four B
  fragments per wave,1.25 LDS fragment reads/WMMA at full occupancy of its
  token tile; hot128x128 has1.125. Historical LDS-only1.25 probes reached39–44,
  below the register peak even before decode/global traffic. Partial buckets
  further reduce reuse. These microbenchmarks do not establish a current
  complete-kernel ceiling. Main dense large shapes reach35.9–38.6 usefulTFLOPS
  in the current profile, consistent with better reuse but not a causal test.
- **Ragged inputs leave preparation work:** current code skips wholly dead
  16-token WMMA subtiles, but still fetches/stages all64/128 B rows (padding
  clamps to the last valid row) and decodes the full weight tile per bucket.
  The saved synthetic routing has20,480 useful rows,23,648 executed WMMA rows
  (+15.47%) and35,520 B staging slots (+73.44%). This is logical work, not
  DRAM bytes or current model-wide percentages; repeated padded loads may hit
  caches. The current trace has no bucket counts; launch bounds cannot supply
  them. The historical dbg-counts.json is logits metadata, not that histogram.
- **Decode competes for issue resources:** historical WMMA/VALU probes and
  routed ablations support this mechanism. Removing decode changed3.6→3.25ms;
  removing decode+weight fetch changed3.6→2.85ms. These are old operator
  interventions, not current causal percentages; their deltas are not additive.
  Neither “decode costs23%” nor a claimed2.05ms floor is a measured current limit.
- **Not an established spill or CPU-gap issue:** audited matrix specializations
  have zero private bytes and zero spill counters. The current forward is
  99.676% kernel-active, leaving4.65ms kernel gaps. Active time is not occupancy;
  instruction/memory stalls and side-stream contention remain unquantified.
  Side-stream temporal overlap of340.76ms is not net saved time.

### Ordered improvement plan

Steps1–5 below are complete for the current bounded batch. The combined
qualification above includes its surviving HC and K5 staging candidates.

1. **Complete the work accounting offline first.** Reuse the saved release ISA,
   operator outputs/routing, current timeline and existing ablations. Report
   useful versus executed WMMA rows, B staging and decode work separately by
   cold/hot class. Inspect whether dead-row fetch/store guards and loop-invariant
   index arithmetic survive in emitted ISA. Static instruction counts are not
   stall counters. Do not repeat the full-model profile. If exact production
   bucket counts become necessary, capture them with the next already-needed
   model validation, outside its timing region, not another standalone sweep.
2. **First candidate: omit wholly dead B subtiles inside the existing kernels.**
   Keep the existing64/128 classes, grid and K16 accumulation order. Predicate
   global fetch and LDS store only for16-row subtiles never consumed; preserve
   the last live subtile's padding and WMMA pairing. Inspect branch/address
   overhead and generated resources before running. This differs from rejected
   compact-class dispatch (more launches/building) and direct-global B (removes
   LDS reuse). Preserve all rows below ceil(count/16)*16; row XOR(ks&3)
   stays within that16-row group. Guard B fetch/store only; A production and
   block barriers remain unconditional. Requires proof every live LDS read is
   initialized. Reject if
   compiler output adds cost without removing loads/stores. On the saved
   synthetic routes, this can omit11,872 of35,520 staged slots (33.42%); the
   final live subtile padding remains. Neither this ratio nor the73% excess
   over useful rows is an operator speedup promise.
3. **Second candidate, conditional on ISA: simplify trellis load/address work.**
   Focus on duplicated address construction, load scheduling and value lifetimes
   in K5 cold/hot, retaining exact decoded halves and each output's K order.
   The hash's two-op form is already optimized. Do not re-run deeper stages,
   double buffers, private decode, fragment hoists, broad tile grids or the
   FP16 cache unchanged. If ISA exposes no distinct removable work, stop this
   candidate rather than manufacture another tuning sweep. Transfer a proven
   mechanism to dense X3 only after its own shape/resource inspection.
4. **One bounded operator diagnostic batch for surviving candidates.** Reuse
   cached Nix10.1 controls only with matching compiler, harness, allocations,
   routing, input/output and cache/concurrency history. Include exact outputs,
   guards, ragged16-row boundaries and both gate/up and down; preserve existing
   shared-side competition. Compare complete paired span, not isolated main
   speed. Synthetic operator evidence screens candidates; it is not model
   qualification. If a new diagnostic condition has no matched cached control,
   measure just that condition once and save it. Only if remaining attribution
   changes the decision, add matched small no-decode/no-B-stage interventions
   and register/LDS calibration to this same batch; never treat ablated outputs
   as correctness evidence or sum intervention deltas into a budget.
5. **Batch promotion and final validation.** Incremental Nix10.1 builds and
   controls stay preserved. Combine passing exact changes into one final
   accuracy/prefill performance window; reuse the qualified baseline wherever
   matching, and rerun only affected failures. Accept neutral simplifications
   when quality and performance are preserved; do not add neutral complexity.
   Record rejected candidates too. No additional TG/MTP optimization in this
   work; retain completed implementations. Keep the150GiB disk floor by deleting
   only identified unused cache, never needed controls or incremental objects.

The gate/up path is21.13% of the1434.41ms profile span: reducing its time20%
would remove60.63ms (4.23% of span) if other costs stayed fixed. All routed
matrices20% lower saves92.69ms; routed+dense20% lower saves151.83ms. These are
budgets, not predictions under overlap. Even the impossible no-overhead55.34
reference puts gate/up at114.14ms, saving189.00ms, less than the409.04ms needed
for2000tok/s from the qualified1429.13. Therefore retain HC-family mix/combine
and projections (326.25ms, including fused routed output combine) as the next
major target if matrix changes plateau; do not claim a matrix-only path to the
whole goal. No fixed achievable ceiling or promised speedup is established.

## Register pressure and kernel fission

The actual preserved release has now been audited: all79 EXL3 matrix variants
have zero private bytes and both spill counters zero. KVarN's80B addressable
arrays were removed and qualified; selector18SGPR spills/private0 and unrelated
fused attention/Q4_K exceptions remain explicitly scoped. See the
[actual release audit](F3-RELEASE-RESOURCE-AUDIT.json). Do not repeat it unchanged.

This is an explicit candidate direction, not a diagnosis that the current
prefill bottleneck is spilling. The saved Nix ROCm 10.1 stage-screen assembly
contains 13 matrix specializations with 89–214 VGPRs and **zero private frame,
SGPR spills and VGPR spills**; see [extracted metadata](F3-REGISTER-AUDIT.json).
This is a static inspection of preserved operator artifacts, not a new build,
profile or performance run, and not proof the final image is globally spill-free.

Private frame bytes, register counts and spill counts are distinct fields in
[LLVM's metadata](https://llvm.org/docs/AMDGPUUsage.html#code-object-v3-metadata).
Private memory can also hold addressable local arrays; zero private bytes does
not exclude scalar-register spilling into vector registers. Use ISA and hot-loop
location as well as metadata, rather than treating any one count as traffic.

Already retained: GDN named q/k vectors replaced a 112-byte private array;
bounded selector-head loads reduced SGPR spills from 304 to 18 with private
scratch still zero on Clang 24; bounded sparse-QK scheduling reduced VGPRs
227→221 without spills. Conversely, cached-FP16 smaller consumer tiles spilled
and lost performance, and most tested K32/smaller-wave matrix tiles also lost.
These existing results must not be repeated unchanged.

Candidate order:

1. Inspect preserved release objects for hot prefill kernels not covered by the
   saved resource inventory. If hot-loop spill/private traffic exists, shorten
   value lifetimes, bound unrolling/prefetch and scalarize addressable arrays
   before splitting launches. Do not force a lower VGPR limit blindly.
2. Where accumulator/temporary pressure is the cause, partition independent
   output channels or token rows, or process smaller groups sequentially inside
   the kernel. Preserve each dot product's K order and complete output rotation,
   normalization and recurrence dependencies. Smaller tiles with fewer waves
   can increase per-thread pressure; verify generated resources for the exact
   shape and compiler. Merely dividing the launch grid while keeping the same
   kernel does not reduce its per-thread register requirements.
3. Consider splitting fused stages only when the removed live ranges/spill work
   can repay extra launches, synchronization and intermediate global reads/writes.
   Count any partial-output/rotation pass in the complete operator span.
4. Evaluate resource fit together with [stream candidates](F3-STREAM-OVERLAP.md):
   a spill-free kernel may still occupy enough LDS/registers to prevent useful
   co-residency. A new paired workload is a distinct reason to revisit a layout,
   not permission to repeat the old standalone sweep.

Split-K/Stream-K changes the reduction organization and belongs to the separate
numerical-quality scope above. Splitting recurrent time into the previously
rejected approximate scan is not an exact spill fix. Screen genuine new changes
with small operator/state checks; retain controls and batch final prefill
qualification. No speedup is claimed by this audit.

## Reuse and preservation

This offline ceiling diagnosis ran no GPU work. The earlier batch2 collector
ran once without a new full-model profile or old baseline inference; its unchanged 512-row scope was reused for the final grid change,
whose 2048-row path was checked against saved outputs through the actual release
object. A packaging permission fault was fixed without rerunning completed
checks. Only unexecuted benchmarks resumed.

Nix ROCm10.1 incremental release/unit trees, changed-object predecessors,
images, controls and receipts remain intact. Only identified unused unshared
Docker packaging-cache records were reclaimed: 4,719,734,784 bytes. The root
filesystem remains above the 150 GiB window floor. No push or deployment.
