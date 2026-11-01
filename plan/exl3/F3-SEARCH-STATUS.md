# F3 prefill search status, 2026-11-01

The qualified result is **1429.13 ±1.13 tok/s at pp2048**. The target of
2000 tok/s is **not reached**: it would still require about 40% more throughput,
or 29% lower latency. No claim of hardware impossibility is made.

The current continuation has screened the concrete matrix mechanisms supported
by the existing profiles and retained the qualified changes. The remaining
ideas below are research leads, not known improvements waiting to be enabled.
Do not repeat the rejected experiments unchanged or collect another full-model
profile merely to rediscover the same bottlenecks.

## Retained and qualified

1. K5 singleton N640/K2560 narrow tile through 2048 rows.
2. Aligned singleton parent-output slicing with the original trellis K-stride.
3. K7 singleton 6144→2560/R2048 token-first tile traversal.

4. Early shared EXL3 MLP beside router/routed MoE, with dedicated operands.
5. KVarN register selection removes80B private arrays without changing arithmetic.

Starting saved batch1400.15→final1429.13 is+2.07%; the stream/private-memory
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
[hardware know-how](../../docs/GFX1151-KNOWHOW.md). They are not new leads.

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
  route to the remaining 29% latency reduction.
- Single-launch grouped routed gate/up remains shelved with an unmeasured
  10–20ms-per-pass estimate; this differs from the measured two-stream screen.
  Independent expert input rotations limit operand reuse.
  HC combine/mix gather latency also remains incompletely explained. Neither
  has a demonstrated fix approaching the additional 409 ms per pass required;
  they are not represented as measured failures or as solved problems.
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

No new full-model profile or old baseline inference ran. The batch2 collector
ran once; its unchanged 512-row scope was reused for the final grid change,
whose 2048-row path was checked against saved outputs through the actual release
object. A packaging permission fault was fixed without rerunning completed
checks. Only unexecuted benchmarks resumed.

Nix ROCm10.1 incremental release/unit trees, changed-object predecessors,
images, controls and receipts remain intact. Only identified unused unshared
Docker packaging-cache records were reclaimed: 4,719,734,784 bytes. The root
filesystem remains above the 150 GiB window floor. No push or deployment.
