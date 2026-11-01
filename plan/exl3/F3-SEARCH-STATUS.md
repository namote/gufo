# F3 prefill search status, 2026-11-01

The qualified result is **1416.51 ±0.82 tok/s at pp2048**. The target of
2000 tok/s is **not reached**: it would still require about 41% more throughput,
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

Starting saved batch 1400.15→final 1416.51 tok/s is +1.17%. Other qualified
prefill cases are pp512 1063.05, pp8192 1447.47 and d32K pp2048 1291.03 tok/s.
All existing TG/MTP work remains. [Qualification and accuracy scopes](F3-PREFILL-GRID.md).

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

- Stream-K remains unimplemented. It changes reduction structure and needs a
  fresh numerical/quality argument; existing full-grid/persistent results do
  not establish a useful scheduling tail to recover. It is not a validated
  route to the remaining 29% latency reduction.
- Paired gate/up launches remain shelved with an unmeasured estimate of
  10–20 ms per pass. Independent expert input rotations limit operand reuse.
  HC combine/mix gather latency also remains incompletely explained. Neither
  has a demonstrated fix approaching the additional 422 ms per pass required;
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
