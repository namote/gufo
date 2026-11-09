# C6 audit: KVarN rollback/requantization and cache retention/checkpoint/ledger

This audit reads the code only. Nothing was built or run, and no GPU or server
was used. It covers HEAD of `ai-tools/port-u4f` after 321431fb. Other agents
were editing in parallel, so line numbers are those at the time of reading.
Scope is CLEANUP.md item C6: (1) the KVarN k4v4/k6v6 path that rewinds over
rejected rows and requantizes the group boundary, in Qwen3.8-27B
(`src/models/qwen/hip`) and Flash-Next (`src/models/qwen38_flash_next`), and
(2) `src/cache/{retention,checkpoint,ledger}.cpp`, including the R1 change
(e60d9b22, 06e5d1b0, 4ac374c3). The design R1's reviews already settled is
not reopened here: pins, `AlignedTail` selection, exact replacement ignoring
pins, and `TryTouch`.

## Summary

- The audit found no high- or medium-severity bug. Today's callers cannot
  corrupt KV, leak a charge or free one twice through any path read here.
- There are four confirmed findings, all low severity. Two are KVarN invariants
  that 27B maintains more weakly than Flash-Next. Today's call sites cannot
  reach them, but nothing enforces that. The other two are an eviction-order
  inconsistency and a missing bounds guard in retention.
- There are two suspicions. Each needs a runtime check to settle.
- The rewind and requantization design itself is sound. See "Verified" below.

## Confirmed findings, ranked

### F1 (low, latent): 27B does not keep the ring-reach invariant that the KVarN rewind guard relies on

- **Where.** The guard is `src/core/hip/kvarn_layout.hpp:183-191`
  (`PlanKvarnPass`) and `:239-248` (`PlanKvarnPrefill`). It refuses a rewind
  only when `reopened + 256 < written_end`, which assumes the tail ring holds
  every FP16 row in `[reopened, written_end)`. 27B breaks that assumption in
  two places:
  - The 27B prefill store (`src/models/qwen/hip/kernels/attention_kvarn.hip:1261-1285`)
    quantizes each group that completes inside a chunk straight from the
    staging buffer. Only the rows of the incomplete last group go into the
    ring. `PrepareKvarnPrefill` (`src/models/qwen/hip/arena.cpp:1382-1383`)
    then still raises `written_end` to the chunk end.
  - The 27B restore sets `written_end = position`
    (`src/core/hip/kvarn_checkpoint.hpp:50-51`, used by
    `src/models/qwen/hip/arena.cpp:1100` and
    `src/models/qwen/continuation_adapter.cpp:1344-1355`). The ring holds only
    `[base, position)`.
- **Failure scenario.**
  1. A WMMA prompt chunk writes `[0, 1000)`. That leaves `flushed_end = 896`
     and `written_end = 1000`, and the ring holds only rows 896..999.
  2. Some later write starts at any `s` in `[768, 896)`.
  3. `PlanKvarnPass` reopens group 768 (`768 + 256 = 1024` is not below
     1000).
  4. The boundary flush then requantizes group 6 from ring rows 768..s-1 that
     this sequence never wrote. They are stale rows from an earlier request,
     or zeros.
  5. A prefill chunk would also copy those rows into staging and attend over
     them as FP16 context (`attention_kvarn.hip:1243-1253`).

  The restore case is the same. After a restore at `P = 1024` with
  `base = 1024`, a write at 1023 reopens group 896, whose ring rows were never
  restored.
- **Why it is not reachable today.** In 27B, the only write that starts below
  the committed frontier is the next pass after an MTP verification rejects
  drafts. Its start is at or above the committed position, which is at or
  above the prompt end or the restored position. So `floor(start)` is at or
  above the prefill `flushed_end` or the restored base, and every row from
  there on was written into the ring by a pass or restored. Hybrid (GDN)
  recurrent state also rules out a live truncation below the frontier. Any
  earlier start goes through a checkpoint restore, which resets the tail.
- **Flash-Next keeps the invariant explicitly.** Its prefill copies into the
  ring every row a later pass may rewind into
  (`src/models/qwen38_flash_next/runtime/executor_attention.cpp:493-509`). Its
  restore raises `written_end` to `base + 129` so that a rewind below the base
  is refused (`runtime/executor_snapshot.cpp:1129-1141`).
- **Suggested fix.** Choose one:
  - (a) After a 27B prompt chunk that quantized groups from staging, set
    `written_end = max(written_end, complete_end + kKvarnTailRingTokens -
    kKvarnGroupTokens + 1)`. Do the same on restore, as Flash-Next does. The
    guard then refuses exactly the rewinds it cannot serve.
  - (b) Copy the reach rows into the ring as Flash-Next does. This costs one
    extra copy per chunk.

  Option (a) is free and keeps the shared guard honest. Better still, move the
  restore rule into `KvarnRestoredTail` so both models share it.
- **Check that settles it (CPU).** In `tests/core/kvarn_layout_test.cpp`,
  apply the arena's tail update after `PlanKvarnPrefill(tail{128,0}, 0, 1000)`.
  Then assert that `PlanKvarnPass(tail, 800, 8)` returns an error. Assert the
  same for `PlanKvarnPass(KvarnRestoredTail(1024, rows), 1023, 1)`. Both return
  no error at HEAD.

### F2 (low, latent): the 27B write guard and unshare range miss the lagging group that a pass flushes first

- **Where.**
  - `src/models/qwen/continuation_adapter.cpp:208-216` guards records only from
    `floor(first)` and calls `UnshareKvRows(first, end)`.
  - `PrepareKvarnPass` and `PrepareKvarnPrefill`
    (`src/models/qwen/hip/arena.cpp:1358-1359,1380-1381`) first call
    `LaunchKvarnFlush` for groups `[flushed_end, floor(first))`. In 27B,
    `flushed_end` can lag one group behind the frontier:
    - A pass that ends exactly at a group end has no in-pass boundary.
      `PlanKvarnPass` then sets `flushed_end = target`. For example, a pass
      over 500..511 leaves `flushed_end = 384` at position 512.
    - 27B has no counterpart to Flash-Next's `FinishKvarn` group-end flush
      (`runtime/executor.cpp:1079-1087`).
  - So the next write at 512 quantizes group 3 into the block that covers rows
    256..511. Neither `UnshareKvRows(512, …)` nor the guard covers that block.
  - Flash-Next widens both ranges with `min(first, flushed_end)`
    (`src/models/qwen38_flash_next/continuation_adapter.cpp:142-145`) and also
    in `ReserveKvRows` (`runtime/executor.cpp:288-289`).
- **Why it is benign today.**
  - No retained checkpoint can hold that group as valid records. Its records
    end at `KvarnTailBase ≤ flushed_end` when it is captured. `flushed_end`
    only drops through a reopen to `floor(start) ≥ floor(P_checkpoint)`. The
    group being flushed starts at or above the current `flushed_end`.
  - A shared block can contain that group only through a checkpoint with
    `records ≥ group end`. A session restored from such a checkpoint has
    `flushed_end` at or beyond the group end, so it never flushes the group
    again.
  - The only remaining hazard is a write into a physically shared block. Such a
    write could only be byte-identical requantization: the quantizer is
    deterministic, and the FP16 rows are identical because they come from the
    same checkpoint.
- **Suggested fix.** Use `min(first, flushed_end)` in the 27B `BeforeWrite`,
  exactly as Flash-Next does. Alternatively, flush at group end in
  `PrepareKvarnPass` the way Flash-Next's `FinishKvarn` does. The second
  option also makes 27B checkpoints at group ends carry records instead of 128
  FP16 tail rows.
- **Check that settles it.** Add a CPU adapter test with a fake arena tail
  `{flushed_end=384}` at position 512. Assert that `BeforeWrite(512, 513)`
  guards or unshares from row 384, the start of the group being flushed.

### F3 (low): retention's `forked` flag depends on admission order and is never cleared

- **Where.**
  - `src/cache/retention.cpp:227-249` (`Relate`) sets `forked` only on the
    deepest retained prefix of the record being added. It does so only when a
    sibling that diverges from the new record is already retained.
  - It never evaluates whether the record being added is itself a fork, that
    is, whether two of its retained extensions diverge.
  - `Erase` (`:576-595`) clears the `below` bits but leaves `forked` set on
    every other record.
  - Exact replacement (`:686-688`, then `:706`) recreates the record with
    `forked = false`.
- **Failure scenarios.**
  - Lost fork. Records C1 and C2 diverge after a shared kHistory prefix P. If P
    is admitted after them, or is republished by exact replacement, then
    `P.forked == false`. `Branch()` recovers fork status only for
    continuation and branch-point purposes (`:163-184`). A history P therefore
    loses its `shared` and `fork` ordering in `PressureOrder` (`:288-298`) and
    can be evicted before the leaves that depend on it.
  - Sticky fork. After C2 is evicted, P keeps `forked == true` indefinitely.
    `PressureOrder` keeps ranking it as shared, and `AlignedTail` (`:528`)
    never selects it, even though nothing forks there any more.
  - Neither scenario leaks or corrupts anything. Only the eviction order
    differs from what the comments describe.
- **Relation to R1.** R1 started reading `t.forked` in `AlignedTail`, so the
  staleness now also affects R1's tail selection. Two concrete cases:
  - A tail whose diverging sibling was evicted is never replaced.
  - A tail re-admitted by exact replacement loses its fork protection.
- **Suggested fix.** Compute fork status on demand from the `below` bits:
  record P is a fork when two of its `below` peers are unrelated to each
  other. That is the same O(limit²) bit work `Relate` already does, over at
  most `kRecordLimit` records. Alternatively, recompute `forked` for the added
  record in `Relate` and for its old parent in `Erase`.
- **Check that settles it (CPU, `cache_retention_test`).** Admit C1 and C2 with
  a shared prefix, then admit P. Assert that `Rank` and the eviction order
  match the order C1→C2→P. Then remove C2 and assert that P is no longer
  treated as a fork.

### F4 (very low, hardening): `Admit` indexes `records_[victim]` without checking `victim != limit_`

- **Where.** `src/cache/retention.cpp:689-700`. Under record pressure, the
  victim comes from `Victim(r, true)`, which can return `limit_`. That would be
  out of bounds when `limit_ == kRecordLimit`.
- **Why it is not reachable today.**
  - `:646` and `:661` already returned when no victim exists.
  - The production caller (`src/cli/serve/component_text_cache.cpp:2446-2449`)
    holds the cache mutex across the whole `Admit`.
  - Its capture lambda returns a checkpoint built beforehand, so the
    ResourceExhausted retry loop runs only in tests. Any erase inside that
    loop lowers `size_` first.
- **Suggested fix.** Before `Erase(victim, …)`, add
  `if (victim == limit_) return refuse(RetentionReason::kRecordCapacity);`.
  This releases the new index entry the same way the other refusals do. Check
  that `index_.Insert` at `:684` is undone on that path; at HEAD, a refusal
  after `Insert` would leave a resident index entry with no record. The
  cheapest form of this fix is to move the victim check above `Insert`.

## Suspicions, to settle at runtime

### S1: the eviction order differs because 27B leaves 128 FP16 rows in checkpoints at group ends

A 27B checkpoint at a group end that a pass reached exactly has
`base = flushed_end = P − 128`. Its records stop one group short, and it
carries 128 FP16 tail rows: 128 × 2 planes × layers × kv_width bytes. A
Flash-Next checkpoint at the same point carries none. This is consistent and
restores correctly. However, it makes such 27B checkpoints larger, by about
1 MiB per group across the full-attention layers (estimate). It also changes
`UniquePayloadBytes`, which feeds the R1 half-size test in `AlignedTail`
(`retention.cpp:552-554`). Only 27B KVarN checkpoint payload sizes from a
`cache` functional run would show whether the effect matters. It does not
affect correctness. Fixing F2 the `FinishKvarn` way would remove it.

### S2: capture refuses instead of recapturing when a shared chunk has retired

`ExecutionHistory::Capture` (`src/cache/checkpoint.cpp:589-590`) and
`NewPayloadBytes` (`:701-702`) both throw when a predecessor chunk is still
alive but its borrowed storage is no longer valid. `Find` already treats an
expired chunk as a hole and recaptures it. A live but invalid chunk instead
blocks every later capture on that history until a restore replaces the
history. Static reading cannot show whether `BorrowedRows` can become invalid
while a checkpoint still holds the chunk, because that depends on the preserve
and failure paths in the adapters. If it can, a long conversation stops getting
checkpoints after one failed preservation. To settle it: count
`capture_failure` refusals in the `cache` and `pi_agent` functional logs, or
add a CPU test that invalidates a borrowed chunk and captures again.

## Verified: no defect found

### KVarN rollback and requantization (both models, K4V4 and K6V6)

- **A rollback that ends inside a group.** Nothing is rolled back eagerly. The
  tail keeps `flushed_end` past the committed position. The next pass sees
  `start < flushed_end` and reopens `floor(start)`
  (`kvarn_layout.hpp:182-192`, `:238-248`). That group's stale record is never
  read meanwhile:
  - `KvarnReadGroups` depends only on the position
    (`kvarn_layout.hpp:135-138`), so positions in the reopened group never
    read it.
  - Checkpoints use `KvarnTailBase = min(flushed_end, floor(P))`
    (`kvarn_checkpoint.hpp:26-39`), so they exclude it and take its committed
    rows from the ring.

  The group's scale and zero data are not separate state. They live in the
  record and are rewritten whole when the group is requantized.
- **Requantization after a rollback compared with a cold fill.**
  - All paths that quantize a group call the same
    `kvarn_detail::KvarnQuantizeTile`:
    - the ring flush before a pass;
    - the boundary flush inside a pass;
    - the group-end flush in Flash-Next's `FinishKvarn`;
    - the prefill quantize from staging.

    The call sites are `attention_kvarn.hip:150-167,1062-1085,1261-1272` and
    `attention_prep.hip.cpp:254-303,578-626`.
  - That tile runs with `fp contract(off)`, fixed reductions and an LDS
    `atomicOr` that is order-independent (`kvarn_device.hpp:274-300,640-644`).
    So identical FP16 rows give byte-identical records, whatever the launch
    shape.
  - The only intended difference from a cold fill: a WMMA prompt chunk attends
    to its own rows as FP16, while a decode or verification row at the same
    position reads completed groups quantized. This is documented in
    `ops/kvarn.hpp:80-93` and in GFX1151-KNOWHOW ("chunk-matched fake quant is
    authoritative"), and is the reason CLEANUP leaves restore==cold alignment
    alone.
  - FP16 rows of accepted drafts come from the multi-row verification GEMMs,
    not the AR GEMV. That is the general MTP numerical difference, not a KVarN
    one.
- **Ring wrap.**
  - A pass writes at most 128 rows, and `end − floor(start) ≤ 256` is
    enforced. Groups before `floor(start)` are flushed before any row is
    written.
  - So a pass never overwrites a ring row that is still needed: the reopened
    group, the lagging group, or the committed tail.
  - `written_end` never decreases, so the guard is conservative after a
    rejection. It never refuses a legitimate MTP rewind, because
    `max end ≤ floor(earliest start) + 256 ≤ floor(start) + 256`.
  - The 27B snapshot path refuses ring rows that have already been overwritten
    (`arena.cpp:282-292`).
  - The staging copy in the 27B prefill (`start − target < 128` rows from a
    group-aligned ring row) never wraps.
- **Snapshot and restore of a partial group.**
  - `KvarnRestoredTail` requires a group-aligned base, a tail of at most 256
    rows and an exact `KvarnRowsAt` match (`kvarn_checkpoint.hpp:42-52`).
  - The 27B component restore also checks the header version,
    `flushed_end` and `written_end == position`
    (`continuation_adapter.cpp:1340-1354`).
  - Flash-Next's restore rebuilds the tail from the group floor, which
    `FinishKvarn` and the prefill guarantee are present in the records.
  - The exception is the F1 weakness, in 27B only.
- **MTP verification commit.**
  - Rows after the boundary read the group as quantized from the draft rows
    before it (`attention_kvarn.hip:1148-1172`; the Flash-Next boundary kernel
    takes start from the device, `attention_prep.hip.cpp:279-303`). This is
    correct, because a row is accepted only if every earlier row was.
  - Rejection leaves the tail for the next pass to reopen.
  - The MTP head's own KV is FP16 in both models (27B `mtp/executor.cpp` has no
    KVarN path; for Flash-Next, see KNOWHOW "FP16 MTP KV").
  - Batched verification calls `PrepareKvarnPass` per sequence after the
    `ContinuationWrite` guards (`batched_decode.cpp:606-630`).
- **K6V6.**
  - The token-major record layout includes 6-bit keys.
  - The read path indexes the low-nibble words and the high-2-bit plane
    correctly (`kernels_internal.hpp:324-378`).
  - Rollback uses the same tile and plans as K4.
- **Failure inside a pass.** `PrepareKvarnPass` advances the tail before the
  layers run. If a pass throws partway, `WriteFailed` marks the slot invalid,
  and its reset clears the tail (`arena.cpp:1955`). No later pass trusts the
  partly written records.

### `ledger.cpp`

- **Reserve.** Admission against the limits uses subtraction, so it cannot
  overflow. `ConstrainLimits` only lowers limits, and never below current use,
  so those subtractions never underflow. `Fail()` throws before any counter
  changes. An allocation that is never admitted releases nothing.
- **Backing life cycle.** The cycle is Free → assigned (reserved) → converted →
  Free on token release. Each step moves bytes between categories and keeps
  `total` and `ram` unchanged. The pooled `ChargeToken` releases its
  `Allocation` references only after its lock scope ends, so it never takes the
  lock twice. An `Allocation` that a pooled token outlives returns to Free
  before it is destroyed.
- **Pins.** A pin holds its charge, so `pins == 0` holds whenever a charge or
  allocation is destroyed. Pin admission checks every block before it changes
  any counter.
- **Locking.** Every read takes the mutex: `Info`, `FindFreeBacking`,
  `Snapshot` and the limits.

### `checkpoint.cpp`

- **Reference counts.** `ChunkReference` and `CheckpointChunk` keep atomic
  per-kind counts. Copies take a count, moves do not, and self-assignment is
  handled.
- **Metadata charges.** Capture reserves every charge before building
  anything, verifies capacities, and publishes without throwing.
- **Entry reuse.** Reuse (`entry_capacity == entries_.size()`) is sound,
  because positions never decrease: equal totals mean the same per-component
  sets. An expired weak entry forces `replace_entries`, which rebuilds from the
  checkpoint under construction.
- **Identifiers.** `NextId` and `ObserveDurableId` handle exhaustion and keep
  identifiers monotonic.

### `retention.cpp`, apart from F3 and F4

- **Retained payload total.** The incremental `retained_payload_bytes_`
  matches the full recount:
  - Both deduplicate chunks by identity at the same index within a component.
  - `UniquePayloadBytes` is computed before the record is cleared on erase, and
    after it is inserted on admit.
  - `RamBytes()` cannot change after a payload is created, because `bytes_`
    and `device_` are fixed.

  Add A, then B (sharing X), then remove them in either order: X is counted
  once and subtracted once.
- **Index bounds.** The `Branch` index `Tokens()[Boundary()]` stays in range,
  because `below` implies a strict prefix with a smaller boundary.
- **Concurrency.** Every caller of `Pin`, `Unpin`, `Reclaim`, `ReclaimWhere`,
  `Admit` and `TryTouch` holds `cache->mutex`
  (`component_text_cache.cpp:1086,1426,1583,1592,1897,2229,2448`). The lock
  order is always the cache mutex, then the ledger mutex: `Emit` takes
  `Snapshot`, and record destructors release charges. Nothing takes them the
  other way round.

## What static reading could not verify, and the check that would

| Question | Check |
| --- | --- |
| F1 and F2 cannot be reached in any 27B path, including a KVarN draft *model* (`draft->executor_` in the 27B adapter) if one can be configured with KVarN | CPU: the tests under F1 and F2. GPU, in the final combined pass: the 27B KVarN MTP `cache` and `long-context` functional suites, with a rejection that crosses a group end, comparing records byte for byte against a cold AR replay of the accepted tokens |
| Requantized records after a rollback match a cold fill byte for byte on 27B | GPU `kvarn_ops_test` case: write rows, flush across a boundary, reopen, rewrite the same rows, flush again, then compare records with `memcmp`. The existing `TestVerifyMatchesDecode` covers attention, not the requantization after a reopen. Flash-Next already records passing "boundary rewinds" (KNOWHOW, 2026-11-03) |
| How often S2 happens | `capture_failure` refusal counts in the `cache` and `pi_agent` functional logs |
| How large S1 is | 27B KVarN checkpoint payload bytes at group-end boundaries in a `cache` run |
