# Held write waits on decode (brief, 2026-11-16)

Follow-up to cd89eb6a (large frontier write-back on the single writer, see
`plan/serving/LARGE-WRITEBACK-BRIEF.md` and the knowhow entry "Large frontier
write-back on the single writer").

## Symptom (combined pass, `build/b1/pass-20261116/server.log`, read only)

2 sessions. A (325K tokens, 3.2 GB state) was captured on reassignment and its
slot held (SlotHold) until the single writer read every piece. B decoded in
the other slot meanwhile. C, an unrelated short request, needed A's slot:

```
request=r4 ... queue_ms=58593.5 ttft_ms=58872.4          <- C waited ~59 s
event=published tokens=324988 payload_bytes=3215196304 yield_ms=6.7
    stream_wait_ms=46533.2 filesystem_ms=1569.0            <- the held write
event=disk_restore tokens=324988 bytes=3215196304 pieces=3268
    stream_wait_ms=202.1                                    <- same bytes, idle GPU
event=published tokens=325050 payload_bytes=3216200584 stream_wait_ms=131.7  <- idle GPU
```

46.5 s / 3268 pieces = ~14 ms per piece, about one decode step. With the GPU
idle the same 3.2 GB moves in ~0.2 s. So device reads for the held write wait
behind the other slot's model work piece by piece.

## Where to look

- `src/cache/streaming.cpp` `StreamedStore::Write` (~472-662): read-ahead queue,
  `Settle`, per-piece in-place reads, pool leasing ("Queue only while another
  stream stays free").
- `src/core/hip/transfer_pool.hpp/.cpp`: `TransferStream::BindTo` binds a lease
  to an existing execution stream. Check whether live/held slot reads bind to
  (or wait on) the shared model execution stream, so each copy queues behind
  the other slot's decode kernels.
- The adapter copy for live/held slot reads (`copy_` sources built in
  `src/cli/serve/component_text_cache.cpp` around `CaptureDurable` /
  SlotHold ~1176-1410, and the model adapters' read paths, e.g.
  `src/models/qwen38_flash_next/continuation_adapter.cpp`).
- Read-ahead sizing: `ReadAheadBytes`, `cache::kPersistenceReadAheadBytes`,
  transfer pool capacity.

## Goal

A held write of a ~3.2 GB state must finish in a few seconds (target: stream
wait under ~3 s) while another slot decodes, without hurting that decode
(no long device-side stalls; B's max token gap stayed 0.085 s and must stay
small). Find the actual reason each piece waits ~one decode step and fix it at
the root, generically (all adapters), e.g.:

- a held slot runs no model work, so its reads should only wait for that
  slot's last work (an event), not queue on the shared execution stream; or
- queue many piece copies before waiting (deeper read-ahead / larger
  transfers), so one decode step drains many pieces instead of one.

Prefer the root cause over tuning. Explain with evidence (code paths) which it is.
If a measurement is needed to decide, say so in the report instead of running it.

## Rules

- Do not touch production (`~/.local/share/gufo/`), the running server, or
  `~/projects/_ai.tools`. Never modify `tools/exl3_quant/hc/`.
- No GPU, model, session or timing runs. Build only the affected targets and
  run CPU/unit tests (`tests/cache`, `tests/cli` serve cache tests, and the
  transfer pool test if CPU-runnable); add/adjust unit tests for the new path.
- Keep code style; run `tools/ci/check-format.py` for C++ changes.
- Do not commit; leave the diff in the working tree for review. Report: root
  cause with evidence, the change, tests run with results, open risks.
- Reply/report in English; the main session translates.

## Decision (user, same day): option B, no queue changes

Root cause found in round 1: every write piece is copied on a transfer-pool
stream, and under `GPU_MAX_HW_QUEUES=2` those streams share hardware queues
with the peer slot's decode, so each copy waits behind one decode round
(~54 ms MTP round). Read-ahead holds only 8 x 1 MiB pieces; other pieces are
read one at a time. (Correction: the 46.5 s write was A's prompt-end capture
while B decoded, not the held reassignment write, which published at 131.7 ms
stream wait on an idle GPU. Same mechanism.)

Option A (high-priority transfer streams, extra HIP queue pool) is rejected:
it touches the queue setup the user froze on 2026-11-10 (U4-3 stall risk).
Its patch is kept in `~/projects/_backup/held-write-priority-queues-20261116/`.
Do not change stream priorities, queue counts or `GPU_MAX_HW_QUEUES`.

Implement option B generically in `StreamedStore::Write`: keep many more
device copies in flight so one decode round drains many pieces, e.g.
- put sub-pieces of large sources and private-state pieces on the read-ahead
  queue too (not only whole-piece sources), and
- size the read-ahead so ~64 MiB (or what the numbers justify; explain) can be
  in flight, bounded and charged to the ledger as today, without starving
  foreground restores of transfer streams (keep the "one stream stays free"
  rule or equivalent), and settle everything on every exit path.
Target from the numbers: 3.2 GB under ~3 s at ~54 ms per round needs ~64 MiB
in flight. Say what the change predicts and what it leaves (filesystem time,
staging memory). Same rules as above; add unit tests that the queue keeps
many sub-pieces in flight with a test adapter, stays bounded, and drains on
cancel/error.

## Round 3: durable drop runs too early (pass 4, `build/b1/pass-20261116d/server.log`, read only)

Pass 4: B max gap 0.517 s (pass 3: 0.504), `event=checkpoint_dropped` never
logged. Timeline: C's `Acquire` ran at 12:57:51 (`live_frontier_captured
reason=reassignment slot=1`), so `DropDurableBorrowers` ran then; nothing was
durable yet. The prompt-end checkpoint (tokens=328416, id 51285) published at
12:58:03. C's slot `Acquire` returned at 12:58:23 after the held write released
the slot, and `Slot::PrepareReassignment` (`src/cache/slot.cpp` ~346) preserved
the borrowed rows in the foreground then, without re-checking durability.

Fix: re-run the durable drop right before the slot preserves borrowed rows,
i.e. after the wait for the slot hold, generically (all adapters). For example
wait for the slot to become acquirable (or have `Slot::Acquire` take a
callback run just before `PrepareReassignment`), then under the cache mutex
drop durable borrowers and redo the lookup if needed. Keep lock order safe (no
cache mutex held while waiting on the slot; no deadlock with the writer's
SlotHold). Restore candidates already pinned for this request must stay valid:
if a dropped checkpoint was the selected RAM candidate, the request must still
restore correctly (from disk or a re-lookup), never from dropped rows.
Also consider: a checkpoint whose publication is queued/in flight when the
reassignment happens is not durable yet; say whether waiting briefly for it is
worth it (do not implement unbounded waits).

Add a unit test: a checkpoint borrowing from a held slot becomes durable while
the acquirer waits for the hold; on release it is dropped, not copied
(`checkpoint_dropped` / no foreground copy bytes). Same rules as above; leave
the diff uncommitted; report in English.

## Round 4 (user, 2026-11-17): snapshot the held slot in the foreground, write later

Shipped state: abba3337 (production runtime-20261116b). Pass 4
(`build/b1/pass-20261116d/server.log`, read only): C waited 32.5 s =
~12 s for the active prompt-end job on the single writer (A's 3.2 GB
prompt-end capture, device reads slow while B decodes) + ~20 s for the held
reassignment job's device reads (~270 MB/s while B decodes). On an idle GPU
the same 3.2 GB of device reads takes ~0.13-0.2 s. Rounds 1-3 notes above.

Goal: a request that needs a slot held by a pending durable write starts in
about a second, not 30+ s, and the peer slot pauses at most ~0.5 s once
(today's reassignment foreground copy already costs ~0.5 s; see
`foreground_bytes`/`residual_wait_ms` in C's metrics).

Design the user approved:
1. At reassignment (where `PrepareAdmission` today captures with `hold=true`
   and queues a slot-holding job), copy the whole state that would be held
   (all device sources of the capture, every adapter, generic) into a
   temporary host buffer in the foreground, the same way the existing
   foreground row preservation runs (the scheduler does not run the peer's
   decode meanwhile, so the GPU is idle and the copy runs at full speed).
   Then release the slot immediately and give the payload to the persistence
   writer, which writes it to disk from host memory without touching the
   device. No SlotHold for this path, so C never waits on the writer queue
   (removes both the ~12 s and the ~20 s).
2. Temporary memory: Strix Halo has 128 GB unified memory. Bound the
   temporary buffers with their own cap (explain the value; a few GB, enough
   for one ~3.5 GB state plus margin), not the RAM cache budget; charge and
   release them on every path (publish, failure, cancel, shutdown). If the
   allocation fails or the cap is reached, fall back to today's held write
   (SlotHold), unchanged.
3. The writer should free each part of the buffer as soon as that part is
   on disk if that is simple; otherwise free it at publication.
4. Do not change stream priorities, queue counts, `GPU_MAX_HW_QUEUES` or the
   transfer pool rules. Keep the prompt-end/write-back captures as they are
   unless the same mechanism clearly applies; say so.
5. Log the foreground snapshot: bytes, ms, and the fallback reason if taken.

Tests (CPU, test adapter): the reassignment capture snapshots to host and
releases the slot before the write runs (a second acquirer is not blocked by
a slow writer); cap reached / allocation failure falls back to the held path;
buffers are released on success, failure and cancel; restore from the
published checkpoint is byte-identical. Same rules as above (no GPU/model/
server runs, check-format, no commit). Report in English: the change, the
predicted C wait and peer pause, memory cost, tests and results, open risks.

## Round 5: pass 5 result and the regression it found (`build/b1/pass-20261117a/`, read only)

Binary 8f5cbb4f… (round 4 + review fixes). Results:
- C: 2.9 s total, queue 961 ms (was 34.4 s). `slot_snapshot bytes=3279220932
  ms=627 alloc_ms=383 copy_ms=236 arenas=51 reused_bytes=0`. Publication of
  the snapshot: stream wait 0, filesystem 1.99 s. Arenas trimmed after idle.
- B max gap 1.172 s (one gap over 1 s): snapshot 0.63 s + C's foreground
  preservation (`foreground_bytes=1331145216 foreground_copy_ms=252
  residual_wait_ms=289`).
- **Regression:** A's next turn took 192 s: `cached_tokens=163840`,
  ~168K tokens re-prefilled. Log at 23:13:33, right after the snapshot:
  `event=persistence_incomplete checkpoint_id=51805 reason="borrowed rows retired"`.
  51805 is A's prompt-end capture (tokens ~331674), still queued/being
  written on the writer and reading A's slot rows. The snapshot released and
  reset the slot, so that job lost its rows. Only the 331721-token frontier
  (prompt + generated reasoning, not a prefix of the next prompt) was
  published, so the next turn fell back to an old 163840 checkpoint. Before
  round 4 the held write kept the slot until the writer had read everything,
  so the prompt-end job finished first.

Fix (generic, all adapters), in this order of preference:
1. In the same foreground step, before the slot is released, also copy to
   host every device source that any queued or active persistence job reads
   from this slot (the prompt-end job above is the usual case: it is always
   in flight right after a long prefill). Pieces identical to ones already in
   the frontier snapshot (same disk file id / chunk) must be shared, not
   copied twice; only the differing pieces (e.g. its private state, tail)
   need new bytes. Switch those jobs' sources to the host copies safely
   while the writer may be mid-job (the active job: under the existing
   piece/slot locks, or by pausing it at a piece boundary). Charge the pool.
2. If that cannot be done safely or room is short, fall back to today's held
   write for that reassignment (never lose a pending job's checkpoint).
Also: no `persistence_incomplete reason="borrowed rows retired"` may happen
because of a snapshot; add a test where a prompt-end job borrowing the slot
is queued (and one where it is mid-write) when the reassignment snapshot runs,
and both checkpoints publish and restore byte-identically.

Peer pause (secondary): first-time `alloc_ms` 383 ms. Say whether the pool
can be warmed off the scheduler thread when a long state appears (e.g. when
a slot's state passes ~1 GB, on the trim thread), bounded by the same cap and
idle trim; implement if simple. Do not change the 1.33 GB foreground
preservation in this round (left for later).
Same rules as before; do not commit; report in English.
