# RAM cache ledger stuck over capacity; requests fail with "cache resource exhausted" (brief, 2026-11-18)

Production runtime-20261118b (a3f775a8: `DiskStore::ValidateCandidate`, see
`plan/serving/DECODE-STUTTER-BRIEF.md` and the knowhow entry "Disk writes
failing once the RAM cache is full"). Flash-Next, 1M, 4 sessions,
`--kv-pool-bytes 30000000000`, `--cache-ram-bytes 4294967296`, disk cache
100 GiB (~99.7 GiB used). Log snapshot (read only, UTC):
`build/b1/stutter-20261118/server-prod-b.log` (started 03:46, ~400 requests).

## Symptoms

1. **User requests fail.** r394 (prompt 237,994, image) streamed 1,384
   tokens, then at release: `[ERROR] [chat] cache resource exhausted`,
   `outcome=stream_error error_code=generation_failed` (07:49:15). Same for
   r398 at 07:50:53. A cache admission failure must never fail a request.
2. **The RAM ledger stays over its capacity for hours.** `configured
   capacity_bytes=4294967296`; `cache_ledger_total_bytes` passes it at ~05:30
   and stays ~4.34 GB. At 07:49: backing_free 3.06 GB (free pool blocks),
   private_state 0.71 GB, metadata 0.49 GB (was 0.29 GB at start, 0.37 at
   05:20), staging 51 MB. With 3 GB of free backing blocks, admissions still
   fail: 138 `persistence_refused reason=cache resource exhausted`, 38
   `persistence_incomplete` with the same reason, 22
   `live_frontier_write_back_failed`, 27 `restore_failed source=ram` (e.g.
   07:07:48, 17 RAM candidates in a row), many `capture_refused
   reason=byte_capacity`. Long conversations (230K tokens) are then not saved.
3. Not seen with runtime-20261118a, but there every disk publication failed,
   so the disk catalog (and its RAM metadata) did not grow. Check whether
   metadata grows with each publication (disk index charged to the RAM
   ledger) and whether the a3f775a8 change is related.

## Goal (generic, all adapters)

- Find what holds the ledger over capacity and fix it at the root, e.g. free
  backing blocks must be given back (pool shrinks) when metadata or other
  categories need room, metadata of the disk catalog must be bounded or not
  compete with the RAM cache's data, and admissions must evict/reuse before
  refusing. Explain with log/code evidence what each category is and why
  retention did not reclaim.
- Every cache failure on the request path (release, capture, write-back,
  restore) must degrade to "not cached" and never surface as a request
  error. Find the exact throw site behind `[chat] cache resource exhausted`
  at release and fix it; audit the other request-path cache calls.
- Also explain `restore_failed source=ram reason="cache resource exhausted"`
  (why a restore needs ledger room).

## Rules

- Do not touch production (`~/.local/share/gufo/`), the running server, or
  `~/projects/_ai.tools`. Never modify `tools/exl3_quant/hc/` or
  `plan/exl3/HC-Q8-CALIBRATION.md`. Never use `pgrep -f`/`pkill -f`.
- No GPU, model, session or timing runs. Build with `build/b1/build.sh
  <targets>` and run CPU/unit tests with `build/b1/build.sh --run <cmd>`;
  format check `build/b1/checkfmt.sh`. Add unit tests: a long run of
  publications with a full RAM cache keeps the ledger within capacity and
  admissions succeed after eviction; an admission failure at release does
  not fail the request (scheduler/test adapter).
- Read docs/GFX1151-KNOWHOW.md (cache entries, November 16-18) first.
- Have a second reviewer (Codex thread) review the diff, fix, re-review.
- Do not commit; leave the diff in the working tree. Report in English:
  root causes with evidence, the change, tests and results, what the GPU
  pass must check, open risks.
