# Decode stutters while another request prefills; checkpoint saves fail (brief, 2026-11-18)

Production: runtime-20261118a (622e6cf8), Flash-Next, 1M context, 4 sessions,
`--kv-pool-bytes 30000000000`, RAM cache 4 GiB, MTP 7 drafts, K4V4.
Log snapshot (read only, UTC timestamps): `build/b1/stutter-20261118/server.log`.
The user runs several web-chat conversations at once and sees generation
"write a bit, stop, write a bit, stop".

## Item 1: checkpoint persistence fails with `std::bad_alloc`

From 01:15:25 on, every queued disk write ends in
`event=persistence_incomplete reason="std::bad_alloc"` (10 in a few minutes),
right after `capture_refused reason=byte_capacity tokens=36193` /
`snapshot_skipped reason=3`. RAM-cache `ledger_bytes` sits at ~4.13-4.14 GB
(cap 4 GiB). Consequence: request r14 (prompt 36,193) restored only 8,444
cached tokens and re-prefilled 27,749 tokens (ttft 61 s), which is the
prefill that made the other conversations stutter for a minute.

Find why a disk write needs RAM-ledger (or other) memory it cannot get once
the RAM cache is full, and fix it generically (all adapters): a durable disk
write must not fail because the RAM cache is full (stage through bounded
staging outside the RAM budget, the snapshot pool, or evict RAM entries that
are already durable first). Also check why r14 found only an 8,444-token
checkpoint: was its conversation's newer state never saved because of these
failures, evicted, or a different prefix? Explain with log evidence.
Also say what `io_lock_ms=26997` on the first publication (line ~49) was.

## Item 2: decode stalls while another request prefills

Evidence (01:14:44-49): request 7 decodes (`chunk_tps` ~39) while request 8
prefills 27,749 tokens in 512-token chunks (~0.58 s each, ~900 tok/s).
Request 7 gets ~50 tokens per ~2.5 s wall (~20 tok/s visible, visible pauses
of 1-2 s). The scheduler already alternates prefill chunks with decode time
(`PrefillPolicy::decode_time_share_min/max`, `text_generation_scheduler.cpp`
~2760-2810, docs/SERVER.md ~1793), i.e. roughly 50/50 time sharing.

Goal: a decoding conversation keeps most of its solo speed and no pause over
~0.3 s while another request prefills, without making the prefill much
slower. Time-sharing alone cannot do both (giving decode 75% of the time makes
a 30 s prefill take ~120 s). Preferred root fix: let the active decoders'
rows ride along in the prefill pass (one combined pass: shared expert/weight
streaming once, each session's mixers/attention/GatedDeltaNet/MTP on its own
rows), the way packed scoring already runs several requests in one pass
(docs/SERVER.md ~1780, Flash-Next packed scoring). Prefill is compute bound
and decode is weight-bandwidth bound, so the decode rows should cost little.

First assess feasibility (MTP draft/verify rows, rollback, K4V4/pool, image
sessions, sampling, graphs) and say what it would take. If it is feasible in
a bounded change, implement it for Flash-Next with a generic scheduler hook.
If not, implement the best bounded alternative and explain the trade-off
with numbers, e.g. smaller prefill chunks while a decoder is active
(say what chunk size does to prefill tok/s; check the knowhow first), or a
decode floor in the time share. Do not change stream priorities, queue
counts or `GPU_MAX_HW_QUEUES` (frozen 2026-11-10).

## Rules

- Do not touch production (`~/.local/share/gufo/`), the running server, or
  `~/projects/_ai.tools`. Never modify `tools/exl3_quant/hc/` or
  `plan/exl3/HC-Q8-CALIBRATION.md`. Never use `pgrep -f`/`pkill -f`.
- No GPU, model, session or timing runs (production is live on this GPU).
  Build only affected targets and run CPU/unit tests (`tests/cache`,
  `tests/cli` serve tests); add unit tests for the new paths (item 1: disk
  write succeeds with a full RAM cache; item 2: the scheduler interleaving /
  combined pass contract with a test adapter). Docker test commands need
  `--entrypoint bash --user $(id -u):$(id -g)` if used.
- Read docs/GFX1151-KNOWHOW.md first; do not re-measure what it records.
- Keep code style; run `tools/ci/check-format.py` for C++ changes.
- Do not commit; leave the diff in the working tree. Report in English: root
  causes with evidence, the change, tests and results, predicted effect,
  what the combined GPU pass must measure, open risks.
