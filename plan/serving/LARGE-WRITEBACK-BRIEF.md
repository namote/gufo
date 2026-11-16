# Large live-frontier write-back fails on Flash-Next (brief, 2026-11-16)

## Symptom (production, runtime-20261112a, 1M context, 2 sessions, RAM cache 4 GiB, disk cache 100 GiB)

The user runs several web-chat conversations at once. One is ~340–350K tokens.
When a slot is reassigned, the long conversation's state (~3.4 GB) is not
saved. When it returns, it is re-prefilled from an old checkpoint (172K) or
from the system prompt only (`cached_tokens=9796`): 300K+ tokens at ~800 tok/s,
6–10 min, and other conversations' decode stalls between prefill chunks.

Log: `~/.local/share/gufo/logs/25afb7e5ec97/server.log` (UTC timestamps; read only).

```
event=live_frontier_write_back_failed slot=1 tokens=342829 error=std::bad_alloc
event=live_frontier_capture_failed reason=reassignment slot=2 tokens=347164 error=checkpoint input changed within inherited prefix
schema=component-cache-v1 event=capture_refused reason=byte_capacity tokens=349484
schema=component-cache-v1 event=snapshot_skipped reason=3 tokens=349484 bytes=3447442000
event=live_frontier_captured reason=reassignment slot=2 tokens=13545 bytes=269069892 disk_queued_bytes=269069892 ms=16.26   <- small states succeed
http ... prompt_tokens=347331 prefill_tokens=337535 cached_tokens=9796 duration_ms=410320   <- full re-prefill
```

## Cause found so far

`ComponentTextCache::Lease::Impl::CaptureDurable` (`src/cli/serve/component_text_cache.cpp` ~936)
is the direct-to-disk path, but rows the slot cannot lend are staged in RAM
backing inside the RAM cache budget, all at once. Flash-Next
`ContinuationAdapter::BorrowRowsView` (`src/models/qwen38_flash_next/continuation_adapter.cpp` ~904)
returns empty for every component when `kv_pool_` is set and for KVarN records,
so the whole 3.4 GB is staged in RAM -> `bad_alloc` / `byte_capacity`.
Flash-Next already has `RetainsRows`/`RetainRows` (device block references).

## Goal

1. Large durable captures (write-back and reassignment capture) must succeed
   without needing RAM-cache room for the whole state: lend/retain device rows
   (shared KV pool blocks, KVarN records) or stage and persist chunk by chunk
   under a small bounded staging budget. Strix Halo has unified memory.
   The RAM cache stays for fast restores of small states.
2. Find and fix why reassignment capture fails with
   `checkpoint input changed within inherited prefix` (is it a real input change or a bug?).
3. Explain why intermediate disk checkpoints were missing (restore fell back
   to 9796 tokens although earlier 155K/172K checkpoints existed; disk cache is
   at 99/100 GB). If eviction removes the active conversation's checkpoints
   first, fix the order.

## Scope (user, same day)

Fix it in the shared cache path so every model adapter benefits whenever rows
cannot be lent, not Flash-Next only. Adapter-specific lending (Flash-Next
`RetainRows`) is only an optional optimization on top. Unit tests use a test
adapter that lends nothing.

## Rules

- Do not touch production (`~/.local/share/gufo/`), the running server, or
  `~/projects/_ai.tools`. Never modify `tools/exl3_quant/hc/`.
- No GPU, model, session or timing runs. Build only the affected targets and
  run CPU/unit tests (`tests/cache`, `tests/cli` serve cache tests); add unit
  tests for the new path (a large capture under a small RAM budget succeeds;
  staging stays bounded).
- Keep code style; run `tools/ci/check-format.py` for C++ changes.
- Do not commit; leave the diff in the working tree for review. Report: root
  causes (1–3) with evidence, the change, tests run with results, open risks.
- Reply/report in English; the main session translates.
