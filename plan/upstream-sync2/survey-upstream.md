# Upstream Gufo survey: 6d793814 (v0.31.0) .. upstream/main 3e3708b5 (v0.35.0, 2026-11-19)

Fork: `ai-tools/port-u4f` at `ff594d14`. 114 upstream commits reviewed by subject. Every
commit that touches Flash-Next, 27B, serve, cache, JSON or core memory was reviewed by
diff. `git apply --check` was run against HEAD to test whether each one cherry-picks
cleanly. This was read-only: nothing was edited, built or run on the GPU.

## 0. Status of the old range (c92b19c5..6d793814, plan/upstream-sync)

- `git cherry HEAD 6d793814 c92b19c5` gives 4 commits with equal patch-ids and 135 `+`.
  The ports were done by hand, so patch-ids do not match. Matching on subjects instead
  (`(upstream #NNN)` in our log) shows every planned U-item has a commit, except these:
  - **U2-4 (#903, #1063): still not ported, deferred on purpose** (PLAN.md:98). Our
    backing admission was rewritten. The 2 GiB RAM cache is committed at start-up.
    Nothing has changed this; keep it deferred.
  - U2-1 (#876): not applicable. The equivalent `dense_prefix` from `cef16057` is
    already in our tree.
  - U4-1 (#895): reverted in `4f49f025` (output not byte-identical). U4-2 and U4-5
    (#894): rejected. U4-3 (#961, #993): reverted, then re-landed in `21c138cc` with the
    `8a18e528`/`6e2c504e` mitigations. The stall persists on kernel 7.0.0-43 (memory note).
- The "won't do" list (PLAN.md:343) still applies. The one exception: the UD 3.1 weights
  are now in use (see rejected list).

## 1. Ranked candidates (benefit / effort)

Legend: effort CP = clean cherry-pick (`git apply --check` passes on src/), HP = hand
port, N/A = not applicable.

### 1. 6ecfa504 (#1249): keep GPU-visible host buffers out of compaction. **Bug/safety (stall)**
- Upstream: `HostAllocations::Allocate` no longer uses `hipHostMalloc`. It takes anonymous
  pages (`CommitZeroedPages`), pins them long-term (io_uring, `PinHostPages`), then calls
  `hipHostRegister(Mapped)`. If the pin fails it falls back to hipHostMalloc. The header
  says why: *"hipHostMalloc pages stay candidates for compaction even when pinned: every
  failed migration invalidates their GPU mapping and evicts all of the process's GPU
  queues, which froze the GPU for 45-109 s in a probe under memory pressure."*
- Our tree has the old code: `src/core/hip/host_allocations.cpp:11-40`. Flash-Next keeps
  GPU-read control and descriptor buffers in `host_allocations_`:
  `runtime/executor.cpp:1230` (n-gram), `runtime/executor_batch.cpp:160` (GDN batch
  descriptors), `executor.hpp:354`/`1428` (kv table, control).
- **Why it matters to us:** it fits our open symptoms closely:
  - U4-3 "queue-level stall, no fault, HQD frozen, released when another process
    changes the queue set" (PLAN.md:282-307).
  - `compaction_proactiveness=50` costs MTP about 18% but not AR (KNOWHOW:1260).
  - "Host/driver latency hurts MTP, not AR" after fragmentation.

  This is a hypothesis, not a proof, but it is the cheapest test we have.
- Lands in: CP of `src/core/hip/host_allocations.{cpp,hpp}` (+~50 lines; `CommitZeroedPages`
  and `ReleaseCommittedPages` already exist in `src/core/platform/pages.hpp:22-26`).
  Upstream already routes every model allocation through it. Ours still has raw sites to
  convert, about 20 more lines:
  - `src/cli/serve/inference_backend.cpp:1155` (fork-only `snapshot_allocate`, 64 MiB
    pinned arenas up to 4 GiB)
  - `src/models/qwen/hip/arena.cpp:1275` (27B KV block table, GPU-read)
  - `src/models/qwen/hip/model_loader.cpp:99` (load staging only; low priority)
- **Deployment caveat:** the fix only works where the pin succeeds. Our supervisors run
  plain `docker run` with no `--ulimit memlock` and no seccomp override
  (`tools/serving/qwen27b/gufo-server.py:143-165`). Docker's default seccomp blocks
  `io_uring_setup`/`io_uring_register`, and upstream's docs/internal/cache.md says the
  same about Podman. Then `PinHostPages` returns -1 and the code falls back to
  hipHostMalloc, so nothing changes. To get the benefit, add `--ulimit memlock=-1` and a
  seccomp profile that allows those two syscalls, then check that the pin took (add a log
  of pinned bytes).
- Risk: low. Exact same GPU arithmetic. Pinned anonymous memory is never migrated.
  Watch RLIMIT_MEMLOCK accounting.
- Check after porting: the U4-3 `hang_loop.sh` stress and MTP d0/d32K at proactiveness
  20 against 50.

### 2. 800ce328 (#1230): keep compacted MTP inject rows out of the norm buffer. **Bug (GPU race)**
- `MtpForwardBatch` compact tail sets `tail.inject = base.mtp_h`. `mtp_h` aliases `s_.xn`
  (our `runtime/executor.cpp:1326`). The FFN combine writes `xn` while its blocks still
  read the inject partials, so this is a read/write race inside one kernel. Upstream
  instead compacts inject rows inside `base.inject` and skips copies that are already in
  place.
- Ours: the same bug is at `runtime/executor_batch.cpp:469-487`.
  `compact_tail = rows > 8 && rows % 8 != 1`, which is true for multi-session MTP batches
  such as our 4-session serving.
- Effort: HP of 7 lines (file moved; the hunk is otherwise identical).
- Benefit: corrupt MTP-block FFN input → wrong draft-head KV/residual for batched
  sessions → lower acceptance at C2-C4. Target output stays exact, because the target
  verifies every draft. May also explain part of the old "serial vs batch snapshot tail
  differs in draft K/V rows" finding. No upstream number.
- Risk: very low. Run `mtp_ops`/session batch tests.

### 3. 4e95aecb (#1146): return tool-call numbers as the model wrote them. **API + cache reuse**
- `json::Value::Numbers::kAsParsed` keeps the original spelling of each number token.
  Arguments replayed in the next turn then re-render byte-identical to what the model
  generated (`1.0` is not rewritten as `1`; integers above 2^53 survive), so the
  continuation checkpoint is still reused. Schema checks use kCanonical.
- Ours: missing. `src/cli/serve/openai_chat.cpp:459` and `:1811` use `tojson()`/`dump()`;
  `src/core/json.hpp` has no `Numbers`.
- Effort: **CP; the whole commit applies cleanly** (~110 src lines plus tests).
- Benefit: client-visible correctness, and fewer cache misses in agent tool loops whose
  arguments contain floats.
- Risk: low.

### 4. 390639b3 (#1273): host reserve inside free CMA pages. **Bug (memory budget)**
- On Ubuntu 26.04 7.0 kernels, KHO scratch shows up as `CmaFree`, up to 8.7 GiB after
  boot. Upstream now subtracts `max(CmaFree, 4 GiB)` from MemAvailable instead of the sum.
- Ours subtracts both: `MeminfoAvailableBytes` subtracts CmaFree
  (`src/cli/serve/text_model_runner.cpp:108-129`), then the ceiling subtracts
  `kHostSnapshotHeadroomBytes` (4 GiB) again (`:211`). **This host right now:
  CmaTotal 0, CmaFree 6.18 GiB, kernel 7.0.0-43.** Our snapshot/pinnable ceiling is
  therefore about 4 GiB smaller than intended.
- Effort: HP of about 20-30 lines (upstream moved this into `src/ext/host_budget.cpp`; ours
  keeps it in text_model_runner). Port the formula, not the file move.
- Benefit: about 4 GiB more host-snapshot/RAM-cache room. Fewer `snapshot_room` refusals.
- Risk: low to medium. Free CMA pages hold only movable pages, so they are still reclaim
  room for the OS. Confirm with a pressure run.

### 5. 53193819 (#1168): cap JSON numbers at llama.cpp's 16 digits. **Safety (constrained decode)**
- The grammar's integer, fraction and exponent were unbounded (`Repeat(digits)`). A model
  that loops on digits under a strict schema never closes the value. The fix caps each
  part at 16 digits, matching llama.cpp, and bounds open numeric intervals.
- Ours: unbounded (`src/core/json_constraint.cpp:179-184`).
- Effort: **CP** (src/ applies cleanly; the fixture JSON may need a hand merge).
  ~60 src lines.
- Benefit: no runaway constrained tool calls. Matches the llama.cpp grammar.
- Risk: low. A legitimate value with more than 16 digits is rejected, as in llama.cpp.

### 6. 72442dce (#1102): a value starting with `</parameter>` is one call. **API (tool parse)**
- After a canonical `<parameter=v>\n` header, the closer search starts past the frame
  newline. Before this, `<parameter=v>\n</parameter>\n` split one constrained call into
  several.
- Ours: `src/models/qwen/chat_template.cpp:433` has the old `body.find("\n</parameter>")`.
- Effort: **CP after a path rewrite** (`src/families/qwen/text/` → `src/models/qwen/`;
  `git apply --check` passes). 11 lines plus tests.
- Risk: very low.

### 7. 850b5271 (#1158): continue the live frontier after a re-tokenized answer. **Speed (agent loops)**
- Re-tokenizing the client's echoed history can split our previous answer differently from
  the tokens the model sampled. A recurrent (GDN) frontier cannot rewind to the first
  differing token, so the turn falls back to an older checkpoint and re-prefills.
  `AlignToLiveFrontier` swaps in the frontier's own tokens when the text matches, and
  handles whitespace trimmed by the template at reply edges.
- Ours: missing (no `AlignToLiveFrontier`; `src/cli/serve/component_text_cache.cpp`).
- Effort: HP of about 250 lines (component_text_cache diverged by ~2.7K lines; hooks in
  `inference_backend.cpp`, `text_model_runner.{hpp,cpp}`, scheduler).
- Benefit: avoids a whole-tail re-prefill on turns where BPE re-splits the answer (common
  with code and whitespace). Upstream gives no number. Verify with `pi_agent.py`
  prefill-token counts.
- Risk: medium. Wrong alignment means a wrong continuation. Upstream adds 147 test lines.
  Interacts with our `continue_final_message` work (`28e3873a`) and U2-2 alignment.

### 8. 49b5df18 (#912): recover stale MTP backoff after batch observations. **Speed (MTP)**
- When a jointly planned greedy batch keeps observing profitable drafts, it clears the
  session's old AR backoff (`retry_tokens = 0`). Without this, the session stays in AR
  for up to 16 tokens after the batch ends.
- Ours: missing (`src/models/qwen38_flash_next/mtp_policy.hpp:80-112`, call at
  `engine.cpp:2112`; the same `batch_drafts` structure is at `engine.cpp:1725`).
- Effort: HP of about 15 lines plus the `mtp_sampling` test.
- Benefit: small but free, for multi-session greedy then solo. No number.
- Risk: very low. Greedy output unchanged.

### 9. fe2694bc (#1104): let optional copies displace stale conversations. **Bug (cache)**
- Record capacity can fill before byte pressure applies, which models with small
  checkpoints hit. The table then "freezes and refuses every later retry copy and grid
  point". The fix adds a `stale()` victim: not used within one record-table turnover.
- Ours: missing (`src/cache/retention.cpp:402` Victim). **CP: applies cleanly.**
  ~30 lines.
- Benefit: related to our U5 retry regression (retry copies refused/evicted).
- Risk: low. Run the retention tests and the functional `batch` retry rows.

### 10. 77621b27 (#1181): resume captured prefills before repeating peer work. **Bug (fairness/TTFT)**
- A request whose capture finished re-enters `prefilling` before the last-served peer
  instead of behind it.
- Ours: arrivals already do this (`text_generation_scheduler.cpp:1433-1442`), but the
  capture-resume path still uses `push_back` (`:2631`). HP of 8 lines.
- Benefit: less TTFT inflation for a request whose capture waited behind a long peer
  prefill. Related to C4. Risk: very low.

### 11. 3016d652 (#1117): keep snapshot copies inside one host registration. **Bug (restore)**
- A restore joins host-adjacent chunks from different `hipHostRegister` ranges.
  hipMemcpyAsync rejects a host range that spans two registrations, even on Linux. This
  hits the small or unaligned fallback. The fix splits each copy at registration and
  alias boundaries.
- Ours: old code (`src/core/hip/snapshot_copy.hip:23`). **CP: applies cleanly.**
  ~80 lines.
- Benefit: removes a rare `restore_failed` on segmented backing. Risk: low (host-side
  splitting only).

### 12. bd6b0fdd (#1128): match llama.cpp nested object defaults. **API (strict tools)**
- Nested tool objects that declare `properties` without `additionalProperties` become
  closed. Before, a misspelled key could turn into an unbounded string. The fix also
  accepts llama.cpp's required-first member order.
- Ours: missing. HP, because `src/core/json_constraint.cpp:1775` conflicts (our U1-2 port
  differs); ~120 lines. Risk: low to medium (it changes accepted grammars; run the
  llama_parity fixture).

### 13. 0a473ebf (#1288) + cc405337 (#1290): retention grace for new short conversations / keep edit history. **Bug (cache)**
- #1288: a new cheap (GDN state > 8 x rows) conversation keeps its recency for 1/16 of the
  RAM budget. Before, the next unrelated request evicted it and its follow-up missed. Our
  Flash-Next short chats are exactly "cheap".
- #1290: a short resumed tail's retry copy may not displace its own conversation's history
  below the boundary it resumed from, which an edit of the last turn needs.
- Ours: missing (`src/cache/retention.cpp` around `:280` `reused_at`, `:419`
  `resumed_tail`). HP of about 40 plus 15 lines. They rely partly on #1183 fields
  (`used_bytes`, generated-copy rank). Our tree already has `frontier_used`, `forked` and
  `kGenerated`.
- Benefit: fewer misses on short follow-ups and edits under RAM pressure.
- Risk: medium (retention policy). Replay with `tests/cache/retention_replay`.

### 14. a33323f2 (#1209): draft MTP proposals with a Q4_K copy of the head prefix. **Speed (MTP)**
- Upstream measured: the subset head is 1.17 ms per draft. The Q4_K copy saves about
  1 ms per round with unchanged acceptance, **+2.6% at C1 sampled**. Proposals only, so
  the output distribution is exact.
- Ours: the 98,304-row subset head exists (`draft_vocab.hpp`, `runtime/executor_mtp.cpp:497`).
  Our head is an EXL3 **K8** linear (`exl3_weights.cpp:448`; EXPERIMENTS.md:783 "current
  K8 head unchanged"), about 252 MB per draft for the prefix. A 4-bit copy (Q4_K via the
  existing GGUF GEMV, or EXL3 K4) would halve that read.
- Effort: HP of about 300-400 lines. Upstream's device Q4_K quantizer and `draft_head.hip.cpp`
  work from Q8_0; we would dequant EXL3 K8 → F16 → Q4_K at load. Plus graph-key plumbing.
- Benefit: probably +2-4% MTP decode (we draft up to 7, with greedy and sampled both using
  the subset). Measure acceptance.
- Risk: low for correctness. The acceptance change needs a measurement. ~130 MB more VRAM.

### 15. fdbc654e (#1270): learn a shared branch point after 192 common tokens. **Speed (cache)**
- Splits the 512-token rule: waiting for a peer still needs 512 tokens, but publishing a
  branch checkpoint for later prompts now needs only 192.
- Ours: `text_model_runner.hpp:794`, `.cpp:2577`/`2729`. HP of about 10 lines.
- Benefit: short shared system prompts (192-511 tokens) get a reusable checkpoint. For
  Flash-Next each checkpoint carries the full GDN state (tens of MB), so the value is
  uncertain. Risk: more RAM churn. Measure with functional `cache-shared-prefix`.

### 16. 11a51d2f (#1285): commit backing for a disk restore's capture before the restore. **Speed (serving-thread stall)**
- A capture after a disk restore shares no RAM chunks and must commit backing for the
  whole checkpoint. Upstream grows backing on the provider's thread first
  (`prepare_capture`), so the serving thread does not pay 56-58 ms per GiB (upstream
  cache.md measurement).
- Ours: missing (`src/core/hip/committed_backing.cpp`, `component_text_cache.cpp`). HP of
  about 100 lines; upstream's committed_backing grew by 333 lines in this range, so port
  only the hook.
- Benefit: removes a stall of tens to hundreds of ms on peers after long-context disk
  restores. Risk: medium.

### 17. 2e50748f (#1265) + 425e3513 (#1274) [+ infra from 842295e4 (#1251)]: lone prompts in 14K-row passes on transient scratch. **Speed (prefill)**
- Upstream measured: served prompts of **28K-85K tokens prefill 2-7% faster** than in
  4096-row passes. The extra row buffers (4.6 GiB) are mapped only while a wide pass runs
  with no decoder beside it, and freed at the next decode step or when idle (#1274).
- Ours: `kPrefillChunkTokens = 4096` (`engine.cpp:37`). We measured the 16K cap ourselves:
  **+2.8% PP16384, +1.3% PP8192, -0.5% PP2048, +6.38 GiB peak**, not adopted "below the 3%
  bar for its memory" (KNOWHOW:1670-1676). Upstream's transient-scratch design removes
  exactly that memory objection.
- Effort: large. Port `ScratchArena` (`src/core/hip/scratch_arena.{hpp,cpp}`, ~260 lines),
  which needs demand-mapped VMM (`mem::DemandMapped`) from #1251 (3.4K lines; only the
  arena part is needed). Our EXL3 executor scratch carve-up in `runtime/executor.cpp`
  (~300 lines). StepCapacity/PassChunk plumbing in engine and runner. Our U2-2 alignment
  and in-pass checkpoint rules must hold at the new pass width.
- Benefit: prefill is the user's top priority; about +1.5-3% on long prompts on our EXL3
  pack (our data), up to 7% upstream on GGUF.
- Risk: medium to high (VMM mapping on gfx1151, mapping and unmapping latency, KVarN 1M
  sessions leaving no room → falls back to narrow).

### 18. Disk-cache throughput series (medium effort each, deep conflicts)
- f5c0b20b (#1271): read disk checkpoints in parallel (8 x 1 MiB pieces) and warm the
  page cache ahead of admission. `streaming.cpp` +510 lines. Conflicts with our held-write
  and reassignment work. Benefit: faster long-context disk restores (ours restores 328K
  tokens in about 3 s). No upstream number.
- 70bd0fda (#1205): fsync new payloads concurrently ("one at a time each small file waits
  for its own commit", #1180). `disk.cpp` +127 lines. Our `disk.cpp:815` syncs one at a
  time. Benefit: the disk tier catches up faster. Low risk.
- ba0cb3b6 (#1283): keep queued turn ends until the persistence queue is full (coalesce
  generated frontiers only when there is no room). `streaming.cpp` HP of about 170 lines.
  Ties in with our held-write queue.
- ab68f0de (#1269): persist one grid checkpoint so another question about the same long
  document restores from disk. HP of about 90 lines.

### 19. a59df291 (#1224): 27B captures prompt checkpoints inside the prefill step. **Speed (27B scheduling)**
- A prefill step runs the passes that boundary-ending steps would run and captures between
  them. Layer groups span passes, so decoders wait less. Captures cost 4.0-7.1 ms each
  (upstream measurement; pass cost 120 ms + 2.25 ms/row, UD-Q4_K_XL).
- Ours: no 27B in-pass capture (`src/models/qwen/runner`, `hip/prefill.cpp`). HP of about
  600-1000 lines plus our KVarN checkpoint rules (C6 F1/F2). Low benefit/effort for now.

### 20. Ideas from origin (franzmoca/gufo, secondary; neither is upstream)
- e2ecbf12 + 41d7d621 (`origin/core/prompt-lookup-agreement`): greedy prompt-lookup copies
  stop where other matching occurrences disagree (repeated syntax such as log timestamps).
  `src/core/speculative/prompt_lookup.hpp` (+41 lines), `engine.cpp` AppendLookup
  (+11 lines), DFlash backend. Missing in ours. It may cut rejected long copies and wasted
  64-row wide verifications (U4-3). No numbers; needs a pi_agent/rewrite measurement.
- f0cd0a3b (`origin/perf/cache-in-place-restore-v2`): restore same-slot checkpoints in
  place on Flash-Next (no copy when the checkpoint lives in the slot being restored).
  ~330 lines, adapter-level. No numbers.

## 2. Rejected or not applicable (one line each)

- 365cc5a5 (#1112) decode shared-expert overlap calibration: Q8_0 (GGUF) shared experts
  only; the motivation is Windows. We already overlap EXL3 decode (`runtime/executor_moe.cpp:280`).
- f01a4570 (#969) fused batched FFN HC pair: built on the one-launch `FusedMixer` from
  #894, which we rejected (U4-5).
- 5f12c753 (#958) Q8_1 rows from batched HC mixing: they feed Q8_0 GGUF GEMVs; the EXL3
  MoE in our batch path does not consume `mixed_q8_ready_`.
- e39a314e (#1185) 27B KV tiles shared across verify rows: FP16-KV split-K kernel only;
  our 27B serves `kvarn-k4v4` (`attention_kvarn.hip`). The idea could move to the KVarN
  verify kernel as its own project.
- bc35aa81 (#1184): only for runners without prefix reuse; ours have it.
- 6858788b (#1207) `--max-pending 0` semantics: we do not use 0. Optional.
- d69ceda9 (#1210) trim the separator after a prompt-closed think block: extension opt-in;
  no built-in runner returns `kTextAfterThought`.
- 017e9249 (#1083) + be3a81af (#1164) prompt-state budget refusal: #1164 removes the charge
  for preallocated runners (Flash-Next, 27B), so the net effect for us is nothing.
- 0d238d4f (#1175) required scheduling profile: an ext API refactor with no behavior gain.
- 12e8b99f (#934), 1b0e4926 (#1272): disk-only (RAM cache off) mode; we run a 2 GiB RAM cache.
- b97f3b5c (#1121): already present (`ValidateCandidate`, `src/cache/disk.cpp:714`).
- 07e1191f (#1194), e8f2fe0e (#1196): start-up accounting for the automatic budget and
  recovery metadata performance; small for our disk sizes. Revisit only if start-up
  refusals appear.
- 7674e71a (#1195) shared transfer streams: we already cap `GPU_MAX_HW_QUEUES=2`; gain unclear.
- 48aeeee9 (#1183) generated-frontier retention and coordinated RAM/disk (3.4K lines): we
  have the fork equivalent (kGenerated live-frontier copies from U2-2/U3-3, held-write).
  Mine it only for specific sub-fixes (see 13 and 18).
- 842295e4 (#1251), f6346864 (#1258), e4a6856f (#1275): demand-mapped session state,
  sharing state when sessions do not fit, RAM-cache relief under pressure. Strategic
  memory infrastructure (~7K lines), not a port. Consider it if we ever need more sessions
  than fit.
- ca6955c4 (#1127) AVX-512 sampler block maxima and n-gram hash: microseconds per token;
  upstream says "do not establish an end-to-end inference speedup".
- Rust changes (4b030049 #1159 httparse, 5ace393c #1176 multipart, b50ef159 #1149
  websocket, 1b4bc1c2 #1261 / f055766f #1266 GGUF header, 77b6ccf6 #1118 / 768a482f #1110
  audio, 580d053b #1182): the fork has no Rust toolchain. If the HTTP hardening is wanted,
  re-implement its rules in C++ `http_server.cpp`: 400 on Transfer-Encoding, on
  non-digit or conflicting Content-Length and on control bytes; 431 above 100 header
  fields.
- f4bc1adb (#1124) deprecate `--draft-tokens`/`--min-draft-tokens`/`--draft-policy`: CLI
  churn; check our supervisors before ever adopting it.
- dbfe834d (#1167) UD 3.1 Q4_K_XL revision: already in use (our file is 17,659,407,264 B =
  16.45 GiB, the new size).
- e74e16b5 (#1190) SSM post-norm kernel into the shared lib: structure only (`src/families`
  does not exist in ours).
- fc26f52c (#945) Clang 23 WMMA builtin signatures: needed only when the toolchain moves
  to Clang 23 (one line per file); keep in mind for a ROCm bump.
- afaca7d2 (#1267) diagnose compute-unit count, 466d1bc9 (#1222) bench lambdas,
  17d6fa25/718a388d/6a021263 bench and test harness: cosmetic for us.
- Gemma, DeepSeek (#1264, #1106, #1105), NPU/XRT (#1216, #1186, #1225, #1229, #1237,
  #1282), ASR/TTS (#1259, #1256, #1217, #1268), image (#1236, #1212, #861, #1214), H3
  (#998), HRX builds, Windows (#1191, #1215, #1204, #1189, #1119, #910, #1122), extensions
  (#1203, #1163, #1162, #1151, #1050, #1108), CI and releases: out of scope.

## 3. Suggested batch (implement → review → one combined GPU pass)

Batch A (small, about one day):
- #1249 with our raw hipHostMalloc sites, plus the docker memlock/seccomp change and a
  pinned-bytes log.
- #1230, #1146, #1273, #1168, #1102, #912, #1104, #1181, #1117.

All are clean cherry-picks or ports under 30 lines. Then one GPU pass:
- U4-3 hang_loop.
- MTP d0/d32K at compaction_proactiveness 20 and 50.
- 4-session MTP acceptance.
- functional `batch`, `cache`, tool-schema and pi_agent.

Batch B (medium): #1158, #1128, #1288/#1290, #1209, #1285, #1205.

Batch C (large, prefill): #1265 transient wide passes, in its own measured window.
