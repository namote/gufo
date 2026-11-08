# Flash-Next shared KV and durable K4V4 continuation

User scope: Pi serves native5.05/K6 on Nix ROCm10.1, C4, 262144 tokens per
session, KVarN k4v4, a shared physical pool of two contexts and 50 GiB of stored
continuations. No weight precision, kernel arithmetic or TG/MTP policy change.

## Ownership and representation

The existing KvBlockPool owns physical record, MTP FP16 and pooled-index planes.
Each session owns exclusive mapped blocks, a stable device table and pinned host
table, plus its recurrent, convolution, PLE, raw indexer and KVarN sink/tail state.
Admission reserves future blocks; writes map only required rows before KVarN
flush or graph replay. Reset/destruction drains work before releasing mappings.
Successful restoration trims unreachable blocks. Checkpoints copy into host
backing and do not retain device block references; admission accounts for this.

A KVarN append row is a completed group, starting at group1. Group0 is the FP16
sink. Transfers split at physical boundaries, including the group1 offset;
private state contains canonical valid sink/tail rows, with unused bytes zero.
Restoration reconstructs the same conservative rewind frontier as typed snapshots.
Compatibility identity includes codec, value order, group size and layout schema3.
Shared sessions use component snapshots; legacy typed snapshots reject them.
KVarN in-pass checkpoints remain unsupported. Persistent continuation storage is
not transparent swapping of active attention pages.

Production geometry is 2420736 bytes per 256-token block. A 256K context requires
1025 blocks including pooled-index guard; two contexts require2050 blocks,
4962508800 bytes (4.62 GiB). Session count is independent of physical pool size.
The50 GiB disk limit is a retention quota, not a preallocation.

## Verification record (completed, November 2, 2026)

- Incremental host objects link against retained qualified Nix10.1 kernel objects;
  previous binaries and intermediates remain under build/f3-tgmtp3.
- Pinned clang-tools21.1.8 shared formatting check passed764 files.
- CPU continuation inventory passes FP16 and KVarN boundaries0/127/128/129,
  255/256/257, sparse2048 boundary and context endpoints.
- Independent reviewer approved mapping/guard/copy design and two fixes:
  geometry-derived context guard count and short-restore block release.
- Shared GPU fixture passed92.04s: AR/MTP, four slots/two-context capacity,
  reservation exhaustion/release, interleaved physical blocks, nine boundaries,
  exact component recapture, private-session greedy/seeded output comparison,
  short restoration, cancellation and complete block return.
- Numerical collector32760 positions passed153.23s;101424960 output bytes equal
  saved fn-tgmtp3 baseline. No baseline execution was repeated.
- First combined launch stopped after the fixture because the Python driver
  lacked /workspace on PYTHONPATH. Only the unstarted checks resumed.
- Sampling aggregate stopped at a mixed EOS assertion combining token policy and
  physical batch width. The old log cannot identify the failed clause. Focused
  rerun with diagnostics passes:1 stopped token,8 continuing tokens, all trailing
  tokens EOS, widths2/2. No production change between these runs. Original failed
  evidence is retained; focused EOS and serving sampling pass without a production
  EOS change. The initial failure is not proven harmless noise.

Evidence root: ai.tools/tmp/test-results/gufo-upstream-port-20261023/
fn-shared-kv-nix101, fn-shared-kv-qualify-nix101,
fn-shared-kv-remaining-nix101. Build recipes/logs: build/f3-shared-kv.

- Remaining combined qualification passed: serving sampling100.4s, mixed batches
  C2/C4/C6/C8 174.9s, deep state32768 83.4s, MTP RAM/serialized cache and policy
  replay165s; PP/AR/MTP53.3/99.2/93.9s. Completed windows healthPASS.
- Final shared fixture103.511s also covers a completely saturated two-context
  pool: both4095-token sessions admit and evaluate their final token exactly.
  CPU admission tests cover zero rows, pooled-index guards and context ceiling.
- Adjacent logical transfer runs coalesce only with contiguous physical and
  payload offsets. Private sessions retain their original one-copy behavior.
- Final capability guards disable unsupported in-pass checkpoints for KVarN
  and shared sessions in both adapter and native execution; zero byte estimates
  remain zero. No kernel arithmetic change. Independent review approved.
- Final image serves C4 with eight first/repeat HTTP requests passing. The API
  long-context suite passes at12359 tokens; after process restart12352 tokens
  restore from disk and only7 prefill, identical output,0.486s.
- Real Pi1.4.0 regression passes3 tasks/9 tool calls/12 HTTP requests, including
  file writes/reads, code edits and independent Python result checks. The harness
  accepts an explicit version while preserving its historical0.87.0 default.
- Full two-times-256K inference was not run. Capacity is configured and boundary
  admission is tested; HTTP uses12K and native deep-state qualification32K.

## Saved performance comparison

Native5.05/K6, Nix ROCm10.1, KVarN k4v4; previous baseline reused, not rerun.
These standard model measurements establish arithmetic/performance continuity;
shared serving is additionally covered by the fixtures and HTTP cases above.

| Scope | Saved tok/s | Candidate tok/s |
| --- | ---: | ---: |
| PP2048 | 1426.88 | 1425.09 |
| PP8192 | 1460.75 | 1462.00 |
| AR128 d0 | 25.20 | 25.20 |
| AR128 d32768 | 24.49 | 24.47 |
| MTP128 d0 | 40.85 | 40.92 |
| MTP128 d32768 | 79.48 | 79.27 |

Maximum absolute difference0.264%; all output hashes exact. The d32K MTP
fixture is lookup-heavy, not a general prose speed claim. Final guard-only
changes were checked with CPU/shared fixture and final-image HTTP/Pi rather
than another complete benchmark cycle.

## Current local execution and provenance

Image `ai-tools/gufo-eval:f3-shared-kv-4757e04f-rocm101`, image ID
`sha256:222716db02d82ecff916c25a0322ef5722245a5381572ebb76345cc509bccba1`,
ELF `4757e04fa0cc57066bcb33588026cdf4ef3fb13709777fe6fffe1ff30b6b201d`.
Built incrementally from9c96f9d6 plus this change; build recipes, object/link
manifests and source diff identity remain in `build/f3-shared-kv/`.
The old unit driver's source_revision is a stale ignored workspace stamp;
use the image/ELF and captured source identity, not that field.

`/home/m2on/projects/pitest`: `./server status`, `./server start`, `./pi`.
Server binds0.0.0.0:18740; Pi connects127.0.0.1:18740/v1. C4, per-session
context262144, shared pool4962508800B, RAM cache4GiB, disk quota50GiB.
Native embedded n-gram and actual mapped Nix HIP/ROCr10.1 verified; no GGUF.
Persistent identity cache reduces startup85.34s cold to35.15s warm. Each server
window lasts one hour under the existing resource/health supervisor.

Additional evidence: `fn-shared-kv-final-nix101`, `fn-shared-kv-http` including
RESULT/PERFORMANCE, C4, continuation-disk, long-context-disk and Pi reports.
Permanent receipts: ai.tools `test/experiments/gufo_upstream_port_20261023/`
`shared_kv_artifacts/`. Preserve old qualified images and Nix objects.
