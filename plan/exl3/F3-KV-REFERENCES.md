# Flash-Next physical GPU prefix sharing (2026-11-04)

User requested the Gufo27B shared prefix capability for Flash-Next. Kyojin
remains the default service (C2, native256K per request, FP16 shared512K pool).
Gufo qualification temporarily owns the GPU and restores Kyojin afterwards.

## Contract

The shared allocator now also shares physical KV blocks across checkpoints and
branches. Retained component spans alias one reference-counted block owner;
restore maps those IDs into the destination block table. Partial/mixed-source
blocks copy on write, including all target, draft and pooled-indexer planes.
The stable device block-table pointer remains valid for captured graphs.

Only blocks complete across target, draft and both pooled-key frontiers are
retained. This deliberately leaves the writable frontier private, including
the dense-to-sparse indexer backlog. Capturing after row reservation cannot add
future COW debt. Short dense prefixes (up to 2048 tokens) still use host row
checkpoints. Recurrent GDN, convolution, PLE, raw indexer, KVarN sink/tail and
MTP hidden/residual/policy state remain private exact snapshots.

KVarN chunks use one 128-token group because append row zero denotes physical
group one (group zero is the private FP16 sink). Pooled keys use 64 rows per
256-token GPU block; FP16 key/value chunks use 256 rows. Layout identity/version
is 4; old v3 disk checkpoints are not reused. Numerics and model weights are
unchanged.

Shared ownership and progressive admission are separate capabilities. Admission
converts compressed component rows into token ranges, intersects retained ranges
across every plane, then counts complete physical blocks. Host-only early chunks
do not hide later GPU references; partial blocks separated by gaps cannot be
combined into fictitious capacity. Whole-checkpoint restoration and monotonic
component history preserve coherent physical ownership across inherited planes.

## Qualification

Incremental Nix ROCm10.1 build and four linked targets pass. Scheduler, runner,
component-cache CPU suites pass (34.83s final total), including retained+progressive
admission and non-token component row units. Continuation layout and repository
format check (767 files) pass. Independent agent review found no remaining
blocking code issue after allocation-failure ordering fixes.

Core combined GPU run passes:88 boundary/reference checks,32760 numerical
records byte-identical to saved K6, standard PP/AR/MTP and GPU health. Fixtures cover actual retained pointers, reverse
component restore, nearly-full-pool admission, reservation-before-capture with
all unreserved blocks occupied, source release, divergent child branches and
AR/MTP boundary replay for K4, K6 and FP16. Saved K6 numerical/speed baselines
are reused. Artifacts and exact commands: `build/fn-kv-sharing/`.

## Reclaim correction found by branch/image validation

The first two branch runs passed text branches and uncached controls, but the
following image request was incorrectly rejected with HTTP400. Its body was
`Flash-Next shared KV pool reservation exhausted`, not an oversized prompt.
The generic HTTP length-error mapping mislabeled this as context exhaustion.
ReclaimRows protected every token-prefix match, including independent physical
copies from cold histories. It now protects only the actual selected checkpoint
or live slot, using the same input identity and stable boundary as admission.
Other candidate pins are released before reclamation. CPU regression covers
token-equal text/image histories; second review passes. Failed attempts are
preserved in gpu/http-branches/ and gpu/http-branches-a02/. Only the affected
branch/vision history and unstarted Pi/disk checks are resumed with a03; passed
core/numerical/speed and C4 memory-pressure checks are not repeated.

## Final serving result

Final image `ai-tools/gufo-eval:fn-kv-references-a03-rocm101` is selected in the
permanent Gufo launcher; its previous controller and all candidate binaries are
preserved in `build/fn-kv-sharing/`. Kyojin remains the default and is restored
with FP16, shared512K, native256K per conversation, C2, medium and MTPmix7.

C4 memory qualification passes20 completed requests, four unlimited streams
and three pressure streams: waiter25 resumes after guarantor cancellation.
Long9637-token cached replay, greedy/seeded MTP C1 repeats, C4 progress, vision
and medium pass. Final a03 branch fixture passes13 requests: separate and
parallel children reuse4845+ common tokens, uncached controls agree, image
parent/child/cold agree, and the previous text branch survives subsequent cold
histories. A real restart restores4867/4867 prompt tokens from disk with zero
prefill and the same answer (0.198s request wall observation).

Pi1.5.0 passes2 tasks/7 model requests/6 actual tool calls: write/cat/read a file;
write a CommonJS mean/median module and four assertions, then run Node with
exit0 and `All tests passed`. The retained session and tool results were
inspected. No GPU errors in qualification; all candidate processes terminate.
Only disposable qualification continuation payloads were removed after results
were saved. Final receipt: `build/fn-kv-sharing/RESULT.json`.

Limits: no new full256K replay, task-quality benchmark or matched idle-machine
speed claim. Core reference fixtures reach4K; standard decode covers32K depth;
HTTP uses16K contexts with a two-context pool. Private recurrent and incomplete
frontier state remains per conversation; sharing reduces occupancy inside the
fixed pool rather than its backing allocation size. Dense prefixes up to2048
still use host checkpoint rows to protect the future pooled-indexer backlog.
