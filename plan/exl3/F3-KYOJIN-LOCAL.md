# Local Kyojin execution, November 2, 2026

Kyojin ran on the local gfx1151 with its complete model and native n-gram on
internal NVMe. All 12 requests completed and GPU health passed. **The paired
greedy AR/MTP token and text equality check failed on all three prompts.**
These are observed speeds of that configuration, not an accepted replacement
or a claim of quality parity. Gufo native `b4ebe27c` remains selected.

| Workload | AR tok/s | MTP tok/s | First divergent token (zero-based) |
|---|---:|---:|---:|
| Prose, 256 output tokens | 31.30 | 42.52 | 57 |
| Chat, 256 output tokens | 31.33 | 49.44 | 76 |
| Code, 256 output tokens | 31.28 | 63.22 | 149 |

Real-text AR prefill: 2048 tokens **1550.88 tok/s**, 8192 tokens **1386.46 tok/s**,
from server phase timing after one warm pass each. These inputs differ from
Gufo's fixed-pattern PP fixture. Decode uses `(completion_tokens - 1)` divided
by first-to-last useful SSE event time. One measured pass per case; no baseline
rerun or confidence interval. Model load/JIT is excluded from request timings.

## Exact scope and failure

Frozen upstream revision `2132bd248e3bfe07258a408669608f3d2cfdda79` (1.8.3),
unmodified source, approximately 4 bpw pack with K4 PLE, context 16384,
sessions 1, quant K4V4, `draft-policy mix`, no vision/thinking/uncensor,
temperature 0, seed 314159, no prompt-cache hits. A single resident target/MTP
served AR then MTP requests; each pair differs only in `speculative`.
Independent receipt review confirmed 256 token IDs per response and real text
differences. Actual accepted/rejected drafts: 138/105, 160/61 and 194/36.

This establishes failure of exact AR/MTP equivalence in this run. It does not
isolate the cause, prove deterministic reproduction, or quantify general quality
loss. No extra GPU run was used to turn this result into a pass. The window's
`FAILED_no_fallback` means the equality gate failed; it was not a process crash
(container exit 0, GPU health PASS). No automatic alternate execution occurred.

Actual runtime paths belong to Kyojin's source-pinned AMD ROCm
`10.1.0a20260822` wheel SDK and built extension. Torch reports HIP `7.16.26332`
and the compiler is AMD clang23. This is **not Gufo's pinned Nix ROCm10.1
release/clang24 runtime**, although its container supplies the base Python.
Kyojin used `trellis_disk` native K4 n-gram (24.4 GiB), HC sidecar 96 sites
plus 1 plain site, and `draft2s.st`. Startup was 172.05 seconds including
warmup; the dense warmup has a 90-second budget, so first-use tuning outside
that budget remains possible. No swap was observed.

The saved Gufo K-suite MTP rates on these three prompts were
35.41/41.99/56.70 tok/s. The observed Kyojin rates are 20.1%/17.7%/11.5% higher,
but that Gufo record used 5.05 weights, old IQ4 PLE, KVarN and context49152.
It is a historical cross-configuration comparison, not current native parity
or an isolated engine speedup. Full quality/long-context comparison remains undone.

## Local storage and retained artifacts

User-authorized removal freed **106.28 GiB** from five unused internal
Flash-Next GGUF payloads. Their external originals, saved checksums and
provenance remain. No native Gufo pack was removed.
Kyojin's **88.82 GiB / 43 manifest files** were copied in 265.69 seconds to
`/home/m2on/projects/ai.tools/storage/models/llm/flash-next-exl3-kyojin`.
This includes `ngram_embedding.safetensors`; the external source is retained.
Source-byte hashes were checked against prior HF identities while streaming
the copy; destination sizes were checked and files fsynced. This is not a
second destination hash pass. About **166 GiB** remains free after retaining
wheel/runtime caches. The first external load was user-stopped before any
benchmark request; its health check passed and its stop receipt is preserved.

The 458.03-second CPU-only build and its native extension, source and SDK
are retained in ai.tools
`test/experiments/gufo_upstream_port_20261023/kyojin_runtime/`.
JIT caches remain in the completed receipt (`receipt/triton-cache/` and
`receipt/home/.cache/`). Inputs/wheels/source manifests identify the reusable bundle. Gufo Nix objects,
images, existing baselines and results are preserved. Infrastructure checks:
**432 passed**. No Gufo inference code changed.

[Machine-readable results and receipt hashes](F3-KYOJIN-LOCAL.json).
Raw receipts under ai.tools `tmp/test-results/gufo-upstream-port-20261023/`:
`kyojin-local-prepare/`, `kyojin-local-rocm101/` (user-stopped), and
`kyojin-local-ssd-rocm101/` (completed measurements, failed equality).
