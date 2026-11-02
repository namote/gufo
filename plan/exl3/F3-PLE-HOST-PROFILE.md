# Native PLE host timing, November 2, 2026

The old GPU profiles begin at the first EmbedKernel and do not measure the
preceding CPU work. Ordinary token embeddings run on GPU; native PLE rows
are fetched and decoded on CPU, including compressed-cache hits. PLE output
uses coherent pinned memory read directly by the GPU quantizer, without a
separate embedding upload. This distinction was missing from the earlier
bottleneck attribution.

A bounded diagnostic on native `1c16241f`, Nix ROCm10.1 and the internal-NVMe
5.05/K6 pack timed Forward, PleFetch, WaitPle, PrefetchPle, StartRead and
WaitRead. Copied source files add only host RAII clocks/logging; production
source, arithmetic, graph eligibility and synchronization remain unchanged.
Two CPU translation units were rebuilt and linked with the preserved native,
batch6 and TG/MTP kernel objects. The production image remains selected.

The generic harness RESULT.json carries inherited source/binary labels and is
not authoritative for this mounted diagnostic executable. Use INPUTS.json,
MANIFEST.json and the report's `inputs`: diagnostic ELF SHA256
`80f861d0` (prefix; full digest in JSON), with copied-source and compile/link
hashes. The mounted diagnostic ELF differs intentionally from the production
image ELF; this run does not qualify a new production image.

| Phase | Overall time | PLE host time | Share |
|---|---:|---:|---:|
| PP2048 measured | 1420.88 ms | 6.54 ms | 0.46% |
| PP8192 measured, two chunks | 5558.23 ms | 25.48 ms | 0.46% |
| AR128 decode, prefix excluded | 5073.33 ms | 88.31 ms | 1.74% |
| MTP128 decode, prefix excluded | 3182.50 ms | 53.73 ms | 1.69% |

PLE host time sums the nonoverlapping PleFetch, WaitPle and PrefetchPle scopes;
StartRead/WaitRead are nested and are not added again. PP denominator is the
Forward sum; TG denominator is the full benchmark interval reconstructed from
its rounded rate, including draft/acceptance work outside Forward. PP warmup
and TG16-token prefix are retained separately in the JSON. First untimed
PP2048 host PLE was3.89ms/1464.94ms; this is not a controlled cold-cache trial.

AR and MTP outputs both match saved native SHA256
`64c0a14b57711d22c337c8a496f6fd1c7805d751c51b722909cdcae195ec5748`.
MTP retains54cycles,114drafts,74accepted and0lookup. All three commands and
hardware health pass; pinned ROCr/CLR and no GGUF mappings are verified.
Diagnostic rates1441.35/1473.84,25.23,40.22tok/s are not new clean baselines.
No old baseline or full numerical qualification was repeated.

These measurements do not support CPU PLE waiting as the dominant cause of
the large throughput gap on these inputs. They do not rule out other embedding
costs: host waiting overlaps GPU prefix execution; logging changes scheduling;
worker/SSD shared-bandwidth interference and GPU reads of mapped embeddings
are unmeasured. CPU decode versus SSD service is not separately timed. The
fixed-pattern PP and shallow generation do not establish long-context prose
or cold-table behavior. Avoid extrapolating the percentages to those cases.

Preserve the production path. Before a larger loader rewrite, use a real
prose/cold-row trace to attribute worker decode, I/O and GPU quantizer traffic,
with buffered timing records and GPU activity alignment. Cache hits still
re-decode, small cached gathers decode synchronously, and native scalar ring
unpacking/FP16 conversion are concrete candidates only if that trace justifies
their cost. Large gains should continue to be sought in the separately measured
matrix/draft/verify work rather than assumed from this CPU path.

[Numbers, identities, raw receipt hashes and limitations](F3-PLE-HOST-PROFILE.json).
Receipt: `/home/m2on/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/fn-ple-profile-nix101`.
Independent reviewer `indexer_review` checked instrumentation, ABI/relink,
runner and analysis; nested timing and logging caveats are incorporated.
