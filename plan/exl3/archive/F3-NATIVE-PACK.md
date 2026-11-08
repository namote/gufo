# Native EXL3 pack loading

User request, 2026-11-02: use the selected pack's native n-gram embedding and
remove the GGUF execution dependency. Completed: native 5.05 text/MTP qualification, image packaging and CPU
contract checks. Short-context MTP has a recorded workload-specific slowdown.

The directory entry point reads config/tokenizer/template directly and keeps
an owned typed metadata adapter for existing validators. It writes no GGUF and
never substitutes a legacy PLE source. Native `ngram_embedding.safetensors`
K1–K8 mul1 rings support a single tensor or sequential shard tensors, per-head
bias, existing bounded compressed-row cache and asynchronous readers. Rows
round to FP16 at the upstream embedding boundary before entering Gufo's FP32
buffers. Native MTP must use the trunk pack, including its local mixer patch
when the indexed weights omit that tensor. Persistent identity includes all
native source files and full weight payload digests, cached by file identity.

The 5.05 K6 n-gram is already on internal NVMe. The Kyojin K4 pack remains on
its existing external disk; CPU contract tests cover its sharded table. No
protected GGUF model, baseline or prior candidate is deleted. Optional vision
projector support remains separate; this is native text/MTP loading, not an
implementation of EXL3 vision encoding.

Applied: pinned Nix ROCm10.1, gfx1151, prior dense/MoE/prefill/TG kernels,
HIP graphs, MTP/lookup policies and KVarN choices. New candidate: native input
metadata, native PLE rows and full source identity. Deferred: moving PLE decode
to GPU, new kernel tuning, full local Kyojin comparison. Unsupported native
metadata/tokenizer contracts fail explicitly, without automatic fallback.

Validation: focused CPU codec/formula, shard/head boundaries, cache/dedup,
padding/short-file rejection, owned metadata moves, content identity and native
tokenizer/template checks. Then one guarded combined window with **no GGUF
mounts**: saved BF16-reference quality corpus, sampled/MTP/batch/deep replay,
plus prefill2048/8192 and TG128 AR/MTP depth0/32768. Reuse saved baseline records;
weights changed, so old output hashes are not an exact-output oracle. Retain
per-run timing, actual ELF/runtime maps, quality tails and hardware health.

## Qualification result

One guarded936.5s combined window completed; all eight commands and GPU health
passed, with zero `.gguf` mappings in every captured process map. Source base
`3667e1b2`; pinned ROCm10.1, prior optimized kernels retained. Candidate and
exported image ELF SHA are identical. Image:
`ai-tools/gufo-eval:f3-native-b4ebe27c-rocm101`, entry point
`/opt/gufo-native/bin/gufo`. Original images, models and incremental objects remain.
Only this export's unused BuildKit records were pruned; free disk returned above
150GiB. [Machine-readable evidence](F3-NATIVE-PACK.json).

CPU: K1–K8 in both layouts, independent bit-at-a-time ring reconstruction,
head/shard boundaries, compressed cache and duplicate rows, output guards,
padded-row/truncation rejection. The actual 5.05 K6 and Kyojin K4 files passed
33 and287 boundary-row checks (51,200 exact FP16-widened elements). Both packs'
metadata, NFC tokenizer, every added atom and compiled template passed. Owned
metadata lifetime/moves, source contract rejection and full-content identity
mutation/relocation checks passed. Legacy GGUF reader/identity, Qwen tokenizer
and n-gram regressions passed; shared clang-format21.1.8 check covered763files;
ai.tools infrastructure432tests passed. Independent review found no remaining
blocker after MTP pairing, tokenizer contract and source-stamp fixes.

Model checks passed sampled/EOS/RNG replay; C2/4/6/8 AR/MTP and lookup state;
32K independent/mixed execution and serialized replay; learned MTP policy and
cache restoration. Benchmark repeat hashes are stable and AR/MTP agree at both
depths. No saved baseline was executed again.

| Case | Saved IQ4_NL PLE | Native K6 PLE | Change |
|---|---:|---:|---:|
| Prefill2048 | 1435.59 | 1423.76 ±4.48 | −0.82% |
| Prefill8192 | 1466.18 | 1461.67 ±0.54 | −0.31% |
| AR128 d0 | 25.24 | 25.20 ±0.00 | −0.16% |
| AR128 d32768 | 24.51 | 24.43 ±0.01 | −0.33% |
| MTP128 d0 | 46.70 ±3.90 | 40.09 ±0.08 | −14.15% |
| MTP128 d32768 | 79.31 ±0.94 | 78.53 ±1.03 | −0.98% |

All rates are tok/s, r2, KVarN k4v4. TG uses C1, context32897, temperature0,
seed314159. PP uses maximum context8193; the new PP list omits the old initial
512 case, so the first2048 cache history is not perfectly matched. This is a
native-weight transition, not a proof that all performance differences arise
from the loader. The CLI's tensorless metadata size field currently displays
0.00GiB; it is not model residency or the native artifact's file size.

Short MTP output changed (`b2c54baf…`→`64c0a14b…`); accepted drafts88/113→74/114
(77.88%→64.91%), cycles40→54. The lower completed-token rate is material and is
retained, without reverting to a different PLE table. The unchanged neural
kernels remain much faster than the older pre-TG/MTP baseline. At32K the output,
17cycles and111accepted tokens remain identical, including101lookup copies:
that rate is a repetitive lookup-heavy workload, not ordinary prose throughput.
Further MTP acceptance tuning must use the native pack and real prompts.

Quality reuses the saved BF16 reference and saved IQ4_NL PLE candidate records:
32760 scored positions, eight4095-row windows,64 full-head replays. Full-vocab
NLL2.175202→2.167751 (PPL8.80396→8.73861); pairedΔNLL−.007451,
window95%CI[−.017370,−.001770]. KL256+.rest decreases.026481→.025464;
its paired windowCI spans zero. Top-1 agreement92.0543%→91.9750% (−.0794pp),
windowCI[−.3175,+.2260]pp. KL>1 top-1 flips17; all category/tail records remain
in the report. These support this transition; they do not establish exact
full-model HF parity or universal task quality superiority. Kyojin's full model
and end-to-end speed have not been newly qualified.
