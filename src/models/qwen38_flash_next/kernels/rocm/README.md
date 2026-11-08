# Flash-Next HIP kernels

**Start with [`kernels.hpp`](kernels.hpp)**: every launcher the runtime calls,
grouped by stage, with its layout and rounding contract. Each translation unit
holds one stage:

| File | Stage |
| --- | --- |
| `embed_norm.hip.cpp` | Token embedding, RMSNorm rows |
| `hc.hip.cpp` | Hyper-connection mix epilogues, combines, deferred residual |
| `activation.hip.cpp` | SiLU, SwiGLU, sigmoid gates, F32 to F16/BF16 narrowing |
| `small_gemm.hip.cpp` | Small-batch GGUF GEMMs and the stable HC down route |
| `w8a8.hip.cpp`, `w8a8_wave64.hip.cpp` | Q8_0 x Q8 activation GEMMs (wave32 and the wave64 variant), activation tiling |
| `dense_gemm.hip.cpp` | Dense F16/BF16 WMMA GEMM and its fused epilogues (QKV + norm + RoPE, HC up/mix, SSM projection, EXL3 plain prefill) |
| `routed_f16.hip.cpp` | Routed expert GEMMs over GGUF weights |
| `exl3_moe.hip.cpp` (+ `exl3_moe.hpp`, `exl3_prefill_combine.hpp`) | EXL3 routed expert decode GEMVs and prefill buckets, MoE combine |
| `moe_glue.hip.cpp` | Router top-k, expert counts, expert-output epilogues |
| `recurrent.hip.cpp` | Gated DeltaNet, SSM convolution, PLE gate/conv/inject, rolling history |
| `attention_prep.hip.cpp` | Q/gate unpack, RoPE, K/V stores, KVarN rotation and quantization |
| `attention_select.hip.cpp` | Sparse-attention indexer: key rows, pooled blocks, block selection |
| `attention.hip.cpp` | Selected-block decode attention and WMMA causal prefill attention |
| `mtp.hip.cpp` | MTP head glue, argmax, draft logits, candidate selection |
| `kernels_internal.hpp` | Private device helpers shared by the TUs above |
| `mmq/` | Quantized GEMMs adapted from llama.cpp ([VENDOR.md](mmq/VENDOR.md)) |

Device code must stay byte-identical when code moves between these files; see
the note at the top of `kernels_internal.hpp`.
