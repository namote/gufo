# Qwen3.8 Flash-Next (`qwen4exp`)

Model package for Qwen3.8-Flash-Next: hyper-connection (HC) residual streams,
Gated DeltaNet and sparse-attention layers, routed MoE with a shared expert,
per-layer embeddings (PLE), MTP drafting, and GGUF or native EXL3 weights.
Everything here is in `gufo::models::qwen38_flash_next`; the GPU code is in its
`rocm` namespace. User documentation, benchmarks and quality records are in
[docs/models/qwen3.8-flash-next/](../../../docs/models/qwen3.8-flash-next/).

| Path | Contents |
| --- | --- |
| `config.*`, `rope_scaling.*`, `native_metadata.*` | Architecture parameters from GGUF or a native EXL3 pack, static YaRN, native pack metadata |
| `weights.*`, `exl3_weights.*`, `exl3_ngram.*`, `ngram.*` | Host weight views (GGUF and EXL3 descriptors) and the PLE n-gram embedding tables |
| `engine.*` | The Gufo-facing API: one resident model, sessions, shared projection batches |
| `continuation_*` | Continuation-cache adapter: state inventory, capacity checks and hooks |
| `mtp_*.hpp`, `draft_vocab.hpp`, `continuation_hooks.hpp` | MTP draft policy, cost table, sampling and the draft vocabulary; host callbacks of adapter-owned sessions |
| `reference.*`, `cpu_ops.*` | Scalar F32 reference model (`gufo_qwen38_flash_next_reference`, built only by validation targets) |
| [`kernels/rocm/`](kernels/rocm/) | HIP device code. `kernels.hpp` declares the model kernel launchers, `exl3_moe.hpp` the EXL3 routed expert launchers and `exl3_prefill_combine.hpp` the EXL3 prefill MoE combine. The `*.hip.cpp` files hold one stage each: `embed_norm`, `hc`, `activation`, `small_gemm`, `w8a8`, `routed_f16`, `dense_gemm`, `recurrent` (GDN, SSM convolution, PLE), `attention_prep`, `attention_select` (indexer), `attention` (decode and WMMA prefill), `moe_glue` and `mtp`. They share the private `kernels_internal.hpp`. `exl3_moe.hip` holds the EXL3 routed expert GEMMs, and `w8a8_wave64.hip` the wave64 W8A8 GEMM |
| `kernels/rocm/mmq/` | Quantized GEMM kernels adapted from llama.cpp in the `qfn_mmq` namespace; see its [VENDOR.md](kernels/rocm/mmq/VENDOR.md) |
| [`runtime/`](runtime/) | Host runtime. `device_model.*` uploads the model and `blaslt.*` wraps hipBLASLt. `executor.hpp` is the executor. Its stages are split into `executor.cpp` (lifetime, sessions, K/V rows), `executor_snapshot.cpp`, `executor_dense.cpp`, `executor_moe.cpp`, `executor_hc.cpp`, `executor_ple.cpp`, `executor_gdn.cpp`, `executor_attention.cpp`, `executor_forward.cpp`, `executor_mtp.cpp` and `executor_batch.cpp` (batched multi-session decode). They share the private `executor_internal.hpp` and `executor_snapshot.hpp` |

Tests are in [tests/models/qwen38_flash_next/](../../../tests/models/qwen38_flash_next/README.md).
Benchmarks and model tools are in
[tools/qwen38_flash_next/](../../../tools/qwen38_flash_next/README.md), and the
EXL3 conversion pipeline is in [tools/exl3_quant/](../../../tools/exl3_quant/README.md).

Device code is compiled with `-O3 -ffast-math -fno-finite-math-only`. Moving a
kernel or a shared helper between translation units can change its code,
because interprocedural optimization sees a different set of callers. After
such a move, compare the per-kernel code objects. See the note at the top of
`kernels/rocm/kernels_internal.hpp`.
