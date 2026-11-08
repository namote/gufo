# Flash-Next host runtime

**Start with [`executor.hpp`](executor.hpp)**: the `Executor` (one resident
model) and its `Session`s. The implementation is split by stage:

| File | Contents |
| --- | --- |
| `device_model.{hpp,cpp}` | Uploads GGUF or EXL3 weights to the GPU |
| `blaslt.{hpp,cpp}` | hipBLASLt plans for dense 16-bit projections |
| `executor.cpp` | Executor and session lifetime, sizing, K/V rows, rollback depth, KVarN planning, side stream, diagnostics |
| `executor_forward.cpp` | The forward pass: graph capture/replay, per-layer body, speculative commit and rollback |
| `executor_dense.cpp` | Dense projections: Q8 inputs, GGUF/F16/EXL3 routes, rotated-input slabs |
| `executor_hc.cpp`, `executor_ple.cpp` | Hyper-connection combine and mix; per-layer embeddings |
| `executor_gdn.cpp`, `executor_attention.cpp`, `executor_moe.cpp` | Linear-attention, full-attention and MoE layers |
| `executor_mtp.cpp` | MTP drafting and verification |
| `executor_snapshot.{hpp,cpp}` | Session snapshots, prefill checkpoints, the saved-session format |
| `executor_batch.cpp` | Batched decode across sessions |
| `executor_internal.hpp` | Private helpers shared by the executor files |
