# Qwen3.8 Flash-Next tools

Benchmarks and analysis tools for one model. **Start with `bench/`**: each file
there is a CMake target named `qwen38_flash_next_<file stem>`. The model code
is in [src/models/qwen38_flash_next/](../../src/models/qwen38_flash_next/README.md),
its tests in [tests/models/qwen38_flash_next/](../../tests/models/qwen38_flash_next/README.md),
and the EXL3 conversion in [tools/exl3_quant/](../exl3_quant/README.md).

| Path | Contents |
| --- | --- |
| `bench/<topic>_bench.{cpp,hip}` | Benchmarks of the production library, declared `EXCLUDE_FROM_ALL` in `tests/models/qwen38_flash_next/CMakeLists.txt` (`qwen38_flash_next_<topic>_bench`): `dense_gemm`, `f16_gemm`, `routed_gemm`, `attn`, `decode_attn`, `gdn_prefill`, `overlap`, `hc_decode`, `hc_prefill`, `moe_glue`, and `exl3_wmma_ceiling` (the B7 screen) |
| `bench/experiments/` | Standalone W3/F3 experiments and offline sweeps (`<topic>_bench.hip`), kept as the evidence behind recorded decisions. They are built one at a time with `tools/bench/build.sh <path>`. `hc_combine`, `plain_tiles`, `moe_glue`, `moe_combine` and `routed_gemv` `#include` a production translation unit (`hc.hip.cpp`, `dense_gemm.hip.cpp` or `exl3_moe.hip.cpp`) and compare variants against it. `prefill_tile_variants` (X3 tiles) and `dense_decode_verify` (K7 decode/verify) include the shared EXL3 prefill and GEMV sources. `prefill_shapes_bench.sh` links `prefill_shapes_bench.hip` against a build tree. `dense_blaslt_sweep`, `projection_plans` and `moe_gemm_i8` are hipBLASLt shape sweeps and an int8 WMMA prototype. `plain_tiles`, `moe_glue`, `moe_combine` and `routed_gemv` do not link on their own, because the TU they include calls launchers defined elsewhere, and `moe_combine` does not compile. These failures predate B9 |
| `coding_eval/` | R5 coding steering set tools: model answers to Python problems, hidden edge-case tests in a Docker sandbox, and pass-rate comparison between settings |
| `spec/` | W9 speculative-decoding analysis: MTP chain-trace replay (`spec_sim.py`), policy sweeps, sensitivity, statistics, ABBA timing verdicts and the prompt suites (`suite_*.json`, built by `build_suites.py`) |
| `rq_quality/bf16_reference.py` | CPU, disk-streamed HF BF16 oracle for the R0 a01 windows (reuses `tools/qwen27b/rq_quality/`) |
| `f32_reference.py` | Writes and scores the F32 reference fixtures in `tests/fixtures/qwen38_flash_next/f32_reference/` |
