# Qwen3.8 Flash-Next tools

Model-specific benchmarks and analysis tools. The model code is in
[src/models/qwen38_flash_next/](../../src/models/qwen38_flash_next/README.md),
its tests in [tests/models/qwen38_flash_next/](../../tests/models/qwen38_flash_next/README.md),
and the EXL3 conversion in [tools/exl3_quant/](../exl3_quant/README.md).

| Path | Contents |
| --- | --- |
| `bench/*.cpp`, `bench/flash_next_*.hip`, `bench/gfx1151_exl3_wmma_ceiling.hip` | CMake targets `qwen38_flash_next_*_bench` (declared in `tests/models/qwen38_flash_next/CMakeLists.txt`, `EXCLUDE_FROM_ALL`). They link the production library: dense, F16 and routed GEMMs, prefill and decode attention, GDN prefill, stream overlap, decode HC projections, prefill HC combine and MoE glue, and the B7 EXL3 WMMA ceiling screen |
| `bench/fn_*_bench.hip`, `bench/exl3_x3_tiles_bench.hip`, `bench/exl3_k7_bench.hip` | Standalone W3/F3 experiments that `#include` a production translation unit (`hc.hip.cpp`, `dense_gemm.hip.cpp`, `exl3_moe.hip`, or for X3 tiles and K7 the shared `prefill_fp16.hip` and EXL3 GEMV sources) and compare variants against it. Build them with `tools/bench/build.sh <path>`. `fn_plain_tiles`, `fn_moe_glue`, `fn_moe_combine` and `fn_routed_gemv` no longer link on their own: the TU they include calls launchers defined in other sources (`LaunchPlainF16PrefillGemm`, the EXL3 fast-GEMV launchers); `fn_moe_combine` does not compile either. These failures predate B9 |
| `bench/exl3_prefill_shapes_bench.{hip,sh}` | EXL3 prefill shapes, linked against a configured build tree by the script |
| `bench/dense_blaslt_sweep.hip`, `bench/projection_plans.hip`, `bench/moe_gemm_i8_bench.hip` | Offline hipBLASLt shape sweeps and the int8 WMMA grouped-GEMM prototype (`tools/bench/build.sh <path>`) |
| `spec/` | W9 speculative-decoding analysis: MTP chain trace replay (`spec_sim.py`), policy sweeps, sensitivity, statistics, ABBA timing verdicts and the prompt suites (`suite_*.json`, built by `build_suites.py`) |
| `rq_quality/bf16_reference.py` | CPU, disk-streamed HF BF16 oracle for the R0 a01 windows (reuses `tools/qwen27b/rq_quality/`) |
| `f32_reference.py` | Writes and scores the F32 reference fixtures in `tests/fixtures/qwen38_flash_next/f32_reference/` |
