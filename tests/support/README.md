# Shared test support

Headers that tests of several models include:

| File | Used by |
| --- | --- |
| `kvarn_fake_quant.hpp` | KVarN fake-quantization reference (Qwen27B, Qwen HIP attention and Flash-Next KVarN tests, the KVarN decode bench) |
| `sampling_cases.hpp` | Shared sampling cases (CLI logit sampler, Qwen27B and Flash-Next sampling and serving tests) |
| `temporary_directory.hpp` | A self-removing temporary directory for tests |
