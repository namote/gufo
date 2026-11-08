# Gufo: The ***Open Source*** Strix Halo AI engine 

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufo logo" width="180">
</p>

Gufo is a vertical local inference engine for Linux and Windows, specifically built and optimized for AMD Strix Halo hardware:
Ryzen AI MAX+ 395 systems with Radeon 8060S (`gfx1151`), up to 128 GiB of unified memory.

**Contributions are welcome!**

See the [changelog](CHANGELOG.md) and [GitHub Releases](https://github.com/gufo-org/gufo/releases)
for user-facing changes and release history.

> [!TIP]
> Check out **[Gufo RDMA for Dual Strix Halo](https://github.com/neuhaus/gufo)**, a Gufo variant that has not been merged yet!

## This fork: native EXL3 on Strix Halo

[`namote/gufo`](https://github.com/namote/gufo) is an independent fork. It
adds native [EXL3](https://github.com/turboderp-org/exllamav3) (trellis)
inference to Gufo, tuned for Qwen3.8 Flash-Next on one `gfx1151`.

**Qwen3.8 Flash-Next, EXL3 c03 pack (4.04 bpw over the main weights), FP16 KV, one user**

| Workload | Start (Nov 7) | Now (B10, Nov 8) | Gain |
| --- | ---: | ---: | ---: |
| Prefill 2,048 tokens, MTP off | 1,471.5 tok/s | **1,516.3 tok/s** | +3.0% |
| Prefill 8,192 tokens, MTP off | 1,499.7 tok/s | **1,549.3 tok/s** | +3.3% |
| Decode, AR, empty context | 27.31 tok/s | **32.87 tok/s** | +20.4% |
| Decode, AR, 32K context | 26.75 tok/s | **32.56 tok/s** | +21.7% |
| Decode, MTP, empty context | 51.36 tok/s | **67.91 tok/s** | +32.2% |
| Decode, MTP, 32K context¹ | 87.62 tok/s | **104.69 tok/s** | +19.5% |

The numbers come from `gufo bench` on pinned Nix ROCm 10.1: 128 generated
tokens, temperature 0, two repetitions per cell.
- **Start** is the baseline of the first batch (`f24b10ea`); **Now** is the
  B10 candidate, which is the binary the server now runs.
- Each of the ten batches was timed against the batch before it, in the same
  window. The Gain column therefore chains those steps; Start and Now were not
  measured side by side.
- Quality against the BF16 reference stayed within noise at every step: KL\@256
  moved from 0.0415 to 0.0419, and top-1 agreement is 90.9%.

¹ The 32K MTP prompt is repetitive, so n-gram lookup drafts carry much of it.
It is not general long-context throughput.

Runs and rejected attempts are recorded in
[the benchmarks](docs/models/qwen3.8-flash-next/BENCHMARKS.md),
[the gfx1151 know-how](docs/GFX1151-KNOWHOW.md) and
[the B-series summary](plan/exl3/B-SERIES-SUMMARY.md).

**What the fork adds**

- **Native EXL3 packs.** Gufo loads ExLlamaV3 pack directories for two
  models:
  - Flash-Next: the trunk, routed and shared experts, the MTP head, the n-gram
    embeddings and the embedded vision tower.
  - Qwen3.8 27B: the text projections.

  On the GPU, the trellis lane decoder matches the CPU dequantizer bit for bit
  at K1–K8 and K1.5–K3.5. Production projections accept integer K1–K8, and
  routed experts also accept K3.5.
- **EXL3 kernels for gfx1151.**
  - Decode and MTP verification use trellis GEMVs that keep the next tile
    group in flight.
  - The decode HC down projection runs as W8A8 on resident Q8_0 copies.
  - Prefill uses WMMA GEMMs over expert buckets, with 128-token buckets for
    hot experts.
  - Router, restore and combine glue is fused into the neighbouring kernels,
    and side branches overlap the shared MLP and the attention indexer.
- **Quantized KV cache.** `--kv-cache-type` accepts `fp16`, `kvarn-k4v2`,
  `kvarn-k4v4` and `kvarn-k6v6`, where the model stores them. Flash-Next
  takes `fp16`, `kvarn-k4v4` and `kvarn-k6v6`; Qwen3.8 27B defaults to
  `kvarn-k4v2`. KVarN types store rotated quantized records after an FP16 sink
  group.
- **Block-paged, shared KV.** Attention KV lives in 256-token blocks in one
  physical pool.
  - Sessions map blocks only when they write them, and the number of sessions
    does not depend on the pool size.
  - Requests with a common prompt prefix share that prefix's complete
    physical blocks, copy-on-write.
  - With K4V4, two 256K contexts fit in a 4.62 GiB pool; FP16 needs 13.4 GiB.
    Admission is tested at that capacity, but full two-times-256K inference
    has not been run.
- **Long context.** Flash-Next has served native 256K contexts per
  conversation, with K6V6 on the 5.05 pack and FP16 on c03. No full-256K
  stress test has been run.
- **Persistent continuations.** Finished conversations are kept in a RAM cache
  and a disk cache in the session's KV format (FP16, K4V4 or K6V6) and are
  restored on the next turn. One measured K4V4 disk restore (5.05 pack) served
  12,352 cached tokens plus 7 prefilled tokens in 0.486 s, with identical
  output.
- **MTP that pays off.**
  - Draft chains stop when draft confidence drops.
  - KV-only catch-up keeps the draft head in step with the trunk.
  - Verification computes only the rows that are still live.
  - GDN verify state is written once, for the kept prefix.
- **Request memory budgeting.** Before a request starts, admission reserves
  its future KV blocks and its checkpoint memory, so a request that would not
  fit is refused up front.
- **Reproducible tooling in the repo.**
  - [`tools/exl3_quant`](tools/exl3_quant/README.md): the Flash-Next
    conversion pipeline and a pinned, hash-verified Kyojin v1.8.6 converter.
  - [`tools/docker`](tools/docker/README.md): the pinned ROCm 10.1 build and
    GPU-run container.
  - [`tools/serving/flash_next`](tools/serving/flash_next/README.md): the
    serving supervisor, with an installer that records its git revision and
    file hashes.
  - [`tools/qwen38_flash_next`](tools/qwen38_flash_next/README.md): the
    model's benches, probes and specs.
- **Readable model code.** Flash-Next code is split by stage:
  - [`kernels/rocm`](src/models/qwen38_flash_next/kernels/rocm/README.md)
    holds HC, GDN, attention, MoE glue, MTP and the GEMMs.
  - [`runtime/`](src/models/qwen38_flash_next/runtime/README.md) holds the
    executor.
  - After the split, all 7,059 device symbols were identical to the build
    before it.

This fork's results are not upstream claims. Nothing here is proposed back
upstream.

## Models and benchmarks

All model documentation lives under [docs/models](docs/models/README.md):

| Model | Inference modes | Hugging Face weights | Benchmarks | Quality |
| --- | --- | --- | --- | --- |
| [Qwen3.8 27B](docs/models/qwen3.8-27b/README.md) | Q4/Q8, images, AR, DFlash2 | Unsloth [Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf) / [Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q8_K_XL.gguf) · [DFlash2 Q4_K_M](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/blob/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) | Q4: **648.89 tok/s pp**; up to **71.55 tok/s tg** single user and **128.42 aggregated tok/s** on 8 concurrent requests with DFlash2 · [Benchmarks](docs/models/qwen3.8-27b/BENCHMARKS.md) | [Quality](docs/models/qwen3.8-27b/QUALITY.md) |
| [Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next/README.md) | Q4, images, AR, MTP | Unsloth [Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q4_K_XL) · [MTP Q8_0](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf) | **1,801.01 tok/s pp** (AR); up to **116.93 tok/s tg** single user and **245.81 aggregated tok/s** on 8 concurrent requests with MTP · [Benchmarks](docs/models/qwen3.8-flash-next/BENCHMARKS.md) | [Quality](docs/models/qwen3.8-flash-next/QUALITY.md) |
| [DeepSeek V4 Flash](docs/models/deepseek-v4-flash/README.md) | AR, DSpark | [antirez Flash 0731 IQ2XXS](https://huggingface.co/antirez/deepseek-v4-gguf/blob/1cd7b564460821938add0475a60b942c409295e0/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf) · [DSpark](https://huggingface.co/antirez/deepseek-v4-gguf/blob/e7f04037032990db0346398d249baf9fb9df1ccc/DeepSeek-V4-Flash-DSpark-support-0731.gguf) | **484.62 tok/s pp**; up to **26.62 tok/s tg** single user and **54.74 aggregated tok/s** on 8 concurrent requests with DSpark · [Benchmarks](docs/models/deepseek-v4-flash/BENCHMARKS.md) | [Quality](docs/models/deepseek-v4-flash/QUALITY.md) |
| [Qwen3-ASR 1.7B](docs/models/qwen3-asr/README.md) | Speech recognition | [BF16](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5) | **15.27× realtime** · [Benchmarks](docs/models/qwen3-asr/BENCHMARKS.md) | [Quality](docs/models/qwen3-asr/QUALITY.md) |
| [EmbeddingGemma 2](docs/models/embeddinggemma-2/README.md) | Text and image embeddings | [BF16](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b) | Up to **41.5k tokens/s** over HTTP; **13.0 ms** for a 404-token request · [Benchmarks](docs/models/embeddinggemma-2/BENCHMARKS.md) | [Quality](docs/models/embeddinggemma-2/QUALITY.md) |
| [Qwen3-TTS 1.7B](docs/models/qwen3-tts/README.md) | Speech synthesis and voice cloning | BF16 [CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) / [VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) / [Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Up to **2.54× realtime**; **201 ms** to first audio (CustomVoice) · [Benchmarks](docs/models/qwen3-tts/BENCHMARKS.md) | [Quality](docs/models/qwen3-tts/QUALITY.md) |
| [Qwen-Image-2.1](docs/models/qwen-image-2.1/README.md) | BF16 image generation and editing | [Complete pipeline](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/b3179ad355be050328e483a9dfdd9e60cd62adfa) | In progress · [Benchmarks](docs/models/qwen-image-2.1/BENCHMARKS.md) | [Quality](docs/models/qwen-image-2.1/QUALITY.md) |
| [MiniMax H3](docs/models/minimax-h3/README.md) | BF16 text to video/audio | [FL2VA pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA) | In progress · [Benchmarks](docs/models/minimax-h3/BENCHMARKS.md) | [Quality](docs/models/minimax-h3/QUALITY.md) |

Peak measured workloads; text pp is autoregressive (AR), while tg uses the
named speculative mode. Peaks include repetitive output; aggregate tg sums
individual request decode rates. Qwen27B's single-user peak uses the short-prompt
C1 workload. Audio excludes loading.
Each model guide lists the required files and complete benchmark settings.

## Philosophy

- Contributions are welcome! We need the help of Strix Halo community to keep improving gufo!
- We would like this to be the one-stop shop for Strix Halo Local AI enthusiasts: batteries included for text, audio, image, and video models.
- Build and optimize specifically for the Strix Halo 128 GiB hardware. Smaller memory configurations should still work and preserve the speed benefits for models that can fit on memory.
- Support only the best available models for their size that can run on this hardware: less code to maintain, more focused optimization and testing work.
- Preserve quality when optimizing. Each model's quality report records independent numerical checks, execution consistency and unresolved gaps. Don't reuse kernels across different models to limit blast radius of a code change.
- Treat concurrent requests, cancellation and conversation caching as first-class workloads.
- Keep production dependencies small and development tools separate.

## Quickstart

```sh
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-GGUF
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF \
  Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --revision 2d9571f8ce46e151f61c6499c99dee6079e1d610 \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-DFlash2-GGUF
podman pull ghcr.io/gufo-org/toolboxes/gufo-runtime:latest
podman run --rm \
  --userns=keep-id:uid=1000,gid=1000 \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add keep-groups \
  --ulimit memlock=-1 \
  -p 8080:8080 \
  -v ./models:/models:ro \
  ghcr.io/gufo-org/toolboxes/gufo-runtime:latest \
  gufo serve --host 0.0.0.0 --port 8080 llm \
  --model /models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --speculative dflash2 \
  --dflash-model /models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

Rootless Podman needs `crun` for `--group-add keep-groups`. Your host user must
have read/write access to `/dev/kfd` and `/dev/dri/renderD*`, usually through the
`render` and `video` groups; log out and back in after changing membership.
Container groups named `video`/`render` do not preserve host supplementary
groups. Check `id`, `ls -l /dev/kfd /dev/dri/renderD*`, and
`podman info --format '{{.Host.OCIRuntime.Name}}'` if ROCm reports no device.
See [Podman's rootless group-access guidance](https://github.com/containers/podman/blob/main/troubleshooting.md#20-passed-in-devices-or-files-cant-be-accessed-in-rootless-container).

On Fedora or another SELinux-enforcing host, GPU enumeration can succeed while
SELinux blocks mapping `/dev/kfd`, causing ROCr to report a misleading
“Memory critical” error. Check the **host** audit log:

```sh
sudo ausearch -m avc -ts recent | grep -E '/dev/kfd|hsa_device_t'
```

If it shows a denied `map` for the container, Podman documents this fix:

```sh
sudo setsebool -P container_use_devices true
```

This persistently allows containers to access device labels for devices passed
into them; it affects all containers on that host. Review that policy scope
before enabling it. See [Podman's device documentation](https://docs.podman.io/en/latest/markdown/podman-run.1.html#device-host-device-container-device-permissions)
and the [SELinux container policy](https://github.com/containers/container-selinux/blob/main/container.te).

Then, from another terminal, ask it something through the OpenAI-compatible API:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-UD-Q8_K_XL",
    "messages": [{"role": "user", "content": "Say something"}]
  }'
```

For Open WebUI, VS Code, OpenAI SDK and Responses-API clients (for example
Codex), set the API base URL to `http://localhost:8080/v1`. Chat Completions and
Responses support text, images, function tools, structured output and streaming.
See the [API contract](docs/SERVER.md).

The text server uses the model's native context by default and generates until
EOS or the context is full. `--context N` sets context capacity per session;
`--max-tokens N` sets a default response limit that clients can override.
Reasoning tokens count toward that response limit.

## Harness notes

- **Qwen with Codex:** Codex sends developer messages mid-conversation, after
  compaction or a settings change. Qwen's template accepts only one leading
  system turn, so Gufo moves them there. The request that introduces one is
  prefilled again; later requests reuse the cache. See the
  [API contract](docs/SERVER.md).
- **Qwen with Claude Code:** not supported. Claude Code adds a system message
  to `messages` on every turn, so Gufo moves a new one to the leading system
  turn each time and every request prefills the whole history again (#522).

## Build from source

Production packages target Linux x86-64 on AMD Strix Halo (`gfx1151`).
Windows x86-64 is a supported native build target for development; see
[BUILD-WINDOWS.md](BUILD-WINDOWS.md). CMake owns
one production configuration for both Nix and ordinary Linux builds. Tests,
profilers, tuning executables and Python reference runners are not installed
with the production package. No `build.sh` wrapper is needed.

### With Nix

```sh
nix build
./result/bin/gufo diagnose
./result/bin/gufo serve llm --model /path/to/model.gguf
```

[flake.lock](flake.lock) pins the dependencies. `nix develop` adds profiling,
model-download and independent evaluation tools; these are not runtime
requirements. Optional benchmark baselines are selected separately with
`nix shell .#ds4-reference`, `.#llama-cpp-reference` or
`.#llama-cpp-mtp-reference`; see [benchmarking](docs/BENCHMARKS.md).
See [testing](docs/TESTING.md) for the small hosted CI suite and
explicit local quality checks.

Prebuilt store paths are published to the `gufo` Cachix cache. The flake
declares it in `nixConfig`: users listed in `trusted-users` are asked to accept
it, or pass `--accept-flake-config`. Nix ignores it for other users; add it
system-wide with `cachix use gufo` or in `/etc/nix/nix.conf`:

```
extra-substituters = https://gufo.cachix.org
extra-trusted-public-keys = gufo.cachix.org-1:rfZe3n4lHINgyoACKzs5EthIZmBNvEyCCtufUpth0PE=
```

The Nix package and development shell use ROCm 10.1.0 (HIP Clang 24)
from the pinned [nixpkgs ROCm 10.1 update](https://github.com/NixOS/nixpkgs/pull/573885).
Python, Python packages and general build dependencies come from the pinned
`nixos-unstable` input. The ROCm package set comes from the separate fork,
retaining its compiler/runtime dependencies to reuse the gfx1151 binaries on
`gufo.cachix.org`. All Nix workflows configure this cache explicitly.
Both include a [queue-memory patch](.devops/nix/patches/rocr-fixed-cwsr-rocm10.patch)
for Strix Halo. It keeps queue save/restore buffers, launch/copy arenas and
code-loading staging resident during host memory compaction, while retaining
managed-memory support and the ordinary fine-grained host-allocation policy.

### Without Nix

Install a C++20 compiler, CMake 3.21+, Ninja, pkg-config and the
following development libraries. The pinned Nix toolchain uses GCC
16.2 and ROCm 10.1.0 (HIP Clang 24). Attention and audio convolution kernels are compiled
directly from HIP. Python, Triton/AOTriton, Composable Kernel and MIOpen are
not production build or runtime requirements.
For the same protection against queue evictions as the Nix build, apply the
ROCr patch above when building your runtime. Flash-Next's host buffers also use
Linux long-term pins, with ordinary HIP registration retained when system
limits disallow those pins. Gufo leaves `HSA_USERPTR_FOR_PAGED_MEM` unchanged.

| Dependency | Used for |
| --- | --- |
| ROCm HIP compiler/runtime, hipBLAS, hipBLASLt, rocBLAS | GPU execution and matrix multiplication |
| hipCUB, rocPRIM, rocWMMA headers | Compiled GPU kernels |
| ICU, libcurl, OpenSSL, libpng, libjpeg, libwebp | Tokenization, HTTPS, hashing and images |
| libsndfile, libsoxr | Audio decoding and reference-compatible resampling |
| FFmpeg and ffprobe | Video/audio output; invoked as separate executables |

Install ROCm using [AMD's Linux instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).
Use the development packages for the libraries above. ROCm normally installs
under `/opt/rocm`.

For example, on Debian/Ubuntu the ordinary system libraries are:

```sh
sudo apt install build-essential cmake ninja-build pkg-config \
  libicu-dev libcurl4-openssl-dev libssl-dev libpng-dev libjpeg-dev libwebp-dev libsndfile1-dev libsoxr-dev ffmpeg

# ROCm libraries from the table, named as AMD's repository ships them.
sudo apt install hipblas-dev hipblaslt-dev rocblas-dev \
  hipcub-dev rocprim-dev rocwmma-dev

cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
./build/release/gufo serve llm --model /path/to/model.gguf
```

Configuring fails at `find_package(hipblas)` when those ROCm packages are
missing. Other distributions name them `-devel` instead of `-dev`.
For nonstandard installations, pass ordinary CMake paths, for example
`cmake --preset release -DCMAKE_PREFIX_PATH="/opt/rocm"`.
If compiler discovery picks a system Clang, also pass
`-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++`.
Use `cmake --install build/release` to install Gufo,
its runtime data and license notices. The GPU driver must allow your user to
access `/dev/kfd` and `/dev/dri`; model weights are acquired separately.

The same source, compiler flags and install rules serve both builds. Nix pins
the complete toolchain for reproducible comparisons; changing the compiler or
math libraries requires the affected model's quality checks.

## License

Gufo's original code is [MIT licensed](LICENSE). Adapted code and dependencies
retain their own notices in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and `licenses/`, installed under `share/licenses/gufo`. Model weights are not
bundled and retain their publishers' terms.

## Reference Projects

The initial design is informed by the following open source projects:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for compact model serving, GGUF, and CPU/GPU correctness paths.
- [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp), [Nathanw1014/strix-halo-llamacpp](https://github.com/Nathanw1014/strix-halo-llamacpp), and [gaetan-puleo/llama-cpp-strix-halo](https://github.com/gaetan-puleo/llama-cpp-strix-halo) for Strix Halo optimization inspiration.
- [vLLM](https://github.com/vllm-project/vllm) for continuous batching and paged request scheduling.
- [hipEngine](https://github.com/shisa-ai/hipEngine) for torch-free HIP execution, and native speculative-cycle work.
- [ds4](https://github.com/antirez/ds4) for DeepSeek V4 Flash, MoE scheduling, and DSpark.
- [audio.cpp](https://github.com/0xShug0/audio.cpp) for audio models for tts and asr tasks.
