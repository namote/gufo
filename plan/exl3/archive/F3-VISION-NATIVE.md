# Native EXL3 vision (2026-11-04)

User requested direct embedded EXL3 vision, eliminating the mandatory BF16
GGUF sidecar in Gufo. Kyojin remains the selected default service; it is
stopped for one combined qualification window and restored afterwards.

Implementation: validate native vision geometry/preprocessor and selected
safetensor shapes; split quantized QKV takes precedence over retained original
fused weights. Preserve padded FFN channels and native FP16 activations with
FP32 residuals, positional interpolation dtype and tanh merger activation.
Keep existing BF16 GGUF execution unchanged for explicit sidecars.
Use shared EXL3 rotations/prefill kernels for supported rates. Other supported
EXL3 formats decode into a bounded per-image inner-weight scratch slab and use
FP16 BLAS, retaining packed resident weights. No failed-path retry.

Checks: synthetic CPU loader/identity/malformed inputs and memory admission;
independent second review; incremental pinned Nix10.1 compile; one serialized
vision numerical/latency window plus image HTTP and text smoke. Reuse saved
legacy/reference artifacts where contract matches; no duplicate full baseline.
Raw EXL3 and GGUF BF16 are different weights/rounding, so equality between them
is not a quality gate. Same-pack upstream stage comparison must be explicit.

Artifacts: build/vision-exl3. Existing objects and source changes are preserved.
Qualification completed for the tested packs and fixtures:

- CPU native metadata/identity/padding/corruption tests, existing image input
  tests, and request memory budget tests pass. Independent review/re-review
  completed with no remaining blocking findings. Global C++ format: 770 files.
- Nix ROCm10.1 incremental build reused retained objects. Binary SHA256
  `62f0c59a15f3aee3aeb5336ab8e85a15406629dc89bc1cdce95d9a95672a3341`.
- One 320x224 RGB fixture, 280 patches: native Turboderp K6 and Vemloir K8
  compared against pinned local ExLlamaV3 vision-only execution. All 29 stages
  pass the preselected relative RMS <= 0.01 gate. Final embeddings 0.00602344
  and0.00621150 respectively. Upstream/source/environment provenance is in
  each REFERENCE.json. Not bit-exact; backend/FP16 rounding differs.
- The same fixture on candidate GGUF and retained old encoder matches all
  29 stage files byte-for-byte. Baseline computed once and saved.
- Full Gufo profile: native 256K, C2, shared 512K FP16, medium, MTP7, vision,
  0.0.0.0:18740. Run 2f22bfcb6617: API advertises image; mapped files contain
  no GGUF. Text 17+25=42; red-left/blue-right image correct. Identical repeat
  reuses 91/91 prompt tokens, prefills 0, answer unchanged. No new GPU error.
- Native K6 resident vision 352601088 B versus saved BF16 903984064 B: saves
  551382976 B (525.84 MiB). This reflects different stored precision; no general
  quality or speed superiority claim. Native request scratch is accounted
  separately, including the generic K8 decoded-inner slab.

Image `ai-tools/gufo-eval:native-exl3-vision-rocm101`, image ID
`sha256:e7a761a40187efb13515a0275362ab09d084a129f4544aa4300ec5cccb8c7468`.
Permanent Gufo launcher now selects this image and embedded vision without
`--mmproj`; previous controller is saved in
`~/.local/share/gufo/vision-exl3/gufo-server.before.py`.
Default remains Kyojin; restored run `873264740ec0` returned health OK.
No model/vision original files were deleted.

Limits: not a full image benchmark, high-resolution/multi-image sweep, 27B pack
qualification, or full 256K stress test. Vemloir was tested as vision-only;
its full Gufo language/MTP pack compatibility is not established here.
Candidate cold HTTP text/image/repeat wall observations are saved; not a
matched performance comparison. No full text baseline rerun.

Retained harness failures: first Docker nested read-only mount failed before
GPU execution; first controller retained a sidecar preflight assertion and
failed before loading; first HTTP harness name shadowed Python's http module
and was renamed before requests. These did not relax numerical gates.
