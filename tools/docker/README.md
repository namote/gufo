# Pinned ROCm 10.1 container

**Start with [`pinned_rocm101.py`](pinned_rocm101.py)**: the one definition of
the image, Nix volume, environment and `docker run` flags that the Flash-Next
B1–B9 builds and GPU passes used. `build.sh` runs an incremental CPU build (or
`--run CMD`) in it, and `gpu-run.sh` runs one recorded GPU command. Usage,
mounts and limits are described in
[tools/README.md](../README.md#pinned-rocm-101-container); the CPU contract test
is `tests/tools/pinned_rocm101_test.py`.
