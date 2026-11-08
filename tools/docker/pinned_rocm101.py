#!/usr/bin/env python3
"""Pinned ROCm 10.1 build and GPU run environment for gufo (standard library only).

Every Flash-Next B1-B7 build and GPU pass ran in one retained closure: the
`ai-tools/gufo-eval:native-exl3-vision-rocm101` image (IMAGE) with the
`gufo-nix` Docker volume mounted read-only at /nix. The volume holds the gufo
development environment (ENV, a `nix print-dev-env --json` dump), the ROCm
10.1 runtime (ROCR_LIB), glibc and the GCC prefix the HIP compiler links
against. This module was the `final_current_build.environment()` and
`run_stage.py` Docker flags of the retired ai.tools experiment
`flash_next_quant_20261104`; it is now the one source for both.

Host side (builds a `docker run` command line and executes it):

  pinned_rocm101.py run [--gpu] [--build-dir DIR] [--source DIR]
                        [--mount SRC:DST[:ro]]... [--raw ENTRYPOINT] -- CMD...

  Without --raw, CMD runs through `pinned_rocm101.py exec` inside the
  container, with the pinned environment. With --raw ENTRYPOINT, CMD is the
  argument list of that entrypoint, run with only the GPU runtime variables.

Container side:

  pinned_rocm101.py exec -- CMD...      run CMD in /build/source with the env
  pinned_rocm101.py configure-args REV  print the pinned CMake configure line

The build directory is mounted at /build (objects in /build/obj) and the
source tree at /build/source. CPU runs have no network and no GPU devices.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

IMAGE = "sha256:e7a761a40187efb13515a0275362ab09d084a129f4544aa4300ec5cccb8c7468"
IMAGE_TAG = "ai-tools/gufo-eval:native-exl3-vision-rocm101"
NIX_VOLUME = "gufo-nix"
ENV = "/nix/store/65ksrp4rfgpz1wfyxdqfs3syl2m13p80-gufo-0-unstable-dirty-env"
ROCR_LIB = "/nix/store/88jjsz00km321bcl9851aiwwdig79b4l-rocm-runtime-10.1/lib"
GLIBC = "/nix/store/pqyiidz5mi73wv5f91d48s9plcqyj5bz-glibc-2.44-57"
GCC_PREFIX = "/nix/store/l3msfz6rylwsyhnjzsbr8ylfshh1xw5f-gcc-prefix"
PYTHON = "/opt/venv/bin/python3"
SELF = "/build/source/tools/docker/pinned_rocm101.py"
CPU_LIMITS = ("--memory", "16g", "--memory-swap", "16g", "--cpus", "8")
GPU_LIMITS = ("--memory", "100g", "--memory-swap", "100g")
DROPPED = {"HOME", "PWD", "OLDPWD", "TMPDIR", "TMP", "TEMP", "NIX_BUILD_TOP"}


def environment(env_dump: dict | None = None, base: dict | None = None) -> tuple[dict, str]:
    """Return (process environment, cmakeFlags) of the pinned closure."""
    if env_dump is None:
        env_dump = json.loads(Path(ENV).read_text())
    data = env_dump["variables"]
    env = {k: v["value"] for k, v in data.items()
           if v["type"] == "exported" and isinstance(v["value"], str) and k not in DROPPED}
    env.update(NIX_BUILD_TOP="/build", LD_LIBRARY_PATH=ROCR_LIB, TMPDIR="/build/tmp", TMP="/build/tmp",
               TEMP="/build/tmp", PYTHONDONTWRITEBYTECODE="1", XDG_CACHE_HOME="/build/cache")
    env["PATH"] = env.get("PATH", "") + ":/opt/venv/bin:/usr/bin:/bin"
    return {**(os.environ if base is None else base), **env}, data["cmakeFlags"]["value"]


def configure_args(flags: str, revision: str) -> list[str]:
    """The RelWithDebInfo configure line used for /build/obj."""
    return ["cmake", "-S", "/build/source", "-B", "/build/obj", "-G", "Ninja", *shlex.split(flags),
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            f"-DCMAKE_HIP_FLAGS=--gcc-toolchain={GCC_PREFIX} -B{GLIBC}/lib",
            f"-DCMAKE_EXE_LINKER_FLAGS=-L{GLIBC}/lib -Wl,--dynamic-linker,{GLIBC}/lib/ld-linux-x86-64.so.2"
            f" -Wl,-rpath,{GLIBC}/lib",
            "-DCMAKE_C_COMPILER=gcc", "-DCMAKE_CXX_COMPILER=g++", "-DBUILD_TESTING=ON",
            "-DGUFO_BUILD_TOOLS=OFF", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCMAKE_HIP_ARCHITECTURES=gfx1151",
            "-DGUFO_REVISION=" + revision, "-DGUFO_RELEASE_VERSION=development",
            "-DGUFO_ROCR_LIBRARY_DIR=" + ROCR_LIB, "-DCMAKE_INSTALL_PREFIX=/build/install",
            "-DPython3_EXECUTABLE=" + PYTHON, "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -g -gz=zlib -DNDEBUG",
            "-DCMAKE_HIP_FLAGS_RELWITHDEBINFO=-O2 -g -gz=zlib -DNDEBUG"]


def gpu_groups(dev: Path = Path("/dev")) -> list[int]:
    """Group ids owning /dev/kfd and the render nodes, for --group-add."""
    nodes = [dev / "kfd", *sorted((dev / "dri").glob("renderD*"))]
    return sorted({p.stat().st_gid for p in nodes if p.exists()})


def mount(src: str, dst: str, readonly: bool) -> list[str]:
    if "," in src or "," in dst:
        raise ValueError(f"mount paths may not contain commas: {src}:{dst}")
    return ["--mount", f"type=bind,src={src},dst={dst}" + (",readonly" if readonly else "")]


def parse_mount(text: str) -> tuple[str, str, bool]:
    parts = text.split(":")
    if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] != "ro") or not all(parts[:2]):
        raise ValueError(f"expected SRC:DST[:ro], got {text!r}")
    if not parts[1].startswith("/"):
        raise ValueError(f"container path must be absolute: {text!r}")
    return str(Path(parts[0]).resolve()), parts[1], len(parts) == 3


def docker_argv(build_dir: Path, source: Path, command: list[str], *, gpu: bool = False,
                source_readonly: bool = False, mounts: tuple[tuple[str, str, bool], ...] = (),
                raw_entrypoint: str | None = None, groups: list[int] | None = None) -> list[str]:
    """`docker run` argv for one pinned CPU or GPU container."""
    argv = ["docker", "run", "--rm", "--pull", "never", "--network", "none", "--user", "1000:1000"]
    if gpu:
        argv += [*GPU_LIMITS, "--device", "/dev/kfd", "--device", "/dev/dri", "--ipc", "host"]
        for gid in gpu_groups() if groups is None else groups:
            argv += ["--group-add", str(gid)]
    else:
        argv += CPU_LIMITS
    argv += ["--tmpfs", "/tmp:rw,size=2g,mode=1777"]
    if gpu:
        argv += ["-e", "HOME=/tmp", "-e", "LD_LIBRARY_PATH=" + ROCR_LIB]
    argv += ["--mount", f"type=volume,src={NIX_VOLUME},dst=/nix,readonly"]
    argv += mount(str(build_dir), "/build", False)
    argv += mount(str(source), "/build/source", source_readonly)
    for src, dst, readonly in mounts:
        argv += mount(src, dst, readonly)
    if raw_entrypoint is None:
        argv += ["--entrypoint", PYTHON, IMAGE, "-u", SELF, "exec", "--", *command]
    else:
        argv += ["--entrypoint", raw_entrypoint, IMAGE, *command]
    return argv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="host: run CMD in a pinned container")
    run.add_argument("--gpu", action="store_true", help="add /dev/kfd, /dev/dri, render groups and 100 GiB")
    run.add_argument("--build-dir", type=Path, required=True, help="host directory mounted at /build")
    run.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2],
                     help="host source tree mounted at /build/source (default: this repository)")
    run.add_argument("--source-readonly", action="store_true")
    run.add_argument("--mount", action="append", default=[], metavar="SRC:DST[:ro]")
    run.add_argument("--raw", metavar="ENTRYPOINT", help="run CMD as arguments of ENTRYPOINT, without the env")
    run.add_argument("--print", action="store_true", help="print the docker command instead of running it")
    run.add_argument("command", nargs=argparse.REMAINDER)
    ex = sub.add_parser("exec", help="container: run CMD with the pinned environment")
    ex.add_argument("command", nargs=argparse.REMAINDER)
    cfg = sub.add_parser("configure-args", help="container: print the pinned configure command")
    cfg.add_argument("revision")
    a = ap.parse_args(argv)
    command = a.command[1:] if getattr(a, "command", [])[:1] == ["--"] else getattr(a, "command", [])
    if a.action == "configure-args":
        print(shlex.join(configure_args(environment()[1], a.revision)))
        return 0
    if not command:
        ap.error("missing command after --")
    if a.action == "exec":
        env, _ = environment()
        for name in ("tmp", "cache"):
            Path("/build", name).mkdir(exist_ok=True)
        return subprocess.run(command, cwd="/build/source", env=env).returncode
    if not a.print:
        a.build_dir.mkdir(parents=True, exist_ok=True)
    full = docker_argv(a.build_dir.resolve(), a.source.resolve(), command, gpu=a.gpu,
                       source_readonly=a.source_readonly, mounts=tuple(parse_mount(m) for m in a.mount),
                       raw_entrypoint=a.raw)
    if a.print:
        print(shlex.join(full))
        return 0
    os.execvp(full[0], full)
    return 127


if __name__ == "__main__":
    sys.exit(main())
