#!/usr/bin/env python3
"""CPU contract of tools/docker/pinned_rocm101.py: docker argv, environment and configure line."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "docker"))

import pinned_rocm101 as p  # noqa: E402

DUMP = {"variables": {
    "PATH": {"type": "exported", "value": "/nix/store/a-cmake/bin"},
    "CC": {"type": "exported", "value": "gcc"},
    "HOME": {"type": "exported", "value": "/homeless-shelter"},
    "TMPDIR": {"type": "exported", "value": "/tmp/nix"},
    "LOCAL": {"type": "var", "value": "x"},
    "ARR": {"type": "exported", "value": ["a"]},
    "cmakeFlags": {"type": "exported", "value": "-DA=1 '-DB=two words'"},
}}


class Test(unittest.TestCase):
    def test_environment(self):
        env, flags = p.environment(DUMP, base={"HOME": "/me", "KEEP": "1"})
        self.assertEqual(flags, "-DA=1 '-DB=two words'")
        self.assertEqual(env["HOME"], "/me")
        self.assertEqual(env["KEEP"], "1")
        self.assertEqual(env["CC"], "gcc")
        self.assertEqual(env["TMPDIR"], "/build/tmp")
        self.assertEqual(env["LD_LIBRARY_PATH"], p.ROCR_LIB)
        self.assertEqual(env["PATH"], "/nix/store/a-cmake/bin:/opt/venv/bin:/usr/bin:/bin")
        self.assertNotIn("LOCAL", env)
        self.assertNotIn("ARR", env)

    def test_configure_args(self):
        args = p.configure_args("-DA=1 '-DB=two words'", "abc123")
        self.assertEqual(args[:8], ["cmake", "-S", "/build/source", "-B", "/build/obj", "-G", "Ninja", "-DA=1"])
        self.assertIn("-DB=two words", args)
        self.assertIn("-DGUFO_REVISION=abc123", args)
        self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1151", args)
        self.assertIn("-DGUFO_ROCR_LIBRARY_DIR=" + p.ROCR_LIB, args)

    def test_cpu_argv(self):
        argv = p.docker_argv(Path("/b"), Path("/s"), ["cmake", "--build", "/build/obj"])
        joined = " ".join(argv)
        self.assertIn("--network none", joined)
        self.assertNotIn("/dev/kfd", joined)
        self.assertIn("--cpus 8", joined)
        self.assertIn("type=volume,src=gufo-nix,dst=/nix,readonly", argv)
        self.assertIn("type=bind,src=/b,dst=/build", argv)
        self.assertIn("type=bind,src=/s,dst=/build/source", argv)
        i = argv.index(p.IMAGE)
        self.assertEqual(argv[i - 1], p.PYTHON)
        self.assertEqual(argv[i + 1:], ["-u", p.SELF, "exec", "--", "cmake", "--build", "/build/obj"])

    def test_gpu_raw_argv(self):
        argv = p.docker_argv(Path("/b"), Path("/s"), ["--version"], gpu=True, source_readonly=True,
                             mounts=(("/m", "/models/fnx", True),), raw_entrypoint="/build/obj/gufo",
                             groups=[44, 990])
        joined = " ".join(argv)
        for part in ("--device /dev/kfd", "--device /dev/dri", "--ipc host", "--group-add 44",
                     "--group-add 990", "--memory 100g", "HOME=/tmp", "LD_LIBRARY_PATH=" + p.ROCR_LIB):
            self.assertIn(part, joined)
        self.assertIn("type=bind,src=/s,dst=/build/source,readonly", argv)
        self.assertIn("type=bind,src=/m,dst=/models/fnx,readonly", argv)
        self.assertEqual(argv[-3:], ["/build/obj/gufo", p.IMAGE, "--version"])

    def test_main_print_strips_separator_and_creates_nothing(self):
        import contextlib, io, tempfile
        with tempfile.TemporaryDirectory() as t:
            build = Path(t) / "absent"
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = p.main(["run", "--build-dir", str(build), "--source", t, "--print", "--",
                             "cmake", "--build", "/build/obj"])
            self.assertEqual(rc, 0)
            self.assertFalse(build.exists())
            self.assertTrue(out.getvalue().rstrip().endswith(
                p.SELF + " exec -- cmake --build /build/obj"), out.getvalue())

    def test_parse_mount(self):
        self.assertEqual(p.parse_mount("/a:/b:ro"), ("/a", "/b", True))
        self.assertEqual(p.parse_mount("/a:/b"), ("/a", "/b", False))
        for bad in ("/a", "/a:b", "/a:/b:rw", ":/b"):
            with self.assertRaises(ValueError):
                p.parse_mount(bad)
        with self.assertRaises(ValueError):
            p.mount("/a,b", "/c", False)


if __name__ == "__main__":
    unittest.main()
