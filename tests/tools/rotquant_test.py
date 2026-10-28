#!/usr/bin/env python3
"""Synthetic-tensor checks for tools/rotquant (no model files, no GPU).

Run with NumPy and PyYAML: python3 -I tests/tools/rotquant_test.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from gufo import gguf as shared_gguf  # noqa: E402
from rotquant import convert  # noqa: E402
from rotquant.gguf_io import OutTensor, U32, as_float32, encode_value, open_gguf, write_gguf  # noqa: E402
from rotquant.hadamard import (BLOCK_SIZE, bf16_bits_to_f32, f32_to_bf16_bits,  # noqa: E402
                               rotate_last_axis, signs, sylvester)
from rotquant.plan import (PlanError, build_plan, check_rotation, load_groups,  # noqa: E402
                           quantize_command, read_rot_signs, tensor_type_lines)
from rotquant.recipe import RecipeError, parse_recipe, parse_rotate_selection  # noqa: E402

GROUPS = load_groups("qwen27b")
ALL_GROUPS = GROUPS.names
HIDDEN = 256
TOKENS = ["<a>", "<b>", "한글"]


def kv_entries(extra=()):
    entries = [
        ("general.architecture", encode_value("qwen35")),
        ("general.name", encode_value("toy")),
        ("qwen35.block_count", encode_value(U32(2))),
        ("tokenizer.ggml.tokens", struct.pack("<IIQ", 9, 8, len(TOKENS)) +
         b"".join(struct.pack("<Q", len(t.encode())) + t.encode() for t in TOKENS)),
    ]
    return entries + list(extra)


def toy_tensors(rng, *, tied=False, attn_output_k=384):
    """A two-layer qwen35-shaped model: DeltaNet layer 0, attention layer 1."""
    def bf16(n, k):
        return "BF16", (k, n), f32_to_bf16_bits(rng.standard_normal((n, k)).astype(np.float32) * 0.05)

    def f32(*ne):
        return "F32", ne, rng.standard_normal(int(np.prod(ne))).astype("<f4")

    t = {"token_embd.weight": bf16(64, HIDDEN), "output_norm.weight": f32(HIDDEN)}
    if not tied:
        t["output.weight"] = bf16(64, HIDDEN)
    for layer in (0, 1):
        p = f"blk.{layer}."
        t[p + "attn_norm.weight"] = f32(HIDDEN)
        t[p + "post_attention_norm.weight"] = f32(HIDDEN)
        if layer == 0:
            t[p + "attn_qkv.weight"] = bf16(384, HIDDEN)
            t[p + "attn_gate.weight"] = bf16(256, HIDDEN)
            t[p + "ssm_alpha.weight"] = bf16(4, HIDDEN)
            t[p + "ssm_beta.weight"] = bf16(4, HIDDEN)
            t[p + "ssm_out.weight"] = bf16(HIDDEN, 256)
            t[p + "ssm_a"] = f32(4)
            t[p + "ssm_conv1d.weight"] = f32(4, 384)
            t[p + "ssm_dt.bias"] = f32(4)
            t[p + "ssm_norm.weight"] = f32(64)
        else:
            t[p + "attn_q.weight"] = bf16(512, HIDDEN)
            t[p + "attn_k.weight"] = bf16(128, HIDDEN)
            t[p + "attn_v.weight"] = bf16(128, HIDDEN)
            t[p + "attn_output.weight"] = bf16(HIDDEN, attn_output_k)
            t[p + "attn_q_norm.weight"] = f32(64)
            t[p + "attn_k_norm.weight"] = f32(64)
        t[p + "ffn_gate.weight"] = bf16(512, HIDDEN)
        t[p + "ffn_up.weight"] = bf16(512, HIDDEN)
        t[p + "ffn_down.weight"] = bf16(HIDDEN, 512)
    return t


def out_tensors(tensors):
    result = []
    for name, (type_name, ne, data) in tensors.items():
        payload = np.ascontiguousarray(data).tobytes()
        result.append(OutTensor(name, tuple(ne), type_name, lambda f, b=payload: f.write(b)))
    return result


def write_split(directory, tensors, shards=2, extra_kv=()):
    """Write a gguf-split style set: metadata in shard 0, split keys everywhere."""
    names = list(tensors)
    per = -(-len(names) // shards)
    paths = []
    for index in range(shards):
        split_kv = [("split.no", struct.pack("<IH", 2, index)),
                    ("split.count", struct.pack("<IH", 2, shards)),
                    ("split.tensors.count", struct.pack("<Ii", 5, len(names)))]
        kv = (kv_entries(extra_kv) if index == 0 else []) + split_kv
        chunk = {n: tensors[n] for n in names[index * per:(index + 1) * per]}
        path = Path(directory) / f"toy-BF16-{index + 1:05d}-of-{shards:05d}.gguf"
        write_gguf(path, kv, out_tensors(chunk), 32)
        paths.append(path)
    return paths


def write_single(path, tensors, extra_kv=(), kv=None):
    write_gguf(path, kv if kv is not None else kv_entries(extra_kv), out_tensors(tensors), 32)
    return Path(path)


def run_cli(*args):
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = convert.main([str(a) for a in args])
    return code, stdout.getvalue(), stderr.getvalue()


class HadamardTest(unittest.TestCase):
    def test_orthogonal_symmetric_normalized(self):
        h = sylvester(BLOCK_SIZE)
        np.testing.assert_allclose(h @ h.T, np.eye(BLOCK_SIZE), atol=1e-12)
        np.testing.assert_array_equal(h, h.T)
        np.testing.assert_allclose(np.abs(h), 1 / np.sqrt(BLOCK_SIZE), rtol=0, atol=1e-15)
        np.testing.assert_allclose(h @ h, np.eye(BLOCK_SIZE), atol=1e-12)

    def test_signs_are_deterministic_per_instance(self):
        a = signs(7, "attn_in:0", 512)
        self.assertTrue(np.isin(a, (-1.0, 1.0)).all())
        np.testing.assert_array_equal(a, signs(7, "attn_in:0", 512))
        self.assertFalse(np.array_equal(a, signs(7, "attn_in:1", 512)))
        self.assertFalse(np.array_equal(a, signs(8, "attn_in:0", 512)))
        with self.assertRaises(ValueError):
            signs(7, "x", 200)

    def test_rotation_preserves_matmul_fp32(self):
        rng = np.random.default_rng(1)
        for k in (128, 640, 5120):
            w = rng.standard_normal((48, k)).astype(np.float32)
            x = rng.standard_normal((k, 3)).astype(np.float32)
            d = signs(3, f"k{k}", k)
            w_rot = rotate_last_axis(w, d)                 # W·D·H
            x_rot = rotate_last_axis(x.T, d).T             # H·D·x
            y_ref = w.astype(np.float64) @ x.astype(np.float64)
            y = (w_rot @ x_rot).astype(np.float64)
            rel = np.abs(y - y_ref).max() / np.abs(y_ref).max()
            self.assertLess(rel, 1e-5, f"K={k}")
            # The rotation is invertible: (W·D·H)·H·D = W.
            inverse = (w_rot.astype(np.float64).reshape(48, k // 128, 128) @ sylvester()).reshape(48, k) * d
            np.testing.assert_allclose(inverse, w, atol=1e-5)

    def test_bf16_storage_error_is_rounding_level(self):
        rng = np.random.default_rng(2)
        k = 5120
        w = bf16_bits_to_f32(f32_to_bf16_bits(rng.standard_normal((64, k)).astype(np.float32) * 0.02))
        x = rng.standard_normal(k).astype(np.float32)
        d = signs(4, "bf16", k)
        w_rot = rotate_last_axis(w, d)
        stored = bf16_bits_to_f32(f32_to_bf16_bits(w_rot))
        # Elementwise: round-to-nearest BF16 is within half an ulp (2^-8 relative).
        self.assertTrue((np.abs(stored - w_rot) <= np.abs(w_rot) * 2.0 ** -8).all())
        y_ref = w.astype(np.float64) @ x
        y = stored.astype(np.float64) @ rotate_last_axis(x.astype(np.float64), d)
        rel = np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)
        self.assertLess(rel, 4e-3)
        self.assertGreater(rel, 1e-6)  # BF16 rounding happened, so not bit exact

    def test_bf16_round_to_nearest_even(self):
        values = np.array([1.0, 1.0 + 2.0 ** -8, 1.0 + 3 * 2.0 ** -8, -2.5], dtype=np.float32)
        self.assertEqual(f32_to_bf16_bits(values).tolist(), [0x3F80, 0x3F80, 0x3F82, 0xC020])
        with self.assertRaises(ValueError):
            f32_to_bf16_bits(np.array([np.nan], np.float32))


class RecipeTest(unittest.TestCase):
    allowed = GROUPS.recipe_types()

    def parse(self, text):
        return parse_recipe(textwrap.dedent(text), ALL_GROUPS, self.allowed)

    def test_parse(self):
        recipe = self.parse(r"""
            version: 1
            rotate: [ffn_in, attn_in]
            types:
              - match: '^blk\.\d+\.ssm_out\.weight$'
                type: Q6_K
              - match: '\.weight$'
                type: Q4_K
            """)
        self.assertEqual(recipe.rotate, ("attn_in", "ffn_in"))  # definition order
        self.assertEqual([r.type_name for r in recipe.rules], ["Q6_K", "Q4_K"])
        self.assertEqual(self.parse("version: 1\nrotate: all\n").rotate, ALL_GROUPS)
        self.assertEqual(self.parse("version: 1\nrotate: none\n").rotate, ())
        self.assertIsNone(self.parse("version: 1\n").rotate)

    def test_rejects(self):
        cases = {
            "version: 2\n": "version",
            "version: 1\nextra: 1\n": "unknown recipe key",
            "version: 1\nrotate: [attn_in, nope]\n": "unknown input group",
            "version: 1\nrotate: [attn_in, attn_in]\n": "repeats",
            "version: 1\nrotate: no\n": "rotate must be",
            "version: 1\ntypes: [{match: x, type: Q2_K}]\n": "not supported",
            "version: 1\ntypes: [{match: x, type: Q8_K}]\n": "not supported",
            "version: 1\ntypes: [{match: x, type: q4_k}]\n": "not supported",
            "version: 1\ntypes: [{match: '(', type: Q4_K}]\n": "invalid pattern",
            "version: 1\ntypes: [{match: x}]\n": "exactly the keys",
            "version: 1\ntypes: {a: b}\n": "must be a list",
            "[1, 2]\n": "mapping",
            "version: 1\n  bad: [\n": "not valid YAML",
        }
        for text, message in cases.items():
            with self.subTest(text=text), self.assertRaisesRegex(RecipeError, message):
                self.parse(text)

    def test_cli_selection(self):
        self.assertEqual(parse_rotate_selection("ffn_in,attn_in", ALL_GROUPS), ("attn_in", "ffn_in"))
        with self.assertRaises(RecipeError):
            parse_rotate_selection("", ALL_GROUPS)

    def test_unknown_model_is_refused(self):
        for name in ("nosuch", "../qwen27b", ""):
            with self.subTest(name=name), self.assertRaisesRegex(PlanError, "definition"):
                load_groups(name)


class ConverterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.tensors = toy_tensors(np.random.default_rng(5))
        self.split = write_split(self.dir, self.tensors)

    def tearDown(self):
        self.tmp.cleanup()

    def rotate_all(self, seed=11, name="rot.gguf", groups="all"):
        out = self.dir / name
        code, _, err = run_cli("--groups", "qwen27b", "--input", self.split[1], "--rotate-groups", groups,
                               "--seed", seed, "--rotate-only", "--output", out)
        self.assertEqual(code, 0, err)
        return out

    def recipe(self, text, name="r.yaml"):
        path = self.dir / name
        path.write_text(textwrap.dedent(text))
        return path

    def test_split_input_is_read_as_one_model(self):
        with open_gguf(self.split[0]) as model:
            self.assertEqual(len(model.paths), 2)
            self.assertEqual({t.name for t in model.tensors}, set(self.tensors))
            self.assertEqual({t.shard for t in model.tensors}, {0, 1})
            for tensor in model.tensors:
                type_name, ne, data = self.tensors[tensor.name]
                self.assertEqual(bytes(model.tensor_bytes(tensor)), np.ascontiguousarray(data).tobytes())

    def test_rotate_only_roundtrip(self):
        out = self.rotate_all()
        # Independent reader (tools/gufo/gguf.py) sees the markers and shapes.
        state = shared_gguf.parse_gguf(out)
        kv = {**state["metadata"], **state["user_kv"]}
        self.assertEqual(kv["gufo.rotation.version"], 1)
        self.assertEqual(kv["gufo.rotation.kind"], "hadamard")
        self.assertEqual(kv["gufo.rotation.block_size"], 128)
        self.assertEqual(kv["gufo.rotation.side"], "input")
        self.assertEqual(kv["gufo.rotation.source_converter"], "rotquant 1")
        self.assertEqual(kv["gufo.rotation.source_file"], self.split[0].name)
        self.assertEqual(kv["gufo.rotation.source_sha256"], convert.file_sha256(self.split))
        self.assertEqual(len(kv["gufo.rotation.source_recipe_sha256"]), 64)
        self.assertFalse(any(key.startswith("split.") for key in kv))
        rot = {t["name"]: t for t in state["tensors"] if t["name"].endswith(".rot")}
        projections = [n for n in self.tensors if GROUPS.instance(n)[0] is not None]
        self.assertEqual(set(rot), {n + ".rot" for n in projections})
        for name in projections:
            self.assertEqual(rot[name + ".rot"]["type_name"], "F16")
            self.assertEqual(rot[name + ".rot"]["shape"], [self.tensors[name][1][0]])

        with open_gguf(out) as model, open_gguf(self.split[0]) as source:
            self.assertEqual(check_rotation(model, GROUPS), ALL_GROUPS)
            # Tokenizer and other metadata are copied byte for byte.
            self.assertEqual(dict(model.kv_raw)["tokenizer.ggml.tokens"],
                             dict(source.kv_raw)["tokenizer.ggml.tokens"])
            by_name = model.by_name
            sign = {n: read_rot_signs(model, by_name[n + ".rot"]) for n in projections}
            group0 = ["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_alpha.weight",
                      "blk.0.ssm_beta.weight"]
            for name in group0[1:]:
                np.testing.assert_array_equal(sign[name], sign[group0[0]])
            np.testing.assert_array_equal(sign["blk.1.attn_q.weight"], sign["blk.1.attn_v.weight"])
            np.testing.assert_array_equal(sign["blk.0.ffn_gate.weight"], sign["blk.0.ffn_up.weight"])
            self.assertFalse(np.array_equal(sign["blk.0.ffn_gate.weight"], sign["blk.1.ffn_gate.weight"]))
            np.testing.assert_array_equal(sign["blk.0.attn_qkv.weight"], signs(11, "attn_in:0", HIDDEN))

            rng = np.random.default_rng(9)
            for name in projections:
                w = as_float32(source, source.by_name[name]).astype(np.float64)
                w_rot = as_float32(model, by_name[name]).astype(np.float64)
                x = rng.standard_normal(w.shape[1])
                y_ref = w @ x
                y = w_rot @ rotate_last_axis(x, sign[name])
                rel = np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)
                self.assertLess(rel, 4e-3, name)
                # Stored values are the BF16 rounding of the exact rotation (half an ulp,
                # plus FP32 accumulation error), and match the FP32 path bit for bit.
                exact = rotate_last_axis(w, sign[name])
                bound = np.abs(exact) * 2.0 ** -8 + 1e-6 * np.abs(exact).max()
                self.assertTrue((np.abs(w_rot - exact) <= bound).all(), name)
                stored = np.frombuffer(bytes(model.tensor_bytes(by_name[name])), "<u2")
                expected = f32_to_bf16_bits(rotate_last_axis(w.astype(np.float32), sign[name]))
                np.testing.assert_array_equal(stored, expected.reshape(-1), name)
            for name in self.tensors:
                if name not in projections:
                    self.assertEqual(bytes(model.tensor_bytes(by_name[name])),
                                     bytes(source.tensor_bytes(source.by_name[name])), name)

    def test_partial_groups_and_no_rotation(self):
        out = self.rotate_all(groups="ffn_in,head_in", name="partial.gguf")
        with open_gguf(out) as model:
            self.assertEqual(check_rotation(model, GROUPS), ("ffn_in", "head_in"))
            names = {t.name for t in model.tensors}
            self.assertIn("output.weight.rot", names)
            self.assertNotIn("blk.0.ffn_down.weight.rot", names)
        out = self.rotate_all(groups="none", name="none.gguf")
        with open_gguf(out) as model:
            self.assertEqual(check_rotation(model, GROUPS), ())
            self.assertNotIn("gufo.rotation.version", model.kv)
            self.assertIn("gufo.rotation.source_sha256", model.kv)

    def test_rotation_metadata_rejections(self):
        out = self.rotate_all()
        with open_gguf(out) as model:
            kv_raw = list(model.kv_raw)
            tensors = {t.name: (t.type_name, t.ne, np.frombuffer(bytes(model.tensor_bytes(t)), np.uint8))
                       for t in model.tensors}

        def variant(name, kv=kv_raw, tensors=tensors):
            return write_single(self.dir / name, tensors, kv=kv)

        flipped = dict(tensors)
        bits = flipped["blk.0.attn_gate.weight.rot"][2].view("<u2").copy()
        bits[0] ^= 0x8000
        flipped["blk.0.attn_gate.weight.rot"] = ("F16", (HIDDEN,), bits)
        no_version = [(k, v) for k, v in kv_raw if k != "gufo.rotation.version"]
        bad_block = [(k, encode_value(U32(64)) if k == "gufo.rotation.block_size" else v) for k, v in kv_raw]
        embd_rot = dict(tensors)
        embd_rot["token_embd.weight.rot"] = tensors["output.weight.rot"]
        mixed = {k: v for k, v in tensors.items() if k != "blk.0.ssm_beta.weight.rot"}
        layer_only = {k: v for k, v in tensors.items() if not k.startswith("blk.1.ffn_down.weight.rot")}
        not_pm1 = dict(tensors)
        not_pm1["blk.0.ffn_down.weight.rot"] = ("F16", (512,), np.full(512, 0x4000, "<u2"))
        wrong_len = dict(tensors)
        wrong_len["output.weight.rot"] = ("F16", (128,), np.full(128, 0x3C00, "<u2"))
        markers_only = {k: v for k, v in tensors.items() if not k.endswith(".rot")}
        cases = {
            "different sign vector": variant("flip.gguf", tensors=flipped),
            "missing gufo.rotation.version": variant("nover.gguf", kv=no_version),
            "unsupported gufo.rotation.block_size": variant("block.gguf", kv=bad_block),
            "no rotation path": variant("embd.gguf", tensors=embd_rot),
            "mixes rotated and unrotated": variant("mixed.gguf", tensors=mixed),
            "some layers only": variant("layer.gguf", tensors=layer_only),
            "other than": variant("pm1.gguf", tensors=not_pm1),
            "128 values": variant("len.gguf", tensors=wrong_len),
            "no .rot tensors": variant("markers.gguf", tensors=markers_only),
        }
        for message, path in cases.items():
            with self.subTest(message=message), open_gguf(path) as model:
                with self.assertRaisesRegex(PlanError, message):
                    check_rotation(model, GROUPS)
        rot_only = [(k, v) for k, v in kv_raw if not k.startswith("gufo.rotation.")]
        with open_gguf(variant("nokv.gguf", kv=rot_only)) as model:
            with self.assertRaisesRegex(PlanError, "no gufo.rotation metadata"):
                check_rotation(model, GROUPS)

    def test_plan_rejections(self):
        tied = write_single(self.dir / "tied.gguf", toy_tensors(np.random.default_rng(1), tied=True))
        other_arch = write_single(self.dir / "llama.gguf", self.tensors,
                                  kv=[("general.architecture", encode_value("llama"))])
        q4 = self.recipe("version: 1\ntypes: [{match: '\\.weight$', type: Q4_K}]\n", "q4.yaml")
        unused = self.recipe("version: 1\ntypes: [{match: '\\.weight$', type: Q8_0}, "
                             "{match: 'nothing', type: Q4_K}]\n", "unused.yaml")
        partial = self.recipe("version: 1\ntypes: [{match: 'ffn', type: Q8_0}]\n", "partial.yaml")
        rot = self.rotate_all()
        cases = [
            ("tied to token_embd.weight", ["--input", tied, "--rotate-groups", "head_in", "--seed", 1,
                                           "--rotate-only", "--output", self.dir / "x.gguf"]),
            ("architecture", ["--input", other_arch, "--rotate-groups", "none", "--rotate-only",
                              "--output", self.dir / "x.gguf"]),
            ("requires --seed", ["--input", self.split[0], "--rotate-groups", "all", "--rotate-only",
                                 "--output", self.dir / "x.gguf"]),
            ("choose the rotated groups", ["--input", self.split[0], "--rotate-only",
                                           "--output", self.dir / "x.gguf"]),
            ("not a multiple of the Q4_K block", ["--input", self.split[0], "--rotate-groups", "none",
                                                  "--recipe", q4, "--dry-run"]),
            ("match no tensor", ["--input", self.split[0], "--rotate-groups", "none",
                                 "--recipe", unused, "--dry-run"]),
            ("have no type in the recipe", ["--input", self.split[0], "--rotate-groups", "none",
                                            "--recipe", partial, "--dry-run"]),
            ("already rotated for groups", ["--input", rot, "--rotate-groups", "ffn_in",
                                            "--recipe", q4, "--dry-run"]),
            ("cannot change an already rotated", ["--input", rot, "--rotate-groups", "all", "--seed", 2,
                                                  "--recipe", q4, "--dry-run"]),
            ("nothing to do", ["--input", rot, "--rotate-groups", "all", "--rotate-only",
                               "--output", self.dir / "x.gguf"]),
            ("requires --llama-quantize", ["--input", self.split[0], "--rotate-groups", "none",
                                           "--recipe", q4, "--output", self.dir / "x.gguf"]),
            ("does not apply to --rotate-only", ["--input", self.split[0], "--rotate-groups", "none",
                                                 "--rotate-only", "--llama-quantize", "/bin/true",
                                                 "--output", self.dir / "x.gguf"]),
            ("refusing to overwrite", ["--input", self.split[0], "--rotate-groups", "all", "--seed", 1,
                                       "--rotate-only", "--output", rot]),
        ]
        for message, args in cases:
            with self.subTest(message=message):
                code, _, err = run_cli("--groups", "qwen27b", *args)
                self.assertEqual(code, 2, err)
                self.assertIn(message, err)

    def test_float_target_types_follow_loader_roles(self):
        # A BF16 norm would be reinterpreted by the loader's float kernels.
        tensors = dict(self.tensors)
        tensors["output_norm.weight"] = ("BF16", (HIDDEN,), np.zeros(HIDDEN, "<u2"))
        path = write_single(self.dir / "bf16norm.gguf", tensors)
        code, _, err = run_cli("--groups", "qwen27b", "--input", path, "--rotate-groups", "none",
                               "--rotate-only", "--dry-run")
        self.assertEqual(code, 2)
        self.assertIn("output_norm.weight: rotated type BF16 is not supported for the norm role", err)

    def test_dry_run_table(self):
        recipe = self.recipe(r"""
            version: 1
            rotate: [attn_in, ffn_in, ffn_down_in]
            types:
              - match: 'attn_output|ssm_alpha|ssm_beta'
                type: Q8_0
              - match: 'ssm_out'
                type: Q6_K
              - match: '^token_embd'
                type: Q8_0
              - match: '\.weight$'
                type: Q4_K
            """)
        code, out, err = run_cli("--groups", "qwen27b", "--input", self.split[0], "--recipe", recipe,
                                 "--seed", 3, "--dry-run")
        self.assertEqual(code, 0, err)
        rows = {line.split()[0]: line.split() for line in out.splitlines() if line.startswith(("blk.", "token", "output"))}
        self.assertEqual(rows["blk.0.attn_qkv.weight"][1:], ["384x256", "projection", "attn_in:0", "yes",
                                                             "BF16", "Q4_K", "0.05", "4.500"])
        self.assertEqual(rows["blk.1.attn_output.weight"][3:7], ["attn_out_in:1", "no", "BF16", "Q8_0"])
        self.assertEqual(rows["blk.0.ssm_out.weight"][6], "Q6_K")
        self.assertEqual(rows["output.weight"][3:7], ["head_in", "no", "BF16", "Q4_K"])
        self.assertEqual(rows["blk.0.ssm_conv1d.weight"][2:7], ["ssm_parameter", "-", "no", "F32", "F32"])
        self.assertEqual(rows["token_embd.weight"][3:7], ["-", "no", "BF16", "Q8_0"])
        self.assertIn("rotated groups: attn_in, ffn_in, ffn_down_in", out)
        self.assertIn(".rot sign tensors (F16): 13,", out)
        self.assertIn("llama-quantize base type: Q4_K", out)
        # Rotate-only dry runs keep the source types.
        code, out, err = run_cli("--groups", "qwen27b", "--input", self.split[0], "--recipe", recipe,
                                 "--seed", 3, "--rotate-only", "--dry-run")
        self.assertEqual(code, 0, err)
        self.assertIn("BF16    BF16", out)
        self.assertNotIn("base type", out)

    def test_copy_recipe_from_header(self):
        quantized = {}
        for name, (type_name, ne, data) in self.tensors.items():
            role = GROUPS.role(name)
            if role in ("projection", "embedding"):
                target = "Q8_0" if ne[0] % 256 else ("IQ4_XS" if "ffn_down" in name else "Q5_K")
                block, size = {"Q8_0": (32, 34), "IQ4_XS": (256, 136), "Q5_K": (256, 176)}[target]
                quantized[name] = (target, ne, np.zeros(int(np.prod(ne)) // block * size, np.uint8))
            else:
                quantized[name] = (type_name, ne, data)
        header = write_single(self.dir / "ud.gguf", quantized)
        code, out, err = run_cli("--groups", "qwen27b", "--input", self.split[0], "--rotate-groups", "all",
                                 "--seed", 1, "--copy-recipe-from", header, "--dry-run")
        self.assertEqual(code, 0, err)
        for line in out.splitlines():
            parts = line.split()
            if parts and parts[0] in quantized:
                self.assertEqual(parts[6], quantized[parts[0]][0], parts[0])
        recipe = self.recipe("version: 1\ntypes: [{match: '.', type: Q8_0}]\n")
        code, _, err = run_cli("--groups", "qwen27b", "--input", self.split[0], "--rotate-groups", "all",
                               "--seed", 1, "--copy-recipe-from", header, "--recipe", recipe, "--dry-run")
        self.assertEqual(code, 2)
        self.assertIn("not both", err)
        missing = {k: v for k, v in quantized.items() if k != "blk.1.ffn_up.weight"}
        code, _, err = run_cli("--groups", "qwen27b", "--input", self.split[0], "--rotate-groups", "all",
                               "--seed", 1, "--copy-recipe-from", write_single(self.dir / "m.gguf", missing),
                               "--dry-run")
        self.assertEqual(code, 2)
        self.assertIn("1 tensor(s) have no type in the --copy-recipe-from file: blk.1.ffn_up.weight", err)

    def fake_quantize(self):
        log = self.dir / "argv.json"
        script = self.dir / "fake-llama-quantize"
        script.write_text(f"#!{sys.executable}\nimport json, sys\n"
                          f"open({str(log)!r}, 'w').write(json.dumps(sys.argv[1:]))\n")
        script.chmod(0o755)
        return script, log

    def test_quantize_command_from_rotated_file(self):
        rot = self.rotate_all()
        recipe = self.recipe(r"""
            version: 1
            rotate: all
            types:
              - match: 'attn_output'
                type: Q8_0
              - match: '\.weight$'
                type: IQ4_XS
            """)
        script, log = self.fake_quantize()
        imatrix = self.dir / "rot.imatrix"
        imatrix.write_bytes(b"x")
        out = self.dir / "rot-iq4.gguf"
        code, stdout, err = run_cli("--groups", "qwen27b", "--input", rot, "--recipe", recipe,
                                    "--imatrix", imatrix, "--llama-quantize", script, "--output", out)
        self.assertEqual(code, 0, err)
        argv = json.loads(log.read_text())
        recipe_json = (self.dir / "rot-iq4.recipe.json").read_text().strip()
        digest = hashlib.sha256(recipe_json.encode()).hexdigest()
        self.assertEqual(argv, [
            "--pure", "--tensor-type-file", str(self.dir / "rot-iq4.tensor-types.txt"),
            "--imatrix", str(imatrix),
            "--override-kv", "gufo.rotation.source_converter=str:rotquant 1",
            "--override-kv", f"gufo.rotation.source_recipe_sha256=str:{digest}",
            str(rot), str(out), "IQ4_XS"])
        self.assertFalse((self.dir / "rot-iq4.rotated.gguf").exists())
        effective = json.loads(recipe_json)
        self.assertEqual(effective["rotate"], list(ALL_GROUPS))
        self.assertEqual(effective["seed"], None)
        self.assertEqual(effective["types"]["blk.1.attn_output.weight"], "Q8_0")
        lines = (self.dir / "rot-iq4.tensor-types.txt").read_text().splitlines()
        rules = dict(line.rsplit("=", 1) for line in lines)
        self.assertEqual(rules[r"^blk\.0\.attn_qkv\.weight$"], "IQ4_XS")
        self.assertEqual(rules[r"^blk\.1\.attn_output\.weight$"], "Q8_0")
        self.assertEqual(rules[r"^blk\.0\.ssm_conv1d\.weight$"], "F32")
        self.assertEqual(rules[r"^token_embd\.weight$"], "IQ4_XS")
        self.assertFalse(any(".rot" in line for line in lines))
        matrices = [n for n, (_, ne, _) in self.tensors.items() if len(ne) >= 2]
        self.assertEqual(len(lines), len(matrices))
        for line in lines:
            self.assertEqual(line.split("=")[0], line.split("=")[0].lower())

    def test_quantize_from_bf16_rotates_first(self):
        script, log = self.fake_quantize()
        recipe = self.recipe("version: 1\nrotate: [ffn_down_in]\ntypes: [{match: '.', type: Q8_0}]\n")
        out = self.dir / "q8.gguf"
        code, _, err = run_cli("--groups", "qwen27b", "--input", self.split[0], "--recipe", recipe,
                               "--seed", 4, "--llama-quantize", script, "--output", out)
        self.assertEqual(code, 0, err)
        rotated = self.dir / "q8.rotated.gguf"
        argv = json.loads(log.read_text())
        self.assertEqual(argv[-3:], [str(rotated), str(out), "Q8_0"])
        with open_gguf(rotated) as model:
            self.assertEqual(check_rotation(model, GROUPS), ("ffn_down_in",))
            self.assertNotEqual(model.kv["gufo.rotation.source_recipe_sha256"],
                                argv[argv.index("--override-kv") + 3].split("str:")[1])
        # Without rotation the input is quantized directly and keeps its provenance.
        recipe = self.recipe("version: 1\nrotate: none\ntypes: [{match: '.', type: Q8_0}]\n", "none.yaml")
        out = self.dir / "plain.gguf"
        code, _, err = run_cli("--groups", "qwen27b", "--input", self.split[0], "--recipe", recipe,
                               "--llama-quantize", script, "--output", out)
        self.assertEqual(code, 0, err)
        argv = json.loads(log.read_text())
        self.assertEqual(argv[-3:], [str(self.split[0]), str(out), "Q8_0"])
        self.assertIn(f"gufo.rotation.source_sha256=str:{convert.file_sha256(self.split)}", argv)
        self.assertFalse((self.dir / "plain.rotated.gguf").exists())

    def test_quantize_command_rejects_float_only_recipe(self):
        with open_gguf(self.split[0]) as model:
            plan = build_plan(model, GROUPS, (), seed=None, rotate_only=False,
                              rules=parse_recipe("version: 1\ntypes: [{match: '.', type: BF16}]\n",
                                                 ALL_GROUPS, GROUPS.recipe_types()).rules)
        self.assertEqual(len(tensor_type_lines(plan)), sum(len(ne) >= 2 for _, ne, _ in self.tensors.values()))
        with self.assertRaisesRegex(PlanError, "quantizes no tensor"):
            quantize_command(plan, "q", "in.gguf", "out.gguf", "t.txt")


if __name__ == "__main__":
    unittest.main()
