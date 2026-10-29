#!/usr/bin/env python3
"""Synthetic-tensor checks for tools/rotquant (no model files, no GPU).

Run with NumPy: python3 -I tests/tools/rotquant_test.py
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
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from gufo import gguf as shared_gguf  # noqa: E402
from rotquant import check_rotated, convert, realign  # noqa: E402
from rotquant.gguf_io import OutTensor, U32, as_float32, encode_value, open_gguf, write_gguf  # noqa: E402
from rotquant.hadamard import (BLOCK_SIZE, bf16_bits_to_f32, f32_to_bf16_bits,  # noqa: E402
                               rotate_last_axis, signs, sylvester)
from rotquant.plan import (ROTATION_KV, PlanError, build_plan, check_rotation,  # noqa: E402
                           load_groups, quantize_command, read_rot_signs, tensor_type_lines)
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


FAKE_QUANTIZE = """#!{python}
import json, re, sys
sys.path.insert(0, {tools!r})
from rotquant.gguf_io import SPLIT_KEYS, OutTensor, encode_value, open_gguf, type_size, write_gguf
args = sys.argv[1:]
open({log!r}, "w").write(json.dumps(args))
rules, overrides = [], {{}}
i = 0
while i < len(args) - 3:
    if args[i] == "--tensor-type-file":
        rules = [line.rsplit("=", 1) for line in open(args[i + 1]).read().split()]
    elif args[i] == "--override-kv":
        key, value = args[i + 1].split("=str:", 1)
        overrides[key] = value
    i += 2 if args[i] != "--pure" else 1
source, output = args[-3], args[-2]
model = open_gguf(source)
tensors = []
for t in model.tensors:
    new = next((typ for pattern, typ in rules if re.search(pattern, t.name)), t.type_name)
    if t.name == {wrong!r}:
        new = "Q4_K"
    if new == t.type_name:
        payload = bytes(model.tensor_bytes(t))
        if {change_rot!r} and t.name.endswith(".rot"):
            payload = payload[:-2] + bytes([payload[-2], payload[-1] ^ 0x80])
    else:
        n = 1
        for d in t.ne:
            n *= d
        payload = bytes(type_size(new, n, t.ne[0]))
    tensors.append(OutTensor(t.name, t.ne, new, lambda f, b=payload: f.write(b)))
kv = [(k, raw) for k, raw in model.kv_raw if k not in SPLIT_KEYS and k not in overrides]
kv += [(k, encode_value(v)) for k, v in overrides.items()]
write_gguf(output, kv, tensors, model.alignment)
"""


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

    def parse(self, data):
        text = data if isinstance(data, str) else json.dumps(data)
        return parse_recipe(text, ALL_GROUPS, self.allowed)

    def test_parse(self):
        recipe = self.parse(r"""
            {"version": 1,
             "rotate": ["ffn_in", "attn_in"],
             "types": [{"match": "^blk\\.\\d+\\.ssm_out\\.weight$", "type": "Q6_K"},
                       {"match": "\\.weight$", "type": "Q4_K"}]}
            """)
        self.assertEqual(recipe.rotate, ("attn_in", "ffn_in"))  # definition order
        self.assertEqual([r.match for r in recipe.rules], [r"^blk\.\d+\.ssm_out\.weight$", r"\.weight$"])
        self.assertEqual([r.type_name for r in recipe.rules], ["Q6_K", "Q4_K"])
        self.assertEqual(self.parse({"version": 1, "rotate": "all"}).rotate, ALL_GROUPS)
        self.assertEqual(self.parse({"version": 1, "rotate": "none"}).rotate, ())
        self.assertIsNone(self.parse({"version": 1}).rotate)

    def test_rejects(self):
        cases = {
            '{"version": 2}': "version",
            '{"version": 1.0}': "version",
            '{"version": true}': "version",
            '{"version": 1, "extra": 1}': "unknown recipe key",
            '{"version": 1, "rotate": ["attn_in", "nope"]}': "unknown input group",
            '{"version": 1, "rotate": ["attn_in", "attn_in"]}': "repeats",
            '{"version": 1, "rotate": false}': "rotate must be",
            '{"version": 1, "types": [{"match": "x", "type": "Q2_K"}]}': "not supported",
            '{"version": 1, "types": [{"match": "x", "type": "Q8_K"}]}': "not supported",
            '{"version": 1, "types": [{"match": "x", "type": "q4_k"}]}': "not supported",
            '{"version": 1, "types": [{"match": "(", "type": "Q4_K"}]}': "invalid pattern",
            '{"version": 1, "types": [{"match": "x"}]}': "exactly the keys",
            '{"version": 1, "types": [{"match": "x", "type": "Q4_K", "type": "Q8_0"}]}': "repeats the key",
            '{"version": 1, "version": 1}': "repeats the key",
            '{"version": 1, "types": {"a": "b"}}': "must be a list",
            "[1, 2]": "JSON object",
            "version: 1\n": "not valid JSON",
            '{"version": 1,}': "not valid JSON",
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

    def recipe(self, data, name="r.json"):
        path = self.dir / name
        path.write_text(json.dumps(data))
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
        q4 = self.recipe({"version": 1, "types": [{"match": r"\.weight$", "type": "Q4_K"}]}, "q4.json")
        unused = self.recipe({"version": 1, "types": [{"match": r"\.weight$", "type": "Q8_0"},
                                                      {"match": "nothing", "type": "Q4_K"}]}, "unused.json")
        partial = self.recipe({"version": 1, "types": [{"match": "ffn", "type": "Q8_0"}]}, "partial.json")
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
        recipe = self.recipe({"version": 1, "rotate": ["attn_in", "ffn_in", "ffn_down_in"], "types": [
            {"match": "attn_output|ssm_alpha|ssm_beta", "type": "Q8_0"},
            {"match": "ssm_out", "type": "Q6_K"},
            {"match": "^token_embd", "type": "Q8_0"},
            {"match": r"\.weight$", "type": "Q4_K"}]})
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
        recipe = self.recipe({"version": 1, "types": [{"match": ".", "type": "Q8_0"}]})
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

    def fake_quantize(self, wrong_type=None, change_rot=False):
        """A stand-in llama-quantize: records argv and writes an output GGUF
        with the requested types (zero payloads for converted tensors)."""
        log = self.dir / "argv.json"
        script = self.dir / "fake-llama-quantize"
        script.write_text(FAKE_QUANTIZE.format(python=sys.executable, tools=str(ROOT / "tools"), log=str(log),
                                               wrong=wrong_type, change_rot=change_rot))
        script.chmod(0o755)
        return script, log

    def test_quantize_command_from_rotated_file(self):
        rot = self.rotate_all()
        recipe = self.recipe({"version": 1, "rotate": "all", "types": [
            {"match": "attn_output", "type": "Q8_0"},
            {"match": r"\.weight$", "type": "IQ4_XS"}]})
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
        recipe = self.recipe({"version": 1, "rotate": ["ffn_down_in"],
                              "types": [{"match": ".", "type": "Q8_0"}]})
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
        recipe = self.recipe({"version": 1, "rotate": "none", "types": [{"match": ".", "type": "Q8_0"}]},
                             "none.json")
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
                              rules=parse_recipe('{"version": 1, "types": [{"match": ".", "type": "BF16"}]}',
                                                 ALL_GROUPS, GROUPS.recipe_types()).rules)
        self.assertEqual(len(tensor_type_lines(plan)), sum(len(ne) >= 2 for _, ne, _ in self.tensors.values()))
        with self.assertRaisesRegex(PlanError, "quantizes no tensor"):
            quantize_command(plan, "q", "in.gguf", "out.gguf", "t.txt")

    def test_quantized_output_is_checked(self):
        rot = self.rotate_all()
        recipe = self.recipe({"version": 1, "rotate": "all", "types": [{"match": r"\.weight$", "type": "Q8_0"}]})
        script, _ = self.fake_quantize()
        code, stdout, err = run_cli("--groups", "qwen27b", "--input", rot, "--recipe", recipe,
                                    "--llama-quantize", script, "--output", self.dir / "ok.gguf")
        self.assertEqual(code, 0, err)
        self.assertIn("types match the plan", stdout)
        self.assertIn("(16 .rot)", stdout)
        cases = [("wrong", dict(wrong_type="blk.0.ffn_gate.weight"), "blk.0.ffn_gate.weight Q4_K != Q8_0"),
                 ("rot", dict(change_rot=True), "unquantized tensor(s) changed bytes")]
        for name, kwargs, message in cases:
            with self.subTest(name=name):
                script, _ = self.fake_quantize(**kwargs)
                code, _, err = run_cli("--groups", "qwen27b", "--input", rot, "--recipe", recipe,
                                       "--llama-quantize", script, "--output", self.dir / f"bad-{name}.gguf")
                self.assertEqual(code, 2)
                self.assertIn(message, err)

    def test_check_rotated_on_a_converted_file(self):
        rot = self.rotate_all(seed=11)
        report = self.dir / "check.json"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = check_rotated.main(["--groups", "qwen27b", "--seed", "11", "--source", str(self.split[0]),
                                       "--rotated", str(rot), "--rows", "0", "--json", str(report)])
        self.assertEqual(code, 0, out.getvalue())
        summary = json.loads(report.read_text())
        self.assertEqual(summary["rotated_groups"], list(ALL_GROUPS))
        self.assertEqual({r["group"] for r in summary["tensors"]}, set(ALL_GROUPS))
        for r in summary["tensors"]:
            self.assertEqual(r["bit_mismatches"], 0, r["tensor"])
            self.assertTrue(r["signs_match_seed"], r["tensor"])
            self.assertLess(max(r["random"]["rel_l2"] + r["outlier"]["rel_l2"]), 4e-3, r["tensor"])
            self.assertLessEqual(r["weight_rel_rounding_max"], 2.0 ** -8 * (1 + 1e-3), r["tensor"])
        # The wrong seed is reported.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check_rotated.main(["--groups", "qwen27b", "--seed", "12", "--source",
                                                 str(self.split[0]), "--rotated", str(rot)]), 1)

    def test_output_is_256_byte_aligned(self):
        # Inputs use the GGUF default of 32; the converter writes 256.
        with open_gguf(self.split[0]) as model:
            self.assertEqual(model.alignment, 32)
        for groups, seed in (("all", ["--seed", 3]), ("none", [])):
            with self.subTest(groups=groups):
                out = self.dir / f"aligned-{groups}.gguf"
                code, _, err = run_cli("--groups", "qwen27b", "--input", self.split[0], "--rotate-groups", groups,
                                       *seed, "--rotate-only", "--output", out)
                self.assertEqual(code, 0, err)
                state = shared_gguf.parse_gguf(out)
                self.assertEqual(state["metadata"]["general.alignment"], 256)
                self.assertEqual(state["data_offset"] % 256, 0)
                raw = out.read_bytes()
                with open_gguf(out) as model, open_gguf(self.split[0]) as source:
                    self.assertEqual(model.shards[0].data_offset, state["data_offset"])
                    for tensor in model.tensors:
                        start = state["data_offset"] + tensor.offset
                        self.assertEqual(start % 256, 0, tensor.name)
                        self.assertEqual(raw[start:start + tensor.nbytes], bytes(model.tensor_bytes(tensor)))
                        if groups == "none":
                            self.assertEqual(bytes(model.tensor_bytes(tensor)),
                                             bytes(source.tensor_bytes(source.by_name[tensor.name])), tensor.name)
                    self.assertEqual(sum(key == "general.alignment" for key, _ in model.kv_raw), 1)
        # An input that already records an alignment is rewritten, not duplicated.
        aligned = self.dir / "a64.gguf"
        write_gguf(aligned, kv_entries(), out_tensors(self.tensors), 64)
        with open_gguf(aligned) as model:
            self.assertEqual(model.alignment, 64)
        out = self.dir / "from-a64.gguf"
        code, _, err = run_cli("--groups", "qwen27b", "--input", aligned, "--rotate-groups", "none",
                               "--rotate-only", "--output", out)
        self.assertEqual(code, 0, err)
        with open_gguf(out) as model:
            self.assertEqual(model.alignment, 256)
            self.assertEqual(sum(key == "general.alignment" for key, _ in model.kv_raw), 1)
        with self.assertRaises(ValueError):
            write_gguf(self.dir / "bad.gguf", [], [], 48)

    def test_realign_keeps_bytes(self):
        out = self.dir / "realigned.gguf"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(realign.main(["--input", str(self.split[1]), "--output", str(out)]), 0)
        state = shared_gguf.parse_gguf(out)
        self.assertEqual(state["metadata"]["general.alignment"], 256)
        self.assertNotIn("split.count", state["metadata"])
        with open_gguf(out) as model, open_gguf(self.split[0]) as source:
            self.assertEqual([t.name for t in model.tensors], [t.name for t in source.tensors])
            for tensor in model.tensors:
                self.assertEqual((model.shards[0].data_offset + tensor.offset) % 256, 0)
                self.assertEqual(bytes(model.tensor_bytes(tensor)),
                                 bytes(source.tensor_bytes(source.by_name[tensor.name])))
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(realign.main(["--input", str(self.split[0]), "--output", str(out)]), 2)
        self.assertIn("refusing to overwrite", err.getvalue())

    def test_nextn_blocks_are_not_rotated(self):
        tensors = dict(self.tensors)
        rng = np.random.default_rng(9)
        for short, (n, k) in {"attn_q": (512, HIDDEN), "attn_k": (128, HIDDEN), "attn_v": (128, HIDDEN),
                              "attn_output": (HIDDEN, 384), "ffn_gate": (512, HIDDEN),
                              "ffn_up": (512, HIDDEN), "ffn_down": (HIDDEN, 512)}.items():
            tensors[f"blk.2.{short}.weight"] = (
                "BF16", (k, n), f32_to_bf16_bits(rng.standard_normal((n, k)).astype(np.float32)))
        tensors["blk.2.nextn.eh_proj.weight"] = (
            "BF16", (2 * HIDDEN, HIDDEN), f32_to_bf16_bits(rng.standard_normal((HIDDEN, 2 * HIDDEN)).astype(np.float32)))
        kv = [e for e in kv_entries() if e[0] != "qwen35.block_count"]
        kv += [("qwen35.block_count", encode_value(U32(3))),
               ("qwen35.nextn_predict_layers", encode_value(U32(1)))]
        path = write_single(self.dir / "mtp.gguf", tensors, kv=kv)
        out = self.dir / "mtp-rot.gguf"
        code, _, err = run_cli("--groups", "qwen27b", "--input", path, "--rotate-groups", "all", "--seed", 1,
                               "--rotate-only", "--output", out)
        self.assertEqual(code, 0, err)
        with open_gguf(out) as model, open_gguf(path) as source:
            names = {t.name for t in model.tensors}
            self.assertNotIn("blk.2.attn_q.weight.rot", names)
            self.assertIn("blk.1.attn_q.weight.rot", names)
            self.assertEqual(bytes(model.tensor_bytes(model.by_name["blk.2.ffn_down.weight"])),
                             bytes(source.tensor_bytes(source.by_name["blk.2.ffn_down.weight"])))
            self.assertEqual(check_rotation(model, GROUPS), ALL_GROUPS)
        # A .rot on a NextN block names the tensor and is refused.
        tensors["blk.2.ffn_down.weight.rot"] = ("F16", (512,), np.full(512, 0x3C00, "<u2"))
        bad = write_single(self.dir / "mtp-bad.gguf", tensors,
                           kv=kv + [(k, encode_value(U32(v)) if isinstance(v, int) else encode_value(v))
                                    for k, v in ROTATION_KV.items()])
        with open_gguf(bad) as model, self.assertRaisesRegex(PlanError, "blk.2.ffn_down.weight is rotated"):
            check_rotation(model, GROUPS)
        missing = write_single(self.dir / "noblocks.gguf", self.tensors,
                               kv=[e for e in kv_entries() if e[0] != "qwen35.block_count"])
        code, _, err = run_cli("--groups", "qwen27b", "--input", missing, "--rotate-groups", "none",
                               "--rotate-only", "--dry-run")
        self.assertEqual(code, 2)
        self.assertIn("rotated block range", err)


if __name__ == "__main__":
    unittest.main()
