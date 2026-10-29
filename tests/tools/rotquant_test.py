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
from rotquant import prism  # noqa: E402
from rotquant.gguf_io import I32Array, OutTensor, U32, as_float32, encode_value, open_gguf, write_gguf  # noqa: E402
from rotquant.hadamard import (BLOCK_SIZE, bf16_bits_to_f32, f32_to_bf16_bits,  # noqa: E402
                               rotate_last_axis, signs, sylvester)
from rotquant.plan import (PlanError, build_plan, check_rotation, file_signs,  # noqa: E402
                           load_groups, quantize_command, sign_key, tensor_type_lines)
from rotquant.recipe import RecipeError, parse_recipe, parse_rotate_selection  # noqa: E402
from rotquant.verify import verify_output  # noqa: E402

GROUPS = load_groups("qwen27b")
ALL_GROUPS = GROUPS.names
HIDDEN = 256
TOKENS = ["<a>", "<b>", "한글"]
# Toy input widths: HIDDEN (attn_in, ffn_in, head_in, ssm_out), 384
# (attn_output) and 512 (ffn_down). One sign vector per width.
WIDTHS = (256, 384, 512)


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
        if {change_copy!r} and t.name == "output_norm.weight":
            payload = payload[:-1] + bytes([payload[-1] ^ 0x80])
    else:
        n = 1
        for d in t.ne:
            n *= d
        payload = bytes(type_size(new, n, t.ne[0]))
    tensors.append(OutTensor(t.name, t.ne, new, lambda f, b=payload: f.write(b)))
kv = [(k, raw) for k, raw in model.kv_raw if k not in SPLIT_KEYS and k not in overrides
      and not ({drop_prism!r} and k == "prism.hadamard.sign_values")]
kv += [(k, encode_value(v)) for k, v in overrides.items()]
# Like llama-quantize: keep the input's general.alignment, pack the data at 32.
write_gguf(output, kv, tensors, 32, recorded_alignment=model.alignment)
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
        # Independent reader (tools/gufo/gguf.py) sees the prism.hadamard metadata.
        state = shared_gguf.parse_gguf(out)
        kv = {**state["metadata"], **state["user_kv"]}
        folded = [n for n in self.tensors if GROUPS.instance(n)[0] is not None]
        self.assertNotIn("blk.0.ssm_alpha.weight", folded)
        self.assertEqual(kv["prism.hadamard.version"], 1)
        self.assertEqual(kv["prism.hadamard.block_size"], 128)
        self.assertEqual(kv["prism.hadamard.transform"], "normalized-sylvester-walsh-hadamard")
        self.assertEqual(kv["prism.hadamard.axis"], "input-last-dimension")
        self.assertEqual(kv["prism.hadamard.sign_mode"], "explicit")
        self.assertEqual(kv["prism.hadamard.sign_widths"], list(WIDTHS))
        self.assertEqual(len(kv["prism.hadamard.sign_values"]), sum(WIDTHS))
        self.assertEqual(kv["prism.hadamard.weight_names"], folded)  # file order
        for key in ("tied_output", "inverse_weight_names", "gdn_v_grouped"):
            self.assertNotIn("prism.hadamard." + key, kv)
        self.assertEqual(kv["gufo.rotation.source_converter"], "rotquant 1")
        self.assertEqual(kv["gufo.rotation.source_file"], self.split[0].name)
        self.assertEqual(kv["gufo.rotation.source_sha256"], convert.file_sha256(self.split))
        self.assertEqual(len(kv["gufo.rotation.source_recipe_sha256"]), 64)
        self.assertFalse(any(key.startswith("split.") or key == "gufo.rotation.version" for key in kv))
        # No extra tensors: the file has exactly the source tensors.
        self.assertEqual([t["name"] for t in state["tensors"]], list(self.tensors))
        # The sign values are int32 (the upstream loader reads int32/uint32 arrays).
        raw = dict(open_gguf(out, header_only=True).kv_raw)["prism.hadamard.sign_values"]
        self.assertEqual(struct.unpack_from("<II", raw), (9, 5))

        with open_gguf(out) as model, open_gguf(self.split[0]) as source:
            self.assertEqual(check_rotation(model, GROUPS), ALL_GROUPS)
            # Tokenizer and other metadata are copied byte for byte.
            self.assertEqual(dict(model.kv_raw)["tokenizer.ggml.tokens"],
                             dict(source.kv_raw)["tokenizer.ggml.tokens"])
            by_name = model.by_name
            table = file_signs(model)
            self.assertEqual(sorted(table), list(WIDTHS))
            for k in WIDTHS:
                np.testing.assert_array_equal(table[k], signs(11, sign_key(k), k))
            self.assertFalse(np.array_equal(table[256][:128], table[384][:128]))

            rng = np.random.default_rng(9)
            for name in folded:
                sign = table[by_name[name].ne[0]]
                w = as_float32(source, source.by_name[name]).astype(np.float64)
                w_rot = as_float32(model, by_name[name]).astype(np.float64)
                x = rng.standard_normal(w.shape[1])
                y_ref = w @ x
                y = w_rot @ rotate_last_axis(x, sign)
                rel = np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)
                self.assertLess(rel, 4e-3, name)
                # Stored values are the BF16 rounding of the exact rotation (half an ulp,
                # plus FP32 accumulation error), and match the FP32 path bit for bit.
                exact = rotate_last_axis(w, sign)
                bound = np.abs(exact) * 2.0 ** -8 + 1e-6 * np.abs(exact).max()
                self.assertTrue((np.abs(w_rot - exact) <= bound).all(), name)
                stored = np.frombuffer(bytes(model.tensor_bytes(by_name[name])), "<u2")
                expected = f32_to_bf16_bits(rotate_last_axis(w.astype(np.float32), sign))
                np.testing.assert_array_equal(stored, expected.reshape(-1), name)
            # Everything else, the DeltaNet ssm_alpha/ssm_beta included, is copied.
            for name in self.tensors:
                if name not in folded:
                    self.assertEqual(bytes(model.tensor_bytes(by_name[name])),
                                     bytes(source.tensor_bytes(source.by_name[name])), name)

    def test_partial_groups_and_no_rotation(self):
        out = self.rotate_all(groups="ffn_in,head_in", name="partial.gguf")
        with open_gguf(out) as model:
            self.assertEqual(check_rotation(model, GROUPS), ("ffn_in", "head_in"))
            self.assertEqual(model.kv["prism.hadamard.weight_names"],
                             ["output.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
                              "blk.1.ffn_gate.weight", "blk.1.ffn_up.weight"])
            self.assertEqual(model.kv["prism.hadamard.sign_widths"], [256])
        out = self.rotate_all(groups="none", name="none.gguf")
        with open_gguf(out) as model:
            self.assertEqual(check_rotation(model, GROUPS), ())
            self.assertFalse(any(key.startswith("prism.") for key in model.kv))
            self.assertIn("gufo.rotation.source_sha256", model.kv)

    def rotated_variant(self, name, set_kv=None, drop=(), tensors=None, kv=None):
        """The rotate_all() file with some metadata replaced, dropped or added."""
        rot = self.dir / "rot.gguf"
        if not rot.exists():
            self.rotate_all()
        with open_gguf(rot) as model:
            raw = list(model.kv_raw) if kv is None else kv
            data = {t.name: (t.type_name, t.ne, np.frombuffer(bytes(model.tensor_bytes(t)), np.uint8))
                    for t in model.tensors}
        replaced = {key: encode_value(value) for key, value in (set_kv or {}).items()}
        raw = [(k, replaced.pop(k, v)) for k, v in raw if k not in drop]
        raw += list(replaced.items())
        return write_single(self.dir / name, tensors if tensors is not None else data, kv=raw)

    def test_rotation_metadata_rejections(self):
        with open_gguf(self.rotate_all()) as model:
            names = list(model.kv["prism.hadamard.weight_names"])
            widths = list(model.kv["prism.hadamard.sign_widths"])
            values = list(model.kv["prism.hadamard.sign_values"])
        without = lambda *drop: [n for n in names if n not in drop]  # noqa: E731
        cases = {
            # The converter/gufo subset (the upstream loader accepts these).
            "mixes rotated and unrotated": {"prism.hadamard.weight_names": without("blk.0.attn_gate.weight")},
            "some layers only": {"prism.hadamard.weight_names": without("blk.1.ffn_down.weight")},
            "unsupported prism.hadamard.block_size": {"prism.hadamard.block_size": U32(64)},
            "unsupported prism.hadamard.sign_mode": {"prism.hadamard.sign_mode": "identity"},
            "gdn_v_grouped is not supported": {"prism.hadamard.gdn_v_grouped": True},
            "inverse_weight_names is not supported": {"prism.hadamard.inverse_weight_names": ["token_embd.weight"]},
            "no folded weight uses": {"prism.hadamard.sign_widths": I32Array(widths + [1024]),
                                      "prism.hadamard.sign_values": I32Array(values + [1] * 1024)},
            # Upstream loader rules surface through check_rotation too.
            "not on a verified Hadamard-aware matmul path": {
                "prism.hadamard.weight_names": names + ["blk.0.ssm_alpha.weight"]},
        }
        for index, (message, set_kv) in enumerate(cases.items()):
            with self.subTest(message=message), open_gguf(self.rotated_variant(f"v{index}.gguf", set_kv)) as model:
                with self.assertRaisesRegex(PlanError, message):
                    check_rotation(model, GROUPS)
        stray = self.rotated_variant("stray.gguf", drop=("prism.hadamard.version",))
        with open_gguf(stray) as model, self.assertRaisesRegex(PlanError, "without prism.hadamard.version"):
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
        self.assertIn("prism.hadamard sign vectors: 2 (widths 256, 512; 768 int32 values)", out)
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

    def fake_quantize(self, wrong_type=None, change_copy=False, drop_prism=False):
        """A stand-in llama-quantize: records argv and writes an output GGUF
        with the requested types (zero payloads for converted tensors)."""
        log = self.dir / "argv.json"
        script = self.dir / "fake-llama-quantize"
        script.write_text(FAKE_QUANTIZE.format(python=sys.executable, tools=str(ROOT / "tools"), log=str(log),
                                               wrong=wrong_type, change_copy=change_copy,
                                               drop_prism=drop_prism))
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
        code, stdout, err = run_cli("--groups", "qwen27b", "--input", self.split[0], "--recipe", recipe,
                                    "--llama-quantize", script, "--output", out)
        self.assertEqual(code, 0, err)
        argv = json.loads(log.read_text())
        self.assertEqual(argv[-3:], [str(self.split[0]), str(out), "Q8_0"])
        self.assertIn(f"gufo.rotation.source_sha256=str:{convert.file_sha256(self.split)}", argv)
        self.assertFalse((self.dir / "plain.rotated.gguf").exists())
        # llama-quantize keeps the original's 32-byte alignment; the converter
        # rewrites the output at 256 bytes and removes the intermediate.
        self.assertIn("realigned", stdout)
        self.assertIn("256-byte aligned", stdout)
        self.assertFalse((self.dir / "plain.unaligned.gguf").exists())
        with open_gguf(out) as model:
            self.assertEqual(model.alignment, 256)
            self.assertTrue(all((model.shards[0].data_offset + t.offset) % 256 == 0 for t in model.tensors))

    def test_quantize_command_rejects_float_only_recipe(self):
        with open_gguf(self.split[0]) as model:
            plan = build_plan(model, GROUPS, (), seed=None, rotate_only=False,
                              rules=parse_recipe('{"version": 1, "types": [{"match": ".", "type": "BF16"}]}',
                                                 ALL_GROUPS, GROUPS.recipe_types()).rules)
        self.assertEqual(len(tensor_type_lines(plan)), sum(len(ne) >= 2 for _, ne, _ in self.tensors.values()))
        with self.assertRaisesRegex(PlanError, "quantizes no tensor"):
            quantize_command(plan, "q", "in.gguf", "out.gguf", "t.txt")

    def test_unaligned_output_is_refused(self):
        rot = self.rotate_all()
        recipe = self.recipe({"version": 1, "rotate": "all", "types": [{"match": r"\.weight$", "type": "Q8_0"}]})
        script, _ = self.fake_quantize()
        out = self.dir / "aligned.gguf"
        code, stdout, err = run_cli("--groups", "qwen27b", "--input", rot, "--recipe", recipe,
                                    "--llama-quantize", script, "--output", out)
        self.assertEqual(code, 0, err)
        # llama-quantize packs at 32 even under a 256 key; the converter rewrites it.
        self.assertIn("realigned", stdout)
        self.assertFalse((self.dir / "aligned.unaligned.gguf").exists())
        with open_gguf(rot) as source:
            plan = build_plan(source, GROUPS, ALL_GROUPS, seed=None, rotate_only=False,
                              rules=parse_recipe(recipe.read_text(), ALL_GROUPS, GROUPS.recipe_types()).rules)
        with open_gguf(out) as quantized:
            tensors = [OutTensor(t.name, t.ne, t.type_name, lambda f, b=bytes(quantized.tensor_bytes(t)): f.write(b))
                       for t in quantized.tensors]
            kv = list(quantized.kv_raw)
        narrow = self.dir / "narrow.gguf"
        write_gguf(narrow, kv, tensors, 32)
        with self.assertRaisesRegex(PlanError, "not 256-byte aligned"):
            verify_output(rot, narrow, plan, {})

    def test_quantized_output_is_checked(self):
        rot = self.rotate_all()
        recipe = self.recipe({"version": 1, "rotate": "all", "types": [{"match": r"\.weight$", "type": "Q8_0"}]})
        script, _ = self.fake_quantize()
        code, stdout, err = run_cli("--groups", "qwen27b", "--input", rot, "--recipe", recipe,
                                    "--llama-quantize", script, "--output", self.dir / "ok.gguf")
        self.assertEqual(code, 0, err)
        self.assertIn("types match the plan", stdout)
        self.assertIn("14 folded", stdout)
        cases = [("wrong", dict(wrong_type="blk.0.ffn_gate.weight"), "blk.0.ffn_gate.weight Q4_K != Q8_0"),
                 ("copy", dict(change_copy=True), "unquantized tensor(s) changed bytes"),
                 ("kv", dict(drop_prism=True), "prism.hadamard.sign_values changed or missing")]
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

    def test_llama_quantize_layout_is_read_at_32(self):
        # A 256 general.alignment over 32-byte packing (what llama-quantize writes
        # for a 256-aligned input) is refused as is and realigned when read at 32.
        with open_gguf(self.split[0]) as source:
            tensors = [OutTensor(t.name, t.ne, t.type_name, lambda f, b=bytes(source.tensor_bytes(t)): f.write(b))
                       for t in source.tensors]
            kv = [(k, v) for k, v in source.kv_raw if not k.startswith("split.")]
        mixed = self.dir / "mixed.gguf"
        write_gguf(mixed, kv, tensors, 32, recorded_alignment=256)
        self.assertEqual(shared_gguf.parse_gguf(mixed)["metadata"]["general.alignment"], 256)
        with self.assertRaisesRegex(ValueError, "do not follow 256-byte packing"):
            realign.realign(mixed, self.dir / "bad.gguf")
        out = self.dir / "fixed.gguf"
        summary = realign.realign(mixed, out, layout_alignment=32)
        self.assertEqual((summary["input_alignment"], summary["alignment"]), (32, 256))
        with open_gguf(out) as model, open_gguf(self.split[0]) as source:
            for tensor in model.tensors:
                self.assertEqual((model.shards[0].data_offset + tensor.offset) % 256, 0)
                self.assertEqual(bytes(model.tensor_bytes(tensor)),
                                 bytes(source.tensor_bytes(source.by_name[tensor.name])), tensor.name)

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

    def test_sign_free_rotation(self):
        out = self.dir / "nosign.gguf"
        code, _, err = run_cli("--groups", "qwen27b", "--input", self.split[0], "--rotate-groups", "all",
                               "--signs", "none", "--rotate-only", "--output", out)
        self.assertEqual(code, 0, err)
        with open_gguf(out) as model, open_gguf(self.split[0]) as source:
            self.assertEqual(check_rotation(model, GROUPS), ALL_GROUPS)
            # Written as explicit signs (gufo accepts explicit only), all +1.
            self.assertEqual(model.kv["prism.hadamard.sign_mode"], "explicit")
            self.assertEqual(model.kv["prism.hadamard.sign_widths"], list(WIDTHS))
            self.assertEqual(set(model.kv["prism.hadamard.sign_values"]), {1})
            self.assertEqual(model.kv["gufo.rotation.source_signs"], "none")
            # W' = W·H: each 128-block of a row times the normalized Hadamard matrix.
            h = sylvester(BLOCK_SIZE, np.float32)
            for name in model.kv["prism.hadamard.weight_names"]:
                w = as_float32(source, source.by_name[name])
                n, k = w.shape
                expected = f32_to_bf16_bits((w.reshape(n, k // 128, 128) @ h).reshape(n, k))
                stored = np.frombuffer(bytes(model.tensor_bytes(model.by_name[name])), "<u2").reshape(n, k)
                np.testing.assert_array_equal(stored, expected, name)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertEqual(check_rotated.main(["--groups", "qwen27b", "--source", str(self.split[0]),
                                                 "--rotated", str(out), "--rows", "0"]), 0, log.getvalue())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            check_rotated.main(["--groups", "qwen27b", "--seed", "1", "--source", str(self.split[0]),
                                "--rotated", str(out)])
        # A seeded file records its seed, so check_rotated can verify it without --seed.
        seeded = self.rotate_all(seed=11)
        with open_gguf(seeded) as model:
            self.assertEqual(model.kv["gufo.rotation.source_signs"], "random seed=11")
            self.assertGreater(len(set(model.kv["prism.hadamard.sign_values"])), 1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check_rotated.main(["--groups", "qwen27b", "--source", str(self.split[0]),
                                                 "--rotated", str(seeded)]), 0)
        # The effective recipe records the mode; refusals.
        with open_gguf(self.split[0]) as model:
            plan = build_plan(model, GROUPS, ALL_GROUPS, seed=None, rotate_only=True, sign_mode="none")
        self.assertEqual(json.loads(plan.recipe_json(False))["signs"], "none")
        cases = [("does not apply to --signs none", ["--input", self.split[0], "--rotate-groups", "all",
                                                    "--signs", "none", "--seed", 1, "--rotate-only",
                                                    "--output", self.dir / "x.gguf"]),
                 ("--signs cannot change an already rotated", ["--input", out, "--rotate-groups", "all",
                                                               "--signs", "none", "--dry-run"])]
        for message, args in cases:
            with self.subTest(message=message):
                code, _, err = run_cli("--groups", "qwen27b", *args)
                self.assertEqual(code, 2, err)
                self.assertIn(message, err)

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
            folded = model.kv["prism.hadamard.weight_names"]
            self.assertNotIn("blk.2.attn_q.weight", folded)
            self.assertIn("blk.1.attn_q.weight", folded)
            self.assertEqual(bytes(model.tensor_bytes(model.by_name["blk.2.ffn_down.weight"])),
                             bytes(source.tensor_bytes(source.by_name["blk.2.ffn_down.weight"])))
            self.assertEqual(check_rotation(model, GROUPS), ALL_GROUPS)
        # A NextN weight listed as folded names the tensor and is refused.
        with open_gguf(out) as model:
            raw = [(k, encode_value(model.kv["prism.hadamard.weight_names"] + ["blk.2.ffn_down.weight"])
                    if k == "prism.hadamard.weight_names" else v) for k, v in model.kv_raw]
        bad = write_single(self.dir / "mtp-bad.gguf", tensors, kv=raw)
        with open_gguf(bad) as model, self.assertRaisesRegex(PlanError, "blk.2.ffn_down.weight is rotated"):
            check_rotation(model, GROUPS)
        missing = write_single(self.dir / "noblocks.gguf", self.tensors,
                               kv=[e for e in kv_entries() if e[0] != "qwen35.block_count"])
        code, _, err = run_cli("--groups", "qwen27b", "--input", missing, "--rotate-groups", "none",
                               "--rotate-only", "--dry-run")
        self.assertEqual(code, 2)
        self.assertIn("rotated block range", err)



class PrismUpstreamRulesTest(unittest.TestCase):
    """rotquant/prism.read_prism refuses what the upstream loader refuses.

    Each case names the line of ggml-org/llama.cpp 8ab29fae
    src/llama-model.cpp (load_hparams_hadamard / load_tensors_hadamard)
    whose refusal it mirrors; the message must carry that line.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        split = write_split(self.dir, toy_tensors(np.random.default_rng(5)))
        self.rot = self.dir / "rot.gguf"
        code, _, err = run_cli("--groups", "qwen27b", "--input", split[0], "--rotate-groups", "all",
                               "--seed", 11, "--rotate-only", "--output", self.rot)
        self.assertEqual(code, 0, err)
        with open_gguf(self.rot) as model:
            self.kv_raw = list(model.kv_raw)
            self.names = list(model.kv["prism.hadamard.weight_names"])
            self.widths = list(model.kv["prism.hadamard.sign_widths"])
            self.values = list(model.kv["prism.hadamard.sign_values"])
            self.data = {t.name: (t.type_name, t.ne, np.frombuffer(bytes(model.tensor_bytes(t)), np.uint8))
                         for t in model.tensors}

    def tearDown(self):
        self.tmp.cleanup()

    def variant(self, name, set_kv=(), drop=(), raw=(), untied_output=True):
        encoded = {k: encode_value(v) for k, v in dict(set_kv).items()}
        encoded.update(dict(raw))
        kv = [(k, encoded.pop(k, v)) for k, v in self.kv_raw if k not in drop]
        kv += list(encoded.items())
        data = self.data if untied_output else {k: v for k, v in self.data.items() if k != "output.weight"}
        return write_single(self.dir / name, data, kv=kv)

    def test_converter_output_passes(self):
        with open_gguf(self.rot) as model:
            info = prism.read_prism(model)
        self.assertEqual((info.version, info.block_size, info.sign_mode), (1, 128, "explicit"))
        self.assertEqual(sorted(info.signs), list(WIDTHS))
        self.assertEqual(info.weight_names, tuple(self.names))

    def test_refusals_match_upstream_lines(self):
        widths, values, names = self.widths, self.values, self.names
        no_head = [n for n in names if n != "output.weight"]
        cases = [
            (2078, dict(set_kv={prism.VERSION: U32(3)})),
            (2082, dict(set_kv={prism.VERSION: U32(2)})),
            (2085, dict(set_kv={prism.VERSION: U32(2), prism.TIED_OUTPUT: True})),
            (2224, dict(set_kv={prism.TIED_OUTPUT: True}, drop=(prism.VERSION,))),
            (2101, dict(set_kv={prism.BLOCK_SIZE: U32(96)})),
            (2104, dict(set_kv={prism.TRANSFORM: "walsh"})),
            (2107, dict(set_kv={prism.AXIS: "output"})),
            (2110, dict(set_kv={prism.SIGN_MODE: "random"})),
            (2113, dict(set_kv={prism.WEIGHT_NAMES: []})),
            (2123, dict(set_kv={prism.SIGN_WIDTHS: I32Array(), prism.SIGN_VALUES: I32Array()})),
            (2128, dict(set_kv={prism.SIGN_WIDTHS: I32Array(widths + [200]),
                                prism.SIGN_VALUES: I32Array(values + [1] * 200)})),
            (2131, dict(set_kv={prism.SIGN_WIDTHS: I32Array([256] + widths),
                                prism.SIGN_VALUES: I32Array([1] * 256 + values)})),
            (2137, dict(set_kv={prism.SIGN_VALUES: I32Array([2] + values[1:])})),
            (2143, dict(set_kv={prism.SIGN_VALUES: I32Array(values + [1])})),
            (2159, dict(set_kv={"general.architecture": "gemma3"})),
            (2199, dict(set_kv={prism.WEIGHT_NAMES: names + ["blk.0.ssm_beta.weight"]})),
            (2199, dict(set_kv={prism.WEIGHT_NAMES: names + ["token_embd.weight"]})),
            (2203, dict(set_kv={prism.WEIGHT_NAMES: names + names[:1]})),
            (2213, dict(set_kv={prism.INVERSE_WEIGHT_NAMES: ["output.weight"]})),
            (2217, dict(set_kv={prism.INVERSE_WEIGHT_NAMES: ["token_embd.weight", "token_embd.weight"]})),
            (2228, dict(set_kv={prism.VERSION: U32(2), prism.TIED_OUTPUT: True, prism.WEIGHT_NAMES: no_head},
                        untied_output=False)),
            (2232, dict(set_kv={prism.INVERSE_WEIGHT_NAMES: ["token_embd.weight"], prism.WEIGHT_NAMES: no_head},
                        untied_output=False)),
            (2271, dict(set_kv={prism.WEIGHT_NAMES: names + ["blk.5.ffn_up.weight"]})),
            (2274, dict(set_kv={prism.SIGN_MODE: "identity", prism.BLOCK_SIZE: U32(512)})),
            (2348, dict(set_kv={prism.SIGN_WIDTHS: I32Array([256, 512]),
                                prism.SIGN_VALUES: I32Array(values[:256] + values[256 + 384:])})),
        ]
        for index, (line, spec) in enumerate(cases):
            with self.subTest(line=line, case=index):
                with open_gguf(self.variant(f"u{index}.gguf", **spec)) as model:
                    with self.assertRaisesRegex(prism.PrismError, f"llama-model.cpp:{line}\\)$"):
                        prism.read_prism(model)
        # Keys must carry the GGUF type the upstream getters accept.
        bad_type = self.variant("type.gguf", raw={prism.VERSION: struct.pack("<Ii", 5, 1)})
        with open_gguf(bad_type) as model, self.assertRaisesRegex(prism.PrismError, "wrong GGUF type"):
            prism.read_prism(model)
        # Identity mode needs no sign table; the upstream loader accepts it.
        identity = self.variant("identity.gguf", set_kv={prism.SIGN_MODE: "identity"},
                                drop=(prism.SIGN_WIDTHS, prism.SIGN_VALUES))
        with open_gguf(identity) as model:
            self.assertEqual(prism.read_prism(model).signs, {})


if __name__ == "__main__":
    unittest.main()
