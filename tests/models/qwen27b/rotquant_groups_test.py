#!/usr/bin/env python3
"""The Qwen3.8 27B rotation group table matches the plan and the loader.

Run with NumPy and PyYAML: python3 -I tests/models/qwen27b/rotquant_groups_test.py
"""
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from rotquant.plan import load_groups  # noqa: E402

GROUPS = load_groups("qwen27b")


def layer_names(layer: int, attention: bool) -> list:
    p = f"blk.{layer}."
    names = [p + n for n in ("attn_norm.weight", "post_attention_norm.weight", "ffn_gate.weight",
                             "ffn_up.weight", "ffn_down.weight")]
    if attention:
        names += [p + n for n in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                  "attn_q_norm.weight", "attn_k_norm.weight")]
    else:
        names += [p + n for n in ("attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight",
                                  "ssm_beta.weight", "ssm_out.weight", "ssm_a", "ssm_conv1d.weight",
                                  "ssm_dt.bias", "ssm_norm.weight")]
    return names


class GroupTableTest(unittest.TestCase):
    def test_plan_table(self):
        expected = {
            "attn_in": {"attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "attn_q", "attn_k", "attn_v"},
            "ssm_out_in": {"ssm_out"},
            "attn_out_in": {"attn_output"},
            "ffn_in": {"ffn_gate", "ffn_up"},
            "ffn_down_in": {"ffn_down"},
        }
        self.assertEqual(GROUPS.architecture, "qwen35")
        self.assertEqual(GROUPS.names, ("attn_in", "ssm_out_in", "attn_out_in", "ffn_in",
                                        "ffn_down_in", "head_in"))
        names = ["token_embd.weight", "output_norm.weight", "output.weight"]
        for layer in range(64):
            names += layer_names(layer, attention=layer % 4 == 3)
        seen = {}
        for name in names:
            group, instance = GROUPS.instance(name)
            role = GROUPS.role(name)
            self.assertIsNotNone(role, name)
            if group is None:
                self.assertNotEqual(role, "projection", name)
                continue
            self.assertEqual(role, "projection", name)
            if group == "head_in":
                self.assertEqual((name, instance), ("output.weight", "head_in"))
                continue
            layer, short = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", name).groups()
            self.assertIn(short, expected[group], name)
            self.assertEqual(instance, f"{group}:{layer}")
            seen.setdefault(group, set()).add(short)
        self.assertEqual(seen, expected)
        self.assertEqual(GROUPS.instance("token_embd.weight"), (None, None))
        self.assertEqual(GROUPS.tied_heads, {"head_in": "token_embd.weight"})
        self.assertEqual(GROUPS.role("blk.3.ssm_conv1d.weight"), "ssm_parameter")
        self.assertEqual(GROUPS.role("blk.3.ffn_norm.weight"), "norm")
        self.assertIsNone(GROUPS.role("blk.0.attn_qkv.weight.rot"))

    def test_supported_types_follow_the_loader(self):
        source = (ROOT / "src/models/qwen/weights.cpp").read_text()
        body = source[source.index("constexpr bool SupportsTensorType"):]
        body = body[:body.index("\n}\n")]

        def case(role):
            start = body.index(f"case TensorRole::{role}:")
            ret = body.index("return", start)
            return {m for m in re.findall(r"GgmlType::k(\w+)", body[ret:body.index(";", ret)])}

        projection = case("kProjection")
        self.assertEqual(GROUPS.supported["projection"], projection - {"Q8_K"})
        self.assertIn("Q8_K", projection)
        self.assertEqual(GROUPS.supported["embedding"], case("kEmbedding"))
        self.assertEqual(GROUPS.supported["norm"], case("kSsmParameter"))
        self.assertEqual(GROUPS.supported["ssm_parameter"], {"F32"})


if __name__ == "__main__":
    unittest.main()
