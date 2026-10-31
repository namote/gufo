#!/usr/bin/env python3
"""Flash-Next disk oracle: header checks and optional tiny HF numerical replay.

python3 -I tests/models/qwen38_flash_next/rq_quality_test.py
The numerical tests need torch, accelerate, numpy and transformers==5.17.0.
A dependency skip is not a numerical quality pass.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/qwen38_flash_next/rq_quality"))
import bf16_reference as oracle  # noqa: E402


def checkpoint(directory, tensors):
    """Independent minimal safetensors fixture; tensors: name -> (dtype, shape, bytes)."""
    header, payload = {}, bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = dict(dtype=dtype, shape=shape,
                            data_offsets=[len(payload), len(payload) + len(data)])
        payload.extend(data)
    raw = json.dumps(header).encode()
    (directory / "weights.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    (directory / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "weights.safetensors" for name in tensors}}))


class HeaderTest(unittest.TestCase):
    def test_offsets_and_numeric_shard_order(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            names = [f"embedding.shard_{i}.weight" for i in range(12)]
            checkpoint(path, {name: ("BF16", [2, 3], b"\0" * 12) for name in reversed(names)})
            cp = oracle.Checkpoint(path)
            self.assertEqual(cp.embedding_parts("embedding.weight"), names)
            file, offset, entry = cp.tensor(names[0])
            with file.open("rb") as f:
                f.seek(offset)
                self.assertEqual(f.read(12), b"\0" * 12)
            self.assertEqual(entry["shape"], [2, 3])
            with file.open("r+b") as f:
                f.truncate(file.stat().st_size - 1)
            with self.assertRaisesRegex(ValueError, "invalid tensor range"):
                cp.tensor(names[0])

    def test_missing_shard_is_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            checkpoint(path, {"embedding.shard_1.weight": ("BF16", [1, 1], b"\0\0")})
            with self.assertRaisesRegex(ValueError, "non-contiguous"):
                oracle.Checkpoint(path).embedding_parts("embedding.weight")

    def test_hash_buffers_and_prefix_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            checkpoint(path, {"hash": ("I64", [3], struct.pack("<qqq", 1, 2, 3))})
            self.assertEqual(oracle.Checkpoint(path).tensor("hash")[2]["dtype"], "I64")
        self.assertEqual(oracle.source_key("model.layers.1.linear_attn.A_log"),
                         "model.language_model.layers.1.linear_attn.A_log")
        self.assertEqual(oracle.source_key("lm_head.weight"), "lm_head.weight")


HAS_RUNTIME = all(importlib.util.find_spec(name) is not None
                  for name in ("torch", "transformers", "accelerate", "numpy"))


@unittest.skipUnless(HAS_RUNTIME, "HF numerical runtime absent (not a quality pass)")
class StreamingTest(unittest.TestCase):
    def test_streaming_matches_resident_prefill_decode_and_ple(self):
        import torch
        from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

        torch.set_num_threads(1)
        config = Qwen4ExpTextConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            hc_count=4, hc_lowrank=8, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
            moe_intermediate_size=16, shared_expert_intermediate_size=16,
            num_experts=4, num_experts_per_tok=2, ple_layer_ids=[2],
            ple_embed_dim=32, ngram_size=2, heads_per_ngram=2,
            ngram_vocab_size_base=17, make_ngram_vocab_size_divisible_by=2,
            split_ngram_parts=2, indexer_n_heads=2, indexer_kv_heads=1,
            indexer_head_dim=8, indexer_budget=4, indexer_compress_ratio=4,
            max_position_embeddings=64, eos_token_id=63,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0]})
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            torch.manual_seed(17)
            resident = Qwen4ExpForCausalLM(config).eval()
        finally:
            torch.set_default_dtype(previous)
        tensors = {}
        for name, value in resident.state_dict().items():
            key = oracle.source_key(name)
            chunks = list(value.chunk(2, dim=0)) if name.endswith(".ngram_embedding.weight") else [value]
            for i, chunk in enumerate(chunks):
                target = key.removesuffix(".weight") + f".shard_{i}.weight" if len(chunks) > 1 else key
                dtype = {torch.bfloat16: "BF16", torch.float32: "F32", torch.int64: "I64"}[chunk.dtype]
                tensors[target] = (dtype, list(chunk.shape), chunk.contiguous().view(torch.uint8).numpy().tobytes())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            checkpoint(path, tensors)
            (path / "config.json").write_text(json.dumps({
                "model_type": "qwen4_exp", "text_config": config.to_dict()}))
            streamed, _ = oracle.build_model(oracle.Checkpoint(path))
            ids = torch.tensor([[1, 5, 9, 63, 3, 2, 7, 4, 8, 6]])
            with torch.inference_mode():
                # Include EOS history reset and sparse-indexer budget crossing.
                self.assertTrue(torch.equal(resident(ids, use_cache=False).logits,
                                            streamed(ids, use_cache=False).logits))
                a = resident(ids[:, :6], use_cache=True)
                b = streamed(ids[:, :6], use_cache=True)
                self.assertTrue(torch.equal(a.logits, b.logits))
                for pos in range(6, ids.shape[1]):
                    a = resident(ids[:, pos:pos + 1], past_key_values=a.past_key_values, use_cache=True)
                    b = streamed(ids[:, pos:pos + 1], past_key_values=b.past_key_values, use_cache=True)
                    self.assertTrue(torch.equal(a.logits, b.logits))
                emb_name = "model.layers.1.ple.ple_embedding.ngram_embedding"
                original = resident.get_submodule(emb_name)
                lazy = streamed.get_submodule(emb_name)
                rows = torch.tensor([[0, original.num_embeddings // 2, original.num_embeddings - 1]])
                self.assertTrue(torch.equal(original(rows), lazy(rows)))
            self.assertTrue(all(param.is_meta for param in streamed.parameters()))


if __name__ == "__main__":
    unittest.main()
