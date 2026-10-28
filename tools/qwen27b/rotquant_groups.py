"""Qwen3.8 27B input groups and tensor roles for tools/rotquant.

Matrices that consume the same activation share one sign vector so the
activation is rotated once. A group instance is one group in one layer.
Groups follow the UD-Q4_K_XL header (hidden 5120); see
plan/rotated-quant/PLAN.md, "27B 입력 그룹".
"""
from __future__ import annotations

import re

ARCHITECTURE = "qwen35"

# Group name -> full-match tensor-name pattern. `layer` names the instance.
GROUPS = {
    # attn_norm output: DeltaNet attn_qkv/attn_gate/ssm_alpha/ssm_beta and
    # full-attention attn_q/attn_k/attn_v.
    "attn_in": r"blk\.(?P<layer>\d+)\.(attn_qkv|attn_gate|ssm_alpha|ssm_beta|attn_q|attn_k|attn_v)\.weight",
    # SSM per-head norm and SiLU gate output.
    "ssm_out_in": r"blk\.(?P<layer>\d+)\.ssm_out\.weight",
    # Attention output after its gate.
    "attn_out_in": r"blk\.(?P<layer>\d+)\.attn_output\.weight",
    # post_attention_norm output.
    "ffn_in": r"blk\.(?P<layer>\d+)\.(ffn_gate|ffn_up)\.weight",
    # SwiGLU output.
    "ffn_down_in": r"blk\.(?P<layer>\d+)\.ffn_down\.weight",
    # output_norm output. The DFlash2 drafter shares this head, so its head
    # path must rotate too once the loader accepts rotated heads.
    "head_in": r"output\.weight",
}

# Rotating a head that is tied to the embedding would change the lookup.
# The loader falls back to token_embd when output.weight is absent.
TIED_HEADS = {"head_in": "token_embd.weight"}

# Types accepted by src/models/qwen/weights.cpp SupportsTensorType. Q8_K is
# accepted there for projections but is an activation format that
# llama-quantize cannot target, so recipes cannot select it.
_PROJECTION = {"F32", "BF16", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "IQ4_NL", "IQ4_XS", "IQ3_S"}
_EMBEDDING = {"F32", "BF16", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "IQ4_NL", "IQ4_XS", "IQ3_S"}
SUPPORTED_TYPES = {
    "embedding": _EMBEDDING,
    "projection": _PROJECTION,
    "norm": {"F32"},
    "ssm_parameter": {"F32"},
}
# Roles whose type comes from the recipe; other roles keep the source type.
RECIPE_ROLES = ("embedding", "projection")

_PROJECTION_NAMES = re.compile(
    r"output\.weight|blk\.\d+\.(attn_qkv|attn_gate|ssm_alpha|ssm_beta|attn_q|attn_k|attn_v|"
    r"attn_output|ssm_out|ffn_gate|ffn_up|ffn_down)\.weight")
_NORM_NAMES = re.compile(
    r"output_norm\.weight|blk\.\d+\.(attn_norm|ffn_norm|post_attention_norm|attn_q_norm|attn_k_norm|ssm_norm)\.weight")
_SSM_NAMES = re.compile(r"blk\.\d+\.(ssm_a|ssm_conv1d\.weight|ssm_dt\.bias)")


def role(name: str) -> str | None:
    """Loader role of a tensor, or None for tensors the 27B loader ignores."""
    if name == "token_embd.weight":
        return "embedding"
    if _PROJECTION_NAMES.fullmatch(name):
        return "projection"
    if _NORM_NAMES.fullmatch(name):
        return "norm"
    if _SSM_NAMES.fullmatch(name):
        return "ssm_parameter"
    return None
