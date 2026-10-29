"""Upstream llama.cpp `prism.hadamard.*` metadata: write and validate.

A rotated GGUF stores W' = W·D·H for the folded weights and declares them in
`prism.hadamard.weight_names`; the runtime computes y = W'·(H·(s·x)), where
H is the normalized Sylvester-Hadamard matrix applied in blocks along the
input (K) axis and `s` is one ±1 sign vector **per input width**, shared by
every folded weight with that K. The file has no extra tensors.

`read_prism` applies the refusals of the upstream loader so a file the
converter accepts also loads in llama.cpp. References are to ggml-org/llama.cpp
`8ab29faec401829db0e617f40af2c942f4a9eaab` (format unchanged since the merge
`a9d27ac6`, #29600): `src/llama-model.cpp` `load_hparams_hadamard`
(2073-2234) and `load_tensors_hadamard` (2238-2398).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import struct

import numpy as np

from rotquant.gguf_io import T_ARRAY, T_BOOL, T_I32, T_STRING, T_U32, I32Array, U32, encode_value

UPSTREAM = "llama.cpp 8ab29fae src/llama-model.cpp"
PREFIX = "prism.hadamard."
VERSION = PREFIX + "version"
TIED_OUTPUT = PREFIX + "tied_output"
BLOCK_SIZE = PREFIX + "block_size"
TRANSFORM = PREFIX + "transform"
AXIS = PREFIX + "axis"
SIGN_MODE = PREFIX + "sign_mode"
SIGN_WIDTHS = PREFIX + "sign_widths"
SIGN_VALUES = PREFIX + "sign_values"
WEIGHT_NAMES = PREFIX + "weight_names"
INVERSE_WEIGHT_NAMES = PREFIX + "inverse_weight_names"
GDN_V_GROUPED = PREFIX + "gdn_v_grouped"

TRANSFORM_NAME = "normalized-sylvester-walsh-hadamard"
AXIS_NAME = "input-last-dimension"


# llama-model.cpp:2151-2163: archs whose matmuls all go through build_lora_mm.
ARCHITECTURES = frozenset({"llama", "qwen3", "qwen3moe", "qwen35", "qwen35moe", "qwen3next"})
# llama-model.cpp:2166-2199 (and conversion/base.py:754): foldable weights.
FOLDABLE = re.compile(
    r"output\.weight|blk\.\d+\.("
    r"attn_q|attn_k|attn_v|attn_qkv|attn_gate|attn_output"
    r"|ffn_gate|ffn_up|ffn_down"
    r"|ffn_gate_exps|ffn_up_exps|ffn_down_exps|ffn_gate_up_exps"
    r"|ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp"
    r"|ssm_out"
    r")\.weight")


class PrismError(ValueError):
    """The metadata breaks a rule of the upstream loader (line in the message)."""


@dataclass
class Prism:
    version: int
    block_size: int
    sign_mode: str
    signs: dict = field(default_factory=dict)     # width -> float32 ±1 vector (explicit mode)
    weight_names: tuple = ()
    inverse_weight_names: tuple = ()
    tied_output: bool = False
    gdn_v_grouped: bool = False


def _fail(line: int, message: str):
    raise PrismError(f"{message} ({UPSTREAM}:{line})")


def _tag(model, key):
    raw = dict(model.kv_raw).get(key)
    if raw is None:
        return None, None
    tag = struct.unpack_from("<I", raw)[0]
    sub = struct.unpack_from("<I", raw, 4)[0] if tag == T_ARRAY else None
    return tag, sub


def _get(model, key, tag, sub=None, required=True):
    """Read a key with the GGUF type the upstream getter accepts."""
    got_tag, got_sub = _tag(model, key)
    if got_tag is None:
        if required:
            raise PrismError(f"key not found in model: {key} (llama-model-loader get_key/get_arr)")
        return None
    accepted = sub if isinstance(sub, tuple) else (sub,)
    if got_tag != tag or (tag == T_ARRAY and got_sub not in accepted):
        raise PrismError(f"{key} has the wrong GGUF type (llama-model-loader get_key/get_arr)")
    return model.kv[key]


def read_prism(model) -> Prism | None:
    """Validate like the upstream loader; None when the file declares no folding."""
    tied = bool(_get(model, TIED_OUTPUT, T_BOOL, required=False) or False)
    version = _get(model, VERSION, T_U32, required=False)
    if version is None:
        if tied:
            _fail(2224, "prism.hadamard.tied_output requires version 2")
        return None
    if version not in (1, 2):
        _fail(2078, f"unsupported prism.hadamard.version: {version}")
    if (version == 2) != tied:
        _fail(2082, "prism.hadamard version 2 requires tied_output=true; version 1 forbids it")
    names_present = {t.name for t in model.tensors}
    if tied and "output.weight" in names_present:
        _fail(2085, "prism.hadamard.tied_output requires output.weight to be absent")
    block = _get(model, BLOCK_SIZE, T_U32)
    transform = _get(model, TRANSFORM, T_STRING)
    axis = _get(model, AXIS, T_STRING)
    sign_mode = _get(model, SIGN_MODE, T_STRING)
    weight_names = _get(model, WEIGHT_NAMES, T_ARRAY, T_STRING)
    if block == 0 or block & (block - 1):
        _fail(2101, f"invalid prism.hadamard.block_size: {block}")
    if transform != TRANSFORM_NAME:
        _fail(2104, f"unsupported prism.hadamard.transform: {transform}")
    if axis != AXIS_NAME:
        _fail(2107, f"unsupported prism.hadamard.axis: {axis}")
    if sign_mode not in ("identity", "explicit"):
        _fail(2110, f"unsupported prism.hadamard.sign_mode: {sign_mode}")
    if not weight_names:
        _fail(2113, "prism.hadamard.weight_names is empty")
    signs = {}
    if sign_mode == "explicit":
        widths = _get(model, SIGN_WIDTHS, T_ARRAY, (T_I32, T_U32))
        values = _get(model, SIGN_VALUES, T_ARRAY, (T_I32, T_U32))
        if not widths:
            _fail(2123, "prism.hadamard.sign_mode is explicit but sign_widths is empty")
        offset = 0
        for width in widths:
            if width <= 0 or width % block or offset + width > len(values):
                _fail(2128, f"invalid prism.hadamard sign width: {width}")
            if width in signs:
                _fail(2131, f"duplicate prism.hadamard sign width: {width}")
            vector = np.asarray(values[offset:offset + width], dtype=np.int64)
            if not np.isin(vector, (1, -1)).all():
                _fail(2137, "prism.hadamard sign values must be +/-1")
            signs[width] = vector.astype(np.float32)
            offset += width
        if offset != len(values):
            _fail(2143, "prism.hadamard.sign_values length mismatch")
    gdn = bool(_get(model, GDN_V_GROUPED, T_BOOL, required=False) or False)
    arch = model.kv.get("general.architecture")
    if arch not in ARCHITECTURES:
        _fail(2159, f"prism.hadamard: arch '{arch}' is not verified to apply the activation transform")
    seen = set()
    for name in weight_names:
        if not FOLDABLE.fullmatch(name):
            _fail(2199, f"prism.hadamard: weight '{name}' is not on a verified Hadamard-aware matmul path")
        if name in seen:
            _fail(2203, f"duplicate prism.hadamard weight: {name}")
        seen.add(name)
    inverse = _get(model, INVERSE_WEIGHT_NAMES, T_ARRAY, T_STRING, required=False) or []
    inverse_seen = set()
    for name in inverse:
        if name != "token_embd.weight":
            _fail(2213, f"prism.hadamard: weight '{name}' is not a verified inverse-after-lookup table")
        if name in seen or name in inverse_seen:
            _fail(2217, f"duplicate prism.hadamard inverse weight: {name}")
        inverse_seen.add(name)
    if tied and "token_embd.weight" not in inverse_seen:
        _fail(2228, "prism.hadamard.tied_output requires a latent token embedding")
    if not tied and "token_embd.weight" in inverse_seen and "output.weight" not in names_present:
        _fail(2232, "a tied Hadamard output requires version 2 and tied_output=true")
    by_name = model.by_name
    for name in list(weight_names) + list(inverse):
        tensor = by_name.get(name)
        if tensor is None:
            _fail(2271, f"prism.hadamard weight not found: {name}")
        if tensor.ne[0] % block:
            _fail(2274, f"prism.hadamard block size {block} does not divide input dimension "
                        f"{tensor.ne[0]} for {name}")
        if signs and tensor.ne[0] not in signs:
            _fail(2348, f"prism.hadamard has no sign vector for width {tensor.ne[0]} ({name})")
    return Prism(version, block, sign_mode, signs, tuple(weight_names), tuple(inverse), tied, gdn)


def prism_kv(block_size: int, weight_names, signs_by_width: dict) -> list:
    """(key, raw value) entries for an untied, explicit-sign version 1 file."""
    widths = sorted(signs_by_width)
    values = []
    for width in widths:
        values.extend(int(v) for v in signs_by_width[width])
    entries = [
        (VERSION, U32(1)),
        (BLOCK_SIZE, U32(block_size)),
        (TRANSFORM, TRANSFORM_NAME),
        (AXIS, AXIS_NAME),
        (SIGN_MODE, "explicit"),
        (SIGN_WIDTHS, I32Array(widths)),
        (SIGN_VALUES, I32Array(values)),
        (WEIGHT_NAMES, list(weight_names)),
    ]
    return [(key, encode_value(value)) for key, value in entries]
