#!/usr/bin/env python3
"""Bits per weight of 27B checkpoints by the W2 definition, from tensor headers only.

W2 (R0-RESULTS "bpw"): 2-D weights of the text model including the output head, excluding the
token embedding and the NextN/MTP block. GGUF: tensors with at least two non-unit dimensions,
without token_embd and blk.64 (the scripts reproduce R0's UD-Q4_K_XL 5.1450 and UD-Q4_K_M
4.8032). EXL3 safetensors: every quantized linear of model.language_model.layers.* and lm_head
counts its trellis plus its suh/svh/mul1 bytes over in x out weights; the unquantized 2-D layer
tensors (in_proj_a/b) count as stored; embed_tokens, model.visual.* and mtp.* are excluded.
"""
from __future__ import annotations

import json
import math
import re
import struct
import sys
from collections import Counter
from pathlib import Path

GGML = {0: ('F32', 1, 4), 1: ('F16', 1, 2), 2: ('Q4_0', 32, 18), 3: ('Q4_1', 32, 20), 6: ('Q5_0', 32, 22),
        7: ('Q5_1', 32, 24), 8: ('Q8_0', 32, 34), 10: ('Q2_K', 256, 84), 11: ('Q3_K', 256, 110),
        12: ('Q4_K', 256, 144), 13: ('Q5_K', 256, 176), 14: ('Q6_K', 256, 210), 15: ('Q8_K', 256, 292),
        16: ('IQ2_XXS', 256, 66), 17: ('IQ2_XS', 256, 74), 18: ('IQ3_XXS', 256, 98), 19: ('IQ1_S', 256, 50),
        20: ('IQ4_NL', 32, 18), 21: ('IQ3_S', 256, 110), 22: ('IQ2_S', 256, 82), 23: ('IQ4_XS', 256, 136),
        24: ('I8', 1, 1), 25: ('I16', 1, 2), 26: ('I32', 1, 4), 29: ('IQ1_M', 256, 56), 30: ('BF16', 1, 2)}
ST = {'BF16': 2, 'F16': 2, 'F32': 4, 'I16': 2, 'I32': 4, 'I8': 1, 'U8': 1}
SCALAR = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def _rd(f, fmt):
    return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))


def _skip_value(f, t):
    if t in SCALAR:
        f.seek(struct.calcsize(SCALAR[t]), 1)
    elif t == 8:
        (n,) = _rd(f, 'Q'); f.seek(n, 1)
    elif t == 9:
        et, n = _rd(f, 'IQ')
        if et in SCALAR:
            f.seek(n * struct.calcsize(SCALAR[et]), 1)
        else:
            for _ in range(n):
                _skip_value(f, et)
    else:
        raise ValueError(f'unknown GGUF value type {t}')


def gguf_tensors(path):
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise ValueError(f'{path}: not GGUF')
        _, nt, nkv = _rd(f, 'IQQ')
        for _ in range(nkv):
            (n,) = _rd(f, 'Q'); f.seek(n, 1)
            (t,) = _rd(f, 'I'); _skip_value(f, t)
        out = []
        for _ in range(nt):
            (n,) = _rd(f, 'Q'); name = f.read(n).decode()
            (nd,) = _rd(f, 'I'); dims = _rd(f, 'Q' * nd); ty, _ = _rd(f, 'IQ')
            out.append((name, dims, ty))
    return out


def gguf_bpw(path) -> dict:
    total_b = total_e = 0
    types = Counter()
    for name, dims, ty in gguf_tensors(path):
        tname, be, bb = GGML[ty]
        n = math.prod(dims)
        if len([d for d in dims if d > 1]) < 2 or name.startswith('token_embd') or name.startswith('blk.64.'):
            continue
        total_e += n; total_b += n // be * bb; types[tname] += n // be * bb
    return dict(file=str(path), bpw=total_b * 8 / total_e, bytes=total_b, elements=total_e,
                bytes_by_type=dict(types.most_common()))


def safetensors_headers(directory):
    h = {}
    for p in sorted(Path(directory).glob('*.safetensors')):
        with open(p, 'rb') as f:
            (n,) = struct.unpack('<Q', f.read(8))
            h.update(json.loads(f.read(n)))
    h.pop('__metadata__', None)
    return h


def exl3_bpw(directory) -> dict:
    h = safetensors_headers(directory)
    nb = lambda n: math.prod(h[n]['shape']) * ST[h[n]['dtype']]
    total_b = total_e = 0
    bits = Counter()
    for n in h:
        if not n.startswith(('model.language_model.layers', 'lm_head')):
            continue
        if n.endswith('.trellis'):
            base = n[:-len('.trellis')]
            e = h[base + '.suh']['shape'][0] * h[base + '.svh']['shape'][0]
            total_e += e
            total_b += nb(n) + sum(nb(base + s) for s in ('.suh', '.svh', '.mul1') if base + s in h)
            bits[h[n]['shape'][2] // 16] += e
        elif n.startswith('model.language_model.layers') and len(h[n]['shape']) == 2 and n.endswith('.weight'):
            total_e += math.prod(h[n]['shape']); total_b += nb(n)
    return dict(dir=str(directory), bpw=total_b * 8 / total_e, bytes=total_b, elements=total_e,
                trellis_bits_share={k: v / total_e for k, v in sorted(bits.items())})


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        report = exl3_bpw(arg) if Path(arg).is_dir() else gguf_bpw(arg)
        print(json.dumps(report))
