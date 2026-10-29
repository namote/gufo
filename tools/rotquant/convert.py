#!/usr/bin/env python3
"""Rotated-weight converter: Hadamard-rotate input groups, then quantize.

Stage 1 (rotate) streams a BF16 GGUF (single or split) one tensor at a time
and writes W' = W·D·H for every tensor in a rotated input group, with the
upstream llama.cpp prism.hadamard metadata: one ±1 sign vector D per input
width K and the list of folded weights (rotquant/prism.py).
Stage 2 (quantize) writes exact per-tensor --tensor-type rules, runs the
reference llama-quantize on the rotated file and reads the output back
(rotquant/verify.py): types, unchanged bytes and prism/gufo metadata must match.
An already rotated input skips stage 1, so one rotated BF16 file serves
many recipes. NextN (MTP) blocks are never rotated.

Examples (see plan/rotated-quant/R1A-RESULTS.md):
  convert.py --groups qwen27b --input BF16-00001-of-00002.gguf --rotate-groups all \\
      --seed 1 --rotate-only --output rot-bf16.gguf
  convert.py --groups qwen27b --input rot-bf16.gguf --rotate-groups all \\
      --recipe q4.json --llama-quantize llama-quantize --output rot-q4.gguf
  convert.py --groups qwen27b --input BF16-00001-of-00002.gguf --rotate-groups all \\
      --copy-recipe-from UD-Q4_K_XL.gguf --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from rotquant import CONVERTER_VERSION  # noqa: E402
from rotquant import prism  # noqa: E402
from rotquant.gguf_io import (SPLIT_KEYS, WRITE_ALIGNMENT, OutTensor, as_float32,  # noqa: E402
                              encode_value, layout_is_packed, open_gguf, write_gguf)
from rotquant.hadamard import f32_to_bf16_bits, rotate_last_axis, signs  # noqa: E402
from rotquant.hadamard import BLOCK_SIZE  # noqa: E402
from rotquant.plan import (PlanError, build_plan, copied_recipe, dry_run_table,  # noqa: E402
                           load_groups, quantize_command, sign_key, tensor_type_lines)
from rotquant.recipe import RecipeError, load_recipe, parse_rotate_selection  # noqa: E402
from rotquant.realign import realign  # noqa: E402
from rotquant.verify import verify_output  # noqa: E402

CHUNK_BYTES = 64 << 20
QUANTIZE_LAYOUT = 32  # llama-quantize's data alignment, whatever general.alignment says


def file_sha256(paths) -> str:
    """SHA-256 of the input bytes, shards concatenated in order."""
    digest = hashlib.sha256()
    for path in paths:
        with Path(path).open("rb") as source:
            while block := source.read(CHUNK_BYTES):
                digest.update(block)
    return digest.hexdigest()


def _copy_writer(model, tensor):
    def write(out):
        raw = model.tensor_bytes(tensor)
        for start in range(0, len(raw), CHUNK_BYTES):
            out.write(raw[start:start + CHUNK_BYTES])
    return write


def _rotate_writer(model, tensor, sign):
    def write(out):
        k = tensor.ne[0]
        rows = tensor.n_elements // k
        step = max(1, CHUNK_BYTES // (k * 4))
        for start in range(0, rows, step):
            block = as_float32(model, tensor, slice(start, min(rows, start + step)))
            rotated = rotate_last_axis(block, sign)
            if tensor.type_name == "BF16":
                out.write(f32_to_bf16_bits(rotated).tobytes())
            elif tensor.type_name == "F16":
                if not np.isfinite(rotated).all() or np.abs(rotated).max() >= 65504:
                    raise PlanError(f"{tensor.name}: rotated values exceed F16")
                out.write(rotated.astype("<f2").tobytes())
            else:
                out.write(np.ascontiguousarray(rotated, dtype="<f4").tobytes())
    return write


def rotate_file(model, plan, output, source_sha256: str) -> None:
    """Write the rotated float GGUF for a plan whose input is not rotated."""
    sign_by_width = {}
    folded = []
    tensors = []
    by_name = model.by_name
    for entry in plan.entries:
        tensor = by_name[entry.name]
        if not entry.rotate:
            tensors.append(OutTensor(entry.name, tensor.ne, tensor.type_name, _copy_writer(model, tensor)))
            continue
        k = tensor.ne[0]
        if not prism.FOLDABLE.fullmatch(entry.name):
            raise PlanError(f"{entry.name}: upstream prism.hadamard cannot fold this tensor kind")
        sign = sign_by_width.setdefault(k, signs(plan.seed, sign_key(k), k))
        folded.append(entry.name)
        tensors.append(OutTensor(entry.name, tensor.ne, tensor.type_name, _rotate_writer(model, tensor, sign)))
    kv_raw = [(key, raw) for key, raw in model.kv_raw if key not in SPLIT_KEYS]
    if folded:
        kv_raw += prism.prism_kv(BLOCK_SIZE, folded, sign_by_width)
    added = provenance(plan, with_types=False, source_sha256=source_sha256,
                       source_file=Path(model.paths[0]).name)
    kv_raw += [(key, encode_value(value)) for key, value in added.items()]
    write_gguf(output, kv_raw, tensors)


def provenance(plan, *, with_types: bool, source_sha256=None, source_file=None) -> dict:
    values = {"gufo.rotation.source_converter": CONVERTER_VERSION,
              "gufo.rotation.source_recipe_sha256": plan.recipe_sha256(with_types)}
    if source_sha256 is not None:
        values["gufo.rotation.source_sha256"] = source_sha256
        values["gufo.rotation.source_file"] = source_file
    return values


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", required=True, help="model definition: tools/<name>/rotquant_groups.py")
    parser.add_argument("--input", required=True, type=Path, help="BF16 GGUF (any shard of a split set) "
                        "or a rotated GGUF from --rotate-only")
    parser.add_argument("--output", type=Path, help="output GGUF (not overwritten)")
    parser.add_argument("--recipe", type=Path, help="JSON recipe: rotate groups and type rules")
    parser.add_argument("--rotate-groups", help="all, none, or comma-separated groups; overrides the recipe")
    parser.add_argument("--seed", type=int, help="sign seed for newly rotated groups (stored in the file)")
    parser.add_argument("--imatrix", type=Path, help="importance matrix for llama-quantize")
    parser.add_argument("--rotate-only", action="store_true", help="write the rotated float GGUF only")
    parser.add_argument("--copy-recipe-from", type=Path, help="take per-tensor types from this GGUF's header")
    parser.add_argument("--dry-run", action="store_true", help="print the per-tensor plan and exit")
    parser.add_argument("--llama-quantize", type=Path, help="reference llama-quantize executable")
    return parser.parse_args(argv)


def prepare(args):
    """Validate options and build the plan; returns (model, plan)."""
    groups = load_groups(args.groups)
    allowed = groups.recipe_types()
    recipe = load_recipe(args.recipe, groups.names, allowed) if args.recipe else None
    if args.rotate_groups is not None:
        rotate = parse_rotate_selection(args.rotate_groups, groups.names)
    elif recipe is not None and recipe.rotate is not None:
        rotate = recipe.rotate
    else:
        raise PlanError("choose the rotated groups with --rotate-groups or the recipe's rotate key")
    if args.rotate_only:
        for flag, value in (("--imatrix", args.imatrix), ("--copy-recipe-from", args.copy_recipe_from),
                            ("--llama-quantize", args.llama_quantize)):
            if value is not None:
                raise PlanError(f"{flag} does not apply to --rotate-only")
    copied = None
    if args.copy_recipe_from is not None:
        if recipe is not None and recipe.rules:
            raise PlanError("use either recipe type rules or --copy-recipe-from, not both")
        copied = copied_recipe(args.copy_recipe_from, groups)
    if not args.dry_run:
        if args.output is None:
            raise PlanError("--output is required")
        if args.output.suffix != ".gguf":
            raise PlanError("--output must end in .gguf")
        if not args.rotate_only and args.llama_quantize is None:
            raise PlanError("quantizing requires --llama-quantize (or use --rotate-only)")
        if args.imatrix is not None and not args.imatrix.is_file():
            raise PlanError(f"--imatrix {args.imatrix} does not exist")
    model = open_gguf(args.input)
    try:
        plan = build_plan(model, groups, rotate, seed=args.seed, rotate_only=args.rotate_only,
                          rules=recipe.rules if recipe else (), copied_types=copied)
    except Exception:
        model.close()
        raise
    return model, plan


def sidecars(output: Path) -> dict:
    stem = output.with_suffix("")
    return {"rotated": Path(f"{stem}.rotated.gguf"),
            "tensor_types": Path(f"{stem}.tensor-types.txt"),
            "recipe": Path(f"{stem}.recipe.json"),
            "unaligned": Path(f"{stem}.unaligned.gguf")}


def run(args) -> int:
    model, plan = prepare(args)
    with model:
        if args.dry_run:
            print(dry_run_table(plan))
            return 0
        paths = sidecars(args.output)
        targets = [args.output] + ([] if args.rotate_only else list(paths.values()))
        existing = [str(p) for p in targets if p.exists()]
        if existing:
            raise PlanError(f"refusing to overwrite {', '.join(existing)}")
        if args.rotate_only:
            rotate_file(model, plan, args.output, file_sha256(model.paths))
            print(f"wrote {args.output}")
            return 0
        if plan.input_rotated:
            source = Path(model.paths[0])
            overrides = provenance(plan, with_types=True)
        elif plan.rotate:
            source = paths["rotated"]
            rotate_file(model, plan, source, file_sha256(model.paths))
            print(f"wrote {source}")
            overrides = provenance(plan, with_types=True)
        else:
            source = Path(model.paths[0])
            overrides = provenance(plan, with_types=True, source_sha256=file_sha256(model.paths),
                                   source_file=source.name)
    paths["tensor_types"].write_text("\n".join(tensor_type_lines(plan)) + "\n", encoding="utf-8")
    paths["recipe"].write_text(plan.recipe_json(with_types=True) + "\n", encoding="utf-8")
    command = quantize_command(plan, args.llama_quantize, source, args.output, paths["tensor_types"],
                               imatrix=args.imatrix, provenance=overrides)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    # llama-quantize copies general.alignment from its input but always lays
    # the data out at 32 bytes (llama.cpp src/llama-quant.cpp `const size_t
    # align = GGUF_DEFAULT_ALIGNMENT`, b11069:979, 8ab29fae:1059). A 256 key
    # over 32-byte packing is inconsistent (llama.cpp's reader refuses the
    # offsets), so the output is read at 32 and rewritten at 256; tensor
    # bytes do not change.
    with open_gguf(args.output, layout_alignment=QUANTIZE_LAYOUT) as written:
        if not layout_is_packed(written, QUANTIZE_LAYOUT):
            raise PlanError(f"{args.output}: llama-quantize output is not packed at {QUANTIZE_LAYOUT} bytes")
        base = written.shards[0].data_offset
        aligned = (written.kv.get("general.alignment") == WRITE_ALIGNMENT and base % WRITE_ALIGNMENT == 0
                   and all(t.offset % WRITE_ALIGNMENT == 0 for t in written.tensors))
    if not aligned:
        args.output.rename(paths["unaligned"])
        result = realign(paths["unaligned"], args.output, layout_alignment=QUANTIZE_LAYOUT)
        paths["unaligned"].unlink()
        print(f"realigned {args.output}: alignment {result['input_alignment']} -> {result['alignment']}, "
              f"tensor bytes unchanged", flush=True)
    summary = verify_output(source, args.output, plan, overrides)
    print(f"verified {summary['output']}: {summary['tensors']} tensors, {summary['quantized']} quantized, "
          f"{summary['byte_identical']} byte-identical, {summary['folded']} folded, types match the plan, "
          f"prism/gufo metadata kept, {WRITE_ALIGNMENT}-byte aligned", flush=True)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (PlanError, RecipeError, ValueError) as error:
        print(f"rotquant: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
