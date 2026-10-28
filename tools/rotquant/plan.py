"""Per-tensor conversion plan, rotation metadata checks, dry-run table and
the llama-quantize command line."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np

from rotquant import CONVERTER_VERSION
from rotquant.gguf_io import GgufFile, type_size
from rotquant.hadamard import BLOCK_SIZE

TOOLS = Path(__file__).resolve().parents[1]
FLOAT_TYPES = ("F32", "F16", "BF16")
ROT_SUFFIX = ".rot"
ROT_TYPE = "F16"
F16_PLUS_ONE, F16_MINUS_ONE = 0x3C00, 0xBC00
KV_PREFIX = "gufo.rotation."
# The marker keys; their presence means "this file has rotated tensors".
ROTATION_KV = {
    "gufo.rotation.version": 1,
    "gufo.rotation.kind": "hadamard",
    "gufo.rotation.block_size": BLOCK_SIZE,
    "gufo.rotation.side": "input",
}
SOURCE_KEYS = ("gufo.rotation.source_converter", "gufo.rotation.source_recipe_sha256",
               "gufo.rotation.source_sha256", "gufo.rotation.source_file")


class PlanError(ValueError):
    pass


# --------------------------------------------------------------------------
# Model definitions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Groups:
    name: str
    architecture: str
    patterns: dict            # group -> compiled full-match pattern
    tied_heads: dict
    supported: dict           # role -> set of type names
    recipe_roles: tuple
    role: object              # callable(name) -> role or None

    @property
    def names(self) -> tuple:
        return tuple(self.patterns)

    def recipe_types(self) -> set:
        types = set()
        for role in self.recipe_roles:
            types |= self.supported[role]
        return types

    def instance(self, tensor_name: str):
        """(group, instance key) for a tensor, or (None, None)."""
        for group, pattern in self.patterns.items():
            match = pattern.fullmatch(tensor_name)
            if match:
                layer = match.groupdict().get("layer")
                return group, group if layer is None else f"{group}:{layer}"
        return None, None


def load_groups(name: str) -> Groups:
    """Load tools/<name>/rotquant_groups.py; a model without one is refused."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name or ""):
        raise PlanError(f"invalid model definition name {name!r}")
    path = TOOLS / name / "rotquant_groups.py"
    if not path.is_file():
        raise PlanError(f"no rotation group definition for model {name!r} ({path} does not exist)")
    spec = importlib.util.spec_from_file_location(f"rotquant_groups_{name.replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return Groups(name, module.ARCHITECTURE,
                      {g: re.compile(p) for g, p in module.GROUPS.items()},
                      dict(module.TIED_HEADS), dict(module.SUPPORTED_TYPES),
                      tuple(module.RECIPE_ROLES), module.role)
    except AttributeError as error:
        raise PlanError(f"{path} is incomplete: {error}") from None


# --------------------------------------------------------------------------
# Existing rotation metadata (shared by the reuse path and the tests)
# --------------------------------------------------------------------------
def read_rot_signs(model: GgufFile, tensor) -> np.ndarray:
    if tensor.type_name != ROT_TYPE or len(tensor.ne) != 1:
        raise PlanError(f"{tensor.name} must be a one-dimensional {ROT_TYPE} tensor, "
                        f"got {tensor.type_name} {list(tensor.ne)}")
    bits = np.frombuffer(model.tensor_bytes(tensor), dtype="<u2")
    if not np.isin(bits, (F16_PLUS_ONE, F16_MINUS_ONE)).all():
        raise PlanError(f"{tensor.name} holds values other than +1 and -1")
    return np.where(bits == F16_PLUS_ONE, 1.0, -1.0).astype(np.float32)


def check_rotation(model: GgufFile, groups: Groups) -> tuple:
    """Validate rotation metadata like the loader will; return rotated groups.

    Refuses: unknown marker values, `.rot` without markers or markers without
    `.rot`, a `.rot` on a tensor outside every group (this covers a tied
    head: the embedding it falls back to belongs to no group), mixed rotation
    or different signs inside one group instance, and a group rotated in
    some layers only.
    """
    by_name = model.by_name
    rot = {t.name[:-len(ROT_SUFFIX)]: t for t in model.tensors if t.name.endswith(ROT_SUFFIX)}
    markers = {key: model.kv[key] for key in ROTATION_KV if key in model.kv}
    if rot and not markers:
        raise PlanError("file has .rot tensors but no gufo.rotation metadata")
    if markers and not rot:
        raise PlanError("file has gufo.rotation metadata but no .rot tensors")
    if not rot:
        return ()
    for key, expected in ROTATION_KV.items():
        if key not in markers:
            raise PlanError(f"missing {key}")
        if markers[key] != expected:
            raise PlanError(f"unsupported {key} = {markers[key]!r} (expected {expected!r})")
    instances = {}
    for tensor in model.tensors:
        if tensor.name.endswith(ROT_SUFFIX):
            continue
        group, key = groups.instance(tensor.name)
        if group is not None:
            instances.setdefault(key, (group, []))[1].append(tensor)
    for base in rot:
        if base not in by_name:
            raise PlanError(f"{base}{ROT_SUFFIX} has no matching tensor {base}")
        if groups.instance(base)[0] is None:
            raise PlanError(f"{base} is rotated but its role has no rotation path")
    rotated_instances = {}
    for key, (group, members) in instances.items():
        flags = [m.name in rot for m in members]
        if any(flags) and not all(flags):
            raise PlanError(f"group instance {key} mixes rotated and unrotated tensors")
        if not any(flags):
            continue
        reference = None
        for member in members:
            sign = read_rot_signs(model, rot[member.name])
            if sign.shape[0] != member.ne[0]:
                raise PlanError(f"{member.name}{ROT_SUFFIX} has {sign.shape[0]} values, tensor K is {member.ne[0]}")
            if reference is None:
                reference = sign
            elif not np.array_equal(sign, reference):
                raise PlanError(f"group instance {key}: {member.name} uses a different sign vector")
        rotated_instances.setdefault(group, set()).add(key)
    rotated = []
    for group in groups.names:
        all_keys = {k for k, (g, _) in instances.items() if g == group}
        done = rotated_instances.get(group, set())
        if done and done != all_keys:
            raise PlanError(f"group {group} is rotated in some layers only")
        if done:
            rotated.append(group)
    return tuple(rotated)


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------
@dataclass
class Entry:
    name: str
    ne: tuple
    role: str | None
    group: str | None
    instance: str | None
    source_type: str
    rotate: bool              # rotate in this run
    rotated: bool             # rotated in the output (this run or earlier)
    rotated_type: str         # type in the rotated (pre-quantize) file
    target_type: str          # type after quantize (rotated_type when rotate-only)

    @property
    def n_elements(self) -> int:
        count = 1
        for dim in self.ne:
            count *= dim
        return count

    def nbytes(self, type_name: str) -> int:
        return type_size(type_name, self.n_elements, self.ne[0])


@dataclass
class Plan:
    groups: Groups
    rotate: tuple             # groups rotated in the output
    input_rotated: bool       # the input already carries the rotation
    seed: int | None
    rotate_only: bool
    entries: list             # weight tensors (no .rot), input order

    def effective(self, with_types: bool) -> dict:
        data = {"converter": CONVERTER_VERSION, "model": self.groups.name,
                "rotate": list(self.rotate), "seed": self.seed}
        if with_types:
            data["types"] = {e.name: e.target_type for e in self.entries}
        return data

    def recipe_json(self, with_types: bool) -> str:
        return json.dumps(self.effective(with_types), sort_keys=True, separators=(",", ":"))

    def recipe_sha256(self, with_types: bool) -> str:
        return hashlib.sha256(self.recipe_json(with_types).encode()).hexdigest()

    @property
    def rot_entries(self) -> list:
        return [e for e in self.entries if e.rotated]


def build_plan(model: GgufFile, groups: Groups, rotate: tuple, *, seed, rotate_only: bool,
               rules=(), copied_types: dict | None = None) -> Plan:
    arch = model.kv.get("general.architecture")
    if arch != groups.architecture:
        raise PlanError(f"input architecture {arch!r} does not match the {groups.name} "
                        f"definition ({groups.architecture!r})")
    existing = check_rotation(model, groups)
    input_rotated = bool(existing) or any(key in model.kv for key in ROTATION_KV)
    if input_rotated:
        if tuple(rotate) != existing:
            raise PlanError(f"input is already rotated for groups {', '.join(existing)}; "
                            f"requested {', '.join(rotate) or 'none'}")
        if seed is not None:
            raise PlanError("--seed cannot change an already rotated input")
        if rotate_only:
            raise PlanError("input is already rotated; --rotate-only has nothing to do")
    elif rotate and seed is None:
        raise PlanError("rotation requires --seed")
    if any(key.startswith(KV_PREFIX) for key in model.kv) and not input_rotated:
        raise PlanError("input has gufo.rotation provenance keys but no rotation; "
                        "convert from the original file")

    present = {groups.instance(t.name)[0] for t in model.tensors}
    for group in rotate:
        if group not in present:
            if group in groups.tied_heads:
                raise PlanError(f"group {group} has no tensor: the head is tied to "
                                f"{groups.tied_heads[group]} and cannot be rotated")
            raise PlanError(f"group {group} has no tensor in the input")

    entries = []
    used_rules = set()
    missing = []
    for tensor in model.tensors:
        if tensor.name.endswith(ROT_SUFFIX):
            continue
        role = groups.role(tensor.name)
        group, instance = groups.instance(tensor.name)
        rotated = group in rotate
        do_rotate = rotated and not input_rotated
        if do_rotate:
            if tensor.type_name not in FLOAT_TYPES:
                raise PlanError(f"{tensor.name}: cannot rotate a {tensor.type_name} tensor; "
                                f"the input must be F32, F16 or BF16")
            if tensor.ne[0] % BLOCK_SIZE:
                raise PlanError(f"{tensor.name}: K = {tensor.ne[0]} is not a multiple of {BLOCK_SIZE}")
        target = tensor.type_name
        if not rotate_only and role in groups.recipe_roles:
            if copied_types is not None:
                target = copied_types.get(tensor.name)
                if target is None:
                    missing.append(tensor.name)
                    continue
            else:
                for index, rule in enumerate(rules):
                    if rule.matches(tensor.name):
                        target = rule.type_name
                        used_rules.add(index)
                        break
                else:
                    missing.append(tensor.name)
                    continue
        entries.append(Entry(tensor.name, tensor.ne, role, group, instance, tensor.type_name,
                             do_rotate, rotated, tensor.type_name, target))
    if missing:
        head = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        source = "--copy-recipe-from file" if copied_types is not None else "recipe"
        raise PlanError(f"{len(missing)} tensor(s) have no type in the {source}: {head}")
    if not rotate_only and copied_types is None:
        unused = [rules[i].match for i in range(len(rules)) if i not in used_rules]
        if unused:
            raise PlanError(f"recipe rule(s) match no tensor: {', '.join(unused)}")
    for entry in entries:
        if entry.role is None:
            continue
        allowed = groups.supported[entry.role]
        for label, type_name in (("rotated", entry.rotated_type), ("target", entry.target_type)):
            if type_name not in allowed:
                raise PlanError(f"{entry.name}: {label} type {type_name} is not supported for the "
                                f"{entry.role} role; allowed: {', '.join(sorted(allowed))}")
        try:
            entry.nbytes(entry.target_type)
        except ValueError as error:
            raise PlanError(f"{entry.name}: {error}") from None
    return Plan(groups, tuple(rotate), input_rotated, seed, rotate_only, entries)


def copied_recipe(path, groups: Groups) -> dict:
    """Tensor types from another GGUF's header (data is not read)."""
    from rotquant.gguf_io import open_gguf
    other = open_gguf(path, header_only=True)
    allowed = groups.recipe_types()
    types = {}
    for tensor in other.tensors:
        if tensor.name.endswith(ROT_SUFFIX) or groups.role(tensor.name) not in groups.recipe_roles:
            continue
        if tensor.type_name not in allowed:
            raise PlanError(f"--copy-recipe-from: {tensor.name} is {tensor.type_name}, "
                            f"which the converter cannot target")
        types[tensor.name] = tensor.type_name
    return types


# --------------------------------------------------------------------------
# Dry-run table
# --------------------------------------------------------------------------
def _shape(ne) -> str:
    return "x".join(str(d) for d in reversed(ne))


def dry_run_table(plan: Plan) -> str:
    lines = []
    header = f"{'tensor':<34} {'shape':>13} {'role':<13} {'group':<15} {'rot':<4} {'source':<7} {'target':<7} {'MiB':>10} {'bpw':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    total_bytes = total_elements = 0
    for e in plan.entries:
        nbytes = e.nbytes(e.target_type)
        total_bytes += nbytes
        total_elements += e.n_elements
        rot = "yes" if e.rotated else "no"
        if e.rotated and not e.rotate:
            rot = "kept"
        lines.append(f"{e.name:<34} {_shape(e.ne):>13} {e.role or '-':<13} {e.instance or '-':<15} "
                     f"{rot:<4} {e.source_type:<7} {e.target_type:<7} {nbytes / 2**20:>10.2f} "
                     f"{nbytes * 8 / e.n_elements:>6.3f}")
    rot_entries = plan.rot_entries
    rot_bytes = sum(type_size(ROT_TYPE, e.ne[0], e.ne[0]) for e in rot_entries)
    lines.append("-" * len(header))
    lines.append(f"tensors: {len(plan.entries)}  rotated: {len(rot_entries)}  "
                 f"rotated groups: {', '.join(plan.rotate) or 'none'}")
    lines.append(f"weights: {total_bytes / 2**30:.3f} GiB, {total_bytes * 8 / max(total_elements, 1):.4f} bpw")
    lines.append(f".rot sign tensors ({ROT_TYPE}): {len(rot_entries)}, {rot_bytes / 2**20:.3f} MiB")
    lines.append(f"expected data size: {(total_bytes + rot_bytes) / 2**30:.3f} GiB (+ header and padding)")
    if not plan.rotate_only:
        lines.append(f"llama-quantize base type: {base_type(plan)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# llama-quantize
# --------------------------------------------------------------------------
def base_type(plan: Plan) -> str:
    """Positional llama-quantize type: the quantized type with the most bytes.

    Every tensor also gets an exact --tensor-type rule, so this only fills
    general.file_type and keeps llama-quantize's override path enabled (it
    applies overrides only when the base type is quantized).
    """
    totals = {}
    for e in plan.entries:
        if e.target_type not in FLOAT_TYPES:
            totals[e.target_type] = totals.get(e.target_type, 0) + e.nbytes(e.target_type)
    if not totals:
        raise PlanError("the recipe quantizes no tensor; use --rotate-only for a float output")
    return max(sorted(totals), key=lambda t: totals[t])


def tensor_type_lines(plan: Plan) -> list:
    """One anchored rule per matrix. llama-quantize lowercases the pattern,
    so names must already be lowercase."""
    lines = []
    for e in plan.entries:
        if len(e.ne) < 2:
            continue
        pattern = "^" + re.escape(e.name) + "$"
        if pattern != pattern.lower():
            raise PlanError(f"{e.name}: llama-quantize lowercases tensor patterns; name must be lowercase")
        lines.append(f"{pattern}={e.target_type}")
    return lines


def quantize_command(plan: Plan, llama_quantize, rotated_path, output_path, tensor_types_path,
                     imatrix=None, provenance: dict | None = None) -> list:
    command = [str(llama_quantize), "--pure", "--tensor-type-file", str(tensor_types_path)]
    if imatrix is not None:
        command += ["--imatrix", str(imatrix)]
    for key, value in (provenance or {}).items():
        if len(value) > 127 or len(key) > 127:
            raise PlanError(f"--override-kv value for {key} exceeds llama.cpp's 127 characters")
        command += ["--override-kv", f"{key}=str:{value}"]
    command += [str(rotated_path), str(output_path), base_type(plan)]
    return command
