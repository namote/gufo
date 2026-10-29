"""Per-tensor conversion plan, rotation metadata checks, dry-run table and
the llama-quantize command line."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import re


from rotquant import CONVERTER_VERSION
from rotquant.gguf_io import GgufFile, type_size
from rotquant import prism
from rotquant.hadamard import BLOCK_SIZE

TOOLS = Path(__file__).resolve().parents[1]
FLOAT_TYPES = ("F32", "F16", "BF16")
KV_PREFIX = "gufo.rotation."
SOURCE_KEYS = ("gufo.rotation.source_converter", "gufo.rotation.source_recipe_sha256",
               "gufo.rotation.source_sha256", "gufo.rotation.source_file", "gufo.rotation.source_signs")
# Sign vectors D: "random" draws ±1 per input width from --seed; "none" uses
# D = I (Hadamard only). Both are written as explicit prism.hadamard signs.
SIGN_MODES = ("random", "none")
SIGNS_KEY = "gufo.rotation.source_signs"


def signs_label(sign_mode: str, seed) -> str:
    """Value of gufo.rotation.source_signs: "random seed=<n>" or "none"."""
    return "none" if sign_mode == "none" else f"random seed={seed}"


def sign_key(k: int) -> str:
    """Seed derivation key of the sign vector for input width k."""
    return f"width:{k}"


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
    rotated_layers: object    # callable(kv) -> blocks [0, n) that rotate
    layer_limit: int | None = None  # set by for_model

    def for_model(self, kv: dict) -> "Groups":
        """Bind the rotated block range read from a model header."""
        try:
            limit = int(self.rotated_layers(kv))
        except (KeyError, TypeError, ValueError) as error:
            raise PlanError(f"cannot read the rotated block range from the header: {error!r}") from None
        return replace(self, layer_limit=limit)

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
                if layer is not None and self.layer_limit is not None and int(layer) >= self.layer_limit:
                    return None, None
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
                      tuple(module.RECIPE_ROLES), module.role, module.rotated_layers)
    except AttributeError as error:
        raise PlanError(f"{path} is incomplete: {error}") from None


# --------------------------------------------------------------------------
# Existing rotation metadata (shared by the reuse path and the tests)
# --------------------------------------------------------------------------
def check_rotation(model: GgufFile, groups: Groups) -> tuple:
    """Validate prism.hadamard metadata; return the rotated groups.

    First the upstream llama.cpp loader rules (rotquant/prism.py), then the
    subset this converter writes and gufo executes: version 1, untied, block
    128, explicit signs, no inverse tables or GDN permutation, every folded
    weight in a group of the model definition, and each group folded in all
    of its trunk instances or in none (a partially folded activation would
    need both bases at run time).
    """
    groups = groups.for_model(model.kv)
    try:
        info = prism.read_prism(model)
    except prism.PrismError as error:
        raise PlanError(str(error)) from None
    if info is None:
        stray = sorted(key for key in model.kv if key.startswith(prism.PREFIX))
        if stray:
            raise PlanError(f"{', '.join(stray)} without {prism.VERSION}")
        return ()
    if info.version != 1 or info.tied_output:
        raise PlanError("only untied prism.hadamard version 1 files are supported")
    if info.block_size != BLOCK_SIZE:
        raise PlanError(f"unsupported {prism.BLOCK_SIZE} = {info.block_size} (expected {BLOCK_SIZE})")
    if info.sign_mode != "explicit":
        raise PlanError(f"unsupported {prism.SIGN_MODE} = {info.sign_mode!r} (expected 'explicit')")
    if info.inverse_weight_names:
        raise PlanError(f"{prism.INVERSE_WEIGHT_NAMES} is not supported")
    if info.gdn_v_grouped:
        raise PlanError(f"{prism.GDN_V_GROUPED} is not supported: rotation is defined in the GGUF "
                        "(tiled) value-head order")
    folded = set(info.weight_names)
    for name in info.weight_names:
        if groups.instance(name)[0] is None:
            raise PlanError(f"{name} is rotated but its role has no rotation path")
    instances = {}
    for tensor in model.tensors:
        group, key = groups.instance(tensor.name)
        if group is not None:
            instances.setdefault(key, (group, []))[1].append(tensor.name)
    rotated_instances = {}
    for key, (group, members) in instances.items():
        flags = [name in folded for name in members]
        if any(flags) and not all(flags):
            raise PlanError(f"group instance {key} mixes rotated and unrotated tensors")
        if all(flags):
            rotated_instances.setdefault(group, set()).add(key)
    rotated = []
    for group in groups.names:
        all_keys = {k for k, (g, _) in instances.items() if g == group}
        done = rotated_instances.get(group, set())
        if done and done != all_keys:
            raise PlanError(f"group {group} is rotated in some layers only")
        if done:
            rotated.append(group)
    unused = sorted(set(info.signs) - {model.by_name[n].ne[0] for n in folded})
    if unused:
        raise PlanError(f"{prism.SIGN_WIDTHS} lists width(s) no folded weight uses: {unused}")
    return tuple(rotated)


def file_signs(model: GgufFile) -> dict:
    """Width -> sign vector of a rotated file (empty when unrotated)."""
    info = prism.read_prism(model)
    return {} if info is None else dict(info.signs)


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
    entries: list             # tensors, input order
    sign_mode: str = "random"  # "random" (seeded) or "none" (D = I)

    def effective(self, with_types: bool) -> dict:
        data = {"converter": CONVERTER_VERSION, "model": self.groups.name,
                "rotate": list(self.rotate), "seed": self.seed}
        if self.sign_mode != "random":
            data["signs"] = self.sign_mode
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
               rules=(), copied_types: dict | None = None, sign_mode: str | None = None) -> Plan:
    arch = model.kv.get("general.architecture")
    if arch != groups.architecture:
        raise PlanError(f"input architecture {arch!r} does not match the {groups.name} "
                        f"definition ({groups.architecture!r})")
    groups = groups.for_model(model.kv)
    existing = check_rotation(model, groups)
    input_rotated = bool(existing)
    if input_rotated:
        if tuple(rotate) != existing:
            raise PlanError(f"input is already rotated for groups {', '.join(existing)}; "
                            f"requested {', '.join(rotate) or 'none'}")
        if seed is not None:
            raise PlanError("--seed cannot change an already rotated input")
        if sign_mode is not None:
            raise PlanError("--signs cannot change an already rotated input")
        if rotate_only:
            raise PlanError("input is already rotated; --rotate-only has nothing to do")
    elif sign_mode not in (None, *SIGN_MODES):
        raise PlanError(f"--signs must be one of {', '.join(SIGN_MODES)}")
    elif sign_mode == "none":
        if seed is not None:
            raise PlanError("--seed does not apply to --signs none")
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
    return Plan(groups, tuple(rotate), input_rotated, seed, rotate_only, entries, sign_mode or "random")


def copied_recipe(path, groups: Groups) -> dict:
    """Tensor types from another GGUF's header (data is not read)."""
    from rotquant.gguf_io import open_gguf
    other = open_gguf(path, header_only=True)
    allowed = groups.recipe_types()
    types = {}
    for tensor in other.tensors:
        if groups.role(tensor.name) not in groups.recipe_roles:
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
    widths = sorted({e.ne[0] for e in rot_entries})
    lines.append("-" * len(header))
    lines.append(f"tensors: {len(plan.entries)}  rotated: {len(rot_entries)}  "
                 f"rotated groups: {', '.join(plan.rotate) or 'none'}")
    lines.append(f"weights: {total_bytes / 2**30:.3f} GiB, {total_bytes * 8 / max(total_elements, 1):.4f} bpw")
    lines.append(f"prism.hadamard sign vectors: {len(widths)} (widths {', '.join(map(str, widths)) or '-'}; "
                 f"{sum(widths)} int32 values)")
    lines.append(f"expected data size: {total_bytes / 2**30:.3f} GiB (+ header and padding)")
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
