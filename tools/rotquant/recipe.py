"""Recipe files: per-tensor quantization types and per-group rotation.

```yaml
version: 1
rotate: [attn_in, ffn_in]        # group names, or all / none
types:                           # first matching rule wins (Python re.search)
  - match: '^blk\\.\\d+\\.ssm_out\\.weight$'
    type: Q6_K
  - match: '\\.weight$'
    type: Q4_K
```

Rules apply only to tensors whose model role takes a recipe type
(embeddings and projections). Every such tensor must match a rule unless the
run is rotate-only; there is no implicit default type.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import yaml

RECIPE_VERSION = 1
RECIPE_KEYS = {"version", "rotate", "types"}
RULE_KEYS = {"match", "type"}


class RecipeError(ValueError):
    pass


@dataclass(frozen=True)
class Rule:
    match: str
    type_name: str

    def matches(self, name: str) -> bool:
        return re.search(self.match, name) is not None


@dataclass(frozen=True)
class Recipe:
    rotate: tuple | None          # group names; None when the recipe does not choose
    rules: tuple                  # Rule, in priority order


def parse_rotate_selection(value, known_groups) -> tuple:
    """Parse `all`, `none`, a comma list, or a YAML list into group names."""
    if isinstance(value, str):
        if value == "all":
            return tuple(known_groups)
        if value == "none":
            return ()
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        items = value
    else:
        raise RecipeError(f"rotate must be all, none, or a list of groups, got {value!r}")
    if not items or any(not isinstance(item, str) or not item for item in items):
        raise RecipeError(f"rotate list is empty or malformed: {value!r}")
    if len(set(items)) != len(items):
        raise RecipeError(f"rotate list repeats a group: {value!r}")
    unknown = [item for item in items if item not in known_groups]
    if unknown:
        raise RecipeError(f"unknown input group(s) {', '.join(unknown)}; "
                          f"known: {', '.join(known_groups)}")
    # Keep the model definition's order so hashes do not depend on spelling.
    return tuple(group for group in known_groups if group in items)


def parse_recipe(text: str, known_groups, allowed_types) -> Recipe:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise RecipeError(f"recipe is not valid YAML: {error}") from None
    if not isinstance(data, dict):
        raise RecipeError("recipe must be a mapping")
    unknown = set(data) - RECIPE_KEYS
    if unknown:
        raise RecipeError(f"unknown recipe key(s): {', '.join(sorted(map(str, unknown)))}")
    if data.get("version") != RECIPE_VERSION:
        raise RecipeError(f"recipe version must be {RECIPE_VERSION}, got {data.get('version')!r}")
    rotate = None
    if "rotate" in data:
        rotate = parse_rotate_selection(data["rotate"], known_groups)
    rules = []
    types = data.get("types", [])
    if not isinstance(types, list):
        raise RecipeError("types must be a list of {match, type} rules")
    for index, rule in enumerate(types):
        if not isinstance(rule, dict) or set(rule) != RULE_KEYS:
            raise RecipeError(f"types[{index}] must have exactly the keys match and type")
        pattern, type_name = rule["match"], rule["type"]
        if not isinstance(pattern, str) or not isinstance(type_name, str):
            raise RecipeError(f"types[{index}]: match and type must be strings")
        try:
            re.compile(pattern)
        except re.error as error:
            raise RecipeError(f"types[{index}]: invalid pattern {pattern!r}: {error}") from None
        if type_name not in allowed_types:
            raise RecipeError(f"types[{index}]: type {type_name} is not supported by the model loader; "
                              f"allowed: {', '.join(sorted(allowed_types))}")
        rules.append(Rule(pattern, type_name))
    return Recipe(rotate, tuple(rules))


def load_recipe(path, known_groups, allowed_types) -> Recipe:
    return parse_recipe(Path(path).read_text(encoding="utf-8"), known_groups, allowed_types)
