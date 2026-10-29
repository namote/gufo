"""Check a llama-quantize output against the plan that produced it.

llama-quantize can change a tensor type on its own (name heuristics, the
`tensor_type_fallback` for rows a block does not tile) and only warns. The
converter therefore reads the output back and refuses it unless:

- it holds exactly the source tensors, `.rot` sign tensors included;
- every tensor has the planned type (`.tensor-types.txt` for matrices, the
  source type for everything llama-quantize leaves alone);
- every tensor whose type did not change, `.rot` included, is byte-identical
  to the source;
- every `gufo.*` key of the source is still present with the same bytes, and
  each `--override-kv` value reads back exactly.
"""
from __future__ import annotations

from pathlib import Path

from rotquant.gguf_io import encode_value, open_gguf
from rotquant.plan import ROT_SUFFIX, ROT_TYPE, PlanError


def verify_output(source, output, plan, overrides: dict) -> dict:
    """Raise PlanError on the first class of mismatch; return a summary."""
    expected = {e.name: e.target_type for e in plan.entries}
    with open_gguf(source) as src, open_gguf(output) as out:
        src_names = [t.name for t in src.tensors]
        out_by_name = out.by_name
        missing = sorted(set(src_names) - set(out_by_name))
        extra = sorted(set(out_by_name) - set(src_names))
        if missing or extra:
            raise PlanError(f"{output}: tensor set differs from the source; "
                            f"missing {missing[:5]}, extra {extra[:5]}")
        wrong = []
        changed = []
        copied = 0
        for tensor in src.tensors:
            got = out_by_name[tensor.name]
            if tuple(got.ne) != tuple(tensor.ne):
                wrong.append(f"{tensor.name} shape {list(got.ne)} != {list(tensor.ne)}")
                continue
            if tensor.name.endswith(ROT_SUFFIX):
                want = ROT_TYPE
            else:
                want = expected.get(tensor.name)
                if want is None:
                    raise PlanError(f"{tensor.name} is not in the plan")
            if got.type_name != want:
                wrong.append(f"{tensor.name} {got.type_name} != {want}")
                continue
            if got.type_name == tensor.type_name:
                if out.tensor_bytes(got) != src.tensor_bytes(tensor):
                    changed.append(tensor.name)
                copied += 1
        if wrong:
            raise PlanError(f"{output}: {len(wrong)} tensor(s) do not match the plan: "
                            + "; ".join(wrong[:8]))
        if changed:
            raise PlanError(f"{output}: {len(changed)} unquantized tensor(s) changed bytes: "
                            + ", ".join(changed[:8]))
        out_kv = dict(out.kv_raw)
        kv_errors = []
        for key, raw in src.kv_raw:
            if key.startswith("gufo.") and key not in overrides and out_kv.get(key) != raw:
                kv_errors.append(f"{key} changed or missing")
        for key, value in overrides.items():
            if out_kv.get(key) != encode_value(value):
                kv_errors.append(f"{key} != {value!r}")
        if kv_errors:
            raise PlanError(f"{output}: metadata check failed: " + "; ".join(kv_errors))
        rot = sum(name.endswith(ROT_SUFFIX) for name in src_names)
    return {"tensors": len(src_names), "rot": rot, "byte_identical": copied,
            "quantized": len(src_names) - copied, "output": str(Path(output))}
