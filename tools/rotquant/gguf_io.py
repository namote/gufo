"""GGUF header access with raw metadata spans, split inputs, and a writer.

Value decoding and type tables come from `tools/gufo/gguf.py`. This module
keeps every metadata entry as raw bytes so the converter can copy the
tokenizer and hyperparameters verbatim, which the shared inspector skips.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import mmap
from pathlib import Path
import re
import struct

import numpy as np

from gufo.gguf import GGML_TYPE_BLOCK, GGML_TYPE_NAME, Reader

TYPE_ID = {name: type_id for type_id, name in GGML_TYPE_NAME.items()}
SPLIT_KEYS = ("split.no", "split.count", "split.tensors.count")
SPLIT_NAME = re.compile(r"^(.*)-([0-9]+)-of-([0-9]+)\.gguf$")
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# GGUF metadata value tags.
T_U32, T_I32, T_F32, T_BOOL, T_STRING, T_ARRAY, T_U64 = 4, 5, 6, 7, 8, 9, 10


def type_size(type_name: str, n_elements: int, row: int) -> int:
    """Bytes for n_elements of type_name, refusing rows the blocks do not tile."""
    block, size = GGML_TYPE_BLOCK[TYPE_ID[type_name]]
    if row % block:
        raise ValueError(f"row of {row} values is not a multiple of the {type_name} block {block}")
    return n_elements // block * size


@dataclass
class TensorInfo:
    name: str
    ne: tuple            # GGUF order: ne[0] is the contiguous (K) dimension
    type_name: str
    offset: int          # relative to the shard data section
    shard: int = 0

    @property
    def n_elements(self) -> int:
        count = 1
        for dim in self.ne:
            count *= dim
        return count

    @property
    def nbytes(self) -> int:
        return type_size(self.type_name, self.n_elements, self.ne[0])

    @property
    def shape(self) -> tuple:
        """NumPy (row-major) shape: reversed GGUF dimensions."""
        return tuple(reversed(self.ne))


@dataclass
class Shard:
    path: Path
    file: object
    data: mmap.mmap
    data_offset: int


@dataclass
class GgufFile:
    """One GGUF model, single-file or split, opened read-only."""
    paths: list
    kv: dict                      # key -> decoded value (shard 0)
    kv_raw: list                  # (key, raw tag+value bytes) in file order (shard 0)
    tensors: list
    alignment: int
    shards: list = field(default_factory=list)

    @property
    def by_name(self) -> dict:
        return {t.name: t for t in self.tensors}

    def tensor_bytes(self, tensor: TensorInfo) -> memoryview:
        shard = self.shards[tensor.shard]
        start = shard.data_offset + tensor.offset
        end = start + tensor.nbytes
        if end > len(shard.data):
            raise ValueError(f"tensor {tensor.name} overruns {shard.path}")
        return memoryview(shard.data)[start:end]

    def close(self):
        for shard in self.shards:
            shard.data.close()
            shard.file.close()
        self.shards = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _parse_header(data, path, keep_raw: bool):
    if data[0:4] != b"GGUF":
        raise ValueError(f"{path} is not a GGUF file")
    r = Reader(data)
    r.pos = 4
    version = r.u32()
    if version not in (2, 3):
        raise ValueError(f"{path}: unsupported GGUF version {version}")
    n_tensors = r.u64()
    n_kv = r.u64()
    kv, kv_raw = {}, []
    for _ in range(n_kv):
        key = r.string()
        start = r.pos
        tag = r.u32()
        if key.startswith("tokenizer."):
            r.skip_value(tag)
        else:
            kv[key] = r.read_value(tag)
        if keep_raw:
            kv_raw.append((key, bytes(data[start:r.pos])))
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.u32()
        ne = tuple(r.u64() for _ in range(n_dims))
        type_id = r.u32()
        offset = r.u64()
        if type_id not in GGML_TYPE_NAME or type_id not in GGML_TYPE_BLOCK:
            raise ValueError(f"{path}: tensor {name} has unsupported type {type_id}")
        tensors.append(TensorInfo(name, ne, GGML_TYPE_NAME[type_id], offset))
    alignment = int(kv.get("general.alignment", DEFAULT_ALIGNMENT))
    data_offset = (r.pos + alignment - 1) // alignment * alignment
    return kv, kv_raw, tensors, alignment, data_offset


def _shard_paths(path: Path, count: int) -> list:
    match = SPLIT_NAME.match(path.name)
    if not match:
        raise ValueError(f"split GGUF name does not match <name>-00001-of-00002.gguf: {path.name}")
    prefix, width_no, width_count = match.group(1), len(match.group(2)), len(match.group(3))
    return [path.parent / f"{prefix}-{i + 1:0{width_no}d}-of-{count:0{width_count}d}.gguf"
            for i in range(count)]


def open_gguf(path, header_only: bool = False) -> GgufFile:
    """Open a GGUF file; a split set is opened from any of its shards."""
    path = Path(path)
    paths = [path]
    first = _open_one(path)
    try:
        kv = first[1]
        count = int(kv.get("split.count", 1))
        if count > 1:
            paths = _shard_paths(path, count)
            if int(kv["split.no"]) != paths.index(path):
                raise ValueError(f"{path}: split.no does not match its file name")
    except Exception:
        first[0].data.close(); first[0].file.close()
        raise
    shards, tensors = [], []
    kv0 = kv_raw0 = alignment0 = None
    try:
        for index, shard_path in enumerate(paths):
            opened = first if shard_path == path else _open_one(shard_path)
            shard, skv, skv_raw, stensors, alignment = opened
            shards.append(shard)
            if count > 1 and (int(skv.get("split.no", -1)) != index or
                              int(skv.get("split.count", -1)) != count):
                raise ValueError(f"inconsistent split metadata in {shard_path}")
            if index == 0:
                kv0, kv_raw0, alignment0 = skv, skv_raw, alignment
            for tensor in stensors:
                tensor.shard = index
                tensors.append(tensor)
        if count > 1 and len(tensors) != int(kv0["split.tensors.count"]):
            raise ValueError(f"split tensor count mismatch: {len(tensors)} != {kv0['split.tensors.count']}")
        names = [t.name for t in tensors]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tensor names across the input")
    except Exception:
        for shard in shards:
            shard.data.close(); shard.file.close()
        raise
    model = GgufFile(paths, kv0, kv_raw0, tensors, alignment0, shards)
    if header_only:
        model.close()
    return model


def _open_one(path: Path):
    source = path.open("rb")
    try:
        data = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
    except Exception:
        source.close()
        raise
    try:
        kv, kv_raw, tensors, alignment, data_offset = _parse_header(data, path, keep_raw=True)
    except Exception:
        data.close(); source.close()
        raise
    return Shard(path, source, data, data_offset), kv, kv_raw, tensors, alignment


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------
def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def encode_value(value) -> bytes:
    """Tag + payload for the metadata types the converter adds."""
    if isinstance(value, bool):
        return struct.pack("<IB", T_BOOL, int(value))
    if isinstance(value, U32):
        return struct.pack("<II", T_U32, int(value))
    if isinstance(value, int):
        return struct.pack("<IQ", T_U64, value)
    if isinstance(value, str):
        return struct.pack("<I", T_STRING) + _string(value)
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return struct.pack("<IIQ", T_ARRAY, T_STRING, len(value)) + b"".join(_string(v) for v in value)
    raise TypeError(f"unsupported metadata value {value!r}")


class U32(int):
    """Marks an integer metadata value to be written as uint32."""


@dataclass
class OutTensor:
    name: str
    ne: tuple
    type_name: str
    write: object        # callable(file) writing exactly nbytes

    @property
    def nbytes(self) -> int:
        count = 1
        for dim in self.ne:
            count *= dim
        return type_size(self.type_name, count, self.ne[0])


def write_gguf(path, kv_raw: list, tensors: list, alignment: int) -> None:
    """Write a single-file GGUF v3; tensor payloads are streamed by callbacks."""
    header = bytearray(b"GGUF")
    header += struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(kv_raw))
    for key, raw in kv_raw:
        header += _string(key) + raw
    offset = 0
    offsets = []
    for tensor in tensors:
        offsets.append(offset)
        header += _string(tensor.name)
        header += struct.pack("<I", len(tensor.ne))
        header += b"".join(struct.pack("<Q", d) for d in tensor.ne)
        header += struct.pack("<IQ", TYPE_ID[tensor.type_name], offset)
        offset += (tensor.nbytes + alignment - 1) // alignment * alignment
    header += b"\0" * (-len(header) % alignment)
    with Path(path).open("wb") as out:
        out.write(header)
        for tensor, expected in zip(tensors, offsets):
            start = out.tell() - len(header)
            if start != expected:
                raise AssertionError(f"{tensor.name}: offset {start} != {expected}")
            tensor.write(out)
            written = out.tell() - len(header) - start
            if written != tensor.nbytes:
                raise AssertionError(f"{tensor.name}: wrote {written} bytes, expected {tensor.nbytes}")
            out.write(b"\0" * (-written % alignment))


def as_float32(model: GgufFile, tensor: TensorInfo, rows: slice | None = None) -> np.ndarray:
    """Decode a float tensor (F32, F16, BF16) as float32 of NumPy shape (rows, K)."""
    raw = model.tensor_bytes(tensor)
    k = tensor.ne[0]
    rows_total = tensor.n_elements // k
    rows = rows or slice(0, rows_total)
    width = {"F32": 4, "F16": 2, "BF16": 2}.get(tensor.type_name)
    if width is None:
        raise ValueError(f"{tensor.name}: {tensor.type_name} is not a float type")
    chunk = raw[rows.start * k * width:rows.stop * k * width]
    if tensor.type_name == "F32":
        values = np.frombuffer(chunk, dtype="<f4")
    elif tensor.type_name == "F16":
        values = np.frombuffer(chunk, dtype="<f2").astype(np.float32)
    else:
        values = (np.frombuffer(chunk, dtype="<u2").astype(np.uint32) << np.uint32(16)).view(np.float32)
    return values.reshape(rows.stop - rows.start, k)
