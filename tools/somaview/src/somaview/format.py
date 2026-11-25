# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""The SOMA file format read with h5py: root metadata, the part table, fields and mesh values.

A SOMA mesh file keeps everything under the group ``/soma``. The datasets written by every part
are the concatenation of the parts' rows in part order, and ``/soma/parts/offsets`` (N + 1 rows,
one column per such dataset) says where each part starts. This module knows that layout and
nothing about meshes; :mod:`somaview.mesh` builds meshes from it.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import h5py
import numpy as np

#: Newest ``format_version`` this reader understands.
FORMAT_VERSION = 1

#: ``soma::DataType`` codes, in enum order.
DTYPES = (np.int8, np.uint8, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64)

#: Slot of the entity sets in a field's dimension list (``soma::set_dim``).
SET_DIM = 4


class Topology(enum.IntEnum):
    """``soma::Topology`` codes, as stored in ``dimD/topology``."""

    vertex = 0
    edge = 1
    tri = 2
    quad = 3
    polygon = 4
    tet = 5
    pyramid = 6
    prism = 7
    knife = 8
    hex = 9
    polyhedron = 10


class LayoutKind(enum.IntEnum):
    """``soma::Layout::Kind``: where a field's rows sit on its entities."""

    per_entity = 0
    element_nodes = 1
    shared_nodes = 2
    quadrature = 3


#: ``soma::NodeFamily::gll`` and ``soma::NodeOrdering::tensor_lexicographic``.
FAMILY_GLL = 1
ORDERING_TENSOR = 1
#: Largest order of a node table (``soma::max_node_order``).
MAX_NODE_ORDER = 32


class FileKind(enum.Enum):
    mesh = "mesh"
    amr = "amr"


@dataclass(frozen=True)
class FieldInfo:
    """One field of a SOMA mesh file (``/soma/fields/fI``)."""

    index: int
    name: str
    dtype: np.dtype
    ncomp: int
    variable: bool
    handle: bool
    layout: LayoutKind
    order: int
    full_bits: int
    dims: tuple[int, ...]
    default: np.ndarray | None = None
    global_value: np.ndarray | None = None
    family: int = 0
    ordering: int = 0
    serendipity: bool = False

    @property
    def group(self) -> str:
        return f"fields/f{self.index}"

    @property
    def drawable(self) -> bool:
        """True for fields with one fixed-size row per entity: the ones a renderer can show."""
        return self.layout == LayoutKind.per_entity and not self.variable and not self.handle

    @property
    def nodal_on_quads(self) -> bool:
        """True for a spectral-element field: GLL values on quads (see :mod:`somaview.nodal`)."""
        return (self.layout == LayoutKind.element_nodes and not self.variable and not self.handle
                and self.family == FAMILY_GLL and self.ordering == ORDERING_TENSOR and not self.serendipity
                and 1 <= self.order <= MAX_NODE_ORDER and 2 in self.dims)

    def full_on(self, dim: int) -> bool:
        """True when every part's domain covers all its owned entities of ``dim``."""
        return bool((self.full_bits >> dim) & 1)


@dataclass(frozen=True)
class MeshValue:
    """A mesh value (``/soma/values/vI``): a named constant of the whole mesh."""

    name: str
    dtype: np.dtype
    ncomp: int
    value: np.ndarray | None


@dataclass(frozen=True)
class Geometry:
    """The surface a mesh is on: ``kind`` "affine" (the elements' own maps), "sphere" or "unknown" (of a
    kind not known here); ``declared`` by the application, or recognized from the vertices on writing."""

    kind: str
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    declared: bool = True


@dataclass(frozen=True)
class RootMeta:
    """The words of the ``/soma`` attribute ``meta``."""

    format_version: int
    dim: int
    nparts: int
    written: tuple[int, ...]
    input: tuple[int, ...]
    nodes: tuple[int, ...]
    graphs: int
    nsets: int
    nfields: int
    nvalues: int
    ordering: int


def _bits(mask: int) -> tuple[int, ...]:
    return tuple(d for d in range(4) if (int(mask) >> d) & 1)


def _text(a) -> str:
    if isinstance(a, bytes):
        return a.decode()
    if isinstance(a, np.ndarray):
        return _text(a.item()) if a.shape == () else _text(a.tolist()[0])
    return str(a)


def _typed(raw, dtype) -> np.ndarray | None:
    if raw is None:
        return None
    return np.frombuffer(np.asarray(raw, dtype=np.uint8).tobytes(), dtype=np.dtype(dtype).newbyteorder("<")).copy()


def detect(path: str | os.PathLike) -> FileKind | None:
    """``FileKind.mesh`` or ``FileKind.amr`` for a SOMA file, None for anything else."""
    try:
        with h5py.File(path, "r") as f:
            if "soma" not in f:
                return None
            a = f["soma"].attrs
            return FileKind.amr if "amr" in a else FileKind.mesh if "meta" in a else None
    except (OSError, ValueError):
        return None


def runs(sorted_ints: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs [a, b) of consecutive values in a sorted array of distinct integers."""
    if len(sorted_ints) == 0:
        return []
    v = np.asarray(sorted_ints, dtype=np.int64)
    cut = np.flatnonzero(np.diff(v) != 1) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [len(v)]))
    return [(int(v[s]), int(v[e - 1]) + 1) for s, e in zip(starts, ends)]


class SomaFile:
    """A SOMA file open for reading.

    ``SomaFile(path)`` opens it with the default HDF5 driver; pass ``driver="mpio", comm=...``
    to open it collectively with a parallel h5py. The file is closed by :meth:`close` or on
    leaving a ``with`` block.
    """

    def __init__(self, path: str | os.PathLike, driver: str | None = None, comm=None, **kw):
        self.path = os.fspath(path)
        args = {"driver": driver, "comm": comm} if driver == "mpio" else ({"driver": driver} if driver else {})
        self._h5 = h5py.File(self.path, "r", **args, **kw)
        if "soma" not in self._h5:
            self._h5.close()
            raise ValueError(f"{self.path}: not a SOMA file (no /soma group)")
        self.root = self._h5["soma"]
        a = self.root.attrs
        if "amr" in a:
            self.kind = FileKind.amr
            self.meta = None
            words = np.asarray(a["amr"], dtype=np.int64)
            self.format_version = int(words[0])
            self.nparts = int(words[2])
        elif "meta" in a:
            self.kind = FileKind.mesh
            w = [int(x) for x in np.asarray(a["meta"], dtype=np.int64)]
            self.meta = RootMeta(w[0], w[1], w[2], _bits(w[3]), _bits(w[4]), _bits(w[5]), w[6], w[7], w[8], w[9],
                                 w[10] if len(w) > 10 else 0)
            self.format_version = self.meta.format_version
            self.nparts = self.meta.nparts
        else:
            self._h5.close()
            raise ValueError(f"{self.path}: /soma has neither the attribute meta nor amr")
        if self.format_version > FORMAT_VERSION:
            self._h5.close()
            raise ValueError(f"{self.path}: format_version {self.format_version} is newer than this reader "
                             f"({FORMAT_VERSION})")
        self._table = None
        self._columns: dict[str, int] = {}
        if "parts/offsets" in self.root:
            t = self.root["parts/offsets"]
            self._table = np.asarray(t[...], dtype=np.int64)
            cols = _text(t.attrs["columns"]).split("\n")
            self._columns = {c: k for k, c in enumerate(c for c in cols if c)}
        self.fields: list[FieldInfo] = self._read_fields() if self.kind == FileKind.mesh else []
        self.values: dict[str, MeshValue] = self._read_values() if self.kind == FileKind.mesh else {}

    # -- context management -------------------------------------------------------------------
    def close(self) -> None:
        if self._h5:
            self._h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        if self.kind == FileKind.amr:
            return f"SomaFile({self.path!r}, AMR checkpoint, {self.nparts} parts)"
        m = self.meta
        return (f"SomaFile({self.path!r}, dim {m.dim}, {self.nparts} parts, dims written {list(m.written)}, "
                f"{len(self.fields)} fields)")

    # -- mesh-level queries -------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return self.meta.dim

    @property
    def written_dims(self) -> tuple[int, ...]:
        return self.meta.written

    def has(self, path: str) -> bool:
        return path in self.root

    def dataset(self, path: str) -> h5py.Dataset:
        return self.root[path]

    def is_column(self, path: str) -> bool:
        """True for a dataset every part writes rows of (one listed in the part table)."""
        return path in self._columns

    def part_rows(self, path: str, part: int | None = None) -> np.ndarray | tuple[int, int]:
        """Rows [start, stop) of ``path`` that ``part`` wrote, or the N + 1 starts when part is None."""
        k = self._columns[path]
        if part is None:
            return self._table[:, k]
        return int(self._table[part, k]), int(self._table[part + 1, k])

    def count(self, dim: int, parts: Iterable[int] | None = None) -> int:
        """Owned entities of ``dim`` over ``parts`` (all by default)."""
        path = f"dim{dim}/gid"
        if path not in self._columns:
            return 0
        starts = self.part_rows(path)
        if parts is None:
            return int(starts[-1])
        return int(sum(starts[p + 1] - starts[p] for p in parts))

    def counts_by_part(self, dim: int) -> np.ndarray:
        path = f"dim{dim}/gid"
        if path not in self._columns:
            return np.zeros(self.nparts, dtype=np.int64)
        return np.diff(self.part_rows(path))

    # -- reading rows -------------------------------------------------------------------------
    def read(self, path: str, parts: Sequence[int] | None = None) -> np.ndarray:
        """The rows ``parts`` wrote of a per-part dataset, in part order (every row by default).

        Consecutive parts are read in one hyperslab each. A dataset of width 1 comes back 1-D.
        """
        ds = self.root[path]
        if parts is None:
            out = ds[...]
        else:
            pieces = []
            starts = self.part_rows(path)
            for a, b in runs(np.unique(np.asarray(parts, dtype=np.int64))):
                lo, hi = int(starts[a]), int(starts[b])
                if hi > lo:
                    pieces.append(ds[lo:hi])
            out = np.concatenate(pieces) if pieces else np.empty((0,) + ds.shape[1:], dtype=ds.dtype)
        return _squeeze(out)

    def read_rows(self, path: str, rows: np.ndarray, gap: int = 4096) -> np.ndarray:
        """Rows at global row indices ``rows`` (any order, repeats allowed) of a dataset.

        The distinct rows are coalesced into hyperslabs, merging runs closer than ``gap`` rows,
        so a scattered selection costs a few large reads rather than many small ones.
        """
        ds = self.root[path]
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return _squeeze(np.empty((0,) + ds.shape[1:], dtype=ds.dtype))
        u, inverse = np.unique(rows, return_inverse=True)
        cut = np.flatnonzero(np.diff(u) > gap) + 1
        starts = np.concatenate(([0], cut))
        ends = np.concatenate((cut, [len(u)]))
        parts = []
        for s, e in zip(starts, ends):
            lo, hi = int(u[s]), int(u[e - 1]) + 1
            block = ds[lo:hi]
            parts.append(block[u[s:e] - lo])
        got = np.concatenate(parts)
        return _squeeze(got[inverse])

    # -- fields and values --------------------------------------------------------------------
    def _read_fields(self) -> list[FieldInfo]:
        out = []
        for i in range(self.meta.nfields):
            g = self.root[f"fields/f{i}"]
            m = [int(x) for x in np.asarray(g.attrs["meta"], dtype=np.int64)]
            dtype = np.dtype(DTYPES[m[0]])
            ndims = m[16]
            dims = tuple(m[17:17 + ndims])
            out.append(FieldInfo(
                index=i, name=_text(g.attrs["name"]), dtype=dtype, ncomp=m[1], variable=bool(m[2]),
                handle=bool(m[3]), layout=LayoutKind(m[4]), order=m[6], full_bits=m[15], dims=dims,
                default=_typed(g.attrs.get("default"), dtype) if m[13] else None,
                global_value=_typed(g.attrs.get("global"), dtype) if m[14] else None,
                family=m[5], ordering=m[7], serendipity=bool(m[8])))
        return out

    def _read_values(self) -> dict[str, MeshValue]:
        out = {}
        for i in range(self.meta.nvalues):
            g = self.root[f"values/v{i}"]
            m = [int(x) for x in np.asarray(g.attrs["meta"], dtype=np.int64)]
            dtype = np.dtype(DTYPES[m[0]])
            raw = g.attrs.get("global") if m[7] else (g.attrs.get("default") if m[6] else None)
            name = _text(g.attrs["name"])
            out[name] = MeshValue(name, dtype, m[1], _typed(raw, dtype))
        return out

    def fields_on(self, dim: int, drawable_only: bool = True) -> list[FieldInfo]:
        """The fields with values on entities of ``dim``."""
        return [f for f in self.fields if dim in f.dims and (f.drawable or not drawable_only)]

    @property
    def geometry(self) -> Geometry | None:
        """The surface the file says its mesh is on (``soma:geometry``), if it says."""
        v = self.values.get("soma:geometry")
        if v is None or v.value is None:
            return None
        w = np.asarray(v.value, float).ravel()
        if len(w) < 3:
            raise ValueError(f"{self.path}: soma:geometry is cut short")
        if w[0] != 1 or w[1] not in (0, 1):
            return Geometry("unknown", declared=bool(w[2]))
        if w[1] == 0:
            return Geometry("affine", declared=bool(w[2]))
        if len(w) < 7 or not np.isfinite(w[3:7]).all() or w[6] <= 0:
            raise ValueError(f"{self.path}: soma:geometry gives no sphere")
        return Geometry("sphere", tuple(float(c) for c in w[3:6]), float(w[6]), bool(w[2]))

    def time(self) -> float | None:
        """The value of a scalar mesh value named ``time`` (any case), if the file has one."""
        for name, v in self.values.items():
            if name.lower() == "time" and v.value is not None and v.value.size >= 1:
                return float(v.value[0])
        return None


def _squeeze(a: np.ndarray) -> np.ndarray:
    return a[:, 0] if a.ndim == 2 and a.shape[1] == 1 else a
