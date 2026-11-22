# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Meshes from SOMA files: any subset of the parts as points, cells and fields in VTK's terms.

``read_piece`` reads the parts a process is given and nothing else. A part holds only the
vertices it owns, so the vertices its elements share with other parts (the halo) are found
through the sharing the file records: an owned shared vertex lists the parts holding copies of
it. Two ways to use that:

* scan (default): read the sharing tables of the other parts in bounded chunks and keep the
  rows whose copies are on the parts being read; cost proportional to the interface size;
* exchange (``comm`` given): each process reads only its own parts' sharing tables and sends its
  owned shared vertices, with their point field values, to the processes holding copies, in one
  all-to-all, as SOMA's own reader does. Cost proportional to the process's own interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .cells import vtk_cells
from .format import SET_DIM, FieldInfo, SomaFile, runs


def parts_for_rank(nparts: int, nranks: int, rank: int) -> list[int]:
    """The parts process ``rank`` of ``nranks`` reads: part q goes to floor(q nranks / nparts)."""
    q = np.arange(nparts, dtype=np.int64)
    return [int(x) for x in q[(q * nranks) // nparts == rank]]


def rank_of_parts(nparts: int, nranks: int) -> np.ndarray:
    return (np.arange(nparts, dtype=np.int64) * nranks) // nparts


class GidIndex:
    """Positions of global ids in an array of distinct gids, looked up vectorised."""

    def __init__(self, gids: np.ndarray):
        g = np.asarray(gids, dtype=np.int64)
        self.n = len(g)
        self._dense = None
        if self.n == 0:
            self._sorted = g
            self._order = g
            return
        lo, hi = int(g.min()), int(g.max())
        if hi - lo + 1 <= 4 * self.n + 1024:  # dense enough for a direct table
            self._lo = lo
            self._dense = np.full(hi - lo + 1, -1, dtype=np.int64)
            self._dense[g - lo] = np.arange(self.n, dtype=np.int64)
        else:
            self._order = np.argsort(g, kind="stable")
            self._sorted = g[self._order]

    def lookup(self, q: np.ndarray) -> np.ndarray:
        """Position of each gid of ``q``, or -1 where it is not in the index."""
        q = np.asarray(q, dtype=np.int64)
        if self.n == 0:
            return np.full(q.shape, -1, dtype=np.int64)
        if self._dense is not None:
            k = q - self._lo
            ok = (k >= 0) & (k < len(self._dense))
            out = np.full(q.shape, -1, dtype=np.int64)
            out[ok] = self._dense[k[ok]]
            return out
        pos = np.searchsorted(self._sorted, q)
        pos = np.minimum(pos, self.n - 1)
        hit = self._sorted[pos] == q
        return np.where(hit, self._order[pos], -1)


@dataclass
class CellBlock:
    """The elements of one dimension: VTK cells over the piece's points, and their data."""

    dim: int
    gids: np.ndarray
    part: np.ndarray
    types: np.ndarray
    offsets: np.ndarray
    connectivity: np.ndarray
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.types)


@dataclass
class MeshPiece:
    """What one process holds of a SOMA mesh: points with their data, cell blocks per dimension."""

    path: str
    dim: int
    parts: tuple[int, ...]
    points: np.ndarray
    point_gids: np.ndarray
    point_part: np.ndarray
    point_data: dict[str, np.ndarray]
    blocks: dict[int, CellBlock]
    field_data: dict[str, np.ndarray]
    halo_points: int = 0
    skipped: list[str] = field(default_factory=list)
    time: float | None = None

    @property
    def cells(self) -> CellBlock:
        """The block of elements of the mesh's dimension (or the highest one read)."""
        return self.blocks[self.dim] if self.dim in self.blocks else self.blocks[max(self.blocks)]

    @property
    def num_points(self) -> int:
        return len(self.points)

    @property
    def num_cells(self) -> int:
        return sum(len(b) for b in self.blocks.values())


# -- field values --------------------------------------------------------------------------------

def _float_fill(values: np.ndarray, present: np.ndarray, info: FieldInfo) -> np.ndarray:
    """Values with the rows of non-members set to NaN (integers promoted to float64)."""
    if present.all():
        return values
    out = values.astype(np.float64) if not np.issubdtype(values.dtype, np.floating) else values.copy()
    out[~present] = np.nan
    return out


def _values_at(sf: SomaFile, info: FieldInfo, dim: int, owner: np.ndarray, local: np.ndarray,
               cache: dict | None = None) -> np.ndarray:
    """Values of ``info`` on the entities (owner part, local row) of ``dim``; NaN off its domain."""
    base = f"{info.group}/dim{dim}"
    n = len(owner)
    shape = (n,) if info.ncomp == 1 else (n, info.ncomp)
    values_path = base + "/values"
    present = np.ones(n, dtype=bool)
    rows = np.empty(n, dtype=np.int64)
    if n == 0:
        return np.empty(shape, dtype=info.dtype)
    vstart = sf.part_rows(values_path)
    if info.full_on(dim):
        rows[:] = vstart[owner] + local
    else:
        key = ("full", info.index, dim)
        full = cache.get(key) if cache is not None else None
        if full is None:
            full = sf.read(base + "/full").astype(bool)
            if cache is not None:
                cache[key] = full
        order = np.argsort(owner, kind="stable")
        so = owner[order]
        bounds = np.flatnonzero(np.diff(so)) + 1
        for grp in np.split(order, bounds):
            q = int(owner[grp[0]])
            if full[q]:
                rows[grp] = vstart[q] + local[grp]
                continue
            dom = sf.read(base + "/domain", [q])
            pos = np.searchsorted(dom, local[grp])
            pos_c = np.minimum(pos, max(len(dom) - 1, 0))
            hit = (len(dom) > 0) & (dom[pos_c] == local[grp]) if len(dom) else np.zeros(len(grp), dtype=bool)
            present[grp] = hit
            rows[grp] = vstart[q] + pos_c
    got = np.empty(shape, dtype=info.dtype)
    if present.any():
        got[present] = sf.read_rows(values_path, rows[present])
    return _float_fill(got, present, info)


def _values_owned(sf: SomaFile, info: FieldInfo, dim: int, parts: Sequence[int], owner: np.ndarray,
                  local: np.ndarray, cache: dict) -> np.ndarray:
    if info.full_on(dim):  # aligned with the owned rows: one hyperslab per run of parts
        return sf.read(f"{info.group}/dim{dim}/values", parts)
    return _values_at(sf, info, dim, owner, local, cache)


# -- sets as cell labels -------------------------------------------------------------------------

def _set_labels(sf: SomaFile, info: FieldInfo, nsets: int, contains: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Per set, the value of the set-valued field ``info`` it inherits (NaN where none).

    A set takes the value of every set of the field's domain that contains it, directly or
    through other sets; where several do, the last in domain order wins.
    """
    label = np.full(nsets, np.nan)
    dom = sf.read(f"{info.group}/sets/domain")
    vals = sf.read(f"{info.group}/sets/values")
    starts, members = contains
    for s, v in zip(dom.tolist(), np.asarray(vals, dtype=np.float64).tolist()):
        seen = {s}
        stack = [s]
        while stack:
            t = stack.pop()
            for c in members[starts[t]:starts[t + 1]].tolist():
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        label[list(seen)] = v
    return label


# -- the reader ----------------------------------------------------------------------------------

class PieceReader:
    """Reads one process's parts of a SOMA mesh file in steps (see :func:`read_piece`).

    ``read_owned`` reads the parts' own rows; ``outgoing`` gives, per destination process, the
    owned shared vertices it holds copies of; ``finish`` gathers the halo (from received
    messages, or by scanning the file) and returns the :class:`MeshPiece`.
    """

    def __init__(self, sf: SomaFile, parts: Sequence[int] | None = None, dims: Iterable[int] | None = None,
                 point_fields: Iterable[str] | None = None, cell_fields: Iterable[str] | None = None,
                 high_order: bool = True, set_fields: Iterable[str] | bool = True, chunk_rows: int = 1 << 22):
        if sf.kind.value != "mesh":
            raise ValueError(f"{sf.path}: an AMR checkpoint, not a mesh (use somaview.amr)")
        self.sf = sf
        self.parts = sorted(set(range(sf.nparts) if parts is None else (int(p) for p in parts)))
        if any(p < 0 or p >= sf.nparts for p in self.parts):
            raise ValueError(f"parts must be in [0, {sf.nparts})")
        self.all_parts = len(self.parts) == sf.nparts
        written = [d for d in sf.written_dims if d > 0]
        self.dims = sorted(set(dims)) if dims is not None else [sf.dim]
        missing = [d for d in self.dims if d not in written]
        if missing:
            raise ValueError(f"{sf.path}: dimension(s) {missing} not in the file (written: {written})")
        self.high_order = high_order
        self.chunk_rows = chunk_rows
        self._cache: dict = {}
        self.skipped: list[str] = []
        self.point_fields = self._choose(0, point_fields)
        self.cell_fields = {d: self._choose(d, cell_fields) for d in self.dims}
        self.set_fields = self._choose_sets(set_fields)
        self._set_dtype = {f.name: f.dtype for f in self.set_fields}

    def _choose(self, dim: int, names) -> list[FieldInfo]:
        chosen = []
        for f in self.sf.fields:
            if dim not in f.dims or (names is not None and f.name not in names):
                continue
            if not f.drawable:
                why = "variable-length" if f.variable else "handle" if f.handle else f.layout.name
                self.skipped.append(f"{f.name} (dim {dim}: {why})")
                continue
            chosen.append(f)
        return chosen

    def _choose_sets(self, names) -> list[FieldInfo]:
        if names is False:
            return []
        out = []
        for f in self.sf.fields:
            if SET_DIM not in f.dims or f.handle or f.variable or f.ncomp != 1:
                continue
            if names is True or names is None or f.name in names:
                out.append(f)
        return out

    # step 1 -----------------------------------------------------------------------------------
    def read_owned(self) -> "PieceReader":
        sf, parts = self.sf, self.parts
        per = sf.counts_by_part(0)[parts]
        self.owned_gids = sf.read("dim0/gid", parts).astype(np.int64)
        self.owned_xyz = sf.read("dim0/coords", parts).reshape(-1, 3).astype(np.float64, copy=False)
        self.owned_part = np.repeat(np.asarray(parts, dtype=np.int32), per)
        self.owned_local = _local_rows(per)
        self.owned_index = GidIndex(self.owned_gids)
        self.owned_data = {f.name: _values_owned(sf, f, 0, parts, self.owned_part, self.owned_local, self._cache)
                           for f in self.point_fields}
        self.raw_cells = {}
        needed = []
        for d in self.dims:
            g = f"dim{d}"
            c = {"gids": sf.read(g + "/gid", parts).astype(np.int64),
                 "topology": sf.read(g + "/topology", parts),
                 "counts": sf.read(g + "/counts", parts).astype(np.int64),
                 "conn": sf.read(g + "/conn", parts).astype(np.int64)}
            pc = sf.counts_by_part(d)[parts]
            c["part"] = np.repeat(np.asarray(parts, dtype=np.int32), pc)
            c["local"] = _local_rows(pc)
            if self.high_order and sf.has(g + "/nodes"):
                c["node_counts"] = sf.read(g + "/node_counts", parts).astype(np.int64)
                c["nodes"] = sf.read(g + "/nodes", parts).astype(np.int64)
                needed.append(c["nodes"])
            needed.append(c["conn"])
            self.raw_cells[d] = c
        allv = np.unique(np.concatenate(needed)) if needed else np.empty(0, dtype=np.int64)
        self.missing = allv[self.owned_index.lookup(allv) < 0]
        return self

    # step 2 (exchange only) -------------------------------------------------------------------
    def outgoing(self, nranks: int, me: int, part_rank: np.ndarray | None = None) -> list[dict | None]:
        """Per destination process, the owned shared vertices it holds copies of (None: nothing)."""
        sf, parts = self.sf, self.parts
        if part_rank is None:
            part_rank = rank_of_parts(sf.nparts, nranks)
        index = sf.read("dim0/shared/index", parts).astype(np.int64)
        counts = sf.read("dim0/shared/counts", parts).astype(np.int64)
        dest = sf.read("dim0/shared/parts", parts).astype(np.int64)
        out: list[dict | None] = [None] * nranks
        if len(index) == 0:
            return out
        rows_per = np.diff(np.asarray([sf.part_rows("dim0/shared/index", p) for p in parts]), axis=1)[:, 0]
        owner = np.repeat(np.asarray(parts, dtype=np.int64), rows_per)
        # Position in this process's owned arrays of each shared row.
        first = np.zeros(len(parts) + 1, dtype=np.int64)
        np.cumsum(sf.counts_by_part(0)[parts], out=first[1:])
        slot = {p: k for k, p in enumerate(parts)}
        pos = first[np.fromiter((slot[int(q)] for q in owner), dtype=np.int64, count=len(owner))] + index
        row_of = np.repeat(np.arange(len(index)), counts)
        r = part_rank[dest]
        keep = r != me
        pairs = np.unique(np.stack([r[keep], row_of[keep]], axis=1), axis=0) if keep.any() else np.empty((0, 2), int)
        for rr in np.unique(pairs[:, 0]) if len(pairs) else []:
            sel = pos[pairs[pairs[:, 0] == rr, 1]]
            out[int(rr)] = {"gid": self.owned_gids[sel], "xyz": self.owned_xyz[sel],
                            "part": self.owned_part[sel], "data": {k: v[sel] for k, v in self.owned_data.items()}}
        return out

    # step 3 -----------------------------------------------------------------------------------
    def finish(self, received: list[dict | None] | None = None) -> MeshPiece:
        sf = self.sf
        halo = self._halo_from(received) if received is not None else self._halo_by_scan()
        points = np.concatenate([self.owned_xyz, halo["xyz"]]) if len(halo["gid"]) else self.owned_xyz
        gids = np.concatenate([self.owned_gids, halo["gid"]])
        ppart = np.concatenate([self.owned_part, halo["part"].astype(np.int32)])
        pdata = {}
        for f in self.point_fields:
            a, b = self.owned_data[f.name], halo["data"][f.name]
            if a.dtype != b.dtype:
                a, b = a.astype(np.float64), b.astype(np.float64)
            pdata[f.name] = np.concatenate([a, b])
        index = GidIndex(gids) if len(halo["gid"]) else self.owned_index
        blocks = {}
        set_info = self._sets_info() if self.set_fields else None
        for d, c in self.raw_cells.items():
            conn = index.lookup(c["conn"])
            nodes = index.lookup(c["nodes"]) if "nodes" in c else None
            if (conn < 0).any() or (nodes is not None and (nodes < 0).any()):
                raise RuntimeError(f"{sf.path}: elements of dimension {d} use vertices no part shares with "
                                   f"parts {self._parts_text()}")
            types, offsets, vconn = vtk_cells(c["topology"], c["counts"], conn, c.get("node_counts"), nodes,
                                              self.high_order)
            cdata = {f.name: _values_owned(sf, f, d, self.parts, c["part"], c["local"], self._cache)
                     for f in self.cell_fields[d]}
            if set_info is not None:
                cdata.update(self._set_cell_data(d, c, set_info))
            blocks[d] = CellBlock(d, c["gids"], c["part"], types, offsets, vconn, cdata)
        fdata = {k: v.value for k, v in sf.values.items() if v.value is not None}
        return MeshPiece(sf.path, sf.dim, tuple(self.parts), points, gids, ppart, pdata, blocks, fdata,
                         halo_points=len(halo["gid"]), skipped=sorted(set(self.skipped)), time=sf.time())

    def _parts_text(self) -> str:
        return ", ".join(f"{a}" if b == a + 1 else f"{a}-{b - 1}" for a, b in runs(np.asarray(self.parts)))

    def _empty_halo(self) -> dict:
        return {"gid": np.empty(0, np.int64), "xyz": np.empty((0, 3)), "part": np.empty(0, np.int32),
                "data": {f.name: np.empty((0,) if f.ncomp == 1 else (0, f.ncomp), f.dtype) for f in self.point_fields}}

    def _halo_from(self, received: list[dict | None]) -> dict:
        msgs = [m for m in received if m is not None and len(m["gid"])]
        if not msgs or len(self.missing) == 0:
            if len(self.missing):
                raise RuntimeError(f"{self.sf.path}: no process sent the {len(self.missing)} halo vertices of parts "
                                   f"{self._parts_text()}")
            return self._empty_halo()
        gid = np.concatenate([m["gid"] for m in msgs])
        gid_u, first = np.unique(gid, return_index=True)
        want = np.isin(gid_u, self.missing)
        sel = first[want]
        if want.sum() != len(self.missing):
            raise RuntimeError(f"{self.sf.path}: {len(self.missing) - int(want.sum())} halo vertices of parts "
                               f"{self._parts_text()} were not received")
        cat = lambda k: np.concatenate([m[k] for m in msgs])[sel]
        return {"gid": gid[sel], "xyz": cat("xyz"), "part": cat("part"),
                "data": {f.name: np.concatenate([m["data"][f.name] for m in msgs])[sel] for f in self.point_fields}}

    def _halo_by_scan(self) -> dict:
        """The halo vertices, from the other parts' sharing tables read in chunks of parts."""
        sf = self.sf
        if len(self.missing) == 0:
            return self._empty_halo()
        mine = np.asarray(self.parts, dtype=np.int64)
        others = np.setdiff1d(np.arange(sf.nparts), mine)
        idx_rows = sf.part_rows("dim0/shared/index")
        owners, locals_ = [], []
        for group in _chunks_of_parts(others, idx_rows, self.chunk_rows):
            index = sf.read("dim0/shared/index", group).astype(np.int64)
            if len(index) == 0:
                continue
            counts = sf.read("dim0/shared/counts", group).astype(np.int64)
            dest = sf.read("dim0/shared/parts", group).astype(np.int64)
            hit_rows = np.unique(np.repeat(np.arange(len(index)), counts)[np.isin(dest, mine)])
            rows_per = np.asarray([idx_rows[p + 1] - idx_rows[p] for p in group])
            owner = np.repeat(np.asarray(group, dtype=np.int64), rows_per)
            owners.append(owner[hit_rows])
            locals_.append(index[hit_rows])
        owner = np.concatenate(owners) if owners else np.empty(0, np.int64)
        local = np.concatenate(locals_) if locals_ else np.empty(0, np.int64)
        gstart = sf.part_rows("dim0/gid")
        gid = sf.read_rows("dim0/gid", gstart[owner] + local).astype(np.int64) if len(owner) else np.empty(0, np.int64)
        keep = np.isin(gid, self.missing)
        gid_k, first = np.unique(gid[keep], return_index=True)
        sel = np.flatnonzero(keep)[first]
        if len(gid_k) != len(self.missing):  # sharing incomplete: find the rest by their gids
            rest = np.setdiff1d(self.missing, gid_k)
            o2, l2 = self._scan_gids(rest, others)
            owner = np.concatenate([owner[sel], o2])
            local = np.concatenate([local[sel], l2])
            gid = np.concatenate([gid[sel], rest])
        else:
            owner, local, gid = owner[sel], local[sel], gid[sel]
        rows = gstart[owner] + local
        return {"gid": gid, "xyz": sf.read_rows("dim0/coords", rows).reshape(-1, 3), "part": owner.astype(np.int32),
                "data": {f.name: _values_at(sf, f, 0, owner, local, self._cache) for f in self.point_fields}}

    def _scan_gids(self, wanted: np.ndarray, others: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(owner part, local row) of the gids ``wanted``, found by reading the other parts' gids."""
        sf = self.sf
        idx = GidIndex(wanted)
        found_o = np.full(len(wanted), -1, np.int64)
        found_l = np.full(len(wanted), -1, np.int64)
        gstart = sf.part_rows("dim0/gid")
        for group in _chunks_of_parts(others, gstart, self.chunk_rows):
            g = sf.read("dim0/gid", group).astype(np.int64)
            k = idx.lookup(g)
            hit = np.flatnonzero(k >= 0)
            if len(hit):
                per = np.asarray([gstart[p + 1] - gstart[p] for p in group])
                owner = np.repeat(np.asarray(group, np.int64), per)
                found_o[k[hit]] = owner[hit]
                found_l[k[hit]] = _local_rows(per)[hit]
        if (found_o < 0).any():
            raise RuntimeError(f"{sf.path}: {(found_o < 0).sum()} vertices used by parts {self._parts_text()} "
                               "are in no part")
        return found_o, found_l

    def _sets_info(self):
        sf = self.sf
        nsets = len(sf.read("sets/gid"))
        if nsets == 0:
            return None
        # Containment as CSR over local set indices: unordered sets list contained sets' gids in
        # set_members; ordered sets list them in entries as (set, position, gid).
        sidx = GidIndex(sf.read("sets/gid").astype(np.int64))
        mc = sf.read("sets/set_members_counts").astype(np.int64)
        owner = np.repeat(np.arange(nsets, dtype=np.int64), mc)
        child = sidx.lookup(sf.read("sets/set_members").astype(np.int64))
        entries = sf.read("sets/entries").reshape(-1, 3).astype(np.int64)
        if len(entries):
            owner = np.concatenate([owner, entries[:, 0]])
            child = np.concatenate([child, sidx.lookup(entries[:, 2])])
        ok = child >= 0
        owner, child = owner[ok], child[ok]
        order = np.argsort(owner, kind="stable")
        starts = np.zeros(nsets + 1, np.int64)
        np.cumsum(np.bincount(owner, minlength=nsets), out=starts[1:])
        contains = (starts, child[order])
        members = sf.read("sets/members", self.parts).reshape(-1, 4).astype(np.int64)
        mrows = np.diff(sf.part_rows("sets/members"))[self.parts]
        mpart = np.repeat(np.asarray(self.parts, np.int64), mrows)
        labels = {f.name: _set_labels(sf, f, nsets, contains) for f in self.set_fields}
        return members, mpart, labels

    def _set_cell_data(self, d: int, c: dict, info) -> dict[str, np.ndarray]:
        members, mpart, labels = info
        on = members[:, 1] == d
        if not on.any():
            return {}
        m, mp = members[on], mpart[on]
        # Position of (part, local row) among this block's cells.
        first = {p: k for p, k in zip(self.parts, np.concatenate(([0], np.cumsum(self.sf.counts_by_part(d)[self.parts])))[:-1])}
        cell = np.fromiter((first[int(p)] for p in mp), np.int64, len(mp)) + m[:, 2]
        out = {}
        for name, label in labels.items():
            v = label[m[:, 0]]
            ok = ~np.isnan(v)
            if not ok.any():
                continue
            a = np.full(len(c["gids"]), np.nan)
            a[cell[ok]] = v[ok]
            dtype = self._set_dtype[name]
            if np.issubdtype(dtype, np.integer) and not np.isnan(a).any():
                a = a.astype(dtype)  # every cell labelled: keep the field's integer type
            out[f"set:{name}"] = a
        return out


def _local_rows(per_part: np.ndarray) -> np.ndarray:
    """0, 1, ... within each part, for parts of the given sizes laid end to end."""
    per_part = np.asarray(per_part, dtype=np.int64)
    total = int(per_part.sum())
    starts = np.repeat(np.cumsum(per_part) - per_part, per_part)
    return np.arange(total, dtype=np.int64) - starts


def _chunks_of_parts(parts: np.ndarray, starts: np.ndarray, chunk_rows: int):
    """Groups of consecutive entries of ``parts`` holding at most about chunk_rows rows each."""
    group, rows = [], 0
    for p in parts.tolist():
        n = int(starts[p + 1] - starts[p])
        if group and rows + n > chunk_rows:
            yield group
            group, rows = [], 0
        group.append(p)
        rows += n
    if group:
        yield group


def read_piece(source: str | SomaFile, parts: Sequence[int] | None = None, *, dims: Iterable[int] | None = None,
               point_fields: Iterable[str] | None = None, cell_fields: Iterable[str] | None = None,
               high_order: bool = True, set_fields: Iterable[str] | bool = True, comm=None,
               rank: int | None = None, nranks: int | None = None) -> MeshPiece:
    """Read parts of a SOMA mesh file as a :class:`MeshPiece`.

    Parameters
    ----------
    source : path or :class:`SomaFile`
    parts : parts to read. Default: all, or with ``comm`` (or ``rank``/``nranks``) the parts of this
        process, part q on process floor(q P / N) as SOMA's reader assigns them.
    dims : element dimensions to read (default: the mesh's dimension). Lower ones give extra
        blocks, e.g. the boundary faces a mesh was built with.
    point_fields, cell_fields : names to read (default: every field a renderer can show).
    high_order : make quadratic VTK cells of elements with high-order nodes.
    set_fields : label cells with the values of fields on entity sets (``MATERIAL_SET`` ...), as
        cell arrays named ``set:<name>``; True for all, False for none, or a list of names.
    comm : an mpi4py communicator; the halo is then exchanged rather than scanned for.
    """
    own = not isinstance(source, SomaFile)
    sf = SomaFile(source) if own else source
    try:
        if comm is not None:
            if parts is not None:
                raise ValueError("read_piece: with comm, each process reads the parts SOMA assigns it; "
                                 "do not pass parts")
            rank, nranks = comm.Get_rank(), comm.Get_size()
        if parts is None and rank is not None:
            parts = parts_for_rank(sf.nparts, nranks, rank)
        reader = PieceReader(sf, parts, dims, point_fields, cell_fields, high_order, set_fields).read_owned()
        if comm is not None and nranks > 1:  # collective: every process takes part
            return reader.finish(comm.alltoall(reader.outgoing(nranks, rank)))
        return reader.finish()
    finally:
        if own:
            sf.close()
