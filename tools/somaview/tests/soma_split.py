# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Test helper: rewrite a one-part SOMA mesh file as N parts, as N ranks would have written it.

SOMA writes one part per rank; without MPI a test cannot make multi-part files with point
fields, so this rewrites a file SOMA wrote on one rank. Elements of the mesh's dimension go to
parts in slabs along an axis; a vertex is owned by the lowest part using it and lists the
others as holding copies. The result is checked by SOMA's own reader in the tests (it must read
back the same mesh), not trusted on its own. Files with written dimensions other than 0 and the
mesh's dimension are refused.
"""

import h5py
import numpy as np


def _sq(a):
    return a[:, 0] if a.ndim == 2 and a.shape[1] == 1 else a


def split(src, dst, nparts, axis=0):
    with h5py.File(src, "r") as f:
        r = f["soma"]
        meta = np.array(r.attrs["meta"])
        D = int(meta[1])
        assert int(meta[2]) == 1, "source must have one part"
        assert int(meta[3]) == (1 | (1 << D)), "only files written with dims 0 and D"
        cols = [c for c in r["parts/offsets"].attrs["columns"].decode().split("\n") if c]
        old = {c: r[c][...] for c in cols}
    g = f"dim{D}"
    vg, xyz = _sq(old["dim0/gid"]), old["dim0/coords"]
    vpos = {int(x): k for k, x in enumerate(vg)}
    cnt = _sq(old[g + "/counts"]).astype(np.int64)
    cs = np.concatenate([[0], np.cumsum(cnt)])
    conn = _sq(old[g + "/conn"])
    has_nodes = g + "/nodes" in old
    ncnt = _sq(old[g + "/node_counts"]).astype(np.int64) if has_nodes else np.zeros(len(cnt), np.int64)
    ns = np.concatenate([[0], np.cumsum(ncnt)])
    nodes = _sq(old[g + "/nodes"]) if has_nodes else np.empty(0, np.int64)
    ncell = len(cnt)
    cent = np.array([xyz[[vpos[int(v)] for v in conn[cs[e]:cs[e + 1]]], axis].mean() for e in range(ncell)])
    rank = np.empty(ncell, np.int64)
    rank[np.argsort(cent, kind="stable")] = np.arange(ncell)
    cpart = (rank * nparts) // ncell
    users = [set() for _ in vg]
    for e in range(ncell):
        for v in list(conn[cs[e]:cs[e + 1]]) + list(nodes[ns[e]:ns[e + 1]]):
            users[vpos[int(v)]].add(int(cpart[e]))
    vpart = np.array([min(u) if u else 0 for u in users], np.int64)
    # Per part: entity rows in original order; new local id of each original row.
    order = {0: [np.flatnonzero(vpart == p) for p in range(nparts)], D: [np.flatnonzero(cpart == p) for p in range(nparts)]}
    newlocal = {}
    for d in (0, D):
        nl = np.empty(len(vpart if d == 0 else cpart), np.int64)
        owner = vpart if d == 0 else cpart
        for p in range(nparts):
            nl[order[d][p]] = np.arange(len(order[d][p]))
        newlocal[d] = (owner, nl)

    new = {c: [] for c in cols}

    def put(c, p, arr):
        new[c].append((p, np.asarray(arr)))

    for p in range(nparts):
        rows0, rowsD = order[0][p], order[D][p]
        put("dim0/gid", p, old["dim0/gid"][rows0])
        put("dim0/coords", p, xyz[rows0])
        sh = [(k, sorted(users[v] - {p})) for k, v in enumerate(rows0) if users[v] - {p}]
        put("dim0/shared/index", p, np.array([k for k, _ in sh], np.int64).reshape(-1, 1))
        put("dim0/shared/counts", p, np.array([len(q) for _, q in sh], np.uint32).reshape(-1, 1))
        put("dim0/shared/parts", p, np.array([x for _, q in sh for x in q], np.int32).reshape(-1, 1))
        for name in ("gid", "topology", "counts"):
            put(f"{g}/{name}", p, old[f"{g}/{name}"][rowsD])
        put(f"{g}/conn", p, np.concatenate([conn[cs[e]:cs[e + 1]] for e in rowsD]).reshape(-1, 1)
            if len(rowsD) else np.empty((0, 1), np.int64))
        if has_nodes:
            put(f"{g}/node_counts", p, old[f"{g}/node_counts"][rowsD])
            put(f"{g}/nodes", p, np.concatenate([nodes[ns[e]:ns[e + 1]] for e in rowsD]).reshape(-1, 1)
                if len(rowsD) else np.empty((0, 1), np.int64))
        for name in ("index", "counts", "parts"):
            c = f"{g}/shared/{name}"
            put(c, p, old[c][:0])
    for c in cols:
        if not c.startswith("fields/"):
            continue
        parts_ = c.split("/")
        base, leaf = "/".join(parts_[:3]), parts_[3]
        d = int(parts_[2][3:])
        if leaf != "full":
            continue
        full0 = bool(_sq(old[base + "/full"])[0])
        dom = _sq(old[base + "/domain"]).astype(np.int64)
        vals = old[base + "/values"]
        assert base + "/counts" not in old, "only per-entity fields"
        owner, nl = newlocal[d]
        members = np.arange(len(owner)) if full0 else dom  # original local ids holding values
        for p in range(nparts):
            mine = members[owner[members] == p]
            rows = mine if full0 else np.searchsorted(dom, mine)
            full_p = len(mine) == len(order[d][p])
            # values in new local order: new local ids of mine are increasing (order kept)
            put(base + "/full", p, np.array([[1 if full_p else 0]], np.uint8))
            put(base + "/domain", p, (np.empty((0, 1), np.int64) if full_p else nl[mine].reshape(-1, 1)))
            put(base + "/values", p, vals[rows])
    mem = old["sets/members"].astype(np.int64)
    for p in range(nparts):
        rows = []
        for s, d, l, pos in mem.tolist():
            owner, nl = newlocal[d]
            if owner[l] == p:
                rows.append([s, d, int(nl[l]), pos])
        put("sets/members", p, np.array(rows, np.int64).reshape(-1, 4))
        put("sets/entries", p, old["sets/entries"] if p == 0 else old["sets/entries"][:0])

    table = np.zeros((nparts + 1, len(cols)), np.uint64)
    with h5py.File(src, "r") as f, h5py.File(dst, "w", libver=("v114", "latest")) as out:
        f.copy(f["soma"], out, "soma")
        r = out["soma"]
        m = np.array(r.attrs["meta"]); m[2] = nparts
        r.attrs.modify("meta", m)
        for k, c in enumerate(cols):
            pieces = new[c]
            assert len(pieces) == nparts, (c, len(pieces))
            arr = np.concatenate([a.reshape(-1, *r[c].shape[1:]) for _, a in pieces]).astype(r[c].dtype)
            sizes = [len(a.reshape(-1, *r[c].shape[1:])) for _, a in pieces]
            table[1:, k] = np.cumsum(sizes)
            dt = r[c].dtype
            del r[c]
            r.create_dataset(c, data=arr, dtype=dt)
        attrs = dict(r["parts/offsets"].attrs)
        del r["parts/offsets"]
        t = r.create_dataset("parts/offsets", data=table)
        for k, v in attrs.items():
            t.attrs[k] = v
    return cpart, vpart
