# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Test helper, run as its own process: what SOMA's reader makes of a file, saved as .npz.

Run without h5py in the process (h5py's HDF5 and SOMA's would clash). Saves per-gid vertex
coordinates, element corner gids, and every field's values on vertices and elements, so the
tests can compare somaview's reading with SOMA's own.

    python soma_oracle.py <file.soma> <out.npz>
"""

import sys

sys.modules["mpi4py"] = None  # no MPI here: SOMA built without it

import numpy as np  # noqa: E402
import soma  # noqa: E402


def main(path, out):
    m = soma.read_file(soma.Comm.self(), path)
    d = m.dim
    n = m.owned_count(d)
    save = {"dim": np.array(d), "vgid": np.asarray(m.gids(0)), "xyz": np.asarray(m.coords()),
            "cgid": np.asarray(m.gids(d))[:n], "topo": np.asarray(m.topology(d))[:n]}
    off = np.asarray(m.conn_offsets(d))[: n + 1].astype(np.int64)
    conn = np.asarray(m.conn(d)).astype(np.int64)
    save["coff"] = off
    save["cconn_gid"] = np.asarray(m.gids(0))[conn[off[0]:off[-1]]]
    for info in m.fields():
        name = info["name"]
        if info["variable"] or name.startswith("soma:"):
            continue
        for dd in info["dims"]:
            if dd in (0, d):
                try:
                    save[f"field:{dd}:{name}"] = np.asarray(m.field(name, dd))
                except Exception as e:  # partial domains etc.
                    save[f"err:{dd}:{name}"] = np.array(str(e))
    np.savez(out, **save)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
