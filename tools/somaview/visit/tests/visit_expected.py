# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""What VisIt should report for each file, from somaview's Python core: cells, scalar ranges,
and for AMR checkpoints the number of cells not covered by a finer level.

    python visit_expected.py OUT.json FILE[:VAR] ...
"""

import json
import sys

import numpy as np

import somaview as sv
from somaview.amr import AmrFile

out = {}
for spec in sys.argv[2:]:
    path, _, var = spec.partition(":")
    e = {"path": path}
    if sv.detect(path) == sv.FileKind.amr:
        with AmrFile(path) as af:
            cells = [np.prod(b[:, 3:] - b[:, :3] + 1, axis=1).sum() for b in af.boxes]
            covered = 0
            for l in range(1, af.nlevels):  # every fine box covers (cells / ratio^dim) coarse cells
                covered += int(cells[l]) // af.ratio ** af.dim
            e.update(kind="amr", boxes=int(sum(len(b) for b in af.boxes)), cells=int(sum(cells)),
                     uncovered=int(sum(cells)) - covered)
            if var:
                vals = np.concatenate([np.concatenate([a.ravel() for a in af.read(var, l)]) for l in range(af.nlevels)])
                e.update(var=var, min=float(vals.min()), max=float(vals.max()))
    else:
        p = sv.read(path)
        with sv.SomaFile(path) as sf:
            e.update(kind="mesh", parts=sf.nparts, dim=sf.dim)
        e.update(cells=len(p.cells), points=int(len(np.unique(p.point_gids))))
        if var:
            a = p.point_data.get(var)
            if a is None:
                a = p.cells.cell_data[var]
            a = np.asarray(a, float)
            e.update(var=var, min=float(np.nanmin(a)), max=float(np.nanmax(a)))
    out[path] = e
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out))
