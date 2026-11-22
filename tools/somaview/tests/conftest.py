# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "src"), str(HERE)]

#: The SOMA-written ground-truth files (SOMAVIEW_DATA, default tests/data, made by
#: tests/make_data.sh), and SOMA's Python package for the oracle (SOMAVIEW_SOMA_PYTHON: a SOMA
#: build's python/ directory; the oracle tests skip without it).
DATA = Path(os.environ.get("SOMAVIEW_DATA", HERE / "data"))
SOMA_PY = os.environ.get("SOMAVIEW_SOMA_PYTHON", "")
NE20 = os.environ.get("SOMAVIEW_NE20", "")


def data(name):
    p = DATA / name
    if not p.exists():
        pytest.skip(f"ground-truth file {p} not present")
    return str(p)


def oracle(path, tmp_path):
    """SOMA's own reading of ``path`` (run in a separate process; see soma_oracle.py)."""
    if not SOMA_PY or not Path(SOMA_PY, "soma").exists():
        pytest.skip("SOMA's Python package not built")
    out = tmp_path / (Path(path).stem + ".oracle.npz")
    env = dict(os.environ, PYTHONPATH=SOMA_PY, OMP_NUM_THREADS="1", OMP_PROC_BIND="false")
    r = subprocess.run([sys.executable, str(HERE / "soma_oracle.py"), str(path), str(out)], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return dict(np.load(out))
