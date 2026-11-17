# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""SOMA: distributed unstructured meshes, fields and remapping, from Python.

Run scripts under ``mpiexec``; pass mpi4py communicators as ``soma.Comm(MPI.COMM_WORLD)``.
Arrays returned by meshes share SOMA's memory (no copies) unless documented otherwise.
"""

import atexit as _atexit
import gc as _gc

try:  # MPI first: mpi4py initializes MPI; SOMA never does from Python.
    from mpi4py import MPI as _MPI
except ImportError:  # a build without MPI needs no mpi4py
    _MPI = None

from . import _soma
from ._soma import *  # noqa: F401,F403

has_mpi = _soma.has_mpi
Error = _soma.Error


def _check_mpi(soma_version, mpi4py_version):
    """Raises ImportError unless SOMA and mpi4py call the same MPI library."""
    if soma_version != mpi4py_version:
        raise ImportError(
            "soma and mpi4py use different MPI libraries; build mpi4py against SOMA's MPI\n"
            f"  soma:   {soma_version.splitlines()[0] if soma_version else ''}\n"
            f"  mpi4py: {mpi4py_version.splitlines()[0] if mpi4py_version else ''}"
        )


if has_mpi:
    if _MPI is None:
        raise ImportError("this soma was built with MPI and needs mpi4py")
    _check_mpi(_soma.mpi_library_version(), _MPI.Get_library_version())


def _at_exit():
    _gc.collect()
    _soma._finalize()


_atexit.register(_at_exit)


from .vis import to_pyvista  # noqa: E402
