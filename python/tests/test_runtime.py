# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""The module's runtime: communicators, the MPI library check, errors, and exit."""

import os
import subprocess
import sys

import pytest

import soma

MPI = pytest.importorskip("mpi4py.MPI") if soma.has_mpi else None


@pytest.mark.skipif(not soma.has_mpi, reason="a build without MPI")
def test_comm_matches_mpi4py():
    c = soma.Comm(MPI.COMM_WORLD)
    assert c.rank == MPI.COMM_WORLD.Get_rank()
    assert c.size == MPI.COMM_WORLD.Get_size()
    assert soma.Comm.self().size == 1
    w = soma.Comm.world()
    assert (w.rank, w.size) == (c.rank, c.size)


def test_comm_rejects_non_communicators():
    with pytest.raises(TypeError):
        soma.Comm(3)
    with pytest.raises(TypeError):
        soma.Comm("x")


@pytest.mark.skipif(not soma.has_mpi, reason="a build without MPI")
def test_mpi_library_is_shared():
    assert soma._soma.mpi_library_version() == MPI.Get_library_version()
    with pytest.raises(ImportError):
        soma._check_mpi("a", "b")
    soma._check_mpi("a", "a")


@pytest.mark.parametrize(
    "kind, error",
    [("invalid_argument", ValueError), ("out_of_range", IndexError), ("runtime_error", soma.Error)],
)
def test_errors_map(kind, error):
    with pytest.raises(error, match="soma test error: " + kind):
        soma._soma.testing_raise(kind)
    assert issubclass(soma.Error, RuntimeError)


def test_exit_with_live_object_is_clean():
    if soma.has_mpi and MPI.COMM_WORLD.Get_rank() != 0:
        pytest.skip("runs once, on rank 0")
    # A fresh interpreter, outside mpiexec: MPI starts as a singleton.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "HYDRA_", "MPIR_"))}
    code = "import soma\nkeep = soma._soma.testing_live_object()\nassert soma._soma.live_objects() >= 1\n"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=50)
    assert r.returncode == 0, r.stderr
    assert r.stderr == ""


@pytest.mark.skipif(not soma.has_mpi, reason="a build without MPI")
def test_import_failure_exits_cleanly():
    if MPI.COMM_WORLD.Get_rank() != 0:
        pytest.skip("runs once, on rank 0")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "HYDRA_", "MPIR_"))}
    code = (
        "import mpi4py.MPI as M\n"
        "M.Get_library_version = lambda: 'another MPI'\n"
        "try:\n"
        "    import soma\n"
        "except ImportError as e:\n"
        "    assert 'different MPI libraries' in str(e)\n"
        "else:\n"
        "    raise SystemExit('imported')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=50)
    assert r.returncode == 0, r.stderr
    assert "terminating" not in r.stderr
