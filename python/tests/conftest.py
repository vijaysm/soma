# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Helpers shared by the tests: the world communicator and reductions, with or without MPI."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import soma  # noqa: E402

if soma.has_mpi:
    from mpi4py import MPI


def comm_world():
    return soma.Comm(MPI.COMM_WORLD) if soma.has_mpi else soma.Comm.self()


def allsum(x):
    return MPI.COMM_WORLD.allreduce(x) if soma.has_mpi else x


def bcast(x):
    return MPI.COMM_WORLD.bcast(x, root=0) if soma.has_mpi else x


@pytest.fixture
def shared_tmp(tmp_path):
    """A directory every rank names alike: rank 0's tmp_path."""
    return Path(bcast(str(tmp_path)))
