# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""A ParaView Python reader plugin for SOMA files.

Loads with ``Tools > Manage Plugins`` or ``LoadPlugin(path)`` in pvpython/pvbatch; registers the
extension ``.soma``. A mesh file becomes a ``vtkPartitionedDataSetCollection`` (one partitioned
dataset per selected element dimension, named ``dim0``.."dimN" in a ``vtkDataAssembly``, matching
``somaview.vtkconv.to_vtk_collection``); an AMR checkpoint becomes a ``vtkOverlappingAMR``. The
output type is chosen per file in ``RequestDataObject``, from ``somaview.detect``.

Properties: a repeatable ``FileNames`` list (each name one time step: its ``time`` mesh value if
every file in the list has one, else the file's position in the list); ``PointArrayStatus`` and
``CellArrayStatus`` (point and cell fields of a mesh file); ``ElementBlocks`` (the written element
dimensions, named ``dim1``.. as in the core); ``AMRArrayStatus`` (AMR checkpoint fields);
``HighOrderCells``, ``SetFieldLabels``, ``PartIds``, ``GlobalIds``, ``AMRBlanking`` booleans.

Distributed reading follows ``UPDATE_PIECE_NUMBER``/``UPDATE_NUMBER_OF_PIECES`` from the output
information: piece k of P reads the parts ``somaview.parts_for_rank(N, P, k)`` assigns it (an
mpi4py communicator sized P does the halo exchange in one alltoall; otherwise the core scans the
sharing tables for it). AMR boxes are split across pieces by ``AmrFile.box_owner``.

This module does not import ``somaview`` until load time, so that LoadPlugin can locate the
package relative to this file (or via ``SOMAVIEW_PATH``) before the import is attempted.
"""

from __future__ import annotations

import os
import sys
import weakref
from pathlib import Path

paraview_plugin_version = "1.0"


def _locate_somaview() -> None:
    """Make ``import somaview`` succeed: it may already be on ``sys.path`` (installed, or a
    developer's PYTHONPATH); otherwise try ``SOMAVIEW_PATH`` and ``<this file>/../src``, the
    layout this plugin ships in."""
    try:
        import somaview  # noqa: F401
        # A directory named somaview with no __init__.py (this repository's own root, when a
        # script beside it puts its parent on sys.path) imports as an empty namespace package
        # that would shadow the real one: discard it and keep looking.
        if getattr(somaview, "__file__", None):
            return
        for name in [m for m in sys.modules if m == "somaview" or m.startswith("somaview.")]:
            del sys.modules[name]
    except ImportError:
        pass
    here = Path(__file__).resolve().parent
    candidates = []
    env = os.environ.get("SOMAVIEW_PATH")
    if env:
        candidates.append(Path(env))
    candidates.append(here.parent / "src")
    for c in reversed(candidates):
        if c.exists():
            if str(c) in sys.path:
                sys.path.remove(str(c))
            sys.path.insert(0, str(c))
    import somaview  # noqa: F401  (raises ImportError with a clear message if still missing)


_locate_somaview()

import numpy as np  # noqa: E402
from vtkmodules.vtkCommonCore import vtkDataArraySelection  # noqa: E402
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkOverlappingAMR, vtkPartitionedDataSetCollection  # noqa: E402,E501

from paraview.util.vtkAlgorithm import VTKPythonAlgorithmBase, smdomain, smhint, smproperty, smproxy  # noqa: E402

import somaview as sv  # noqa: E402
from somaview.amr import AmrFile, to_vtk_amr  # noqa: E402
from somaview.format import FileKind  # noqa: E402
from somaview.vtkconv import to_vtk_collection  # noqa: E402


def _modified_callback(obj):
    """A callback for ``vtkDataArraySelection.AddObserver`` that marks ``obj`` modified without
    keeping it alive through the observer (as ``paraview.algorithms.eam_reader`` does)."""
    ref = weakref.ref(obj)

    def _cb(*_args):
        o = ref()
        if o is not None:
            o.Modified()

    return _cb


@smproxy.reader(name="SomaReader", label="SomaReader", extensions="soma",
                file_description="SOMA mesh or AMR checkpoint file")
class SomaReader(VTKPythonAlgorithmBase):
    """Reads a SOMA mesh file or AMR checkpoint (or a file series of either).

    A mesh file's output is a ``vtkPartitionedDataSetCollection``, one partitioned dataset per
    selected element dimension. An AMR checkpoint's output is a ``vtkOverlappingAMR``. The kind
    is read from the file itself (``somaview.detect``); ``RequestDataObject`` creates whichever
    output type the current file needs.
    """

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=0, nOutputPorts=1, outputType=None)
        self._filenames: list[str] = []
        self._metadata_key = None
        self._kind: FileKind | None = None
        self._mesh_dim = None
        self._time_values: list[float] = []

        self._point_sel = vtkDataArraySelection()
        self._cell_sel = vtkDataArraySelection()
        self._dim_sel = vtkDataArraySelection()
        self._amr_sel = vtkDataArraySelection()
        for sel in (self._point_sel, self._cell_sel, self._dim_sel, self._amr_sel):
            sel.AddObserver("ModifiedEvent", _modified_callback(self))

        self._high_order = True
        self._set_labels = True
        self._part_ids = True
        self._global_ids = True
        self._amr_blanking = True

    # -- FileNames: a repeatable file list, one entry per time step -----------------------------
    @smproperty.stringvector(name="FileNames", repeat_command="1", clean_command="RemoveAllFileNames",
                             number_of_elements_per_command="1", animateable="0", default_values="")
    @smdomain.filelist()
    @smhint.filechooser(extensions="soma", file_description="SOMA mesh or AMR checkpoint file")
    def AddFileName(self, name: str) -> None:
        if name and name != "None" and name not in self._filenames:
            self._filenames.append(name)
            self.Modified()

    def RemoveAllFileNames(self) -> None:
        if self._filenames:
            self._filenames = []
            self.Modified()

    # -- array and element-dimension selection ---------------------------------------------------
    @smproperty.dataarrayselection(name="PointArrayStatus")
    def GetPointArrayStatus(self):
        return self._point_sel

    @smproperty.dataarrayselection(name="CellArrayStatus")
    def GetCellArrayStatus(self):
        return self._cell_sel

    @smproperty.dataarrayselection(name="ElementBlocks")
    def GetElementBlocks(self):
        return self._dim_sel

    @smproperty.dataarrayselection(name="AMRArrayStatus")
    def GetAMRArrayStatus(self):
        return self._amr_sel

    # -- toggles (int vector + boolean domain, the usual ParaView idiom for a checkbox) ---------
    @smproperty.intvector(name="HighOrderCells", default_values=1)
    @smdomain.boolean()
    def SetHighOrderCells(self, value) -> None:
        v = bool(value)
        if v != self._high_order:
            self._high_order = v
            self.Modified()

    @smproperty.intvector(name="SetFieldLabels", default_values=1)
    @smdomain.boolean()
    def SetSetFieldLabels(self, value) -> None:
        v = bool(value)
        if v != self._set_labels:
            self._set_labels = v
            self.Modified()

    @smproperty.intvector(name="PartIds", default_values=1)
    @smdomain.boolean()
    def SetPartIds(self, value) -> None:
        v = bool(value)
        if v != self._part_ids:
            self._part_ids = v
            self.Modified()

    @smproperty.intvector(name="GlobalIds", default_values=1)
    @smdomain.boolean()
    def SetGlobalIds(self, value) -> None:
        v = bool(value)
        if v != self._global_ids:
            self._global_ids = v
            self.Modified()

    @smproperty.intvector(name="AMRBlanking", default_values=1)
    @smdomain.boolean()
    def SetAMRBlanking(self, value) -> None:
        v = bool(value)
        if v != self._amr_blanking:
            self._amr_blanking = v
            self.Modified()

    # -- time steps (exposed to the animation UI; RequestInformation sets the executive key too) -
    @smproperty.xml("""
        <DoubleVectorProperty name="TimestepValues"
                              repeatable="1"
                              information_only="1">
          <TimeStepsInformationHelper/>
        </DoubleVectorProperty>""")
    def GetTimestepValues(self):
        return self._time_values

    # -- metadata: read once per distinct file list, cheaply (attributes and tables only) -------
    def _refresh_metadata(self) -> None:
        key = tuple(self._filenames)
        if key == self._metadata_key:
            return
        self._metadata_key = key
        self._kind = None
        self._mesh_dim = None
        self._time_values = []
        for sel in (self._point_sel, self._cell_sel, self._dim_sel, self._amr_sel):
            sel.RemoveAllArrays()
        if not self._filenames:
            return
        path0 = self._filenames[0]
        kind = sv.detect(path0)
        if kind is None:
            raise ValueError(f"{path0}: not a SOMA file")
        self._kind = kind
        if kind == FileKind.mesh:
            self._refresh_mesh_metadata(path0)
        else:
            self._refresh_amr_metadata(path0)

    def _refresh_mesh_metadata(self, path0: str) -> None:
        with sv.SomaFile(path0) as sf:
            self._mesh_dim = sf.dim
            elem_dims = sorted(d for d in sf.written_dims if d > 0)
            for d in elem_dims:
                name = f"dim{d}"
                self._dim_sel.AddArray(name)
                if d == sf.dim:
                    self._dim_sel.EnableArray(name)
                else:
                    self._dim_sel.DisableArray(name)
            for f in sf.fields_on(0):
                self._point_sel.AddArray(f.name)
                self._point_sel.EnableArray(f.name)
            cell_names = sorted({f.name for d in elem_dims for f in sf.fields_on(d)})
            for name in cell_names:
                self._cell_sel.AddArray(name)
                self._cell_sel.EnableArray(name)
        self._time_values = self._time_steps_for(self._read_mesh_time)

    def _refresh_amr_metadata(self, path0: str) -> None:
        with AmrFile(path0) as af:
            for f in af.fields:
                self._amr_sel.AddArray(f.name)
                self._amr_sel.EnableArray(f.name)
        self._time_values = self._time_steps_for(lambda _p: None)  # AMR checkpoints have no time value

    def _read_mesh_time(self, path: str) -> float | None:
        with sv.SomaFile(path) as sf:
            return sf.time()

    def _time_steps_for(self, time_of) -> list[float]:
        """One time step per file: ``time_of(path)`` for every file if every file has one, else
        the file's position in the list. A single file gets no time steps at all."""
        if len(self._filenames) <= 1:
            return []
        times = [time_of(p) for p in self._filenames]
        if all(t is not None for t in times):
            return [float(t) for t in times]
        return [float(i) for i in range(len(self._filenames))]

    # -- array-selection helpers ------------------------------------------------------------------
    @staticmethod
    def _enabled(sel: vtkDataArraySelection) -> list[str]:
        names = [sel.GetArrayName(i) for i in range(sel.GetNumberOfArrays())]
        return [n for n in names if sel.ArrayIsEnabled(n)]

    # -- the vtkAlgorithm pipeline passes --------------------------------------------------------
    def FillOutputPortInformation(self, port, info):
        # The output type (vtkPartitionedDataSetCollection or vtkOverlappingAMR) is decided per
        # file in RequestDataObject; VTKPythonAlgorithmBase's default implementation looks up
        # self.OutputType, which is None here. Their common ancestor, vtkCompositeDataSet, is
        # abstract (vtkDataObjectTypes.NewDataObject cannot instantiate it, which the pipeline
        # treats as fatal), so the port is declared with vtkPartitionedDataSetCollection, a
        # concrete placeholder RequestDataObject is free to replace with a vtkOverlappingAMR.
        info.Set(vtkDataObject.DATA_TYPE_NAME(), "vtkPartitionedDataSetCollection")
        return 1

    def RequestDataObject(self, request, inInfo, outInfo):
        self._refresh_metadata()
        if self._kind is None:
            return 1
        info = outInfo.GetInformationObject(0)
        wanted = vtkOverlappingAMR if self._kind == FileKind.amr else vtkPartitionedDataSetCollection
        current = info.Get(vtkDataObject.DATA_OBJECT()) if info.Has(vtkDataObject.DATA_OBJECT()) else None
        if current is None or not current.IsA(wanted.__name__):
            info.Set(vtkDataObject.DATA_OBJECT(), wanted())
        return 1

    def RequestInformation(self, request, inInfo, outInfo):
        self._refresh_metadata()
        info = outInfo.GetInformationObject(0)
        executive = self.GetExecutive()
        info.Remove(executive.TIME_STEPS())
        info.Remove(executive.TIME_RANGE())
        if self._time_values:
            for t in self._time_values:
                info.Append(executive.TIME_STEPS(), t)
            info.Append(executive.TIME_RANGE(), self._time_values[0])
            info.Append(executive.TIME_RANGE(), self._time_values[-1])
        # vtkPVCompositeDataPipeline (composite outputs: our vtkPartitionedDataSetCollection and
        # vtkOverlappingAMR) does not have CAN_HANDLE_PIECE_REQUEST -- it always forwards
        # UPDATE_PIECE_NUMBER/UPDATE_NUMBER_OF_PIECES to the algorithm, unlike
        # vtkStreamingDemandDrivenPipeline (plain vtkDataSet outputs), where this key is needed.
        if hasattr(executive, "CAN_HANDLE_PIECE_REQUEST"):
            info.Set(executive.CAN_HANDLE_PIECE_REQUEST(), 1)
        return 1

    def _time_index(self, outInfo) -> int:
        executive = self.GetExecutive()
        info = outInfo.GetInformationObject(0)
        if len(self._time_values) > 1 and info.Has(executive.UPDATE_TIME_STEP()):
            t = info.Get(executive.UPDATE_TIME_STEP())
            return min(range(len(self._time_values)), key=lambda i: abs(self._time_values[i] - t))
        return 0

    def RequestData(self, request, inInfo, outInfo):
        self._refresh_metadata()
        if not self._filenames or self._kind is None:
            return 1
        info = outInfo.GetInformationObject(0)
        executive = self.GetExecutive()
        piece = info.Get(executive.UPDATE_PIECE_NUMBER()) if info.Has(executive.UPDATE_PIECE_NUMBER()) else 0
        npieces = (info.Get(executive.UPDATE_NUMBER_OF_PIECES())
                  if info.Has(executive.UPDATE_NUMBER_OF_PIECES()) else 1)
        idx = self._time_index(outInfo)
        path = self._filenames[idx if idx < len(self._filenames) else 0]
        out = self.GetOutputData(outInfo, 0)
        if self._kind == FileKind.mesh:
            self._request_mesh(path, piece, max(npieces, 1), out)
        else:
            self._request_amr(path, piece, max(npieces, 1), out)
        return 1

    def _request_mesh(self, path: str, piece: int, npieces: int, out) -> None:
        dims = sorted(int(name[3:]) for name in self._enabled(self._dim_sel))
        if not dims:
            dims = [self._mesh_dim]
        point_fields = self._enabled(self._point_sel)
        cell_fields = self._enabled(self._cell_sel)
        comm = self._mpi_comm_for(npieces)
        kw = dict(dims=dims, point_fields=point_fields, cell_fields=cell_fields,
                 high_order=self._high_order, set_fields=bool(self._set_labels))
        if comm is not None:
            piece_obj = sv.read(path, comm=comm, **kw)
        else:
            piece_obj = sv.read(path, rank=piece, nranks=npieces, **kw)
        collection = to_vtk_collection(piece_obj, gids=self._global_ids, part=self._part_ids)
        out.ShallowCopy(collection)

    def _request_amr(self, path: str, piece: int, npieces: int, out) -> None:
        names = self._enabled(self._amr_sel)
        with AmrFile(path) as af:
            amr = to_vtk_amr(af, fields=names, rank=piece, nranks=npieces, blank=self._amr_blanking)
        out.ShallowCopy(amr)

    @staticmethod
    def _mpi_comm_for(npieces: int):
        """``MPI.COMM_WORLD`` if it is running with exactly ``npieces`` processes (so
        ``somaview.read``'s halo exchange, not the sharing-table scan, finds the halo); None
        under pvpython/pvbatch --no-mpi, where every piece is read in this one process.

        Only looks at ``mpi4py`` if it is already imported: importing it fresh calls
        ``MPI_Init`` as a side effect (mpi4py's default ``rc.initialize``), which aborts the
        whole process on a host where MPI cannot open its own sockets (such as any
        single-process pvpython/pvbatch run not launched under mpiexec).
        """
        if "mpi4py" not in sys.modules or "mpi4py.MPI" not in sys.modules:
            return None
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm if comm.Get_size() == npieces else None
