# SOMA reader plugin for ParaView

`SomaReader.py` is a single-file ParaView Python reader plugin for SOMA files (`.soma`):
mesh files and AMR checkpoints, over the somaview core in `../src`. Its user documentation is
SOMA's `docs/visualization/paraview.md`; this file keeps the details and implementation notes.

## Install

Load it from `Tools > Manage Plugins > Load New...` in the ParaView GUI, or from a script:

```python
from paraview.simple import LoadPlugin
LoadPlugin("/path/to/somaview/paraview/SomaReader.py", remote=False, ns=globals())
```

(`remote=True`, the default, loads it on the server side too -- use that when the data is on a
remote `pvserver`.) The plugin needs the `somaview` core package. It tries `import somaview`
first (so an installed `pip install somaview` or a `PYTHONPATH` entry is used if present);
otherwise it inserts `<plugin dir>/../src` (the layout this package ships in) and the directory
named by the environment variable `SOMAVIEW_PATH`, in that order, before trying the import again.
Under `pvserver`, the plugin's Python runs on every server rank, so `somaview` (or
`SOMAVIEW_PATH`) must be reachable from each of them.

## Use

Once loaded, the reader is registered under the name `SomaReader` (extension `.soma`):

```python
from paraview.simple import SomaReader, Show, ColorBy, Render, SaveScreenshot

reader = SomaReader(FileNames=["mesh.soma"])
reader.ElementBlocks = ["dim3"]          # which element dimensions to read (default: the mesh's own)
reader.PointArrayStatus = ["pressure"]   # point fields to read (default: all)
reader.CellArrayStatus = []              # cell fields to read (default: all)
reader.HighOrderCells = 1                # quadratic VTK cells for high-order elements
reader.SetFieldLabels = 1                # cell arrays "set:<name>" from entity-set fields
reader.PartIds = 1                       # cell array "soma_part"
reader.GlobalIds = 1                     # GlobalIds point/cell arrays from the gids
reader.CloseHangingCells = 1             # close cells at hanging vertices (no GlobalIds in 3D then)
reader.NodalResolution = 0               # gll_p<order> blocks: 0 the GLL points, n > 0 n x n sub-quads
reader.BoundedInterpolation = 1          # ... kept within the range of the element's own values
reader.LagrangeCells = 0                 # gll_p<order> blocks: VTK Lagrange cells instead
reader.UpdatePipeline()
```

A mesh file's output is a `vtkPartitionedDataSetCollection`, one partitioned dataset per
selected element dimension, named `dim0`, `dim1`, ... (matching `somaview.vtkconv`'s own
naming) in a `vtkDataAssembly`. An AMR checkpoint's output is a `vtkOverlappingAMR`; its field
selection is the separate `AMRArrayStatus` property, `AMRBlanking` toggles coarse-cell
blanking under finer boxes, and `AMRNodeValues` (on) adds `<field>:smooth`, node values continuous across
boxes and levels; `AMRClosed` (off) gives one unstructured mesh, closed where levels meet, instead of the boxes. Spectral-element fields are point data of blocks `gll_p<order>`, read by default in place
of `dim2`. The kind is read from the file itself (`somaview.detect`); loading a
different file on the same reader instance switches the output type if needed.

### File series (time steps)

`FileNames` is a plain, repeatable list, not tied to any filename pattern: give it the paths of
a time series in order and each becomes one time step.

```python
reader = SomaReader(FileNames=["run_0000.soma", "run_0001.soma", "run_0002.soma"])
```

The time value of step *i* is the mesh value named `time` (any case) of file *i*, if every file
in the list has one; otherwise it is just *i*. AMR checkpoints have no `time` mesh value, so a
series of them always uses the file index.

### Distributed reading

The reader honours `UPDATE_PIECE_NUMBER`/`UPDATE_NUMBER_OF_PIECES` from the output information:
piece *k* of *P* reads the parts `somaview.parts_for_rank(N, P, k)` assigns it (SOMA's own
part-to-rank rule). Under `pvserver`/`pvbatch` with `mpiexec -n P`, if `mpi4py` is already
imported in that server process, the reader passes an `mpi4py.MPI.COMM_WORLD` sized *P* to
`somaview.read`, which exchanges the halo in one `alltoall`; otherwise it passes `rank`/`nranks`
and the core finds the halo by scanning the other parts' sharing tables. AMR boxes are split
across pieces the same way, by `AmrFile.box_owner`. The reader never imports `mpi4py` itself
unless `mpi4py.MPI` is already in `sys.modules`: importing it fresh calls `MPI_Init`, which
aborts a process that was not launched under `mpiexec` (see **Findings** below).

## `examples/render_soma.py`

A `pvbatch`-ready script: loads the plugin, reads one file, extracts the surface, colors it by
a named array, and saves a screenshot.

```
pvbatch --mpi examples/render_soma.py path/to/file.soma array_name output.png
mpiexec -n 4 pvbatch --mpi examples/render_soma.py path/to/file.soma array_name output.png
```

See the note at the end of that file for running it against a real `pvserver` under MPI
(`Connect`, and where the plugin and `somaview` need to be reachable).

## Gallery

`examples/pv_gallery.py` renders with ParaView's own renderer under `pvbatch` (offscreen, EGL),
with IceT compositing when run on several MPI ranks:

```
mpiexec -n 4 pvbatch examples/pv_gallery.py SomaReader.py out_dir \
    hexmpi=hex_4p.soma bricks=bricks_4p.soma highorder=highorder3d.soma amr=amr3d_4p.soma
mpiexec -n 8 pvbatch examples/pv_gallery.py SomaReader.py out_dir conus=conus.soma
```

## Tests

`tests/test_paraview.py` (run with `pvpython --dr --no-mpi tests/test_paraview.py`) checks the
reader's data, not pixels: counts, arrays, values against the core, pieces, time steps, AMR. All
checks pass with ParaView 6.1.0 (macOS) and 6.1.1 (Linux). Under MPI, `pvbatch` on 1, 2, 4 and 7
ranks reading SOMA-MPI-written files (hex at 4 and 7 parts, quadrilaterals at 7, MOAB bricks at
4, an ocean mesh at 3) covers every cell exactly once with correct coordinates.

## Findings

- **A directory named `somaview` can shadow the package.** A script run from the directory that
  holds this repository puts it on `sys.path`, and the repository root (no `__init__.py`)
  imports as an empty namespace package; `from somaview.amr import ...` then fails. The plugin
  discards such a namespace package and inserts `<plugin dir>/../src` first.
- **`ProcessIds` (ParaView >= 5.11) generates point data only by default** (`PointProcessIds`);
  set `GenerateCellData = 1` to color cells by `CellProcessIds`.
- **Composite output types need a concrete `FillOutputPortInformation`.** `VTKPythonAlgorithmBase`
 with `outputType=None` (needed here, since the output type is only known once a file's kind
 is detected) falls over if the declared port type is `vtkCompositeDataSet`: that class is
 abstract, and `vtkDataObjectTypes.NewDataObject` refuses to instantiate it, which the pipeline
 treats as fatal (not a catchable Python exception -- it aborts the whole process). Declaring
 `vtkPartitionedDataSetCollection` instead (concrete, and `RequestDataObject` is free to replace
 it with a `vtkOverlappingAMR` when needed) fixed it; see `FillOutputPortInformation` in
 `SomaReader.py`.
- **`vtkPVCompositeDataPipeline` has no `CAN_HANDLE_PIECE_REQUEST`.** That executive key, the
 usual way a reader tells `vtkStreamingDemandDrivenPipeline` it will do its own piece slicing,
 does not exist on the composite-data executive ParaView uses for `vtkPartitionedDataSetCollection`
 /`vtkOverlappingAMR` outputs (`vtkPVCompositeDataPipeline`). Setting it unconditionally raises
 `AttributeError`; `RequestInformation` now guards it with `hasattr`. The composite pipeline
 forwards `UPDATE_PIECE_NUMBER`/`UPDATE_NUMBER_OF_PIECES` to the algorithm either way, so piece
 requests still work (verified: P = 1, 2, 4, 7 on the real ne20 file partition its 581522 cells
 exactly, no gaps or overlaps).
- **Importing `mpi4py` fresh calls `MPI_Init`.** `mpi4py`'s default `rc.initialize = True` means
 `from mpi4py import MPI` initializes MPI as a side effect, even just to check
 `COMM_WORLD.Get_size()`. On a single `pvpython`/`pvbatch` process not launched under `mpiexec`
 (any one-process use of this plugin), that call
 aborts the entire process with an unrecoverable `MPI_Init_thread` failure -- not a Python
 exception, so there is nothing to catch. The reader now only looks at `mpi4py.MPI` when it is
 already in `sys.modules` (i.e. something else, such as a real MPI job launcher, imported it
 first); otherwise it falls back to the sharing-table scan path unconditionally.
- A default value is pushed to every property when a ParaView proxy is created, including
 repeat-command ones; leaving `default_values` unset on the `FileNames` property made ParaView
 generate `default_values="None"` in the property XML and call `AddFileName("None")` once at
 construction. Giving it an explicit `default_values=""` (and guarding `AddFileName` against an
 empty or literal `"None"` name) avoided a spurious file-not-found error on every fresh reader.
