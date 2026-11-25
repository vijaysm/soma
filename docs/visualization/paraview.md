# ParaView

`tools/somaview/paraview/SomaReader.py` is a
ParaView Python reader plugin for `.soma` files, mesh files and AMR checkpoints alike, over the
[somaview core](overview.md#one-reading-core). It is tested with ParaView 6.1.

## Loading it

From the GUI: `Tools > Manage Plugins > Load New...` and choose `SomaReader.py`. From a script:

```python
from paraview.simple import LoadPlugin
LoadPlugin("tools/somaview/paraview/SomaReader.py", remote=False, ns=globals())
```

`remote=True` (the default) loads it on a `pvserver` too, as a remote session needs. The plugin
finds the core by `import somaview`, else in `../src` beside it, else in the directory
`SOMAVIEW_PATH` names; under `pvserver` every server rank needs it. ParaView ships numpy; h5py
must be importable by ParaView's Python as well.

## Reading

```python
from paraview.simple import SomaReader

reader = SomaReader(FileNames=["mesh.soma"])
reader.ElementBlocks = ["dim3"]          # element dimensions to read (default: the mesh's own, or gll_p<order>)
reader.PointArrayStatus = ["pressure"]   # point fields (default: all)
reader.CellArrayStatus = []              # cell fields (default: all)
reader.HighOrderCells = 1                # quadratic VTK cells for high-order elements
reader.SetFieldLabels = 1                # cell arrays set:<name> from entity-set fields
reader.PartIds = 1                       # cell array soma_part
reader.GlobalIds = 1                     # GlobalIds from the gids
reader.CloseHangingCells = 1             # close cells at hanging vertices
reader.NodalResolution = 0               # spectral-element fields: 0 the GLL points, n > 0 n x n sub-quads
reader.BoundedInterpolation = 1          # ... kept within the range of the element's own values
reader.LagrangeCells = 0                 # spectral-element fields on VTK Lagrange cells instead
reader.UpdatePipeline()
```

A mesh file gives a `vtkPartitionedDataSetCollection`, one partitioned data set per element
dimension read (`dim0`, `dim1`, ...). An AMR checkpoint gives a `vtkOverlappingAMR`, its fields
chosen by `AMRArrayStatus`, with `AMRBlanking` to blank the coarse cells under finer boxes and
`AMRNodeValues` to add `<field>:smooth`, [node values](overview.md#block-amr) to contour; with `AMRClosed` it gives one unstructured mesh instead, whose
3D contours have no seams between levels. A file with
[spectral-element fields](overview.md#between-the-nodes) has a block `gll_p<order>` per order, read by
default in place of the quads under it; the fields are chosen by `PointArrayStatus`. The
kind is read from the file. `FileNames` is a list: several files are the steps of a time series,
at the mesh value `time` of each file when they all have one, else at 0, 1, 2, ...

## In parallel

The reader answers ParaView's piece requests: piece $k$ of $P$ reads the parts SOMA assigns to
rank $k$ of $P$. Under `pvbatch` or `pvserver` run by `mpiexec`, when mpi4py is already imported in
the server, the halo comes by one all-to-all over `MPI.COMM_WORLD`; otherwise each piece scans the
sharing tables. (The reader never imports mpi4py itself: a fresh import starts MPI, which aborts a
process not run under `mpiexec`.) IceT composites the image.

```bash
mpiexec -n 16 pvbatch tools/somaview/paraview/examples/render_soma.py mesh.soma pressure out.png
```

`paraview/examples/pv_gallery.py` renders the figure below on 4 ranks (8 for the large mesh):

```bash
mpiexec -n 4 pvbatch tools/somaview/paraview/examples/pv_gallery.py \
  tools/somaview/paraview/SomaReader.py out_dir hexmpi=hex_4p.soma bricks=bricks_4p.soma \
  highorder=highorder3d.soma amr=amr3d_4p.soma
```

```{figure} ../figures/img/visualization-paraview.png
:alt: Six ParaView renders: a hex box in four horizontal slabs coloured by reading rank; the same box clipped, coloured by pressure with velocity arrows; a cube of 64 bricks coloured by material set; five curved quadratic cells; a slice of a 3D AMR hierarchy with its boxes; and a sphere coloured by cell size with a dark refined region over North America
:width: 100%

ParaView 6.1.1 `pvbatch` on Linux with EGL. Top: a hex mesh written by SOMA on 4 ranks, coloured by the rank that
read each cell; the same mesh clipped, with pressure and velocity glyphs; the 64-brick MOAB mesh by material set.
Bottom: quadratic cells (tet10 to hex27) drawn curved; a slice of a 3D AMR checkpoint; the 200 m CONUS mesh,
50.3 million cells read by 8 ranks, coloured by grid spacing.
```

The [CONUS mesh's figures](../apps/conus-200m.md#looking-at-the-mesh) are made the same way, on
16 ranks.

## Offscreen rendering

- Run `pvbatch` from the EGL (GPU) or OSMesa build of ParaView, with `DISPLAY` unset: with a
  stale `DISPLAY` some ranks try X and render nothing.
- Merge many small sources (the lines of a graticule, say) into one with `AppendGeometry` before
  showing them: one actor each exhausts an offscreen context's texture units, and the view comes
  out a single colour.
- `ProcessIds` gives point data only unless `GenerateCellData = 1`; colour cells by the reading
  rank with its `CellProcessIds`.
- Colours drawn unlit (`Ambient = 1`, `Diffuse = 0`) match a colour key drawn elsewhere exactly.

## Tests

`paraview/tests/test_paraview.py` (`pvpython --dr --no-mpi paraview/tests/test_paraview.py`) checks
the reader's data against the core: counts, arrays and values on every test file, the default
selections, pieces, time series and AMR. `tests/mpi/pv_mpi_check.py` checks under
`mpiexec pvbatch` that the pieces cover every cell exactly once with the right coordinates, at 1,
2, 4 and 7 ranks on files written by 3 to 7 ranks. `tests/test_paraview_reader.py` checks the
spectral-element blocks, closed cells and AMR node values by pytest, through VTK's own Python algorithm
class without ParaView.
