# pyvista, yt and VTKHDF

The [somaview core](overview.md#one-reading-core) also gives SOMA files to pyvista and yt in a
Python session, a command line to look at and convert files, and an exporter to VTKHDF, the HDF5
format VTK and ParaView read natively. Install with `pip install -e tools/somaview[all]`.

## pyvista

```python
import somaview as sv

sv.plot("mesh.soma", scalars="pressure", clip="x", screenshot="out.png")
sv.plot("checkpoint.soma", scalars="phi", box_outlines=True)   # an AMR checkpoint
grid = sv.read_pyvista("mesh.soma", parts=[0, 1])               # a pyvista.UnstructuredGrid
```

`plot` draws the outer surface of a volume mesh, so a mesh of millions of cells stays interactive,
and clips the whole volume first when asked, so the cut face shows the field. Integer labels
(`soma_part`, `set:<name>`) get one colour each. A name that is no point or cell array is an error
listing the arrays there are. For an AMR checkpoint, `somaview.plot.to_pyvista_amr` gives the
hierarchy as a `pyvista.MultiBlock` (`level<l>` then `box<b>`) without the cells finer boxes
cover, and `somaview.plot.amr_box_outlines` the boxes as lines. `plot` renders offscreen when
given `screenshot`, which needs an OpenGL context: a GPU, EGL or OSMesa.

## yt

```python
import somaview.yt as svyt

ds = svyt.load("checkpoint.soma")   # an AMR checkpoint: yt.load_amr_grids
ds = svyt.load("mesh.soma")         # a mesh file: yt.load_unstructured_mesh
```

An AMR checkpoint's cell-centred fields become yt grid fields as they are, node-centred ones are
averaged to the cells (`ds.parameters["averaged_from_nodal"]`), and a vector field is one yt field
per component, `<field>_0`, `<field>_1`, ... A mesh file becomes one yt mesh per kind of linear
cell yt supports (tetrahedra, pyramids, prisms, hexahedra); other cells are counted in
`ds.parameters["skipped_cells"]`.

```{figure} ../figures/img/visualization-pyvista-yt.png
:alt: Five pyvista renders and one yt slice: a hex box coloured by pressure, a cube of 64 bricks by material set, curved quadratic cells with edges, five linear cell types, a square coloured by phi with AMR box outlines, and a yt slice of a 3D AMR field with three round features
:width: 100%

pyvista, offscreen with EGL, and yt: a hex mesh by pressure; the 64-brick MOAB mesh by material set; quadratic
cells drawn curved; linear cells of every kind; a 2D AMR checkpoint by `phi`, with its boxes; a yt slice of a 3D
AMR checkpoint's `u`. `tools/somaview/examples/gallery.py` makes them.
```

## The command line

```bash
somaview info mesh.soma                               # parts, dimensions, fields; AMR levels
somaview render mesh.soma -s pressure -o out.png --clip x
somaview convert mesh.soma out.vtkhdf                 # or out.vtu, out.pvtu
```

## VTKHDF files

```python
from somaview.vtkhdf import write_vtkhdf

write_vtkhdf("mesh.soma", "mesh.vtkhdf")                            # one partition per part
from mpi4py import MPI
write_vtkhdf("mesh.soma", "mesh.vtkhdf", comm=MPI.COMM_WORLD)       # under mpiexec, collectively
```

The output is a partitioned VTKHDF UnstructuredGrid (format version 2.2), one partition per SOMA
part, each holding its halo, so VTK's `vtkHDFReader` reads one partition per rank without
communication. As the format requires, each partition's `Connectivity` indexes its own points and
its `Offsets` start at 0. Knives become convex point sets, since the VTKHDF polyhedron extension
is newer than VTK 9.5 and ParaView 6.1. With `comm=`, every rank writes its parts' partitions into
one file through h5py's `mpio` driver, which needs an h5py built against parallel HDF5; every rank
creates every dataset and attribute, since HDF5 metadata operations are collective.
