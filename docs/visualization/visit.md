# VisIt

`tools/somaview/visit/` is a VisIt database plugin for `.soma`
files, mesh files and AMR checkpoints, tested with VisIt 3.5:

- `soma_reader.hpp`, a header-only C++17 reader over the HDF5 C API, with no VTK or VisIt in it,
  that reads what the [somaview core](overview.md#one-reading-core) reads and is tested against
  it;
- `SOMA.xml`, `avtSOMAFileFormat.h` and `avtSOMAFileFormat.C`, the plugin itself: single
  timestep, multiple domains (STMD).

## Building it

Against a VisIt binary install, with VisIt's own HDF5:

```bash
mkdir SOMA && cp tools/somaview/visit/{SOMA.xml,avtSOMAFileFormat.*,soma_reader.hpp,visit_macos.cmake} SOMA/ && cd SOMA
<visit>/bin/xml2info -clobber SOMA.xml
<visit>/bin/xml2cmake -clobber SOMA.xml
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ .
make -j
```

On macOS, add `-DCMAKE_PROJECT_INCLUDE=$PWD/visit_macos.cmake` to the `cmake` line and use
`/usr/bin/clang++`: VisIt 3.5.0's own CMake files there give HDF5 no library, misname the VTK
libraries and take whichever MPI they find, and `visit_macos.cmake` points them at VisIt's own.
Run `xml2info` first on every platform: it writes the `SOMA*PluginInfo.C` files that the generated
`CMakeLists.txt` names.

`make` installs `libISOMADatabase`, `libMSOMADatabase`, `libESOMADatabase_ser` and
`libESOMADatabase_par` into `~/.visit/<version>/<arch>/plugins/databases/`, where VisIt finds
them. Build with the compiler VisIt was built with, not one from another environment on the
`PATH`.

## Domains, meshes and variables

- A mesh file's domains are its parts. VisIt's engine hands them to its ranks; each rank reads
  only the parts it is given, with their halo found by scanning.
- The mesh of the file's own dimension is `mesh`; each lower dimension the file wrote is a mesh
  `elements_dim<D>`. A domain's meshes share one array of points.
- Global node and zone ids are given, so VisIt builds ghost zones between domains.
- A knife is given as a degenerate hexahedron, corners [0 1 2 3 4 5 6 5], which has the knife's
  faces and volume: VisIt 3.5's filters cannot take VTK polyhedra.
- A field with one row per entity is a scalar (1 component), a vector (2 or 3) or one scalar per
  component. VisIt names variables by name alone, so a name several fields share (MOAB's
  `GLOBAL_ID`, written once per dimension) is qualified by where it lives: `nodal/GLOBAL_ID`,
  `mesh/GLOBAL_ID`, `elements_dim1/GLOBAL_ID`.
- `soma_part` and `set:<name>` are zonal scalars on `mesh`.
- An AMR checkpoint's domains are its boxes over every level, grouped by level, each a
  rectilinear grid; coarse cells under finer boxes are blanked through VisIt's domain nesting.

```{figure} ../figures/img/visualization-visit.png
:alt: Six VisIt renders with VisIt's rainbow colour bars: a hex box and a tet box coloured by pressure with their meshes, five linear cell types in a row, an ocean mesh of the globe coloured by node id with continents missing, a square coloured from blue to red, and a globe coloured by cell size with a refined region over North America
:width: 100%

VisIt 3.5 on a 4-rank engine: SOMA-written hex (7 parts) and tet (4 parts) meshes by pressure; linear cells of
every kind, the knife last; an ocean mesh (3 parts) by node id; a 2D AMR checkpoint; and the CONUS 200 m mesh,
50.3 million cells, on an 8-rank engine, by grid spacing.
```

## Tests

- `visit/tests/compare_with_python.py` and `compare_amr.py` check `soma_reader.hpp`, through the
  small program `visit/tests/cpp_driver.cpp` (`visit/tests/build.sh` builds it), against the
  Python core on every test file, whole and part by part: coordinates, cell corners, fields, set
  labels and AMR boxes, in both prism conventions.
- `visit/tests/run_visit_tests.sh` builds the plugin against a VisIt install and runs
  `visit_test.py` in `visit -nowin -cli` on a serial engine and a 4-rank engine (and a large file
  on 8 ranks): the zones read and a field's range must equal the core's, and each file is
  rendered.

VisIt's `Query("NumZones", use_actual_data=1)` counts the zones after VisIt's own processing,
with the ghost zones it adds between domains and the linear pieces it cuts quadratic 2D cells
into; `use_actual_data=0` counts the zones read.
