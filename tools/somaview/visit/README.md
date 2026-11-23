# SOMA VisIt database plugin

Reads SOMA mesh files and AMR checkpoints into VisIt (built and run against VisIt 3.5.0, Linux
x86_64, serial and parallel engines). Two layers:

- `soma_reader.hpp`: a header-only C++17 reader over the HDF5 C API, with no VTK or VisIt
  dependency. It mirrors somaview's Python core (`format.py`, `cells.py`, `mesh.py`, `amr.py`):
  the same part table, field and layout handling, VTK cell-type mapping (prism corner order by
  VTK version, every quadratic cell, polygon, convex point set), halo by scan, and entity sets as
  cell labels.
- `SOMA.xml`, `avtSOMAFileFormat.h`, `avtSOMAFileFormat.C`: the STMD (single timestep, multiple
  domain) database plugin on top of it.

Its user documentation is SOMA's `docs/visualization/visit.md`; this file keeps the details and
implementation notes.

## Domains and meshes

- Mesh files: one domain per SOMA part. The main mesh is named `mesh` (the file's own
  dimension); every lower dimension the file wrote (boundary faces, edges) is one more mesh,
  `elements_dim<D>`. A domain's meshes share one point array (the part's own points and its
  halo), read once per domain and cached (`HaloCache`, `CachedPiece`).
- VisIt's engine distributes domains over its MPI ranks itself; the plugin never calls MPI.
  Each rank reads only the parts it is given, with one hyperslab per dataset.
- Global node and zone ids are supplied through `GetAuxiliaryData`, so VisIt builds ghost zones
  between domains (this is why `Query("NumZones", use_actual_data=1)` on a multi-domain 3D mesh
  reports more zones than the file holds; `use_actual_data=0` reports the file's own count).
- A knife (7 corners, 5 quadrilateral faces) is handed to VisIt as a degenerate hexahedron,
  corners [0 1 2 3 4 5 6 5]: the hexahedron whose corner 7 coincides with corner 5 has exactly
  the knife's faces and volume (checked against VTK's polyhedron volume). VisIt 3.5's own
  filters cannot re-insert `VTK_POLYHEDRON` cells and crash on them.
- AMR checkpoints: one domain per box over every level (level-major, `AVT_AMR_MESH`), each a
  `vtkRectilinearGrid` (VisIt's face-list filter crashes on a 2D `vtkUniformGrid`); groups are
  levels. Coarse cells under a finer box are blanked through an `avtStructuredDomainNesting`
  built once from the boxes and the refinement ratio.

## Fields

- A field with one fixed-size row per entity is drawable: `ncomp == 1` a scalar, 2 or 3 a vector,
  any other `ncomp` that many scalars `<field>_0`, `<field>_1`, ...
- VisIt keys variables by name alone, so a name that several fields share (MOAB's `GLOBAL_ID`,
  written once per dimension) is qualified by where it lives: `nodal/GLOBAL_ID`,
  `mesh/GLOBAL_ID`, `elements_dim1/GLOBAL_ID` (VisIt shows the '/' as a submenu).
- `soma_part` (owning part of each cell) and `set:<name>` (entity-set labels) are zonal scalars
  on the main mesh.

## Build

Against a VisIt binary install (no VisIt source tree needed); VisIt's own HDF5 is used:

```
mkdir SOMA && cp SOMA.xml avtSOMAFileFormat.* soma_reader.hpp SOMA/ && cd SOMA
<visit>/bin/xml2info -clobber SOMA.xml
<visit>/bin/xml2cmake -clobber SOMA.xml
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ .
make -j
```

`make` installs `libISOMADatabase.so`, `libMSOMADatabase.so`, `libESOMADatabase_ser.so` and
`libESOMADatabase_par.so` into `~/.visit/<version>/<arch>/plugins/databases/`. Use the system
compiler VisIt was built with (a conda compiler in `PATH` links against a different libstdc++).
`SOMA.xml` carries `dbtype="STMD"`, `-std=c++17` and VisIt's `${HDF5_INCLUDE_DIR}`,
`${HDF5_LIBRARY_DIR}` and `${HDF5_LIB}`.

## Tests

1. C++ core against the Python core (`tests/cpp_driver.cpp`, `tests/compare_with_python.py`,
   `tests/compare_amr.py`; build `cpp_driver` with `tests/build.sh`): coordinates by gid, cell
   corner gid sets, every field, set labels, AMR box data; whole file, every part alone, and the
   cached path. 618 mesh checks and 24 AMR checks pass, in both prism conventions
   (`SOMAVIEW_VTK_WEDGE=legacy|current`); 844 further checks pass on 8 files written by SOMA's
   own MPI writer (3 to 7 parts).
2. Malformed files (`tests/check_malformed.py <cpp_driver> <good.soma> <workdir>`, with `cpp_driver`
   built with `-fsanitize=address`): damaged copies of a mesh file (short root, field or value
   metadata, a part table shorter than its parts, corner counts that do not sum to the
   connectivity) each end with an error, never an out-of-bounds read; 7 cases.
3. VisIt itself (`tests/visit_test.py`, expectations from `tests/visit_expected.py`, driven by
   `tests/run_visit_tests.sh`): 16 files on a serial engine and on a 4-rank parallel engine, and the
   50,304,352-cell, 48-part CONUS mesh on an 8-rank engine. For every file the zone count of
   the data read equals the Python core's and the MinMax of a field equals the Python core's;
   every file renders. Files: hex, mixed linear 3D (tet, pyramid, prism, hex, knife), quadratic
   3D and 2D, polygons, the 64-brick MOAB mesh with sets, an ocean mesh, SOMA-MPI-written hex,
   tet, quadrilateral and brick files at 3 to 7 parts, and 2D and 3D AMR checkpoints.

## Findings

- VisIt calls `GetAuxiliaryData` with domain -1 for whole-dataset items; reading part -1 for it
  reads outside the part table (HDF5 "selection + offset not within extent"). Only one valid
  domain's ids are answered now.
- VisIt's NumZones on actual data counts its generated ghost zones and, for quadratic 2D cells,
  the linear zones it tessellates them into; compare file contents with `use_actual_data=0`.
- VTK 9.7 changed the corner order of `vtkWedge`: a SOMA prism needs [0 2 1 3 5 4] for VTK
  <= 9.6 and no reordering for VTK >= 9.7; the plugin picks by `VTK_VERSION_NUMBER`.
