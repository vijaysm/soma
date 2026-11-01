# Provenance

SOMA is a clean-room implementation. The following were copied or ported from moabX
(git@bitbucket.org:fathomteam/moabx.git, branch `vijaysm/codex/pp-moabx-foundation`,
commit `c4a45fd`), authored by the same team:

| SOMA path | Source | Treatment |
|---|---|---|
| `include/soma/topology/canonical.hpp` | `include/moabx/mesh/subentity_topology.hpp` | tables ported, device-callable rewrite |
| `tests/reference/test_canonical_moab.cpp` | `src/moabx_subentity_topology_tests.cpp` | ported to GoogleTest |
| `cmake/CPM.cmake` | `cmake/CPM.cmake` | verbatim, v0.43.2 (MIT, CPM.cmake project) |
| `cmake/SomaAppleOpenMP.cmake` | `CMakeLists.txt` OpenMP block, `cmake/moabxConfig.cmake.in` (c6e82ba) | adapted |
| `tests/packaging/{CMakeLists.txt,main.cpp}` | `tests/packaging/*` | adapted |
| `tests/packaging/run_packaging_test.cmake` | `tests/packaging/run_packaging_test.cmake` | rewritten from the described logic, no text carried over |
| `cmake/somaConfig.cmake.in` | `cmake/moabxConfig.cmake.in` | libomp replay adapted |
| `data/*.h5m`, `*.vtk`, `*.msh` | `data/` | verbatim |
| `tests/golden/serial_derivation.{hpp,cpp}` | staged as `porting/moabx-c4a45fd/subentity_derivation.{hpp,cpp}` (`moabx/mesh/subentity_derivation.hpp`) | ported: `std::unordered_map` for `phmap`, SOMA types and canonical tables |
| `tests/reference/test_adjacency_moab.cpp` (ring neighbourhoods) | staged as `porting/moabx-c4a45fd/native_reference.hpp` (`moabx/mesh/native_reference.hpp`) | vertex and element rings rewritten as a test; edge rings dropped |
| `bench/harness.hpp` | staged as `porting/moabx-c4a45fd/harness.hpp` | ported onto `soma::Comm`, untimed setup added |
| `porting/moabx-c4a45fd/*` | various | staging copies, deleted once ported (the sub-entity topology, derivation, MOAB reference and harness files are gone) |

`porting/` holds reference copies only: they are never compiled or installed, and each file is
deleted in the commit that finishes its port. Text from MOAB-derived (LGPL) files must not be
copied into SOMA sources.

The crystal router (design §7.2) is a clean-room implementation of the algorithm in gslib
(Nek5000, BSD-3, UChicago Argonne), as described by Tufo & Fischer. No MOAB source (LGPL-3) is
copied into SOMA.

The legacy VTK files in `data/vtk_reference/` were written by ParaView 6.1.0-RC1's VTK 9.6
writer (`vtkUnstructuredGridWriter`, ASCII, file versions 5.1 and 4.2) from
`data/vtk_reference/make_reference.py`, run with ParaView's `pvpython`.

Large meshes are not stored in this repository. Point `-DSOMA_LARGE_DATA_DIR` at a directory
containing `outCS_ne1024.h5m`, `oRRS18to6v3.170111.p512.h5m`, and
`xylar_oEC60to30wLI_scrip.h5m` (available in the moabX `data/` directory).
