# Provenance

SOMA is a clean-room implementation. The following were copied or ported from moabX
(git@bitbucket.org:fathomteam/moabx.git, branch `vijaysm/codex/pp-moabx-foundation`,
commit `c4a45fd`), authored by the same team:

| SOMA path | Source | Treatment |
|---|---|---|
| `include/soma/topology/canonical.hpp` | `include/moabx/mesh/subentity_topology.hpp` | tables ported, device-callable rewrite |
| `tests/reference/test_canonical_moab.cpp` | `src/moabx_subentity_topology_tests.cpp` | ported to GoogleTest |
| `cmake/CPM.cmake` | `cmake/CPM.cmake` | verbatim (MIT, CPM.cmake project) |
| `cmake/SomaAppleOpenMP.cmake` | `CMakeLists.txt` OpenMP block, `cmake/moabxConfig.cmake.in` (c6e82ba) | adapted |
| `tests/packaging/*` | `tests/packaging/*` | adapted |
| `data/*.h5m`, `*.vtk`, `*.msh` | `data/` | verbatim |
| `porting/moabx-c4a45fd/*` | various | staging copies, deleted once ported |

`porting/` holds reference copies only: they are never compiled or installed, and each file is
deleted in the commit that finishes its port. Text from MOAB-derived (LGPL) files must not be
copied into SOMA sources.

The crystal router (design §7.2) is a clean-room implementation of the algorithm in gslib
(Nek5000, BSD-3, UChicago Argonne), as described by Tufo & Fischer. No MOAB source (LGPL-3) is
copied into SOMA.

Large meshes are not stored in this repository. Point `-DSOMA_LARGE_DATA_DIR` at a directory
containing `outCS_ne1024.h5m`, `oRRS18to6v3.170111.p512.h5m`, and
`xylar_oEC60to30wLI_scrip.h5m` (available in the moabX `data/` directory).
