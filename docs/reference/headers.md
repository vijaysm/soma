# Public headers

Every header under `include/soma/`, grouped by directory, with what it provides and the page that
documents it. Each public header's name links to its page in the [API reference](api/index.md). Include headers by their path from `include/`, for example
`#include "soma/core/mesh.hpp"`. The comment at the top of each header (and above each
declaration) is the authoritative contract; the pages summarise it.

Headers marked **internal** declare only `soma::detail` (or another detail namespace): they are
installed because public headers include them, but their contents are implementation details and
may change without notice.

The generated header `soma/config.hpp` (written by CMake into the build tree and installed beside
the others) records the build options that change the ABI or the public headers:
`SOMA_ENABLE_MPI`, `SOMA_LO_64`, `SOMA_OFFSET_64`, `SOMA_DEBUG`, `SOMA_ENABLE_KOKKOSKERNELS`,
`SOMA_ENABLE_PETSC`, `SOMA_ENABLE_ARBORX`, the version numbers and `SOMA_GPU_AWARE_MPI_DEFAULT`
(see [CMake options](cmake-options.md)).

## `soma/core`

The mesh data model. Documented in [the mesh model](../concepts/mesh-model.md) unless noted.

| Header | Provides |
| --- | --- |
| [`soma/core/types.hpp`](api/core/types.md) | the ordinal types: `LO` (local id; 64-bit with `SOMA_LO_64`), `Offset` (CSR offsets), `GO` (global id), `Real` |
| [`soma/core/view_types.hpp`](api/core/view_types.md) | the Kokkos view aliases `View1`, `View2`, `UView1`; every SOMA view is `LayoutLeft` |
| [`soma/core/topology.hpp`](api/core/topology.md) | `Topology`, the element types with MOAB's `EntityType` values; `dimension`, `num_vertices`, `has_fixed_topology` |
| [`soma/core/segments.hpp`](api/core/segments.md) | `Ownership` (interior, shared, shared copy, ghost), `Segment`, `SegmentTable`, `make_segments`, `split_segments` |
| [`soma/core/mesh.hpp`](api/core/mesh.md) | `Mesh<MemSpace>`, the move-only owner of one rank's mesh in one memory space; `HostMesh`, `DeviceMesh`, `EntityBlock`, `Renumbering`; `clone()` and `mirror<Space>()` |
| [`soma/core/mesh_view.hpp`](api/core/mesh_view.md) | `MeshView`, the device-copyable kernel handle of a mesh |
| [`soma/core/transfer.hpp`](api/core/transfer.md) | `TransferOptions`: what a mirror copies (dimensions, coordinates, sets, fields, adjacency graphs) |
| [`soma/core/mesh_builder.hpp`](api/core/mesh_builder.md) | `MeshBuilder`: vertices and elements (high-order included) to a mesh. See [building meshes](../concepts/building-meshes.md) |
| [`soma/core/ordering.hpp`](api/core/ordering.md) | `Ordering` (`file`, `type_major_sfc`, `tiled`), the entity order a build uses. See [building meshes](../concepts/building-meshes.md) |
| [`soma/core/sfc.hpp`](api/core/sfc.md) | Hilbert curve keys (`soma::sfc`, `KeyBox`, `hilbert_index`) for ordering entities by locality. See [building meshes](../concepts/building-meshes.md) |
| [`soma/core/locality.hpp`](api/core/locality.md) | `VertexSpanReport`, `vertex_span_report`: how far apart in memory an element's vertices are. See [building meshes](../concepts/building-meshes.md) |
| [`soma/core/mesh_edit.hpp`](api/core/mesh_edit.md) | `MeshEdit` (batched additions and removals made visible by `apply()`), `EditReport`, `removed_fraction`. See [editing meshes](../concepts/editing.md) |
| [`soma/core/sets.hpp`](api/core/sets.md) | entity sets, as MOAB's meshsets: `EntitySets`, `SetsView`, `MissingSetMembers`. See [entity sets](../concepts/sets.md) |
| [`soma/core/set_builder.hpp`](api/core/set_builder.md) | `SetBuilder`, `SetSpec`, `SetItem`, `SetRun`: collect sets on the host and install them at once. See [entity sets](../concepts/sets.md) |
| [`soma/core/lid_runs.hpp`](api/core/lid_runs.md) | a sorted set of local ids held as full, intervals or list, shared by domain blocks and set rows. See [entity sets](../concepts/sets.md) |
| [`soma/core/mesh_values.hpp`](api/core/mesh_values.md) | `MeshValue`, `MeshValues`: values of the mesh rather than of its entities (MOAB class-3 tags). See [fields](../concepts/fields.md) |
| [`soma/core/handle_ref.hpp`](api/core/handle_ref.md) | handles naming an entity or set by slot and gid (`make_handle`), and `HandleRef`, which resolves handle fields to entities and sets on the host and in kernels. See [fields](../concepts/fields.md) |
| [`soma/core/point_key.hpp`](api/core/point_key.md) | exact keys of point coordinates in device code (`point_keys`): IEEE bits as an unsigned integer that sorts as the coordinate does. See [matching and merging](../parallel/matching-and-merging.md) |
| [`soma/core/hash3d_exact.hpp`](api/core/hash3d_exact.md) | namespace `spatial_exact`: exact coordinate keys and serial id assignment by exact match or tolerance (`assign_ids_exact`, `assign_ids_tol`). See [matching and merging](../parallel/matching-and-merging.md) |
| `soma/core/search.hpp` | **internal**: `lower_bound` and `upper_bound` over views |

## `soma/topology`

Adjacencies and canonical numbering. Documented in [topology](../concepts/topology.md).

| Header | Provides |
| --- | --- |
| [`soma/topology/adjacency.hpp`](api/topology/adjacency.md) | adjacency graphs between dimensions (downward, upward, bridge) and the request naming them: `AdjacencyRequest`, `AdjacencyGraph`, `AdjacencyStore`, `MissingAdjacency` |
| [`soma/topology/build.hpp`](api/topology/build.md) | `build_adjacencies`, the one entry point that derives sub-entities and builds the requested graphs, with the strong exception guarantee |
| [`soma/topology/canonical.hpp`](api/topology/canonical.md) | canonical side tables of fixed-topology elements (`soma::canonical`), as MOAB's `CN::SubEntityVertexIndices`; device-callable |
| [`soma/topology/nodes.hpp`](api/topology/nodes.md) | MOAB's high-order node layouts: `num_nodes`, `mid_node_bits`, `node_parent`, `NodeParent` |
| [`soma/topology/orientation.hpp`](api/topology/orientation.md) | relative orientation of two vertex lists naming one entity (`connectivity_match`, `Match`), as MOAB's `CN::ConnectivityMatch` |
| [`soma/topology/sides.hpp`](api/topology/sides.md) | the sides of an entity as mesh vertex ids in canonical order (`side_of`, `side_count`, `Side`); device-callable |
| `soma/topology/derive.hpp` | **internal**: sub-entities (edges, faces) from elements by the lowest-creator rule |
| `soma/topology/bridge.hpp` | **internal**: bridge adjacency (entities sharing an entity of the bridge dimension) |
| `soma/topology/transpose.hpp` | **internal**: upward graphs as the transpose of a downward CSR, made independent of the thread order |

## `soma/fields`

Fields on entities. Documented in [fields](../concepts/fields.md).

| Header | Provides |
| --- | --- |
| [`soma/fields/data_type.hpp`](api/fields/data_type.md) | `DataType` (`i8` ... `f64`), `data_type_of`, `dispatch_data_type` |
| [`soma/fields/layout.hpp`](api/fields/layout.md) | `Layout` (how many value points each entity carries), `NodeFamily`, `NodeOrdering` |
| [`soma/fields/domain.hpp`](api/fields/domain.md) | `Domain`: an immutable set of local entity ids per dimension, held as full, intervals or list |
| [`soma/fields/space.hpp`](api/fields/space.md) | `Space` = domain × layout: the row numbering of every field defined on it |
| [`soma/fields/domain_mesh.hpp`](api/fields/domain_mesh.md) | mesh-aware construction of domains and spaces (`SegmentFilter`, `make_space`) |
| [`soma/fields/field.hpp`](api/fields/field.md) | `Field<T, MemSpace>` (typed handle on one field), `FieldOptions`, `FieldInfo` |
| [`soma/fields/var_field.hpp`](api/fields/var_field.md) | `VarField<T, MemSpace>`: a variable-length field, a CSR over a per-entity space |
| [`soma/fields/registry.hpp`](api/fields/registry.md) | `FieldRegistry`, the host-side owner of a mesh's fields and their families |
| [`soma/fields/resolver.hpp`](api/fields/resolver.md) | entity-to-value lookup across the members of a field family |
| [`soma/fields/errors.hpp`](api/fields/errors.md) | the exceptions `FieldOverlap`, `FieldClash`, `StaleField` |
| [`soma/fields/tag_format.hpp`](api/fields/tag_format.md) | `TagFormat`: how a `.h5m` file stores a family's tag, recorded by `read_h5m` and restored by `write_h5m` |
| [`soma/fields/node_tables.hpp`](api/fields/node_tables.md) | reference-element node coordinates for high-order layouts (`NodeTables`, `make_node_table`) |
| `soma/fields/overlap.hpp` | **internal**: the smallest local id two domain blocks share |

## `soma/parallel`

Distribution over MPI ranks.

| Header | Provides |
| --- | --- |
| [`soma/parallel/comm.hpp`](api/parallel/comm.md) | `Comm`, the single communication seam (single-rank identities without MPI), `Request`, `SplitComm`, `Op`, `CommCounts`. See [communicators](../parallel/comm.md) |
| [`soma/parallel/distributed.hpp`](api/parallel/distributed.md) | the local part of a distributed mesh: `build_local_part`, `build_local_part_deferred`, `build_distributed_adjacencies`. See [distributed meshes](../parallel/distributed-meshes.md) |
| [`soma/parallel/sharing.hpp`](api/parallel/sharing.md) | `Sharing`, `resolve_sharing`: which ranks hold an entity, from the gids each rank holds. See [distributed meshes](../parallel/distributed-meshes.md) |
| [`soma/parallel/partition.hpp`](api/parallel/partition.md) | `partition_by_curve`: partitions without a graph, along a Morton curve of element centroids. See [distributed meshes](../parallel/distributed-meshes.md) |
| [`soma/parallel/migrate.hpp`](api/parallel/migrate.md) | `migrate`, `rebalance`, `place_closure`, `Placement`: moving elements between ranks with their closure. See [distributed meshes](../parallel/distributed-meshes.md) |
| [`soma/parallel/ghosts.hpp`](api/parallel/ghosts.md) | ghost layers: `GhostOptions`, `add_ghosts_by_rebuild`. See [ghosts and halos](../parallel/ghosts-and-halos.md) |
| [`soma/parallel/halo_exchange.hpp`](api/parallel/halo_exchange.md) | `HaloExchange`: `update` (owners to copies and ghosts) and `reduce` over the shared entities of one dimension. See [ghosts and halos](../parallel/ghosts-and-halos.md) |
| [`soma/parallel/point_match.hpp`](api/parallel/point_match.md) | `match_points`, `PointIds`, `PointTolerance`, `PointNorm`: global ids of points from their coordinates. See [matching and merging](../parallel/matching-and-merging.md) |
| [`soma/parallel/merge.hpp`](api/parallel/merge.md) | `merge_vertices`, `number_elements`, `merge_mesh`, `MergeOptions`, `MergeGids`: coincident vertices made one. See [matching and merging](../parallel/matching-and-merging.md) |
| [`soma/parallel/exact_sum.hpp`](api/parallel/exact_sum.md) | `ExactSums`, `ExactSumReducer`, `allreduce_exact`: sums of doubles independent of the order of the terms. See [reproducible sums](../parallel/reproducible-sums.md) |
| [`soma/parallel/transport/route.hpp`](api/parallel/transport/route.md) | `transport::route`, `RecordBatch`, `Strategy`: sparse personalized exchange of fixed-width records. See [transport](../parallel/transport.md) |
| [`soma/parallel/transport/comm_plan.hpp`](api/parallel/transport/comm_plan.md) | `transport::CommPlan`, `learn_plan`: a routed exchange replayed by point-to-point messages. See [transport](../parallel/transport.md) |
| [`soma/parallel/transport/rendezvous.hpp`](api/parallel/transport/rendezvous.md) | `transport::rendezvous`, `rendezvous_to`: a distributed directory keyed by a hash of the key. See [transport](../parallel/transport.md) |
| [`soma/parallel/transport/sort.hpp`](api/parallel/transport/sort.md) | `transport::rank_keys`, `rank_keys_held`, `KeyRanks`: global ranks of keys by a sample sort. See [transport](../parallel/transport.md) |

## `soma/io`

Files. See [input and output](../io/overview.md).

| Header | Provides |
| --- | --- |
| [`soma/io/mesh_file.hpp`](api/io/mesh_file.md) | `read_file`, `write_file`: read and write a mesh without naming its format. See [I/O overview](../io/overview.md) |
| [`soma/io/soma_file.hpp`](api/io/soma_file.md) | the SOMA format: `write_soma`, `read_soma`, `convert_mesh_file`, `detect_format`, `SomaWriteOptions`, `SomaReadOptions`. See [the SOMA format](../io/soma-format.md) |
| [`soma/io/soma_amr.hpp`](api/io/soma_amr.md) | AMR checkpoints in the SOMA format: `write_soma` for a hierarchy, `read_soma_amr`, `AmrCheckpoint`. See [AMR checkpoints](../io/amr-checkpoints.md) |
| [`soma/io/h5m.hpp`](api/io/h5m.md) | MOAB `.h5m` input and output with the HDF5 C API: `read_h5m`, `write_h5m`, `h5m_available`. See [MOAB .h5m files](../io/h5m.md) |
| [`soma/io/vtk_writer.hpp`](api/io/vtk_writer.md) | legacy VTK output: `write_vtk`, `VtkWriteOptions`, `VtkLegacyVersion`, `VtkField`. See [VTK files](../io/vtk.md) |
| [`soma/io/vtk_reader.hpp`](api/io/vtk_reader.md) | legacy VTK input: `read_vtk`, `legacy_vtk_header`. See [VTK files](../io/vtk.md) |
| [`soma/io/id_runs.hpp`](api/io/id_runs.md) | `soma::io`: file ids written as (start, count) runs, used by the writers. See [I/O overview](../io/overview.md) |

## `soma/algorithms`

| Header | Provides |
| --- | --- |
| [`soma/algorithms/laplacian.hpp`](api/algorithms/laplacian.md) | `laplacian_smooth`, `SmoothOptions`, `SmoothReport`: Laplacian smoothing of vertex coordinates. See [smoothing](../algorithms/smoothing.md) |
| [`soma/algorithms/quality_smooth.hpp`](api/algorithms/quality_smooth.md) | `quality_smooth`, `QualitySmoothOptions`, `QualitySmoothReport`: optimization-based smoothing of surface meshes. See [smoothing](../algorithms/smoothing.md) |
| [`soma/algorithms/fem.hpp`](api/algorithms/fem.md) | `soma::fem`: P1 and Q1 elements on single cells (`cell_of`, `stiffness`, `mass`, `load`, `Quadrature`). See [finite elements](../algorithms/fem.md) |
| [`soma/algorithms/fem_space.hpp`](api/algorithms/fem_space.md) | `FemSpace`: linear finite elements over a mesh's vertices (`laplacian`, `load`, `boundary`, `update`). See [finite elements](../algorithms/fem.md) |
| [`soma/algorithms/fv.hpp`](api/algorithms/fv.md) | `FvSpace`, `FvScheme`: cell-centred finite volumes, first-order upwind or bounded second order (`normal_flux`, `max_dt`, `step`). See [finite volumes](../algorithms/finite-volumes.md) |
| [`soma/algorithms/csr.hpp`](api/algorithms/csr.md) | `CsrMatrix`, `apply_dirichlet`: a rank's rows of a sparse matrix over vertices, products the same bit for bit on any number of ranks and threads. See [linear algebra](../algorithms/linear-algebra.md) |
| [`soma/algorithms/cg.hpp`](api/algorithms/cg.md) | `cg`, `CgOptions`, `CgReport`: Jacobi-preconditioned conjugate gradients over a `FemSpace`. See [linear algebra](../algorithms/linear-algebra.md) |
| [`soma/algorithms/kokkos_sparse.hpp`](api/algorithms/kokkos_sparse.md) | `to_kokkos_sparse`: a `CsrMatrix` as a `KokkosSparse::CrsMatrix` (`SOMA_ENABLE_KOKKOSKERNELS`). See [linear algebra](../algorithms/linear-algebra.md) |
| [`soma/algorithms/petsc.hpp`](api/algorithms/petsc.md) | `to_petsc_matrix`, `to_petsc_vector`, `from_petsc_vector` (`SOMA_ENABLE_PETSC`). See [linear algebra](../algorithms/linear-algebra.md) |
| [`soma/algorithms/grid_index.hpp`](api/algorithms/grid_index.md) | `GridIndex`: a uniform grid of bins for the boxes a query box meets. See [spatial search](../algorithms/spatial-search.md) |
| [`soma/algorithms/arborx_index.hpp`](api/algorithms/arborx_index.md) | `ArborXIndex`: the same queries by ArborX's bounding volume hierarchy (`SOMA_ENABLE_ARBORX`). See [spatial search](../algorithms/spatial-search.md) |
| [`soma/algorithms/remap.hpp`](api/algorithms/remap.md) | `RemapOperator`, `RemapOptions`: first-order conservative remap between plane, volume or sphere meshes. See [remap](../algorithms/remap.md) |
| [`soma/algorithms/refine.hpp`](api/algorithms/refine.md) | `refine_uniform`, `refine_marked`, `mark_fraction`, `mark_threshold`, `infer_births`, `RefineOptions`, `MarkedOptions`, `RefineResult`. See [refinement](../algorithms/refinement.md) |
| [`soma/algorithms/refine_graded.hpp`](api/algorithms/refine_graded.md) | `refine_graded`, `refine_levels`, `band_widths`, `level_sizes`, `GradedOptions`, `GradedTarget`, `GradedHook`, `GradedResult`: graded all-quad refinement on the unit sphere toward a target given by a callback. See [refinement](../algorithms/refinement.md) |
| `soma/algorithms/polygon.hpp` | **internal**: plane polygons for intersection (clipping, triangulation), callable in kernels. See [remap](../algorithms/remap.md) |
| `soma/algorithms/tet_clip.hpp` | **internal**: the overlap of two tetrahedra, for volume remap. See [remap](../algorithms/remap.md) |
| `soma/algorithms/convex_clip.hpp` | **internal**: the overlap of two convex cells with flat faces, for volume remap. See [remap](../algorithms/remap.md) |
| `soma/algorithms/sphere_clip.hpp` | **internal**: great-circle polygons on the unit sphere, by gnomonic projection or great circles. See [remap](../algorithms/remap.md) |
| `soma/algorithms/vertex_graph.hpp` | **internal**: host-side graphs over a mesh's vertices (neighbours, cells around a vertex, skin) ordered by gid |
| `soma/algorithms/to_space.hpp` | **internal**: a host vector copied into a view of a memory space |

## `soma/sem`

| Header | Provides |
| --- | --- |
| [`soma/sem/gll.hpp`](api/sem/gll.md) | `gll_weights_1d`, `gll_derivative_1d`: the 1D GLL weights and derivative matrix. See [spectral elements](../algorithms/spectral-elements.md) |
| [`soma/sem/cubed_sphere.hpp`](api/sem/cubed_sphere.md) | `CubedSphereSpec`, `cubed_sphere_mesh`: the equiangular cubed sphere, linear or quadratic, split across the ranks. See [spectral elements](../algorithms/spectral-elements.md) |
| [`soma/sem/sem_space.hpp`](api/sem/sem_space.md) | `SemSpace`, `SemGeometry`: GLL points and geometry over quads in the plane or on a sphere; `dss`, `mass`, `integral`. See [spectral elements](../algorithms/spectral-elements.md) |
| [`soma/sem/operators.hpp`](api/sem/operators.md) | `gradient`, `divergence`, `vorticity`, `project_tangent`, `laplacian`, `hyperviscosity`. See [spectral elements](../algorithms/spectral-elements.md) |
| [`soma/sem/boundary.hpp`](api/sem/boundary.md) | `SemBoundary`, `make_boundary`, `remove_normal`: the coast points of a `SemSpace`, their outward normals and the no-slip corners; removes the normal flow from a vector field. See [spectral elements](../algorithms/spectral-elements.md) |

## `soma/amr`

Patch-based AMR, namespace `soma::amr`. Documented in [block-structured AMR](../amr/block-amr.md).

| Header | Provides |
| --- | --- |
| [`soma/amr/box.hpp`](api/amr/box.md) | `IntVect`, `IndexType`, `Box`: the integer index space; device-callable |
| [`soma/amr/box_list.hpp`](api/amr/box_list.md) | `BoxList` and set operations: `subtract`, `complement`, `intersect`, `refine`, `coarsen`, `grow`, `covers` |
| [`soma/amr/box_array.hpp`](api/amr/box_array.md) | `BoxArray` (a level's boxes along a space-filling curve), `Curve`, `chop`, `Distribution` |
| [`soma/amr/cluster.hpp`](api/amr/cluster.md) | `cluster`, `ClusterOptions`: Berger–Rigoutsos clustering of tagged cells |
| [`soma/amr/level_field.hpp`](api/amr/level_field.md) | `LevelField`, `LevelView`, `Patch`, `Region`, `for_each_point`: data on the boxes of one level |
| [`soma/amr/copy.hpp`](api/amr/copy.md) | `CopyPlan`, `CopyOp`, `Periodicity`, `copy`, `begin_copy`, `PendingCopy` |
| [`soma/amr/halo.hpp`](api/amr/halo.md) | `HaloPlan`, `fill_halo`, `begin_fill_halo`, `BoundaryRegion`: same-level halo exchange |
| [`soma/amr/interlevel.hpp`](api/amr/interlevel.md) | `CoarseFinePlan`, `fill_coarse_fine`, `AverageDownPlan`, `average_down`, `Interp` |
| [`soma/amr/flux_register.hpp`](api/amr/flux_register.md) | `FluxRegister`: refluxing across a coarse-fine interface |
| [`soma/amr/hierarchy.hpp`](api/amr/hierarchy.md) | `Hierarchy`, `Geometry`, `HierarchyOptions`, `MultiLevelField`, `TagField`, `NoBoundary`: levels, fills, restriction and regrids |
| [`soma/amr/vtk_amr.hpp`](api/amr/vtk_amr.md) | `write_vtk_amr`, `VtkAmrField`: VTK overlapping AMR output (`.vthb`) |

## `soma/runtime`

Documented in [runtime](../concepts/runtime.md).

| Header | Provides |
| --- | --- |
| [`soma/runtime/scope_guard.hpp`](api/runtime/scope_guard.md) | `ScopeGuard`: reads SOMA's options, then initializes MPI (in MPI builds) and Kokkos unless they run already, and stops what it started |
| [`soma/runtime/options.hpp`](api/runtime/options.md) | process-wide options: `parse_options`, `compress_threshold`, `set_compress_threshold`, `reproducible`, `set_reproducible` |

## `soma/testing`

| Header | Provides |
| --- | --- |
| [`soma/testing/box_mesh.hpp`](api/testing/box_mesh.md) | `soma::testing::box_mesh`, `BoxSpec`, `BoxCounts`, `box_counts`: structured box meshes (hexes, tets, quads, triangles) with closed-form counts, coordinates and orientation, for tests. See [building meshes](../concepts/building-meshes.md) |

## `soma/compat`

MOAB's API over SOMA meshes (`SOMA_ENABLE_MOAB_COMPAT`; installed only then). Documented in
[the MOAB compatibility layer](../compat/moab-compat.md).

| Header | Provides |
| --- | --- |
| [`soma/compat/types.hpp`](api/compat/types.md) | MOAB's types and constants with MOAB's values: `EntityHandle`, `EntityType`, `ErrorCode`, `TagType`, `DataType`, `make_handle`, `type_of`, `id_of` |
| [`soma/compat/core.hpp`](api/compat/core.md) | `Interface`, `Core`: MOAB's interface over a SOMA host mesh |
| [`soma/compat/range.hpp`](api/compat/range.md) | `Range`: MOAB's sorted set of handles kept as runs |
| [`soma/compat/cn.hpp`](api/compat/cn.md) | `CN`: the part of MOAB's canonical numbering applications call |
| [`soma/compat/scd.hpp`](api/compat/scd.md) | `ScdInterface`, `ScdBox`, `HomCoord`: MOAB's structured box construction |
| [`soma/compat/prog_options.hpp`](api/compat/prog_options.md) | `ProgOptions`: the part of MOAB's command-line options its examples use |
| [`soma/compat/error.hpp`](api/compat/error.md) | MOAB's error macros under `SOMA_` names (`SOMA_MB_CHK_ERR`, `SOMA_MB_SET_ERR`, ...) |
| `soma/compat/serial_mpi/mpi.h` | a one-rank stand-in for the MPI calls of MOAB's serial examples, on the drop-in include path of a build without MPI |
| `soma/compat/dropin/moab/*.hpp`, `moab/MOABConfig.h` | drop-in headers at MOAB's include paths (`moab/Core.hpp`, `moab/Interface.hpp`, `moab/Range.hpp`, `moab/CN.hpp`, `moab/ScdInterface.hpp`, `moab/HomXform.hpp`, `moab/ProgOptions.hpp`, `moab/Types.hpp`, `moab/EntityHandle.hpp`, `moab/EntityType.hpp`, `moab/Error.hpp`, `moab/ErrorHandler.hpp`, `moab/Forward.hpp`): `namespace moab = soma::compat` and MOAB's macro names |
| `soma/compat/dropin/MBTagConventions.hpp` | MOAB's conventional tag names |
| `soma/compat/dropin/moab_mpi.h` | includes `<mpi.h>` |
