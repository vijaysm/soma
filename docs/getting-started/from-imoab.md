# From iMOAB

iMOAB is MOAB's C interface for applications and couplers, called from Fortran in E3SM and
similar codes. This page places each of its 62 functions, as declared in `moab/iMOAB.h` of MOAB
5.6.0, on the SOMA [Fortran module](fortran.md). The C API has the same operations under `soma_*`
names.

The two interfaces are organized differently:

- **No application registry.** iMOAB keeps applications in a registry and addresses them by an
  integer id. In SOMA, an application holds a `soma_comm` and its `soma_mesh` handles, and
  freeing a mesh (or letting it go out of scope) releases it.
- **Fields instead of tags.** A tag defined with `iMOAB_DefineTagStorage` is a field made with
  `%add_field`. A field is a pointer into the mesh's memory, so reading or writing values needs no
  copy. `%get_field` and `%set_field` copy, interleaved, as iMOAB's `Get`/`Set...TagStorage` do.
- **Ids are already 1-based.** Connectivity, segment bounds and other local ids from the Fortran
  module start at 1, as iMOAB's do from Fortran. `x(conn(k), :)` needs no shift.
- **Errors.** Each call takes an optional `ierr`. Without it, a failure stops the program with
  SOMA's message; there is no error code to check after every call.

This release covers a single application's mesh. iMOAB's coupler calls exchange meshes and
fields between applications on different communicators, through communication graphs, coverage
meshes and map files; those are planned for the next release. Within one communicator,
`soma_migrate` and `soma_remap` already do the repartitioning and the conservative remap.

## Runtime and applications

| iMOAB | SOMA |
|---|---|
| `iMOAB_Initialize`, `iMOAB_InitializeFortran` | `soma_initialize()`, after `MPI_Init` |
| `iMOAB_Finalize` | `soma_finalize()`, before `MPI_Finalize` |
| `iMOAB_RegisterApplication`, `iMOAB_RegisterApplicationFortran` | `soma_comm(fcomm)` for the application's communicator; meshes are created from it |
| `iMOAB_DeregisterApplication`, `iMOAB_DeregisterApplicationFortran` | `%free()` on the application's meshes and communicator, or let them go out of scope |
| `iMOAB_SetGlobalInfo`, `iMOAB_GetGlobalInfo` | none: keep these values in the application |

## Files

| iMOAB | SOMA |
|---|---|
| `iMOAB_ReadHeaderInfo` | `soma_detect_format(path)` gives the format only; there is no header query of parts and tags |
| `iMOAB_LoadMesh` | `soma_read_file(comm, path)`, collective. A MOAB `.h5m` file is read on every rank, its elements placed by its `PARALLEL_PARTITION` sets or by the curve and its tags read as fields. Ghost layers are added after the read with `%add_ghosts` |
| `iMOAB_WriteMesh` | `soma_write_file(comm, mesh, path)`, collective, to `.soma`, `.h5m` or `.vtk` |
| `iMOAB_WriteLocalMesh` | none: there is no write of one rank's part on its own |

## Building a mesh

| iMOAB | SOMA |
|---|---|
| `iMOAB_CreateVertices`, `iMOAB_CreateElements` | `soma_builder(dim)` with `%add_vertices(gids, xyz)` and `%add_elements(topology, gids, vertex_gids)`, then `soma_build_local_part(comm, builder)`. To add to an existing mesh, `soma_edit(mesh)` with `%add_vertices`, `%add_elements` and `%apply` |
| `iMOAB_ResolveSharedEntities` | part of `soma_build_local_part`, which finds the shared entities from the global ids |
| `iMOAB_UpdateMeshInfo` | none needed: counts are read when asked for |
| `iMOAB_MergeVertices` | none in the bindings |
| `iMOAB_DetermineGhostEntities` | `%add_ghosts(comm, layers, bridge_dim)`, collective; `%remove_ghosts(comm)` drops them |

## Querying a mesh

| iMOAB | SOMA |
|---|---|
| `iMOAB_GetMeshInfo` | `%count(d)` (all held entities), `%owned_count(d)`, `%interior_count(d)`, and `%segments(d)` for the runs of shared, copied and ghost entities |
| `iMOAB_GetVisibleVerticesCoordinates` | `%coords()`, shared as `x(n, 3)`, or `%get_coords(xyz)`, an interleaved copy into `xyz(3, n)` as iMOAB gives |
| `iMOAB_GetBlockID`, `iMOAB_GetBlockInfo` | `%segments(d)`: runs of elements of one topology and arity. Material sets are not exposed in the bindings |
| `iMOAB_GetBlockElementConnectivities` | `%conn(d, conn, offsets)`, with the elements of a segment from `first` to `last` |
| `iMOAB_GetElementConnectivity` | `%conn(d, conn, offsets)`: element `e` is `conn(offsets(e) : offsets(e + 1) - 1)` |
| `iMOAB_GetVisibleElementsInfo` | `%gids(d)` for the global ids, `%owner_ranks(comm, d)` for the owners; blocks as above |
| `iMOAB_GetVertexID`, `iMOAB_GetElementID` | `%gids(0)`, `%gids(d)` |
| `iMOAB_GetVertexOwnership`, `iMOAB_GetElementOwnership` | `%owner_ranks(comm, 0)`, `%owner_ranks(comm, d)` |
| `iMOAB_GetNeighborElements`, `iMOAB_GetNeighborVertices` | not in the bindings: `%build_adjacencies` derives the graphs, but reading them needs the C++ library |
| `iMOAB_GetPointerToSurfaceBC`, `iMOAB_GetPointerToVertexBC` | not in the bindings: entity sets (Neumann and Dirichlet sets among them) are read, but only C++ reaches them |

## Tags and fields

| iMOAB | SOMA |
|---|---|
| `iMOAB_DefineTagStorage` | `%add_field(name, dim, ncomp, u)`: `u` points at the new field, `u(rows, ncomp)` |
| `iMOAB_SetIntTagStorage`, `iMOAB_SetDoubleTagStorage` | write into the field pointer, or `%set_field(name, u)` from an interleaved `u(ncomp, rows)` |
| `iMOAB_GetIntTagStorage`, `iMOAB_GetDoubleTagStorage` | read the field pointer from `%field(name, u)`, or `%get_field(name, u)` into an interleaved `u(ncomp, rows)` |
| `iMOAB_SetDoubleTagStorageWithGid` | none: map the global ids to local ids with `%gids(d)`, then write the field |
| `iMOAB_SynchronizeTags` | `soma_halo(comm, mesh, dim)`, then `%update(comm, u)`: owners' values to every copy |
| `iMOAB_ReduceTagsMax` | `%reduce(comm, u, SOMA_MAX)` on the same exchange; `SOMA_SUM`, `SOMA_MIN` and `SOMA_PROD` too |

## Coupling and remap

| iMOAB | SOMA |
|---|---|
| `iMOAB_ComputeMeshIntersectionOnSphere`, `iMOAB_ComputeScalarProjectionWeights` | `soma_remap(comm, source, target, surface=SOMA_SURFACE_SPHERE, radius=r)` intersects the meshes and builds a first-order conservative operator, with both meshes on one communicator |
| `iMOAB_ApplyScalarProjectionWeights` | `%apply(comm, u_source, u_target)` on that operator |
| `iMOAB_SendMesh`, `iMOAB_ReceiveMesh`, `iMOAB_FreeSenderBuffers` | next release. Within one communicator, `soma_migrate(comm, mesh, dest)` moves elements to the ranks given |
| `iMOAB_SendElementTag`, `iMOAB_ReceiveElementTag` | next release |
| `iMOAB_ComputeCommGraph`, `iMOAB_CoverageGraph`, `iMOAB_DumpCommGraph` | next release |
| `iMOAB_MigrateMapMesh`, `iMOAB_ComputeCoverageMesh`, `iMOAB_GetCoverageMeshInfo`, `iMOAB_WriteCoverageMesh`, `iMOAB_SetDoubleTagStorageOnCoverage`, `iMOAB_SetMapGhostLayers` | next release |
| `iMOAB_ComputePointDoFIntersection` | next release |
| `iMOAB_LoadMapFile`, `iMOAB_WriteMapFile` | next release |
