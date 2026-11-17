# Fortran and C

The Fortran module `soma` runs SOMA from Fortran 2008 programs: build or read a mesh, edit it,
smooth or refine it, remap fields between meshes and write it, on any number of ranks under
`mpiexec`. It sits on a C API, `soma/c/soma.h`, which C programs can call directly. Both offer the
operations of the [Python package](python.md); the work is SOMA's, and coordinates, connectivity
and fields are handed out as pointers into the mesh's own memory.

This release runs on the host Kokkos backends (Serial, OpenMP, Threads). Codes coming from iMOAB
will find each of its calls placed on SOMA's in [From iMOAB](from-imoab.md).

## Building and installing

The module needs a Fortran 2008 compiler (gfortran 16 is the version tested) and, with MPI, the
same MPI library as SOMA:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/kokkos -DSOMA_ENABLE_MPI=ON -DSOMA_ENABLE_HDF5=ON \
  -DSOMA_ENABLE_FORTRAN=ON -DCMAKE_Fortran_COMPILER=gfortran
cmake --build build -j
cmake --install build --prefix /path/to/soma
```

`SOMA_ENABLE_FORTRAN` turns on `SOMA_ENABLE_C_API`; the C API alone is `-DSOMA_ENABLE_C_API=ON`.
The install adds `libsoma_c`, `libsoma_fortran`, the headers `soma/c/soma.h` and
`soma/c/soma_config.h`, and the module files in `include/soma/fortran`. Module files belong to the
compiler that wrote them, so a program using the module is compiled with the same Fortran compiler
as SOMA. A CMake project finds both as components:

```cmake
find_package(soma REQUIRED COMPONENTS c fortran)
target_link_libraries(my_c_program PRIVATE soma::soma_c)
target_link_libraries(my_fortran_program PRIVATE soma::soma_fortran)
```

## A first program

Every rank runs the same program, and calls named *collective* in the module's comments must be
made by every rank:

```fortran
program box
  use mpi
  use soma
  implicit none
  integer :: ierr
  call MPI_Init(ierr)
  call soma_initialize()
  block
    type(soma_mesh) :: m
    real(c_double), pointer :: x(:, :)
    m = soma_box_mesh(soma_comm(MPI_COMM_WORLD), 8, 8, 8, element=SOMA_TET)
    x => m%coords()
    print *, 'owned cells:', m%owned_count(3), ' first vertex:', x(1, :)
  end block
  call soma_finalize()
  call MPI_Finalize(ierr)
end program
```

`soma_comm(fcomm)` takes the integer handle of `use mpi` (from `mpi_f08`, pass `comm%MPI_VAL`) and
keeps a reference to it; `soma_comm_world()` and `soma_comm_self()` are shortcuts. A build without
MPI accepts only `soma_comm_self()`. The program initializes MPI before `soma_initialize` and
finalizes it after `soma_finalize`. The `block` lets the mesh go out of scope, and so be freed,
before `soma_finalize`.

## Conventions

- **1-based ids.** Local ids, CSR offsets and segment bounds start at 1, and SOMA's "no id" is 0.
  Global ids and MPI ranks are never shifted.
- **Coordinates** are shared with the mesh as `x(n, 3)`: `x(v, :)` is vertex `v`, and writing into
  `x` moves the mesh. `m%get_coords(xyz)` and `m%set_coords(xyz)` copy to and from an interleaved
  `xyz(3, n)`, the layout iMOAB uses.
- **Connectivity** is a 1-based copy, made once each time the mesh's topology changes and shared
  until then: after `call m%conn(3, conn, off)`, the corners of element `e` are
  `conn(off(e) : off(e + 1) - 1)`, and `x(conn(k), :)` are their coordinates.
- **Fields** are shared as `u(rows, ncomp)` (`real(c_double)`, `integer(c_int32_t)` or
  `integer(c_int64_t)`); `get_field` and `set_field` copy to and from an interleaved
  `u(ncomp, rows)`.
- **Pointers** stay valid until a call changes the mesh's topology or numbering (an edit's
  `apply`, `compress`, `add_ghosts`, `remove_ghosts`, `build_adjacencies`) or the mesh is freed;
  field pointers also until a field is added or removed. Take them again after such a call.
- **Kinds.** `soma_lo` (local ids), `soma_offset` (CSR offsets) and `soma_go` (global ids) follow
  the widths SOMA was built with (`SOMA_LO_64`, `SOMA_OFFSET_64`); `soma_go` is always 64 bits.
- **Handles** (`soma_mesh`, `soma_comm`, `soma_halo`, ...) are reference counted. Assignment shares
  the object, and the last handle to go (by `free` or out of scope) frees it. A halo exchange,
  an edit or a remap operator keeps its meshes alive.

## Errors

Every procedure takes an optional `integer :: ierr`. When it is present it receives `SOMA_OK` or a
`SOMA_ERR_*` status (`INVALID_ARGUMENT`, `OUT_OF_RANGE`, `TYPE`, `STATE`, `RUNTIME`) and
`soma_error_message()` gives SOMA's message. When it is absent, a failure writes
`soma: <message>` to standard error and stops the program with `error stop 1`. Collective calls
agree on their arguments first, so a bad argument on one rank fails the call on every rank, and
every rank returns the same status.

## What a program can do

| Group | Fortran | C |
|---|---|---|
| Build | `soma_builder(dim)`, `%add_vertices(gids, xyz)`, `%add_elements(topology, gids, vertex_gids)`, `soma_build_local_part(comm, builder)`; `soma_box_mesh(comm, nx, ny, nz, element, lx, ly, lz)` | `soma_builder_*`, `soma_build_local_part`, `soma_box_mesh` |
| Query | `%dim()`, `%count(d)`, `%owned_count(d)`, `%interior_count(d)`, `%segments(d)`, `%coords()`, `%conn(d, conn, offsets)`, `%gids(d)`, `%topology(d)`, `%owner_ranks(comm, d)`, `%build_adjacencies(comm, pairs, bridges)` | `soma_mesh_*` |
| Edit | `soma_edit(mesh)`: `%add_vertices`, `%add_elements`, `%add_polygons`, `%remove`, `%apply(added, removed)`; `%compress()` | `soma_edit_*`, `soma_mesh_compress` |
| Fields | `%add_field(name, dim, ncomp, u)`, `%field(name, u, dim)`, `%field_names()`, `%remove_field(name)`, `%get_field`, `%set_field` | `soma_mesh_add_field`, `soma_mesh_field`, ... |
| Files | `soma_read_file(comm, path)`, `soma_write_file(comm, mesh, path, vtk_fields)`, `soma_detect_format(path)`, `soma_convert_mesh_file(comm, src, dst)` | `soma_read_file`, ... |
| Parallel | `soma_partition_by_curve`, `soma_migrate`, `soma_rebalance`, `%add_ghosts(comm, layers, bridge_dim)`, `%remove_ghosts(comm)`, `soma_halo(comm, mesh, dim)`: `%update(comm, u)`, `%reduce(comm, u, op)` | `soma_partition_by_curve`, ..., `soma_halo_*` |
| Smoothing | `soma_laplacian_smooth(comm, mesh, iterations, bridge_dim, tolerance, fixed)` | `soma_laplacian_smooth` |
| Refinement | `soma_refine_uniform(comm, mesh, levels)`, `soma_refine_marked(comm, mesh, marks)`, then `%mesh()`, `%parent()`, `%children()`, `%hanging()`, ... | `soma_refine_*` |
| Remap | `soma_remap(comm, source, target, ...)`, `%apply(comm, u_source, u_target)` | `soma_remap_*` |

The C++ pages describe each operation in depth: [the mesh model](../concepts/mesh-model.md),
[building meshes](../concepts/building-meshes.md), [editing](../concepts/editing.md),
[fields](../concepts/fields.md), [reading and writing](../io/overview.md),
[distributed meshes](../parallel/distributed-meshes.md),
[ghosts and halo exchange](../parallel/ghosts-and-halos.md),
[smoothing](../algorithms/smoothing.md), [refinement](../algorithms/refinement.md) and
[remap](../algorithms/remap.md). The comments in `fortran/soma.F90` and `include/soma/c/soma.h`
give each call's arguments.

## The C API

`soma/c/soma.h` is plain C99. Objects are opaque handles (`soma_mesh_t*`, ...) made by a
`soma_*_create`, `soma_box_mesh`, `soma_read_file` and so on, shared by `soma_*_clone` and released
by `soma_*_free`. Every call returns an `int` status and writes its results through pointers;
`soma_last_error(buf, len)` gives the message of the calling thread's last failure. Ids are
0-based, as in C++, and "no id" is -1. `soma_mesh_conn1` gives the 1-based connectivity copy the
Fortran module uses, for C codes that index from 1.

## A complete program

`fortran/examples/pipeline.F90` (built with `SOMA_ENABLE_EXAMPLES` and MPI) builds a box of tets,
perturbs and smooths it, remaps a linear field to a second box and writes both meshes, the steps of
`examples/pipeline.cpp`. The test `fortran.pipeline` runs both at one and three ranks and requires
the same digests of the coordinates and remapped values, to the last bit.
