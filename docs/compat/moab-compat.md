# MOAB compatibility layer

`soma::compat` offers the part of MOAB's C++ interface that MOAB applications commonly call
(`Core`/`Interface`, `Range`, `CN`, `ScdInterface`, `ProgOptions`, the error macros and the type
and constant definitions) over a SOMA host mesh. Use it to run an existing MOAB application on
SOMA, or to port one step by step: in drop-in mode the application's `#include <moab/Core.hpp>`
resolves to SOMA's headers and the code builds unchanged. It is not a full reimplementation of
MOAB; the limits are listed [below](#limits).

## Enabling it

The layer is optional. Configure with `SOMA_ENABLE_MOAB_COMPAT=ON` (default `OFF`), which builds
two more targets:

| Target | Alias / exported name | What it is |
| --- | --- | --- |
| `soma_compat` | `soma::compat` | the library: headers under `soma/compat/`, namespace `soma::compat` |
| `soma_compat_dropin` | `soma::compat_dropin` | an interface target that puts `include/soma/compat/dropin` on the include path (and, in a build without MPI, `include/soma/compat/serial_mpi`), and links `soma_compat` |

Both are installed and exported with `soma` when the option is on; without it the
`soma/compat` headers are not installed.

```bash
cmake -S . -B build -DSOMA_ENABLE_MOAB_COMPAT=ON -DSOMA_ENABLE_HDF5=ON
```

HDF5 is not required, but without `SOMA_ENABLE_HDF5` only `.vtk` files can be read and written.
See [CMake options](../reference/cmake-options.md) for the other options.

## Two ways to use it

**Namespaced.** Include `soma/compat/core.hpp` (and `range.hpp`, `cn.hpp`, `scd.hpp`,
`prog_options.hpp`, `error.hpp` as needed), link `soma::compat`, and use the names in
`soma::compat`: `soma::compat::Core`, `soma::compat::Range`, `soma::compat::MB_SUCCESS`, and so on.
The error macros carry a `SOMA_` prefix (`SOMA_MB_CHK_ERR`, `SOMA_MB_SET_ERR`, ...). This is how
`tests/reference/test_compat_core_moab.cpp` uses the layer side by side with MOAB, from which:

```cpp
#include "soma/compat/core.hpp"
namespace sc = soma::compat;

sc::Core a;
ASSERT_EQ(a.load_file((data + "/64bricks_1khex.h5m").c_str()), sc::MB_SUCCESS);
sc::Range elems;
a.get_entities_by_dimension(0, 3, elems);
```

**Drop-in.** Link `soma::compat_dropin` instead of MOAB and leave the application's sources
alone. The drop-in headers sit at MOAB's include paths:

| Drop-in header | Provides |
| --- | --- |
| `moab/Core.hpp`, `moab/Interface.hpp` | `soma/compat/core.hpp` |
| `moab/Range.hpp` | `soma/compat/range.hpp` |
| `moab/CN.hpp` | `soma/compat/cn.hpp` |
| `moab/ScdInterface.hpp`, `moab/HomXform.hpp` | `soma/compat/scd.hpp` |
| `moab/ProgOptions.hpp` | `soma/compat/prog_options.hpp`, with `ProgOptions` in the global namespace as in MOAB |
| `moab/Types.hpp`, `moab/EntityHandle.hpp`, `moab/EntityType.hpp`, `moab/Error.hpp`, `moab/ErrorHandler.hpp`, `moab/Forward.hpp` | the types; `namespace moab = soma::compat;`; MOAB's error macro names (`MB_CHK_ERR`, `MB_SET_ERR`, `MB_CHK_SET_ERR`, ... ) defined as the `SOMA_` ones |
| `moab/MOABConfig.h` | `MOAB_VERSION` `"5.6.0"` and its major, minor and patch numbers |
| `MBTagConventions.hpp` | MOAB's conventional tag names (`MATERIAL_SET_TAG_NAME`, `GLOBAL_ID_TAG_NAME`, ...) |
| `moab_mpi.h` | `<mpi.h>` |

In a build without MPI, `<mpi.h>` itself is a one-rank stand-in (`soma/compat/serial_mpi/mpi.h`)
for the MPI calls MOAB's serial examples make (`MPI_Init`, `MPI_Initialized`, `MPI_Finalize`,
`MPI_Finalized`, `MPI_Comm_rank`, `MPI_Comm_size`, `MPI_Barrier`, `MPI_Bcast`, `MPI_Allreduce`,
`MPI_Reduce`, `MPI_Abort`): every communicator has one rank and collectives copy their input. In a
build with MPI the application gets the MPI SOMA was built with.

```cmake
find_package(soma REQUIRED)
add_executable(HelloMOAB HelloMOAB.cpp)
target_link_libraries(HelloMOAB PRIVATE soma::compat_dropin)
```

## The model

The SOMA mesh (a `HostMesh`) holds the entities and their coordinates. Tags and sets are kept by
the compat layer, MOAB's way: they are read from a file's fields and sets by `load_file`, and put
back on a copy of the mesh by `write_file`. `Interface::soma_mesh()` returns the mesh as it stands
(staged creations applied), without the tags and sets kept on the compat side.

Created entities are staged in a `MeshEdit` and made visible by the next query, so a burst of
creations costs one apply; deletions apply at once. The compat layer never compresses its mesh.
Deleting an entity takes it out of every set (MOAB leaves it in sets that do not track their
members).

The first instance initializes Kokkos if the application has not, and finalizes it at exit; an
instance must not outlive `main()`.

### Handles and ids

Handles keep MOAB's layout (`soma/compat/types.hpp`): `EntityHandle` is `std::uint64_t`, with the
`EntityType` in the top `MB_TYPE_WIDTH = 4` bits and a per-type id, from `MB_START_ID = 1`, below.

```cpp
constexpr EntityHandle make_handle(EntityType type, EntityID id) noexcept;
constexpr EntityType type_of(EntityHandle h) noexcept;
constexpr EntityID id_of(EntityHandle h) noexcept;
```

Ids are given in the order the entities were read or created, so a file read by `soma::compat`
and by MOAB gives the same handles. Ids are allocated as MOAB allocates them, in blocks per type
and node count, and the id of a deleted entity can be given to an entity created later, as in
MOAB. Reading or writing a tag on a deleted entity fails with `MB_ENTITY_NOT_FOUND`.

`EntityType` (`MBVERTEX` ... `MBENTITYSET`, `MBMAXTYPE`), `ErrorCode` (`MB_SUCCESS` ...
`MB_FAILURE`), `TagType`, `DataType` and `EntitySetProperty` have MOAB's values.

## What `Interface` provides

`soma::compat::Interface` (and `Core`, derived from it) declares the following groups of MOAB
calls, with MOAB's signatures, handles, return codes and results. The header
`soma/compat/core.hpp` is the complete list.

| Group | Calls |
| --- | --- |
| Files | `load_file`, `load_mesh`, `write_file` (two overloads), `write_mesh`, `delete_mesh` |
| Handles | `type_from_handle`, `id_from_handle`, `dimension_from_handle`, `handle_from_id`, `get_dimension`, `set_dimension` |
| Coordinates | `get_vertex_coordinates`, `get_coords` (three overloads), `set_coords` |
| Connectivity | `get_connectivity` (corners then higher-order nodes; `corners_only` for corners) |
| Adjacencies | `get_adjacencies` (with `create_if_missing`, which creates the sides a downward query lacks, as MOAB does; `INTERSECT` or `UNION`), `side_number` |
| Entity queries | `get_entities_by_dimension`, `get_entities_by_type`, `get_entities_by_handle`, `get_entities_by_type_and_tag`, `get_number_entities_by_dimension`, `get_number_entities_by_type`, `get_number_entities_by_handle` |
| Creation and deletion | `create_vertex`, `create_vertices`, `create_element`, `delete_entities` |
| Tags | `tag_get_handle`, `tag_get_name`, `tag_get_length`, `tag_get_bytes`, `tag_get_type`, `tag_get_data_type`, `tag_get_default_value`, `tag_get_tags`, `tag_get_tags_on_entity`, `tag_get_data`, `tag_set_data`, `tag_get_by_ptr`, `tag_set_by_ptr`, `tag_clear_data`, `tag_delete_data`, `tag_delete`, `tag_iterate`, `tag_commit`, `tag_commit_all`, `tag_has_pending` |
| Sets | `create_meshset`, `clear_meshset`, `get_meshset_options`, `set_meshset_options`, `add_entities`, `remove_entities`, `contains_entities`, parent/child/contained queries and counts, `add_parent_meshset`, `add_child_meshset`, `add_parent_child`, `remove_parent_child`, `remove_parent_meshset`, `remove_child_meshset` |
| Other | `api_version`, `query_interface(ScdInterface*&)`, `release_interface`, `get_last_error` |

Tag values are stored as they are set, so `tag_commit` has nothing to do and `tag_has_pending` is
false. `tag_iterate` gives the values of a dense run of entities of one type in place.

### Range, CN, ScdInterface, ProgOptions, errors

- `Range` (`soma/compat/range.hpp`): MOAB's sorted set of handles kept as closed runs. The runs
  live in a vector, so inserting in increasing order appends, while an insertion in the middle
  moves the later runs; as in MOAB, iterators are invalidated by any change.
  `tests/reference/test_compat_range_moab.cpp` checks it against MOAB's `Range`.
- `CN` (`soma/compat/cn.hpp`): the part of MOAB's canonical numbering that applications call
  (`EntityTypeName`, `EntityTypeFromName`, `Dimension`, `VerticesPerEntity`, `NumSubEntities`,
  `SubEntityType`, `SubEntityVertexIndices`, `HasMidNodes`, `SideNumber`, `TypeDimensionMap`),
  from SOMA's own canonical tables, which tests check against MOAB's.
- `ScdInterface` (`soma/compat/scd.hpp`): `construct_box` and `get_boxes`, and `ScdBox`'s ijk
  lookups (`get_element`, `get_vertex`, `box_dims`, `box_size`, ...). A box of hexes, or of quads
  when its k extent is flat, is built as explicit entities with MOAB's ijk numbering of handles (i
  fastest, from the box's low corner). Without coordinates, vertex $(i, j, k)$ sits at
  $(i, j, k)$, as in MOAB. Only on an empty instance, without periodicity or parallel data;
  anything else gives `MB_NOT_IMPLEMENTED`.
- `ProgOptions` (`soma/compat/prog_options.hpp`): options with a long name, a short one or both
  (`"dim,d"`, `",n"`), of type `int`, `double`, `std::string` or `void` (a flag), given as
  `--dim 2`, `--dim=2`, `-d 2` or `-d2`; `-h` and `--help` print the options and exit. Required
  positional arguments come in the order they were added. A malformed command line prints a
  message and exits with status 1, as MOAB's does.
- Error macros (`soma/compat/error.hpp`): an error is reported on stderr in MOAB's format, and each
  `SOMA_MB_CHK_ERR` it passes through adds its own location line. Unlike MOAB's macros, which
  evaluate a failing expression a second time, these evaluate it once.

## Limits

- One file per instance: `load_file` on an instance that holds a mesh gives `MB_NOT_IMPLEMENTED`.
- File options other than none give `MB_UNHANDLED_OPTION`; reading part of a file (set tags,
  blocks) or writing part of a mesh (output sets, tag lists) gives `MB_NOT_IMPLEMENTED`.
- Formats: `.h5m` (with `SOMA_ENABLE_HDF5`) and `.vtk` only. `write_file` to `.vtk` writes the
  integer and double tags that have a value on every vertex or on every element of the mesh
  dimension, as MOAB does.
- Host meshes only, and no parallel calls (no `ParallelComm`).
- Elements are as SOMA stores them: polyhedra and knives cannot be created.

## How it is tested

With `SOMA_ENABLE_MOAB_COMPAT` and `SOMA_ENABLE_MOAB_REFERENCE` on, the tests compare the layer
with MOAB itself:

- `test_compat_core_moab` makes the same calls on `soma::compat::Core` and `moab::Core` on the same
  files and expects the same handles, values and return codes; `test_compat_range_moab` does the
  same for `Range`.
- The example gate (`tests/compat/examples.cmake`) builds MOAB examples from MOAB's
  `share/examples` (`SOMA_MOAB_EXAMPLES_DIR`) twice, against MOAB and against
  `soma_compat_dropin`, runs both on the same input, and requires the same standard output and
  output files that read the same in MOAB (`tests/compat/moab_describe.cpp` compares them). The
  examples are `HelloMOAB`, `GetEntities`, `SetsNTags`, `DeleteEdges`, `ReadPartFile`,
  `StructuredMeshSimple`, `mbex1` to `mbex4`; the tests are named `compat.gate.*`.

`test_compat_error` and `test_compat_prog_options` run with the layer alone. See
[building and testing](../developer/building-and-testing.md).
