# MOAB .h5m files

SOMA reads and writes MOAB's native HDF5 format, `.h5m`, with the HDF5 C API
and without MOAB. Use it to bring MOAB meshes into SOMA and to hand meshes back
to MOAB-based tools. For checkpoints and large parallel output prefer
[the SOMA format](soma-format.md): `.h5m` is written by one process. The
functions need a build with `SOMA_ENABLE_HDF5=ON`; without it `read_h5m` and
`write_h5m` throw `std::runtime_error` and `h5m_available()` returns `false`.

## API

The header is `soma/io/h5m.hpp`:

```cpp
bool h5m_available() noexcept;

HostMesh read_h5m(const std::filesystem::path& path, const H5mReadOptions& options = {});
HostMesh read_h5m(const std::filesystem::path& path, const H5mReadOptions& options,
                  H5mReport& report);

HostMesh read_h5m(const Comm& comm, const std::filesystem::path& path, const H5mReadOptions& options = {});
HostMesh read_h5m(const Comm& comm, const std::filesystem::path& path, const H5mReadOptions& options,
                  H5mReport& report);

void write_h5m(const HostMesh& mesh, const std::filesystem::path& path, const H5mWriteOptions& options = {});
```

`H5mReport` lists what a read left out, one entry (`name`, `reason`) per tag
or tag part not read:

```cpp
soma::H5mReport report;
soma::HostMesh m = soma::read_h5m("64bricks_1khex.h5m", {}, report);
for (const auto& s : report.skipped) std::printf("%s: %s\n", s.name.c_str(), s.reason.c_str());
```

`read_file` and `write_file` call these for `.h5m` files (see
[Reading and writing meshes](overview.md)), and `convert_mesh_file` /
`soma-convert` converts between `.h5m` and SOMA files.

## Read options

`H5mReadOptions`:

| Name | Default | Meaning |
| --- | --- | --- |
| `ordering` | `Ordering::file` | The local ordering of the mesh built. |
| `tags` | `true` | Read tags as fields and mesh values. |
| `adjacencies` | `true` | Install the adjacency graphs the file caches (written by `write_h5m`). |
| `file_parts` | `true` | Parallel read: place elements by the file's `PARALLEL_PARTITION` sets when it has at least as many as ranks; `false`: always by the curve. |
| `collective` | unset | Parallel read: open the file with MPI-IO and read its slices with collective transfers. Unset: when the HDF5 is parallel. The serial read ignores it. |

## What is read

- **Entities.** The vertices (`/tstt/nodes/coordinates`, 1 to 3 columns,
  missing ones zero) and every element group whose type is Edge, Tri, Quad,
  Polygon, Tet, Pyramid, Prism or Hex. Polygons come as fixed-arity groups
  (`Polygon5`, ...) or as one list with `poly_indices`. Groups with mid-edge,
  mid-face and mid-region nodes (`Quad9`, say) are read with their high-order
  nodes. The mesh dimension is the largest element dimension; elements of
  lower dimensions (an `Edge2` group beside `Hex8`, say) are kept as explicit
  entities.
- **Global ids.** Every entity's gid is its file id.
- **Entity sets.** `/tstt/sets` with contents, children and parents, whether
  stored id by id or as ranges (flag `0x8`, not kept). A set's gid is its file
  id.
- **Tags** (with `tags`). Each tag on vertices, elements or sets becomes one
  `per_entity` field per dimension (and on the set slot), named after the tag,
  over all entities of that dimension when each has a value and over the
  listed ones otherwise. Dense and sparse storage of one slot join one field.
  Value types: 8-, 32- and 64-bit integers, f32 and f64, arrays of those,
  opaque tags (as u8 with one component per byte) and bit tags (one u8 per
  entity). Default and global values are kept. Variable-length tags become
  `VarField`s, and handle tags become GO fields with `FieldOptions::handle`
  whose values stay file ids. A tag with values on no entity becomes a mesh
  value in `mesh.values()`.
- **Cached adjacencies** (with `adjacencies`) under `/tstt/soma`, as written
  by `write_h5m`.

Not read, and listed in the report: 16-bit integers, reals other than f32 and
f64, bit tags wider than 8 bits, arrays and opaque values of an unusable size,
variable-length tags of several components per value, and other HDF5 type
classes.

Files that cannot be read throw `std::runtime_error` naming the file: a file
that is not HDF5 or has no `/tstt`, no nodes or no elements, Polyhedron or
Knife elements, ids that do not name a vertex or entity, overlapping file id
ranges, malformed index lists, an ordered set stored as ranges, and two values
of one tag on one entity, among others listed in the header.

## Parallel reads

`read_h5m(comm, path, ...)` is collective over `comm`. Every rank reads the
metadata and a slice of each dataset, and returns its local part:

- Elements of the mesh's dimension go by the file's parts when it has at
  least as many `PARALLEL_PARTITION` sets as ranks and `file_parts` is set:
  part i of n (by tag value, then set id) to rank floor(iP/n); an element in
  several parts goes with the first. Otherwise they go by the curve of their
  corners' centroids (`partition_by_curve`).
- Explicit entities of lower dimensions go to every rank holding all their
  corners, and vertices to every rank holding an entity that uses them; one
  that no rank holds stays with the rank that read it.
- Each rank builds its part with `build_local_part`, so every dimension has
  its ownership and sharing.
- Sets are on every rank with the members held there; tag values reach every
  rank holding their entity (values on sets, every rank); cached graphs are
  built on each part rather than read.

Gids are file ids, as in the serial read, and each rank's entities, members
and values equal the serial read's by gid. With `collective` every rank takes
part in every slice read; the mesh is the same either way. A failure on any
rank throws on every rank (`std::runtime_error` on the others), and
`collective = true` with a serial HDF5 throws `std::invalid_argument`.

## Writing

`write_h5m` runs on one process and writes a file that MOAB's `load_file` and
`read_h5m` read. On several ranks, `write_file` gathers the mesh on rank 0 and
writes it there.

`H5mWriteOptions`:

| Name | Default | Meaning |
| --- | --- | --- |
| `dims` | all unset | `dims[d]` set to `true` writes dimension d, `false` leaves it out; unset follows the rules below. |
| `compression` | `0` | Deflate level in [0, 9]. Above 0, datasets are chunked at about 1 MiB with shuffle and deflate; 0 stores them contiguous, as MOAB does. |
| `adjacencies` | `{}` | An `AdjacencyRequest` naming graphs to cache under `/tstt/soma`, which MOAB ignores. The mesh must hold them. |

What is written:

- **Entities.** The vertices, the elements of `mesh.dim()`, of every dimension
  the mesh was built with, and of every dimension that fields have values on,
  sets hold or handles name. A dimension can be left out only when every
  entity of it is a side of a written entity of a higher dimension. One group
  per element type and node count (`Edge2`, `Tri3`, `Quad4`, `Polygon5`,
  `Hex8`, ...). Knives and polyhedra throw.
- **File ids** are 1-based and contiguous: the vertices, then the groups, each
  in gid order; sets follow. A mesh that `read_h5m` read in file order is
  written under its file ids when they were laid out so.
- **Sets** with their flags; unordered contents as sorted file ids, stored as
  (start, count) pairs when MOAB would (`io::moab_ranged` in
  `soma/io/id_runs.hpp`); ordered sets as their sequence.
- **Tags**: one per field family, `per_entity` layouts only, dense where every
  entity of a block has a value and sparse otherwise; variable-length fields
  always sparse. Mesh values become tags with no entity values. Handle values
  are written as file ids, with `is_handle`.
- **`GLOBAL_ID`** is always written: the mesh's own `GLOBAL_ID` field, or else
  the gids of the written vertices and elements as i32. A gid that does not
  fit an i32 throws `std::invalid_argument`.

No history is written, so equal meshes give byte-equal files. An edited mesh
is written as its compressed copy would be. Every check runs first: a mesh
that cannot be written throws `std::invalid_argument` before any file is
touched. The file is written to a temporary next to `path` and renamed over
it, so a failure leaves neither a partial file nor a damaged earlier one; I/O
failures throw `std::runtime_error`.

## Limits

- Writing is serial; parallel output through `write_file` gathers the whole
  mesh on rank 0.
- No polyhedra or knives, in either direction.
- Fields with layouts other than `per_entity` are not written; neither are
  SOMA's ownership and sharing (use the SOMA format to keep parts).
- The parallel read places elements by the file's partition only when it has
  at least as many parts as ranks.
