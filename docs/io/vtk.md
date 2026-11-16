# Legacy VTK files

SOMA reads and writes ASCII legacy VTK files of `DATASET UNSTRUCTURED_GRID`,
with no third-party dependency. Use them to look at meshes and fields in
ParaView or VisIt, or to read meshes other tools wrote. The format is serial
and keeps no sets, no global ids and no partition; for restarts use
[the SOMA format](soma-format.md).

## API

The reader is in `soma/io/vtk_reader.hpp`, the writer in
`soma/io/vtk_writer.hpp`:

```cpp
HostMesh read_vtk(const std::filesystem::path& path);

struct VtkField {
  std::string name;
  std::optional<int> dim{};  // needed only when the name is on several dimensions
};

enum class VtkLegacyVersion { v3_0, v5_1 };

struct VtkWriteOptions {
  VtkLegacyVersion version = VtkLegacyVersion::v5_1;
};

void write_vtk(const HostMesh& mesh, const std::filesystem::path& path,
               std::span<const std::string> fields = {}, const VtkWriteOptions& options = {});
void write_vtk(const HostMesh& mesh, const std::filesystem::path& path,
               std::span<const VtkField> fields, const VtkWriteOptions& options = {});
void write_vtk(const HostMesh& mesh, const std::filesystem::path& path,
               const VtkWriteOptions& options);
```

`write_vtk` writes only the fields it is given; without a list it writes the
mesh alone. Adapted from `tests/unit/test_vtk_writer.cpp`:

```cpp
mesh.fields().add_tag<std::int64_t>("i", 0, 1);
const std::vector<std::string> names{"i"};
soma::write_vtk(mesh, file(), names);                     // version 5.1, the default
soma::write_vtk(mesh, file(), names, {.version = soma::VtkLegacyVersion::v3_0});
```

Both functions work on one process. On several ranks use `read_file` and
`write_file` (see [Reading and writing meshes](overview.md) and below).

## Writing

The writer has two layouts, both read by current VTK:

| `VtkLegacyVersion` | Header | `CELLS` | 64-bit integer arrays |
| --- | --- | --- | --- |
| `v5_1` (default) | `# vtk DataFile Version 5.1` | `ncells + 1` and the connectivity size, then `OFFSETS vtktypeint64` and `CONNECTIVITY vtktypeint64` | `vtktypeint64`, `vtktypeuint64` |
| `v3_0` | `# vtk DataFile Version 3.0` | `ncells` and the list size, one `count ids...` line per cell | `long`, `unsigned_long` |

**Cells.** Every element of dimension `mesh.dim()` that the mesh holds, in
local id order, by its corner vertices in the element's canonical order:

| SOMA topology | VTK cell type |
| --- | --- |
| `edge` | 3 |
| `tri` | 5 |
| `polygon` | 7 |
| `quad` | 9 |
| `tet` | 10 |
| `hex` | 12 |
| `prism` | 13 (wedge) |
| `pyramid` | 14 |

Knives and polyhedra have no legacy cell type and throw. Wedges follow the
VTK 9.7 convention; readers older than VTK 9.7 (ParaView 6.1 and earlier, for
one) show them with inverted orientation.

**Fields.** A `per_entity` field over a full domain on dimension 0 goes to
`POINT_DATA`, one on `mesh.dim()` to `CELL_DATA`. Each is written as `FIELD`
data (`name ncomp ntuples type`), in the order of the selection. Value types
map as:

| SOMA | VTK |
| --- | --- |
| f64, f32 | `double`, `float` |
| i32, u32, u8 | `int`, `unsigned_int`, `unsigned_char` |
| i64, u64 | see the version table above |
| i8 | `signed_char` in 5.1; refused in 3.0 (widen it to i32 or write 5.1) |

Reals are written in the shortest form that reads back bitwise equal. A NaN,
an infinity or a subnormal value, in a field or in the coordinates, throws
`std::invalid_argument`, since legacy VTK readers cannot read them.

**Errors.** Any other field (another dimension, a subset domain, a layout that
is not `per_entity`, pending writes, an unknown or ambiguous name) throws
`std::invalid_argument`, and a stale field throws `StaleField`. In version
3.0, more than `INT32_MAX` points or `CELLS` integers throws
`std::length_error`. All checks run before any file is touched.

**Atomic replacement.** The file is written to a new temporary next to `path`
(`path` plus `.` and six random characters, created exclusively) and renamed
over `path`, so a failure leaves neither a partial file nor a damaged earlier
one, and concurrent writers do not collide. The suffix takes 7 bytes, so where
file names are limited to 255 bytes a name of 249 to 255 bytes throws
`std::runtime_error`. Nothing is `fsync`ed. I/O failures throw
`std::runtime_error` naming the path.

An edited mesh is written as its compressed copy would be. The writer needs
floating-point `std::to_chars` (libstdc++ 11+, libc++ 14+, macOS 13.3+) and is
POSIX only.

## Reading

`read_vtk` reads ASCII files of versions 2.0, 3.0, 4.x and 5.1, written by
`write_vtk` or by VTK.

- **Cells.** Both the 5.1 layout (`OFFSETS` and `CONNECTIVITY` of any integer
  type) and the older one. Cell types 3, 5, 7 (3 to 65535 vertices), 9, 10,
  12, 13 and 14, as in the table above; polyhedra (42) and any other code
  throw. Wedges follow the VTK 9.7 convention, so a wedge written by VTK 9.6 or
  older comes in with inverted orientation. Degenerate and duplicate cells are
  read as they are.
- **Mesh.** Built with `Ordering::file`; its dimension is the largest cell
  dimension. Points get gids 1 to npoints in file order, and cells gids 1 to n
  in file order within each dimension. A file without cells throws.
- **Fields.** `POINT_DATA` and `CELL_DATA` arrays become `per_entity` tags on
  dimension 0 and on the cell dimension: `FIELD` arrays, `SCALARS`, `VECTORS`,
  `NORMALS`, `TENSORS`, `TENSORS6` and `TEXTURE_COORDINATES`. `CELL_DATA`
  needs every cell to have one dimension. `GLOBAL_IDS`, and integer
  `PEDIGREE_IDS`, become i64 tags named after their array. String and variant
  arrays are skipped.
- **Skipped.** Dataset-level `FIELD` data, `LOOKUP_TABLE` and `COLOR_SCALARS`
  blocks, and `METADATA` blocks.

`BINARY` files, other datasets, unknown sections or cell types, and malformed,
missing or out-of-range values throw `std::runtime_error` naming the file, the
line and the offending token. The header lists the full set of rules,
including the value type mapping.

## Parallel use

Both formats are serial; `read_file` and `write_file` make them usable from
any number of ranks:

- `read_file` reads a VTK file on rank 0 and deals its cells to the ranks
  along the curve (`partition_by_curve`, `migrate`), with their vertex and cell
  fields. Every rank gets a distributed part with ownership and sharing.
- `write_file` to a `.vtk` path gathers the mesh on rank 0 (`migrate`) and
  writes it there. `WriteFileOptions::vtk_fields` selects the fields; unset, it
  writes every field `write_vtk` can write.

A failure on rank 0 reaches every rank as an exception. The whole mesh must fit
on one rank, so for large meshes write [SOMA files](soma-format.md).
