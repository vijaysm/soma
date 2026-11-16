# Reading and writing meshes

SOMA reads and writes three mesh file formats: its own parallel HDF5 format
([the SOMA format](soma-format.md), the recommended one), MOAB's
[.h5m](h5m.md) and [legacy VTK](vtk.md). This page covers `read_file` and
`write_file`, which pick the format for you, the `detect_format` query, the
`soma-convert` tool, and what each format keeps. Block-AMR hierarchies have
their own checkpoint files, described in [AMR checkpoints](amr-checkpoints.md).

## Reading and writing without naming the format

The header `soma/io/mesh_file.hpp` declares:

```cpp
struct ReadFileOptions {
  SomaReadOptions soma{};
  H5mReadOptions h5m{};
};

struct WriteFileOptions {
  SomaWriteOptions soma{};
  H5mWriteOptions h5m{};
  VtkWriteOptions vtk{};
  std::optional<std::vector<std::string>> vtk_fields;  // default: every field VTK can hold
};

[[nodiscard]] HostMesh read_file(const Comm& comm, const std::filesystem::path& path,
                                 const ReadFileOptions& options = {});
[[nodiscard]] HostMesh read_file(const std::filesystem::path& path, const ReadFileOptions& options = {});

void write_file(const Comm& comm, const HostMesh& mesh, const std::filesystem::path& path,
                const WriteFileOptions& options = {});
void write_file(const HostMesh& mesh, const std::filesystem::path& path, const WriteFileOptions& options = {});
```

The overloads without a `Comm` run on `Comm::self()`. The example programs
use this pair, so they accept any of the three formats; from
`examples/laplacian_smooth.cpp`:

```cpp
soma::HostMesh mesh = soma::read_file(comm, in);
mesh.add_ghosts(comm, soma::GhostOptions{1, 0, false});
const soma::SmoothReport report = soma::laplacian_smooth(comm, mesh, options);
mesh.remove_ghosts(comm);
soma::write_file(comm, mesh, out, {.vtk_fields = std::vector<std::string>{}});
```

### Reading: by content

`read_file` looks at the file's content, not its name. Rank 0 calls
`detect_format` and broadcasts the answer, so every rank takes the same path:

| Content | Read with | Distribution |
| --- | --- | --- |
| SOMA mesh file | `read_soma` | on every rank; see [the SOMA format](soma-format.md) |
| MOAB .h5m | `read_h5m(comm, ...)` | on every rank; see [MOAB .h5m files](h5m.md) |
| legacy VTK | `read_vtk` | on rank 0; its cells are then dealt to the ranks along the curve (`partition_by_curve`, `migrate`), with their vertex and cell fields |
| SOMA AMR checkpoint | — | throws `std::invalid_argument` (use `read_soma_amr`) |
| anything else | — | throws `std::invalid_argument` |

`ReadFileOptions::soma` and `ReadFileOptions::h5m` are passed to the reader
chosen; legacy VTK reading has no options.

### Writing: by extension

`write_file` goes by the extension of `path`, compared without regard to case:

| Extension | Written with | Distribution |
| --- | --- | --- |
| `.soma` | `write_soma` | on every rank, each writing its part |
| `.h5m` | `write_h5m` | on rank 0, the mesh gathered there first (`migrate`) when there is more than one rank |
| `.vtk` | `write_vtk` | on rank 0, gathered likewise |

Any other extension throws `std::invalid_argument` on every rank before any
file is touched.

For `.vtk`, `WriteFileOptions::vtk_fields` names the fields written. Left
unset, it writes every field `write_vtk` can write: one value per vertex or
per cell over a full domain, `per_entity` layout, neither variable-length nor
handle fields, and none of the library's own `soma:` fields. An empty vector,
as in the snippet above, writes the mesh alone.

Gathering the whole mesh on rank 0 is what the serial formats need; at scale
write `.soma` instead.

### Collective behaviour and errors

Both functions are collective over `comm`: call them on every rank. They
throw on every rank as the readers and writers they call do. When only
rank 0 does the work (reading a VTK file, writing `.h5m` or `.vtk`), a failure
there is sent to the others, which throw `std::runtime_error` with the same
message; rank 0 rethrows its own exception.

## Detecting a file's format

`detect_format`, declared in `soma/io/soma_file.hpp`, tells the format from
the content:

```cpp
enum class MeshFormat : std::uint8_t { legacy_h5m, soma, soma_amr, legacy_vtk, unknown };

MeshFormat detect_format(const std::filesystem::path& path);
```

- `legacy_vtk`: the file opens with a legacy VTK header, `# vtk DataFile`
  (case aside).
- `soma`: an HDF5 file with a `/soma` group holding a mesh.
- `soma_amr`: an HDF5 file whose `/soma` group carries an AMR checkpoint (its
  `amr` attribute).
- `legacy_h5m`: an HDF5 file with a `/tstt` group.
- `unknown`: anything else, or a file that cannot be opened. In a build
  without HDF5 every HDF5 file is `unknown`.

It is not collective; `read_file` calls it on rank 0 only.

## Converting between .h5m and SOMA files

`convert_mesh_file` (in `soma/io/soma_file.hpp`) converts in either direction,
chosen by the input's content:

```cpp
MeshFormat convert_mesh_file(const Comm& comm, const std::filesystem::path& in, const std::filesystem::path& out,
                             int compression = 0);
```

- A MOAB .h5m input is read in parallel (`read_h5m`) and written with
  `write_soma`.
- A SOMA input is read with `read_soma`, gathered on rank 0 and written there
  by `write_h5m`, without the `soma:sharing` and `soma:owner` fields.

`compression` is the deflate level of the file written. It returns the format
written, and throws `std::invalid_argument` on every rank for an input of
neither format (a VTK file included), and otherwise as the readers and
writers do. It is collective over `comm`.

The `soma-convert` command-line tool (`tools/soma_convert.cpp`) wraps it. It
is built when SOMA is configured with `SOMA_ENABLE_HDF5=ON`, and installed
with the library when `SOMA_INSTALL` is on. Run it under MPI to convert in parallel:

```bash
mpiexec -n 8 soma-convert --compression 4 ocean.h5m ocean.soma
soma-convert ocean.soma ocean_again.h5m
```

Usage is `soma-convert [--compression N] <in> <out>`. The exit status is 0
when done, 1 when the conversion failed (rank 0 prints the message) and 2 for
bad arguments.

## What each format keeps

| | SOMA (`.soma`) | MOAB `.h5m` | legacy VTK (`.vtk`) |
| --- | --- | --- | --- |
| Written in parallel | yes, every rank writes its part | no, rank 0 writes | no, rank 0 writes |
| Read in parallel | yes | yes | no, rank 0 reads, then the cells are dealt |
| Global ids | kept | file ids become gids | renumbered in file order |
| Parts and sharing | kept: a read at the writing rank count gives each rank its part | elements placed by `PARALLEL_PARTITION` sets or by the curve | no |
| Fields | every field, with domains, defaults, global values and formats | `per_entity` fields, as tags | `per_entity` fields over a full domain on vertices and cells |
| Entity sets | yes, with their order | yes | no |
| Mesh values | yes | yes, as tags with no entity values | no |
| High-order nodes | yes | yes | no: cells are written by their corners |
| Needs HDF5 | yes | yes | no |

The pages for each format give the details and limits. Formats that need HDF5
throw `std::runtime_error` from their readers and writers in a build without
`SOMA_ENABLE_HDF5`.
