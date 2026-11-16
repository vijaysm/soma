# AMR checkpoints

A block-AMR hierarchy and its fields can be saved in a SOMA file and read back
at any number of ranks. Use these checkpoints to restart an
[AMR computation](../amr/block-amr.md); they use the same HDF5 machinery as
[the SOMA mesh format](soma-format.md) but hold a hierarchy, not a mesh. They
need a build with `SOMA_ENABLE_HDF5=ON`; without it both functions throw
`std::runtime_error`.

## API

The header is `soma/io/soma_amr.hpp`:

```cpp
// A field to write: its name and every level of it.
struct AmrFieldRef {
  std::string name;
  AmrValues<AmrFieldPointer> levels;

  template <class T>
  AmrFieldRef(std::string n, const amr::MultiLevelField<T>& f);
};

// A field read back.
struct AmrField {
  std::string name;
  AmrValues<amr::MultiLevelField> levels;
};

struct AmrCheckpoint {
  amr::Hierarchy hierarchy;
  std::vector<AmrField> fields;

  template <class T>
  const amr::MultiLevelField<T>& field(const std::string& name) const;
};

struct SomaAmrReadOptions {
  std::function<double(int level, const amr::Box& box)> cost;
};

void write_soma(const Comm& comm, const amr::Hierarchy& hierarchy, const std::vector<AmrFieldRef>& fields,
                const std::filesystem::path& path, const SomaWriteOptions& options = {});

AmrCheckpoint read_soma_amr(const Comm& comm, const std::filesystem::path& path, const SomaAmrReadOptions& options = {});
```

A field may hold any of the value types i8, u8, i32, u32, i64, u64, f32 and
f64 (`AmrValues`). `AmrCheckpoint::field<T>(name)` returns the field of that
name and type, and throws `std::invalid_argument` when there is no field of
that name or it holds another value type.

## Usage

From the header's example:

```cpp
soma::write_soma(comm, h, {{"density", rho}, {"velocity", u}}, "run.soma");
auto c = soma::read_soma_amr(comm, "run.soma");    // any rank count
const auto& rho = c.field<double>("density");
c.hierarchy.fill_halo(rho, amr::Interp::linear, boundary);
```

The fields come back in the order written. Their ghost points are zero after
the read: fill them (`fill_halo`) before use.

To cut the levels over the reading ranks by your own cost rather than by point
count, give a cost per box; adapted from `tests/unit/test_soma_amr.cpp`:

```cpp
soma::SomaAmrReadOptions o;
o.cost = [](int level, const soma::amr::Box& b) { return b.lo(0) < 32 ? 1.0 : 5.0; };
const auto weighted = soma::read_soma_amr(comm, "run.soma", o);
```

The cost is kept in the hierarchy's options.

## Writing

`write_soma` for a hierarchy is collective over `comm`. It checks everything
on every rank first: the fields belong to the hierarchy and are distributed as
its levels, every rank gives the same names and kinds, names are not repeated,
and the deflate level is in [0, 9]. A failed check throws
`std::invalid_argument` on every rank before any file is touched. The file is
then written to a temporary beside `path` and renamed over it, or the call
throws `std::runtime_error` naming the path on every rank.

`SomaWriteOptions` applies as for meshes: `compression`, `hints` and
`aggregators` (see [the SOMA format](soma-format.md#write-options)).

## Reading

`read_soma_amr` is collective over the reading communicator, which may have
any number of ranks. It rebuilds the hierarchy over that communicator, cutting
each level along the curve by `options.cost` (point counts when empty); each
rank reads one run of rows per field and level. A file that is not an AMR
checkpoint (a SOMA mesh file, an `.h5m` file), a newer `format_version`, or
datasets that disagree with the boxes throw `std::runtime_error` on every
rank. Conversely, `read_soma` and `read_file` refuse an AMR checkpoint, and
`detect_format` returns `MeshFormat::soma_amr` for one.

## What the file holds

The root group `/soma` carries an attribute `amr` instead of a mesh's `meta`:
the format version, the hierarchy's geometry and options, and the number of
levels and fields. Under `/soma/amr`:

- every level's boxes in curve order (`amr/levels/lL`);
- per field (`amr/fields/fJ`) its name, value type, index type, number of
  components and ghost width, and the valid points of every box of every
  level, box after box in curve order, each box with i fastest, then j, k,
  then the component.

Ghost points are not written. Part p of the file is what rank p owned, so the
rows of the parts follow the curve.
