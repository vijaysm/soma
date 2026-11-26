# Quick start

This page walks through a small SOMA program: start the runtime, build a distributed mesh (or read
one from a file), query it on the host and in a Kokkos kernel, write it out, and run it under
MPI. It assumes SOMA is [installed](installation.md); the programs in
[examples](examples.md) go further.

## The program

The code below builds an $n \times n$ grid of quads on the unit square, its rows of cells dealt
to the ranks in blocks, the way `examples/remap.cpp` builds its target grid. Given a file name it
reads that mesh instead.

```cpp
#include <cstdint>
#include <cstdio>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh_builder.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/runtime/scope_guard.hpp"

namespace {

// This rank's rows of an n x n grid of quads on the unit square, with every vertex they use.
soma::MeshBuilder grid(const soma::Comm& comm, int n) {
  const auto id = [n](int i, int j) { return static_cast<soma::GO>(j * (n + 1) + i + 1); };
  const int first = n * comm.rank() / comm.size();
  const int last = n * (comm.rank() + 1) / comm.size();
  std::vector<soma::GO> verts, cells, conn;
  std::vector<soma::Real> xyz;
  for (int j = first; j < last; ++j) {
    for (int i = 0; i < n; ++i) {
      cells.push_back(static_cast<soma::GO>(j * n + i + 1));
      conn.insert(conn.end(), {id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)});
    }
  }
  for (int j = first; first < last && j <= last; ++j) {
    for (int i = 0; i <= n; ++i) {
      verts.push_back(id(i, j));
      xyz.insert(xyz.end(), {static_cast<double>(i) / n, static_cast<double>(j) / n, 0.0});
    }
  }
  soma::MeshBuilder b(2);
  b.add_vertices(verts, xyz);
  b.add_elements(soma::Topology::quad, cells, conn);
  return b;
}

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);  // MPI, then Kokkos
  const soma::Comm comm = soma::Comm::world();

  soma::HostMesh mesh =
      argc > 1 ? soma::read_file(comm, argv[1]) : soma::build_local_part(comm, grid(comm, 8));

  // Counts: each entity is owned by one rank, so the owned counts add up to the whole mesh's.
  const int d = mesh.dim();
  const soma::LO owned_cells = mesh.segment_table(d).owned_end;
  const auto cells = comm.allreduce(std::int64_t{owned_cells}, soma::Op::sum);
  const auto vertices = comm.allreduce(std::int64_t{mesh.segment_table(0).owned_end}, soma::Op::sum);

  // A lean copy on the default device (topology and coordinates), and a kernel over its owned
  // cells: the area of each polygon, projected on the xy plane, by the shoelace formula.
  double area = 0;
  if (d == 2) {
    const soma::DeviceMesh device = mesh.mirror<Kokkos::DefaultExecutionSpace::memory_space>();
    const auto v = device.view();
    Kokkos::parallel_reduce(
        "area", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, owned_cells),
        KOKKOS_LAMBDA(const soma::LO e, double& acc) {
          const soma::LO n = v.num_vertices(2, e);
          double twice = 0;
          for (soma::LO k = 0; k < n; ++k) {
            const soma::LO a = v.vertex(2, e, k), b = v.vertex(2, e, (k + 1) % n);
            twice += v.x(a, 0) * v.x(b, 1) - v.x(b, 0) * v.x(a, 1);
          }
          acc += twice / 2;
        },
        area);
    area = comm.allreduce(area, soma::Op::sum);
  }

  soma::write_file(comm, mesh, "quickstart.vtk");
  if (comm.rank() == 0) {
    std::printf("%lld cells, %lld vertices, area %.15g on %d rank(s)\n", static_cast<long long>(cells),
                static_cast<long long>(vertices), area, comm.size());
  }
  return 0;
}
```

## What each step does

**Starting the runtime.** `soma::ScopeGuard` (`soma/runtime/scope_guard.hpp`) initialises MPI
(in MPI builds) and then Kokkos, unless the application already did, and finalises what it
started in reverse order when it goes out of scope. Kokkos forbids creating a View before it is
initialised, so the guard comes first in every SOMA program, and every mesh must be destroyed
before it: declaring the guard first in `main` does that. The guard also reads SOMA's own
options (`--soma-reproducible`, `--soma-compress-threshold=<x>` and the matching environment
variables `SOMA_REPRODUCIBLE` and `SOMA_COMPRESS_THRESHOLD`; see `soma/runtime/options.hpp`), and
Kokkos removes its own `--kokkos-*` options, so parse the remaining arguments after constructing
the guard. See [runtime](../concepts/runtime.md).

**The communicator.** `soma::Comm::world()` is every rank; in a build without MPI it is one rank,
and the same program runs unchanged. `Comm::self()` is a one-rank communicator. Reductions such
as `allreduce` are defined for `std::int32_t`, `std::int64_t`, `std::uint64_t` and `double`. See
[Comm](../parallel/comm.md).

**Building a mesh.** `soma::MeshBuilder` (`soma/core/mesh_builder.hpp`) takes vertices as gids
with interleaved `x, y, z` coordinates, and elements as gids with the gids of their corners
(`add_elements(Topology, gids, vertex_gids)`). Each rank adds the elements it holds and every
vertex they use; vertices on the cut between two ranks are added on both.
`soma::build_local_part(comm, std::move(builder))` (`soma/parallel/distributed.hpp`) is
collective: it builds this rank's part, finds which ranks share each vertex (the lowest rank
holding one owns it), and orders each dimension interior, shared and owned, shared copies, then
ghosts. For a mesh that only ever lives on one rank, `std::move(builder).build()` builds it
without that collective step. See
[building meshes](../concepts/building-meshes.md) and
[distributed meshes](../parallel/distributed-meshes.md).

**Reading a mesh.** `soma::read_file(comm, path)` (`soma/io/mesh_file.hpp`) picks the reader by
the file's content: a SOMA file or a MOAB `.h5m` file is read in parallel on every rank (both need
`SOMA_ENABLE_HDF5`); a legacy VTK file is read on the first rank and its cells dealt to the ranks
along a space-filling curve. It is collective and throws on every rank when it fails. See
[I/O](../io/overview.md).

**Querying the mesh.** `mesh.dim()` is the dimension of the cells and `mesh.count(d)` the number
of entities of dimension `d` this rank holds. `mesh.segment_table(d)` gives the ends of the
ownership ranges: lids `[0, owned_end)` are the entities this rank owns, so summing `owned_end`
over the ranks counts each entity once. `mesh.gid(d)` and `mesh.coords()` are host Views of the
gids and the coordinates. See [the mesh model](../concepts/mesh-model.md).

**Running a kernel.** `mesh.mirror<Space>()` copies the mesh into another memory space; by
default the copy is lean (all dimensions, coordinates and sets, no fields and no adjacency
graphs) and `soma::TransferOptions` (`soma/core/transfer.hpp`) asks for more. `view()` returns a
`soma::MeshView`, which a kernel captures by value; its members (`num_vertices`, `vertex`, `x`,
`gid`, `owned`, ...) are callable on the device. On a build without a GPU backend the default
memory space is the host's and the same code runs on host threads.

**Writing.** `soma::write_file(comm, mesh, path)` picks the writer by the extension: `.soma` is
written in parallel; `.h5m` and `.vtk` are serial formats, so the mesh is gathered on the first
rank and written there. For `.vtk`, every field VTK can hold is written unless
`WriteFileOptions::vtk_fields` names them. Any other extension throws `std::invalid_argument` on
every rank before a file is touched.

Errors are reported by exceptions; collective SOMA calls throw on every rank, so a
`try`/`catch` around them, as in the [example programs](examples.md), lets every rank exit
cleanly.

## Building it

With SOMA installed, a `CMakeLists.txt` next to `quickstart.cpp`:

```cmake
cmake_minimum_required(VERSION 3.20...3.24)
project(quickstart LANGUAGES CXX)

find_package(soma REQUIRED)

add_executable(quickstart quickstart.cpp)
target_link_libraries(quickstart PRIVATE soma::soma)
```

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/soma/prefix
cmake --build build
```

## Running it

```bash
./build/quickstart                      # one rank
mpiexec -n 4 ./build/quickstart         # four ranks (MPI build)
mpiexec -n 4 ./build/quickstart mesh.h5m
./build/quickstart --kokkos-num-threads=8
```

Whatever the number of ranks, the built grid has $8 \times 8 = 64$ cells and $9 \times 9 = 81$
vertices and an area of 1 up to rounding; the program writes `quickstart.vtk`, which ParaView
opens.

```{note}
The kernel above sits in `main`. With the CUDA backend, `KOKKOS_LAMBDA` cannot appear in a
private member function (a GoogleTest `TEST` body is one); put kernels in free functions, as
SOMA's own tests do.
```
