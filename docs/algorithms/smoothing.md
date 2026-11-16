# Laplacian smoothing

`soma::laplacian_smooth` improves the shape of a mesh's cells by moving each free vertex to the
mean of its neighbours' coordinates. Use it to relax a mesh after moving its boundary,
refining it or reading it from a generator that leaves badly shaped cells. The result is the
same, bit for bit, on any number of ranks and threads.

## What it does

Each sweep is a Jacobi iteration: every free owned vertex moves to the mean of its neighbours'
coordinates as they were before the sweep. The algorithm is the one in MOAB's
`LaplacianSmoother` example.

The neighbours of a vertex are the other corners of the entities of dimension `bridge_dim`
around it:

- by default the cells, so an interior vertex of a hex mesh has 26 neighbours;
- with `bridge_dim = 1`, the edges, so only the vertices joined to it by an edge.

Neighbours are read off the cells' canonical sides (see [topology](../concepts/topology.md)),
so the mesh need not hold edges or faces. Each vertex sums its neighbours in ascending
gid order, which is what makes the result independent of the partition and thread count.

## API

Header: `soma/algorithms/laplacian.hpp`.

```cpp
struct SmoothOptions {
  int iterations = 1;
  int bridge_dim = -1;
  Real tolerance = 0;
  std::span<const std::uint8_t> fixed;
};

struct SmoothReport {
  int iterations = 0;     // sweeps made
  Real largest_move = 0;  // in the last sweep, over every rank; 0 unless tolerance > 0
};

template <class MemSpace>
SmoothReport laplacian_smooth(const Comm& comm, Mesh<MemSpace>& mesh, const SmoothOptions& options = {});
```

| Option | Default | Meaning |
| --- | --- | --- |
| `iterations` | `1` | Number of sweeps to make, at most. |
| `bridge_dim` | `-1` | Dimension of the entities that make two vertices neighbours: `-1` for the mesh's dimension (the cells), otherwise 1 to `dim`. |
| `tolerance` | `0` | If greater than 0, stop once no vertex moved farther than this in a sweep. A length in the mesh's units. |
| `fixed` | empty | One entry per vertex (by lid); a nonzero entry keeps that vertex where it is. Empty: the skin stays fixed. |

`SmoothReport::iterations` is the number of sweeps made; it is less than `options.iterations`
when the tolerance stopped the iteration early. `largest_move` is the largest distance a
vertex moved in the last sweep, over every rank, and is computed only when `tolerance > 0`.

## Usage

From `examples/laplacian_smooth.cpp`, which reads a mesh on any number of ranks, smooths it
and writes it back:

```cpp
soma::HostMesh mesh = soma::read_file(comm, in);
mesh.add_ghosts(comm, soma::GhostOptions{1, 0, false});
soma::SmoothOptions options;
options.iterations = 10;
const soma::SmoothReport report = soma::laplacian_smooth(comm, mesh, options);
mesh.remove_ghosts(comm);
soma::write_file(comm, mesh, out, {.vtk_fields = std::vector<std::string>{}});
```

The example takes `--iterations K`, `--bridge-dim B` and `--tolerance T`:

```bash
mpiexec -n 4 ./laplacian_smooth --iterations 20 --tolerance 1e-8 in.h5m out.soma
```

## Boundary handling

Fixed vertices do not move. They are:

- the vertices whose entry in `options.fixed` is nonzero, if a mask is given;
- otherwise the skin: the corners of the sides of dimension `dim - 1` that only one cell has.

A vertex with no neighbours also stays where it is. To keep a different set of vertices fixed
(for example, to let boundary vertices slide), pass a mask with one entry per vertex of the
local part, ghosts included.

## Parallel behaviour

`laplacian_smooth` is collective: every rank must call it, with the same `iterations`,
`bridge_dim` and `tolerance`.

On more than one rank the mesh must hold a ghost layer through vertices,
`GhostOptions{1, 0}` (see [ghosts and halos](../parallel/ghosts-and-halos.md)). Every cell
around a vertex the rank holds is then also held, so all of a vertex's neighbours are present
and the skin found on each rank is the global one. Owners move their vertices; after every
sweep a halo update gives shared copies and ghosts their owners' new coordinates. On return,
copies and ghosts therefore hold their owners' coordinates, which is what
[`FemSpace`](fem.md) expects of a mesh whose coordinates have moved.

Because each vertex sums its neighbours in gid order and only owners compute, the coordinates
after smoothing are the same, bit for bit, on any number of ranks and threads. This holds
whether or not `soma::reproducible()` is on (see [reproducible sums](../parallel/reproducible-sums.md)).

## Memory spaces

`laplacian_smooth` accepts a mesh in any memory space. The neighbour graph and the skin are
built on the host (a device mesh is mirrored to the host for this), once per call; the sweeps
then run as Kokkos kernels in the mesh's memory space.

## Errors

`laplacian_smooth` throws `std::invalid_argument` on every rank if any rank passes:

- a negative iteration count;
- a `bridge_dim` other than `-1` or a value in `[1, dim]`;
- a negative or non-finite tolerance;
- a `fixed` mask whose length is not the vertex count;
- a mesh of dimension 0, or an edited mesh (changed by `MeshEdit::apply` and not compressed
  since with `compress()`; see [editing](../concepts/editing.md));
- on more than one rank, a mesh without a ghost layer through vertices;

or if ranks differ in `iterations`, `bridge_dim` or `tolerance`. All ranks take part in the
check (one reduction), so no rank is left waiting.
