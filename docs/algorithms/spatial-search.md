# Spatial search

SOMA has two indexes over axis-aligned boxes that answer the same question: for each query
box, which of the indexed boxes does it meet? `soma::GridIndex` bins the boxes in a uniform
grid and is always available; `soma::ArborXIndex` uses ArborX's bounding volume hierarchy and
needs SOMA built with ArborX. Both give the same pairs in the same order.
[Conservative remap](remap.md) uses them to find candidate overlapping cells.

## What "meets" means

Two boxes meet when, on every axis, each one's low end is at most the other's high end plus
the tolerance:

$$
q^{\text{lo}}_a \le b^{\text{hi}}_a + \varepsilon
\quad\text{and}\quad
b^{\text{lo}}_a \le q^{\text{hi}}_a + \varepsilon
\qquad \text{for } a \in \{x, y, z\}.
$$

Boxes that touch at a face, edge or corner therefore meet even with $\varepsilon = 0$. The
tolerance is a length in the boxes' units, finite and not negative.

## Box layout and results

Boxes and queries are `Kokkos::View<const Real**, Kokkos::LayoutLeft, MemSpace>` with 6
columns: low $x, y, z$, then high $x, y, z$. A query returns the pairs in compressed sparse
rows:

```cpp
struct Pairs {
  View1<Offset, MemSpace> offsets;  // queries + 1
  View1<LO, MemSpace> boxes;
};
```

The boxes query `q` meets are `pairs.boxes(k)` for `k` in
`[pairs.offsets(q), pairs.offsets(q + 1))`, each once and in ascending order. The result is the
same at any thread count.

The indexes do not copy the boxes: the view passed to the constructor must outlive the index.
Both are built and queried by Kokkos kernels in `MemSpace`'s execution space.

## `GridIndex`

Header: `soma/algorithms/grid_index.hpp`.

```cpp
template <class MemSpace>
class GridIndex {
 public:
  using Boxes = Kokkos::View<const Real**, Kokkos::LayoutLeft, MemSpace>;
  explicit GridIndex(const Boxes& boxes, Real tolerance = 0);
  LO size() const;
  Real tolerance() const;
  std::array<int, 3> bins() const;
  Pairs query(const Boxes& queries) const;
};
```

```cpp
const soma::GridIndex<Mem> index(boxes, tolerance);  // n x 6
const auto pairs = index.query(queries);              // m x 6
```

The grid spans the boxes. A bin's width on each axis is about the boxes' mean extent along
it, with one bin across an axis the boxes do not extend along (such as $z$ for a planar mesh),
and there are at most twice as many bins as boxes. A box goes into every bin it covers. A
query visits the bins covered by its box widened by the tolerance, and reports a box only from
the bin holding the low corner of the two boxes' overlap, so each pair is reported once.

Since bins are about the mean extent wide, a box much larger than the mean goes into many
bins; the grid suits boxes of similar size, as the cells of one mesh usually are.

The constructor throws `std::invalid_argument` unless the boxes have 6 columns, the tolerance is
finite and not negative, and every box is finite with its low ends at most its high ends.
`query` throws `std::invalid_argument` unless the queries have 6 columns.

## `ArborXIndex`

Header: `soma/algorithms/arborx_index.hpp`. Needs SOMA configured with
`-DSOMA_ENABLE_ARBORX=ON` (ArborX built with the same installed Kokkos SOMA uses; see
[CMake options](../reference/cmake-options.md)). Without it the header declares nothing.

```cpp
template <class MemSpace>
class ArborXIndex {
 public:
  using Boxes = typename GridIndex<MemSpace>::Boxes;
  using Pairs = typename GridIndex<MemSpace>::Pairs;
  explicit ArborXIndex(const Boxes& boxes, Real tolerance = 0);
  LO size() const;
  Real tolerance() const;
  Pairs query(const Boxes& queries) const;
};
```

It returns the same boxes, pairs and order as `GridIndex`. The tree holds the boxes widened by
the tolerance and two units in the last place more, so that it finds every pair that meets;
the exact test above then decides. A query box that is not finite meets nothing. The
constructor and `query` throw as `GridIndex`'s do.

## Choosing between them

Because the two return identical results, the choice is only one of cost and availability:

- `GridIndex` needs nothing beyond Kokkos and is the default in remap.
- `ArborXIndex` builds a tree rather than bins of one width, so it does not rely on the boxes
  being of similar size; it needs SOMA built with ArborX.

In `RemapOperator`, `RemapOptions::search` selects the index (`SpatialSearch::grid`, the
default, or `SpatialSearch::arborx`); the operator is the same bit for bit either way. The
changelog records that between uniform grids the two take about the same time, since clipping
dominates. Asking for `SpatialSearch::arborx` in a SOMA built without ArborX throws
`std::invalid_argument` on every rank. See [conservative remap](remap.md).
