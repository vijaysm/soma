# Gallery

Figures from across these pages, each made by SOMA's own examples and tools; follow the links for
how each was made and what it shows.

## A 200 m mesh of the contiguous United States

[The CONUS application](../apps/conus-200m.md): a cubed sphere refined from 47 km to 200 m over
the contiguous United States by conforming refinement in five passes, with the mesh optimized as
each level is made, 50 million cells on 48 MPI ranks in under nine minutes.

```{figure} ../figures/img/conus-telescope.png
:alt: Six panels zooming in on the refined mesh over the contiguous United States, from the whole refined region to a window 4.7 km high at the corner of the 200 m region, each window outlined in the panel before it, the cells coloured by refinement level
:width: 100%

Zooming in from the whole refined region to the corner of the 200 m region, cells coloured by refinement level.
```

```{figure} ../figures/img/conus-min-angle.png
:alt: The refined region in stereographic projection coloured by each cell's least angle, near-right angles inside the region and cyan bands of cells under 72 degrees along the transitions between levels
:width: 100%

Each cell's least angle over the refined region; see [Quality](../apps/conus-200m.md#quality).
```

```{figure} ../figures/img/conus-transition-zoom.png
:alt: A close-up of the corner of the 200 m region coloured by least angle, three levels of cells joined by strips of transition cells drawn in cyan where their least angle is under 72 degrees
:width: 90%

The corner of the 200 m region: three levels and the transition templates between them.
```

```{figure} ../figures/img/conus-p3-p4.png
:alt: Four close-ups of one window at the edge of the 200 m region, with and without the area ratio bound, coloured by least angle and by area ratio to an edge neighbour
:width: 100%

The same window with the area ratio bound (top) and without it (bottom).
```

## Visualizing SOMA files

[ParaView, VisIt, pyvista and yt](../visualization/overview.md) read SOMA files directly, in
parallel, part by part.

```{figure} ../figures/img/visualization-paraview.png
:alt: Six ParaView renders of SOMA files: a hex box coloured by reading rank, the box clipped with pressure and velocity arrows, a cube of 64 bricks by material set, curved quadratic cells, a slice of a 3D AMR hierarchy, and a globe coloured by cell size
:width: 100%

[ParaView](../visualization/paraview.md), `pvbatch` on 4 and 8 ranks.
```

```{figure} ../figures/img/visualization-visit.png
:alt: Six VisIt renders of SOMA files: hex and tet boxes by pressure, linear cells of every kind, an ocean mesh by node id, a 2D AMR checkpoint, and a globe coloured by cell size
:width: 100%

[VisIt](../visualization/visit.md), on 4- and 8-rank engines.
```

```{figure} ../figures/img/visualization-pyvista-yt.png
:alt: Five pyvista renders and a yt slice of SOMA files: a hex box by pressure, 64 bricks by material set, curved quadratic cells, linear cells, a 2D AMR checkpoint with its boxes, and a slice of a 3D AMR field
:width: 100%

[pyvista and yt](../visualization/python.md).
```

## Algorithms

```{figure} ../figures/img/regional-levels.png
:alt: The contiguous United States coloured by refinement level: a yellow region of the finest cells ringed by a teal band inside purple coarse cells
:width: 70%

[Regional refinement](../algorithms/refinement.md) of a cubed sphere to 12 km.
```

```{figure} ../figures/img/regional-quality.png
:alt: Three close-ups of the same transition between levels coloured by least angle: as refined, after centroid smoothing, and after optimization, the transition cells turning lighter after optimization
:width: 100%

Least angles [as refined, after centroid smoothing and after optimization](../algorithms/refinement.md).
```

```{figure} ../figures/img/marked-refinement.png
:alt: A Y-shaped mesh refined towards one corner, non-conforming with hanging vertices on the left and conforming with transition triangles on the right
:width: 90%

[Marked refinement](../algorithms/refinement.md), non-conforming and conforming.
```

```{figure} ../figures/img/laplacian-smoothing.png
:alt: A perturbed grid of quads coloured by cell area, and the same grid after smoothing, now regular
:width: 80%

[Laplacian smoothing](../algorithms/smoothing.md) of a perturbed grid.
```

```{figure} ../figures/img/poisson-solution.png
:alt: The back half of a box of hexes coloured by the solution of a Poisson problem, zero on the outer faces and largest at the centre
:width: 60%

[Finite elements](../algorithms/fem.md): the Poisson example on hexes.
```

```{figure} ../figures/img/advected-bump.png
:alt: A bump of a field advected half a turn by an upwind scheme and by a bounded scheme, the upwind result wider and lower
:width: 100%

[Finite volumes](../algorithms/finite-volumes.md): upwind and bounded advection.
```

```{figure} ../figures/img/remap-field.png
:alt: A Y-shaped source mesh coloured by a smooth field, and the field remapped onto a grid of triangles covering it
:width: 90%

[Conservative remap](../algorithms/remap.md) in the plane.
```

```{figure} ../figures/img/remap-sphere.png
:alt: Two spheres coloured by the same smooth field, an icosphere of triangles and a cubed sphere of quads
:width: 80%

[Conservative remap](../algorithms/remap.md) on the sphere.
```

## Meshes and parallel meshes

```{figure} ../figures/img/box-mesh-gids.png
:alt: A box of 4 by 3 by 2 hexes, each coloured by its global id
:width: 60%

[A box mesh](../concepts/building-meshes.md) coloured by global id.
```

```{figure} ../figures/img/cell-and-vertex-fields.png
:alt: A plane mesh coloured by a field on its cells, one colour per cell, and by a field on its vertices, varying inside each cell
:width: 80%

[Fields](../concepts/fields.md) on cells and on vertices.
```

```{figure} ../figures/img/partition-4-ranks.png
:alt: A plane mesh divided into four parts coloured by owning rank
:width: 60%

[A mesh distributed](../parallel/distributed-meshes.md) over four ranks.
```
