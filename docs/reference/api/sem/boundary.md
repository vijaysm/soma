# `soma/sem/boundary.hpp`

`SemBoundary`, `make_boundary`, `remove_normal`: the boundary of a `SemSpace`: which GLL points lie on the mesh's
skin (a coast), the unit outward normal there in the tangent plane, and the no-slip points where the coast turns too
sharply to have one; `remove_normal` applies it to a vector field. See
[spectral elements](../../../algorithms/spectral-elements.md).

```{doxygenfile} soma/sem/boundary.hpp
```
