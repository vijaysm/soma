# `soma/algorithms/conformity.hpp`

`detail::nonconforming_sides`: the sides of a mesh that tile without matching (hanging vertices, a quad
against two triangles); `detail::split_sides`: the hex faces two triangles tile, as the composite hex
takes them; `detail::conforming_skin`: the boundary, sides between such cells left out. See
[finite volumes](../../../algorithms/finite-volumes.md) and [finite elements](../../../algorithms/fem.md).

```{doxygenfile} soma/algorithms/conformity.hpp
```
