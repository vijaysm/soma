# `soma/runtime/scope_guard.hpp`

`ScopeGuard`: reads SOMA's options, then initializes MPI (in MPI builds) and Kokkos unless they run
already, and stops what it started.

```{doxygenfile} soma/runtime/scope_guard.hpp
```
