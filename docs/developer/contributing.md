# Contributing

The conventions SOMA's code, tests, documentation and history follow. They are taken from the code
as it is; when in doubt, match the file you are editing. See [architecture](architecture.md) for
how the library is organised and [building and testing](building-and-testing.md) for how to build
and run the tests.

## Code style

- **Language.** C++20 (`CMAKE_CXX_STANDARD 20`, no compiler extensions). SOMA targets are built
  with `-Wall -Wextra` where the compiler accepts them; new code should not add warnings.
- **Formatting.** Two-space indentation, access specifiers indented by one space (` public:`),
  lines of at most 120 columns, and `}  // namespace soma` closing comments. The repository has no
  `.clang-format`; format by hand to match the surrounding code rather than running a formatter
  over whole files.
- **File header.** SOMA's sources, headers and CMake files start with an SPDX line and a
  copyright line (with `#` comments in CMake files); give every new file the same two lines:

  ```cpp
  // SPDX-License-Identifier: BSD-3-Clause
  // Copyright (c) 2022-2026, ...
  ```

- **Include guards.** Headers use `#ifndef SOMA_<DIR>_<NAME>_HPP_` / `#define` / `#endif  // SOMA_<DIR>_<NAME>_HPP_`,
  for example `SOMA_AMR_BOX_HPP_` in `include/soma/amr/box.hpp`. Headers are included by their
  path from `include/`: `#include "soma/amr/box.hpp"`.
- **Naming.** Types in `CamelCase` (`MeshBuilder`, `HaloPlan`); functions, variables and
  enumerators in `snake_case` (`build_adjacencies`, `Ownership::shared_copy`, `Curve::morton`);
  private data members with a trailing underscore (`boxes_`); macros with a `SOMA_` prefix.
  Public declarations live in `soma` or a sub-namespace (`soma::amr`, `soma::transport`,
  `soma::compat`); implementation details go in a `detail` namespace.
- **Spelling.** Comments and messages use British spelling (neighbour, centre, normalise).
- **Kernels.** Kernels are `KOKKOS_LAMBDA`s or `KOKKOS_INLINE_FUNCTION`s and get a label naming
  where they come from (`"soma::cg::zero"`, `"amr::copy"`). They capture device-copyable handles
  (`MeshView`, views, `LevelView`), never owning objects such as `Mesh`. Only
  `include/soma/parallel/comm.hpp`, `src/parallel/comm_*.cpp` and `src/runtime/scope_guard.cpp`
  may test `SOMA_ENABLE_MPI`.
- **Errors.** Throw standard exceptions (`std::invalid_argument` for bad input,
  `std::runtime_error` for the environment, `std::logic_error` for a call in the wrong state) with a
  message naming the function and what was wrong. A collective function that rejects input on one
  rank must throw on every rank; see [collectives and errors](architecture.md#collectives-and-errors).

### Header comments are the API contract

Each public header opens, after its include guard, with a prose comment saying what it provides,
how to use it (often with a short indented example) and the behaviour a caller relies on: which
calls are collective, what throws and on which ranks, determinism, tolerances and units, and
memory spaces. Each declaration has a comment of its own with the same kind of contract.
`include/soma/amr/halo.hpp` is a typical example:

```cpp
// Same-level halo exchange. A HaloPlan, computed once for a level's boxes, an
// index type and a ghost width, lists the copies that fill the ghost points
// of every box from the valid points of the others (faces, edges and
// corners), across periodic directions too, ...
```

These comments are the authoritative description of the API; the pages under `docs/` summarise
them. When a change alters behaviour, update the comment in the same change.

## Adding a test

Tests use GoogleTest. Put a unit test in `tests/unit/test_<topic>.cpp` and register it in
`tests/CMakeLists.txt` with `soma_add_test`:

```cmake
# soma_add_test(<name> SOURCES ... [RANKS 1 2 4] [LIBS ...])
soma_add_test(test_partition SOURCES unit/test_partition.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_point_key SOURCES unit/test_point_key.cpp)
```

- Without `RANKS`, or in a build without MPI, the test cases are discovered individually by CTest.
- In an MPI build with `RANKS`, the executable is registered once per rank count, as
  `<name>.np<n>`, run under `mpiexec` with a 60-second timeout. Code that communicates should be
  tested at several rank counts, including odd ones; the existing tests use `1 2 3 4 5 7 8`.
- `LIBS` adds libraries: `soma_compat` for the compat layer, `${SOMA_MOAB_TARGET}` for tests that
  compare with MOAB (these go in `tests/reference/` and are registered only when
  `SOMA_ENABLE_MOAB_REFERENCE` is on).
- Every test links a shared `main` that holds a `ScopeGuard` (so Kokkos and MPI are initialized
  before any view is made) and fails the run on every rank if any rank fails.
- Test data are read through the `SOMA_TEST_DATA_DIR` definition, which names `data/`.
- Kernels in tests go in named free functions, not in `TEST` bodies: CUDA refuses device lambdas
  there, and the test `policy.device_lambdas` rejects them.

For a correctness change, add a test that fails without the change. Where MOAB defines the same
result, a reference test against MOAB is the stronger check.

## Documentation

The documentation is Sphinx with MyST Markdown, one page per topic under `docs/`, organised as the
table of contents in `docs/index.md`. Build it with warnings as errors before submitting:

```bash
python3 -m pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html
```

Pages describe the library as it is. Name only types, functions, options and behaviour that exist,
quote signatures from the headers, and prefer snippets adapted from `examples/`, `tests/` or
`bench/`, naming the file they come from. Cross-link pages with relative links.

## Changelog

`docs/CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Changes are
recorded under the unreleased version, grouped by component (`### Build`, `### Mesh`,
`### Patch-based AMR`, `### Algorithms`, ...), as short entries that say what a user can now do or
what behaves differently. A user-visible change adds or updates its entry in the same commit. The
changelog is also published as the [changelog page](../reference/changelog.md).

## Commit messages

A commit message is a one-line header saying what the change does, in the imperative
("Add rebalance and Doerfler and threshold marking"). If needed, follow it with a blank line and
one or two short lines on why or on what matters, for example:

```text
Add ArborX's bounding volume hierarchy as an optional remap search

SOMA_ENABLE_ARBORX adds ArborXIndex (GridIndex's pairs and order) and RemapOptions::search; the grid stays the default.
```
