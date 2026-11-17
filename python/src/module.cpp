// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The extension module: one registration call per group of operations.

#include "common.hpp"

NB_MODULE(_soma, m) {
  m.doc() = "SOMA's compiled bindings; import the package soma instead.";
  soma_py::bind_runtime(m);
  auto mesh = soma_py::bind_mesh(m);
  soma_py::bind_edit(m, mesh);
  soma_py::bind_fields(m, mesh);
  soma_py::bind_io(m);
  soma_py::bind_parallel(m, mesh);
  soma_py::bind_algorithms(m);
}
