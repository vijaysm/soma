// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The extension module: one registration call per group of operations.

#include "common.hpp"

NB_MODULE(_soma, m) {
  m.doc() = "SOMA's compiled bindings; import the package soma instead.";
  soma_py::bind_runtime(m);
  soma_py::bind_mesh(m);
}
