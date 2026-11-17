// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The C API's file I/O.

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "handles.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/io/soma_file.hpp"

namespace {

std::filesystem::path path_of(const char* path, const char* what) {
  if (path == nullptr) throw std::invalid_argument(std::string(what) + ": a null path");
  return path;
}

}  // namespace

using soma::c::call;
using soma::c::need;
using soma::c::out;

extern "C" {

int soma_read_file(const soma_comm_t* comm, const char* path, soma_mesh_t** result) {
  return call([&] {
    constexpr const char* what = "soma_read_file";
    const soma::Comm& c = *need(comm, what).comm;
    soma_mesh_t*& r = out(result, what);
    r = nullptr;
    std::filesystem::path p;
    soma::c::agree(c, what, [&] {
      p = path_of(path, what);
      if (!std::filesystem::exists(p)) throw std::invalid_argument(std::string(what) + ": no file " + p.string());
    });
    r = soma_c_new_mesh(soma::read_file(c, p));
  });
}

int soma_write_file(const soma_comm_t* comm, const soma_mesh_t* mesh, const char* path, int nvtk,
                    const char* const* vtk_fields) {
  return call([&] {
    constexpr const char* what = "soma_write_file";
    const soma::Comm& c = *need(comm, what).comm;
    const soma_mesh* m = nullptr;
    std::filesystem::path p;
    soma::WriteFileOptions options;
    soma::c::agree(c, what, [&] {
      m = &need(mesh, what);
      p = path_of(path, what);
      if (nvtk >= 0) {
        soma::c::check_array(vtk_fields, nvtk, what);
        std::vector<std::string> names;
        for (int i = 0; i < nvtk; ++i) {
          if (vtk_fields[i] == nullptr) throw std::invalid_argument(std::string(what) + ": a null field name");
          names.emplace_back(vtk_fields[i]);
        }
        options.vtk_fields = std::move(names);
      } else if (nvtk != -1) {
        throw std::invalid_argument(std::string(what) + ": nvtk " + std::to_string(nvtk));
      }
    });
    soma::write_file(c, m->get(), p, options);
  });
}

int soma_detect_format(const char* path, int* format) {
  return call([&] {
    constexpr const char* what = "soma_detect_format";
    out(format, what) = static_cast<int>(soma::detect_format(path_of(path, what)));
  });
}

int soma_convert_mesh_file(const soma_comm_t* comm, const char* src, const char* dst, int compression, int* written) {
  return call([&] {
    constexpr const char* what = "soma_convert_mesh_file";
    const soma::Comm& c = *need(comm, what).comm;
    int& w = out(written, what);
    std::filesystem::path in, to;
    soma::c::agree(c, what, [&] {
      in = path_of(src, what);
      to = path_of(dst, what);
      if (!std::filesystem::exists(in)) throw std::invalid_argument(std::string(what) + ": no file " + in.string());
    });
    w = static_cast<int>(soma::convert_mesh_file(c, in, to, compression));
  });
}

}  // extern "C"
