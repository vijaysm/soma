// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Helper process of test_vtk_writer. It lowers its own file size limit, so that
// a write fails part-way through, and tries to write a mesh twice:
//   vtk_limit_child <existing-file> <new-file> <limit-bytes>
// The exit code is 0 when both writes threw std::runtime_error naming their
// path, and otherwise says which expectation failed; any other exception ends
// the process abnormally. The limit is a property of
// this process alone; the test that starts it keeps no global state.

#include <csignal>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/resource.h>

#include <Kokkos_Core.hpp>

#include "mesh_fixture.hpp"
#include "soma/core/mesh.hpp"
#include "soma/io/vtk_writer.hpp"

namespace {

enum Exit : int {
  failed_as_expected = 0,
  wrote_without_error = 10,
  wrong_error = 11,
  usage = 20,
  limit_not_set = 21,
};

int attempt(const soma::HostMesh& mesh, const std::filesystem::path& path,
            const std::vector<std::string>& names, std::string& note) {
  try {
    soma::write_vtk(mesh, path, names);
  } catch (const std::runtime_error& e) {
    if (std::string(e.what()).find(path.string()) == std::string::npos) {
      note = std::string("the message does not name the path: ") + e.what();
      return wrong_error;
    }
    return failed_as_expected;
  }
  note = "write_vtk returned normally";
  return wrote_without_error;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: vtk_limit_child <existing-file> <new-file> <limit-bytes>\n";
    return usage;
  }
  // Kokkos only: this process must not join the MPI job of the test that starts it.
  const Kokkos::ScopeGuard kokkos(argc, argv);
  auto mesh = soma_test::two_quads_and_a_triangle(soma::Ordering::file);
  const auto f = mesh.fields().add_tag<double>("t", 0, 3);
  for (std::size_t r = 0; r < f.view().extent(0); ++r) {
    for (std::size_t c = 0; c < 3; ++c) f.view()(r, c) = static_cast<double>(r) / 7.0 + static_cast<double>(c);
  }
  const std::vector<std::string> names{"t"};

  std::signal(SIGXFSZ, SIG_IGN);
  rlimit lowered{};
  getrlimit(RLIMIT_FSIZE, &lowered);
  lowered.rlim_cur = static_cast<rlim_t>(std::strtoull(argv[3], nullptr, 10));
  if (setrlimit(RLIMIT_FSIZE, &lowered) != 0) return limit_not_set;

  std::string first_note;
  std::string second_note;
  const int first = attempt(mesh, argv[1], names, first_note);
  const int second = attempt(mesh, argv[2], names, second_note);

  if (first != failed_as_expected) std::cerr << argv[1] << ": " << first_note << '\n';
  if (second != failed_as_expected) std::cerr << argv[2] << ": " << second_note << '\n';
  return first != failed_as_expected ? first : second;
}
