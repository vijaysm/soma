// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// What soma_figure_data's sets share: the checks, the writers and the registry
// of sets. Each area of the documentation registers its sets in a file of its
// own (sets_concepts.cpp, sets_parallel.cpp, sets_algorithms.cpp), so the
// areas can grow independently. See figure_data.cpp.

#ifndef SOMA_DOCS_FIGURES_FIGURE_DATA_HPP_
#define SOMA_DOCS_FIGURES_FIGURE_DATA_HPP_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

#include "soma/core/mesh.hpp"
#include "soma/core/segments.hpp"
#include "soma/parallel/comm.hpp"

namespace figures {

namespace fs = std::filesystem;

// A failed check of a set's data; main prints it and exits 1.
struct SetError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Throws a SetError naming the check unless ok.
void require(bool ok, const std::string& what);

// A cell field from a function of the cell's local id.
void put_cells(soma::HostMesh& mesh, const std::string& name, const std::function<double(soma::LO)>& value);

// The text of a JSON file, written on the first rank.
void write_json(const soma::Comm& comm, const fs::path& path, const std::string& text);

// A JSON list of a view's or vector's values.
template <class V>
std::string json_list(const V& v, std::size_t begin, std::size_t end) {
  std::ostringstream js;
  js << "[";
  for (std::size_t i = begin; i < end; ++i) js << (i > begin ? ", " : "") << v(i);
  js << "]";
  return js.str();
}

const char* ownership_name(soma::Ownership o);

// A set: the rank count it runs on and the function that writes its files.
struct Set {
  int ranks;
  std::function<void(const soma::Comm&, const fs::path&)> make;
};
using Sets = std::map<std::string, Set>;

// Each area adds its sets.
void add_concept_sets(Sets& sets);
void add_parallel_sets(Sets& sets);
void add_algorithm_sets(Sets& sets);

}  // namespace figures

#endif  // SOMA_DOCS_FIGURES_FIGURE_DATA_HPP_
