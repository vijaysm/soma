// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_IO_SOMA_AMR_HPP_
#define SOMA_IO_SOMA_AMR_HPP_

/// \file
/// Checkpoints of AMR hierarchies in the SOMA format, written from any number
/// of ranks and read back at any number. Needs a build with SOMA_ENABLE_HDF5;
/// without it these throw std::runtime_error.
///
/// The file's root group carries an attribute `amr` instead of a mesh's
/// `meta`: the hierarchy's geometry and options, every level's boxes in curve
/// order, and per field its name, value type, index type, components and
/// ghost width, with the valid points of every box of every level, box after
/// box in curve order (each box i fastest, then j, k, then the component).
/// Ghost points are not written. Part p of a file is what rank p owned, so
/// the rows of the parts follow the curve.
///
/// ```
///   soma::write_soma(comm, h, {{"density", rho}, {"velocity", u}}, "run.soma");
///   auto c = soma::read_soma_amr(comm, "run.soma");    // any rank count
///   const auto& rho = c.field<double>("density");
///   c.hierarchy.fill_halo(rho, amr::Interp::linear, boundary);
/// ```
///
/// write_soma is collective, checks everything on every rank first (fields
/// of the hierarchy, distributed as its levels, the same names and kinds on
/// every rank; a deflate level in [0, 9]) and throws std::invalid_argument
/// on every rank before touching a file; then it writes a temporary beside
/// `path` and renames it over `path`, or throws std::runtime_error naming the
/// path on every rank. read_soma_amr rebuilds the hierarchy over the reading
/// communicator, cut along the curve by `options.cost` (point counts by
/// default), each rank reading one run of rows per field and level; the
/// fields' ghost points are zero until the caller fills them. A file that is
/// not an AMR checkpoint, a newer format_version, or datasets that disagree
/// with the boxes throw std::runtime_error on every rank.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "soma/amr/hierarchy.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

/// The value types a checkpointed field may hold, in DataType order.
template <template <class> class F>
using AmrValues = std::variant<F<std::int8_t>, F<std::uint8_t>, F<std::int32_t>, F<std::uint32_t>, F<std::int64_t>,
                               F<std::uint64_t>, F<float>, F<double>>;

template <class T>
using AmrFieldPointer = const amr::MultiLevelField<T>*;

/// A field to write: its name and every level of it.
struct AmrFieldRef {
  std::string name;
  AmrValues<AmrFieldPointer> levels;

  template <class T>
  AmrFieldRef(std::string n, const amr::MultiLevelField<T>& f) : name(std::move(n)), levels(&f) {}
};

/// A field read back.
struct AmrField {
  std::string name;
  AmrValues<amr::MultiLevelField> levels;
};

struct AmrCheckpoint {
  amr::Hierarchy hierarchy;
  std::vector<AmrField> fields;

  /// The field named `name`, of value type T; throws std::invalid_argument
  /// when there is none.
  template <class T>
  const amr::MultiLevelField<T>& field(const std::string& name) const {
    for (const AmrField& f : fields) {
      if (f.name != name) continue;
      if (const auto* u = std::get_if<amr::MultiLevelField<T>>(&f.levels)) return *u;
      throw std::invalid_argument("AmrCheckpoint::field: " + name + " holds another value type");
    }
    throw std::invalid_argument("AmrCheckpoint::field: no field " + name);
  }
};

struct SomaAmrReadOptions {
  /// The cost of a box of a level for the cut over the reading ranks (empty:
  /// its point count); kept in the hierarchy's options.
  std::function<double(int level, const amr::Box& box)> cost;
};

void write_soma(const Comm& comm, const amr::Hierarchy& hierarchy, const std::vector<AmrFieldRef>& fields,
                const std::filesystem::path& path, const SomaWriteOptions& options = {});

AmrCheckpoint read_soma_amr(const Comm& comm, const std::filesystem::path& path, const SomaAmrReadOptions& options = {});

}  // namespace soma

#endif  // SOMA_IO_SOMA_AMR_HPP_
