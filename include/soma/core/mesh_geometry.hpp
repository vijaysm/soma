// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_MESH_GEOMETRY_HPP_
#define SOMA_CORE_MESH_GEOMETRY_HPP_

/// \file
/// The analytic surface a mesh of dimension 2 discretizes, kept as the mesh
/// value "soma:geometry" and so carried by mirrors, merges, migration and
/// files:
///
/// ```
///   soma::set_geometry(mesh.values(), soma::MeshGeometry::sphere(6.371e6));
///   if (const auto g = soma::geometry(mesh.values())) ...
/// ```
///
/// soma::sem::cubed_sphere_mesh and SemSpace declare theirs, and write_soma
/// writes it, or what it recognizes from the vertices when none is declared
/// (SomaWriteOptions::geometry). The value is f64: the layout's version (1),
/// the kind, 1 if declared or 0 if recognized, then the kind's parameters
/// (sphere: centre, radius). A reader that does not know the version or kind
/// takes the mesh to have none.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include "soma/core/mesh_values.hpp"
#include "soma/core/types.hpp"

namespace soma {

inline constexpr std::string_view geometry_value = "soma:geometry";

struct MeshGeometry {
  /// affine: the elements' own maps of their nodes, on no other surface.
  enum class Kind : std::uint8_t { affine = 0, sphere = 1 };
  Kind kind = Kind::affine;
  std::array<Real, 3> center{};
  Real radius = 0;
  bool declared = true;  ///< by the application, rather than recognized from the vertices

  static MeshGeometry affine() noexcept { return MeshGeometry{}; }
  /// Throws std::invalid_argument for a radius not positive and finite or a
  /// centre not finite.
  static MeshGeometry sphere(Real radius, const std::array<Real, 3>& center = {});

  friend bool operator==(const MeshGeometry&, const MeshGeometry&) = default;
};

/// Sets the value, replacing any other geometry. The same on every rank.
void set_geometry(MeshValues& values, const MeshGeometry& g);

/// The geometry held, if there is one of a version and kind known here.
/// Throws std::invalid_argument when the value is cut short or not f64, or
/// its sphere has no positive, finite radius.
std::optional<MeshGeometry> geometry(const MeshValues& values);

namespace detail {

/// The vertices' extent by which a mesh's surface is recognized: each field
/// the largest over the vertices (0: there are none), so ranks' combine by max.
/// Flat in z comes before the sphere: the corners of a square are on one, too.
struct GeometryExtent {
  /// Of r, -r, z and -z, integers in the order of the doubles, then whether a
  /// vertex is not finite.
  std::array<std::uint64_t, 5> order_keys{};

  void add(Real x, Real y, Real z);
  std::optional<MeshGeometry> recognize() const;
};

}  // namespace detail

}  // namespace soma

#endif  // SOMA_CORE_MESH_GEOMETRY_HPP_
