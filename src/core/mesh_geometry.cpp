// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/core/mesh_geometry.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace soma {

namespace {

constexpr double layout_version = 1;
// A sphere's vertices differ in their distance from its centre by at most
// this much of it: twice what SemSpace allows one of its corners.
constexpr Real sphere_tolerance = 2e-8;
constexpr Real flat_tolerance = 1e-12;

// Unsigned integers in the order of the doubles, 0 below all of them.
std::uint64_t key_of(double v) {
  constexpr std::uint64_t sign = std::uint64_t{1} << 63;
  const auto b = std::bit_cast<std::uint64_t>(v);
  return (b & sign) != 0 ? ~b : b | sign;
}

double value_of(std::uint64_t k) {
  constexpr std::uint64_t sign = std::uint64_t{1} << 63;
  return std::bit_cast<double>((k & sign) != 0 ? k & ~sign : ~k);
}

}  // namespace

MeshGeometry MeshGeometry::sphere(Real radius, const std::array<Real, 3>& center) {
  if (!(radius > 0) || !std::isfinite(radius) || !std::all_of(center.begin(), center.end(), [](Real c) { return std::isfinite(c); })) {
    throw std::invalid_argument("MeshGeometry::sphere: the radius must be positive and finite, the centre finite");
  }
  return MeshGeometry{Kind::sphere, center, radius, true};
}

void set_geometry(MeshValues& values, const MeshGeometry& g) {
  std::vector<double> w{layout_version, static_cast<double>(g.kind), g.declared ? 1.0 : 0.0};
  if (g.kind == MeshGeometry::Kind::sphere) {
    (void)MeshGeometry::sphere(g.radius, g.center);
    w.insert(w.end(), {static_cast<double>(g.center[0]), static_cast<double>(g.center[1]),
                       static_cast<double>(g.center[2]), static_cast<double>(g.radius)});
  }
  MeshValue v;
  v.type = DataType::f64;
  v.ncomp = static_cast<int>(w.size());
  v.format.storage_class = 3;
  v.global_value = MeshValue::bytes_of<double>(w);
  values.set(std::string(geometry_value), std::move(v));
}

std::optional<MeshGeometry> geometry(const MeshValues& values) {
  if (!values.contains(geometry_value)) return std::nullopt;
  const auto bad = [](const char* what) {
    return std::invalid_argument("the mesh value soma:geometry " + std::string(what));
  };
  const MeshValue& v = values.at(geometry_value);
  if (v.type != DataType::f64) throw bad("is not f64");
  const auto w = v.global_as<double>();
  if (!w || w->size() < 3) throw bad("is cut short");
  if ((*w)[0] != layout_version) return std::nullopt;
  MeshGeometry g;
  g.declared = (*w)[2] != 0;
  if ((*w)[1] == static_cast<double>(MeshGeometry::Kind::affine)) return g;
  if ((*w)[1] != static_cast<double>(MeshGeometry::Kind::sphere)) return std::nullopt;
  if (w->size() < 7) throw bad("is cut short");
  const bool declared = g.declared;
  g = MeshGeometry::sphere(static_cast<Real>((*w)[6]), {static_cast<Real>((*w)[3]), static_cast<Real>((*w)[4]), static_cast<Real>((*w)[5])});
  g.declared = declared;
  return g;
}

void detail::GeometryExtent::add(Real x, Real y, Real z) {
  const auto r = static_cast<double>(std::sqrt(x * x + y * y + z * z));
  if (!std::isfinite(r)) {
    order_keys[4] = 1;
    return;
  }
  const std::array<double, 4> v{r, -r, static_cast<double>(z), -static_cast<double>(z)};
  for (std::size_t i = 0; i < v.size(); ++i) order_keys[i] = std::max(order_keys[i], key_of(v[i]));
}

std::optional<MeshGeometry> detail::GeometryExtent::recognize() const {
  if (order_keys[4] != 0 || order_keys[0] == 0) return std::nullopt;
  const double far = value_of(order_keys[0]), near = -value_of(order_keys[1]);
  const double top = value_of(order_keys[2]), bottom = -value_of(order_keys[3]);
  MeshGeometry g;
  if (top - bottom <= flat_tolerance * far) {
    g = MeshGeometry::affine();
  } else if (near > 0 && far - near <= sphere_tolerance * near) {
    g = MeshGeometry::sphere(static_cast<Real>(near + (far - near) / 2));
  } else {
    return std::nullopt;
  }
  g.declared = false;
  return g;
}

}  // namespace soma
