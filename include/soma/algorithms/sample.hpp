// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_SAMPLE_HPP_
#define SOMA_ALGORITHMS_SAMPLE_HPP_

/// \file
/// Sampling a function on a mesh, for initial states and convergence studies:
///
/// ```
///   auto u = mesh.fields().add_tag<double>("u", 2, 1);
///   const auto f = soma::pointwise([](double x, double y, double) { return std::sin(x) * y; });
///   soma::sample(mesh, u, f);                                     // at each cell's centre
///   soma::sample(mesh, u, f, {.mode = soma::Sampling::average});  // cell averages
/// ```
///
/// The function is a field rule (soma/fields/field_rule.hpp), the field's own
/// when none is given, called once per dimension of the field's domain with
/// every entity, element node or quadrature point at once.
///
/// - Sampling::point: a per-entity field gets the rule of each member entity
///   (its corners; soma::pointwise evaluates at their average); an
///   element_nodes field (GLL or other nodes) gets the rule at each node,
///   placed by the element's map.
/// - Sampling::average and Sampling::integral: a per-entity field of
///   floating values gets, for each member entity of dimension 1 or more,
///   (1/|e|) int_e f or int_e f by a quadrature exact for polynomials of
///   `degree` on the reference cell, through the element's map: linear on
///   simplices, multilinear on quads, hexes and prisms, rational on pyramids,
///   biquadratic on quads of 8 or 9 nodes (as SemSpace), polygons as
///   triangles about their corners' average (polyhedra are not mapped yet).
///   On a mesh whose
///   surface is a sphere (soma/core/mesh_geometry.hpp), curves and surfaces
///   are mapped onto it, so the measure is the sphere's. Vertices get the
///   rule's value (average) or throw (integral). The rule is then called
///   with points (dim 0, no gids), so it must be a function of position.
///
/// Every member is evaluated, ghosts included, the same way on every rank,
/// so copies agree without communication. Host meshes only for now.

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"
#include "soma/fields/field.hpp"
#include "soma/fields/field_rule.hpp"
#include "soma/fields/registry.hpp"
#include "soma/fields/space.hpp"

namespace soma {

/// The storage of an EntityBatch: batch() views it, so keep it alive while
/// the batch is used.
struct EntityBatchData {
  int dim = 0;
  std::vector<Topology> topology;  ///< empty for dim 0
  std::vector<LO> offsets{0};
  std::vector<Real> xyz;
  std::vector<GO> gid;

  EntityBatch batch() const { return EntityBatch{dim, topology, offsets, xyz, gid}; }
};

/// Entities `lids` of dimension `dim` of the mesh, in that order, with their
/// corner vertices' coordinates and their gids. Throws std::out_of_range for
/// a dimension or local id the mesh does not have.
EntityBatchData entity_batch(const HostMesh& mesh, int dim, std::span<const LO> lids);

/// Every entity of dimension `dim`, in local id order.
EntityBatchData entity_batch(const HostMesh& mesh, int dim);

/// Gives entities `lids` of dimension `dim` the values of the rules of the
/// fields that hold one value per entity of that dimension (a full domain,
/// per-entity layout); fields without a rule are left alone. Returns the
/// names of the fields given values. MeshEdit::apply calls it for the
/// entities it adds. Throws as the rules do, std::logic_error included when
/// one returns the wrong number of values.
std::vector<std::string> fill_by_rules(HostMesh& mesh, int dim, std::span<const LO> lids);

/// The mesh value write_soma records the names of the fields that have a rule
/// in: rules are functions, which files cannot hold.
inline constexpr std::string_view rules_value = "soma:rules";

/// Replaces the soma:rules value with `names` (removes it when empty).
void set_rule_names(MeshValues& values, const std::vector<std::string>& names);

/// The fields the soma:rules value names that have no rule on the mesh: on a
/// mesh read back from a file, the rules to set again. Empty when the value
/// is absent.
std::vector<std::string> missing_rules(const HostMesh& mesh);

enum class Sampling : std::uint8_t { point, average, integral };

struct SampleOptions {
  Sampling mode = Sampling::point;
  /// average, integral: the quadrature is exact for polynomials of this
  /// degree on the reference cell (0 to 60).
  int degree = 4;
};

namespace detail {

/// What sample calls the rule with for one block of the field's domain, and
/// where the values go: point mode, value i to row rows[i]; average and
/// integral, entity i's values are those of points [offsets[i], offsets[i + 1])
/// weighted by weights, to row rows[i].
struct SampleBlock {
  EntityBatchData batch;
  std::vector<LO> rows;
  std::vector<LO> offsets;
  std::vector<Real> weights;
};

/// Throws std::invalid_argument for what sample cannot do (see there).
std::vector<SampleBlock> sample_plan(const HostMesh& mesh, const Space<Kokkos::HostSpace>& space,
                                     const std::string& field, const SampleOptions& options);

}  // namespace detail

/// Gives `field`, a field of `mesh`, the values of `rule` (the field's own
/// rule when empty) on every member of its domain. The rule's value type is
/// the field's, checked when the rule is built. Throws std::invalid_argument
/// when there is no rule, for averages or integrals of an integer field or a
/// layout other than per-entity, an element_nodes layout other than on
/// elements, entities the maps above do not cover, or a degree out of range;
/// std::logic_error when the field is not current on the mesh or the rule
/// returns the wrong number of values.
template <class T>
void sample(const HostMesh& mesh, const Field<T, Kokkos::HostSpace>& field,
            std::type_identity_t<FieldRule<T>> rule = {}, const SampleOptions& options = {}) {
  if (!mesh.fields().is_current(field)) {
    throw std::logic_error("sample: the field is not a current field of the mesh");
  }
  const FieldInfo& info = mesh.fields().info(field.id());
  if (!rule) rule = mesh.fields().template rule<T>(field.id());
  if (!rule) throw std::invalid_argument("sample: field '" + info.name + "' has no rule and none was given");
  if (!std::is_floating_point_v<T> && options.mode != Sampling::point) {
    throw std::invalid_argument("sample: averages and integrals of field '" + info.name +
                                "' need floating-point values");
  }
  const auto to = field.view();
  const auto ncomp = static_cast<std::size_t>(field.ncomp());
  for (const detail::SampleBlock& b : detail::sample_plan(mesh, field.space(), info.name, options)) {
    if (b.rows.empty()) continue;
    const std::vector<T> v = rule(b.batch.batch(), static_cast<int>(ncomp));
    if (b.offsets.empty()) {
      for (std::size_t i = 0; i < b.rows.size(); ++i) {
        for (std::size_t k = 0; k < ncomp; ++k) to(static_cast<std::size_t>(b.rows[i]), k) = v[i * ncomp + k];
      }
      continue;
    }
    for (std::size_t i = 0; i < b.rows.size(); ++i) {
      for (std::size_t k = 0; k < ncomp; ++k) {
        Real sum = 0, measure = 0;
        for (auto p = static_cast<std::size_t>(b.offsets[i]); p < static_cast<std::size_t>(b.offsets[i + 1]); ++p) {
          sum += b.weights[p] * static_cast<Real>(v[p * ncomp + k]);
          measure += b.weights[p];
        }
        if (options.mode == Sampling::average && !(measure != 0)) {
          throw std::invalid_argument("sample: field '" + info.name + "': an average over a cell of no measure (row " +
                                      std::to_string(b.rows[i]) + ")");
        }
        to(static_cast<std::size_t>(b.rows[i]), k) =
            static_cast<T>(options.mode == Sampling::average ? sum / measure : sum);
      }
    }
  }
}

}  // namespace soma

#endif  // SOMA_ALGORITHMS_SAMPLE_HPP_
