// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_FIELD_RULE_HPP_
#define SOMA_FIELDS_FIELD_RULE_HPP_

/// \file
/// A field's rule: how values come about for entities the field has no value
/// for yet (those refinement or an edit adds), and for sampling
/// (soma::sample). A rule is called once with every entity at hand, so its
/// loop can vectorize:
///
/// ```
///   soma::FieldOptions<double> o;
///   o.rule = [](const soma::EntityBatch& b) {
///     std::vector<double> v(b.size());
///     for (std::size_t i = 0; i < b.size(); ++i) {
///       const auto p = b.center(i);
///       v[i] = std::sin(p[0]) * std::cos(p[1]);
///     }
///     return v;
///   };
///   mesh.fields().add_tag<double>("u", 0, 1, std::move(o));
/// ```
///
/// A rule for a field of value type T must return exactly std::vector<T>,
/// ncomp values per entity, entity by entity: a static_assert holds it to that.
/// soma::pointwise makes a rule of a function of one point. Rules live on the
/// host, in the field registry, and copies, mirrors and renumbering carry them;
/// files do not hold them (write_soma records which fields had one, and
/// soma::missing_rules names those a mesh read back is without).

#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"

namespace soma {

/// The entities a rule gives values to, in order: entity i has the corners
/// offsets[i] .. offsets[i + 1] of xyz (three coordinates each). For dim 0
/// each entity is a point, its own one corner: a vertex, an element node or a
/// quadrature point. gid is empty when the points are not entities of the mesh.
struct EntityBatch {
  int dim = 0;
  std::span<const Topology> topology;  ///< per entity; empty for points
  std::span<const LO> offsets;         ///< size() + 1 entries
  std::span<const Real> xyz;           ///< 3 per corner
  std::span<const GO> gid;             ///< per entity, or empty

  std::size_t size() const noexcept { return offsets.empty() ? 0 : offsets.size() - 1; }
  LO corners(std::size_t i) const noexcept { return offsets[i + 1] - offsets[i]; }
  /// Coordinate k of corner c of entity i.
  Real x(std::size_t i, LO c, int k) const noexcept {
    return xyz[3 * static_cast<std::size_t>(offsets[i] + c) + static_cast<std::size_t>(k)];
  }
  /// The average of the entity's corners: the point itself for dim 0.
  std::array<Real, 3> center(std::size_t i) const noexcept {
    std::array<Real, 3> p{};
    const LO n = corners(i);
    for (LO c = 0; c < n; ++c) {
      for (int k = 0; k < 3; ++k) p[static_cast<std::size_t>(k)] += x(i, c, k);
    }
    for (Real& v : p) v /= static_cast<Real>(n);
    return p;
  }
};

/// A rule for a field of value type T: given a batch of n entities, n * ncomp
/// values. Empty (false) when none is set.
template <class T>
class FieldRule {
 public:
  using value_type = T;
  using function_type = std::function<std::vector<T>(const EntityBatch&)>;

  FieldRule() = default;

  template <class F>
    requires(!std::is_same_v<std::remove_cvref_t<F>, FieldRule>)
  FieldRule(F f) {  // NOLINT(google-explicit-constructor): a lambda is a rule
    static_assert(std::is_invocable_v<F&, const EntityBatch&>,
                  "a field rule is called with one argument, a const soma::EntityBatch&");
    static_assert(std::is_same_v<std::invoke_result_t<F&, const EntityBatch&>, std::vector<T>>,
                  "a field rule must return std::vector<T>, T being exactly the field's value type");
    fn_ = std::move(f);
  }

  explicit operator bool() const noexcept { return static_cast<bool>(fn_); }

  /// The rule's values for the batch. Throws std::logic_error when the rule
  /// does not return ncomp values per entity, and std::bad_function_call when
  /// there is no rule.
  std::vector<T> operator()(const EntityBatch& batch, int ncomp) const {
    std::vector<T> v = fn_(batch);
    const std::size_t want = batch.size() * static_cast<std::size_t>(ncomp);
    if (v.size() != want) {
      throw std::logic_error("field rule: returned " + std::to_string(v.size()) + " values for " +
                             std::to_string(batch.size()) + " entities of " + std::to_string(ncomp) +
                             " components; it must return " + std::to_string(want));
    }
    return v;
  }

 private:
  function_type fn_;
};

/// A rule of a function of one point, f(x, y, z), returning T (one
/// component) or std::array<T, N> (N components). It is evaluated at each
/// entity's center: the point itself for points and vertices. The value type
/// is the one f returns, and FieldRule's static_assert checks it against the
/// field's.
template <class F>
auto pointwise(F f) {
  using R = std::invoke_result_t<F&, Real, Real, Real>;
  static_assert(!std::is_void_v<R>, "soma::pointwise: f(x, y, z) must return a value");
  return [f = std::move(f)](const EntityBatch& b) {
    if constexpr (requires { std::tuple_size<R>::value; }) {
      using T = typename R::value_type;
      constexpr std::size_t n = std::tuple_size<R>::value;
      std::vector<T> v(b.size() * n);
      for (std::size_t i = 0; i < b.size(); ++i) {
        const auto p = b.center(i);
        const R r = f(p[0], p[1], p[2]);
        for (std::size_t k = 0; k < n; ++k) v[i * n + k] = r[k];
      }
      return v;
    } else {
      std::vector<R> v(b.size());
      for (std::size_t i = 0; i < b.size(); ++i) {
        const auto p = b.center(i);
        v[i] = f(p[0], p[1], p[2]);
      }
      return v;
    }
  };
}

}  // namespace soma

#endif  // SOMA_FIELDS_FIELD_RULE_HPP_
