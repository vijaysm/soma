// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_LAYOUT_HPP_
#define SOMA_FIELDS_LAYOUT_HPP_

#include <cstdint>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "soma/core/topology.hpp"

namespace soma {

/// Highest supported element order. Node counts at this order fit an int.
inline constexpr int max_node_order = 32;

enum class NodeFamily : std::uint8_t { lagrange_equispaced, gll, gauss };
enum class NodeOrdering : std::uint8_t { moab_native, tensor_lexicographic, hierarchical };

/// How many value points each entity carries. Only per_entity and element_nodes
/// can back a Space for now; the other kinds are representable so that code
/// written against them compiles, and are rejected where a Space is built.
struct Layout {
  enum class Kind : std::uint8_t { per_entity, element_nodes, shared_nodes, quadrature } kind =
      Kind::per_entity;
  NodeFamily family = NodeFamily::gll;
  int order = 1;
  NodeOrdering ordering = NodeOrdering::tensor_lexicographic;
  bool serendipity = false;  ///< Quad8, Hex20, Prism15, Pyramid13
  int quad_rule = 0;

  static constexpr Layout per_entity() noexcept { return Layout{}; }

  static constexpr Layout element_nodes(
      NodeFamily family, int order,
      NodeOrdering ordering = NodeOrdering::tensor_lexicographic,
      bool serendipity = false) noexcept {
    return Layout{Kind::element_nodes, family, order, ordering, serendipity, 0};
  }

  static constexpr Layout shared_nodes(
      NodeFamily family, int order,
      NodeOrdering ordering = NodeOrdering::hierarchical,
      bool serendipity = false) noexcept {
    return Layout{Kind::shared_nodes, family, order, ordering, serendipity, 0};
  }

  static constexpr Layout quadrature(int rule) noexcept {
    return Layout{Kind::quadrature, NodeFamily::gauss, 1, NodeOrdering::tensor_lexicographic,
                  false, rule};
  }

  /// Compares only the members that the kind uses.
  friend constexpr bool operator==(const Layout& a, const Layout& b) noexcept {
    if (a.kind != b.kind) return false;
    switch (a.kind) {
      case Kind::per_entity: return true;
      case Kind::quadrature: return a.quad_rule == b.quad_rule;
      case Kind::element_nodes:
      case Kind::shared_nodes:
        return a.family == b.family && a.order == b.order && a.ordering == b.ordering &&
               a.serendipity == b.serendipity;
    }
    return false;
  }
};

constexpr const char* name(Layout::Kind k) noexcept {
  switch (k) {
    case Layout::Kind::per_entity: return "per_entity";
    case Layout::Kind::element_nodes: return "element_nodes";
    case Layout::Kind::shared_nodes: return "shared_nodes";
    case Layout::Kind::quadrature: return "quadrature";
  }
  return "unknown";
}

constexpr const char* name(NodeFamily f) noexcept {
  switch (f) {
    case NodeFamily::lagrange_equispaced: return "lagrange_equispaced";
    case NodeFamily::gll: return "gll";
    case NodeFamily::gauss: return "gauss";
  }
  return "unknown";
}

constexpr const char* name(NodeOrdering o) noexcept {
  switch (o) {
    case NodeOrdering::moab_native: return "moab_native";
    case NodeOrdering::tensor_lexicographic: return "tensor_lexicographic";
    case NodeOrdering::hierarchical: return "hierarchical";
  }
  return "unknown";
}

inline std::string describe(const Layout& l) {
  switch (l.kind) {
    case Layout::Kind::per_entity: return "per_entity";
    case Layout::Kind::quadrature: return "quadrature(rule " + std::to_string(l.quad_rule) + ")";
    case Layout::Kind::element_nodes:
    case Layout::Kind::shared_nodes: break;
  }
  return std::string(name(l.kind)) + "(" + name(l.family) + ", order " + std::to_string(l.order) +
         (l.serendipity ? ", serendipity" : "") + ")";
}

/// Value points per entity of one element topology. Returns 0 for a combination
/// that has no node table: an order outside [1, max_node_order], a family the
/// topology does not support, serendipity where no such element exists, knife
/// and polyhedron, and layout kinds that are not element-local. `arity` is the
/// vertex count of a polygon and is ignored otherwise.
KOKKOS_INLINE_FUNCTION constexpr int num_element_nodes(Topology t, const Layout& l,
                                                      int arity = 0) noexcept {
  if (l.kind == Layout::Kind::per_entity) return 1;
  if (l.kind != Layout::Kind::element_nodes) return 0;
  const int p = l.order;
  if (p < 1 || p > max_node_order || l.family == NodeFamily::gauss) return 0;

  if (l.serendipity) {
    if (p != 2) return 0;
    switch (t) {
      case Topology::quad: return 8;
      case Topology::hex: return 20;
      case Topology::prism: return 15;
      case Topology::pyramid: return 13;
      default: return 0;
    }
  }
  switch (t) {
    case Topology::edge: return p + 1;
    case Topology::tri: return (p + 1) * (p + 2) / 2;
    case Topology::quad: return (p + 1) * (p + 1);
    case Topology::tet: return (p + 1) * (p + 2) * (p + 3) / 6;
    case Topology::prism: return (p + 1) * (p + 1) * (p + 2) / 2;
    case Topology::hex: return (p + 1) * (p + 1) * (p + 1);
    case Topology::pyramid:
      if (p == 1) return 5;
      if (p == 2 && l.family == NodeFamily::lagrange_equispaced) return 14;
      return 0;
    case Topology::polygon: return (p == 1 && arity >= 3) ? arity : 0;
    default: return 0;
  }
}

/// Throws std::invalid_argument naming the combination when an element_nodes
/// layout has no node table for the topology. Other kinds pass unchecked.
inline void validate(Topology t, const Layout& l, int arity = 0) {
  if (l.kind != Layout::Kind::element_nodes) return;
  if (num_element_nodes(t, l, arity) > 0) return;
  std::string why = std::string("unsupported layout ") + describe(l) + " on " + name(t);
  if (t == Topology::polygon) why += " (arity " + std::to_string(arity) + ")";
  if (l.order > max_node_order) {
    why += " (the maximum order is " + std::to_string(max_node_order) + ")";
  }
  throw std::invalid_argument(why);
}

}  // namespace soma

#endif  // SOMA_FIELDS_LAYOUT_HPP_
