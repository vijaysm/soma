// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TOPOLOGY_CANONICAL_HPP_
#define SOMA_TOPOLOGY_CANONICAL_HPP_

// Canonical sub-entity ("side") tables for fixed-topology elements.
//
// Values mirror moab::CN::SubEntityVertexIndices and are verified
// positionally against it by tests/reference/test_canonical_moab.cpp. Vertex
// order encodes the side's sense relative to the parent; never sort it.
//
// Written as switch statements returning values (no namespace-scope arrays) so
// every function is callable from CUDA, HIP, and SYCL kernels.

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "soma/core/topology.hpp"

namespace soma::canonical {

struct SideDef {
  Topology topology = Topology::vertex;
  std::uint8_t num_vertices = 0;
  std::uint8_t v[4] = {0, 0, 0, 0};
};

namespace detail {
KOKKOS_INLINE_FUNCTION constexpr SideDef e(std::uint8_t a, std::uint8_t b) noexcept {
  return SideDef{Topology::edge, 2, {a, b, 0, 0}};
}
KOKKOS_INLINE_FUNCTION constexpr SideDef t(std::uint8_t a, std::uint8_t b, std::uint8_t c) noexcept {
  return SideDef{Topology::tri, 3, {a, b, c, 0}};
}
KOKKOS_INLINE_FUNCTION constexpr SideDef q(std::uint8_t a, std::uint8_t b, std::uint8_t c,
                                           std::uint8_t d) noexcept {
  return SideDef{Topology::quad, 4, {a, b, c, d}};
}
}  // namespace detail

KOKKOS_INLINE_FUNCTION constexpr int num_sides(Topology topo, int dim) noexcept {
  if (dim == 1) {
    switch (topo) {
      case Topology::edge: return 1;
      case Topology::tri: return 3;
      case Topology::quad: return 4;
      case Topology::tet: return 6;
      case Topology::pyramid: return 8;
      case Topology::prism: return 9;
      case Topology::knife: return 10;
      case Topology::hex: return 12;
      default: return 0;
    }
  }
  if (dim == 2) {
    switch (topo) {
      case Topology::tri:
      case Topology::quad: return 1;
      case Topology::tet: return 4;
      case Topology::pyramid:
      case Topology::prism:
      case Topology::knife: return 5;
      case Topology::hex: return 6;
      default: return 0;
    }
  }
  return 0;
}

// Side i of dimension dim. Returns a default SideDef when out of range.
KOKKOS_INLINE_FUNCTION constexpr SideDef side(Topology topo, int dim, int i) noexcept {
  using detail::e;
  using detail::q;
  using detail::t;
  if (i < 0 || i >= num_sides(topo, dim)) {
    return SideDef{};
  }
  if (dim == 1) {
    switch (topo) {
      case Topology::edge: return e(0, 1);
      case Topology::tri: {
        constexpr SideDef s[] = {e(0, 1), e(1, 2), e(2, 0)};
        return s[i];
      }
      case Topology::quad: {
        constexpr SideDef s[] = {e(0, 1), e(1, 2), e(2, 3), e(3, 0)};
        return s[i];
      }
      case Topology::tet: {
        constexpr SideDef s[] = {e(0, 1), e(1, 2), e(2, 0), e(0, 3), e(1, 3), e(2, 3)};
        return s[i];
      }
      case Topology::pyramid: {
        constexpr SideDef s[] = {e(0, 1), e(1, 2), e(2, 3), e(3, 0),
                                 e(0, 4), e(1, 4), e(2, 4), e(3, 4)};
        return s[i];
      }
      case Topology::prism: {
        constexpr SideDef s[] = {e(0, 1), e(1, 2), e(2, 0), e(0, 3), e(1, 4),
                                 e(2, 5), e(3, 4), e(4, 5), e(5, 3)};
        return s[i];
      }
      case Topology::knife: {
        constexpr SideDef s[] = {e(0, 1), e(1, 2), e(2, 3), e(3, 0), e(0, 4),
                                 e(1, 5), e(2, 6), e(3, 5), e(4, 5), e(5, 6)};
        return s[i];
      }
      case Topology::hex: {
        constexpr SideDef s[] = {e(0, 1), e(1, 2), e(2, 3), e(3, 0), e(0, 4), e(1, 5),
                                 e(2, 6), e(3, 7), e(4, 5), e(5, 6), e(6, 7), e(7, 4)};
        return s[i];
      }
      default: return SideDef{};
    }
  }
  switch (topo) {  // dim == 2
    case Topology::tri: return t(0, 1, 2);
    case Topology::quad: return q(0, 1, 2, 3);
    case Topology::tet: {
      constexpr SideDef s[] = {t(0, 1, 3), t(1, 2, 3), t(0, 3, 2), t(0, 2, 1)};
      return s[i];
    }
    case Topology::pyramid: {
      constexpr SideDef s[] = {t(0, 1, 4), t(1, 2, 4), t(2, 3, 4), t(3, 0, 4), q(0, 3, 2, 1)};
      return s[i];
    }
    case Topology::prism: {
      constexpr SideDef s[] = {q(0, 1, 4, 3), q(1, 2, 5, 4), q(0, 3, 5, 2), t(0, 2, 1),
                               t(3, 4, 5)};
      return s[i];
    }
    case Topology::knife: {
      constexpr SideDef s[] = {q(0, 1, 5, 4), q(1, 2, 6, 5), q(2, 3, 5, 6), q(3, 0, 4, 5),
                               q(0, 3, 2, 1)};
      return s[i];
    }
    case Topology::hex: {
      constexpr SideDef s[] = {q(0, 1, 5, 4), q(1, 2, 6, 5), q(2, 3, 7, 6),
                               q(3, 0, 4, 7), q(0, 3, 2, 1), q(4, 5, 6, 7)};
      return s[i];
    }
    default: return SideDef{};
  }
}

}  // namespace soma::canonical

#endif  // SOMA_TOPOLOGY_CANONICAL_HPP_
