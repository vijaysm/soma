// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_ATM_MODEL_HPP_
#define SWE_ATM_MODEL_HPP_

// The shallow-water model on the cubed sphere: depth h and Cartesian velocity
// u (tangent) at the GLL points of a spectral-element space of order p,
// stepped by SSP-RK3 in the vector-invariant form
//   du/dt = -(zeta + f) n x u - grad(g (h + hs) + |u|^2 / 2) - nu del^4 u
//   dh/dt = -div(h u) - nu del^4 h
// with zeta the vorticity, n the outward normal and del^4 HOMME's
// hyperviscosity (soma::sem::vector_hyperviscosity and hyperviscosity). Every
// tendency is made continuous by direct stiffness summation, so a run has the
// same bits on any number of ranks and threads.

#include <cstdint>
#include <memory>
#include <optional>

#include "cases.hpp"
#include "diagnostics.hpp"
#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"

namespace swe {

struct ModelOptions {
  int ne = 4;     // elements along a cube face's edge
  int order = 3;  // GLL order p
  Case test = Case::williamson2;
  double alpha = 0;  // case 2's rotation angle
  double cfl = 0.4;  // dt = cfl * (smallest GLL gap) / (largest |u| + sqrt(g h) at t = 0)
  std::optional<double> dt;  // s; replaces the CFL step
  std::optional<double> nu;  // m^4/s; default 1e15 (30 / ne)^3.2, and 0 for case 2
};

class SweModel {
 public:
  // Collective: builds the cubed sphere (with its ghost layer and distributed
  // edges on more than one rank), the space and the state at t = 0. Throws
  // std::invalid_argument for ne < 1, order < 1, cfl <= 0, dt <= 0 or nu < 0.
  SweModel(const soma::Comm& comm, const ModelOptions& options);
  SweModel(const SweModel&) = delete;
  SweModel& operator=(const SweModel&) = delete;
  ~SweModel();

  // Collective: steps to t_end (s) exactly, the last step shortened when dt
  // does not divide the interval. Throws std::runtime_error on every rank when
  // a depth is not positive or a value not finite after a step ("swe: blew up
  // at step N, t = T s"); the state is then that step's.
  void advance(const soma::Comm& comm, double t_end);

  // Collective: whole steps toward t (s), stopping at the last one not past t,
  // for a report on the way to t_end >= t. A step that advance(t_end) would
  // shorten or stretch to land on t_end is left to it, so stopping here leaves
  // the bits of the run unchanged. Throws as advance does.
  void advance_whole_steps(const soma::Comm& comm, double t, double t_end);

  double time() const noexcept;
  double dt() const noexcept;
  double nu() const noexcept;
  long steps() const noexcept;
  const ModelOptions& options() const noexcept;

  soma::HostMesh& mesh() noexcept;
  const soma::HostMesh& mesh() const noexcept;
  const Space& space() const noexcept;
  ConstValues h() const;
  ConstValues hs() const;
  ConstValues u() const;  // 3 components

  // Collective: FNV-1a over every owned element's (gid, h, u) in gid order,
  // gathered on every rank: the same on any number of ranks.
  std::uint64_t digest(const soma::Comm& comm) const;

 private:
  // One step of `length` s, after which the time is `time_after`; the
  // collective blow-up check.
  void take_step(const soma::Comm& comm, double length, double time_after);

  struct State;
  std::unique_ptr<State> s_;
};

}  // namespace swe

#endif  // SWE_ATM_MODEL_HPP_
