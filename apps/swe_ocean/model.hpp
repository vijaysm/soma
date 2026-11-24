// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_OCEAN_MODEL_HPP_
#define SWE_OCEAN_MODEL_HPP_

// The barotropic ocean: free surface eta and Cartesian velocity u (tangent) at the GLL points of an ocean
// mesh's space, over the bathymetry depth H > 0, stepped by SSP-RK3 in the vector-invariant form of swe_atm:
//
//   d eta/dt = -div_wk(D u + F),  F = nu grad(L eta)
//   du/dt    = -(zeta + f) n x u - grad(g eta + |u|^2 / 2) + tau / (rho0 D) - drag - nu del^4 u
//
// with D = H + eta the total depth, div_wk the weak divergence (zero flux through the coast, so the integral of
// eta is conserved to rounding), L the weak Laplacian (soma::sem::laplacian; -div_wk(F) is then -nu del^4 eta,
// kept as a flux for the tracers that will share it), zeta the vorticity, n the unit normal a_1 x a_2 (outward
// on the sphere, +z in the plane), f the Coriolis parameter, tau the wind stress, drag C_d |u| u / D
// (quadratic) or r u (linear) and del^4 HOMME's vector hyperviscosity. The linear mode drops zeta and |u|^2 / 2
// and takes D = H in the flux, the stress and the drag. Every tendency is made continuous by direct stiffness
// summation (the weak divergence is assembled); after each stage's update, u loses its normal component at
// coast points and vanishes at no-slip points (soma::sem::remove_normal). A run has the same bits on any number
// of ranks and threads.
//
// The step (when not given): the smaller of the advective and the hyperviscous limit at t = 0,
//   advective     cfl * min over owned points of gap / (|u| + sqrt(g max(D, 0)))
//   hyperviscous  2.5 cfl / (nu rho), rho the spectral radius of del^4 (scalar and vector, the larger;
//                 power iterations as swe_atm's), when nu > 0
// with gap a point's local GLL gap: the least straight-line distance from the point to its neighbours along
// its element's i and j lines. Both minima are exact reductions, the same on any number of ranks.
// The viscosity (when not given): 1e15 (dx_min / dx_30)^3.2 m^4/s, dx_30 = pi a / 60 (the edge of a cubed-
// sphere element at ne = 30) and dx_min the shortest element edge over all ranks (great-circle on the sphere,
// straight in the plane, between the GLL corner points); 0 in the linear mode.

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include "constants.hpp"
#include "ocean_mesh.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/sem/boundary.hpp"
#include "types.hpp"

namespace ocean {

// wind: fills tau (N/m^2, 3 Cartesian components, every row of the space) once, at construction; the stress is
// then made continuous and tangent. None: no stress. drag_quadratic: C_d of C_d |u| u / D; drag_linear: r (1/s)
// of r u, replacing the quadratic drag when set.
struct Forcing {
  std::function<void(const Space&, Values tau)> wind;
  double drag_quadratic = C_d;
  std::optional<double> drag_linear;
};

// linear: the linear mode above; cfl, dt (s), nu (m^4/s): the step and the viscosity, as above; coriolis: f
// (1/s) at a point's position, by default 2 Omega z / |x| (2 Omega sin(latitude) on the sphere, 0 in the plane).
struct ModelOptions {
  bool linear = false;
  double cfl = 0.4;
  std::optional<double> dt, nu;
  std::function<double(const std::array<double, 3>&)> coriolis;
  Forcing forcing;
};

class OceanModel {
 public:
  // Collective: takes the ocean mesh over (its space may be on the sphere or in the plane; its bathymetry's H,
  // one component per row, is the depth), finds the boundary, evaluates f and the wind stress, sets eta = 0 and
  // u = 0 at t = 0 and picks the viscosity and the step. Throws std::invalid_argument on every rank for cfl <=
  // 0, dt <= 0, nu < 0, a drag coefficient negative (or any of them not finite), or a mesh without a space or
  // with H not one component per row of it (that last on the ranks where it is so, the others may wait); what
  // make_boundary and the wind throw.
  OceanModel(const soma::Comm& comm, OceanMesh&& mesh, const ModelOptions& options);
  OceanModel(const OceanModel&) = delete;
  OceanModel& operator=(const OceanModel&) = delete;
  ~OceanModel();

  // Collective: sets eta to eta0 at every point (made continuous) and picks the step again unless it was given.
  // Throws std::logic_error on every rank once a step has been taken.
  void initialize(const soma::Comm& comm, const std::function<double(const std::array<double, 3>&)>& eta0);

  // Collective: steps to t_end (s) exactly, the last step shortened when dt does not divide the interval.
  // Throws std::runtime_error on every rank when a total depth is not positive or a value not finite after a
  // step ("swe_ocean: blew up at step N, t = T s"); the state is then that step's.
  void advance(const soma::Comm& comm, double t_end);

  // Collective: whole steps toward t (s), stopping at the last one not past t, for a report on the way to t_end
  // >= t; the step that advance(t_end) would shorten or stretch is left to it, as SweModel's. Throws as advance.
  void advance_whole_steps(const soma::Comm& comm, double t, double t_end);

  double time() const noexcept;
  double dt() const noexcept;
  double nu() const noexcept;
  long steps() const noexcept;
  const ModelOptions& options() const noexcept;

  const soma::HostMesh& mesh() const noexcept;
  const Space& space() const noexcept;
  const soma::sem::SemBoundary<Host>& boundary() const noexcept;
  ConstValues eta() const;
  ConstValues u() const;  // 3 components
  ConstValues H() const;
  ConstValues f() const;

  // Collective: the integral of eta (m^3), summed exactly.
  double mass(const soma::Comm& comm) const;

  // Collective: FNV-1a over every owned element's (gid, eta, u) in gid order, gathered on every rank: the same
  // on any number of ranks.
  std::uint64_t digest(const soma::Comm& comm) const;

 private:
  void take_step(const soma::Comm& comm, double length, double time_after);

  struct State;
  std::unique_ptr<State> s_;
};

}  // namespace ocean

#endif  // SWE_OCEAN_MODEL_HPP_
