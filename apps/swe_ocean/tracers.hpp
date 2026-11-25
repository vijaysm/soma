// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_OCEAN_TRACERS_HPP_
#define SWE_OCEAN_TRACERS_HPP_

// Passive tracers on the K levels of a general vertical coordinate (vertical.hpp), carried by the barotropic
// ocean (model.hpp). Level k = 1..K (top first) of a column of depth H has the thickness
//
//   h_k = h0_k (1 + eta / H),   r_k = h0_k / H,   sum_k h0_k = H,
//
// and carries the share F_k = r_k G of the barotropic flux G = D u + F of the continuity equation (F the
// hyperviscous flux of eta), so that the levels sum to d eta/dt = -div_wk(G). With w_k the volume flux per unit
// area through interface k (between levels k and k + 1) from level k + 1 up into level k (positive upward; m/s),
// level k's continuity d(h_k)/dt + div_wk(F_k) + w_(k-1) - w_k = 0 with d(h_k)/dt = r_k d eta/dt gives w from
// the bottom (w_K = 0) up:
//
//   w_(k-1) = w_k - div_wk(F_k) - r_k d eta/dt.
//
// w_0 at the surface is not imposed: it is zero but for rounding (the levels sum to the barotropic equation),
// a check (w_surface_max). A tracer T_k obeys
//
//   d(h_k T_k)/dt = -div_wk(F_k T_k) - (wT)_(k-1) + (wT)_k - nu_T div_wk(kappa_k grad(L_k T_k)),
//   L_k T = div_wk(kappa_k grad T) / h_k,
//
// (wT)_k = w_k times T upwinded at interface k (T_(k+1) when w_k > 0, T_k otherwise), (wT)_0 = (wT)_K = 0: no
// tracer crosses the surface or the bottom, so the vertical fluxes telescope. kappa_k, the weight of the
// thickness-weighted biharmonic, is h_k's least value over the element, at each of its points: with h_k itself
// a z-star level 1e-3 m thick beside one 100 m thick feels its neighbour's flux 1e5 times over, and the
// biharmonic is unstable at the model's step; with kappa_k <= h_k it is no stiffer than eta's.
//
// Carried as an anomaly about the tracer's mean Tbar (set by set): the state is a_k = h_k (T_k - Tbar), and
// h_k T_k = Tbar h_k + a_k with h_k from eta. h_k T_k = Tbar h_k solves the equation above (d(h_k)/dt is the
// level's continuity), so a_k obeys it with T_k - Tbar for T_k. A uniform tracer has a = 0 but for the rounding
// of Tbar and stays uniform to rounding even in a vanished z-star level, whose T is otherwise its neighbours'
// rounding, averaged with weights of both signs (the weak divergence's) and so magnified many times. The
// integral of sum_k h_k T_k is conserved to rounding: eta's is, and the weak divergences integrate to zero.
//
// Each stage of the model's SSP-RK3 takes the explicit tendency (tendency) and then the vertical fluxes
// implicitly, with T of the stage's end (vertical): a z-star level near h_min passes the fluxes of its thick
// neighbours on through w, so w dt / h_k can pass the explicit limit (1.8 on the synthetic Earth of the tests).
// The implicit vertical term is first order in time within each stage (backward Euler on the stage's weight);
// the horizontal terms keep SSP-RK3's order.
// Every row is computed alone (no reductions), so a run has the same bits on any number of ranks and threads.
//
// The views have a row per point of the space: h holds K components (level k in component k - 1), hT and the
// state K per tracer (tracer t, level k in component t K + k - 1), w K + 1 (interface k in component k).

#include <array>
#include <functional>
#include <vector>

#include "soma/parallel/comm.hpp"
#include "types.hpp"
#include "vertical.hpp"

namespace ocean {

class Tracers {
 public:
  // Not collective: the reference thicknesses of every row's depth H (one component per row of space) in the
  // coordinate spec, `count` tracers set to 0 over levels of thickness h0_k (eta = 0), nu_T (m^4/s) the
  // tracers' hyperviscosity. space must outlive the tracers. Throws std::invalid_argument for count < 0, nu_T
  // negative or not finite, spec.levels < 1, or H not one component per row of space; what
  // reference_thicknesses throws.
  Tracers(const Space& space, ConstValues H, const VerticalSpec& spec, int count, double nu_T);

  int levels() const noexcept { return K_; }
  int count() const noexcept { return count_; }
  double nu() const noexcept { return nu_; }
  const VerticalSpec& vertical() const noexcept { return spec_; }

  ConstValues h0() const { return h0_; }  // K components
  ConstValues h() const { return h_; }    // K components, of the free surface now
  ConstValues hT() const { return hT_; }  // K count components: Tbar h + the state
  ConstValues w() const { return w_; }    // K + 1 components, of the last stage evaluated

  // Collective: sets T_k of tracer t at every row to value(x, z, t), x the row's position and z (m, negative
  // down) the middle of level k below the free surface now (eta - h_1 - ... - h_(k-1) - h_k / 2), and hT_k =
  // h_k T_k; Tbar becomes the tracer's integral over the volume's. Not made continuous (a tendency is): direct
  // stiffness summation would round a uniform T differently at each shared point, and a vanished level would
  // magnify that. Throws what value throws.
  void set(const soma::Comm& comm, const std::function<double(const std::array<double, 3>&, double z, int)>& value);

  // Collective: the integral of sum_k hT_k of a tracer (m^3 times its unit), each level's summed exactly. Throws
  // std::out_of_range for a tracer outside [0, count).
  double total(const soma::Comm& comm, int tracer) const;

  // Collective: max over owned points of |w_0| (m/s) of the last stage evaluated; 0 before any.
  double w_surface_max(const soma::Comm& comm) const;

  // For the model. Not collective: h and hT from the free surface eta (1 component) and the state; `keep`
  // first rescales the state so that T is unchanged (when eta is set anew).
  void thicken(ConstValues eta, bool keep = false);

  // For the model. The state a (K count components), stepped by the model's stages. A test may write it (then
  // thicken, to recompose hT) to carry a tracer on the anomaly alone.
  Values state() const { return anomaly_; }

  // For the model. Collective: da, the explicit part of the tendency of the state a at a stage whose free
  // surface is eta (1 component), barotropic flux G (3) and d eta/dt (1, -div_wk(G)): all but the vertical
  // fluxes. Sets w. Every row is written.
  void tendency(const soma::Comm& comm, ConstValues eta, ConstValues G, ConstValues deta, ConstValues a,
                Values da);

  // For the model. Not collective: adds to a, the stage's explicit update, c (s) times the stage's vertical
  // fluxes, implicitly: with T of the result and h_k of the stage's end, whose free surface is eta (1
  // component); w is the stage's (tendency). Every row is written.
  void vertical(ConstValues eta, double c, Values a);

 private:
  // hT = Tbar h + a.
  void compose();

  const Space* space_;
  VerticalSpec spec_;
  int K_, count_;
  double nu_;
  ConstValues H_;
  Values h0_, ratio_, eta_, h_, hT_, anomaly_, w_;
  std::vector<double> mean_;  // Tbar of each tracer
  // A stage's thicknesses and diffusive weights, level fluxes F_k (3 K), their divergences, the anomaly of a
  // level, scratch, and a column's tridiagonal system.
  Values hs_, kappa_, Fk_, divF_, T_, level_, vec_, grad_, div_, lower_, diag_, upper_;
};

}  // namespace ocean

#endif  // SWE_OCEAN_TRACERS_HPP_
