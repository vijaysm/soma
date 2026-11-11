// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/algorithms/csr.hpp"
#include "soma/algorithms/fem.hpp"
#include "soma/algorithms/fem_space.hpp"
#include "soma/algorithms/laplacian.hpp"
#include "soma/algorithms/vertex_graph.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "fem_box.hpp"
#include "soma/testing/box_mesh.hpp"

using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Offset;
using soma::Real;
using soma::Topology;
namespace fem = soma::fem;
using fem_test::box_part;
using fem_test::Part;
using fem_test::solid;
using fem_test::spec_of;
using Space = soma::FemSpace<Kokkos::HostSpace>;
using Matrix = soma::CsrMatrix<Kokkos::HostSpace>;
using Vector = soma::View1<Real, Kokkos::HostSpace>;

namespace {

constexpr std::array<Topology, 4> all = {Topology::tri, Topology::quad, Topology::tet, Topology::hex};

using Entries = std::map<std::pair<GO, GO>, Real>;

// The owned rows by (row gid, column gid).
Entries by_gid(const soma::HostMesh& mesh, const Matrix& a) {
  const auto g = mesh.gid(0);
  Entries out;
  for (LO i = 0; i < a.rows; ++i) {
    for (Offset k = a.offsets(i); k < a.offsets(i + 1); ++k) out[{g(i), g(a.columns(k))}] = a.values(k);
  }
  return out;
}

std::map<GO, Real> owned_by_gid(const soma::HostMesh& mesh, LO rows, const Vector& b) {
  const auto g = mesh.gid(0);
  std::map<GO, Real> out;
  for (LO i = 0; i < rows; ++i) out[g(i)] = b(i);
  return out;
}

// The first entry of `got` that is not `want`'s, bit for bit: an empty
// string when there is none. Loops report through it rather than ASSERT,
// which would leave a rank out of the collectives that follow.
template <class K>
std::string first_difference(const std::map<K, Real>& got, const std::map<K, Real>& want) {
  for (const auto& [k, v] : got) {
    const auto it = want.find(k);
    if (it == want.end() || it->second != v) {
      std::ostringstream o;
      o << "entry " << ::testing::PrintToString(k) << ": " << v << " vs "
        << (it == want.end() ? std::string("none") : std::to_string(it->second));
      return o.str();
    }
  }
  return {};
}

// Cases of the distributed tests: a box of 4 cells a side and one so small
// some ranks hold nothing, dealt in blocks, round robin, and round robin with
// two ghost layers.
std::vector<soma::testing::BoxSpec> boxes(Topology t) {
  return {spec_of(t, 4), soma::testing::BoxSpec{2, 1, 1, t, 1.0, 1.0, 1.0}};
}
const std::vector<Part> parts = {Part{.deal = 0}, Part{.deal = 1}, Part{.deal = 1, .layers = 2}};

struct Load {
  KOKKOS_INLINE_FUNCTION Real operator()(Real x, Real y, Real z) const { return 1 + x * y - 2 * z; }
};

struct One {
  KOKKOS_INLINE_FUNCTION Real operator()(Real, Real, Real) const { return 1; }
};

bool on_box_boundary(const soma::HostMesh& m, LO v, bool three) {
  const auto x = m.coords();
  for (int q = 0; q < (three ? 3 : 2); ++q) {
    if (x(v, q) == 0 || x(v, q) == 1) return true;
  }
  return false;
}

}  // namespace

// The pattern's rows: each owned vertex with every vertex it shares a cell
// with, itself included, ascending by gid.
TEST(FemSpace, PatternIsEveryVertexSharingACell) {
  for (const Topology t : all) {
    SCOPED_TRACE(soma::name(t));
    const soma::HostMesh mesh = box_part(Comm::self(), spec_of(t, 3));
    const Space space(Comm::self(), mesh);
    const Matrix a = space.pattern();
    const int dim = mesh.dim();
    const auto m = mesh.view();
    const auto g = mesh.gid(0);
    std::vector<std::set<GO>> want(static_cast<std::size_t>(mesh.count(0)));
    for (LO e = 0; e < mesh.count(dim); ++e) {
      for (LO p = 0; p < m.num_vertices(dim, e); ++p) {
        for (LO q = 0; q < m.num_vertices(dim, e); ++q) {
          want[static_cast<std::size_t>(m.vertex(dim, e, p))].insert(g(m.vertex(dim, e, q)));
        }
      }
    }
    ASSERT_EQ(a.rows, mesh.count(0));
    ASSERT_EQ(a.cols, mesh.count(0));
    for (LO i = 0; i < a.rows; ++i) {
      std::vector<GO> row;
      for (Offset k = a.offsets(i); k < a.offsets(i + 1); ++k) row.push_back(g(a.columns(k)));
      const auto& w = want[static_cast<std::size_t>(i)];
      EXPECT_EQ(row, std::vector<GO>(w.begin(), w.end())) << "row " << g(i);
    }
  }
}

// Against a dense matrix scattered from fem::stiffness: equal to rounding,
// symmetric bit for bit, rows summing to zero.
TEST(FemSpace, LaplacianIsTheScatteredElementMatrices) {
  for (const Topology t : all) {
    SCOPED_TRACE(soma::name(t));
    const soma::HostMesh mesh = box_part(Comm::self(), spec_of(t, 3));
    const Space space(Comm::self(), mesh);
    const Matrix a = space.laplacian(Comm::self());
    const int dim = mesh.dim();
    const auto m = mesh.view();
    const auto n = static_cast<std::size_t>(mesh.count(0));
    std::vector<Real> dense(n * n, 0);
    for (LO e = 0; e < mesh.count(dim); ++e) {
      const fem::Cell c = fem::cell_of(m, dim, e);
      Real k[fem::max_nodes][fem::max_nodes];
      ASSERT_GT(fem::stiffness(c, k), 0);
      for (int p = 0; p < c.n; ++p) {
        for (int q = 0; q < c.n; ++q) {
          dense[static_cast<std::size_t>(m.vertex(dim, e, p)) * n + static_cast<std::size_t>(m.vertex(dim, e, q))] +=
              k[p][q];
        }
      }
    }
    // A sequential gather: each row summing its cells' entries, the cells by
    // gid. Equal bit for bit, whatever the threads did.
    std::map<std::pair<GO, GO>, Real> sequential;
    const auto g = mesh.gid(0);
    const auto cg = mesh.gid(dim);
    for (LO v = 0; v < mesh.count(0); ++v) {
      std::vector<std::pair<GO, LO>> around;
      for (LO e = 0; e < mesh.count(dim); ++e) {
        for (LO p = 0; p < m.num_vertices(dim, e); ++p) {
          if (m.vertex(dim, e, p) == v) around.emplace_back(cg(e), e);
        }
      }
      std::sort(around.begin(), around.end());
      for (const auto& [ge, e] : around) {
        const fem::Cell c = fem::cell_of(m, dim, e);
        Real k[fem::max_nodes][fem::max_nodes];
        fem::stiffness(c, k);
        int a_v = 0;
        while (m.vertex(dim, e, a_v) != v) ++a_v;
        for (int q = 0; q < c.n; ++q) sequential[{g(v), g(m.vertex(dim, e, q))}] += k[a_v][q];
      }
    }
    EXPECT_EQ(first_difference(by_gid(mesh, a), sequential), "");
    EXPECT_EQ(by_gid(mesh, a).size(), sequential.size());
    std::map<std::pair<LO, LO>, Real> got;
    for (LO i = 0; i < a.rows; ++i) {
      Real sum = 0;
      for (Offset k = a.offsets(i); k < a.offsets(i + 1); ++k) {
        got[{i, a.columns(k)}] = a.values(k);
        sum += a.values(k);
      }
      const Real diag = dense[static_cast<std::size_t>(i) * (n + 1)];
      EXPECT_GT(diag, 0);
      EXPECT_NEAR(sum, 0, 1e-14 * diag) << "row " << i;
    }
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        const auto it = got.find({static_cast<LO>(i), static_cast<LO>(j)});
        const Real v = it == got.end() ? 0 : it->second;
        EXPECT_NEAR(v, dense[i * n + j], 1e-15 * dense[i * (n + 1)]) << i << " " << j;
        if (it != got.end()) EXPECT_EQ(v, got.at({static_cast<LO>(j), static_cast<LO>(i)}));
      }
    }
  }
}

// Against a vector scattered from fem::load; with f = 1 it sums to the
// box's measure.
TEST(FemSpace, LoadIsTheScatteredElementLoads) {
  for (const Topology t : all) {
    SCOPED_TRACE(soma::name(t));
    const soma::HostMesh mesh = box_part(Comm::self(), spec_of(t, 3));
    const Space space(Comm::self(), mesh);
    const Vector b = space.load(Comm::self(), Load{});
    const int dim = mesh.dim();
    const auto m = mesh.view();
    std::vector<Real> want(static_cast<std::size_t>(mesh.count(0)), 0);
    for (LO e = 0; e < mesh.count(dim); ++e) {
      const fem::Cell c = fem::cell_of(m, dim, e);
      Real r[fem::max_nodes];
      ASSERT_GT(fem::load(c, Load{}, r), 0);
      for (int p = 0; p < c.n; ++p) want[static_cast<std::size_t>(m.vertex(dim, e, p))] += r[p];
    }
    for (LO v = 0; v < mesh.count(0); ++v) {
      EXPECT_NEAR(b(v), want[static_cast<std::size_t>(v)], 1e-15) << "vertex " << v;
    }
    const Vector ones = space.load(Comm::self(), One{});
    Real total = 0;
    for (LO v = 0; v < mesh.count(0); ++v) total += ones(v);
    EXPECT_NEAR(total, 1, 1e-14);
  }
}

// On every partition the owned rows of the matrix and the load are the
// serial ones, bit for bit, and every vertex here, ghosts too, has the
// boundary verdict of the box.
TEST(FemSpace, DistributedAssemblyIsTheSerialOne) {
  const Comm c = Comm::world();
  for (const Topology t : all) {
    for (const soma::testing::BoxSpec& spec : boxes(t)) {
      const soma::HostMesh whole = box_part(Comm::self(), spec);
      const Space serial(Comm::self(), whole);
      const Entries want = by_gid(whole, serial.laplacian(Comm::self()));
      const auto want_b = owned_by_gid(whole, serial.rows(), serial.load(Comm::self(), Load{}));
      std::map<GO, std::size_t> length;
      for (const auto& [rc, v] : want) ++length[rc.first];
      for (const Part& part : parts) {
        SCOPED_TRACE(testing::Message() << soma::name(t) << " " << spec.nx << "x" << spec.ny << "x" << spec.nz
                                        << " deal " << part.deal << " layers " << part.layers);
        const soma::HostMesh mesh = box_part(c, spec, part);
        const Space space(c, mesh);
        EXPECT_EQ(first_difference(by_gid(mesh, space.laplacian(c)), want), "");
        // Every owned row whole: as many entries as the serial row.
        const Matrix a = space.pattern();
        for (LO i = 0; i < a.rows; ++i) {
          EXPECT_EQ(static_cast<std::size_t>(a.offsets(i + 1) - a.offsets(i)), length.at(mesh.gid(0)(i)));
        }
        EXPECT_EQ(first_difference(owned_by_gid(mesh, space.rows(), space.load(c, Load{})), want_b), "");
        const auto boundary = space.boundary();
        for (LO v = 0; v < mesh.count(0); ++v) {
          if ((boundary(v) != 0) != on_box_boundary(mesh, v, solid(t))) {
            ADD_FAILURE() << "boundary of vertex " << mesh.gid(0)(v);
            break;
          }
        }
        // Rows partition the vertices.
        const Real rows = c.allreduce(static_cast<Real>(space.rows()), soma::Op::sum);
        EXPECT_EQ(rows, static_cast<Real>(whole.count(0)));
      }
    }
  }
}

// Coordinates moved after the space is made, by smoothing (which leaves
// copies and ghosts with their owners'), assemble as the serial ones do.
TEST(FemSpace, MovedCoordinatesAssembleAsSerialOnes) {
  const Comm c = Comm::world();
  soma::SmoothOptions o;
  o.iterations = 3;
  for (const Topology t : {Topology::quad, Topology::tet}) {
    const soma::testing::BoxSpec spec = spec_of(t, 4);
    soma::HostMesh whole = box_part(Comm::self(), spec);
    const Space serial(Comm::self(), whole);
    (void)soma::laplacian_smooth(Comm::self(), whole, o);
    const Entries want = by_gid(whole, serial.laplacian(Comm::self()));
    SCOPED_TRACE(soma::name(t));
    soma::HostMesh mesh = box_part(c, spec, Part{.deal = 1});
    const Space space(c, mesh);
    (void)soma::laplacian_smooth(c, mesh, o);
    EXPECT_EQ(first_difference(by_gid(mesh, space.laplacian(c)), want), "");
  }
}

// A space in the default execution space's memory assembles what the host
// one does.
TEST(FemSpace, DefaultMemorySpaceAssemblesTheHostMatrix) {
  using Mem = Kokkos::DefaultExecutionSpace::memory_space;
  const Comm c = Comm::world();
  for (const Topology t : {Topology::tri, Topology::hex}) {
    SCOPED_TRACE(soma::name(t));
    const soma::HostMesh host = box_part(c, spec_of(t, 4), Part{.deal = 1});
    const soma::Mesh<Mem> mesh = host.mirror<Mem>(soma::TransferOptions::everything());
    const soma::FemSpace<Mem> space(c, mesh);
    const soma::CsrMatrix<Mem> a = space.laplacian(c);
    const Matrix back{a.rows, a.cols, Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a.offsets),
                      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a.columns),
                      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a.values)};
    const Space reference(c, host);
    EXPECT_EQ(first_difference(by_gid(host, back), by_gid(host, reference.laplacian(c))), "");
  }
}

// A linear function is the discrete solution of Laplace's equation with
// its own boundary values, on any mesh of these elements: after Dirichlet
// elimination, A u = b to rounding on every rank's rows.
TEST(FemSpace, DirichletKeepsLinearFunctionsExact) {
  const Comm c = Comm::world();
  // Eliminated with the linear function's values: matrix and right side.
  const auto eliminate = [](const Comm& comm, const soma::HostMesh& mesh, const Space& space, Vector& u) {
    Matrix a = space.laplacian(comm);
    Vector b = space.load(comm, Load{});
    Vector g = space.vector();
    const auto x = mesh.coords();
    for (LO v = 0; v < mesh.count(0); ++v) u(v) = g(v) = 0.5 + x(v, 0) - 2 * x(v, 1) + 0.75 * x(v, 2);
    soma::apply_dirichlet(a, b, space.boundary(), soma::View1<const Real, Kokkos::HostSpace>(g));
    return std::make_pair(a, b);
  };
  for (const Topology t : all) {
    const soma::HostMesh whole = box_part(Comm::self(), spec_of(t, 4));
    const Space serial(Comm::self(), whole);
    Vector su = serial.vector();
    const auto [sa, sb] = eliminate(Comm::self(), whole, serial, su);
    const Entries want = by_gid(whole, sa);
    const auto want_b = owned_by_gid(whole, serial.rows(), sb);
    for (const int deal : {0, 1}) {
      SCOPED_TRACE(testing::Message() << soma::name(t) << " deal " << deal);
      const soma::HostMesh mesh = box_part(c, spec_of(t, 4), Part{.deal = deal});
      const Space space(c, mesh);
      Vector u = space.vector();
      auto [a, b] = eliminate(c, mesh, space, u);
      EXPECT_EQ(first_difference(by_gid(mesh, a), want), "");
      EXPECT_EQ(first_difference(owned_by_gid(mesh, space.rows(), b), want_b), "");
      // With f = 0 a linear function solves the eliminated system.
      Vector f0 = space.vector();
      Vector g = space.vector();
      Kokkos::deep_copy(g, u);
      a = space.laplacian(c);
      b = f0;
      soma::apply_dirichlet(a, b, space.boundary(), soma::View1<const Real, Kokkos::HostSpace>(g));
      Vector y = space.vector();
      space.multiply(c, a, u, y);
      for (LO i = 0; i < a.rows; ++i) {
        EXPECT_NEAR(y(i), b(i), 1e-12) << "vertex " << mesh.gid(0)(i);
        if (space.boundary()(i) != 0) {
          EXPECT_EQ(b(i), g(i));
          for (Offset k = a.offsets(i); k < a.offsets(i + 1); ++k) EXPECT_EQ(a.values(k), a.columns(k) == i ? 1 : 0);
        } else {
          for (Offset k = a.offsets(i); k < a.offsets(i + 1); ++k) {
            if (space.boundary()(a.columns(k)) != 0) EXPECT_EQ(a.values(k), 0);
          }
        }
      }
      {
        // Symmetric still, bit for bit, where both entries are here.
        std::map<std::pair<LO, LO>, Real> e;
        for (LO i = 0; i < a.rows; ++i) {
          for (Offset k = a.offsets(i); k < a.offsets(i + 1); ++k) e[{i, a.columns(k)}] = a.values(k);
        }
        for (const auto& [ij, v] : e) {
          const auto it = e.find({ij.second, ij.first});
          if (it != e.end() && it->second != v) {
            ADD_FAILURE() << "asymmetric at " << ij.first << " " << ij.second;
            break;
          }
        }
      }
    }
  }
}

// Products are the serial ones bit for bit, dot products to rounding; a
// product takes no collective, a dot product one, an assembly one.
TEST(FemSpace, ProductsAndDotsAreTheSerialOnes) {
  const Comm c = Comm::world();
  for (const Topology t : {Topology::tri, Topology::hex}) {
    const soma::testing::BoxSpec spec = spec_of(t, 4);
    const auto fill = [](const soma::HostMesh& m, const Vector& v) {
      for (LO i = 0; i < m.count(0); ++i) v(i) = std::sin(0.37 * static_cast<Real>(m.gid(0)(i)));
    };
    const soma::HostMesh whole = box_part(Comm::self(), spec);
    const Space serial(Comm::self(), whole);
    const Matrix sa = serial.laplacian(Comm::self());
    Vector sx = serial.vector(), sy = serial.vector();
    fill(whole, sx);
    serial.multiply(Comm::self(), sa, sx, sy);
    const auto want = owned_by_gid(whole, serial.rows(), sy);
    const Real want_dot = serial.dot(Comm::self(), sx, sy);
    const Real want_xx = serial.dot(Comm::self(), sx, sx);
    for (const int deal : {0, 1}) {
      SCOPED_TRACE(testing::Message() << soma::name(t) << " deal " << deal);
      const soma::HostMesh mesh = box_part(c, spec, Part{.deal = deal});
      const Space space(c, mesh);
      const auto before = soma::detail::comm_counts().collectives;
      const Matrix a = space.laplacian(c);
      const auto assembled = soma::detail::comm_counts().collectives;
      Vector x = space.vector(), y = space.vector();
      // Only the owned entries are set: the product's update fills the rest.
      for (LO i = 0; i < space.rows(); ++i) x(i) = std::sin(0.37 * static_cast<Real>(mesh.gid(0)(i)));
      space.multiply(c, a, x, y);
      const auto multiplied = soma::detail::comm_counts().collectives;
      const Real d = space.dot(c, x, y);
      const auto dotted = soma::detail::comm_counts().collectives;
      EXPECT_EQ(assembled - before, 1u);
      EXPECT_EQ(multiplied - assembled, 0u);
      EXPECT_EQ(dotted - multiplied, 1u);
      EXPECT_EQ(first_difference(owned_by_gid(mesh, space.rows(), y), want), "");
      EXPECT_NEAR(d, want_dot, 1e-13 * std::abs(want_dot));
      // x's copies and ghosts hold values now: they must not count.
      EXPECT_NEAR(space.dot(c, x, x), want_xx, 1e-13 * want_xx);
    }
  }
}

// A space of `mesh` throws std::invalid_argument saying `what`.
void expect_message(const Comm& c, const soma::HostMesh& mesh, const std::string& what) {
  try {
    const Space bad(c, mesh);
    ADD_FAILURE() << "accepted: " << what;
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find(what), std::string::npos) << e.what();
  }
}

// Every rank throws when one rank's mesh will not do.
TEST(FemSpace, BadMeshesThrowOnEveryRank) {
  const Comm c = Comm::world();
  const soma::testing::BoxSpec spec = spec_of(Topology::quad, 4);
  if (c.size() > 1) {
    const soma::HostMesh bare = box_part(c, spec, Part{.layers = 0});
    EXPECT_THROW(Space(c, bare), std::invalid_argument);
    const soma::HostMesh by_edges = box_part(c, spec, Part{.bridge = 1});
    EXPECT_THROW(Space(c, by_edges), std::invalid_argument);
  }
  expect_message(c, box_part(c, spec, Part{.orphan = true}), "no cell there has");
  // A quadratic triangle, on the first rank only.
  {
    soma::MeshBuilder b(2);
    if (c.rank() == 0) {
      const std::vector<GO> vg = {1, 2, 3, 4, 5, 6};
      const std::vector<Real> xyz = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0};
      b.add_vertices(vg, xyz);
      const std::vector<GO> one = {1};
      b.add_elements(Topology::tri, one, vg, 6);
    }
    soma::HostMesh quadratic = soma::build_local_part(c, std::move(b));
    if (c.size() > 1) quadratic.add_ghosts(c, soma::GhostOptions{1, 0, false});
    expect_message(c, quadratic, "high-order");
  }
  // A cell turned inside out on the last rank only, before and after.
  soma::HostMesh mesh = box_part(c, spec);
  const Space space(c, mesh), atomic(c, mesh, soma::Assembly::atomic), colored(c, mesh, soma::Assembly::colored);
  const auto x = mesh.coords();
  const Real keep = x(0, 0);
  if (c.rank() == c.size() - 1) x(0, 0) = 50;
  EXPECT_THROW((void)space.laplacian(c), std::invalid_argument);
  EXPECT_THROW((void)atomic.laplacian(c), std::invalid_argument);
  EXPECT_THROW((void)colored.laplacian(c), std::invalid_argument);
  EXPECT_THROW((void)space.load(c, Load{}), std::invalid_argument);
  EXPECT_THROW(Space(c, mesh), std::invalid_argument);
  x(0, 0) = keep;
  EXPECT_NO_THROW((void)space.laplacian(c));
  EXPECT_NO_THROW((void)atomic.laplacian(c));
  EXPECT_NO_THROW((void)colored.laplacian(c));
  // A mesh of prisms, on the first rank only.
  soma::MeshBuilder b(3);
  if (c.rank() == 0) {
    const std::vector<GO> vg = {1, 2, 3, 4, 5, 6};
    const std::vector<Real> xyz = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1};
    b.add_vertices(vg, xyz);
    const std::vector<GO> one = {1};
    b.add_elements(Topology::prism, one, vg);
  }
  soma::HostMesh prisms = soma::build_local_part(c, std::move(b));
  if (c.size() > 1) prisms.add_ghosts(c, soma::GhostOptions{1, 0, false});
  expect_message(c, prisms, "other than a tri, quad, tet or hex");
}

// The local operations refuse vectors too short for the matrix; an update
// any vector not of cols() entries.
TEST(FemSpace, ShortVectorsAreRefused) {
  const soma::HostMesh mesh = box_part(Comm::self(), spec_of(Topology::quad, 3));
  const Space space(Comm::self(), mesh);
  Matrix a = space.laplacian(Comm::self());
  const Vector x = space.vector(), y = space.vector();
  const Vector shorter("shorter", static_cast<std::size_t>(space.cols() - 1));
  using CV = soma::View1<const Real, Kokkos::HostSpace>;
  EXPECT_THROW(soma::multiply(a, CV(shorter), y), std::invalid_argument);
  EXPECT_NO_THROW(soma::multiply(a, CV(x), y));
  const soma::View1<std::uint8_t, Kokkos::HostSpace> few("few", static_cast<std::size_t>(space.cols() - 1));
  EXPECT_THROW(soma::apply_dirichlet(a, shorter, space.boundary(), CV(x)), std::invalid_argument);
  EXPECT_THROW(soma::apply_dirichlet(a, Vector(y), soma::View1<const std::uint8_t, Kokkos::HostSpace>(few), CV(x)),
               std::invalid_argument);
  EXPECT_THROW(soma::apply_dirichlet(a, Vector(y), space.boundary(), CV(shorter)), std::invalid_argument);
  EXPECT_THROW(space.update(Comm::self(), shorter), std::invalid_argument);
  const Vector longer("longer", static_cast<std::size_t>(space.cols() + 1));
  EXPECT_THROW(space.update(Comm::self(), longer), std::invalid_argument);
}

// Colors partition the cells with an owned corner and no two cells of a
// color share an owned vertex (only owned rows are written). Greedy coloring
// by gid needs at least as many colors as there are cells around an owned
// vertex and at most one more than a cell's conflicts; on a whole box of
// quads or hexes exactly 2^d. The small box leaves some ranks no cells.
TEST(FemSpace, CellColorsAreValidAndFew) {
  const Comm c = Comm::world();
  for (const Topology t : all) {
    for (const soma::testing::BoxSpec& spec : {spec_of(t, 6), soma::testing::BoxSpec{2, 1, 1, t, 1.0, 1.0, 1.0}}) {
      for (const int deal : {0, 1}) {
        SCOPED_TRACE(testing::Message() << soma::name(t) << " " << spec.nx << " deal " << deal);
        const soma::HostMesh mesh = box_part(c, spec, Part{.deal = deal, .perturb = false});
        const int dim = mesh.dim();
        const auto m = mesh.view();
        const LO rows = mesh.segment_table(0).owned_end;
        const soma::detail::CellGroups g =
            soma::detail::cell_groups(mesh, rows, true, soma::detail::cells_around(mesh, rows));
        std::set<LO> seen;
        for (std::size_t k = 0; k + 1 < g.offsets.size(); ++k) {
          EXPECT_LT(g.offsets[k], g.offsets[k + 1]) << "color " << k << " empty";
          std::set<LO> used;
          for (Offset i = g.offsets[k]; i < g.offsets[k + 1]; ++i) {
            const LO e = g.cells[static_cast<std::size_t>(i)];
            EXPECT_TRUE(seen.insert(e).second) << "cell " << e << " twice";
            for (LO p = 0; p < m.num_vertices(dim, e); ++p) {
              const LO v = m.vertex(dim, e, p);
              if (v < rows) EXPECT_TRUE(used.insert(v).second) << "color " << k << " shares vertex " << v;
            }
          }
        }
        std::map<LO, std::set<LO>> around;  // owned vertex: its cells
        for (LO e = 0; e < mesh.count(dim); ++e) {
          for (LO p = 0; p < m.num_vertices(dim, e); ++p) {
            if (m.vertex(dim, e, p) < rows) around[m.vertex(dim, e, p)].insert(e);
          }
        }
        std::set<LO> active;
        for (const auto& [v, cells] : around) active.insert(cells.begin(), cells.end());
        EXPECT_EQ(seen, active);
        const std::size_t colors = g.offsets.size() - 1;
        std::size_t clique = 0, conflicts = 0;
        for (const auto& [v, cells] : around) clique = std::max(clique, cells.size());
        for (const LO e : active) {
          std::set<LO> others;
          for (LO p = 0; p < m.num_vertices(dim, e); ++p) {
            const auto it = around.find(m.vertex(dim, e, p));
            if (it != around.end()) others.insert(it->second.begin(), it->second.end());
          }
          conflicts = std::max(conflicts, others.size() - 1);
        }
        EXPECT_GE(colors, clique);
        EXPECT_LE(colors, active.empty() ? 0 : conflicts + 1);
        if (c.size() == 1 && spec.nx == 7 && (t == Topology::quad || t == Topology::hex)) {
          EXPECT_EQ(colors, std::size_t{1} << dim);
        }
      }
    }
  }
}

// Atomic and colored assembly give the gather's matrix to rounding, and
// colored assembly the same bits every time.
TEST(FemSpace, AssemblyMethodsAgree) {
  const Comm c = Comm::world();
  for (const Topology t : all) {
    for (const int deal : {0, 1}) {
      SCOPED_TRACE(testing::Message() << soma::name(t) << " deal " << deal);
      const soma::HostMesh mesh = box_part(c, spec_of(t, 4), Part{.deal = deal});
      const Entries gather = by_gid(mesh, Space(c, mesh).laplacian(c));
      Real scale = 0;
      for (const auto& [rc, v] : gather) scale = std::max(scale, std::abs(v));
      for (const soma::Assembly method : {soma::Assembly::atomic, soma::Assembly::colored}) {
        const Space space(c, mesh, method);
        const Entries got = by_gid(mesh, space.laplacian(c));
        ASSERT_EQ(got.size(), gather.size());
        Real worst = 0;
        for (const auto& [rc, v] : got) worst = std::max(worst, std::abs(v - gather.at(rc)));
        EXPECT_LE(worst, 1e-14 * scale) << "method " << static_cast<int>(method);
        if (method == soma::Assembly::colored) {
          // The colors in turn, each cell's entries added sequentially: the
          // same bits, so no thread count changes them.
          const int dim = mesh.dim();
          const auto m = mesh.view();
          const auto vg = mesh.gid(0);
          const LO rows = mesh.segment_table(0).owned_end;
          const soma::detail::CellGroups groups =
              soma::detail::cell_groups(mesh, rows, true, soma::detail::cells_around(mesh, rows));
          Entries replay;
          for (const LO e : groups.cells) {
            const fem::Cell cell = fem::cell_of(m, dim, e);
            Real k[fem::max_nodes][fem::max_nodes];
            fem::stiffness(cell, k);
            for (int p = 0; p < cell.n; ++p) {
              if (m.vertex(dim, e, p) >= rows) continue;
              for (int q = 0; q < cell.n; ++q) replay[{vg(m.vertex(dim, e, p)), vg(m.vertex(dim, e, q))}] += k[p][q];
            }
          }
          EXPECT_EQ(first_difference(got, replay), "");
          EXPECT_EQ(first_difference(by_gid(mesh, space.laplacian(c)), got), "");
        }
      }
    }
  }
}

namespace {

// soma::reproducible() set for a scope.
struct Reproducible {
  bool saved = soma::reproducible();
  explicit Reproducible(bool on) { soma::set_reproducible(on); }
  ~Reproducible() { soma::set_reproducible(saved); }
};

std::uint64_t bits(Real x) { return std::bit_cast<std::uint64_t>(x); }

}  // namespace

// Reproducible: every assembly gathers (and says so), and dots are the
// serial ones bit for bit, in one allreduce. Not, the dots differ from the
// serial ones in some partition.
TEST(FemSpace, ReproducibleSpacesGatherAndDotExactly) {
  const Comm c = Comm::world();
  int plain_differ = 0;
  for (const bool on : {true, false}) {
    const Reproducible scope(on);
    for (const Topology t : {Topology::tri, Topology::hex}) {
      const soma::testing::BoxSpec spec = spec_of(t, 5);
      const soma::HostMesh whole = box_part(Comm::self(), spec);
      const Space serial(Comm::self(), whole);
      Vector sx = serial.vector(), sy = serial.vector();
      for (LO i = 0; i < whole.count(0); ++i) {
        sx(i) = std::sin(0.37 * static_cast<Real>(whole.gid(0)(i)));
        sy(i) = std::cos(0.11 * static_cast<Real>(whole.gid(0)(i))) * 1e3;
      }
      const Real want = serial.dot(Comm::self(), sx, sy);
      for (const int deal : {0, 1}) {
        SCOPED_TRACE(testing::Message() << on << " " << soma::name(t) << " deal " << deal);
        const soma::HostMesh mesh = box_part(c, spec, Part{.deal = deal});
        const Space gathered(c, mesh);
        EXPECT_EQ(gathered.reproducible(), on);
        const Matrix a = gathered.laplacian(c);
        for (const soma::Assembly m : {soma::Assembly::atomic, soma::Assembly::colored}) {
          const Space other(c, mesh, m);
          EXPECT_EQ(other.assembly(), on ? soma::Assembly::gather : m);
          if (!on) continue;
          const Matrix b = other.laplacian(c);
          LO differ = 0;
          for (std::size_t k = 0; k < a.values.extent(0); ++k) differ += bits(a.values(k)) == bits(b.values(k)) ? 0 : 1;
          EXPECT_EQ(differ, 0);
        }
        Vector x = gathered.vector(), y = gathered.vector();
        for (LO i = 0; i < mesh.count(0); ++i) {
          x(i) = std::sin(0.37 * static_cast<Real>(mesh.gid(0)(i)));
          y(i) = std::cos(0.11 * static_cast<Real>(mesh.gid(0)(i))) * 1e3;
        }
        const auto before = soma::detail::comm_counts().collectives;
        const Real d = gathered.dot(c, x, y);
        EXPECT_EQ(soma::detail::comm_counts().collectives - before, 1u);
        if (on) {
          EXPECT_EQ(bits(d), bits(want));
        } else {
          plain_differ += bits(d) == bits(want) ? 0 : 1;
        }
      }
    }
  }
  if (c.size() > 1) EXPECT_GT(plain_differ, 0);
}

// Ranks that differ in soma::reproducible() cannot make a space: every rank
// throws.
TEST(FemSpace, RanksMustAgreeOnReproducible) {
  const Comm c = Comm::world();
  const soma::HostMesh mesh = box_part(c, spec_of(Topology::quad, 4));
  const Reproducible scope(c.size() > 1 && c.rank() == c.size() - 1);
  if (c.size() > 1) {
    expect_message(c, mesh, "differ in soma::reproducible()");
  } else {
    EXPECT_NO_THROW(Space(c, mesh));
  }
}

// The numbering runs through the ranks' owned vertices in turn; a copy or a
// ghost carries its owner's number, so numbers map one to one onto gids.
TEST(FemSpace, NumberingIsContiguousAndAgreesOnEveryHolder) {
  const Comm c = Comm::world();
  for (const int deal : {0, 1}) {
    SCOPED_TRACE(testing::Message() << "deal " << deal);
    const soma::HostMesh mesh = box_part(c, spec_of(Topology::tet, 4), Part{.deal = deal});
    const Space space(c, mesh);
    const soma::Numbering<Kokkos::HostSpace> n = space.numbering(c);
    EXPECT_EQ(n.total, static_cast<GO>(soma::testing::box_mesh(spec_of(Topology::tet, 4)).count(0)));
    EXPECT_EQ(n.first, c.exscan(static_cast<GO>(space.rows())));
    for (LO v = 0; v < space.rows(); ++v) EXPECT_EQ(n.index(v), n.first + v);
    // gid -> number, from the owners, checked against every holder's.
    std::vector<GO> by_gid(static_cast<std::size_t>(n.total) + 1, 0);
    for (LO v = 0; v < space.rows(); ++v) by_gid[static_cast<std::size_t>(mesh.gid(0)(v))] = n.index(v) + 1;
    c.allreduce(std::span<GO>(by_gid), soma::Op::max);
    LO wrong = 0;
    for (LO v = 0; v < space.cols(); ++v) {
      wrong += by_gid[static_cast<std::size_t>(mesh.gid(0)(v))] == n.index(v) + 1 ? 0 : 1;
    }
    EXPECT_EQ(wrong, 0);
  }
}
