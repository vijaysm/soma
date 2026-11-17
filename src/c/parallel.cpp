// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The C API's partitioning, migration and ghost layers.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "handles.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/parallel/partition.hpp"

using soma::c::call;
using soma::c::check_array;
using soma::c::need;
using soma::c::out;

extern "C" {

int soma_partition_by_curve(const soma_comm_t* comm, soma_lo_t n, const double* xyz, const soma_go_t* gids,
                            int* dest) {
  return call([&] {
    constexpr const char* what = "soma_partition_by_curve";
    const soma::Comm& c = *need(comm, what).comm;
    soma::c::agree(c, what, [&] {
      check_array(xyz, 3 * static_cast<std::int64_t>(n), what);
      check_array(gids, n, what);
      check_array(dest, n, what);
    });
    const std::vector<int> d = soma::partition_by_curve(c, {xyz, 3 * static_cast<std::size_t>(n)},
                                                        {gids, static_cast<std::size_t>(n)});
    std::copy(d.begin(), d.end(), dest);
  });
}

int soma_migrate(const soma_comm_t* comm, const soma_mesh_t* mesh, soma_lo_t n, const int* dest, int ordering,
                 soma_mesh_t** result) {
  return call([&] {
    constexpr const char* what = "soma_migrate";
    const soma::Comm& c = *need(comm, what).comm;
    soma_mesh_t*& r = out(result, what);
    r = nullptr;
    const soma_mesh* m = nullptr;
    soma::Ordering o{};
    soma::c::agree(c, what, [&] {
      m = &need(mesh, what);
      o = soma::c::ordering_of(ordering, what);
      const soma::LO held = m->get().count(m->get().dim());
      if (n != held) {
        throw std::invalid_argument(std::string(what) + ": " + std::to_string(n) + " destinations for " +
                                    std::to_string(held) + " elements held");
      }
      check_array(dest, n, what);
      for (soma_lo_t e = 0; e < n; ++e) {
        if (dest[e] < 0 || dest[e] >= c.size()) {
          throw std::invalid_argument(std::string(what) + ": destination " + std::to_string(dest[e]) +
                                      " is not a rank of the communicator");
        }
      }
    });
    r = soma_c_new_mesh(soma::migrate(c, m->get(), {dest, static_cast<std::size_t>(n)}, o));
  });
}

int soma_rebalance(const soma_comm_t* comm, const soma_mesh_t* mesh, int ordering, soma_mesh_t** result) {
  return call([&] {
    constexpr const char* what = "soma_rebalance";
    const soma::Comm& c = *need(comm, what).comm;
    soma_mesh_t*& r = out(result, what);
    r = nullptr;
    const soma_mesh* m = nullptr;
    soma::Ordering o{};
    soma::c::agree(c, what, [&] {
      m = &need(mesh, what);
      o = soma::c::ordering_of(ordering, what);
    });
    r = soma_c_new_mesh(soma::rebalance(c, m->get(), o));
  });
}

int soma_mesh_add_ghosts(const soma_comm_t* comm, soma_mesh_t* mesh, int layers, int bridge_dim, int sets) {
  return call([&] {
    constexpr const char* what = "soma_mesh_add_ghosts";
    const soma::Comm& c = *need(comm, what).comm;
    soma_mesh* m = nullptr;
    soma::c::agree(c, what, [&] {
      m = &need(mesh, what);
      if (layers < 0) throw std::invalid_argument(std::string(what) + ": layers " + std::to_string(layers) + " < 0");
      if (bridge_dim < 0 || bridge_dim >= m->get().dim()) {
        throw std::invalid_argument(std::string(what) + ": bridge_dim " + std::to_string(bridge_dim) +
                                    " is not below the mesh's dimension " + std::to_string(m->get().dim()));
      }
    });
    m->state->changed();
    m->get().add_ghosts(c, soma::GhostOptions{.layers = layers, .bridge_dim = bridge_dim, .sets = sets != 0});
  });
}

int soma_mesh_remove_ghosts(const soma_comm_t* comm, soma_mesh_t* mesh) {
  return call([&] {
    constexpr const char* what = "soma_mesh_remove_ghosts";
    const soma::Comm& c = *need(comm, what).comm;
    soma_mesh* m = nullptr;
    soma::c::agree(c, what, [&] { m = &need(mesh, what); });
    m->state->changed();
    m->get().remove_ghosts(c);
  });
}

}  // extern "C"
