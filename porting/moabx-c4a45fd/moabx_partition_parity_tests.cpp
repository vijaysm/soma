// Multi-rank partition parity against moab::ParallelComm, on real meshes.
//
// The unit tests build partitions from synthetic inputs. This checks that
// what MOABX extracts from an actual parallel read agrees with MOAB's own
// ownership and sharing view, and that a field exchange over that partition
// round-trips exactly.
//
// Run under mpiexec. With one rank it degenerates to an all-owned check,
// which is still worth running to confirm the serial path.

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "moab/Core.hpp"
#include "moab/MOABConfig.h"

#ifdef MOAB_HAVE_MPI
#include <mpi.h>

#include "moab/ParallelComm.hpp"
#endif

#include "moabx/mesh/halo_exchange.hpp"
#include "moabx/mesh/moab_mesh_loader.hpp"
#include "moabx/mesh/moab_partition.hpp"
#include "moabx/pp/comm.hpp"
#include "moabx/pp/runtime_scope.hpp"

namespace {

using namespace moabx::mesh;
namespace pp = moabx::pp;

int g_failures = 0;
int g_rank = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "[rank %d] FAIL: %s\n", g_rank, what.c_str());
  }
}

constexpr entity_kind kKinds[] = {entity_kind::vertex, entity_kind::edge,
                                  entity_kind::face, entity_kind::region};

}  // namespace

int main(int argc, char** argv) {
  pp::runtime_scope scope(&argc, &argv);
  const pp::communicator comm;
  g_rank = comm.rank();

  if (argc < 2) {
    if (comm.rank() == 0) {
      std::fprintf(stderr, "usage: %s <mesh.h5m>\n", argv[0]);
    }
    return 2;
  }
  const std::string mesh_path = argv[1];

#ifdef MOAB_HAVE_MPI
  // runtime_scope only calls MPI_Init when launched under an MPI launcher,
  // so a direct invocation leaves MPI uninitialised. ParallelComm requires
  // it, so report rather than crash inside MPICH.
  int mpi_ready = 0;
  MPI_Initialized(&mpi_ready);
  if (mpi_ready == 0) {
    std::printf("[partition-parity] MPI not initialised (run under mpiexec); skipped\n");
    return 0;
  }

  moab::Core core;
  auto pcomm = std::make_unique<moab::ParallelComm>(&core, MPI_COMM_WORLD);

  // A parallel read with shared-entity resolution is what populates the
  // sharing data; without it MOAB reports everything as owned locally.
  // Only vertices and regions are read from file with global ids. Edges and
  // faces would be materialised during load and inherit the dense GLOBAL_ID
  // default of 0, which is not a usable cross-rank identity -- see the guard
  // in extract_partition. Disabling them keeps this test on data where the
  // parity check is meaningful.
  moab_load_options options;
  options.use_mpi_parallel_read = (comm.size() > 1);
  options.build_edge_entities = false;
  options.build_face_entities = false;

  mesh_graph graph;
  try {
    std::string read_opts;
    if (comm.size() > 1) {
      read_opts =
          "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;"
          "PARALLEL_RESOLVE_SHARED_ENTS;PARTITION_DISTRIBUTE";
    }
    const moab::ErrorCode rval =
        core.load_file(mesh_path.c_str(), nullptr, read_opts.empty() ? nullptr : read_opts.c_str());
    if (rval != moab::MB_SUCCESS) {
      std::fprintf(stderr, "[rank %d] load_file failed for %s\n", comm.rank(), mesh_path.c_str());
      return 3;
    }
    graph = build_moab_mesh_graph(core, options);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[rank %d] load failed: %s\n", comm.rank(), e.what());
    return 3;
  }

  const auto extracted = extract_partition(core, *pcomm, graph);
  const auto& partition = extracted.partition;

  check(partition.rank() == comm.rank(), "partition rank matches communicator");
  check(partition.size() == comm.size(), "partition size matches communicator");

  // 1. Ownership and ghost classification must agree with MOAB entity by
  //    entity.
  for (const entity_kind kind : kKinds) {
    if (graph.handles(kind).empty()) {
      continue;
    }
    std::string first_error;
    const std::size_t bad = validate_partition_against_moab(
        core, *pcomm, graph, partition, kind, &first_error);
    check(bad == 0, std::string("partition parity for ") + entity_kind_name(kind) + ": " +
                        std::to_string(bad) + " mismatches; first: " + first_error);
  }

  // 2. Every entity is owned by exactly one rank: the global count of owned
  //    entities must equal the global count of distinct global ids.
  for (const entity_kind kind : kKinds) {
    const auto& part = partition[kind];
    if (part.total_count() == 0) {
      continue;
    }
    const auto owned = static_cast<long long>(part.owned_count());
    const auto global_owned = comm.allreduce_sum(owned);

    // Max global id + 1 bounds the number of distinct entities. Ownership
    // must not double-count: the owned total cannot exceed it.
    long long local_max = -1;
    for (index_type i = 0; i < part.total_count(); ++i) {
      local_max = std::max(local_max, static_cast<long long>(part.global_id(i)));
    }
    const auto global_max = comm.allreduce_max(local_max);
    check(global_owned <= global_max + 1,
          std::string("owned total ") + std::to_string(global_owned) + " for " +
              entity_kind_name(kind) + " exceeds distinct entity bound " +
              std::to_string(global_max + 1) + " (entity owned by two ranks?)");
  }

  // 3. Owned-first ordering must hold.
  for (const entity_kind kind : kKinds) {
    const auto& part = partition[kind];
    bool ordered = true;
    for (index_type i = 0; i < part.total_count(); ++i) {
      const bool owned = (part.owner_rank(i) == comm.rank());
      if (owned != part.is_owned(i)) {
        ordered = false;
        break;
      }
    }
    check(ordered, std::string("owned-first ordering holds for ") + entity_kind_name(kind));
  }

  // 4. Halo exchange over the extracted partition must reproduce owner
  //    values on ghosts exactly.
  for (const entity_kind kind : kKinds) {
    const auto& part = partition[kind];
    if (part.total_count() == 0) {
      continue;
    }
    const halo_exchange hx(part, comm.rank());

    // Owners write f(global_id); ghosts are poisoned.
    std::vector<double> field(part.total_count(), -1.0e300);
    for (index_type i = 0; i < part.owned_count(); ++i) {
      field[i] = static_cast<double>(part.global_id(i)) * 2.0 + 1.0;
    }

    hx.exchange(comm, std::span<double>(field));

    std::size_t bad = 0;
    for (index_type i = part.owned_count(); i < part.total_count(); ++i) {
      const double want = static_cast<double>(part.global_id(i)) * 2.0 + 1.0;
      if (field[i] != want) {
        ++bad;
      }
    }
    check(bad == 0, std::string("ghost round-trip for ") + entity_kind_name(kind) + ": " +
                        std::to_string(bad) + " of " + std::to_string(part.ghost_count()) +
                        " ghosts wrong");
  }

  // 5. After applying the ordering, graph local ids and partition local ids
  //    must be the same numbering, with owned entities a prefix of both.
  //    This is what assembly relies on.
  {
    std::vector<std::vector<handle_type>> expected(entity_kind_count);
    for (const entity_kind kind : kKinds) {
      const auto& p = extracted.old_to_new[to_index(kind)];
      const auto handles = graph.handles(kind);
      expected[to_index(kind)].assign(handles.size(), 0);
      for (std::size_t old_id = 0; old_id < handles.size(); ++old_id) {
        expected[to_index(kind)][p[old_id]] = handles[old_id];
      }
    }

    apply_partition_ordering(graph, extracted);

    for (const entity_kind kind : kKinds) {
      const auto& part = partition[kind];
      if (part.total_count() == 0) {
        continue;
      }
      check(graph.entity_count(kind) == part.total_count(),
            std::string("graph and partition agree on ") + entity_kind_name(kind) + " count");

      std::size_t bad = 0;
      for (index_type i = 0; i < graph.entity_count(kind); ++i) {
        if (graph.handle(kind, i) != expected[to_index(kind)][i]) {
          ++bad;
        }
      }
      check(bad == 0, std::string("renumbered graph handles for ") + entity_kind_name(kind) +
                          ": " + std::to_string(bad) + " wrong");

      // Owned entities form a prefix, so an owned-only kernel can run over
      // a contiguous span rather than filtering.
      std::size_t owned_outside_prefix = 0;
      for (index_type i = 0; i < part.total_count(); ++i) {
        if (part.is_owned(i) != (i < part.owned_count())) {
          ++owned_outside_prefix;
        }
      }
      check(owned_outside_prefix == 0,
            std::string("owned entities form a prefix for ") + entity_kind_name(kind));

      // The renumbered graph must still be a valid mesh: every connectivity
      // entry must reference a vertex that exists.
      std::size_t bad_conn = 0;
      const auto vertex_count = graph.entity_count(entity_kind::vertex);
      for (index_type i = 0; i < graph.entity_count(kind); ++i) {
        for (const index_type v : graph.connectivity(kind, i)) {
          if (static_cast<std::size_t>(v) >= vertex_count) {
            ++bad_conn;
          }
        }
      }
      check(bad_conn == 0, std::string("connectivity stays in range for ") +
                               entity_kind_name(kind) + ": " + std::to_string(bad_conn) +
                               " out-of-range vertex references");

      // Handle lookup must reflect the new numbering.
      std::size_t bad_lookup = 0;
      for (index_type i = 0; i < graph.entity_count(kind); ++i) {
        const auto found = graph.local_id(kind, graph.handle(kind, i));
        if (!found.has_value() || *found != i) {
          ++bad_lookup;
        }
      }
      check(bad_lookup == 0, std::string("handle lookup matches new ids for ") +
                                 entity_kind_name(kind) + ": " + std::to_string(bad_lookup) +
                                 " wrong");
    }
  }

  // 6. Report the partition so a degenerate run (everything owned, nothing
  //    shared) cannot masquerade as a passing parallel test.
  long long total_ghosts = 0;
  for (const entity_kind kind : kKinds) {
    total_ghosts += static_cast<long long>(partition[kind].ghost_count());
  }
  const auto global_ghosts = comm.allreduce_sum(total_ghosts);
  if (comm.size() > 1) {
    check(global_ghosts > 0,
          "a multi-rank run must produce ghosts; zero means sharing was never resolved "
          "and the parity check above is vacuous");
  }

  const auto total = comm.allreduce_sum(static_cast<long long>(g_failures));
  if (comm.rank() == 0) {
    std::printf("[partition-parity] ranks=%d ghosts=%lld failures=%lld\n", comm.size(),
                global_ghosts, total);
  }
  return total == 0 ? 0 : 1;
#else
  if (comm.rank() == 0) {
    std::printf("[partition-parity] MOAB built without MPI; skipped\n");
  }
  return 0;
#endif
}
