// Multi-rank halo exchange validation.
//
// Run under mpiexec at 1/2/4/8 ranks. Builds a 1-D ring of entities split
// across ranks, where each rank owns a contiguous block and ghosts its two
// neighbours' adjacent entities. The ring guarantees every rank has two
// distinct neighbours at size >= 3, which is what exposes ordering bugs that
// a 2-rank test cannot.
//
// Exchange moves bytes and does no arithmetic, so ghost values are compared
// for EXACT equality with the owner's. A tolerance here would hide real bugs.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "moabx/mesh/halo_exchange.hpp"
#include "moabx/mesh/partition.hpp"
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

// The field value an owner must produce for a global id. Distinct per id and
// exactly representable, so any mismatch is a real error.
double expected_value(global_id_type gid, std::size_t component) {
  return static_cast<double>(gid) * 8.0 + static_cast<double>(component);
}

// Ring decomposition: `per_rank` entities owned by each rank, plus a halo of
// `halo_depth` ghosts from the left neighbour and the same from the right.
//
// halo_depth > 1 is essential: with a single ghost per neighbour, the order
// of elements within a neighbour message is unobservable, and an ordering
// bug in the schedule cannot be detected. A deeper halo makes the send and
// receive sequences for a pair genuinely order-sensitive.
struct ring {
  kind_partition partition;
  std::vector<global_id_type> gids;  // indexed by new local id
};

ring build_ring(int rank, int size, std::size_t per_rank, std::size_t halo_depth = 3) {
  const auto total = static_cast<global_id_type>(size * per_rank);
  const auto first = static_cast<global_id_type>(rank * per_rank);

  partition_builder b(rank, per_rank + 2);

  // Owned block.
  for (std::size_t i = 0; i < per_rank; ++i) {
    b.add_entity(rank, first + static_cast<global_id_type>(i));
  }

  const int left = (rank - 1 + size) % size;
  const int right = (rank + 1) % size;

  // A halo cannot be deeper than a neighbour's owned block.
  const std::size_t depth = std::min(halo_depth, per_rank);

  // Ghosts from the left neighbour: the last `depth` entities of its block.
  // Added in increasing global id so that insertion order does NOT match the
  // required send order for the reversed-id case -- the schedule must sort.
  if (size > 1) {
    for (std::size_t k = 0; k < depth; ++k) {
      const auto gid = (first - static_cast<global_id_type>(depth - k) + total) % total;
      b.add_entity(left, gid);
    }
    if (size > 2) {
      for (std::size_t k = 0; k < depth; ++k) {
        const auto gid = (first + static_cast<global_id_type>(per_rank + k)) % total;
        b.add_entity(right, gid);
      }
    }
  }

  // Mirror image: this rank's boundary entities are ghosted by its
  // neighbours. The last `depth` owned go right, the first `depth` go left.
  if (size > 1) {
    for (std::size_t k = 0; k < depth; ++k) {
      b.add_remote_copy(per_rank - depth + k, right, static_cast<index_type>(k));
    }
    if (size > 2) {
      for (std::size_t k = 0; k < depth; ++k) {
        b.add_remote_copy(k, left, static_cast<index_type>(k));
      }
    }
  }

  auto result = std::move(b).finish();

  std::vector<global_id_type> gids(result.partition.total_count());
  for (index_type i = 0; i < result.partition.total_count(); ++i) {
    gids[i] = result.partition.global_id(i);
  }
  return ring{std::move(result.partition), std::move(gids)};
}

void run_case(const pp::communicator& comm, std::size_t per_rank, std::size_t components) {
  const int rank = comm.rank();
  const int size = comm.size();
  const auto r = build_ring(rank, size, per_rank);
  const auto& part = r.partition;

  const halo_exchange hx(part, rank);
  const std::string ctx =
      " (per_rank=" + std::to_string(per_rank) + " comps=" + std::to_string(components) + ")";

  check(part.owned_count() == per_rank, "owned count" + ctx);

  // Owners write their values; ghosts are poisoned so a missing update is
  // detected rather than accidentally matching.
  std::vector<double> field(part.total_count() * components, -999.0);
  for (index_type i = 0; i < part.owned_count(); ++i) {
    for (std::size_t c = 0; c < components; ++c) {
      field[static_cast<std::size_t>(i) * components + c] = expected_value(part.global_id(i), c);
    }
  }

  hx.exchange(comm, std::span<double>(field), components);

  // Every ghost must now hold its owner's value exactly.
  for (index_type i = part.owned_count(); i < part.total_count(); ++i) {
    for (std::size_t c = 0; c < components; ++c) {
      const double got = field[static_cast<std::size_t>(i) * components + c];
      const double want = expected_value(part.global_id(i), c);
      check(got == want, "ghost gid=" + std::to_string(part.global_id(i)) + " comp=" +
                             std::to_string(c) + " got=" + std::to_string(got) +
                             " want=" + std::to_string(want) + ctx);
    }
  }

  // Owned values must be untouched by the exchange.
  for (index_type i = 0; i < part.owned_count(); ++i) {
    for (std::size_t c = 0; c < components; ++c) {
      const double got = field[static_cast<std::size_t>(i) * components + c];
      check(got == expected_value(part.global_id(i), c), "owned value modified" + ctx);
    }
  }
}

// Multiple ghosts from the same neighbour, so the order of elements within
// a single message is observable. This is what detects a schedule whose send
// and receive sequences disagree.
void run_deep_halo_case(const pp::communicator& comm) {
  for (const std::size_t depth : {2u, 5u}) {
    const auto r = build_ring(comm.rank(), comm.size(), 8, depth);
    const auto& part = r.partition;
    const halo_exchange hx(part, comm.rank());
    const std::string ctx = " (depth=" + std::to_string(depth) + ")";

    if (comm.size() > 2) {
      check(part.ghost_count() == 2 * depth, "ghost count" + ctx);
    }

    std::vector<double> field(part.total_count(), -999.0);
    for (index_type i = 0; i < part.owned_count(); ++i) {
      field[i] = expected_value(part.global_id(i), 0);
    }
    hx.exchange(comm, std::span<double>(field));

    for (index_type i = part.owned_count(); i < part.total_count(); ++i) {
      check(field[i] == expected_value(part.global_id(i), 0),
            "deep-halo ghost gid=" + std::to_string(part.global_id(i)) +
                " got=" + std::to_string(field[i]) + ctx);
    }
  }
}

// A second exchange must be idempotent and must not accumulate state in the
// reused scratch buffers.
void run_repeat_case(const pp::communicator& comm) {
  const auto r = build_ring(comm.rank(), comm.size(), 4);
  const auto& part = r.partition;
  const halo_exchange hx(part, comm.rank());

  std::vector<double> field(part.total_count(), 0.0);
  for (index_type i = 0; i < part.owned_count(); ++i) {
    field[i] = expected_value(part.global_id(i), 0);
  }

  for (int iter = 0; iter < 5; ++iter) {
    hx.exchange(comm, std::span<double>(field));
  }

  for (index_type i = part.owned_count(); i < part.total_count(); ++i) {
    check(field[i] == expected_value(part.global_id(i), 0),
          "repeated exchange diverged at gid " + std::to_string(part.global_id(i)));
  }
}

// Split begin/end must give the same answer as the fused call.
void run_overlap_case(const pp::communicator& comm) {
  const auto r = build_ring(comm.rank(), comm.size(), 4);
  const auto& part = r.partition;
  const halo_exchange hx(part, comm.rank());

  std::vector<double> fused(part.total_count(), -1.0);
  std::vector<double> split(part.total_count(), -1.0);
  for (index_type i = 0; i < part.owned_count(); ++i) {
    fused[i] = split[i] = expected_value(part.global_id(i), 0);
  }

  hx.exchange(comm, std::span<double>(fused));

  hx.exchange_begin(comm, std::span<double>(split));
  // Interior work would happen here while messages are in flight.
  double interior = 0.0;
  for (index_type i = 0; i < part.owned_count(); ++i) {
    interior += split[i];
  }
  hx.exchange_end(std::span<double>(split));

  check(interior != -1.0, "interior compute ran");
  for (std::size_t i = 0; i < fused.size(); ++i) {
    check(fused[i] == split[i], "overlap path differs from fused at " + std::to_string(i));
  }
}

// Integer fields must survive unchanged; this also exercises a second
// element type through the shared byte buffer.
void run_integer_case(const pp::communicator& comm) {
  const auto r = build_ring(comm.rank(), comm.size(), 3);
  const auto& part = r.partition;
  const halo_exchange hx(part, comm.rank());

  std::vector<int> field(part.total_count(), -7);
  for (index_type i = 0; i < part.owned_count(); ++i) {
    field[i] = static_cast<int>(part.global_id(i)) * 3;
  }

  hx.exchange(comm, std::span<int>(field));

  for (index_type i = part.owned_count(); i < part.total_count(); ++i) {
    check(field[i] == static_cast<int>(part.global_id(i)) * 3,
          "integer ghost mismatch at gid " + std::to_string(part.global_id(i)));
  }
}

void run_schedule_consistency(const pp::communicator& comm) {
  const auto r = build_ring(comm.rank(), comm.size(), 4);
  const halo_exchange hx(r.partition, comm.rank());

  // Total bytes sent must equal total bytes received across all ranks,
  // otherwise some rank is sending into a void.
  const auto sent = static_cast<long long>(hx.total_send_count());
  const auto recvd = static_cast<long long>(hx.total_recv_count());
  const auto global_sent = comm.allreduce_sum(sent);
  const auto global_recv = comm.allreduce_sum(recvd);
  check(global_sent == global_recv,
        "global send count " + std::to_string(global_sent) + " != recv count " +
            std::to_string(global_recv));

  if (comm.size() == 1) {
    check(hx.neighbor_count() == 0, "serial run must have no neighbours");
  } else {
    check(hx.neighbor_count() > 0, "parallel run must have neighbours");
  }
}

void run_mismatched_field_rejected(const pp::communicator& comm) {
  const auto r = build_ring(comm.rank(), comm.size(), 3);
  const halo_exchange hx(r.partition, comm.rank());

  bool threw = false;
  std::vector<double> wrong(r.partition.total_count() + 5, 0.0);
  try {
    hx.exchange(comm, std::span<double>(wrong));
  } catch (const std::exception&) {
    threw = true;
  }
  // Only meaningful when there is something to exchange.
  if (comm.size() > 1) {
    check(threw, "a wrongly sized field must be rejected");
  }
}

}  // namespace

int main(int argc, char** argv) {
  moabx::pp::runtime_scope scope(&argc, &argv);
  const moabx::pp::communicator comm;
  g_rank = comm.rank();

  run_case(comm, 4, 1);
  run_case(comm, 4, 3);
  run_case(comm, 1, 1);
  run_case(comm, 16, 2);
  run_deep_halo_case(comm);
  run_repeat_case(comm);
  run_overlap_case(comm);
  run_integer_case(comm);
  run_schedule_consistency(comm);
  run_mismatched_field_rejected(comm);

  // Any rank failing fails the run.
  const auto total = comm.allreduce_sum(static_cast<long long>(g_failures));
  if (comm.rank() == 0) {
    std::printf("[halo-exchange] ranks=%d failures=%lld\n", comm.size(), total);
  }
  return total == 0 ? 0 : 1;
}
