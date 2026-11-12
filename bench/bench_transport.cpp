// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// transport::route against MOAB's crystal router (gs_data::crystal_data::
// gs_transfer on a TupleList). Every rank sends n records of three ints:
//
//   * dense: to destinations drawn uniformly from all ranks;
//   * sparse: to four neighbours, rank +- 1 and +- 2 (a halo's pattern).
//
// SOMA routes them with each strategy. Every result is checked against
// MOAB's by count and checksum on every rank; a mismatch exits with status 2.
//
//   mpiexec -n P bench_transport [--json] [--repeats N] [--records n]

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "harness.hpp"
#include "moab/ProcConfig.hpp"
#include "moab/TupleList.hpp"
#include "moab/gs.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/transport/route.hpp"
#include "soma/runtime/scope_guard.hpp"

namespace {

namespace tr = soma::transport;
using Host = Kokkos::DefaultHostExecutionSpace;
using Batch = tr::RecordBatch<Host::memory_space>;

struct Sum {
  std::int64_t count = 0;
  std::int64_t checksum = 0;
  void add(int source, const std::int32_t* v) {
    ++count;
    checksum += (static_cast<std::int64_t>(source) * 1000003 + v[0]) * 31 + v[1] * 7 + v[2];
  }
};

std::vector<int> destinations(const std::string& pattern, int rank, int size, std::size_t n) {
  std::mt19937 gen(static_cast<std::mt19937::result_type>(rank + 1));
  std::uniform_int_distribution<int> any(0, size - 1), near(0, 3);
  const int offsets[4] = {-2, -1, 1, 2};
  std::vector<int> d(n);
  for (int& x : d) x = pattern == "dense" ? any(gen) : ((rank + offsets[near(gen)]) % size + size) % size;
  return d;
}

bool run_pattern(const soma::Comm& comm, soma::bench::Runner& runner, const std::string& pattern, std::size_t n,
                 bool json) {
  const std::vector<int> dest = destinations(pattern, comm.rank(), comm.size(), n);
  const auto ops = static_cast<std::int64_t>(n);
  const std::string host_name = Kokkos::DefaultHostExecutionSpace::name();

  moab::ProcConfig pc(comm.native());
  moab::TupleList tl;
  Sum moab_sum;
  const auto tm = runner.run(
      "route_" + pattern, "moab_crystal", "moab", ops,
      [&] {
        tl.reset();
        tl.initialize(4, 0, 0, 0, static_cast<unsigned>(n));
        tl.enableWriteAccess();
        for (std::size_t i = 0; i < n; ++i) {
          const unsigned k = tl.get_n();
          tl.vi_wr[4 * k] = dest[i];
          tl.vi_wr[4 * k + 1] = comm.rank();
          tl.vi_wr[4 * k + 2] = static_cast<int>(i);
          tl.vi_wr[4 * k + 3] = dest[i];
          tl.inc_n();
        }
      },
      [&] { pc.crystal_router()->gs_transfer(1, tl, 0); });
  // After the transfer the first int names the source.
  for (unsigned k = 0; k < tl.get_n(); ++k) moab_sum.add(tl.vi_rd[4 * k], &tl.vi_rd[4 * k + 1]);

  Batch in(n, 3 * sizeof(std::int32_t));
  for (std::size_t i = 0; i < n; ++i) {
    in.dest(i) = dest[i];
    const std::int32_t v[3] = {comm.rank(), static_cast<std::int32_t>(i), dest[i]};
    std::memcpy(&in.payload(i, 0), v, sizeof v);
  }
  bool ok = true;
  for (const auto& [name, strategy] : {std::pair{"crystal", tr::Strategy::crystal}, std::pair{"alltoallv", tr::Strategy::alltoallv},
                                       std::pair{"nbx", tr::Strategy::nbx}}) {
    Batch out;
    const auto ts = runner.run("route_" + pattern, std::string("soma_") + name, host_name, ops, [] {},
                               [&] { out = tr::route(Host{}, comm, in, strategy); });
    runner.compare(tm, ts, "speedup(moab/soma)");
    Sum s;
    for (std::size_t i = 0; i < out.size(); ++i) {
      std::int32_t v[3];
      std::memcpy(v, &out.payload(i, 0), sizeof v);
      s.add(out.source(i), v);
    }
    const bool same = s.count == moab_sum.count && s.checksum == moab_sum.checksum;
    ok = comm.allreduce(static_cast<std::int32_t>(same), soma::Op::min) == 1 && ok;
  }
  if (!ok && comm.rank() == 0) std::cerr << "[bench] route_" << pattern << ": SOMA and MOAB delivered different records\n";
  (void)json;
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  bool json = false;
  int repeats = 5;
  std::size_t records = 100000;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--json") {
      json = true;
    } else if (a == "--repeats" && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
    } else if (a == "--records" && i + 1 < argc) {
      records = static_cast<std::size_t>(std::atoll(argv[++i]));
    }
  }
  const auto env = soma::bench::make_environment(comm, std::to_string(records) + " records per rank", repeats);
  if (!json) {
    soma::bench::print_environment(comm, env);
    soma::bench::print_moab_note(comm, env);
  }
  soma::bench::Runner runner(comm, env, json);
  bool ok = true;
  for (const char* pattern : {"dense", "sparse"}) ok = run_pattern(comm, runner, pattern, records, json) && ok;
  return ok ? 0 : 2;
}
