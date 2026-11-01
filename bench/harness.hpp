// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_BENCH_HARNESS_HPP_
#define SOMA_BENCH_HARNESS_HPP_

// Benchmark harness: warmup, per-repeat statistics, environment disclosure,
// and reduction over ranks. Ported from moabX (bench/harness.hpp).
//
// Reporting rules:
//   * One untimed warmup; the first touch pays cold-cache and page-fault costs.
//   * Every repeat is kept; the median and the MAD are reported, not a total.
//   * Absolute times are printed beside every ratio.
//   * Thread count, backend, build type and compiler are printed with the
//     results, since a speedup means nothing without them.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/parallel/comm.hpp"

namespace soma::bench {

using clock_type = std::chrono::steady_clock;

struct SampleStats {
  double median_sec = 0.0;
  double mad_sec = 0.0;  // median absolute deviation
  double min_sec = 0.0;
  double max_sec = 0.0;
  double mean_sec = 0.0;
  double total_sec = 0.0;
  int repeats = 0;
};

inline double median_of(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t n = values.size();
  return n % 2 == 1 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

inline SampleStats summarize(const std::vector<double>& samples) {
  SampleStats s;
  if (samples.empty()) return s;
  s.repeats = static_cast<int>(samples.size());
  s.median_sec = median_of(samples);
  std::vector<double> dev;
  dev.reserve(samples.size());
  for (const double v : samples) dev.push_back(std::abs(v - s.median_sec));
  s.mad_sec = median_of(std::move(dev));
  const auto [lo, hi] = std::minmax_element(samples.begin(), samples.end());
  s.min_sec = *lo;
  s.max_sec = *hi;
  for (const double v : samples) s.total_sec += v;
  s.mean_sec = s.total_sec / static_cast<double>(samples.size());
  return s;
}

// Runs setup() then fn() once as a discarded warmup, then repeats times with
// only fn() timed. setup() builds the untimed state fn() consumes, such as a
// freshly loaded MOAB instance.
template <class Setup, class Fn>
SampleStats time_repeats(int repeats, Setup&& setup, Fn&& fn) {
  repeats = std::max(repeats, 1);
  setup();
  fn();
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repeats));
  for (int r = 0; r < repeats; ++r) {
    setup();
    const auto t0 = clock_type::now();
    fn();
    samples.push_back(std::chrono::duration<double>(clock_type::now() - t0).count());
  }
  return summarize(samples);
}

// The slowest rank defines the wall time of a bulk synchronous phase, so it
// drives ns_per_op rather than a mean over ranks, which hides imbalance.
struct Timing {
  std::string benchmark;
  std::string workflow;
  SampleStats local{};
  double min_rank_sec = 0.0;
  double max_rank_sec = 0.0;
  double ns_per_op = 0.0;
  std::int64_t ops_total = 0;
};

inline Timing reduce_timing(const Comm& comm, std::string benchmark, std::string workflow,
                            const SampleStats& local, std::int64_t ops_per_repeat) {
  Timing t;
  t.benchmark = std::move(benchmark);
  t.workflow = std::move(workflow);
  t.local = local;
  t.min_rank_sec = comm.allreduce(local.median_sec, Op::min);
  t.max_rank_sec = comm.allreduce(local.median_sec, Op::max);
  t.ops_total = comm.allreduce(ops_per_repeat, Op::sum);
  t.ns_per_op = t.ops_total > 0 ? t.max_rank_sec * 1.0e9 / static_cast<double>(t.ops_total) : 0.0;
  return t;
}

struct Environment {
  std::string build_type;
  std::string compiler;
  std::string backend;
  std::string mesh;
  int ranks = 1;
  int threads = 1;
  int repeats = 0;
};

inline Environment make_environment(const Comm& comm, const std::string& mesh, int repeats) {
  Environment env;
#ifdef SOMA_BENCH_BUILD_TYPE
  env.build_type = SOMA_BENCH_BUILD_TYPE;
#else
  env.build_type = "unknown";
#endif
#ifdef SOMA_BENCH_COMPILER
  env.compiler = SOMA_BENCH_COMPILER;
#else
  env.compiler = "unknown";
#endif
  env.backend = Kokkos::DefaultHostExecutionSpace::name();
  env.mesh = mesh;
  env.ranks = comm.size();
  env.threads = Kokkos::DefaultHostExecutionSpace().concurrency();
  env.repeats = repeats;
  return env;
}

inline void print_environment(const Comm& comm, const Environment& env) {
  if (comm.rank() != 0) return;
  std::cout << "[env] mesh=" << env.mesh << " build_type=" << env.build_type
            << " compiler=" << env.compiler << " backend=" << env.backend
            << " ranks=" << env.ranks << " threads=" << env.threads << " repeats=" << env.repeats
            << " (plus 1 discarded warmup)\n";
  if (env.threads > 1) {
    std::cout << "[env] NOTE: SOMA runs on " << env.threads
              << " threads and MOAB is serial; threads=1 is the data-structure comparison.\n";
  }
}

inline void print_timing(const Comm& comm, const Timing& t) {
  if (comm.rank() != 0) return;
  std::cout << std::fixed << std::setprecision(6) << "[bench] " << t.benchmark
            << " workflow=" << t.workflow << " median_sec=" << t.max_rank_sec
            << " mad_sec=" << t.local.mad_sec << " min_rank_sec=" << t.min_rank_sec
            << std::setprecision(2) << " ns_per_op=" << t.ns_per_op << " ops=" << t.ops_total
            << " repeats=" << t.local.repeats << "\n";
}

inline double relative_spread(const Timing& t) {
  return t.local.median_sec > 0.0 ? t.local.mad_sec / t.local.median_sec : 0.0;
}

// A ratio within the measurement spread is not a result, so it is flagged.
inline void print_comparison(const Comm& comm, const Timing& moab, const Timing& soma) {
  if (comm.rank() != 0) return;
  const double speedup = soma.max_rank_sec > 0.0 ? moab.max_rank_sec / soma.max_rank_sec : 0.0;
  const double noise = std::max(relative_spread(moab), relative_spread(soma));
  std::cout << std::fixed << std::setprecision(3) << "[bench] " << moab.benchmark
            << " speedup(moab/soma)=" << speedup << " noise=" << noise;
  if (noise > 0.10) std::cout << " WARNING:high-variance";
  std::cout << "\n";
}

// One JSON object per line: appendable and greppable.
inline void print_json(const Comm& comm, const Environment& env, const Timing& t) {
  if (comm.rank() != 0) return;
  std::cout << std::fixed << std::setprecision(9) << "{\"benchmark\":\"" << t.benchmark
            << "\",\"workflow\":\"" << t.workflow << "\",\"mesh\":\"" << env.mesh
            << "\",\"build_type\":\"" << env.build_type << "\",\"compiler\":\"" << env.compiler
            << "\",\"backend\":\"" << env.backend << "\",\"ranks\":" << env.ranks
            << ",\"threads\":" << env.threads << ",\"repeats\":" << t.local.repeats
            << ",\"median_sec\":" << t.max_rank_sec << ",\"mad_sec\":" << t.local.mad_sec
            << ",\"min_rank_sec\":" << t.min_rank_sec << ",\"ns_per_op\":" << t.ns_per_op
            << ",\"ops\":" << t.ops_total << "}\n";
}

class Runner {
 public:
  Runner(const Comm& comm, Environment env, bool json)
      : comm_(comm), env_(std::move(env)), json_(json) {}

  template <class Setup, class Fn>
  Timing run(const std::string& benchmark, const std::string& workflow,
             std::int64_t ops_per_repeat, Setup&& setup, Fn&& fn) {
    const SampleStats local =
        time_repeats(env_.repeats, std::forward<Setup>(setup), std::forward<Fn>(fn));
    const Timing t = reduce_timing(comm_, benchmark, workflow, local, ops_per_repeat);
    if (json_) {
      print_json(comm_, env_, t);
    } else {
      print_timing(comm_, t);
    }
    return t;
  }

  void compare(const Timing& moab, const Timing& soma) const {
    if (!json_) print_comparison(comm_, moab, soma);
  }

 private:
  const Comm& comm_;
  Environment env_;
  bool json_ = false;
};

}  // namespace soma::bench

#endif  // SOMA_BENCH_HARNESS_HPP_
