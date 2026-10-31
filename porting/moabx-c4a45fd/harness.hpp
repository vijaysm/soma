#ifndef MOABX_BENCH_HARNESS_HPP_
#define MOABX_BENCH_HARNESS_HPP_

// Benchmark harness: warmup, per-repeat statistics, environment disclosure,
// and MPI reduction.
//
// Replaces a pattern that reported only the summed wall time of N repeats with
// no warmup, labelled `avg_sec` (actually a cross-rank mean of a per-rank
// total). Under that scheme the published figures compared an unoptimized
// MOABX against an -O3 MOAB, and a multi-threaded MOABX against a serial MOAB,
// with neither disclosed.
//
// Reporting rules enforced here:
//   * Discard a warmup iteration; the first touch pays cold-cache and
//     first-touch page-fault costs.
//   * Keep every repeat and report median and MAD, not just a total. Median
//     resists the occasional scheduler outlier; MAD shows whether a reported
//     ratio is meaningful at all.
//   * Report absolute times alongside ratios.
//   * Disclose thread count, build type, and compiler with the results, since
//     a speedup is uninterpretable without them.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "moabx/pp/runtime.hpp"

#if PP_USE_OPENMP
#include <omp.h>
#endif

namespace moabx::bench {

using clock_type = std::chrono::steady_clock;

struct sample_stats {
  double median_sec = 0.0;
  double mad_sec = 0.0;  // median absolute deviation
  double min_sec = 0.0;
  double max_sec = 0.0;
  double mean_sec = 0.0;
  double total_sec = 0.0;
  int repeats = 0;
};

inline double median_of(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t n = values.size();
  if (n % 2 == 1) {
    return values[n / 2];
  }
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

inline sample_stats summarize(const std::vector<double>& samples) {
  sample_stats stats{};
  if (samples.empty()) {
    return stats;
  }
  stats.repeats = static_cast<int>(samples.size());
  stats.median_sec = median_of(samples);

  std::vector<double> deviations;
  deviations.reserve(samples.size());
  for (const double value : samples) {
    deviations.push_back(std::abs(value - stats.median_sec));
  }
  stats.mad_sec = median_of(std::move(deviations));

  const auto [lo, hi] = std::minmax_element(samples.begin(), samples.end());
  stats.min_sec = *lo;
  stats.max_sec = *hi;
  for (const double value : samples) {
    stats.total_sec += value;
  }
  stats.mean_sec = stats.total_sec / static_cast<double>(samples.size());
  return stats;
}

// Runs `fn` once as an untimed warmup, then `repeats` timed iterations.
template <class Fn>
sample_stats time_repeats(int repeats, Fn&& fn) {
  if (repeats < 1) {
    repeats = 1;
  }
  fn();  // warmup, discarded

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repeats));
  for (int r = 0; r < repeats; ++r) {
    const auto t0 = clock_type::now();
    fn();
    samples.push_back(std::chrono::duration<double>(clock_type::now() - t0).count());
  }
  return summarize(samples);
}

// Cross-rank view of one workflow's timing.
struct workflow_timing {
  std::string benchmark;
  std::string workflow;
  sample_stats local{};
  double median_sec = 0.0;      // cross-rank max of per-rank medians
  double min_rank_sec = 0.0;    // fastest rank
  double max_rank_sec = 0.0;    // slowest rank (the critical path)
  double ns_per_op = 0.0;       // from slowest rank, per single op
  std::int64_t ops_per_repeat = 0;
  std::int64_t ops_total = 0;
};

// Reduces across ranks. The slowest rank defines wall time for a bulk
// synchronous phase, so `max_rank_sec` drives the reported ns/op rather than a
// cross-rank mean, which would understate cost under load imbalance.
inline workflow_timing reduce_timing(
    const pp::communicator& comm,
    std::string benchmark,
    std::string workflow,
    const sample_stats& local,
    std::int64_t ops_per_repeat) {
  workflow_timing out{};
  out.benchmark = std::move(benchmark);
  out.workflow = std::move(workflow);
  out.local = local;
  out.ops_per_repeat = ops_per_repeat;

  out.min_rank_sec = comm.allreduce_min(local.median_sec);
  out.max_rank_sec = comm.allreduce_max(local.median_sec);
  out.median_sec = out.max_rank_sec;
  out.ops_total = comm.allreduce_sum(ops_per_repeat);

  out.ns_per_op =
      (out.ops_total > 0) ? (out.max_rank_sec * 1.0e9 / static_cast<double>(out.ops_total)) : 0.0;
  return out;
}

struct environment_info {
  std::string build_type;
  std::string compiler;
  std::string mesh;
  int mpi_ranks = 1;
  int omp_threads = 1;
  int repeats = 0;
};

inline int active_thread_count() {
#if PP_USE_OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

inline environment_info make_environment(
    const pp::communicator& comm,
    const std::string& mesh,
    int repeats) {
  environment_info env{};
#ifdef MOABX_BUILD_TYPE
  env.build_type = MOABX_BUILD_TYPE;
#else
  env.build_type = "unknown";
#endif
#ifdef MOABX_COMPILER_ID
  env.compiler = MOABX_COMPILER_ID;
#else
  env.compiler = "unknown";
#endif
  env.mesh = mesh;
  env.mpi_ranks = comm.size();
  env.omp_threads = active_thread_count();
  env.repeats = repeats;
  return env;
}

// A speedup is not interpretable without these, so print them with the results
// rather than expecting the reader to reconstruct them.
inline void print_environment(const pp::communicator& comm, const environment_info& env) {
  if (comm.rank() != 0) {
    return;
  }
  std::cout << "[env] mesh=" << env.mesh
            << " build_type=" << env.build_type
            << " compiler=" << env.compiler
            << " mpi_ranks=" << env.mpi_ranks
            << " omp_threads=" << env.omp_threads
            << " repeats=" << env.repeats
            << " (plus 1 discarded warmup)\n";
  if (env.omp_threads > 1) {
    std::cout << "[env] NOTE: MOABX query kernels are threaded and native MOAB loops are "
                 "serial; run with OMP_NUM_THREADS=1 for a data-structure comparison.\n";
  }
}

inline void print_timing(const pp::communicator& comm, const workflow_timing& t) {
  if (comm.rank() != 0) {
    return;
  }
  std::cout << std::fixed << std::setprecision(6)
            << "[bench] " << t.benchmark
            << " workflow=" << t.workflow
            << " median_sec=" << t.max_rank_sec
            << " mad_sec=" << t.local.mad_sec
            << " min_rank_sec=" << t.min_rank_sec
            << std::setprecision(2)
            << " ns_per_op=" << t.ns_per_op
            << " ops=" << t.ops_total
            << " repeats=" << t.local.repeats
            << "\n";
}

// Relative spread of the slower side, used to flag ratios that noise could
// plausibly explain.
inline double relative_spread(const workflow_timing& t) {
  return (t.local.median_sec > 0.0) ? (t.local.mad_sec / t.local.median_sec) : 0.0;
}

inline void print_comparison(
    const pp::communicator& comm,
    const workflow_timing& native,
    const workflow_timing& moabx) {
  if (comm.rank() != 0) {
    return;
  }
  const double speedup =
      (moabx.max_rank_sec > 0.0) ? (native.max_rank_sec / moabx.max_rank_sec) : 0.0;
  const double noise = std::max(relative_spread(native), relative_spread(moabx));

  std::cout << std::fixed << std::setprecision(3)
            << "[bench] " << native.benchmark
            << " speedup(native/moabx)=" << speedup
            << " noise=" << noise;
  // A ratio within the measurement spread is not a result.
  if (noise > 0.10) {
    std::cout << " WARNING:high-variance";
  }
  std::cout << "\n";
}

// JSON line per workflow, for regression tracking. One object per line keeps
// the output appendable and greppable without a JSON parser in the loop.
inline void print_json(
    const pp::communicator& comm,
    const environment_info& env,
    const workflow_timing& t) {
  if (comm.rank() != 0) {
    return;
  }
  std::cout << std::fixed << std::setprecision(9)
            << "{\"benchmark\":\"" << t.benchmark
            << "\",\"workflow\":\"" << t.workflow
            << "\",\"mesh\":\"" << env.mesh
            << "\",\"build_type\":\"" << env.build_type
            << "\",\"compiler\":\"" << env.compiler
            << "\",\"mpi_ranks\":" << env.mpi_ranks
            << ",\"omp_threads\":" << env.omp_threads
            << ",\"repeats\":" << t.local.repeats
            << ",\"median_sec\":" << t.max_rank_sec
            << ",\"mad_sec\":" << t.local.mad_sec
            << ",\"min_rank_sec\":" << t.min_rank_sec
            << ",\"ns_per_op\":" << t.ns_per_op
            << ",\"ops\":" << t.ops_total
            << "}\n";
}

// Drives one native-vs-MOABX comparison end to end.
class runner {
 public:
  runner(const pp::communicator& comm, environment_info env, bool emit_json)
      : comm_(comm), env_(std::move(env)), emit_json_(emit_json) {}

  const environment_info& environment() const noexcept { return env_; }

  template <class Fn>
  workflow_timing run(
      const std::string& benchmark,
      const std::string& workflow,
      std::int64_t ops_per_repeat,
      Fn&& fn) {
    const sample_stats local = time_repeats(env_.repeats, std::forward<Fn>(fn));
    const workflow_timing timing = reduce_timing(comm_, benchmark, workflow, local, ops_per_repeat);
    if (emit_json_) {
      print_json(comm_, env_, timing);
    } else {
      print_timing(comm_, timing);
    }
    return timing;
  }

  template <class NativeFn, class MoabxFn>
  void run_pair(
      const std::string& benchmark,
      std::int64_t native_ops,
      NativeFn&& native_fn,
      std::int64_t moabx_ops,
      MoabxFn&& moabx_fn) {
    const auto native = run(benchmark, "native_moab", native_ops, std::forward<NativeFn>(native_fn));
    const auto moabx = run(benchmark, "moabx", moabx_ops, std::forward<MoabxFn>(moabx_fn));
    if (!emit_json_) {
      print_comparison(comm_, native, moabx);
    }
  }

 private:
  const pp::communicator& comm_;
  environment_info env_;
  bool emit_json_ = false;
};

}  // namespace moabx::bench

#endif  // MOABX_BENCH_HARNESS_HPP_
