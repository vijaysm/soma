// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_RUNTIME_OPTIONS_HPP_
#define SOMA_RUNTIME_OPTIONS_HPP_

/// \file
/// Process-wide options, set from the environment and the command line by
/// parse_options (which ScopeGuard calls) or by the setters.
///
/// compress_threshold: MeshEdit::apply compresses the mesh when, in some
/// dimension, at least this fraction of the slots holds removed entities
/// (removed_fraction); 0 never compresses. Edited meshes pay for the extra
/// slots and the mask: on a 80^3 box of hexes (8 threads, Apple M-series), a
/// field write over every live hex costs 1.6x its compact cost with 5 % of the
/// slots removed, 1.8x at 17 % and 2.2x at 23 %, passing 2x near 20 %, which is
/// the default. bench_edit prints the crossover it measures, as a value for
/// another machine.
///
/// ```
///   SOMA_COMPRESS_THRESHOLD=0.3 ./app            # environment
///   ./app --soma-compress-threshold=0.3          # command line, which wins
/// ```
///
/// reproducible: results the same bit for bit on any number of ranks and
/// threads, for one build, where an algorithm can give that at some cost;
/// off (the default), they may differ there in the last bits, faster. On, a
/// FemSpace gathers whatever assembly it is asked for and its dot() and cg
/// sum exactly (soma/parallel/exact_sum.hpp); laplacian_smooth is
/// reproducible either way. Algorithms read it when they start (FemSpace when
/// it is made) and agree on it across ranks; an option of their own
/// (CgOptions::reproducible) overrides it.
///
/// ```
///   SOMA_REPRODUCIBLE=1 ./app                    # 0 or 1
///   ./app --soma-reproducible                    # on; or --soma-reproducible=0|1
///                                                # (a value after a space is not its)
/// ```

namespace soma {

inline constexpr double default_compress_threshold = 0.2;

double compress_threshold() noexcept;
/// Throws std::invalid_argument for a value outside [0, 1] (or NaN).
void set_compress_threshold(double t);

bool reproducible() noexcept;
void set_reproducible(bool on) noexcept;

/// Reads SOMA_COMPRESS_THRESHOLD and SOMA_REPRODUCIBLE, then
/// --soma-compress-threshold=<x> and --soma-reproducible[=0|1] from the
/// command line, removing them from argv (argc shrinks, argv[argc] stays
/// null). Throws std::invalid_argument, naming the option, for a threshold
/// that is not a number in [0, 1] or a switch not 0 or 1; nothing is set then.
void parse_options(int& argc, char**& argv);

}  // namespace soma

#endif  // SOMA_RUNTIME_OPTIONS_HPP_
