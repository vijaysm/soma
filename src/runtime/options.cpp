// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/runtime/options.hpp"

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace soma {

namespace {

std::atomic<double> threshold{default_compress_threshold};

double parse_threshold(const char* text, const char* source) {
  const std::string what = std::string(source) + ": '" + text + "' is not a fraction in [0, 1]";
  char* end = nullptr;
  errno = 0;
  const double v = std::strtod(text, &end);
  if (end == text || *end != '\0' || errno == ERANGE || !(v >= 0.0 && v <= 1.0)) throw std::invalid_argument(what);
  return v;
}

}  // namespace

double compress_threshold() noexcept { return threshold.load(std::memory_order_relaxed); }

void set_compress_threshold(double t) {
  if (!(t >= 0.0 && t <= 1.0)) throw std::invalid_argument("set_compress_threshold: " + std::to_string(t) + " is not in [0, 1]");
  threshold.store(t, std::memory_order_relaxed);
}

void parse_options(int& argc, char**& argv) {
  static constexpr char flag[] = "--soma-compress-threshold=";
  const auto ours = [&](int i) { return std::strncmp(argv[i], flag, sizeof flag - 1) == 0; };
  double t = compress_threshold();
  if (const char* env = std::getenv("SOMA_COMPRESS_THRESHOLD"); env != nullptr) t = parse_threshold(env, "SOMA_COMPRESS_THRESHOLD");
  for (int i = 1; i < argc; ++i) {
    if (ours(i)) t = parse_threshold(argv[i] + sizeof flag - 1, "--soma-compress-threshold");
  }
  // Every value read: only now change argv and the threshold.
  int kept = argc > 0 ? 1 : 0;
  for (int i = kept; i < argc; ++i) {
    if (!ours(i)) argv[kept++] = argv[i];
  }
  for (int i = kept; i < argc; ++i) argv[i] = nullptr;
  argc = kept;
  threshold.store(t, std::memory_order_relaxed);
}

}  // namespace soma
