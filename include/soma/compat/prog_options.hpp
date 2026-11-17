// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_COMPAT_PROG_OPTIONS_HPP_
#define SOMA_COMPAT_PROG_OPTIONS_HPP_

/// \file
/// The part of MOAB's ProgOptions that MOAB's examples use: options with a
/// long name, a short one or both ("dim,d", ",n"), of type int, double,
/// std::string or void (a flag), given as "--dim 2", "--dim=2", "-d 2" or
/// "-d2"; -h and --help print the options and exit. Required positional
/// arguments come in the order they were added. A malformed command line
/// prints a message and exits with status 1, as MOAB's does.

#include <functional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace soma::compat {

class ProgOptions {
 public:
  explicit ProgOptions(const std::string& help = "", const std::string& brief = "") : help_(help), brief_(brief) {}

  template <class T>
  void addOpt(const std::string& names, const std::string& help, T* value = nullptr, int /*flags*/ = 0) {
    Opt o = make_opt(names, help);
    if constexpr (std::is_void_v<T>) {
      o.flag = true;
      o.set = [](const std::string&) { return true; };
    } else {
      o.set = [value](const std::string& text) { return parse(text, value); };
    }
    opts_.push_back(std::move(o));
  }
  /// A flag (no value): addOpt<void>(names, help).
  void addOpt(const std::string& names, const std::string& help) { addOpt<void>(names, help, nullptr); }

  template <class T>
  void addRequiredArg(const std::string& name, const std::string& help, T* value = nullptr, int /*flags*/ = 0) {
    Opt o;
    o.long_name = name;
    o.help = help;
    o.set = [value](const std::string& text) { return parse(text, value); };
    args_.push_back(std::move(o));
  }

  /// Whether the option was given, its value (for valued options) in *value.
  template <class T>
  bool getOpt(const std::string& name, T* value = nullptr) {
    for (const Opt& o : opts_) {
      if (o.long_name == name || o.short_name == name) {
        if constexpr (!std::is_void_v<T>) {
          if (o.given && value != nullptr && !o.flag) parse(o.text, value);
        }
        return o.given;
      }
    }
    return false;
  }
  int numOptSet(const std::string& name) {
    for (const Opt& o : opts_) {
      if (o.long_name == name || o.short_name == name) return o.count;
    }
    return 0;
  }

  void parseCommandLine(int argc, char* argv[]);
  void printHelp(std::ostream& out) const;
  void printUsage(std::ostream& out) const;

 private:
  struct Opt {
    std::string long_name, short_name, help, text;
    bool flag = false;
    bool given = false;
    int count = 0;
    std::function<bool(const std::string&)> set;
  };
  static Opt make_opt(const std::string& names, const std::string& help) {
    Opt o;
    const auto comma = names.find(',');
    o.long_name = names.substr(0, comma);
    if (comma != std::string::npos) o.short_name = names.substr(comma + 1);
    o.help = help;
    return o;
  }
  template <class T>
  static bool parse(const std::string& text, T* value) {
    std::istringstream in(text);
    T v{};
    if constexpr (std::is_same_v<T, std::string>) {
      v = text;
    } else {
      in >> v;
      if (in.fail() || !in.eof()) return false;
    }
    if (value != nullptr) *value = v;
    return true;
  }
  [[noreturn]] void error(const std::string& program, const std::string& message) const;

  std::string help_, brief_;
  std::vector<Opt> opts_;
  std::vector<Opt> args_;
};

}  // namespace soma::compat

#endif  // SOMA_COMPAT_PROG_OPTIONS_HPP_
