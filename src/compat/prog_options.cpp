// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/compat/prog_options.hpp"

#include <cstdlib>
#include <iostream>

namespace soma::compat {

void ProgOptions::error(const std::string& program, const std::string& message) const {
  std::cerr << "Error: " << message << "\n";
  printUsage(std::cerr);
  (void)program;
  std::exit(1);
}

void ProgOptions::printUsage(std::ostream& out) const {
  out << "Usage: [options]";
  for (const Opt& a : args_) out << " <" << a.long_name << ">";
  out << "\n";
}

void ProgOptions::printHelp(std::ostream& out) const {
  if (!brief_.empty()) out << brief_ << "\n";
  if (!help_.empty()) out << help_ << "\n";
  printUsage(out);
  out << "Options:\n  -h [--help]  : Show full help text\n";
  for (const Opt& o : opts_) {
    out << "  ";
    if (!o.short_name.empty()) out << "-" << o.short_name << " ";
    if (!o.long_name.empty()) out << "[--" << o.long_name << "]";
    out << (o.flag ? "" : " <val>") << "  : " << o.help << "\n";
  }
}

void ProgOptions::parseCommandLine(int argc, char* argv[]) {
  const std::string program = argc > 0 ? argv[0] : "";
  std::size_t next_arg = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      printHelp(std::cout);
      std::exit(0);
    }
    Opt* o = nullptr;
    std::string value;
    bool inline_value = false;
    if (a.rfind("--", 0) == 0 && a.size() > 2) {
      std::string name = a.substr(2);
      const auto eq = name.find('=');
      if (eq != std::string::npos) {
        value = name.substr(eq + 1);
        name = name.substr(0, eq);
        inline_value = true;
      }
      for (Opt& x : opts_) {
        if (!x.long_name.empty() && x.long_name == name) o = &x;
      }
      if (o == nullptr) error(program, "Invalid option: " + a);
    } else if (a.size() > 1 && a[0] == '-' && !(a[1] >= '0' && a[1] <= '9')) {
      for (Opt& x : opts_) {
        if (!x.short_name.empty() && a.compare(1, x.short_name.size(), x.short_name) == 0 &&
            (a.size() == 1 + x.short_name.size() || !x.flag)) {
          o = &x;
        }
      }
      if (o == nullptr) error(program, "Invalid option: " + a);
      if (a.size() > 1 + o->short_name.size()) {
        value = a.substr(1 + o->short_name.size());
        inline_value = true;
      }
    } else {
      if (next_arg >= args_.size()) error(program, "Unexpected argument: " + a);
      Opt& r = args_[next_arg++];
      if (!r.set(a)) error(program, "Invalid value for " + r.long_name + ": " + a);
      r.given = true;
      continue;
    }
    if (!o->flag && !inline_value) {
      if (i + 1 >= argc) error(program, "Missing value for option: " + a);
      value = argv[++i];
    }
    if (!o->flag && !o->set(value)) error(program, "Invalid value for option " + a + ": " + value);
    o->text = value;
    o->given = true;
    ++o->count;
  }
  if (next_arg < args_.size()) error(program, "Missing required argument: " + args_[next_arg].long_name);
}

}  // namespace soma::compat
