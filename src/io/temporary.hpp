// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_IO_TEMPORARY_HPP_
#define SOMA_SRC_IO_TEMPORARY_HPP_

// The temporary file that the writers fill and then rename over their target,
// private to the library. Failures throw std::runtime_error with the text
// "<who>: cannot write <path>: <cause>".

#include <cerrno>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace soma::io {

inline std::string cannot_write(const std::string& who, const std::filesystem::path& path, const std::string& why) {
  return who + ": cannot write " + path.string() + ": " + why;
}

// The text of an errno value; unlike strerror, safe to call from any thread.
inline std::string error_text(int e) { return std::system_category().message(e); }

// Names `path` + "." + six random letters or digits.
inline std::function<std::string(int)> random_names(const std::filesystem::path& path) {
  return [base = path.string() + ".", random = std::make_shared<std::random_device>()](int) {
    static constexpr char letters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    std::string name = base;
    for (int k = 0; k < 6; ++k) name += letters[(*random)() % (sizeof letters - 1)];
    return name;
  };
}

// A new file next to `path`, created exclusively (O_EXCL, O_NOFOLLOW) under
// the first of `attempts` names from `candidate` that is free, so that it
// never opens an existing file or follows a symlink. Unlike mkstemp, which
// forces mode 0600, it is created with 0666 so that the umask decides the mode
// of the result. Unless commit() moved it over `path`, it is removed when the
// Temporary goes.
class Temporary {
 public:
  Temporary(std::string who, const std::filesystem::path& path, const std::function<std::string(int)>& candidate,
            int attempts)
      : who_(std::move(who)), path_(path) {
    for (int attempt = 0; attempt < attempts; ++attempt) {
      std::string name = candidate(attempt);
      fd_ = ::open(name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0666);
      if (fd_ >= 0) {
        name_ = std::move(name);
        return;
      }
      const int e = errno;
      if (e != EEXIST && e != EINTR) {
        throw std::runtime_error(cannot_write(who_, path, "cannot create a temporary file: " + error_text(e)));
      }
    }
    throw std::runtime_error(cannot_write(who_, path, "no unused temporary file name was found"));
  }
  Temporary(const Temporary&) = delete;
  Temporary& operator=(const Temporary&) = delete;
  ~Temporary() {
    if (fd_ >= 0) ::close(fd_);
    if (!name_.empty()) ::unlink(name_.c_str());
  }

  int fd() const noexcept { return fd_; }
  const std::string& name() const noexcept { return name_; }

  void close() {
    const int fd = fd_;
    fd_ = -1;
    if (::close(fd) != 0) {
      const int e = errno;
      throw std::runtime_error(cannot_write(who_, path_, "closing the file failed: " + error_text(e)));
    }
  }

  // Moves the file over `path`; afterwards there is nothing left to remove.
  void commit() {
    if (::rename(name_.c_str(), path_.c_str()) != 0) {
      const int e = errno;
      throw std::runtime_error(cannot_write(who_, path_, error_text(e)));
    }
    name_.clear();
  }

 private:
  std::string who_;
  const std::filesystem::path& path_;
  std::string name_;
  int fd_ = -1;
};

}  // namespace soma::io

#endif  // SOMA_SRC_IO_TEMPORARY_HPP_
