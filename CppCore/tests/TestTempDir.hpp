// MIT License
// Copyright 2023--present rgpot developers
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>

namespace rgpot::tests {

class TestTempDir {
public:
  explicit TestTempDir(std::string prefix = "rgpot-test")
      : m_path(create_unique_dir(prefix)) {}

  ~TestTempDir() {
    std::error_code ignored;
    std::filesystem::remove_all(m_path, ignored);
  }

  TestTempDir(const TestTempDir &) = delete;
  TestTempDir &operator=(const TestTempDir &) = delete;
  TestTempDir(TestTempDir &&) = delete;
  TestTempDir &operator=(TestTempDir &&) = delete;

  const std::filesystem::path &path() const noexcept { return m_path; }
  std::string string() const { return m_path.string(); }

private:
  // Creating the directory is the uniqueness check: create_directory reports
  // false when the name is already taken, so two processes can never both
  // believe they own the same directory.
  static std::filesystem::path create_unique_dir(const std::string &prefix) {
    constexpr int max_attempts = 256;
    const auto base = std::filesystem::temp_directory_path();
    std::mt19937_64 rng(std::random_device{}());
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
      const auto stamp =
          std::chrono::steady_clock::now().time_since_epoch().count();
      const auto candidate = base / (prefix + "-" + std::to_string(stamp) +
                                     "-" + std::to_string(rng()));
      std::error_code ec;
      if (std::filesystem::create_directory(candidate, ec)) {
        return candidate;
      }
      if (ec) {
        throw std::runtime_error("failed to create temporary directory " +
                                 candidate.string() + ": " + ec.message());
      }
    }
    throw std::runtime_error(
        "failed to create a unique temporary directory after " +
        std::to_string(max_attempts) + " attempts");
  }

  std::filesystem::path m_path;
};

} // namespace rgpot::tests
