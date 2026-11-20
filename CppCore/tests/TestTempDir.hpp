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
      : m_path(make_unique_path(prefix)) {
    std::filesystem::create_directories(m_path);
  }

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
  static std::filesystem::path make_unique_path(const std::string &prefix) {
    auto base = std::filesystem::temp_directory_path();
    std::mt19937_64 rng(std::random_device{}());
    for (int attempt = 0; attempt < 256; ++attempt) {
      const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
      const auto candidate = base / (prefix + "-" + std::to_string(stamp) + "-" +
                                     std::to_string(rng()));
      std::error_code ec;
      if (!std::filesystem::exists(candidate, ec)) {
        return candidate;
      }
    }
    throw std::runtime_error("failed to create unique temporary directory path");
  }

  std::filesystem::path m_path;
};

} // namespace rgpot::tests
