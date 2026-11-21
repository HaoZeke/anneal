// MIT License
// Copyright 2023--present rgpot developers
#include <atomic>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_all.hpp>

#include "TestTempDir.hpp"

namespace fs = std::filesystem;

TEST_CASE("TestTempDir creates an existing directory and removes it",
          "[TestTempDir]") {
  fs::path where;
  {
    rgpot::tests::TestTempDir dir("rgpot-test-tempdir");
    where = dir.path();
    REQUIRE(fs::is_directory(where));
    REQUIRE(where.filename().string().rfind("rgpot-test-tempdir-", 0) == 0);
  }
  REQUIRE_FALSE(fs::exists(where));
}

TEST_CASE("Concurrent TestTempDir instances get distinct directories",
          "[TestTempDir]") {
  constexpr int n_threads = 16;
  std::vector<std::unique_ptr<rgpot::tests::TestTempDir>> dirs(n_threads);
  std::atomic<bool> go{false};
  std::atomic<int> failures{0};

  std::vector<std::thread> threads;
  for (int i = 0; i < n_threads; ++i) {
    threads.emplace_back([&, i] {
      while (!go.load()) {
        std::this_thread::yield();
      }
      try {
        dirs[i] = std::make_unique<rgpot::tests::TestTempDir>(
            "rgpot-test-tempdir-race");
      } catch (...) {
        ++failures;
      }
    });
  }
  go = true;
  for (auto &t : threads) {
    t.join();
  }

  REQUIRE(failures.load() == 0);
  std::set<std::string> seen;
  for (const auto &d : dirs) {
    REQUIRE(d != nullptr);
    REQUIRE(fs::is_directory(d->path()));
    seen.insert(d->path().string());
  }
  REQUIRE(seen.size() == static_cast<std::size_t>(n_threads));
}
