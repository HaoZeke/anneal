// MIT License
// UmaPot unpacks its .pt2 under UmaConfig::cache_dir, else RGPOT_CACHE_DIR,
// and evaluates from there. Concurrent pots share one unpack, and nothing is
// written to the system temporary directory. Meson builds the package and
// passes its path in UMA_FLOAT64_PT2.

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/UmaPot/UmaPot.hpp"
#include "rgpot/UmaPot/aoti_unpack.hpp"

namespace fs = std::filesystem;
namespace unpack = rgpot::aoti_unpack;

namespace {

std::string package_path() {
  const char *path = std::getenv("UMA_FLOAT64_PT2");
  if (path == nullptr || path[0] == '\0')
    FAIL("UMA_FLOAT64_PT2 is not set");
  return path;
}

class EnvGuard {
public:
  explicit EnvGuard(const char *name) : m_name(name) {
    if (const char *v = std::getenv(name)) {
      m_had = true;
      m_value = v;
    }
  }
  ~EnvGuard() {
    if (m_had)
      setenv(m_name, m_value.c_str(), 1);
    else
      unsetenv(m_name);
  }
  EnvGuard(const EnvGuard &) = delete;
  EnvGuard &operator=(const EnvGuard &) = delete;

private:
  const char *m_name;
  bool m_had = false;
  std::string m_value;
};

class Scratch {
public:
  Scratch() {
    path = fs::current_path() /
           ("uma-cache-dir-test-" + unpack::detail::unique_token());
    fs::create_directories(path);
  }
  ~Scratch() {
    std::error_code ignored;
    fs::remove_all(path, ignored);
  }
  Scratch(const Scratch &) = delete;
  Scratch &operator=(const Scratch &) = delete;
  fs::path path;
};

std::set<std::string> listing(const fs::path &dir) {
  std::set<std::string> out;
  std::error_code ec;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
       it.increment(ec))
    out.insert(it->path().filename().string());
  return out;
}

std::set<std::string> temp_listing() {
  std::set<std::string> out;
  std::error_code ec;
  const auto sys = fs::temp_directory_path(ec);
  if (!ec)
    for (const auto &name : listing(sys))
      out.insert(sys.string() + "|" + name);
  for (const auto &name : listing("/tmp"))
    out.insert("/tmp|" + name);
  return out;
}

double line_energy(rgpot::UmaPot &pot) {
  std::vector<double> pos = {12.5, 12.5, 12.5, 12.5, 12.5,
                             13.5, 12.5, 12.5, 11.5};
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  std::vector<double> force(pos.size(), 0.0);
  rgpot::ForceOut slot{force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{3, pos.data(), z.data(), box};
  pot.forceImpl(in, &slot);
  return slot.energy;
}

rgpot::UmaConfig line_config(const std::string &cache_dir) {
  rgpot::UmaConfig cfg;
  cfg.model_path = package_path();
  cfg.task_name = "omol";
  cfg.charge = 0;
  cfg.spin = 1;
  cfg.cache_dir = cache_dir;
  return cfg;
}

fs::path expected_unpack(const fs::path &root) {
  return root / "aoti" /
         (unpack::sha256_file(package_path()) + ".model");
}

} // namespace

TEST_CASE("UmaConfig::cache_dir wins over RGPOT_CACHE_DIR", "[uma][cache]") {
  EnvGuard env(unpack::kCacheDirEnv);
  Scratch s;
  const auto option = s.path / "option";
  const auto variable = s.path / "variable";
  setenv(unpack::kCacheDirEnv, variable.string().c_str(), 1);
  const auto before = temp_listing();

  rgpot::UmaPot pot(line_config(option.string()));
  REQUIRE_THAT(line_energy(pot), Catch::Matchers::WithinAbs(112.5, 1e-9));
  const auto dir = expected_unpack(option);
  REQUIRE(fs::is_directory(dir));
  REQUIRE(fs::exists(dir / unpack::kManifest));
  REQUIRE_FALSE(fs::exists(variable));

  const auto after = temp_listing();
  std::vector<std::string> added;
  std::set_difference(after.begin(), after.end(), before.begin(), before.end(),
                      std::back_inserter(added));
  for (const auto &name : added)
    FAIL("new entry in the temporary directory: " << name);
}

TEST_CASE("RGPOT_CACHE_DIR places the unpack when the config has none",
          "[uma][cache]") {
  EnvGuard env(unpack::kCacheDirEnv);
  Scratch s;
  const auto variable = s.path / "variable";
  setenv(unpack::kCacheDirEnv, variable.string().c_str(), 1);
  const auto before = temp_listing();

  {
    rgpot::UmaPot pot(line_config(""));
    REQUIRE_THAT(line_energy(pot), Catch::Matchers::WithinAbs(112.5, 1e-9));
  }
  const auto dir = expected_unpack(variable);
  REQUIRE(fs::is_directory(dir));
  // The unpack outlives the pot, so the next process reuses it.
  const auto when = fs::last_write_time(dir / unpack::kManifest);
  {
    rgpot::UmaPot pot(line_config(""));
    REQUIRE_THAT(line_energy(pot), Catch::Matchers::WithinAbs(112.5, 1e-9));
  }
  // libc++ file_time_type counts in __int128, which Catch2 cannot print.
  const bool untouched = fs::last_write_time(dir / unpack::kManifest) == when;
  REQUIRE(untouched);
  REQUIRE(listing(variable / "aoti") ==
          std::set<std::string>{dir.filename().string()});

  const auto after = temp_listing();
  std::vector<std::string> added;
  std::set_difference(after.begin(), after.end(), before.begin(), before.end(),
                      std::back_inserter(added));
  for (const auto &name : added)
    FAIL("new entry in the temporary directory: " << name);
}

TEST_CASE("concurrent pots share one unpack", "[uma][cache]") {
  Scratch s;
  const auto root = s.path / "shared";
  constexpr int threads = 6;
  std::vector<double> energies(threads, 0.0);
  std::vector<std::string> errors(threads);
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t)
    pool.emplace_back([&, t] {
      try {
        rgpot::UmaPot pot(line_config(root.string()));
        energies[t] = line_energy(pot);
      } catch (const std::exception &e) {
        errors[t] = e.what();
      }
    });
  for (auto &th : pool)
    th.join();
  for (const auto &e : errors)
    REQUIRE(e.empty());
  for (double e : energies)
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(112.5, 1e-9));
  REQUIRE(listing(root / "aoti") ==
          std::set<std::string>{expected_unpack(root).filename().string()});
}
