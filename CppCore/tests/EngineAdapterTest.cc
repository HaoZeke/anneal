// MIT License
// Copyright 2023--present rgpot developers
//
// The generic engine table (rgpot/engine/EngineTable.cc) over a stand-in
// backend: argument validation, the transform, batching, error text, identity
// queries, the optional hooks and the thread-safety policy. The engine is
// opened the way a host opens it: dlopen of RGPOT_ENGINE_STUB and dlsym of each
// symbol.

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/engine/Sha256.hpp"
#include "rgpot/engine_c_abi.h"

namespace {

struct Stub {
  void *handle = nullptr;
  int (*abi_version)() = nullptr;
  const char *(*name)() = nullptr;
  const char *(*version)() = nullptr;
  RgpotEnginePot *(*create)(const void *, size_t, char *, size_t) = nullptr;
  void (*destroy)(RgpotEnginePot *) = nullptr;
  int (*force)(RgpotEnginePot *, long, const double *, const int *, double *,
               double *, double *, const double *, rgpot_engine_coord_transform,
               void *) = nullptr;
  int (*force_batch)(RgpotEnginePot *, long, long, const double *, const int *,
                     double *, double *, double *, const double *,
                     rgpot_engine_coord_transform, void *) = nullptr;
  RgpotEngineAbiStamp (*stamp)() = nullptr;
  int (*caps)(const RgpotEnginePot *, RgpotEngineCaps *) = nullptr;
  size_t (*last_error)(const RgpotEnginePot *, char *, size_t) = nullptr;
  size_t (*digest)(const RgpotEnginePot *, char *, size_t) = nullptr;
  int (*set_num_threads)(RgpotEnginePot *, int, int) = nullptr;
  int (*set_charge_spin)(RgpotEnginePot *, int, int) = nullptr;
  int (*select_orbitals)(RgpotEnginePot *, long long) = nullptr;
  int (*copy_orbitals)(RgpotEnginePot *, long long, long long) = nullptr;
  int (*max_concurrent)() = nullptr;
  void (*reset)() = nullptr;
  int (*charge)() = nullptr;
  int (*spin)() = nullptr;
};

template <typename Fn> Fn sym(void *h, const char *name) {
  auto fn = reinterpret_cast<Fn>(dlsym(h, name));
  REQUIRE(fn != nullptr);
  return fn;
}

Stub &stub() {
  static Stub s = [] {
    Stub out;
    const char *path = std::getenv("RGPOT_ENGINE_STUB");
    REQUIRE(path != nullptr);
    out.handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    REQUIRE(out.handle != nullptr);
    auto h = out.handle;
    out.abi_version = sym<int (*)()>(h, "rgpot_engine_abi_version");
    out.name = sym<const char *(*)()>(h, "rgpot_engine_name");
    out.version = sym<const char *(*)()>(h, "rgpot_engine_version");
    out.create = sym<decltype(out.create)>(h, "rgpot_engine_create");
    out.destroy = sym<decltype(out.destroy)>(h, "rgpot_engine_destroy");
    out.force = sym<decltype(out.force)>(h, "rgpot_engine_force");
    out.force_batch =
        sym<decltype(out.force_batch)>(h, "rgpot_engine_force_batch");
    out.stamp = sym<decltype(out.stamp)>(h, "rgpot_engine_abi_stamp");
    out.caps = sym<decltype(out.caps)>(h, "rgpot_engine_caps");
    out.last_error =
        sym<decltype(out.last_error)>(h, "rgpot_engine_last_error");
    out.digest = sym<decltype(out.digest)>(h, "rgpot_engine_model_digest");
    out.set_num_threads =
        sym<decltype(out.set_num_threads)>(h, "rgpot_engine_set_num_threads");
    out.set_charge_spin =
        sym<decltype(out.set_charge_spin)>(h, "rgpot_engine_set_charge_spin");
    out.select_orbitals =
        sym<decltype(out.select_orbitals)>(h, "rgpot_engine_select_orbitals");
    out.copy_orbitals =
        sym<decltype(out.copy_orbitals)>(h, "rgpot_engine_copy_orbitals");
    out.max_concurrent = sym<int (*)()>(h, "stub_max_concurrent");
    out.reset = sym<void (*)()>(h, "stub_reset");
    out.charge = sym<int (*)()>(h, "stub_charge");
    out.spin = sym<int (*)()>(h, "stub_spin");
    return out;
  }();
  return s;
}

/// A word-aligned configuration: reentrancy, hooks cleared, model named.
RgpotEnginePot *make(Stub &s, unsigned char reentrancy,
                     unsigned char no_hooks = 0, unsigned char model = 0) {
  alignas(8) unsigned char cfg[8] = {reentrancy, no_hooks, model, 0,
                                     0,          0,        0,     0};
  char err[256] = {0};
  RgpotEnginePot *pot = s.create(cfg, sizeof(cfg), err, sizeof(err));
  INFO(err);
  REQUIRE(pot != nullptr);
  return pot;
}

std::string error_of(Stub &s, const RgpotEnginePot *pot) {
  const size_t n = s.last_error(pot, nullptr, 0);
  std::string out(n + 1, '\0');
  s.last_error(pot, out.data(), out.size());
  out.resize(n);
  return out;
}

const std::vector<int> kZ = {1, 1};
const std::array<double, 9> kBox = {10, 0, 0, 0, 10, 0, 0, 0, 10};

/// Runs `threads` workers, each making `calls` force calls on `pots[w % n]`.
void hammer(Stub &s, const std::vector<RgpotEnginePot *> &pots, int threads,
            int calls) {
  std::vector<std::thread> workers;
  std::atomic<int> failures{0};
  for (int w = 0; w < threads; ++w) {
    workers.emplace_back([&, w] {
      const double pos[6] = {1, 2, 3, 4, 5, 6};
      double forces[6], energy = 0.0;
      for (int c = 0; c < calls; ++c) {
        if (s.force(pots[w % pots.size()], 2, pos, kZ.data(), forces, &energy,
                    nullptr, kBox.data(), nullptr, nullptr) != 0) {
          ++failures;
        }
      }
    });
  }
  for (auto &t : workers) {
    t.join();
  }
  REQUIRE(failures == 0);
}

} // namespace

TEST_CASE("the table identifies the engine", "[engine][adapter]") {
  Stub &s = stub();
  REQUIRE(s.abi_version() == RGPOT_ENGINE_ABI_VERSION);
  REQUIRE(std::string(s.name()) == "stub");
  const std::string version = s.version();
  REQUIRE(version.rfind("stub/", 0) == 0);
  REQUIRE(version.size() > 5);
  const RgpotEngineAbiStamp stamp = s.stamp();
  REQUIRE(stamp.abi_major == RGPOT_ENGINE_ABI_VERSION);
  REQUIRE(stamp.abi_minor == RGPOT_ENGINE_ABI_MINOR);
  REQUIRE(RGPOT_ENGINE_ABI_MINOR >= 3);
}

TEST_CASE("create refuses a missing or ragged configuration",
          "[engine][adapter]") {
  Stub &s = stub();
  char err[256] = {0};
  REQUIRE(s.create(nullptr, 0, err, sizeof(err)) == nullptr);
  REQUIRE(std::string(err).find("params message") != std::string::npos);
  unsigned char ragged[7] = {0};
  std::memset(err, 0, sizeof(err));
  REQUIRE(s.create(ragged, sizeof(ragged), err, sizeof(err)) == nullptr);
  REQUIRE(!std::string(err).empty());
}

TEST_CASE("force returns the backend result and leaves the caller's buffers",
          "[engine][adapter]") {
  Stub &s = stub();
  RgpotEnginePot *pot = make(s, 0);
  const double pos[6] = {1, 2, 3, 4, 5, 6};
  double forces[6] = {0}, energy = 0.0, variance = 0.0;
  REQUIRE(s.force(pot, 2, pos, kZ.data(), forces, &energy, &variance,
                  kBox.data(), nullptr, nullptr) == 0);
  REQUIRE(energy == Catch::Approx(0.5 * (1 + 4 + 9 + 16 + 25 + 36)));
  REQUIRE(variance == 0.25);
  for (int k = 0; k < 6; ++k) {
    REQUIRE(forces[k] == -pos[k]);
  }

  // The transform edits a scratch copy: the energy follows it, the caller's
  // positions do not change.
  auto shift = [](void *, long n, double *p, double *) {
    for (long k = 0; k < 3 * n; ++k) {
      p[k] += 1.0;
    }
    return 0;
  };
  REQUIRE(s.force(pot, 2, pos, kZ.data(), forces, &energy, nullptr, kBox.data(),
                  shift, nullptr) == 0);
  REQUIRE(energy == Catch::Approx(0.5 * (4 + 9 + 16 + 25 + 36 + 49)));
  REQUIRE(pos[0] == 1.0);

  auto reject = [](void *, long, double *, double *) { return 9; };
  REQUIRE(s.force(pot, 2, pos, kZ.data(), forces, &energy, nullptr, kBox.data(),
                  reject, nullptr) == 9);
  REQUIRE(error_of(s, pot).find("9") != std::string::npos);

  REQUIRE(s.force(pot, 0, pos, kZ.data(), forces, &energy, nullptr, kBox.data(),
                  nullptr, nullptr) == 1);
  REQUIRE(error_of(s, pot).find("invalid argument") != std::string::npos);
  REQUIRE(s.force(pot, 2, pos, kZ.data(), forces, &energy, nullptr, kBox.data(),
                  nullptr, nullptr) == 0);
  REQUIRE(error_of(s, pot).empty());
  s.destroy(pot);
}

TEST_CASE("force_batch evaluates strided systems", "[engine][adapter]") {
  Stub &s = stub();
  RgpotEnginePot *pot = make(s, 0);
  const double pos[12] = {1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0};
  std::vector<double> boxes;
  for (int i = 0; i < 2; ++i) {
    boxes.insert(boxes.end(), kBox.begin(), kBox.end());
  }
  double forces[12] = {0}, energies[2] = {0}, variances[2] = {0};
  REQUIRE(s.force_batch(pot, 2, 2, pos, kZ.data(), forces, energies, variances,
                        boxes.data(), nullptr, nullptr) == 0);
  REQUIRE(energies[0] == Catch::Approx(1.0));
  REQUIRE(energies[1] == Catch::Approx(4.0));
  REQUIRE(forces[6] == -2.0);
  REQUIRE(variances[1] == 0.25);
  s.destroy(pot);
}

TEST_CASE("the capabilities come from the backend", "[engine][adapter]") {
  Stub &s = stub();
  for (unsigned char r = 0; r < 3; ++r) {
    RgpotEnginePot *pot = make(s, r);
    RgpotEngineCaps caps;
    caps.size = sizeof(caps);
    REQUIRE(s.caps(pot, &caps) == 0);
    REQUIRE(caps.reentrancy == r);
    REQUIRE(caps.per_image_instances == 1);
    REQUIRE(caps.orbital_selection == 1);
    REQUIRE(caps.orbital_copy == 1);
    s.destroy(pot);
  }
}

TEST_CASE("the model digest is the SHA-256 of the model file",
          "[engine][adapter]") {
  Stub &s = stub();
  const std::string path = std::string(std::getenv("RGPOT_ENGINE_STUB_MODEL"));
  {
    std::ofstream out(path, std::ios::binary);
    out << "abc";
  }
  RgpotEnginePot *pot = make(s, 0, 0, 1);
  char buf[65] = {0};
  REQUIRE(s.digest(pot, buf, sizeof(buf)) == 64);
  REQUIRE(std::string(buf) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  char small[9];
  REQUIRE(s.digest(pot, small, sizeof(small)) == 64);
  REQUIRE(std::strlen(small) == 8);
  // The digest is computed once: a later change of the file does not move it.
  {
    std::ofstream out(path, std::ios::binary);
    out << "different";
  }
  REQUIRE(s.digest(pot, buf, sizeof(buf)) == 64);
  REQUIRE(std::string(buf).rfind("ba7816bf", 0) == 0);
  s.destroy(pot);

  RgpotEnginePot *none = make(s, 0, 0, 0);
  REQUIRE(s.digest(none, buf, sizeof(buf)) == 0);
  REQUIRE(buf[0] == '\0');
  s.destroy(none);
  std::remove(path.c_str());
  RgpotEnginePot *gone = make(s, 0, 0, 1);
  REQUIRE(s.digest(gone, buf, sizeof(buf)) == 0);
  s.destroy(gone);
}

TEST_CASE("charge, spin and thread settings reach the backend",
          "[engine][adapter]") {
  Stub &s = stub();
  s.reset();
  RgpotEnginePot *pot = make(s, 0);
  REQUIRE(s.set_charge_spin(pot, -1, 2) == 0);
  REQUIRE(s.charge() == -1);
  REQUIRE(s.spin() == 2);
  REQUIRE(s.set_charge_spin(pot, 0, 0) == 1);
  REQUIRE(s.set_charge_spin(pot, 99, 1) == 3);
  REQUIRE(error_of(s, pot).find("refused") != std::string::npos);
  REQUIRE(s.charge() == -1);
  REQUIRE(s.set_num_threads(pot, 2, 1) == 0);
  REQUIRE(s.set_num_threads(pot, 2, 99) == 2);
  REQUIRE(error_of(s, pot).find("refused") != std::string::npos);
  s.destroy(pot);

  RgpotEnginePot *bare = make(s, 0, 1);
  REQUIRE(s.set_charge_spin(bare, 0, 1) == 3);
  REQUIRE(s.set_num_threads(bare, 1, 1) == 3);
  REQUIRE(!error_of(s, bare).empty());
  REQUIRE(s.select_orbitals(bare, 1) == 3);
  REQUIRE(s.copy_orbitals(bare, 1, 2) == 3);
  s.destroy(bare);
}

TEST_CASE("orbital hooks report success and adapter errors",
          "[engine][adapter]") {
  Stub &s = stub();
  RgpotEnginePot *pot = make(s, 0);

  REQUIRE(s.select_orbitals(pot, 7) == 0);
  REQUIRE(s.copy_orbitals(pot, 7, 8) == 0);
  REQUIRE(s.select_orbitals(nullptr, 7) == 1);
  REQUIRE(s.copy_orbitals(nullptr, 7, 8) == 1);

  REQUIRE(s.select_orbitals(pot, -1) == 3);
  REQUIRE(error_of(s, pot).find("refused") != std::string::npos);
  REQUIRE(s.copy_orbitals(pot, -1, 8) == 3);
  REQUIRE(error_of(s, pot).find("refused") != std::string::npos);

  REQUIRE(s.select_orbitals(pot, -2) == 2);
  REQUIRE(error_of(s, pot).find("failed") != std::string::npos);
  REQUIRE(s.copy_orbitals(pot, 7, -2) == 2);
  REQUIRE(error_of(s, pot).find("failed") != std::string::npos);
  s.destroy(pot);
}

TEST_CASE("a shared instance runs calls concurrently",
          "[engine][adapter][threads]") {
  Stub &s = stub();
  s.reset();
  RgpotEnginePot *pot = make(s, 0);
  hammer(s, {pot}, 8, 8);
  REQUIRE(s.max_concurrent() > 1);
  s.destroy(pot);
}

TEST_CASE("a per-instance backend admits one call per handle",
          "[engine][adapter][threads]") {
  Stub &s = stub();
  s.reset();
  RgpotEnginePot *pot = make(s, 1);
  hammer(s, {pot}, 8, 8);
  REQUIRE(s.max_concurrent() == 1);

  // Separate handles do not wait for each other.
  s.reset();
  RgpotEnginePot *other = make(s, 1);
  hammer(s, {pot, other}, 8, 8);
  REQUIRE(s.max_concurrent() > 1);
  s.destroy(pot);
  s.destroy(other);
}

TEST_CASE("a process-serial backend admits one call across handles",
          "[engine][adapter][threads]") {
  Stub &s = stub();
  s.reset();
  RgpotEnginePot *a = make(s, 2);
  RgpotEnginePot *b = make(s, 2);
  hammer(s, {a, b}, 8, 8);
  REQUIRE(s.max_concurrent() == 1);
  s.destroy(a);
  s.destroy(b);
}

TEST_CASE("SHA-256 matches the published vectors", "[engine][sha256]") {
  using rgpot::engine::Sha256;
  Sha256 h;
  REQUIRE(h.hex_digest() ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  h.update("abc", 3);
  REQUIRE(h.hex_digest() ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  const std::string two_blocks =
      "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  h.update(two_blocks.data(), two_blocks.size());
  REQUIRE(h.hex_digest() ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  // A million 'a' in uneven pieces.
  const std::string chunk(1000, 'a');
  for (int i = 0; i < 1000; ++i) {
    h.update(chunk.data(), (i % 2 == 0) ? 1000 : 999);
    if (i % 2 == 1) {
      h.update("a", 1);
    }
  }
  REQUIRE(h.hex_digest() ==
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  REQUIRE(Sha256::of_file("/nonexistent/file").empty());
}
