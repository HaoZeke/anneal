// MIT License
// Copyright 2023--present rgpot developers
//
// The generic engine table of each backend library this build produces:
// classical (lj, ljcluster, morse, zbl and the Fortran potentials), metatomic,
// mopac, nwchem, skala and xtb. Each library is opened the way a host opens
// it: dlopen of the path in RGPOT_ENGINE_<NAME> and dlsym of each symbol.
// RGPOT_ENGINE_EXPECT lists the engines the build made; one of them without a
// path fails the test instead of skipping it. The metatomic cases load the
// models in MTA_<NAME>_PT, which scripts/export_tiny_metatomic_model.py
// writes.

#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJPot.hpp"
#ifdef RGPOT_HAS_WATER
#include "rgpot/Water/WaterPots.hpp"
#endif
#include "rgpot/engine_c_abi.h"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/types/AtomMatrix.hpp"

namespace {

struct Engine {
  void *handle = nullptr;
  int (*abi_version)() = nullptr;
  const char *(*name)() = nullptr;
  const char *(*version)() = nullptr;
  RgpotEnginePot *(*create)(const void *, size_t, char *, size_t) = nullptr;
  void (*destroy)(RgpotEnginePot *) = nullptr;
  int (*force)(RgpotEnginePot *, long, const double *, const int *, double *,
               double *, double *, const double *,
               rgpot_engine_coord_transform, void *) = nullptr;
  int (*hvp)(RgpotEnginePot *, long, const double *, const int *,
             const double *, double *, const double *) = nullptr;
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
};

template <typename Fn> Fn sym(void *h, const char *name) {
  auto fn = reinterpret_cast<Fn>(dlsym(h, name));
  INFO(name);
  REQUIRE(fn != nullptr);
  return fn;
}

/// Opens the engine named `key` ("CLASSICAL"), or reports that the build has
/// none. A listed engine without a path fails.
bool open_engine(const std::string &key, Engine &out) {
  const char *path = std::getenv(("RGPOT_ENGINE_" + key).c_str());
  const char *expect = std::getenv("RGPOT_ENGINE_EXPECT");
  std::string expected = expect ? expect : "";
  const bool listed = expected.find(key) != std::string::npos;
  if (path == nullptr) {
    INFO(key << " is listed in RGPOT_ENGINE_EXPECT but has no path");
    REQUIRE(!listed);
    return false;
  }
  out.handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  // dlerror() is null after a success, and streaming a null char pointer
  // crashes libc++.
  const char *load_error = out.handle ? nullptr : dlerror();
  const std::string load_message = load_error ? load_error : "";
  INFO(load_message);
  REQUIRE(out.handle != nullptr);
  auto h = out.handle;
  out.abi_version = sym<int (*)()>(h, "rgpot_engine_abi_version");
  out.name = sym<const char *(*)()>(h, "rgpot_engine_name");
  out.version = sym<const char *(*)()>(h, "rgpot_engine_version");
  out.create = sym<decltype(out.create)>(h, "rgpot_engine_create");
  out.destroy = sym<decltype(out.destroy)>(h, "rgpot_engine_destroy");
  out.force = sym<decltype(out.force)>(h, "rgpot_engine_force");
  out.hvp = sym<decltype(out.hvp)>(h, "rgpot_engine_hvp");
  out.force_batch = sym<decltype(out.force_batch)>(h, "rgpot_engine_force_batch");
  out.stamp = sym<decltype(out.stamp)>(h, "rgpot_engine_abi_stamp");
  out.caps = sym<decltype(out.caps)>(h, "rgpot_engine_caps");
  out.last_error = sym<decltype(out.last_error)>(h, "rgpot_engine_last_error");
  out.digest = sym<decltype(out.digest)>(h, "rgpot_engine_model_digest");
  out.set_num_threads =
      sym<decltype(out.set_num_threads)>(h, "rgpot_engine_set_num_threads");
  out.set_charge_spin =
      sym<decltype(out.set_charge_spin)>(h, "rgpot_engine_set_charge_spin");
  out.select_orbitals =
      sym<decltype(out.select_orbitals)>(h, "rgpot_engine_select_orbitals");
  out.copy_orbitals =
      sym<decltype(out.copy_orbitals)>(h, "rgpot_engine_copy_orbitals");
  return true;
}

kj::Array<capnp::word> builtin_params(const std::string &name) {
  capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::BuiltinParams>();
  p.setName(name);
  return capnp::messageToFlatArray(msg);
}

kj::Array<capnp::word> xtb_params(const std::string &method) {
  capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::BuiltinParams>();
  p.setName("xtb");
  auto x = p.initConfig().initXtb();
  x.setMethod(method);
  x.setAccuracy(0.1);
  x.setElectronicTemperature(100.0);
  x.setMaxIterations(300);
  return capnp::messageToFlatArray(msg);
}

RgpotEnginePot *create(const Engine &e, const kj::Array<capnp::word> &words,
                       std::string *error = nullptr) {
  auto bytes = words.asBytes();
  char err[512] = {0};
  RgpotEnginePot *pot = e.create(bytes.begin(), bytes.size(), err, sizeof(err));
  if (error) {
    *error = err;
  }
  return pot;
}

std::string error_of(const Engine &e, const RgpotEnginePot *pot) {
  const size_t n = e.last_error(pot, nullptr, 0);
  std::string out(n + 1, '\0');
  e.last_error(pot, out.data(), out.size());
  out.resize(n);
  return out;
}

void check_identity(const Engine &e, const std::string &name) {
  REQUIRE(e.abi_version() == RGPOT_ENGINE_ABI_VERSION);
  REQUIRE(std::string(e.name()) == name);
  REQUIRE(std::string(e.version()).rfind(name + "/", 0) == 0);
  const RgpotEngineAbiStamp stamp = e.stamp();
  REQUIRE(stamp.abi_minor == RGPOT_ENGINE_ABI_MINOR);
}

// Five LJ atoms, Angstrom, reduced units.
const std::vector<double> kCluster = {0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2,
                                      0.0, 0.0, 0.0, 1.3, 1.0, 1.0, 1.0};
const std::vector<int> kZ = {1, 1, 1, 1, 1};
const std::array<double, 9> kBox = {40, 0, 0, 0, 40, 0, 0, 0, 40};

} // namespace

TEST_CASE("classical engine matches the directly compiled LJPot",
          "[engine][classical]") {
  Engine e;
  if (!open_engine("CLASSICAL", e)) {
    SKIP("no classical engine in this build");
  }
  check_identity(e, "classical");
  std::string error;
  RgpotEnginePot *pot = create(e, builtin_params("lj"), &error);
  INFO(error);
  REQUIRE(pot != nullptr);

  RgpotEngineCaps caps;
  caps.size = sizeof(caps);
  REQUIRE(e.caps(pot, &caps) == 0);
  // Neighbour scratch is kept per thread, so one LJPot instance is shared.
  REQUIRE(caps.reentrancy == 0);

  const long n = 5;
  std::vector<double> forces(3 * n, 0.0);
  double energy = 0.0;
  REQUIRE(e.force(pot, n, kCluster.data(), kZ.data(), forces.data(), &energy,
                  nullptr, kBox.data(), nullptr, nullptr) == 0);

  rgpot::LJPot direct;
  rgpot::types::AtomMatrix positions(n, 3);
  for (long i = 0; i < n; ++i) {
    for (int d = 0; d < 3; ++d) {
      positions(i, d) = kCluster[3 * i + d];
    }
  }
  const std::array<std::array<double, 3>, 3> box = {
      {{40, 0, 0}, {0, 40, 0}, {0, 0, 40}}};
  auto [ref_e, ref_f, ref_v] = direct(positions, kZ, box);
  (void)ref_v;
  REQUIRE(energy == Catch::Approx(ref_e).margin(1e-12));
  for (long i = 0; i < n; ++i) {
    for (int d = 0; d < 3; ++d) {
      REQUIRE(forces[3 * i + d] == Catch::Approx(ref_f(i, d)).margin(1e-12));
    }
  }

  // A batch of three copies equals three single calls.
  std::vector<double> R, boxes, F(3 * n * 3, 0.0), U(3, 0.0);
  for (int s = 0; s < 3; ++s) {
    R.insert(R.end(), kCluster.begin(), kCluster.end());
    boxes.insert(boxes.end(), kBox.begin(), kBox.end());
  }
  REQUIRE(e.force_batch(pot, 3, n, R.data(), kZ.data(), F.data(), U.data(),
                        nullptr, boxes.data(), nullptr, nullptr) == 0);
  for (int s = 0; s < 3; ++s) {
    REQUIRE(U[s] == energy);
  }

  // No model, no charge, no tensor runtime.
  char buf[65] = {1};
  REQUIRE(e.digest(pot, buf, sizeof(buf)) == 0);
  REQUIRE(buf[0] == '\0');
  REQUIRE(e.set_charge_spin(pot, 0, 1) == 3);
  REQUIRE(e.set_num_threads(pot, 1, 1) == 3);
  REQUIRE(e.select_orbitals(pot, 1) == 3);
  REQUIRE(e.copy_orbitals(pot, 1, 2) == 3);
  e.destroy(pot);
}

TEST_CASE("classical engine LJPot analytic HVP rejects in-place aliasing",
          "[engine][classical][hvp]") {
  Engine e;
  if (!open_engine("CLASSICAL", e)) {
    SKIP("no classical engine in this build");
  }
  RgpotEnginePot *pot = create(e, builtin_params("lj"));
  REQUIRE(pot != nullptr);

  const long n = 5;
  const std::vector<double> direction = {
      0.2, -0.3, 0.4, -0.1, 0.5, -0.2, 0.3, 0.1,
      -0.4, 0.6, -0.5, 0.2, -0.3, 0.4, -0.1};
  std::vector<double> product(direction.size(), 0.0);
  REQUIRE(e.hvp(pot, n, kCluster.data(), kZ.data(), direction.data(),
                product.data(), kBox.data()) == RGPOT_ENGINE_OK);

  rgpot::LJPot direct;
  std::vector<double> ref_product(direction.size(), 0.0);
  const rgpot::ForceInput fi{.nAtoms = static_cast<size_t>(n),
                             .pos = kCluster.data(),
                             .atmnrs = kZ.data(),
                             .box = kBox.data()};
  direct.hessianVectorImpl(fi, direction.data(), ref_product.data());
  const double tol =
      64.0 * std::numeric_limits<double>::epsilon();
  for (std::size_t k = 0; k < direction.size(); ++k) {
    REQUIRE(product[k] == Catch::Approx(ref_product[k]).margin(tol));
  }

  std::vector<double> aliased = direction;
  REQUIRE(e.hvp(pot, n, kCluster.data(), kZ.data(), aliased.data(),
                aliased.data(), kBox.data()) == RGPOT_ENGINE_ERR_INVALID);
  REQUIRE(error_of(e, pot).find("overlap") != std::string::npos);
  REQUIRE(aliased == direction);
  e.destroy(pot);
}

TEST_CASE("classical engine LJPot reports coincident atoms as evaluation errors",
          "[engine][classical][lj]") {
  Engine e;
  if (!open_engine("CLASSICAL", e)) {
    SKIP("no classical engine in this build");
  }
  RgpotEnginePot *pot = create(e, builtin_params("lj"));
  REQUIRE(pot != nullptr);

  const std::array<double, 6> coincident{};
  const std::array<int, 2> z{1, 1};
  const std::array<double, 6> direction{0.2, -0.3, 0.4, -0.1, 0.5, -0.2};
  std::array<double, 6> forces{};
  std::array<double, 6> product{};
  double energy = 0.0;

  REQUIRE(e.force(pot, 2, coincident.data(), z.data(), forces.data(), &energy,
                  nullptr, kBox.data(), nullptr, nullptr) ==
          RGPOT_ENGINE_ERR_EVAL);
  REQUIRE(error_of(e, pot).find("coincident atoms") != std::string::npos);

  REQUIRE(e.force_batch(pot, 1, 2, coincident.data(), z.data(), forces.data(),
                        &energy, nullptr, kBox.data(), nullptr, nullptr) ==
          RGPOT_ENGINE_ERR_EVAL);
  REQUIRE(error_of(e, pot).find("coincident atoms") != std::string::npos);

  REQUIRE(e.hvp(pot, 2, coincident.data(), z.data(), direction.data(),
                product.data(), kBox.data()) == RGPOT_ENGINE_ERR_EVAL);
  REQUIRE(error_of(e, pot).find("coincident atoms") != std::string::npos);

  const std::array<double, 6> separated{0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
  REQUIRE(e.force(pot, 2, separated.data(), z.data(), forces.data(), &energy,
                  nullptr, kBox.data(), nullptr, nullptr) == RGPOT_ENGINE_OK);
  REQUIRE(error_of(e, pot).empty());
  e.destroy(pot);
}

TEST_CASE("one classical handle serves many threads", "[engine][classical][threads]") {
  Engine e;
  if (!open_engine("CLASSICAL", e)) {
    SKIP("no classical engine in this build");
  }
  RgpotEnginePot *pot = create(e, builtin_params("lj"));
  REQUIRE(pot != nullptr);
  const long n = 5;
  std::vector<double> ref(3 * n, 0.0);
  double ref_e = 0.0;
  REQUIRE(e.force(pot, n, kCluster.data(), kZ.data(), ref.data(), &ref_e,
                  nullptr, kBox.data(), nullptr, nullptr) == 0);
  std::vector<std::thread> workers;
  std::atomic<int> bad{0};
  for (int w = 0; w < 8; ++w) {
    workers.emplace_back([&] {
      std::vector<double> f(3 * n);
      double en = 0.0;
      for (int c = 0; c < 200; ++c) {
        if (e.force(pot, n, kCluster.data(), kZ.data(), f.data(), &en, nullptr,
                    kBox.data(), nullptr, nullptr) != 0 ||
            en != ref_e || f != ref) {
          ++bad;
        }
      }
    });
  }
  for (auto &t : workers) {
    t.join();
  }
  REQUIRE(bad == 0);
  e.destroy(pot);
}

TEST_CASE("classical engine names its potentials", "[engine][classical]") {
  Engine e;
  if (!open_engine("CLASSICAL", e)) {
    SKIP("no classical engine in this build");
  }
  for (const char *name : {"lj", "ljcluster", "morse", "zbl"}) {
    INFO(name);
    RgpotEnginePot *pot = create(e, builtin_params(name));
    REQUIRE(pot != nullptr);
    e.destroy(pot);
  }
  std::string error;
  REQUIRE(create(e, builtin_params("nonesuch"), &error) == nullptr);
  REQUIRE(error.find("nonesuch") != std::string::npos);

  // The Fortran kernels, when built, report one instance per thread.
  std::string probe;
  RgpotEnginePot *sw = create(e, builtin_params("sw"), &probe);
  if (sw != nullptr) {
    RgpotEngineCaps caps;
    caps.size = sizeof(caps);
    REQUIRE(e.caps(sw, &caps) == 0);
    REQUIRE(caps.reentrancy == 1);
    e.destroy(sw);
    for (const char *name :
         {"edip", "lenosky", "tersoff", "eamal", "fehe", "cuh2", "waterh"}) {
      INFO(name);
      RgpotEnginePot *pot = create(e, builtin_params(name));
      REQUIRE(pot != nullptr);
      e.destroy(pot);
    }
  }
}

#ifdef RGPOT_HAS_WATER
namespace {

template <typename Pot>
void check_water(const Engine &e, const std::string &name,
                 const std::vector<double> &pos, const std::vector<int> &z) {
  INFO(name);
  std::string error;
  RgpotEnginePot *pot = create(e, builtin_params(name), &error);
  INFO(error);
  REQUIRE(pot != nullptr);
  const long n = static_cast<long>(z.size());
  std::vector<double> forces(3 * n, 0.0);
  double energy = 0.0;
  REQUIRE(e.force(pot, n, pos.data(), z.data(), forces.data(), &energy, nullptr,
                  kBox.data(), nullptr, nullptr) == 0);
  e.destroy(pot);

  Pot direct;
  rgpot::types::AtomMatrix positions(n, 3);
  for (long i = 0; i < n; ++i) {
    for (int d = 0; d < 3; ++d) {
      positions(i, d) = pos[3 * i + d];
    }
  }
  const std::array<std::array<double, 3>, 3> box = {
      {{40, 0, 0}, {0, 40, 0}, {0, 0, 40}}};
  auto [ref_e, ref_f, ref_v] = direct(positions, z, box);
  (void)ref_v;
  REQUIRE(energy == Catch::Approx(ref_e).margin(1e-12));
  for (long i = 0; i < n; ++i) {
    for (int d = 0; d < 3; ++d) {
      REQUIRE(forces[3 * i + d] == Catch::Approx(ref_f(i, d)).margin(1e-12));
    }
  }
}

} // namespace

TEST_CASE("classical engine serves the water potentials",
          "[engine][classical][water]") {
  Engine e;
  if (!open_engine("CLASSICAL", e)) {
    SKIP("no classical engine in this build");
  }
  // One distorted water, Angstrom. Order is H, H, O.
  const std::vector<double> water = {0.80, 0.10, 0.50, -0.70, 0.05,
                                     0.55, 0.0,  0.0,  0.0};
  check_water<rgpot::TIP4PPot>(e, "tip4p", water, {1, 1, 8});
  check_water<rgpot::SPCEPot>(e, "spce", water, {1, 1, 8});
  // TIP4P-Pt mirrors the charges in z = 0, so the water sits above the
  // platinum atom.
  const std::vector<double> on_pt = {0.80, 0.10, 2.50, -0.70, 0.05, 2.55,
                                     0.0,  0.0,  2.0,  0.0,   0.0,  0.0};
  check_water<rgpot::TIP4PPtPot>(e, "tip4p_pt", on_pt, {1, 1, 8, 78});
}
#endif

namespace {

/// A backend whose library is not installed: the engine is created and the
/// first evaluation fails with a reason.
void check_unavailable_engine(const std::string &key, const std::string &name,
                              const kj::Array<capnp::word> &params,
                              int reentrancy) {
  Engine e;
  if (!open_engine(key, e)) {
    SKIP("no " + name + " engine in this build");
  }
  check_identity(e, name);
  std::string error;
  RgpotEnginePot *pot = create(e, params, &error);
  INFO(error);
  REQUIRE(pot != nullptr);
  RgpotEngineCaps caps;
  caps.size = sizeof(caps);
  REQUIRE(e.caps(pot, &caps) == 0);
  REQUIRE(caps.reentrancy == reentrancy);
  REQUIRE(caps.periodic == 0);
  std::vector<double> forces(6, 0.0);
  double energy = 0.0;
  const double pos[6] = {0, 0, 0, 0, 0, 0.74};
  const int z[2] = {1, 1};
  const int rc = e.force(pot, 2, pos, z, forces.data(), &energy, nullptr,
                         kBox.data(), nullptr, nullptr);
  if (rc != 0) {
    REQUIRE(rc == 2);
    REQUIRE(!error_of(e, pot).empty());
  }
  char buf[65];
  REQUIRE(e.digest(pot, buf, sizeof(buf)) == 0);
  e.destroy(pot);
}

} // namespace

TEST_CASE("mopac engine", "[engine][mopac]") {
  check_unavailable_engine("MOPAC", "mopac", builtin_params("mopac"), 2);
}

TEST_CASE("nwchem engine", "[engine][nwchem]") {
  capnp::MallocMessageBuilder msg;
  msg.initRoot<::NWChemParams>();
  check_unavailable_engine("NWCHEM", "nwchem", capnp::messageToFlatArray(msg), 2);
}

TEST_CASE("skala engine", "[engine][skala]") {
  capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::NWChemParams>();
  p.setBasis("def2-svp");
  check_unavailable_engine("SKALA", "skala", capnp::messageToFlatArray(msg), 2);

  Engine e;
  REQUIRE(open_engine("SKALA", e));
  capnp::MallocMessageBuilder msg2;
  msg2.initRoot<::NWChemParams>();
  RgpotEnginePot *pot = create(e, capnp::messageToFlatArray(msg2));
  REQUIRE(pot != nullptr);
  REQUIRE(e.set_charge_spin(pot, -1, 2) == 0);
  REQUIRE(e.set_num_threads(pot, 1, 1) == 3);
  e.destroy(pot);
}

TEST_CASE("xtb engine matches the generic table contract", "[engine][xtb]") {
  Engine e;
  if (!open_engine("XTB", e)) {
    SKIP("no xtb engine in this build");
  }
  check_identity(e, "xtb");
  RgpotEnginePot *pot = create(e, builtin_params("xtb"));
  REQUIRE(pot != nullptr);
  RgpotEngineCaps caps;
  caps.size = sizeof(caps);
  REQUIRE(e.caps(pot, &caps) == 0);
  REQUIRE(caps.reentrancy == 1);
  REQUIRE(caps.stress == 1);

  // Water, Angstrom.
  const double pos[9] = {0.0, 0.0, 0.1173, 0.0, 0.7572, -0.4692, 0.0, -0.7572,
                         -0.4692};
  const int z[3] = {8, 1, 1};
  const std::array<double, 9> box = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  std::vector<double> forces(9, 0.0);
  double energy = 0.0;
  REQUIRE(e.force(pot, 3, pos, z, forces.data(), &energy, nullptr, box.data(),
                  nullptr, nullptr) == 0);
  REQUIRE(energy < -100.0); // eV, GFN2 water is about -137 eV
  e.destroy(pot);

  std::string error;
  REQUIRE(create(e, xtb_params("nonesuch"), &error) == nullptr);
  REQUIRE(error.find("unknown method 'nonesuch'") != std::string::npos);
  RgpotEnginePot *gfn1 = create(e, xtb_params("gfn1"), &error);
  INFO(error);
  REQUIRE(gfn1 != nullptr);
  double gfn1_energy = 0.0;
  REQUIRE(e.force(gfn1, 3, pos, z, forces.data(), &gfn1_energy, nullptr,
                  box.data(), nullptr, nullptr) == 0);
  REQUIRE(gfn1_energy != energy);
  e.destroy(gfn1);
}

TEST_CASE("metatomic engine keeps its own table and adds the generic one",
          "[engine][metatomic]") {
  Engine e;
  if (!open_engine("METATOMIC", e)) {
    SKIP("no metatomic engine in this build");
  }
  check_identity(e, "metatomic");
  // The Metatomic-specific table is still exported.
  REQUIRE(dlsym(e.handle, "rgpot_mta_abi_version") != nullptr);
  REQUIRE(dlsym(e.handle, "rgpot_mta_create") != nullptr);

  std::string error;
  capnp::MallocMessageBuilder empty;
  empty.initRoot<::MetatomicParams>();
  REQUIRE(create(e, capnp::messageToFlatArray(empty), &error) == nullptr);
  REQUIRE(error.find("modelPath") != std::string::npos);

  // The new fields are readable on the wire.
  capnp::MallocMessageBuilder full;
  auto p = full.initRoot<::MetatomicParams>();
  p.setNSymmetryRotations(4);
  p.setRandomRotation(true);
  p.setSo3ProbeScatter(true);
  p.setTorchDeterminism(::MetatomicParams::TorchDeterminism::DETERMINISTIC);
  p.setIntraopThreads(2);
  p.setInteropThreads(1);
  capnp::MallocMessageBuilder copy;
  copy.setRoot(p.asReader());
  auto r = copy.getRoot<::MetatomicParams>();
  REQUIRE(r.getNSymmetryRotations() == 4);
  REQUIRE(r.getTorchDeterminism() == ::MetatomicParams::TorchDeterminism::DETERMINISTIC);
  REQUIRE(r.getIntraopThreads() == 2);
}

namespace {

/// A model the build exports next to the Metatomic engine.
std::string mta_model(const char *key) {
  const char *path = std::getenv(key);
  INFO(key << " is unset; the build sets it with RGPOT_ENGINE_METATOMIC");
  REQUIRE(path != nullptr);
  return path;
}

kj::Array<capnp::word> metatomic_params(const std::string &model,
                                        int charge = 0, int spin = 1,
                                        bool attach_system_extras = false) {
  capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::MetatomicParams>();
  p.setModelPath(model);
  p.setCharge(charge);
  p.setSpin(spin);
  p.setAttachSystemExtras(attach_system_extras);
  return capnp::messageToFlatArray(msg);
}

// An isolated H and O, Angstrom. A model that reads both gives
// E = 2.5 + 10 * charge + multiplicity eV; the forces are -positions.
const double kPair[6] = {0.0, 0.0, 1.0, 0.0, 2.0, 0.0};
const int kPairZ[2] = {1, 8};
const std::array<double, 9> kNoCell = {};

double pair_energy(const Engine &e, RgpotEnginePot *pot) {
  std::vector<double> forces(6, 0.0);
  double energy = 0.0;
  const int rc = e.force(pot, 2, kPair, kPairZ, forces.data(), &energy,
                         nullptr, kNoCell.data(), nullptr, nullptr);
  INFO(error_of(e, pot));
  REQUIRE(rc == 0);
  for (int k = 0; k < 6; ++k) {
    REQUIRE(forces[k] == Catch::Approx(-kPair[k]).margin(1e-12));
  }
  return energy;
}

/// Two copies of the pair in one call, which is one model forward.
std::array<double, 2> pair_batch_energies(const Engine &e,
                                          RgpotEnginePot *pot) {
  std::vector<double> R(kPair, kPair + 6);
  R.insert(R.end(), kPair, kPair + 6);
  std::vector<double> cells(18, 0.0), forces(12, 0.0);
  std::array<double, 2> energies = {0.0, 0.0};
  const int rc = e.force_batch(pot, 2, 2, R.data(), kPairZ, forces.data(),
                               energies.data(), nullptr, cells.data(), nullptr,
                               nullptr);
  INFO(error_of(e, pot));
  REQUIRE(rc == 0);
  return energies;
}

} // namespace

TEST_CASE("metatomic charge and spin reach a model that reads them",
          "[engine][metatomic][charge]") {
  Engine e;
  if (!open_engine("METATOMIC", e)) {
    SKIP("no metatomic engine in this build");
  }
  const std::string model = mta_model("MTA_CHARGE_SPIN_PT");
  std::string error;
  RgpotEnginePot *pot = create(e, metatomic_params(model), &error);
  INFO(error);
  REQUIRE(pot != nullptr);
  CHECK(pair_energy(e, pot) == Catch::Approx(3.5));

  REQUIRE(e.set_charge_spin(pot, -1, 3) == 0);
  CHECK(pair_energy(e, pot) == Catch::Approx(-4.5));
  const auto batch = pair_batch_energies(e, pot);
  CHECK(batch[0] == Catch::Approx(-4.5));
  CHECK(batch[1] == Catch::Approx(-4.5));
  REQUIRE(e.set_charge_spin(pot, 0, 1) == 0);
  CHECK(pair_energy(e, pot) == Catch::Approx(3.5));
  e.destroy(pot);

  // MetatomicParams.charge and spin hold until the first set_charge_spin.
  pot = create(e, metatomic_params(model, 1, 2), &error);
  REQUIRE(pot != nullptr);
  CHECK(pair_energy(e, pot) == Catch::Approx(14.5));
  e.destroy(pot);

  // The multiplicity also reaches a model that reads it as rgpot::spin.
  pot = create(e, metatomic_params(mta_model("MTA_SPIN_PT")), &error);
  REQUIRE(pot != nullptr);
  REQUIRE(e.set_charge_spin(pot, 0, 3) == 0);
  CHECK(pair_energy(e, pot) == Catch::Approx(5.5));
  e.destroy(pot);

  // attachSystemExtras hands both to a model that reads them without
  // declaring them.
  pot = create(e, metatomic_params(mta_model("MTA_UNDECLARED_PT"), -1, 3, true),
               &error);
  REQUIRE(pot != nullptr);
  CHECK(pair_energy(e, pot) == Catch::Approx(-4.5));
  REQUIRE(e.set_charge_spin(pot, 1, 2) == 0);
  CHECK(pair_energy(e, pot) == Catch::Approx(14.5));
  e.destroy(pot);
}

TEST_CASE("metatomic engine refuses a charge or spin its model does not read",
          "[engine][metatomic][charge]") {
  Engine e;
  if (!open_engine("METATOMIC", e)) {
    SKIP("no metatomic engine in this build");
  }
  std::string error;

  // Declares neither, so the model sees neither and there is none to change.
  // The undeclared model reads its defaults, charge 0 and multiplicity 1.
  for (const auto &[key, energy] : {std::pair{"MTA_PLAIN_PT", 2.5},
                                    std::pair{"MTA_UNDECLARED_PT", 3.5}}) {
    INFO(key);
    RgpotEnginePot *pot = create(e, metatomic_params(mta_model(key)), &error);
    INFO(error);
    REQUIRE(pot != nullptr);
    CHECK(e.set_charge_spin(pot, -1, 3) == 3);
    CHECK(!error_of(e, pot).empty());
    CHECK(e.set_charge_spin(pot, 0, 1) == 3);
    CHECK(pair_energy(e, pot) == Catch::Approx(energy));
    e.destroy(pot);
  }

  // Reads the charge only: the multiplicity must stay 1, and a refused
  // setting changes nothing.
  RgpotEnginePot *pot = create(e, metatomic_params(mta_model("MTA_CHARGE_PT")),
                               &error);
  REQUIRE(pot != nullptr);
  CHECK(e.set_charge_spin(pot, -1, 1) == 0);
  CHECK(pair_energy(e, pot) == Catch::Approx(-7.5));
  CHECK(e.set_charge_spin(pot, 1, 3) == 3);
  CHECK(error_of(e, pot).find("spin") != std::string::npos);
  CHECK(pair_energy(e, pot) == Catch::Approx(-7.5));
  e.destroy(pot);

  // Reads the multiplicity only: the charge must stay 0.
  pot = create(e, metatomic_params(mta_model("MTA_SPIN_PT")), &error);
  REQUIRE(pot != nullptr);
  CHECK(e.set_charge_spin(pot, -1, 3) == 3);
  CHECK(error_of(e, pot).find("charge") != std::string::npos);
  CHECK(pair_energy(e, pot) == Catch::Approx(3.5));
  e.destroy(pot);

  // The same holds for the values in MetatomicParams: create fails.
  for (const auto &[key, charge, spin, word] :
       {std::tuple{"MTA_PLAIN_PT", -1, 1, "charge"},
        std::tuple{"MTA_CHARGE_PT", 0, 2, "spin"},
        std::tuple{"MTA_SPIN_PT", 1, 1, "charge"}}) {
    INFO(key << " charge " << charge << " spin " << spin);
    error.clear();
    pot = create(e, metatomic_params(mta_model(key), charge, spin), &error);
    CHECK(pot == nullptr);
    CHECK(error.find(word) != std::string::npos);
    if (pot != nullptr) {
      e.destroy(pot);
    }
  }
}
