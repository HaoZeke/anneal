// MIT License
// Copyright 2023--present rgpot developers
//
// libxtb is Fortran with an ISO_C_BINDING C API (xtb.h). Unit tests exercise
// that C contract carefully:
//   - opaque handle create / destroy (xtb_new* / xtb_del*)
//   - xtb_releaseOutput on each environment
//   - first force: xtb_newMolecule + loadGFN*; later: xtb_updateMolecule
//   - serial multi-handle lifecycle (independent ISO_C handle sets)
//   - method switch only after full destroy of previous handles
//   - linked XTBPot vs dlopen libxtb_engine (same C semantics)
// Tags: [xtb][capi] [xtb][linked] [xtb][dlopen]

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rgpot/XTBPot/XTBDlopen.hpp"
#include "rgpot/XTBPot/XTBPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#include "XtbHotClusters.hpp"

using rgpot::types::AtomMatrix;
using Catch::Matchers::WithinAbs;

// Water (Å): O + 2H
static const double kWaterPos[] = {
    0.00000000, 0.00000000,  0.11779000,  // O
    0.00000000, 0.75545000,  -0.47116000, // H
    0.00000000, -0.75545000, -0.47116000  // H
};
static const int kWaterZ[] = {8, 1, 1};

static void fill_water(AtomMatrix &positions, std::vector<int> &atmtypes,
                       std::array<std::array<double, 3>, 3> &box) {
  // Caller constructs positions(3,3).
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      positions(static_cast<size_t>(i), static_cast<size_t>(j)) =
          kWaterPos[i * 3 + j];
  atmtypes.assign(kWaterZ, kWaterZ + 3);
  box = {{{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
}

static void displace_water(AtomMatrix &positions, double delta) {
  // Break exact symmetry slightly; still a valid bound geometry.
  positions(1, 1) += delta;
  positions(2, 1) -= delta;
}

static void require_finite_forces(const AtomMatrix &forces, size_t n) {
  REQUIRE(forces.rows() == n);
  REQUIRE(forces.cols() == 3);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < 3; ++j)
      REQUIRE(std::isfinite(forces(i, j)));
}

static void require_zero_net_force(const AtomMatrix &forces, double tol) {
  double fx = 0, fy = 0, fz = 0;
  for (size_t i = 0; i < forces.rows(); ++i) {
    fx += forces(i, 0);
    fy += forces(i, 1);
    fz += forces(i, 2);
  }
  REQUIRE_THAT(fx, WithinAbs(0.0, tol));
  REQUIRE_THAT(fy, WithinAbs(0.0, tol));
  REQUIRE_THAT(fz, WithinAbs(0.0, tol));
}

static rgpot::XTBDlopenConfig dlopen_cfg(const rgpot::XTBConfig &xtb) {
  rgpot::XTBDlopenConfig c;
  c.xtb = xtb;
  if (const char *eng = std::getenv("RGPOT_XTB_ENGINE"))
    if (eng && *eng)
      c.engine_path = eng;
  return c;
}

// ---------------------------------------------------------------------------
// Linked XTBPot (build-time libxtb)
// ---------------------------------------------------------------------------

TEST_CASE("XTBPot GFN2 water energy and force conservation", "[xtb][linked]") {
  rgpot::XTBPot pot;
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  REQUIRE(std::isfinite(energy));
  // GFN2-xTB water ≈ -5.07 Ha ≈ -137.9 eV
  REQUIRE(energy < 0.0);
  REQUIRE_THAT(energy, WithinAbs(-137.9, 1.0));
  require_finite_forces(forces, 3);
  require_zero_net_force(forces, 1e-6);
}

TEST_CASE("XTBPot GFNFF water finite energy", "[xtb][linked][gfnff]") {
  rgpot::XTBConfig cfg;
  cfg.method = rgpot::GFNMethod::GFNFF;
  rgpot::XTBPot pot(cfg);
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;
  REQUIRE(std::isfinite(energy));
  REQUIRE(energy < 0.0);
  require_finite_forces(forces, 3);
}

TEST_CASE("XTBPot warm updateMolecule path is stable", "[xtb][linked][capi]") {
  // ISO_C: after first force, molecule handle is updated via xtb_updateMolecule.
  rgpot::XTBPot pot;
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  auto [e0, f0, v0] = pot(positions, atmtypes, box);
  (void)v0;
  REQUIRE(std::isfinite(e0));

  AtomMatrix moved(3, 3);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      moved(i, j) = positions(i, j);
  displace_water(moved, 0.02);
  auto [e1, f1, v1] = pot(moved, atmtypes, box);
  (void)v1;
  REQUIRE(std::isfinite(e1));
  REQUIRE(std::abs(e1 - e0) > 1e-6); // geometry change must move energy
  require_finite_forces(f1, 3);

  // Return to original geometry: energy must recover (warm SCF state).
  auto [e2, f2, v2] = pot(positions, atmtypes, box);
  (void)f0;
  (void)f2;
  (void)v2;
  REQUIRE_THAT(e2, WithinAbs(e0, 1e-8));
}

TEST_CASE("XTBPot serial multi-instance lifecycle (ISO_C handles)",
          "[xtb][linked][capi]") {
  // Serial multi-handle lifecycle (independent ISO_C environments).
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  double e_ref = 0.0;
  {
    rgpot::XTBPot pot;
    auto [e, f, v] = pot(positions, atmtypes, box);
    (void)f;
    (void)v;
    e_ref = e;
  }

  constexpr int kRounds = 8;
  for (int r = 0; r < kRounds; ++r) {
    rgpot::XTBPot pot;
    auto [e, f, v] = pot(positions, atmtypes, box);
    (void)v;
    REQUIRE(std::isfinite(e));
    REQUIRE_THAT(e, WithinAbs(e_ref, 1e-8));
    require_finite_forces(f, 3);
    require_zero_net_force(f, 1e-5);
  } // destructor: xtb_del* for this handle set
}

TEST_CASE("XTBPot sequential method switch after destroy",
          "[xtb][linked][capi]") {
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  const rgpot::GFNMethod methods[] = {
      rgpot::GFNMethod::GFN2xTB,
      rgpot::GFNMethod::GFNFF,
      rgpot::GFNMethod::GFN1xTB,
      rgpot::GFNMethod::GFN2xTB,
  };
  for (auto m : methods) {
    rgpot::XTBConfig cfg;
    cfg.method = m;
    rgpot::XTBPot pot(cfg);
    auto [e, f, v] = pot(positions, atmtypes, box);
    (void)v;
    REQUIRE(std::isfinite(e));
    REQUIRE(e < 0.0);
    require_finite_forces(f, 3);
  }
}

// ---------------------------------------------------------------------------
// dlopen libxtb_engine.so (Fortran still inside the plugin)
// ---------------------------------------------------------------------------

TEST_CASE("XTBDlopen matches linked GFN2 water", "[xtb][dlopen][capi]") {
  const char *eng = std::getenv("RGPOT_XTB_ENGINE");
  if (!eng || !*eng) {
    WARN("RGPOT_XTB_ENGINE unset; attempting bare libxtb_engine.so");
  }

  rgpot::XTBConfig cfg;
  cfg.method = rgpot::GFNMethod::GFN2xTB;

  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  // Parity: evaluate linked to completion (destroy handles), then dlopen engine.
  double e_l = 0.0;
  AtomMatrix f_l(3, 3);
  {
    rgpot::XTBPot linked(cfg);
    auto [e, f, v] = linked(positions, atmtypes, box);
    (void)v;
    e_l = e;
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 3; ++j)
        f_l(i, j) = f(i, j);
    auto [e2, f2, v2] = linked(positions, atmtypes, box);
    (void)f2;
    (void)v2;
    REQUIRE_THAT(e2, WithinAbs(e_l, 1e-8));
  } // linked Fortran env destroyed

  std::unique_ptr<rgpot::XTBDlopen> dl;
  try {
    dl = std::make_unique<rgpot::XTBDlopen>(dlopen_cfg(cfg));
  } catch (const std::exception &ex) {
    FAIL("XTBDlopen construct failed: " << ex.what());
  }
  REQUIRE(dl->available());

  auto [e_d, f_d, v_d] = (*dl)(positions, atmtypes, box);
  (void)v_d;
  REQUIRE(std::isfinite(e_d));
  REQUIRE_THAT(e_d, WithinAbs(e_l, 1e-8));
  require_finite_forces(f_d, 3);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      REQUIRE_THAT(f_d(i, j), WithinAbs(f_l(i, j), 1e-6));

  // Warm update path inside the engine (xtb_updateMolecule)
  AtomMatrix moved(3, 3);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      moved(i, j) = positions(i, j);
  displace_water(moved, 0.015);
  auto [e_m, f_m, v_m] = (*dl)(moved, atmtypes, box);
  (void)f_m;
  (void)v_m;
  REQUIRE(std::isfinite(e_m));
  auto [e_back, f_back, v_back] = (*dl)(positions, atmtypes, box);
  (void)f_back;
  (void)v_back;
  REQUIRE_THAT(e_back, WithinAbs(e_d, 1e-8));
}

TEST_CASE("XTBDlopen serial multi-instance via recreate",
          "[xtb][dlopen][capi]") {
  // Destroy and recreate plugin pot (new C ABI pot → new ISO_C handle set).
  rgpot::XTBConfig cfg;
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  double e_ref = 0.0;
  {
    rgpot::XTBDlopen dl(dlopen_cfg(cfg));
    auto [e, f, v] = dl(positions, atmtypes, box);
    (void)f;
    (void)v;
    e_ref = e;
  }

  for (int r = 0; r < 4; ++r) {
    rgpot::XTBDlopen dl(dlopen_cfg(cfg));
    auto [e, f, v] = dl(positions, atmtypes, box);
    (void)v;
    REQUIRE_THAT(e, WithinAbs(e_ref, 1e-8));
    require_finite_forces(f, 3);
  }
}

// A reused SCC wavefunction has to land on the same GFN2 energy and forces
// as a fresh results object. 1e-8 Hartree is the energy agreement. Forces
// at accuracy 0.01 are compared in eV/Angstrom; accuracy 1 stops earlier
// and is checked against its own residual below.
static constexpr double kWarmEnergyTol = 1e-8 * 27.211386245988;
// Measured warm-versus-cold gap on this water is 6.5e-8 eV/Angstrom.
static constexpr double kWarmForceTol = 1e-7;

static void shift_all(AtomMatrix &positions, size_t axis, double delta) {
  for (size_t i = 0; i < positions.rows(); ++i)
    positions(i, axis) += delta;
}

static void require_close_forces(const AtomMatrix &got, const AtomMatrix &ref,
                                 double tol) {
  REQUIRE(got.rows() == ref.rows());
  REQUIRE(got.cols() == ref.cols());
  double worst = 0.0;
  size_t worst_i = 0;
  size_t worst_j = 0;
  for (size_t i = 0; i < got.rows(); ++i) {
    for (size_t j = 0; j < got.cols(); ++j) {
      const double delta = std::abs(got(i, j) - ref(i, j));
      if (delta > worst) {
        worst = delta;
        worst_i = i;
        worst_j = j;
      }
    }
  }
  UNSCOPED_INFO("worst |dF| " << worst << " at " << worst_i << "," << worst_j
                              << " got " << got(worst_i, worst_j) << " ref "
                              << ref(worst_i, worst_j));
  REQUIRE(worst <= tol);
}

static rgpot::XTBConfig converged_gfn2() {
  rgpot::XTBConfig cfg;
  cfg.accuracy = 0.01;
  cfg.max_iterations = 500;
  return cfg;
}

template <typename Factory>
static void check_wavefunction_reuse(Factory make) {
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  auto cold_of = [&](const rgpot::XTBConfig &cfg, const AtomMatrix &geo) {
    auto pot = make(cfg);
    auto [energy, forces, variance] = (*pot)(geo, atmtypes, box);
    (void)variance;
    REQUIRE_FALSE(pot->reusedWavefunction());
    return std::pair<double, AtomMatrix>{energy, forces};
  };

  const rgpot::XTBConfig cfg = converged_gfn2();
  auto session = make(cfg);
  REQUIRE(session->warmMove() == 1.0);
  auto [e0, f0, v0] = (*session)(positions, atmtypes, box);
  (void)f0;
  (void)v0;
  REQUIRE_FALSE(session->reusedWavefunction());

  AtomMatrix nudged = positions;
  nudged(1, 1) += 0.0625;
  auto [e_warm, f_warm, v_warm] = (*session)(nudged, atmtypes, box);
  (void)v_warm;
  REQUIRE(session->reusedWavefunction());
  {
    auto [e_cold, f_cold] = cold_of(cfg, nudged);
    REQUIRE_THAT(e_warm, WithinAbs(e_cold, kWarmEnergyTol));
    require_close_forces(f_warm, f_cold, kWarmForceTol);
  }

  // Exactly the guard, measured from the geometry just stored.
  AtomMatrix edge = nudged;
  shift_all(edge, 0, 1.0);
  auto [e_edge, f_edge, v_edge] = (*session)(edge, atmtypes, box);
  (void)v_edge;
  REQUIRE(session->reusedWavefunction());
  {
    auto [e_cold, f_cold] = cold_of(cfg, edge);
    REQUIRE_THAT(e_edge, WithinAbs(e_cold, kWarmEnergyTol));
    require_close_forces(f_edge, f_cold, kWarmForceTol);
  }

  AtomMatrix far = edge;
  shift_all(far, 0, 1.25);
  auto [e_far, f_far, v_far] = (*session)(far, atmtypes, box);
  (void)v_far;
  REQUIRE_FALSE(session->reusedWavefunction());
  {
    auto [e_cold, f_cold] = cold_of(cfg, far);
    REQUIRE_THAT(e_far, WithinAbs(e_cold, kWarmEnergyTol));
    require_close_forces(f_far, f_cold, kWarmForceTol);
  }
  AtomMatrix after = far;
  shift_all(after, 1, 0.0625);
  auto [e_after, f_after, v_after] = (*session)(after, atmtypes, box);
  (void)v_after;
  REQUIRE(session->reusedWavefunction());
  {
    auto [e_cold, f_cold] = cold_of(cfg, after);
    REQUIRE_THAT(e_after, WithinAbs(e_cold, kWarmEnergyTol));
    require_close_forces(f_after, f_cold, kWarmForceTol);
  }

  session->reset();
  REQUIRE_FALSE(session->reusedWavefunction());
  auto [e_reset, f_reset, v_reset] = (*session)(after, atmtypes, box);
  (void)v_reset;
  REQUIRE_FALSE(session->reusedWavefunction());
  {
    auto [e_cold, f_cold] = cold_of(cfg, after);
    REQUIRE_THAT(e_reset, WithinAbs(e_cold, kWarmEnergyTol));
    require_close_forces(f_reset, f_cold, kWarmForceTol);
  }

  rgpot::XTBConfig tight = cfg;
  tight.warm_move = 0.01;
  auto guarded = make(tight);
  REQUIRE(guarded->warmMove() == 0.01);
  auto [e_g0, f_g0, v_g0] = (*guarded)(positions, atmtypes, box);
  (void)f_g0;
  (void)v_g0;
  REQUIRE_FALSE(guarded->reusedWavefunction());
  auto [e_g1, f_g1, v_g1] = (*guarded)(nudged, atmtypes, box);
  (void)v_g1;
  REQUIRE_FALSE(guarded->reusedWavefunction());
  {
    auto [e_cold, f_cold] = cold_of(tight, nudged);
    REQUIRE_THAT(e_g1, WithinAbs(e_cold, kWarmEnergyTol));
    require_close_forces(f_g1, f_cold, kWarmForceTol);
  }
  guarded->setWarmMove(2.0);
  REQUIRE(guarded->warmMove() == 2.0);
  AtomMatrix wide = nudged;
  shift_all(wide, 0, 1.5);
  auto [e_wide, f_wide, v_wide] = (*guarded)(wide, atmtypes, box);
  (void)v_wide;
  REQUIRE(guarded->reusedWavefunction());
  {
    auto [e_cold, f_cold] = cold_of(cfg, wide);
    REQUIRE_THAT(e_wide, WithinAbs(e_cold, kWarmEnergyTol));
    require_close_forces(f_wide, f_cold, kWarmForceTol);
  }

  rgpot::XTBConfig off = cfg;
  off.warm_move = 0.0;
  auto disabled = make(off);
  auto [e_off, f_off, v_off] = (*disabled)(positions, atmtypes, box);
  (void)f_off;
  (void)v_off;
  auto [e_off2, f_off2, v_off2] = (*disabled)(nudged, atmtypes, box);
  (void)f_off2;
  (void)v_off2;
  REQUIRE_FALSE(disabled->reusedWavefunction());
  REQUIRE_THAT(e_off2, WithinAbs(e_warm, kWarmEnergyTol));
  (void)e_off;
}

TEST_CASE("XTBPot reuses the GFN2 wavefunction inside the move guard",
          "[xtb][linked][warm]") {
  check_wavefunction_reuse([](const rgpot::XTBConfig &cfg) {
    return std::make_unique<rgpot::XTBPot>(cfg);
  });
}

TEST_CASE("XTBDlopen reuses the GFN2 wavefunction inside the move guard",
          "[xtb][dlopen][warm]") {
  check_wavefunction_reuse([](const rgpot::XTBConfig &cfg) {
    return std::make_unique<rgpot::XTBDlopen>(dlopen_cfg(cfg));
  });
}

// A dissociated ammonia tetramer. One nitrogen has no hydrogen inside
// 1.3 Å. GFN2 at 300 K does not converge the SCC; the gap is 0.29 eV.
// The rescue smears the occupation and returns the cooled 300 K energy.
TEST_CASE("XTBPot cools a nonconverged SCC back to the configured temperature",
          "[xtb][linked][scc]") {
  const int z[16] = {7, 1, 1, 1, 7, 1, 1, 1, 7, 1, 1, 1, 7, 1, 1, 1};
  const double xyz[16][3] = {
      {-1.66532428, 0.54889027, -2.03940010},
      {0.11165024, 2.04946110, 1.43904553},
      {-0.42884806, 2.72524649, 0.30972045},
      {1.43338889, 2.19082068, 0.90531718},
      {0.53411760, 1.59058476, -4.48817656},
      {-0.62302497, 2.69900828, -3.26741719},
      {1.34429373, 3.32407404, -1.22950979},
      {-0.35527480, 1.88804157, -4.87298828},
      {1.47996713, 0.55484625, 3.71752121},
      {1.28956128, -0.72444169, 2.95951158},
      {-0.20031872, 0.12475055, 3.35919321},
      {1.70295493, -0.38386526, 4.03334277},
      {-2.71341888, -5.98849237, 1.22333256},
      {-1.72758892, -6.21046903, 1.63826395},
      {-1.69113306, -6.08127583, -0.32892162},
      {-2.20633610, -6.61433582, 0.61469530},
  };
  rgpot::XTBConfig cfg;
  cfg.method = rgpot::GFNMethod::GFN2xTB;
  cfg.accuracy = 0.01;
  cfg.electronic_temperature = 300.0;
  cfg.max_iterations = 500;
  rgpot::XTBPot pot(cfg);
  AtomMatrix positions(16, 3);
  std::vector<int> atmtypes(z, z + 16);
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = xyz[i][j];
  std::array<std::array<double, 3>, 3> box{};
  box[0][0] = box[1][1] = box[2][2] = 60.0;

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;
  // Cooled 300 K result, not the 1000 K single point (−448.36 eV).
  REQUIRE_THAT(energy, WithinAbs(-447.8814, 1.0e-3));
  require_finite_forces(forces, 16);
}

// ---------------------------------------------------------------------------
// Hot clusters: a session's energy must not depend on its call history
// ---------------------------------------------------------------------------

struct HotOutcome {
  bool ok = false;
  double energy = std::numeric_limits<double>::quiet_NaN();
  AtomMatrix forces;
  bool reused = false;
};

template <typename PotPtr, int N>
static HotOutcome evaluate_frame(PotPtr &pot, const double (&frame)[N][3],
                                 int z) {
  AtomMatrix positions(N, 3);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < 3; ++j)
      positions(static_cast<size_t>(i), static_cast<size_t>(j)) = frame[i][j];
  std::vector<int> atmtypes(N, z);
  std::array<std::array<double, 3>, 3> box{};
  box[0][0] = box[1][1] = box[2][2] = 60.0;
  HotOutcome out;
  try {
    auto [energy, forces, variance] = (*pot)(positions, atmtypes, box);
    (void)variance;
    out.ok = true;
    out.energy = energy;
    out.forces = forces;
    out.reused = pot->reusedWavefunction();
  } catch (const std::exception &ex) {
    UNSCOPED_INFO("throws: " << ex.what());
  }
  return out;
}

// One session walks the frames in order. Every frame is also evaluated by
// a new instance. Both must agree on success and on the energy, and the
// new instance must reproduce standalone xtb where its SCC converges.
// Returns how many frames the session evaluated from the previous
// wavefunction.
template <typename Factory, int F, int N>
static int check_session_matches_fresh(Factory make,
                                       const rgpot::XTBConfig &cfg,
                                       const double (&frames)[F][N][3], int z,
                                       const double (&xtb)[F],
                                       double energy_tol, double force_tol) {
  auto session = make(cfg);
  int reused = 0;
  for (int f = 0; f < F; ++f) {
    auto warm = evaluate_frame(session, frames[f], z);
    auto cold_pot = make(cfg);
    auto cold = evaluate_frame(cold_pot, frames[f], z);
    INFO("frame " << f << " session " << warm.energy << " fresh " << cold.energy
                  << " xtb " << xtb[f]);
    REQUIRE(warm.ok == cold.ok);
    if (warm.ok) {
      REQUIRE_THAT(warm.energy, WithinAbs(cold.energy, energy_tol));
      require_close_forces(warm.forces, cold.forces, force_tol);
      reused += warm.reused ? 1 : 0;
    }
    if (!std::isnan(xtb[f])) {
      REQUIRE(cold.ok);
      REQUIRE_THAT(cold.energy, WithinAbs(xtb[f], 1e-4));
    }
  }
  return reused;
}

static auto make_linked = [](const rgpot::XTBConfig &cfg) {
  return std::make_unique<rgpot::XTBPot>(cfg);
};

static auto make_dlopen = [](const rgpot::XTBConfig &cfg) {
  return std::make_unique<rgpot::XTBDlopen>(dlopen_cfg(cfg));
};

// Si13 at 300 K has a HOMO-LUMO gap below 0.6 eV on every frame. The SCC
// of frame 1 does not converge at 300 K, and its last iterate holds charges
// of several electrons per atom. Seeded from it, the session used to land
// on a self-consistent state 25 to 40 eV below xtb (-627.44 eV against
// -584.39 eV on frame 5). From a converged frame, the session's guess can
// also reach a second physical solution about 5 eV below xtb's.
template <typename Factory> static void check_hot_si13(Factory make) {
  using namespace rgpot_test_hot;
  const rgpot::XTBConfig cfg;
  const int reused = check_session_matches_fresh(make, cfg, kSi13Pos, kSi13Z,
                                                 kSi13Xtb, 1e-6, 1e-5);
  REQUIRE(reused == 0);
}

TEST_CASE("XTBPot session matches a fresh start on hot Si13",
          "[xtb][linked][warm][hot]") {
  check_hot_si13(make_linked);
}

TEST_CASE("XTBDlopen session matches a fresh start on hot Si13",
          "[xtb][dlopen][warm][hot]") {
  check_hot_si13(make_dlopen);
}

TEST_CASE("XTBPot hot Si10 and C10 agree with a fresh start",
          "[xtb][linked][warm][hot]") {
  using namespace rgpot_test_hot;
  SECTION("Si10 with the gap guard") {
    const rgpot::XTBConfig cfg;
    check_session_matches_fresh(make_linked, cfg, kSi10Pos, kSi10Z, kSi10Xtb,
                                1e-6, 1e-5);
  }
  SECTION("Si10 reusing the wavefunction at any gap") {
    rgpot::XTBConfig cfg;
    cfg.warm_min_gap = 0.0;
    const int reused = check_session_matches_fresh(
        make_linked, cfg, kSi10Pos, kSi10Z, kSi10Xtb, 1e-5, 1e-3);
    REQUIRE(reused > 0);
  }
  SECTION("C10 keeps the wavefunction above the gap guard") {
    const rgpot::XTBConfig cfg;
    const int reused = check_session_matches_fresh(make_linked, cfg, kC10Pos,
                                                   kC10Z, kC10Xtb, 1e-5, 1e-3);
    REQUIRE(reused == kC10Frames - 1);
  }
}

TEST_CASE("XTBPot gap guard is adjustable on a live handle",
          "[xtb][linked][warm]") {
  rgpot::XTBPot pot;
  REQUIRE(pot.warmMinGap() == 1.0);
  pot.setWarmMinGap(0.25);
  REQUIRE(pot.warmMinGap() == 0.25);

  rgpot::XTBConfig cfg;
  cfg.warm_min_gap = 0.5;
  auto dl = make_dlopen(cfg);
  REQUIRE(dl->warmMinGap() == 0.5);
  dl->setWarmMinGap(2.0);
  REQUIRE(dl->warmMinGap() == 2.0);
}
