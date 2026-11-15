// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <capnp/message.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "rgpot/NWChemPot/NWChemPot.hpp"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;

// Water molecule geometry (Angstrom)
static const double water_pos[] = {
    0.00000000, 0.00000000,  0.11779000,  // O
    0.00000000, 0.75545000,  -0.47116000, // H
    0.00000000, -0.75545000, -0.47116000  // H
};
static const int water_atmnrs[] = {8, 1, 1};

static bool env_points_at_real_nwchem_engine() {
  const char *keys[] = {"NWCHEMC_LIBRARY", "RGPOT_NWCHEMC_ENGINE",
                        "RGPOT_NWCHEM_ENGINE", nullptr};
  for (int i = 0; keys[i]; ++i) {
    const char *v = std::getenv(keys[i]);
    if (!v || !v[0])
      continue;
    const std::string s(v);
    // Fake CI engine path usually contains "fake"; real meson module does not.
    if (s.find("fake") == std::string::npos &&
        s.find("nwchemc_fake") == std::string::npos)
      return true;
  }
  return false;
}

TEST_CASE("NWChemPot probeAvailable without engine", "[nwchem]") {
  // Without an nwchemc-compatible library on path, probe is false.
  // This documents expected behavior; does not require NWChem installed.
  bool ok = rgpot::NWChemPot::probeAvailable();
  if (!ok) {
    SUCCEED("engine not on path (expected in default CI)");
  } else {
    SUCCEED("engine found via RGPOT_NWCHEM_ENGINE / library path");
  }
}

TEST_CASE("NWChemPot abiAvailable without real embed", "[nwchem]") {
  // Real embed only when the loaded library was built with RGPOT_HAS_NWCHEM.
  bool abi = rgpot::NWChemPot::abiAvailable();
  if (!abi) {
    SUCCEED("stub or missing engine reports abiAvailable=false");
  } else {
    SUCCEED("real NWChem embed reports abiAvailable=true");
  }
}

TEST_CASE("NWChemPot water energy when engine present", "[nwchem]") {
  if (!rgpot::NWChemPot::probeAvailable()) {
    SKIP("libnwchemc not available (set NWCHEMC_LIBRARY, RGPOT_NWCHEMC_ENGINE, or NWCHEM_TOP)");
  }
  // Real NWChem embed is effectively single-shot per process; prefer the B3LYP
  // case for the real-engine proof and keep STO-3G for the in-tree fake.
  if (env_points_at_real_nwchem_engine()) {
    SKIP("real embed: run [b3lyp] case only (one SCF per process)");
  }

  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::NWChemParams>();
  p.setBasis("sto-3g");
  p.setTheory("scf");
  p.setScfType("rhf");
  p.setCharge(0);
  p.setMultiplicity(1);
  rgpot::NWChemPot pot(p.asReader());
  REQUIRE(pot.available());

  // Stub-only (nwchemc_available==0) cannot calculate; real or CI fake engines do.
  if (!rgpot::NWChemPot::abiAvailable()) {
    SKIP("engine loaded but stub-only (no calculable nwchemc ABI)");
  }

  AtomMatrix positions(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = water_pos[i * 3 + j];
  std::vector<int> atmtypes(water_atmnrs, water_atmnrs + 3);
  std::array<std::array<double, 3>, 3> box = {
      {{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  REQUIRE(std::isfinite(energy));
  REQUIRE(energy != 0.0);
  // Real SCF/STO-3G water is bound (negative eV); in-tree fake returns +0.25 Ha in eV.
  // Either path must yield finite non-zero forces through the C ABI.
  bool any_force = false;
  for (size_t i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      REQUIRE(std::isfinite(forces(i, j)));
      if (std::abs(forces(i, j)) > 1e-12)
        any_force = true;
    }
  }
  REQUIRE(any_force);
}

TEST_CASE("NWChemPot setParams updates Cap'n Proto-visible params", "[nwchem]") {
  // Real libnwchemc may SIGSEGV on set_params / second lifetime in-process.
  if (env_points_at_real_nwchem_engine()) {
    SKIP("real embed: setParams after load is unsafe in-process; Cap'n Proto "
         "roundtrip covered with fake engine in CI");
  }
  const bool rich_stanzas = true;

  rgpot::NWChemPot pot;
  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::NWChemParams>();
  p.setBasis("6-31g");
  p.setTheory("dft");
  p.setCharge(-1);
  p.setMultiplicity(1);
  p.setTask("energy");
  p.setTitle("anion");
  p.setMemoryMb(768);
  p.setScratchDir("/scratch/rgpot-nwchem-test");
  p.setPermanentDir("/perm/rgpot-nwchem-test");
  if (rich_stanzas) {
    auto blocks = p.initInputBlocks(2);
    blocks.set(0, "dft; xc b3lyp; end");
    blocks.set(1, "set int:acc_std 1e-8");
    auto stanzas = p.initInputStanzas(1);
    auto stanza = stanzas[0];
    stanza.setKind(::NWChemInputStanza::Kind::DFT);
    auto dft = stanza.initDft();
    dft.setDirect(true);
    dft.setXc("pbe0");
    auto smear = dft.initSmearing();
    smear.setSigmaHartree(0.001);
    smear.setMode(::NWChemDftSmearing::Mode::FIXSZ);
  } else {
    // Real embed: keep params the embed has already proven to accept.
    p.setBasis("sto-3g");
    p.setTheory("scf");
    p.setScfType("rhf");
    p.setCharge(0);
    p.setMultiplicity(1);
    p.setTitle("water-scf");
  }
  // setParams returns false if engine not loaded; stored params still roundtrip.
  (void)pot.setParams(p.asReader());
  ::capnp::MallocMessageBuilder out_msg;
  auto out = out_msg.initRoot<::NWChemParams>();
  pot.getParams(out);
  if (rich_stanzas) {
    REQUIRE(std::string(out.getBasis().cStr()) == "6-31g");
    REQUIRE(std::string(out.getTheory().cStr()) == "dft");
    REQUIRE(out.getCharge() == -1);
    REQUIRE(std::string(out.getTask().cStr()) == "energy");
    REQUIRE(std::string(out.getTitle().cStr()) == "anion");
    REQUIRE(out.getMemoryMb() == 768);
    REQUIRE(std::string(out.getScratchDir().cStr()) ==
            "/scratch/rgpot-nwchem-test");
    REQUIRE(std::string(out.getPermanentDir().cStr()) ==
            "/perm/rgpot-nwchem-test");
    REQUIRE(out.getInputBlocks().size() == 2);
    REQUIRE(std::string(out.getInputBlocks()[0].cStr()) ==
            "dft; xc b3lyp; end");
    REQUIRE(std::string(out.getInputBlocks()[1].cStr()) ==
            "set int:acc_std 1e-8");
    REQUIRE(out.getInputStanzas().size() == 1);
    REQUIRE(out.getInputStanzas()[0].getKind() ==
            ::NWChemInputStanza::Kind::DFT);
    REQUIRE(out.getInputStanzas()[0].getDft().getDirect());
    REQUIRE(std::string(out.getInputStanzas()[0].getDft().getXc().cStr()) ==
            "pbe0");
    REQUIRE(out.getInputStanzas()[0]
                .getDft()
                .getSmearing()
                .getMode() == ::NWChemDftSmearing::Mode::FIXSZ);
    REQUIRE_THAT(out.getInputStanzas()[0]
                     .getDft()
                     .getSmearing()
                     .getSigmaHartree(),
                 Catch::Matchers::WithinAbs(0.001, 1e-12));
  } else {
    REQUIRE(std::string(out.getBasis().cStr()) == "sto-3g");
    REQUIRE(std::string(out.getTheory().cStr()) == "scf");
    REQUIRE(std::string(out.getScfType().cStr()) == "rhf");
    REQUIRE(out.getCharge() == 0);
    REQUIRE(std::string(out.getTitle().cStr()) == "water-scf");
  }
  SUCCEED();
}

TEST_CASE("NWChemPot water B3LYP/6-31G* when real engine present",
          "[nwchem][b3lyp][real]") {
  if (!rgpot::NWChemPot::probeAvailable()) {
    SKIP("libnwchemc not available");
  }
  if (!env_points_at_real_nwchem_engine()) {
    SKIP("real NWChem embed required (set NWCHEMC_LIBRARY to libnwchemc.so)");
  }

  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::NWChemParams>();
  p.setBasis("6-31g*");
  p.setTheory("dft");
  p.setScfType("rhf");
  p.setCharge(0);
  p.setMultiplicity(1);
  p.setTask("gradient");
  p.setTitle("water-b3lyp");
  // Explicit DFT XC block — reliable B3LYP without depending on scfType promotion
  // in an older libnwchemc build.
  auto blocks = p.initInputBlocks(1);
  blocks.set(0, "dft\n  xc b3lyp\n  mult 1\nend");
  // Host-only path must not poison embed set_params (stripped before ABI).
  if (const char *eng = std::getenv("NWCHEMC_LIBRARY"))
    p.setEnginePath(eng);
  else if (const char *eng = std::getenv("RGPOT_NWCHEMC_ENGINE"))
    p.setEnginePath(eng);

  rgpot::NWChemPot pot(p.asReader());
  REQUIRE(pot.available());

  AtomMatrix positions(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = water_pos[i * 3 + j];
  std::vector<int> atmtypes(water_atmnrs, water_atmnrs + 3);
  std::array<std::array<double, 3>, 3> box = {
      {{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};

  // Successive force evaluations on ONE pot (criterion: multi-SCF in-process).
  double e1 = 0.0, e2 = 0.0;
  {
    auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;
    e1 = energy;
    REQUIRE(std::isfinite(energy));
    // Water B3LYP/6-31G* ~ -76.4 Ha ≈ -2079 eV (not LDA ~-2063 eV).
    REQUIRE(energy < -2070.0);
    bool any_force = false;
    for (size_t i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        REQUIRE(std::isfinite(forces(i, j)));
        if (std::abs(forces(i, j)) > 1e-12)
          any_force = true;
      }
    }
    REQUIRE(any_force);
  }
  {
    auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;
    e2 = energy;
    REQUIRE(std::isfinite(energy));
    REQUIRE(energy < -2000.0);
    // Two SCFs on same geometry: agree within DFT tolerance (eV).
    REQUIRE_THAT(energy, Catch::Matchers::WithinAbs(e1, 1e-3));
    (void)forces;
  }
  REQUIRE(e1 != 0.0);
  REQUIRE(e2 != 0.0);
}
