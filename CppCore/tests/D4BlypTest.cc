// MIT License
// Copyright 2023--present rgpot developers
//
// rgpot::D4Pot with dftd4's BLYP parameters against the dftd4 program
// (4.2.0, `--func blyp --grad`) on a water dimer, benzene and an eight-water
// box in a cubic and a sheared cell, the cells run periodic; finite
// differences of the energy for the forces and the stress; and D4 as a
// composite term. scripts/regen_d4_references.py writes the references in
// CppCore/tests/data/d4/.

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "dftd_tolerances.hpp"
#include "rgpot/D2Pot/D2Pot.hpp"
#include "rgpot/D4Pot/D4Pot.hpp"
#include "rgpot/SumPot/SumPot.hpp"
#include "rgpot/units.hpp"

using rgpot::testtol::kEnergyThr;
using rgpot::testtol::kGradThr;
using rgpot::units::BOHR_TO_ANGSTROM;
using rgpot::units::HARTREE_PER_BOHR3_TO_EV_PER_ANGSTROM3;
using rgpot::units::HARTREE_TO_EV;

namespace {

constexpr const char *kSystems[] = {"water_dimer", "benzene", "water_box_cubic",
                                    "water_box_sheared"};

struct Reference {
  std::vector<int> z;
  std::vector<double> pos; // Angstrom
  std::array<double, 9> box{};
  bool periodic = false;
  double energy = 0.0;            // Ha
  std::vector<double> grad;       // Ha/bohr
  std::array<double, 9> virial{}; // Ha, dE/dstrain
};

Reference load(const std::string &name) {
  const std::string path = "CppCore/tests/data/d4/" + name + ".txt";
  std::ifstream in(path);
  INFO(path);
  REQUIRE(in.good());
  Reference r;
  std::string line;
  int row = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream ss(line);
    std::string key;
    ss >> key;
    if (key == "periodic") {
      int p = 0;
      ss >> p;
      r.periodic = p != 0;
    } else if (key == "box") {
      for (double &v : r.box) {
        ss >> v;
      }
    } else if (key == "atom") {
      int z = 0;
      double x = 0, y = 0, w = 0;
      ss >> z >> x >> y >> w;
      r.z.push_back(z);
      r.pos.insert(r.pos.end(), {x, y, w});
    } else if (key == "energy") {
      ss >> r.energy;
    } else if (key == "grad") {
      double a = 0, b = 0, c = 0;
      ss >> a >> b >> c;
      r.grad.insert(r.grad.end(), {a, b, c});
    } else if (key == "virial") {
      ss >> r.virial[3 * row] >> r.virial[3 * row + 1] >> r.virial[3 * row + 2];
      ++row;
    }
  }
  REQUIRE(r.grad.size() == r.pos.size());
  REQUIRE(row == (r.periodic ? 3 : 0));
  return r;
}

struct Result {
  double energy = 0.0; // eV
  std::vector<double> forces;
  std::array<double, 9> stress{};
  bool has_stress = false;
};

template <typename Pot>
Result run(Pot &&pot, const std::vector<int> &z, const std::vector<double> &pos,
           const std::array<double, 9> &box) {
  Result r;
  r.forces.assign(pos.size(), 0.0);
  const rgpot::ForceInput in{z.size(), pos.data(), z.data(), box.data()};
  rgpot::ForceOut out{};
  out.F = r.forces.data();
  pot.forceBatch(rgpot::ForceBatch{.nSystems = 1, .in = &in, .out = &out});
  r.energy = out.energy;
  r.has_stress = out.has_stress != 0;
  std::copy(out.stress, out.stress + 9, r.stress.begin());
  return r;
}

double volume(const std::array<double, 9> &h) {
  return std::abs(h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                  h[2] * (h[3] * h[7] - h[4] * h[6]));
}

/// Every image of a small molecule in this cell lies past dftd4's
/// real-space cutoffs, so the periodic evaluation is cheap and smooth.
constexpr std::array<double, 9> kWideCell{40.0, 0.0, 0.0, 5.0, 38.0, 0.0, -3.0, 4.0, 39.0};

rgpot::D4Config blyp(bool periodic) {
  return rgpot::D4Config{.functional = "blyp", .periodic = periodic};
}

} // namespace

TEST_CASE("D4 BLYP matches dftd4 energies, gradients and virials", "[d4][reference]") {
  for (const char *name : kSystems) {
    INFO(name);
    const Reference ref = load(name);
    rgpot::D4Pot pot(blyp(ref.periodic));
    REQUIRE(pot.caps().stress == ref.periodic);
    const Result r = run(pot, ref.z, ref.pos, ref.box);

    const double e_ha = r.energy / HARTREE_TO_EV;
    INFO("energy " << e_ha << " Ha, dftd4 " << ref.energy);
    REQUIRE(std::abs(e_ha - ref.energy) <= kEnergyThr);

    const double fconv = HARTREE_TO_EV / BOHR_TO_ANGSTROM;
    for (std::size_t k = 0; k < ref.grad.size(); ++k) {
      INFO("gradient component " << k);
      REQUIRE(std::abs(-r.forces[k] / fconv - ref.grad[k]) <= kGradThr);
    }

    REQUIRE(r.has_stress == ref.periodic);
    if (ref.periodic) {
      const double v = volume(ref.box) / std::pow(BOHR_TO_ANGSTROM, 3);
      for (int k = 0; k < 9; ++k) {
        const double ours = r.stress[k] / HARTREE_PER_BOHR3_TO_EV_PER_ANGSTROM3 * v;
        INFO("virial " << k << " ours " << ours << " dftd4 " << ref.virial[k]);
        // The virial is a gradient times a length of the cell.
        REQUIRE(std::abs(ours - ref.virial[k]) <= kGradThr * std::cbrt(v));
      }
    }
  }
}

template <typename Pot>
void requireForcesAreGradients(Pot &pot, const Reference &ref, const std::array<double, 9> &box) {
  const Result r = run(pot, ref.z, ref.pos, box);
  // Central differences: the error is step^2 times the third derivative
  // plus the rounding E eps / step.
  const double step = std::cbrt(std::numeric_limits<double>::epsilon());
  const double bar = kGradThr * HARTREE_TO_EV / BOHR_TO_ANGSTROM;
  for (std::size_t k = 0; k < ref.pos.size(); ++k) {
    std::vector<double> p = ref.pos;
    p[k] += step;
    const double ep = run(pot, ref.z, p, box).energy;
    p[k] -= 2.0 * step;
    const double em = run(pot, ref.z, p, box).energy;
    INFO("component " << k);
    REQUIRE(std::abs(-(ep - em) / (2.0 * step) - r.forces[k]) <= bar);
  }
}

TEST_CASE("D4 forces and stress are derivatives of its energy", "[d4][fd]") {
  {
    const Reference ref = load("water_dimer");
    rgpot::D4Pot pot(blyp(false));
    requireForcesAreGradients(pot, ref, ref.box);
  }
  // dftd4 drops pairs past its real-space cutoffs (60 bohr for the pair
  // term) without smoothing, so straining a dense cell moves image pairs
  // across them and the energy steps. kWideCell keeps every image beyond
  // the cutoffs while dftd4 still runs its periodic path, so the strain
  // derivative is smooth and checks the stress units, sign and volume.
  const Reference ref = load("water_dimer");
  const std::array<double, 9> box = kWideCell;
  rgpot::D4Pot pot(blyp(true));
  requireForcesAreGradients(pot, ref, box);
  const Result r = run(pot, ref.z, ref.pos, box);
  REQUIRE(r.has_stress);
  const double step = std::cbrt(std::numeric_limits<double>::epsilon());
  const double v = volume(box);
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      auto strained = [&](double e) {
        std::array<double, 9> h = box;
        std::vector<double> p = ref.pos;
        for (int i = 0; i < 3; ++i) {
          h[3 * i + b] += e * box[3 * i + a];
        }
        for (std::size_t n = 0; n < ref.z.size(); ++n) {
          p[3 * n + b] += e * ref.pos[3 * n + a];
        }
        return run(pot, ref.z, p, h).energy;
      };
      const double dE = (strained(step) - strained(-step)) / (2.0 * step);
      INFO("stress " << a << b << " fd " << dE << " ours " << v * r.stress[3 * a + b]);
      REQUIRE(std::abs(dE - v * r.stress[3 * a + b]) <=
              kGradThr * HARTREE_TO_EV / BOHR_TO_ANGSTROM * std::cbrt(v));
    }
  }
}

TEST_CASE("molecular D4 ignores the box and periodic D4 needs one", "[d4]") {
  const Reference ref = load("water_dimer");
  rgpot::D4Pot pot(blyp(false));
  REQUIRE_FALSE(pot.caps().periodic);
  const Result open = run(pot, ref.z, ref.pos, ref.box);
  const std::array<double, 9> wide{40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0};
  const Result boxed = run(pot, ref.z, ref.pos, wide);
  REQUIRE(std::abs(open.energy - boxed.energy) <= kEnergyThr * HARTREE_TO_EV);
  for (std::size_t k = 0; k < open.forces.size(); ++k) {
    REQUIRE(std::abs(open.forces[k] - boxed.forces[k]) <=
            kGradThr * HARTREE_TO_EV / BOHR_TO_ANGSTROM);
  }
  REQUIRE_FALSE(boxed.has_stress);

  rgpot::D4Pot periodic(blyp(true));
  REQUIRE_THROWS_AS(run(periodic, ref.z, ref.pos, ref.box), std::invalid_argument);
}

TEST_CASE("D4 is a composite term whose stress joins D2's", "[d4][composite]") {
  const Reference ref = load("water_dimer");
  const std::array<double, 9> box = kWideCell;
  rgpot::D4Pot d4(blyp(true));
  rgpot::D2Pot d2;
  const Result d4_only = run(d4, ref.z, ref.pos, box);
  const Result d2_only = run(d2, ref.z, ref.pos, box);

  rgpot::SumPot sum;
  sum.add({.name = "d2", .weight = -1.0}, std::make_unique<rgpot::D2Pot>());
  sum.add({.name = "d4"}, std::make_unique<rgpot::D4Pot>(blyp(true)));
  REQUIRE(sum.caps().stress);
  const Result swapped = run(sum, ref.z, ref.pos, box);
  const double scale = std::max(std::abs(d4_only.energy), std::abs(d2_only.energy));
  REQUIRE(std::abs(swapped.energy - (d4_only.energy - d2_only.energy)) <=
          kEnergyThr * HARTREE_TO_EV + scale * std::numeric_limits<double>::epsilon());
  REQUIRE(swapped.has_stress);
  for (int k = 0; k < 9; ++k) {
    REQUIRE(std::abs(swapped.stress[k] - (d4_only.stress[k] - d2_only.stress[k])) <=
            kGradThr * HARTREE_PER_BOHR3_TO_EV_PER_ANGSTROM3);
  }
  const auto components = sum.lastComponents();
  REQUIRE(components.size() == 2);
  REQUIRE(std::abs(components[1].energy - d4_only.energy) <= kEnergyThr * HARTREE_TO_EV);

  rgpot::SumPot molecular;
  molecular.add({.name = "d2"}, std::make_unique<rgpot::D2Pot>());
  molecular.add({.name = "d4"}, std::make_unique<rgpot::D4Pot>(blyp(false)));
  REQUIRE_FALSE(molecular.caps().stress);
  REQUIRE_FALSE(molecular.caps().periodic);
}
