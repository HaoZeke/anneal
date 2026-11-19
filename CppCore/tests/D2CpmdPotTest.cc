// MIT License
// Copyright 2023--present rgpot developers
//
// rgpot::D2CpmdPot against OpenCPMD's own empirical D2 (062582b, ALL DFT-D2
// with BLYP) on water and methane in cubic, orthorhombic and triclinic
// cells, including a pair on either side of half the cell and pairs on
// either side of CPMD's 6 R0 truncation; finite differences for its forces
// and stress; the cusp at half the cell that rgpot::D2Pot does not have;
// and the composite that swaps CPMD's term for the lattice sum.
// scripts/regen_d2_cpmd_references.py writes the water and methane files.
// scripts/regen_d2_cpmd_pbc3.py writes the pbc3 symmetry-cell files.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "dftd_tolerances.hpp"
#include "rgpot/D2Pot/D2CpmdPot.hpp"
#include "rgpot/D2Pot/D2Pot.hpp"
#ifdef RGPOT_HAS_CACHE
#include "rgpot/PotentialCache.hpp"
#include <rocksdb/options.h>
#endif
#include "rgpot/SumPot/SumPot.hpp"
#include "rgpot/units.hpp"

using rgpot::testtol::kEnergyThr;
using rgpot::testtol::kGradThr;
using rgpot::units::BOHR_TO_ANGSTROM;
using rgpot::units::HARTREE_TO_EV;

namespace {

// OpenCPMD 062582b7cfd832d36f88f504cd08e4ead42eb404. Positions, the cell,
// the forces and the stress all use rgpot::kCpmdBohr, so a derivative of
// the returned energy recovers the force without a second bohr factor.
constexpr const char *kOpenCpmdPbc3 = "062582b7cfd832d36f88f504cd08e4ead42eb404";

struct Reference {
  std::vector<int> z;
  std::vector<double> pos; // Angstrom
  std::array<double, 9> box{};
  double energy = 0.0;  // Ha
  double printed = 0.0; // Ha, CPMD's VDW ENERGY line
  std::vector<double> force; // Ha/bohr
  int ibrav = 0;
  bool differs = false;     // pbc3 energy is not the Euclidean image's
  double euclidean = 0.0;   // Ha, Grimme energy of that Euclidean image
  double nearest = 0.0;     // Ha, Grimme energy of the nearest integer image
  std::string version;
};

Reference load(const std::string &name) {
  const std::string path = "CppCore/tests/data/d2cpmd/" + name + ".txt";
  std::ifstream in(path);
  INFO(path);
  REQUIRE(in.good());
  Reference r;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream ss(line);
    std::string key;
    ss >> key;
    if (key == "box") {
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
    } else if (key == "printed") {
      ss >> r.printed;
    } else if (key == "force") {
      double a = 0, b = 0, c = 0;
      ss >> a >> b >> c;
      r.force.insert(r.force.end(), {a, b, c});
    } else if (key == "ibrav") {
      ss >> r.ibrav;
    } else if (key == "differs") {
      int flag = 0;
      ss >> flag;
      r.differs = flag != 0;
    } else if (key == "euclidean") {
      ss >> r.euclidean;
    } else if (key == "nearest") {
      ss >> r.nearest;
    } else if (key == "version") {
      ss >> r.version;
    }
  }
  REQUIRE(r.force.size() == r.pos.size());
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

const char *kSystems[] = {"cubic_water",         "cubic_below_half",
                          "cubic_above_half",    "ortho_water_methane",
                          "triclinic_water_methane", "cubic_truncation"};

// One cell for every code latgen builds that pbc3 handles. The first group
// can be read from the box; symmetry 12 has the shape of 14 and cannot.
const char *kPbc3Inferable[] = {"pbc3_hexagonal",
                                "pbc3_fcc",
                                "pbc3_monoclinic",
                                "pbc3_cubic",
                                "pbc3_orthorhombic",
                                "pbc3_bcc",
                                "pbc3_rhombohedral",
                                "pbc3_tetragonal",
                                "pbc3_body_centred_tetragonal"};

const char *kPbc3Monoclinic12[] = {"pbc3_monoclinic12", "pbc3_monoclinic12_skew"};

std::vector<const char *> all_pbc3() {
  std::vector<const char *> names(std::begin(kPbc3Inferable), std::end(kPbc3Inferable));
  names.insert(names.end(), std::begin(kPbc3Monoclinic12), std::end(kPbc3Monoclinic12));
  return names;
}

rgpot::D2CpmdConfig with_symmetry(int symmetry) {
  rgpot::D2CpmdConfig cfg;
  cfg.symmetry = symmetry;
  return cfg;
}

} // namespace

TEST_CASE("CPMD's D2 matches OpenCPMD's empirical correction", "[d2cpmd][reference]") {
  rgpot::D2CpmdPot pot;
  for (const char *name : kSystems) {
    INFO(name);
    const Reference ref = load(name);
    const Result r = run(pot, ref.z, ref.pos, ref.box);
    // CPMD prints eight decimals, rounded.
    REQUIRE(std::abs(ref.printed - ref.energy) <= 0.5e-8);
    REQUIRE(std::abs(r.energy / HARTREE_TO_EV - ref.energy) <= kEnergyThr);
    const double to_au = rgpot::kCpmdBohr / HARTREE_TO_EV;
    for (std::size_t k = 0; k < ref.pos.size(); ++k) {
      INFO("component " << k);
      REQUIRE(std::abs(r.forces[k] * to_au - ref.force[k]) <= kGradThr);
    }
    REQUIRE(r.has_stress);
  }
}

TEST_CASE("CPMD's D2 forces and stress are derivatives of its energy", "[d2cpmd][fd]") {
  rgpot::D2CpmdPot pot;
  // Fourth order step, compared directly to the reported force and stress.
  // The bar is under the relative gap between kCpmdBohr and
  // rgpot::units::BOHR_TO_ANGSTROM (about 8e-11 for a force, about 2.5e-10
  // for a stress), so converting either with the other constant fails.
  constexpr double kStep = 5.0e-4;
  constexpr double kRel = 4.0e-11;
  auto stencil = [](double em2, double em, double ep, double ep2) {
    return (em2 - 8.0 * em + 8.0 * ep - ep2) / (12.0 * kStep);
  };
  for (const char *name : {"cubic_water", "ortho_water_methane", "triclinic_water_methane"}) {
    INFO(name);
    const Reference ref = load(name);
    const Result r = run(pot, ref.z, ref.pos, ref.box);
    double fscale = 0.0;
    for (double f : r.forces) {
      fscale = std::max(fscale, std::abs(f));
    }
    REQUIRE(fscale > 0.0);
    for (std::size_t k = 0; k < ref.pos.size(); ++k) {
      auto at = [&](double delta) {
        std::vector<double> p = ref.pos;
        p[k] += delta;
        return run(pot, ref.z, p, ref.box).energy;
      };
      const double dE = stencil(at(-2.0 * kStep), at(-kStep), at(kStep), at(2.0 * kStep));
      INFO("component " << k);
      REQUIRE(std::abs(dE + r.forces[k]) <= kRel * fscale);
    }
  }
  // A strain of this step stays on one image in the orthorhombic cell.
  // The cubic and triclinic references have a pair close to a boundary,
  // so the same strain is not a derivative of one smooth branch.
  const Reference ref = load("ortho_water_methane");
  const Result r = run(pot, ref.z, ref.pos, ref.box);
  const double v = volume(ref.box);
  double sscale = 0.0;
  for (double s : r.stress) {
    sscale = std::max(sscale, std::abs(v * s));
  }
  REQUIRE(sscale > 0.0);
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      auto strained = [&](double e) {
        std::array<double, 9> box = ref.box;
        std::vector<double> p = ref.pos;
        for (int i = 0; i < 3; ++i) {
          box[3 * i + b] += e * ref.box[3 * i + a];
        }
        for (std::size_t n = 0; n < ref.z.size(); ++n) {
          p[3 * n + b] += e * ref.pos[3 * n + a];
        }
        return run(pot, ref.z, p, box).energy;
      };
      const double dE =
          stencil(strained(-2.0 * kStep), strained(-kStep), strained(kStep), strained(2.0 * kStep));
      INFO("stress " << a << b);
      REQUIRE(std::abs(dE - v * r.stress[3 * a + b]) <= kRel * sscale);
    }
  }
}

TEST_CASE("CPMD's minimum image has a cusp at half the cell", "[d2cpmd][cusp]") {
  // Two O atoms on the x axis of a 7 Angstrom cube, a hair either side of
  // 3.5 Angstrom apart. The minimum image switches sides between the two, so
  // CPMD's force on the pair flips sign while its energy does not move; the
  // lattice sum has both images at all times and stays smooth.
  const double length = 7.0;
  const std::array<double, 9> box{length, 0, 0, 0, length, 0, 0, 0, length};
  const std::vector<int> z{8, 8};
  const double delta = std::cbrt(std::numeric_limits<double>::epsilon());
  auto at = [&](double x) { return std::vector<double>{0.0, 0.0, 0.0, x, 0.0, 0.0}; };
  const std::vector<double> below = at(0.5 * length - delta);
  const std::vector<double> above = at(0.5 * length + delta);

  rgpot::D2CpmdPot cpmd;
  const Result cb = run(cpmd, z, below, box);
  const Result ca = run(cpmd, z, above, box);
  INFO("CPMD forces " << cb.forces[3] << " and " << ca.forces[3] << " eV/A");
  REQUIRE(std::abs(cb.energy - ca.energy) <= kGradThr * HARTREE_TO_EV);
  REQUIRE(cb.forces[3] < 0.0);
  REQUIRE(ca.forces[3] > 0.0);
  REQUIRE(std::abs(cb.forces[3] + ca.forces[3]) <= 4.0 * delta * std::abs(cb.forces[3]));

  rgpot::D2Pot sum;
  const Result sb = run(sum, z, below, box);
  const Result sa = run(sum, z, above, box);
  INFO("lattice sum forces " << sb.forces[3] << " and " << sa.forces[3] << " eV/A");
  REQUIRE(std::abs(sb.forces[3] - sa.forces[3]) <= kGradThr * HARTREE_TO_EV / BOHR_TO_ANGSTROM);
  REQUIRE(std::abs(sb.forces[3]) <= kGradThr * HARTREE_TO_EV / BOHR_TO_ANGSTROM);
}

TEST_CASE("CPMD's D2 tables, functionals and limits", "[d2cpmd]") {
  REQUIRE(rgpot::d2_cpmd_config_for("blyp").s6 == 1.2);
  REQUIRE(rgpot::d2_cpmd_config_for("PBE").s6 == 0.75);
  REQUIRE(rgpot::d2_cpmd_config_for("revpbe").s6 == 1.25);
  REQUIRE_THROWS_AS(rgpot::d2_cpmd_config_for("b97-d"), std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::D2CpmdPot(rgpot::D2CpmdConfig{.vdweps = 0.0}),
                    std::invalid_argument);

  rgpot::D2CpmdPot pot;
  const std::array<double, 9> none{};
  const std::array<double, 9> box{30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0};
  const std::vector<double> pos{0.0, 0.0, 0.0, 3.0, 0.0, 0.0};
  // Francium to einsteinium have no parameters and give nothing.
  const Result fr = run(pot, std::vector<int>{87, 8}, pos, box);
  REQUIRE(fr.energy == 0.0);
  REQUIRE_THROWS_AS(run(pot, std::vector<int>{100, 8}, pos, box), std::invalid_argument);
  // An all-zero box is a molecule: no image at all.
  const Result mol = run(pot, std::vector<int>{8, 8}, pos, none);
  REQUIRE(mol.energy < 0.0);
  REQUIRE_FALSE(mol.has_stress);
  REQUIRE(mol.energy == run(pot, std::vector<int>{8, 8}, pos, box).energy);

  const std::vector<int> z_pair{8, 8};
  auto run_ptr = [&](rgpot::D2CpmdPot &p, const double *b) {
    Result r;
    r.forces.assign(pos.size(), 0.0);
    const rgpot::ForceInput in{z_pair.size(), pos.data(), z_pair.data(), b};
    rgpot::ForceOut out{};
    out.F = r.forces.data();
    p.forceBatch(rgpot::ForceBatch{.nSystems = 1, .in = &in, .out = &out});
    r.energy = out.energy;
    r.has_stress = out.has_stress != 0;
    return r;
  };
  const Result null_box = run_ptr(pot, nullptr);
  REQUIRE_FALSE(null_box.has_stress);
  REQUIRE(null_box.energy == mol.energy);
  REQUIRE(null_box.forces == mol.forces);
#ifdef RGPOT_HAS_CACHE
  const std::string db_path =
      (std::filesystem::temp_directory_path() /
       ("rgpot_d2cpmd_null_box_" +
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count())))
          .string();
  {
    rgpot::cache::PotentialCache cache(db_path);
    pot.setCache(&cache);
    const Result cold = run_ptr(pot, nullptr);
    const Result warm = run_ptr(pot, nullptr);
    REQUIRE(cache.counts().computed == 1);
    REQUIRE(cache.counts().served == 1);
    REQUIRE(cold.energy == mol.energy);
    REQUIRE(warm.energy == mol.energy);
    REQUIRE(warm.forces == mol.forces);
    REQUIRE_FALSE(warm.has_stress);
    (void)run_ptr(pot, none.data());
    REQUIRE(cache.counts().computed == 2);
    REQUIRE(cache.counts().served == 1);
    const Result cached_box = run_ptr(pot, box.data());
    REQUIRE(cache.counts().computed == 3);
    REQUIRE(cache.counts().served == 1);
    REQUIRE(cached_box.has_stress);
    pot.setCache(nullptr);
  }
  REQUIRE(rocksdb::DestroyDB(db_path, rocksdb::Options()).ok());
#endif

  // The 6 R0 truncation: the O-O sum radius is 2 x 1.342 Angstrom.
  const double cut = 6.0 * 2.684;
  const double rel = std::cbrt(std::numeric_limits<double>::epsilon());
  const Result in = run(pot, std::vector<int>{8, 8},
                        std::vector<double>{0.0, 0.0, 0.0, cut * (1.0 - rel), 0.0, 0.0}, none);
  const Result out = run(pot, std::vector<int>{8, 8},
                         std::vector<double>{0.0, 0.0, 0.0, cut * (1.0 + rel), 0.0, 0.0}, none);
  REQUIRE(in.energy < 0.0);
  REQUIRE(out.energy == 0.0);
}

TEST_CASE("a composite swaps CPMD's D2 for the lattice sum", "[d2cpmd][composite]") {
  const Reference ref = load("cubic_water");
  rgpot::D2Pot lattice;
  rgpot::D2CpmdPot cpmd;
  const Result l = run(lattice, ref.z, ref.pos, ref.box);
  const Result c = run(cpmd, ref.z, ref.pos, ref.box);

  rgpot::SumPot sum;
  sum.add({.name = "undo cpmd", .weight = -1.0}, std::make_unique<rgpot::D2CpmdPot>());
  sum.add({.name = "d2"}, std::make_unique<rgpot::D2Pot>());
  REQUIRE(sum.caps().stress);
  const Result s = run(sum, ref.z, ref.pos, ref.box);
  REQUIRE(s.energy == Catch::Approx(l.energy - c.energy).margin(kEnergyThr * HARTREE_TO_EV));
  for (std::size_t k = 0; k < ref.pos.size(); ++k) {
    REQUIRE(s.forces[k] ==
            Catch::Approx(l.forces[k] - c.forces[k]).margin(kGradThr * HARTREE_TO_EV));
  }
  REQUIRE(sum.lastComponents()[0].energy == c.energy);
}

TEST_CASE("CPMD pbc3 matches OpenCPMD on the symmetry cells", "[d2cpmd][pbc3]") {
  const double to_au = rgpot::kCpmdBohr / HARTREE_TO_EV;
  bool saw_difference = false;
  for (const char *name : all_pbc3()) {
    INFO(name);
    const Reference ref = load(name);
    REQUIRE(ref.version == kOpenCpmdPbc3);
    REQUIRE(ref.ibrav != 0);
    rgpot::D2CpmdPot pot(with_symmetry(ref.ibrav));
    const Result got = run(pot, ref.z, ref.pos, ref.box);
    REQUIRE(std::abs(got.energy / HARTREE_TO_EV - ref.energy) <= kEnergyThr);
    REQUIRE(got.forces.size() == ref.force.size());
    for (std::size_t k = 0; k < ref.force.size(); ++k) {
      INFO("component " << k);
      REQUIRE(std::abs(got.forces[k] * to_au - ref.force[k]) <= kGradThr);
    }
    if (std::string(name) == "pbc3_hexagonal" || std::string(name) == "pbc3_monoclinic") {
      REQUIRE(ref.differs);
    }
    if (ref.differs) {
      saw_difference = true;
      REQUIRE(std::abs(ref.energy - ref.euclidean) > kEnergyThr);
      REQUIRE(std::abs(got.energy / HARTREE_TO_EV - ref.euclidean) > kEnergyThr);
    }
  }
  REQUIRE(saw_difference);
}

TEST_CASE("CPMD pbc3 is read from the box for every code but 12", "[d2cpmd][pbc3]") {
  const double to_au = rgpot::kCpmdBohr / HARTREE_TO_EV;
  rgpot::D2CpmdPot pot;
  for (const char *name : kPbc3Inferable) {
    INFO(name);
    const Reference ref = load(name);
    const Result got = run(pot, ref.z, ref.pos, ref.box);
    REQUIRE(std::abs(got.energy / HARTREE_TO_EV - ref.energy) <= kEnergyThr);
    for (std::size_t k = 0; k < ref.force.size(); ++k) {
      REQUIRE(std::abs(got.forces[k] * to_au - ref.force[k]) <= kGradThr);
    }
  }
}

TEST_CASE("CPMD symmetry 12 takes pbc3's monoclinic branch", "[d2cpmd][pbc3][symmetry12]") {
  const double to_au = rgpot::kCpmdBohr / HARTREE_TO_EV;
  for (const char *name : kPbc3Monoclinic12) {
    INFO(name);
    const Reference ref = load(name);
    REQUIRE(ref.ibrav == 12);
    // The nearest integer image, which is what any other branch of this box
    // that is not the monoclinic one amounts to, is a different energy.
    REQUIRE(std::abs(ref.energy - ref.nearest) > kEnergyThr);
    rgpot::D2CpmdPot pot(with_symmetry(12));
    const Result got = run(pot, ref.z, ref.pos, ref.box);
    REQUIRE(std::abs(got.energy / HARTREE_TO_EV - ref.energy) <= kEnergyThr);
    REQUIRE(std::abs(got.energy / HARTREE_TO_EV - ref.nearest) > kEnergyThr);
    for (std::size_t k = 0; k < ref.force.size(); ++k) {
      INFO("component " << k);
      REQUIRE(std::abs(got.forces[k] * to_au - ref.force[k]) <= kGradThr);
    }
  }
}

TEST_CASE("CPMD symmetry codes that pbc3 does not handle are refused", "[d2cpmd][pbc3]") {
  for (int symmetry : {0, 9, 10, 11, 13, 15, -2, -3}) {
    INFO("symmetry " << symmetry);
    REQUIRE_THROWS_AS(rgpot::D2CpmdPot(with_symmetry(symmetry)), std::invalid_argument);
  }
  for (int symmetry : {rgpot::kCpmdInferSymmetry, 1, 2, 3, 4, 5, 6, 7, 8, 12, 14}) {
    INFO("symmetry " << symmetry);
    REQUIRE_NOTHROW(rgpot::D2CpmdPot(with_symmetry(symmetry)));
  }
}

TEST_CASE("CPMD refuses a box that does not have the shape of its symmetry",
          "[d2cpmd][pbc3]") {
  const Reference water = load("cubic_water");
  const Reference triclinic = load("triclinic_water_methane");
  const Reference hex = load("pbc3_hexagonal");
  const Reference mono = load("pbc3_monoclinic12");
  const Reference ortho = load("pbc3_orthorhombic");
  struct Case {
    int symmetry;
    const Reference *ref;
  };
  const Case wrong[] = {{1, &ortho}, {1, &hex},  {2, &water},     {2, &hex},
                        {4, &mono},  {4, &water}, {8, &hex},       {8, &triclinic},
                        {12, &triclinic}};
  for (const Case &c : wrong) {
    INFO("symmetry " << c.symmetry << " ref " << c.ref->ibrav);
    rgpot::D2CpmdPot pot(with_symmetry(c.symmetry));
    REQUIRE_THROWS_AS(run(pot, c.ref->z, c.ref->pos, c.ref->box), std::invalid_argument);
  }
  // An isolated system has no box to check, whatever the code.
  for (int symmetry : {1, 2, 4, 8, 12, 14}) {
    rgpot::D2CpmdPot pot(with_symmetry(symmetry));
    const std::array<double, 9> none{};
    REQUIRE_NOTHROW(run(pot, water.z, water.pos, none));
  }
}

TEST_CASE("CPMD pbc3 folds a displacement of several cells like one inside the cell",
          "[d2cpmd][pbc3]") {
  const std::array<std::array<double, 3>, 3> kShifts{{
      {2.0, -2.0, 2.0},
      {3.0, -4.0, 2.0},
      {-7.0, 5.0, -6.0},
  }};
  for (const char *name : all_pbc3()) {
    const Reference ref = load(name);
    rgpot::D2CpmdPot pot(with_symmetry(ref.ibrav));
    const Result want = run(pot, ref.z, ref.pos, ref.box);
    for (const auto &shift : kShifts) {
      // pbc3's monoclinic branch folds along three lattice directions in
      // turn and does not always land in the same image for a far
      // displacement; OpenCPMD's own energy for this cell and this shift
      // is not the one inside the cell, and the port reproduces it.
      if (std::string(name) == "pbc3_monoclinic12" && shift[0] == 3.0) {
        continue;
      }
      INFO(name << " shift " << shift[0] << " " << shift[1] << " " << shift[2]);
      std::vector<double> far = ref.pos;
      for (int a = 0; a < 3; ++a) {
        far[3 + a] += shift[0] * ref.box[a] + shift[1] * ref.box[3 + a] + shift[2] * ref.box[6 + a];
      }
      const Result got = run(pot, ref.z, far, ref.box);
      REQUIRE(got.energy == Catch::Approx(want.energy).margin(kEnergyThr * HARTREE_TO_EV));
      REQUIRE(got.forces.size() == want.forces.size());
      for (std::size_t k = 0; k < want.forces.size(); ++k) {
        REQUIRE(got.forces[k] ==
                Catch::Approx(want.forces[k]).margin(kGradThr * HARTREE_TO_EV / rgpot::kCpmdBohr));
      }
    }
  }
}
