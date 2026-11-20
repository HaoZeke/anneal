// MIT License
// Copyright 2023--present rgpot developers
//
// rgpot::D2Pot against the dftd3 program (V3.1, `-func b-lyp -old`, with
// `-pbc` for the cells) on a water dimer, benzene and an eight-water box in a
// cubic and a sheared cell; finite differences of the energy for the forces
// and the stress; and the convergence of the lattice sum with the cutoff
// that fixes D2Config's default. scripts/regen_d2_references.py writes the
// references in CppCore/tests/data/d2/.

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
#include "rgpot/D2Pot/D2Cell.hpp"
#include "rgpot/D2Pot/D2Pot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
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

struct Reference {
  std::vector<int> z;
  std::vector<double> pos; // Angstrom
  std::array<double, 9> box{};
  bool periodic = false;
  double energy = 0.0;               // Ha
  std::vector<double> grad;          // Ha/bohr
  std::array<double, 9> cellgrad{};  // Ha/bohr, row i = dE/da_i
};

Reference load(const std::string &name) {
  const std::string path = "CppCore/tests/data/d2/" + name + ".txt";
  std::ifstream in(path);
  INFO(path);
  REQUIRE(in.good());
  Reference r;
  std::string line;
  int cell_row = 0;
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
    } else if (key == "cellgrad") {
      ss >> r.cellgrad[3 * cell_row] >> r.cellgrad[3 * cell_row + 1] >>
          r.cellgrad[3 * cell_row + 2];
      ++cell_row;
    }
  }
  REQUIRE(r.grad.size() == r.pos.size());
  REQUIRE(cell_row == (r.periodic ? 3 : 0));
  return r;
}

struct Result {
  double energy = 0.0; // eV
  std::vector<double> forces;
  std::array<double, 9> stress{};
  bool has_stress = false;
};

template <typename Pot>
Result run(Pot &&pot, const std::vector<int> &z,
           const std::vector<double> &pos, const std::array<double, 9> &box) {
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

/// dftd3 prints the b-lyp s6 from the single-precision literal `s6=1.2` of
/// its setfuncpar, so the comparison takes the same value.
rgpot::D2Config dftd3Blyp(bool periodic) {
  rgpot::D2Config c;
  c.s6 = static_cast<double>(1.2f);
  if (periodic) {
    // dftd3's default pair threshold, rthr = 9000 bohr^2.
    c.cutoff = std::sqrt(9000.0) * BOHR_TO_ANGSTROM;
  }
  return c;
}

struct Gap {
  double mid = 0.0;
  double half = 0.0;
};

/// The widest gap between pair and image distances in [lo, hi], with every
/// image that can reach hi counted.
Gap widestGap(const Reference &ref, double lo, double hi) {
  std::vector<double> d{lo, hi};
  const std::size_t n = ref.z.size();
  std::array<int, 3> rep{};
  if (ref.periodic) {
    const double v = volume(ref.box);
    for (int k = 0; k < 3; ++k) {
      const double *u = &ref.box[3 * ((k + 1) % 3)];
      const double *w = &ref.box[3 * ((k + 2) % 3)];
      const double c[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                           u[0] * w[1] - u[1] * w[0]};
      const double spacing = v / std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
      rep[k] = static_cast<int>(std::ceil(hi / spacing)) + 1;
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (int t0 = -rep[0]; t0 <= rep[0]; ++t0) {
        for (int t1 = -rep[1]; t1 <= rep[1]; ++t1) {
          for (int t2 = -rep[2]; t2 <= rep[2]; ++t2) {
            double r2 = 0.0;
            for (int a = 0; a < 3; ++a) {
              const double x = ref.pos[3 * i + a] - ref.pos[3 * j + a] + t0 * ref.box[a] +
                               t1 * ref.box[3 + a] + t2 * ref.box[6 + a];
              r2 += x * x;
            }
            const double r = std::sqrt(r2);
            if (r > lo && r < hi) {
              d.push_back(r);
            }
          }
        }
      }
    }
  }
  std::sort(d.begin(), d.end());
  Gap best;
  for (std::size_t k = 1; k < d.size(); ++k) {
    if (0.5 * (d[k] - d[k - 1]) > best.half) {
      best.half = 0.5 * (d[k] - d[k - 1]);
      best.mid = 0.5 * (d[k] + d[k - 1]);
    }
  }
  return best;
}

const char *kSystems[] = {"water_dimer", "benzene", "water_box_cubic", "water_box_sheared"};

} // namespace

TEST_CASE("D2 tables and functional parameters follow dftd3", "[d2]") {
  REQUIRE(rgpot::d2_c6(1) == 0.14);
  REQUIRE(rgpot::d2_r0(1) == 0.91);
  REQUIRE(rgpot::d2_c6(8) == 0.70);
  REQUIRE(rgpot::d2_r0(8) == 1.22);
  // Paper radius = table radius * 1.1: O is 1.342 Angstrom in Table 1.
  REQUIRE(rgpot::d2_r0(8) * 1.1 == Catch::Approx(1.342));
  REQUIRE(rgpot::d2_c6(86) == 56.64);
  REQUIRE_THROWS_AS(rgpot::d2_c6(0), std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::d2_r0(87), std::invalid_argument);

  const rgpot::D2Config blyp = rgpot::d2_config_for("B-LYP");
  REQUIRE(blyp.s6 == 1.2);
  REQUIRE(blyp.d == 20.0);
  REQUIRE(blyp.sr == 1.1);
  REQUIRE(rgpot::d2_config_for("blyp").s6 == 1.2);
  REQUIRE(rgpot::d2_config_for("pbe").s6 == 0.75);
  REQUIRE(rgpot::d2_config_for("b3lyp").s6 == 1.05);
  REQUIRE(rgpot::d2_config_for("dsd-blyp").d == 60.0);
  REQUIRE(rgpot::d2_config_for("blyp").cutoff == rgpot::D2Config{}.cutoff);
  REQUIRE_THROWS_AS(rgpot::d2_config_for("nonesuch"), std::invalid_argument);

  REQUIRE_THROWS_AS(rgpot::D2Pot(rgpot::D2Config{.cutoff = 0.0}), std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::D2Pot(rgpot::D2Config{.sr = 0.0}), std::invalid_argument);
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_THROWS_AS(rgpot::D2Pot(rgpot::D2Config{.sr = inf}), std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::D2Pot(rgpot::D2Config{.sr = nan}), std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::D2Pot(rgpot::D2Config{.cutoff = inf}), std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::D2Pot(rgpot::D2Config{.s6 = inf}), std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::D2Pot(rgpot::D2Config{.d = inf}), std::invalid_argument);
}

TEST_CASE("D2 refuses a cell too thin for its cutoff", "[d2][limits]") {
  using rgpot::d2::detail::Cell;
  using rgpot::d2::detail::kMaxImagesPerAxis;
  const double rc = 100.0; // bohr
  auto cubic = [](double length) {
    const double box[9] = {length, 0, 0, 0, length, 0, 0, 0, length};
    return Cell::from_box(box, 1.0);
  };
  // The image count along a vector is ceil(rc / spacing + 1/2).
  const double edge = kMaxImagesPerAxis - 1.0;
  REQUIRE_NOTHROW(cubic(rc / edge).require_bounded_images(rc));
  REQUIRE_THROWS_AS(cubic(rc / (edge + 1.0)).require_bounded_images(rc), std::invalid_argument);
  // A count that does not fit an int would overflow the conversion.
  const double beyond_int = 2.0 * static_cast<double>(std::numeric_limits<int>::max());
  REQUIRE_THROWS_AS(cubic(rc / beyond_int).require_bounded_images(rc), std::invalid_argument);
  REQUIRE_THROWS_WITH(cubic(rc / beyond_int).require_bounded_images(rc),
                      Catch::Matchers::ContainsSubstring(std::to_string(kMaxImagesPerAxis)));

  const Reference ref = load("water_dimer");
  rgpot::D2Pot pot;
  const double thin = rc * BOHR_TO_ANGSTROM / beyond_int;
  const std::array<double, 9> box{thin, 0, 0, 0, thin, 0, 0, 0, thin};
  REQUIRE_THROWS_AS(run(pot, ref.z, ref.pos, box), std::invalid_argument);
}

TEST_CASE("D2 matches dftd3 energies, gradients and cell gradients", "[d2][reference]") {
  for (const char *name : kSystems) {
    INFO(name);
    const Reference ref = load(name);
    rgpot::D2Pot pot(dftd3Blyp(ref.periodic));
    const Result r = run(pot, ref.z, ref.pos, ref.box);

    const double e_ha = r.energy / HARTREE_TO_EV;
    INFO("energy " << e_ha << " Ha, dftd3 " << ref.energy);
    REQUIRE(std::abs(e_ha - ref.energy) <= kEnergyThr);

    const double fconv = HARTREE_TO_EV / BOHR_TO_ANGSTROM;
    for (std::size_t k = 0; k < ref.grad.size(); ++k) {
      INFO("gradient component " << k);
      REQUIRE(std::abs(-r.forces[k] / fconv - ref.grad[k]) <= kGradThr);
    }

    REQUIRE(r.has_stress == ref.periodic);
    if (ref.periodic) {
      // sigma = (1/V) h^T G, G the gradient by lattice vector (bohr units).
      std::array<double, 9> h{};
      for (int k = 0; k < 9; ++k) {
        h[k] = ref.box[k] / BOHR_TO_ANGSTROM;
      }
      const double v = volume(h);
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          double w = 0.0;
          for (int i = 0; i < 3; ++i) {
            w += h[3 * i + a] * ref.cellgrad[3 * i + b];
          }
          const double ours =
              r.stress[3 * a + b] / rgpot::units::HARTREE_PER_BOHR3_TO_EV_PER_ANGSTROM3 * v;
          INFO("virial " << a << b << " ours " << ours << " dftd3 " << w);
          // The virial is a gradient times a length of the cell.
          REQUIRE(std::abs(ours - w) <= kGradThr * std::cbrt(v));
        }
      }
    }
  }
}

TEST_CASE("D2 forces and stress are derivatives of its energy", "[d2][fd]") {
  for (const char *name : {"water_dimer", "water_box_sheared"}) {
    INFO(name);
    const Reference ref = load(name);
    // The cutoff sits in the widest gap between image distances near 15
    // Angstrom, so no displacement or strain below moves a pair across it
    // and the truncated sum stays smooth.
    const Gap gap = widestGap(ref, 14.0, 16.0);
    INFO("cutoff " << gap.mid << " A, half gap " << gap.half);
    REQUIRE(gap.half > 0.0);
    rgpot::D2Pot pot(rgpot::D2Config{.cutoff = gap.mid});
    const Result r = run(pot, ref.z, ref.pos, ref.box);
    // Central differences: the error is step^2 times the third derivative
    // plus the rounding E eps / step.
    const double step = std::min(0.25 * gap.half, std::cbrt(std::numeric_limits<double>::epsilon()));
    const double bar = kGradThr * HARTREE_TO_EV / BOHR_TO_ANGSTROM;
    for (std::size_t k = 0; k < ref.pos.size(); ++k) {
      std::vector<double> p = ref.pos;
      p[k] += step;
      const double ep = run(pot, ref.z, p, ref.box).energy;
      p[k] -= 2.0 * step;
      const double em = run(pot, ref.z, p, ref.box).energy;
      INFO("component " << k);
      REQUIRE(std::abs(-(ep - em) / (2.0 * step) - r.forces[k]) <= bar);
    }
    if (!ref.periodic) {
      continue;
    }
    const double v = volume(ref.box);
    const double strain = step / (2.0 * gap.mid);
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
        const double dE = (strained(strain) - strained(-strain)) / (2.0 * strain);
        INFO("stress " << a << b);
        // dE/d(strain) is a gradient times a length of the cell.
        REQUIRE(std::abs(dE - v * r.stress[3 * a + b]) <=
                kGradThr * HARTREE_TO_EV / BOHR_TO_ANGSTROM * std::cbrt(v));
      }
    }
  }
}

TEST_CASE("D2's default cutoff converges the lattice sum", "[d2][convergence]") {
  // Doubling the cutoff past the default moves the energy per atom by less
  // than kGradThr Ha and every force component by less than kGradThr
  // Ha/bohr, the bars the reference comparison uses; the tail falls as
  // cutoff^-3 for the energy and cutoff^-4 for the forces, so the sequence
  // below shrinks monotonically.
  const double fbar = kGradThr * HARTREE_TO_EV / BOHR_TO_ANGSTROM;
  const double ebar = kGradThr * HARTREE_TO_EV;
  for (const char *name : {"water_box_cubic", "water_box_sheared"}) {
    INFO(name);
    const Reference ref = load(name);
    const double n = static_cast<double>(ref.z.size());
    const double c0 = rgpot::D2Config{}.cutoff;
    const Result converged = run(rgpot::D2Pot(rgpot::D2Config{.cutoff = 2.0 * c0}), ref.z,
                                 ref.pos, ref.box);
    double last_de = std::numeric_limits<double>::infinity();
    for (double scale : {0.25, 0.5, 1.0}) {
      const Result r = run(rgpot::D2Pot(rgpot::D2Config{.cutoff = scale * c0}), ref.z,
                           ref.pos, ref.box);
      const double de = std::abs(r.energy - converged.energy) / n;
      double df = 0.0;
      for (std::size_t k = 0; k < r.forces.size(); ++k) {
        df = std::max(df, std::abs(r.forces[k] - converged.forces[k]));
      }
      WARN(name << " cutoff " << scale * c0 << " A: |dE|/atom " << de / HARTREE_TO_EV
                << " Ha, max |dF| " << df / (HARTREE_TO_EV / BOHR_TO_ANGSTROM)
                << " Ha/bohr against cutoff " << 2.0 * c0 << " A");
      REQUIRE(de < last_de);
      last_de = de;
      if (scale == 1.0) {
        REQUIRE(de <= ebar);
        REQUIRE(df <= fbar);
      }
    }
  }
}

TEST_CASE("D2 of a molecule needs no cell", "[d2]") {
  const Reference ref = load("water_dimer");
  rgpot::D2Pot pot;
  const Result open = run(pot, ref.z, ref.pos, ref.box);
  REQUIRE_FALSE(open.has_stress);
  // A cell far larger than the cutoff has no image within it.
  std::array<double, 9> big{};
  big[0] = big[4] = big[8] = 4.0 * rgpot::D2Config{}.cutoff;
  const Result boxed = run(pot, ref.z, ref.pos, big);
  REQUIRE(boxed.energy == Catch::Approx(open.energy).epsilon(kEnergyThr));
  REQUIRE(boxed.has_stress);
  std::array<double, 9> singular{};
  singular[0] = singular[4] = 10.0;
  REQUIRE_THROWS_AS(run(pot, ref.z, ref.pos, singular), std::invalid_argument);
  std::vector<int> bad = ref.z;
  bad[0] = 87;
  REQUIRE_THROWS_AS(run(pot, bad, ref.pos, ref.box), std::invalid_argument);

  auto run_ptr = [&](rgpot::D2Pot &p, const double *box) {
    Result r;
    r.forces.assign(ref.pos.size(), 0.0);
    const rgpot::ForceInput in{ref.z.size(), ref.pos.data(), ref.z.data(), box};
    rgpot::ForceOut out{};
    out.F = r.forces.data();
    p.forceBatch(rgpot::ForceBatch{.nSystems = 1, .in = &in, .out = &out});
    r.energy = out.energy;
    r.has_stress = out.has_stress != 0;
    return r;
  };
  const Result null_box = run_ptr(pot, nullptr);
  REQUIRE_FALSE(null_box.has_stress);
  REQUIRE(null_box.energy == open.energy);
  REQUIRE(null_box.forces == open.forces);
#ifdef RGPOT_HAS_CACHE
  const std::string db_path =
      (std::filesystem::temp_directory_path() /
       ("rgpot_d2_null_box_" +
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
    REQUIRE(cold.energy == open.energy);
    REQUIRE(warm.energy == open.energy);
    REQUIRE(warm.forces == open.forces);
    REQUIRE_FALSE(warm.has_stress);
    // Distinct from a real box, including an all-zero one and a periodic cell.
    (void)run_ptr(pot, ref.box.data());
    REQUIRE(cache.counts().computed == 2);
    REQUIRE(cache.counts().served == 1);
    const Result cached_big = run_ptr(pot, big.data());
    REQUIRE(cache.counts().computed == 3);
    REQUIRE(cache.counts().served == 1);
    REQUIRE(cached_big.has_stress);
    pot.setCache(nullptr);
  }
  REQUIRE(rocksdb::DestroyDB(db_path, rocksdb::Options()).ok());
#endif
}

TEST_CASE("a composite with D2 at weight -1 cancels a D2-only run", "[d2][composite]") {
  const Reference ref = load("water_box_sheared");
  rgpot::SumPot sum;
  sum.add({.name = "lj"}, std::make_unique<rgpot::LJPot>());
  sum.add({.name = "d2", .weight = -1.0}, std::make_unique<rgpot::D2Pot>());
  const Result composite = run(sum, ref.z, ref.pos, ref.box);
  const Result d2 = run(rgpot::D2Pot(), ref.z, ref.pos, ref.box);
  const Result lj = run(rgpot::LJPot(), ref.z, ref.pos, ref.box);
  const double scale = std::abs(lj.energy) + std::abs(d2.energy);
  REQUIRE(std::abs(composite.energy + d2.energy - lj.energy) <= kEnergyThr * scale);
  for (std::size_t k = 0; k < lj.forces.size(); ++k) {
    REQUIRE(std::abs(composite.forces[k] + d2.forces[k] - lj.forces[k]) <=
            kEnergyThr * (std::abs(lj.forces[k]) + std::abs(d2.forces[k]) + 1.0));
  }
  REQUIRE(composite.has_stress);
  REQUIRE(sum.lastComponents()[1].energy == d2.energy);
  REQUIRE(sum.caps().stress);
  REQUIRE(sum.caps().reentrancy == rgpot::Reentrancy::PerInstance);
}
