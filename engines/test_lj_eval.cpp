// MIT License
// Copyright 2023--present rgpot developers
//
// Shifted 12-6 Lennard-Jones for one pair, compared with the shared
// evaluator the engines call, and the potential it keeps between calls.

#include "lj_eval.hpp"

#include "rgpot/KeptPotential.hpp"
#include "rgpot/LennardJones/LJPot.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

double shifted(double r, double u0, double cutoff, double psi) {
  const double ar = std::pow(psi / r, 6.0);
  const double ac = std::pow(psi / cutoff, 6.0);
  const double pair = 4.0 * u0 * ar * (ar - 1.0);
  const double shift = 4.0 * u0 * ac * (ac - 1.0);
  return pair - shift;
}

int fail(const char *sentence) {
  std::fprintf(stderr, "%s\n", sentence);
  return 1;
}

struct Params {
  double u0, cutoff, psi;
};

struct Eval {
  int status = 1;
  double energy = 0.0;
  int has_stress = 0;
  std::vector<double> forces;
  double stress[9] = {};
};

Eval entry(const std::vector<double> &xyz, const std::vector<int> &z, const double *box,
           const Params &p) {
  Eval e;
  e.forces.assign(xyz.size(), 0.0);
  e.status = rgpot_lj_eval(xyz.data(), z.data(), z.size(), box, p.u0, p.cutoff, p.psi,
                           e.forces.data(), &e.energy, &e.has_stress, e.stress);
  return e;
}

/// A potential built for this call alone, as rgpot_lj_eval once did.
Eval fresh(const std::vector<double> &xyz, const std::vector<int> &z, const double *box,
           const Params &p) {
  rgpot::LJConfig config;
  config.u0 = p.u0;
  config.cutoff = p.cutoff;
  config.psi = p.psi;
  const rgpot::LJPot pot(config);
  Eval e;
  e.forces.assign(xyz.size(), 0.0);
  rgpot::ForceOut out{e.forces.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl({z.size(), xyz.data(), z.data(), box}, &out);
  e.status = 0;
  e.energy = out.energy;
  e.has_stress = out.has_stress;
  std::memcpy(e.stress, out.stress, sizeof(e.stress));
  return e;
}

bool same(const Eval &a, const Eval &b) {
  return a.status == b.status && a.has_stress == b.has_stress &&
         std::memcmp(&a.energy, &b.energy, sizeof(double)) == 0 &&
         a.forces.size() == b.forces.size() &&
         std::memcmp(a.forces.data(), b.forces.data(), a.forces.size() * sizeof(double)) == 0 &&
         std::memcmp(a.stress, b.stress, sizeof(a.stress)) == 0;
}

/// Every call has the bits of a potential built for it alone. A change of
/// parameters, atom count or atomic numbers builds the kept potential
/// again, and anything else reuses it.
int kept_checks() {
  using Kept = rgpot::Kept<rgpot::LJPot>;
  std::vector<double> xyz;
  std::vector<double> nudged;
  std::vector<double> shifted_xyz;
  for (int i = 0; i < 32; ++i) {
    const int cell[3] = {i % 4, (i / 4) % 4, i / 16};
    for (int k = 0; k < 3; ++k) {
      const double site = 7.0 + 1.12 * cell[k];
      const double phase = 1.7 * static_cast<double>(3 * i + k);
      xyz.push_back(site + 0.03 * std::sin(phase));
      nudged.push_back(site + 0.03 * std::cos(phase));
      // Past half the one angstrom skin, so the kept list is built again.
      shifted_xyz.push_back(site + 0.03 * std::sin(phase) + (k == 0 ? 0.6 : 0.0));
    }
  }
  const std::vector<int> z(32, 18);
  std::vector<int> other_z = z;
  other_z[5] = 36;
  const std::vector<double> fewer(xyz.begin(), xyz.begin() + 3 * 20);
  const std::vector<int> fewer_z(z.begin(), z.begin() + 20);
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  const double wider[9] = {24, 0, 0, 0, 22, 0, 0, 0, 20};
  const Params p{1.0, 2.5, 1.0};
  const Params deeper{1.5, 2.5, 1.0};
  const Params longer{1.0, 3.0, 1.0};
  const Params wider_psi{1.0, 2.5, 1.05};

  struct Step {
    const char *what;
    const std::vector<double> &xyz;
    const std::vector<int> &z;
    const double *box;
    Params p;
    bool build;
  };
  const Step steps[] = {
      {"first call", xyz, z, box, p, true},
      {"small move", nudged, z, box, p, false},
      {"move past the skin", shifted_xyz, z, box, p, false},
      {"new cell", xyz, z, wider, p, false},
      {"same cell again", xyz, z, wider, p, false},
      {"other atomic numbers", xyz, other_z, box, p, true},
      {"atomic numbers back", xyz, z, box, p, true},
      {"fewer atoms", fewer, fewer_z, box, p, true},
      {"well depth", xyz, z, box, deeper, true},
      {"cutoff", xyz, z, box, longer, true},
      {"psi", xyz, z, box, wider_psi, true},
      {"first parameters again", xyz, z, box, p, true},
  };
  Kept::release();
  rgpot::KeptCounts last = Kept::counts();
  for (const Step &s : steps) {
    if (!same(entry(s.xyz, s.z, s.box, s.p), fresh(s.xyz, s.z, s.box, s.p))) {
      std::fprintf(stderr, "%s: the kept potential differs from a fresh one\n", s.what);
      return 1;
    }
    const rgpot::KeptCounts now = Kept::counts();
    if (now.builds != last.builds + (s.build ? 1 : 0) ||
        now.reuses != last.reuses + (s.build ? 0 : 1)) {
      std::fprintf(stderr, "%s: %llu builds and %llu reuses after %llu and %llu\n", s.what,
                   static_cast<unsigned long long>(now.builds),
                   static_cast<unsigned long long>(now.reuses),
                   static_cast<unsigned long long>(last.builds),
                   static_cast<unsigned long long>(last.reuses));
      return 1;
    }
    last = now;
  }

  // A rejected call reaches no potential and leaves the kept one.
  Eval rejected;
  rejected.forces.assign(xyz.size(), 0.0);
  if (rgpot_lj_eval(xyz.data(), z.data(), z.size(), nullptr, p.u0, p.cutoff, p.psi,
                    rejected.forces.data(), &rejected.energy, &rejected.has_stress,
                    rejected.stress) == 0) {
    return fail("a null cell was accepted");
  }
  if (!same(entry(xyz, z, box, p), fresh(xyz, z, box, p)) ||
      Kept::counts().builds != last.builds || Kept::counts().reuses != last.reuses + 1) {
    return fail("a rejected call changed the kept potential");
  }
  return 0;
}

} // namespace

int main() {
  const double u0 = 1.0;
  const double cutoff = 8.0;
  const double psi = 1.0;
  const double r = 1.5;
  const double xyz[6] = {0.0, 0.0, 0.0, r, 0.0, 0.0};
  const int z[2] = {1, 1};
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  double forces[6];
  double energy = 0.0;
  int has_stress = 0;
  double stress[9];
  if (rgpot_lj_eval(xyz, z, 2, box, u0, cutoff, psi, forces, &energy, &has_stress,
                    stress) != 0) {
    return fail("rgpot_lj_eval rejected the dimer");
  }
  const double expect = shifted(r, u0, cutoff, psi);
  if (std::abs(energy - expect) > 1e-12) {
    std::fprintf(stderr, "energy %.16e expect %.16e\n", energy, expect);
    return 1;
  }
  if (!has_stress) {
    return fail("Lennard-Jones did not report stress");
  }
  const double eps = 1e-6;
  double xyz_plus[6] = {0.0, 0.0, 0.0, r + eps, 0.0, 0.0};
  double xyz_minus[6] = {0.0, 0.0, 0.0, r - eps, 0.0, 0.0};
  double unused[6];
  double e_plus = 0.0;
  double e_minus = 0.0;
  int ignored = 0;
  double ignored_stress[9];
  rgpot_lj_eval(xyz_plus, z, 2, box, u0, cutoff, psi, unused, &e_plus, &ignored,
                ignored_stress);
  rgpot_lj_eval(xyz_minus, z, 2, box, u0, cutoff, psi, unused, &e_minus, &ignored,
                ignored_stress);
  const double dEdr = (e_plus - e_minus) / (2.0 * eps);
  // Force on atom 1 is -dE/dr along x. Force on atom 0 is the opposite.
  if (std::abs(forces[3] + dEdr) > 1e-7 || std::abs(forces[0] - dEdr) > 1e-7) {
    std::fprintf(stderr, "force %g %g dEdr %g\n", forces[0], forces[3], dEdr);
    return 1;
  }
  if (rgpot_lj_eval(nullptr, z, 2, box, u0, cutoff, psi, forces, &energy, &has_stress,
                    stress) == 0) {
    return fail("a null coordinate pointer was accepted");
  }
  return kept_checks();
}
