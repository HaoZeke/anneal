// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief EAM images past half a face spacing.
 *
 * Compiled apart from EAMTablePot::compute. That stored pair walk is the
 * large cell kernel. Inlining this walk into it changes the code the
 * large cell runs.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "rgpot/EAMTable/EAMTablePot.hpp"
#include "rgpot/EAMTable/EAMTableTerms.hpp"
#include "rgpot/nlist/MultiImage.hpp"
#include "rgpot/stress.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define RGPOT_EAM_IMAGES __attribute__((noinline))
#elif defined(_MSC_VER)
#define RGPOT_EAM_IMAGES __declspec(noinline)
#else
#define RGPOT_EAM_IMAGES
#endif

namespace rgpot {

RGPOT_EAM_IMAGES
void EAMTablePot::computeAllImages(const ForceInput &in, ForceOut *out,
                                   const PerAtomOut *atoms) const {
  const auto n_size = in.nAtoms;
  double *F = out->F;
  std::fill(F, F + 3 * n_size, 0.0);
  out->energy = 0.0;
  out->variance = 0.0;
  const EAMTable &t = m_table;
  std::vector<std::size_t> el(n_size);
  for (std::size_t i = 0; i < n_size; ++i) {
    const int a = element_of(m_element, in.atmnrs[i]);
    if (a < 0) {
      bad("no element for atomic number " + std::to_string(in.atmnrs[i]));
    }
    el[i] = static_cast<std::size_t>(a);
  }

  const double *box = in.box;
  if (!(cellVolume(box) > 0.0)) {
    bad("periodic evaluation needs a cell");
  }
  const std::array<bool, 3> periodic{true, true, true};
  const bool pairs = n_size >= 1;
  const double cut2 = t.cutoff * t.cutoff;
  const Terms terms{t, m_frho, m_rhor, m_z2r};

  std::vector<double> rho(n_size, 0.0);
  if (pairs) {
    nlist::visitHalfImages(
        in.pos, n_size, box, periodic, t.cutoff,
        [&](int32_t i, int32_t j, double, double, double, double r2) {
          if (!(r2 > 0.0) || !(r2 < cut2)) {
            return;
          }
          const Couple &c = terms.couple(el[static_cast<std::size_t>(i)],
                                         el[static_cast<std::size_t>(j)]);
          const Place k = terms.radial(std::sqrt(r2));
          const double rhoi = Knot(c.atI, k).value();
          if (i == j) {
            // i's own ± lattice images both contribute to rho_i.
            rho[static_cast<std::size_t>(i)] += 2.0 * rhoi;
          } else {
            rho[static_cast<std::size_t>(i)] += rhoi;
            rho[static_cast<std::size_t>(j)] += Knot(c.atJ, k).value();
          }
        });
  }

  std::vector<double> fprime(n_size, 0.0);
  double energy = 0.0;
  for (std::size_t i = 0; i < n_size; ++i) {
    const double embed = terms.embedding(el[i], rho[i], fprime[i]);
    energy += embed;
    if (atoms != nullptr && atoms->energy != nullptr) {
      atoms->energy[i] += embed;
    }
  }

  double strain[6] = {};
  if (pairs) {
    nlist::visitHalfImages(
        in.pos, n_size, box, periodic, t.cutoff,
        [&](int32_t i, int32_t j, double dx, double dy, double dz, double r2) {
          if (!(r2 > 0.0) || !(r2 < cut2)) {
            return;
          }
          const auto ui = static_cast<std::size_t>(i);
          const auto uj = static_cast<std::size_t>(j);
          double phi = 0.0;
          const double fscale =
              terms.pair(terms.couple(el[ui], el[uj]), std::sqrt(r2), fprime[ui],
                         fprime[uj], phi);
          energy += phi;
          F[3 * ui] += fscale * dx;
          F[3 * ui + 1] += fscale * dy;
          F[3 * ui + 2] += fscale * dz;
          F[3 * uj] -= fscale * dx;
          F[3 * uj + 1] -= fscale * dy;
          F[3 * uj + 2] -= fscale * dz;
          if (m_stress) {
            accumulatePairStrain(strain, fscale, dx, dy, dz);
          }
          if (atoms != nullptr) {
            tallyPair(*atoms, ui, uj, phi, fscale, dx, dy, dz);
          }
        });
  }

  out->energy = energy;
  if (m_stress) {
    publishCauchyStress(out, strain, cellVolume(in.box));
  }
}

} // namespace rgpot
