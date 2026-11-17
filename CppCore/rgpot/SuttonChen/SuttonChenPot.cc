// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Sutton-Chen energy and analytic forces.
 *
 * Each unordered pair contributes epsilon * (a/r)^n once, which is the
 * 1/2 sum_i sum_{j != i} term. The density sum has no 1/2. The force
 * on atom i along d = r_i - r_j is fscale * d, with
 * fscale = -(dE/dr) / r.
 *
 * The energy depends on positions only through pair distances, so with
 * SuttonChenConfig::stress the force pass also sums -fscale * d (x) d,
 * which is dE/dstrain for the embedding and pair terms together.
 */

// clang-format off
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>
// clang-format on

#include "rgpot/KeptPotential.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/stress.hpp"

// The pair terms run once per pair and are called from the serial and the
// team walks. Out of line, the call costs as much as the arithmetic.
#if defined(__GNUC__) || defined(__clang__)
#define RGPOT_SC_PAIR_INLINE __attribute__((always_inline))
#else
#define RGPOT_SC_PAIR_INLINE
#endif

namespace {

double ipow(double base, int exp) {
  double result = 1.0;
  while (exp > 0) {
    if ((exp & 1) != 0) {
      result *= base;
    }
    base *= base;
    exp >>= 1;
  }
  return result;
}

/// (a/r)^exp from ar2 = (a/r)^2.
double pow_ar(double ar2, int exp) {
  double p = ipow(ar2, exp / 2);
  if ((exp % 2) != 0) {
    p *= std::sqrt(ar2);
  }
  return p;
}

/// cutoff <= 0 keeps every pair. The unbounded search walks them all
/// without bins or a list.
double searchCutoff(const rgpot::SuttonChenConfig &cfg) {
  return cfg.cutoff > 0.0 ? cfg.cutoff : std::numeric_limits<double>::infinity();
}

rgpot::SuttonChenConfig config_from_c(const rgpot_sc_config &raw) {
  rgpot::SuttonChenConfig c;
  c.epsilon = raw.epsilon;
  c.c = raw.c;
  c.a = raw.a;
  c.n = raw.n;
  c.m = raw.m;
  c.cutoff = raw.cutoff;
  c.periodic = raw.periodic != 0;
  return c;
}

} // namespace

namespace rgpot {

SuttonChenPot::SuttonChenPot(const SuttonChenConfig &c)
    : Potential(PotType::SuttonChen), m_config{c} {
  if (c.n < 1 || c.m < 1) {
    throw std::invalid_argument("Sutton-Chen exponents n and m must be >= 1");
  }
  if (!(c.a > 0.0) || !std::isfinite(c.epsilon) || !std::isfinite(c.c) ||
      !std::isfinite(c.cutoff)) {
    throw std::invalid_argument("Sutton-Chen parameters are not finite");
  }
  if (c.periodic && !(c.cutoff > 0.0)) {
    throw std::invalid_argument(
        "periodic Sutton-Chen needs a positive cutoff");
  }
  Fnv1a fp;
  fp.u64(kKernelVersion);
  fp.f64(c.epsilon);
  fp.f64(c.c);
  fp.f64(c.a);
  fp.u64(static_cast<uint64_t>(c.n));
  fp.u64(static_cast<uint64_t>(c.m));
  fp.f64(c.cutoff);
  fp.u64(c.periodic ? 1 : 0);
  if (c.stress) {
    // Mixed in only when set, so the default key keeps its old value.
    fp.u64(1);
  }
  m_paramsKey = fp.h;
}

void SuttonChenPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  compute(in, out, nullptr);
}

void SuttonChenPot::forcePerAtom(const ForceInput &in, ForceOut *out,
                                 const PerAtomOut &atoms) const {
  clearPerAtom(atoms, in.nAtoms);
  compute(in, out, &atoms);
}

void SuttonChenPot::compute(const ForceInput &in, ForceOut *out,
                            const PerAtomOut *atoms) const {
  const auto n_atoms = static_cast<long>(in.nAtoms);
  const double *R = in.pos;
  double *F = out->F;
  out->energy = 0.0;
  out->variance = 0.0;
  for (long i = 0; i < n_atoms; ++i) {
    F[3 * i] = 0.0;
    F[3 * i + 1] = 0.0;
    F[3 * i + 2] = 0.0;
  }
  const SuttonChenConfig cfg = m_config;
  double strain[6] = {};
  if (n_atoms < 2) {
    if (cfg.stress) {
      publishCauchyStress(out, strain, cellVolume(in.box));
    }
    return;
  }

  const double cutoff = searchCutoff(cfg);
  const double cutoff2 = cutoff * cutoff;
  const std::array<bool, 3> periodic =
      cfg.periodic ? std::array<bool, 3>{true, true, true}
                   : std::array<bool, 3>{false, false, false};

  static constexpr double free_box[9] = {1.0e6, 0.0, 0.0, 0.0, 1.0e6,
                                         0.0,   0.0, 0.0, 1.0e6};
  const bool box_ok = in.box != nullptr && in.box[0] > 0.0 && in.box[4] > 0.0 &&
                      in.box[8] > 0.0;
  if (cfg.periodic && !box_ok) {
    throw std::invalid_argument("periodic Sutton-Chen needs a real cell");
  }
  const double *box_use = box_ok ? in.box : free_box;

  const double epsilon = cfg.epsilon;
  const double c_coef = cfg.c;
  const double lattice = cfg.a;
  const int n_exp = cfg.n;
  const int m_exp = cfg.m;
  const auto n_size = static_cast<std::size_t>(n_atoms);
  double *atom_e = atoms != nullptr ? atoms->energy : nullptr;

  rho_.assign(n_size, 0.0);
  double *rho = rho_.data();
  double epair = 0.0;
  const double skin = mode_ == nlist::Mode::Stream ? 0.0 : cutoff / 5.0;
  nlist::updateCells(neighbours_, R, n_size, box_use, cutoff, skin, nlist::Image::Round,
                     mode_, periodic);
  const bool stream = neighbours_.streaming();

  // From kTeamMinAtoms up, each atom keeps the pair energy and strain of
  // its partners above it and the totals add them in atom order, as in
  // nlist/PairSum.hpp. A team, or a stream, then walks the rows of the
  // atoms; one thread on a stored list or an unbounded search keeps the
  // half walks, whose densities and forces the rows repeat bit for bit.
  const bool rows = n_size >= kTeamMinAtoms;
  const int nt = nlist::teamSize(n_size);
  const bool by_row = rows && (nt > 1 || (stream && !neighbours_.allPairs()));
  std::vector<double> row_e(rows ? n_size : 0, 0.0);
  std::vector<double> row_s(rows && cfg.stress ? 6 * n_size : 0, 0.0);
  double *e_sum = rows ? row_e.data() : &epair;
  double *s_sum = !cfg.stress ? nullptr : (rows ? row_s.data() : strain);
  // The half walks take the stride of the sums as a type, so below
  // kTeamMinAtoms every pair adds to the one total with no index.
  auto by_stride = [rows](auto walk) {
    if (rows) {
      walk(std::integral_constant<std::size_t, 1>{});
    } else {
      walk(std::integral_constant<std::size_t, 0>{});
    }
  };

  // Each walk visits the same pairs in the same order, one lane per
  // slice, so the force pass reads back what the density pass kept.
  const auto lanes = static_cast<std::size_t>(by_row ? nt : 1);
  if (terms_.size() < lanes) {
    terms_.resize(lanes);
  }
  std::size_t expected = 0;
  if (!by_row) {
    expected = neighbours_.allPairs() ? n_size * (n_size - 1) / 2
               : stream               ? 0
                                      : neighbours_.slots() / 2;
  }
  for (std::size_t t = 0; t < lanes; ++t) {
    terms_[t].setLimit(term_limit_ / lanes);
    terms_[t].record(t == 0 ? expected : 0);
  }
  // The force takes the powers of a^2 * (1/r^2), the density those of
  // a^2 / r^2. With a^2 = 1 they are one number; otherwise the last bit
  // can differ, and the kept powers are the force's.
  const double a2 = lattice * lattice;
  const bool unit_a = a2 == 1.0;
  const double en = epsilon * static_cast<double>(n_exp);
  const double em = static_cast<double>(m_exp);
  // The pair lambdas copy their constants and arrays: a walk that copies
  // its callback then keeps them in registers. Only the sums and the
  // lanes are shared by reference.
  auto pair_term = [=](double r2, double &phi, double &rep, Terms &terms) RGPOT_SC_PAIR_INLINE {
    double *kept = terms.push();
    if (unit_a) {
      const double inv_r2 = 1.0 / r2;
      phi = pow_ar(inv_r2, m_exp);
      rep = pow_ar(inv_r2, n_exp);
      if (kept != nullptr) {
        kept[0] = en * rep;
        kept[1] = em * phi;
        kept[2] = inv_r2;
      }
      return;
    }
    const double ar2 = a2 / r2;
    phi = pow_ar(ar2, m_exp);
    rep = pow_ar(ar2, n_exp);
    if (kept != nullptr) {
      const double inv_r2 = 1.0 / r2;
      const double ar2_force = a2 * inv_r2;
      const bool same = ar2_force == ar2;
      kept[0] = en * (same ? rep : pow_ar(ar2_force, n_exp));
      kept[1] = em * (same ? phi : pow_ar(ar2_force, m_exp));
      kept[2] = inv_r2;
    }
  };
  if (!by_row) {
    Terms &lane = terms_[0];
    by_stride([&](auto stride) {
      nlist::visitHalf(neighbours_,
          0, n_atoms,
          [=, &lane](int32_t i, int32_t j, double, double, double,
                     double r2) RGPOT_SC_PAIR_INLINE {
            if (!(r2 > 0.0) || r2 > cutoff2) {
              return;
            }
            double phi = 0.0;
            double rep = 0.0;
            pair_term(r2, phi, rep, lane);
            rho[static_cast<std::size_t>(i)] += phi;
            rho[static_cast<std::size_t>(j)] += phi;
            e_sum[static_cast<std::size_t>(i) * decltype(stride)::value] += epsilon * rep;
            if (atom_e != nullptr) {
              atom_e[static_cast<std::size_t>(i)] += 0.5 * epsilon * rep;
              atom_e[static_cast<std::size_t>(j)] += 0.5 * epsilon * rep;
            }
          });
    });
  } else {
    Terms *lane_terms = terms_.data();
    // Each row belongs to one slice, so only that thread writes its atom.
    neighbours_.forRowTeam(
        nt,
        [=](int t, int32_t i, int32_t j, double, double, double, double r2) RGPOT_SC_PAIR_INLINE {
          if (!(r2 > 0.0) || r2 > cutoff2) {
            return;
          }
          double phi = 0.0;
          double rep = 0.0;
          pair_term(r2, phi, rep, lane_terms[t]);
          rho[static_cast<std::size_t>(i)] += phi;
          if (j > i) {
            e_sum[static_cast<std::size_t>(i)] += epsilon * rep;
          }
          if (atom_e != nullptr) {
            atom_e[static_cast<std::size_t>(i)] += 0.5 * epsilon * rep;
          }
        });
  }

  weight_.assign(n_size, 0.0);
  double *weight = weight_.data();
  double eemb = 0.0;
  for (std::size_t i = 0; i < n_size; ++i) {
    if (!(rho[i] > 0.0)) {
      continue;
    }
    const double root = std::sqrt(rho[i]);
    eemb -= epsilon * c_coef * root;
    weight[i] = epsilon * c_coef / (2.0 * root);
    if (atom_e != nullptr) {
      atom_e[i] -= epsilon * c_coef * root;
    }
  }

  for (std::size_t t = 0; t < lanes; ++t) {
    terms_[t].replay();
  }
  auto apply = [=](auto stride, int32_t i, int32_t j, double dx, double dy, double dz, double r2,
                   double *force, double *virial, bool both, Terms &terms) RGPOT_SC_PAIR_INLINE {
    if (!(r2 > 0.0) || r2 > cutoff2) {
      return;
    }
    double rep_term = 0.0;
    double phi_term = 0.0;
    double inv_r2 = 0.0;
    if (const double *kept = terms.next()) {
      rep_term = kept[0];
      phi_term = kept[1];
      inv_r2 = kept[2];
    } else {
      inv_r2 = 1.0 / r2;
      const double ar2 = a2 * inv_r2;
      rep_term = en * pow_ar(ar2, n_exp);
      phi_term = em * pow_ar(ar2, m_exp);
    }
    const double wi = weight[static_cast<std::size_t>(i)];
    const double wj = weight[static_cast<std::size_t>(j)];
    const double fscale = (rep_term - phi_term * (wi + wj)) * inv_r2;
    const double fx = fscale * dx;
    const double fy = fscale * dy;
    const double fz = fscale * dz;
    force[3 * i] += fx;
    force[3 * i + 1] += fy;
    force[3 * i + 2] += fz;
    if (both) {
      force[3 * j] -= fx;
      force[3 * j + 1] -= fy;
      force[3 * j + 2] -= fz;
    }
    if (virial != nullptr && j > i) {
      accumulatePairStrain(virial + 6 * static_cast<std::size_t>(i) * decltype(stride)::value,
                           fscale, dx, dy, dz);
    }
    if (atoms != nullptr) {
      tallyAtomVirial(*atoms, static_cast<std::size_t>(i), 0.5, fscale, dx, dy, dz);
      if (both) {
        tallyAtomVirial(*atoms, static_cast<std::size_t>(j), 0.5, fscale, dx, dy, dz);
      }
    }
  };
  if (!by_row) {
    Terms &lane = terms_[0];
    double *virial = s_sum;
    by_stride([&](auto stride) {
      nlist::visitHalf(neighbours_,
          0, n_atoms,
          [=, &lane](int32_t i, int32_t j, double dx, double dy, double dz,
                     double r2) RGPOT_SC_PAIR_INLINE {
            apply(stride, i, j, dx, dy, dz, r2, F, virial, true, lane);
          });
    });
  } else {
    double *virial = s_sum;
    Terms *lane_terms = terms_.data();
    // Each row belongs to one slice, so its atom's force is written once.
    neighbours_.forRowTeam(
        nt,
        [=](int t, int32_t i, int32_t j, double dx, double dy, double dz,
            double r2) RGPOT_SC_PAIR_INLINE {
          apply(std::integral_constant<std::size_t, 1>{}, i, j, dx, dy, dz, r2, F, virial, false,
                lane_terms[t]);
        });
  }
  if (rows) {
    for (std::size_t i = 0; i < n_size; ++i) {
      epair += row_e[i];
    }
    if (cfg.stress) {
      for (std::size_t i = 0; i < n_size; ++i) {
        for (std::size_t k = 0; k < 6; ++k) {
          strain[k] += row_s[6 * i + k];
        }
      }
    }
  }

  out->energy = epair + eemb;
  if (cfg.stress) {
    publishCauchyStress(out, strain, cellVolume(in.box));
  }
}

void SuttonChenPot::densityStage(const HalfList &list, double *densities) const {
  const SuttonChenConfig &cfg = m_config;
  const double cutoff = searchCutoff(cfg);
  const double cutoff2 = cutoff * cutoff;
  const double a2 = cfg.a * cfg.a;
  staged::forEachPair(list, [&](int i, int j, double, double, double, double r2, bool both) {
    if (!(r2 > 0.0) || r2 > cutoff2) {
      return;
    }
    const double phi = pow_ar(a2 / r2, cfg.m);
    densities[i] += phi;
    if (both) {
      densities[j] += phi;
    }
  });
}

void SuttonChenPot::forceStage(const HalfList &list, const double *densities, double *forces,
                               StageTally *tally) const {
  const SuttonChenConfig &cfg = m_config;
  const double cutoff = searchCutoff(cfg);
  const double cutoff2 = cutoff * cutoff;
  const double a2 = cfg.a * cfg.a;
  const double ec = cfg.epsilon * cfg.c;
  std::vector<double> weight(list.nAtoms, 0.0);
  for (std::size_t i = 0; i < list.nAtoms; ++i) {
    if (densities[i] > 0.0) {
      weight[i] = ec / (2.0 * std::sqrt(densities[i]));
    }
  }
  if (tally != nullptr) {
    for (std::size_t k = 0; k < list.nRows; ++k) {
      const int i = list.rows[k];
      if (densities[i] > 0.0) {
        staged::addSite(*tally, i, -(ec * std::sqrt(densities[i])));
      }
    }
  }
  const double n_exp = static_cast<double>(cfg.n);
  const double m_exp = static_cast<double>(cfg.m);
  staged::withTally(tally, [&](auto tallied) {
    staged::forEachPair(
        list, [&](int i, int j, double dx, double dy, double dz, double r2, bool both) {
          if (!(r2 > 0.0) || r2 > cutoff2) {
            return;
          }
          const double inv_r2 = 1.0 / r2;
          const double ar2 = a2 * inv_r2;
          const double phi = pow_ar(ar2, cfg.m);
          const double rep = pow_ar(ar2, cfg.n);
          const double wi = weight[static_cast<std::size_t>(i)];
          const double wj = weight[static_cast<std::size_t>(j)];
          const double fscale = (cfg.epsilon * n_exp * rep - m_exp * phi * (wi + wj)) * inv_r2;
          staged::addPair(tallied, forces, tally, i, j, both, cfg.epsilon * rep, fscale, dx,
                          dy, dz);
        });
  });
}

} // namespace rgpot

namespace {

int export_force(const rgpot_sc_config *cfg, int32_t natoms,
                 const double *positions, const double *box, double *energy,
                 double *forces) {
  if (cfg == nullptr || natoms < 0 || energy == nullptr ||
      (natoms > 0 && (positions == nullptr || forces == nullptr))) {
    return 1;
  }
  const auto n = static_cast<std::size_t>(natoms);
  // The potential reads no species. Its neighbour update and pair terms
  // carry over to the next call with the same parameters and atom count.
  rgpot::KeptKey key;
  key.f64(cfg->epsilon)
      .f64(cfg->c)
      .f64(cfg->a)
      .i64(cfg->n)
      .i64(cfg->m)
      .f64(cfg->cutoff)
      .i64(cfg->periodic != 0 ? 1 : 0)
      .atoms(n, nullptr);
  try {
    *energy = rgpot::Kept<rgpot::SuttonChenPot>::with(
        key, [&] { return std::make_unique<rgpot::SuttonChenPot>(config_from_c(*cfg)); },
        [&](const rgpot::SuttonChenPot &pot) {
          rgpot::ForceOut out{};
          out.F = forces;
          out.energy = 0.0;
          out.variance = 0.0;
          out.has_stress = 0;
          const rgpot::ForceInput in{n, positions, nullptr, box};
          pot.forceImpl(in, &out);
          return out.energy;
        });
  } catch (const std::exception &) {
    return 1;
  }
  return 0;
}

} // namespace

extern "C" RGPOT_SC_API int
rgpot_sutton_chen_energy_forces(const rgpot_sc_config *cfg, int32_t natoms,
                        const double *positions, const double *box,
                        double *energy, double *forces) {
  return export_force(cfg, natoms, positions, box, energy, forces);
}
