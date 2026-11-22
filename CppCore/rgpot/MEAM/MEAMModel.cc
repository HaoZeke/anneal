// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief MEAM parameter bookkeeping, reference structures and pair tables.
 *
 * The reference-structure formulas follow I. Huang, M. I. Baskes and
 * coworkers (Modelling Simul. Mater. Sci. Eng. 3, 615 (1995)), which the
 * LAMMPS MEAM package (src/MEAM/meam_setup_done.cpp, meam_funcs.cpp)
 * implements: each element's pair term is fixed by requiring the energy of
 * its reference structure to be the Rose equation of state.
 */

// clang-format off
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
// clang-format on

#include "rgpot/MEAM/MEAMModel.hpp"
#include "rgpot/ParamHash.hpp"

namespace rgpot {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTiny = 1e-20;

bool is_zero(double f) { return std::fabs(f) < kTiny; }

[[noreturn]] void bad(const std::string &what) {
  throw std::invalid_argument("MEAM: " + what);
}

double square(double x) { return x * x; }
double cube(double x) { return x * x * x; }

double powint(double x, int n) {
  double r = 1.0;
  for (int i = 0; i < n; ++i) {
    r *= x;
  }
  return r;
}

/// Cutoff (1 - (1 - x)^4)^2 on [0, 1].
double fcut(double x) {
  if (x >= 1.0) {
    return 1.0;
  }
  if (x <= 0.0) {
    return 0.0;
  }
  double a = 1.0 - x;
  a *= a;
  a *= a;
  a = 1.0 - a;
  return a * a;
}

/// ZBL screened nuclear repulsion in eV for atomic numbers z1, z2.
double zbl(double r, int z1, int z2) {
  constexpr double c[] = {0.028171, 0.28022, 0.50986, 0.18175};
  constexpr double d[] = {0.20162, 0.40290, 0.94229, 3.1998};
  constexpr double azero = 0.4685;
  constexpr double cc = 14.3997;
  const double a = azero / (std::pow(z1, 0.23) + std::pow(z2, 0.23));
  const double x = r / a;
  double result = 0.0;
  for (int i = 0; i < 4; ++i) {
    result += c[i] * std::exp(-d[i] * x);
  }
  if (r > 0.0) {
    result = result * z1 * z2 / r * cc;
  }
  return result;
}

void shape_for(MEAMLattice latt, double sthe, double cthe, double s[3]) {
  switch (latt) {
  case MEAMLattice::FCC:
  case MEAMLattice::BCC:
  case MEAMLattice::B1:
  case MEAMLattice::B2:
  case MEAMLattice::SC:
    s[0] = s[1] = s[2] = 0.0;
    break;
  case MEAMLattice::HCP:
    s[0] = 0.0;
    s[1] = 0.0;
    s[2] = 1.0 / 3.0;
    break;
  case MEAMLattice::CH4:
  case MEAMLattice::DIA:
  case MEAMLattice::DIA3:
    s[0] = 0.0;
    s[1] = 0.0;
    s[2] = 32.0 / 9.0;
    break;
  case MEAMLattice::DIM:
    s[0] = 1.0;
    s[1] = 2.0 / 3.0;
    s[2] = 0.40;
    break;
  case MEAMLattice::LIN:
    s[0] = 0.0;
    s[1] = 8.0 / 3.0;
    s[2] = 0.0;
    break;
  case MEAMLattice::ZIG:
  case MEAMLattice::TRI:
    s[0] = 4.0 * std::pow(cthe, 2);
    s[1] = 4.0 * (std::pow(cthe, 4) + std::pow(sthe, 4) - 1.0 / 3.0);
    s[2] = 4.0 * (std::pow(cthe, 2) * (3 * std::pow(sthe, 4) + std::pow(cthe, 4)));
    s[2] -= 0.6 * s[0];
    break;
  default:
    s[0] = 0.0;
    break;
  }
}

} // namespace

bool meam_lattice_from_name(const std::string &name, bool single,
                            MEAMLattice *out) {
  struct Entry {
    const char *name;
    MEAMLattice lattice;
    bool single;
  };
  static constexpr Entry kNames[] = {
      {"fcc", MEAMLattice::FCC, true},   {"bcc", MEAMLattice::BCC, true},
      {"hcp", MEAMLattice::HCP, true},   {"dim", MEAMLattice::DIM, true},
      {"dia", MEAMLattice::DIA, true},   {"dia3", MEAMLattice::DIA3, true},
      {"lin", MEAMLattice::LIN, true},   {"zig", MEAMLattice::ZIG, true},
      {"tri", MEAMLattice::TRI, true},   {"sc", MEAMLattice::SC, true},
      {"b1", MEAMLattice::B1, false},    {"c11", MEAMLattice::C11, false},
      {"l12", MEAMLattice::L12, false},  {"b2", MEAMLattice::B2, false},
      {"ch4", MEAMLattice::CH4, false},
  };
  for (const Entry &e : kNames) {
    if (name == e.name && (e.single || !single)) {
      *out = e.lattice;
      return true;
    }
  }
  return false;
}

int meam_coordination(MEAMLattice lattice) {
  switch (lattice) {
  case MEAMLattice::FCC:
  case MEAMLattice::HCP:
  case MEAMLattice::L12:
    return 12;
  case MEAMLattice::BCC:
  case MEAMLattice::B2:
    return 8;
  case MEAMLattice::DIA:
  case MEAMLattice::DIA3:
  case MEAMLattice::CH4:
    return 4;
  case MEAMLattice::DIM:
    return 1;
  case MEAMLattice::B1:
  case MEAMLattice::SC:
    return 6;
  case MEAMLattice::C11:
    return 10;
  case MEAMLattice::LIN:
  case MEAMLattice::ZIG:
  case MEAMLattice::TRI:
    return 2;
  }
  return 0;
}

namespace {

/// Nearest-neighbour distance of an element from its lattice constant.
double neighbour_distance(MEAMLattice lattice, double alat) {
  switch (lattice) {
  case MEAMLattice::FCC:
    return alat / std::sqrt(2.0);
  case MEAMLattice::BCC:
    return alat * std::sqrt(3.0) / 2.0;
  case MEAMLattice::DIA:
  case MEAMLattice::DIA3:
    return alat * std::sqrt(3.0) / 4.0;
  default:
    return alat;
  }
}

} // namespace

const char *MEAMParameters::keyword(Key key) {
  static constexpr const char *kKeywords[] = {
      "Ec",     "alpha",          "rho0",          "delta",
      "lattce", "attrac",         "repuls",        "nn2",
      "Cmin",   "Cmax",           "rc",            "delr",
      "augt1",  "gsmooth_factor", "re",            "ialloy",
      "mixture_ref_t", "erose_form", "zbl",        "emb_lin_neg",
      "bkgd_dyn", "theta"};
  return kKeywords[static_cast<int>(key)];
}

bool MEAMParameters::key_from_keyword(const std::string &word, Key *out) {
  for (int k = 0; k <= static_cast<int>(Key::theta); ++k) {
    if (word == keyword(static_cast<Key>(k))) {
      *out = static_cast<Key>(k);
      return true;
    }
  }
  return false;
}

MEAMParameters::MEAMParameters(std::vector<MEAMElement> elements)
    : m_elements(std::move(elements)) {
  const std::size_t n = m_elements.size();
  if (n == 0) {
    bad("no elements");
  }
  m_ec.assign(n * n, 0.0);
  m_re.assign(n * n, 0.0);
  m_alpha.assign(n * n, 0.0);
  m_delta.assign(n * n, 0.0);
  m_attrac.assign(n * n, 0.0);
  m_repuls.assign(n * n, 0.0);
  m_stheta.assign(n * n, 1.0);
  m_ctheta.assign(n * n, 0.0);
  m_lattice.assign(n * n, static_cast<int>(MEAMLattice::FCC));
  m_nn2.assign(n * n, 0);
  m_zbl.assign(n * n, 1);
  m_cmin.assign(n * n * n, 2.0);
  m_cmax.assign(n * n * n, 2.8);

  for (std::size_t i = 0; i < n; ++i) {
    const MEAMElement &e = m_elements[i];
    const int ii = static_cast<int>(i);
    m_lattice[pair(ii, ii)] = static_cast<int>(e.lattice);
    m_alpha[pair(ii, ii)] = e.alpha;
    m_ec[pair(ii, ii)] = e.esub;
    m_re[pair(ii, ii)] = neighbour_distance(e.lattice, e.alat);
    if (e.t[0] != 1.0) {
      bad("t0 of " + e.name + " is not 1");
    }
    if (!(e.rho0 > 0.0)) {
      bad("rho0 of " + e.name + " must be positive");
    }
    switch (e.lattice) {
    case MEAMLattice::B1:
    case MEAMLattice::B2:
    case MEAMLattice::C11:
    case MEAMLattice::L12:
    case MEAMLattice::CH4:
      bad("lattice of " + e.name + " is not an element lattice");
    default:
      break;
    }
  }
}

void MEAMParameters::set(Key key, double value, const std::vector<int> &index) {
  const int n = static_cast<int>(m_elements.size());
  auto need = [&](std::size_t count) {
    if (index.size() < count) {
      bad(std::string("keyword ") + keyword(key) + " expected more indices");
    }
    for (std::size_t c = 0; c < count; ++c) {
      if (index[c] < 0 || index[c] >= n) {
        bad(std::string("keyword ") + keyword(key) +
            " has out of range element index");
      }
    }
  };
  switch (key) {
  case Key::Ec:
    need(2);
    m_ec[pair(index[0], index[1])] = value;
    break;
  case Key::alpha:
    need(2);
    m_alpha[pair(index[0], index[1])] = value;
    break;
  case Key::rho0:
    need(1);
    if (!(value > 0.0)) {
      bad("rho0 must be > 0");
    }
    m_elements[static_cast<std::size_t>(index[0])].rho0 = value;
    break;
  case Key::delta:
    need(2);
    m_delta[pair(index[0], index[1])] = value;
    break;
  case Key::lattce:
    need(2);
    m_lattice[pair(index[0], index[1])] = static_cast<int>(value);
    break;
  case Key::attrac:
    need(2);
    m_attrac[pair(index[0], index[1])] = value;
    break;
  case Key::repuls:
    need(2);
    m_repuls[pair(index[0], index[1])] = value;
    break;
  case Key::nn2:
    need(2);
    m_nn2[pair(std::min(index[0], index[1]), std::max(index[0], index[1]))] =
        static_cast<int>(value);
    break;
  case Key::Cmin:
    need(3);
    m_cmin[triple(index[0], index[1], index[2])] = value;
    break;
  case Key::Cmax:
    need(3);
    m_cmax[triple(index[0], index[1], index[2])] = value;
    break;
  case Key::rc:
    m_rc = value;
    break;
  case Key::delr:
    m_delr = value;
    break;
  case Key::augt1:
    m_augt1 = static_cast<int>(value);
    break;
  case Key::gsmooth_factor:
    m_gsmooth = value;
    break;
  case Key::re:
    need(2);
    m_re[pair(index[0], index[1])] = value;
    break;
  case Key::ialloy:
    m_ialloy = static_cast<int>(value);
    break;
  case Key::mixture_ref_t:
    m_mix_ref_t = static_cast<int>(value);
    break;
  case Key::erose_form:
    m_erose_form = static_cast<int>(value);
    break;
  case Key::zbl:
    need(2);
    m_zbl[pair(std::min(index[0], index[1]), std::max(index[0], index[1]))] =
        static_cast<int>(value);
    break;
  case Key::emb_lin_neg:
    m_emb_lin_neg = static_cast<int>(value);
    break;
  case Key::bkgd_dyn:
    m_bkgd_dyn = static_cast<int>(value);
    break;
  case Key::theta: {
    need(2);
    const std::size_t at =
        pair(std::min(index[0], index[1]), std::max(index[0], index[1]));
    m_stheta[at] = std::sin(value / 2 * kPi / 180.0);
    m_ctheta[at] = std::cos(value / 2 * kPi / 180.0);
    break;
  }
  }
}

void MEAMModel::shape_factors(MEAMLattice lattice, double stheta,
                              double ctheta, double s[3]) {
  shape_for(lattice, stheta, ctheta, s);
}

bool MEAMModel::G(double gamma, int ibar, double *value, double *slope) const {
  switch (ibar) {
  case 0:
  case 4: {
    const double gs = m_p.m_gsmooth;
    const double switchpoint = -gs / (gs + 1);
    if (gamma < switchpoint) {
      if (gamma == 0.0) {
        *value = 0.0;
        *slope = 0.0;
        return true;
      }
      const double g = std::sqrt(1 / (gs + 1) * std::pow(switchpoint / gamma, gs));
      *value = g;
      *slope = -gs * g / (2.0 * gamma);
    } else {
      const double g = std::sqrt(1.0 + gamma);
      *value = g;
      *slope = 1.0 / (2.0 * g);
    }
    return true;
  }
  case 1: {
    const double g = std::exp(gamma / 2.0);
    *value = g;
    *slope = g / 2.0;
    return true;
  }
  case 3: {
    const double g = 2.0 / (1.0 + std::exp(-gamma));
    *value = g;
    *slope = g * (2.0 - g) / 2;
    return true;
  }
  case -5:
    if ((1.0 + gamma) >= 0) {
      const double g = std::sqrt(1.0 + gamma);
      *value = g;
      *slope = 1.0 / (2.0 * g);
    } else {
      const double g = -std::sqrt(-1.0 - gamma);
      *value = g;
      *slope = -1.0 / (2.0 * g);
    }
    return true;
  default:
    *value = 0.0;
    *slope = 1.0;
    return false;
  }
}

double MEAMModel::embedding(int element, double rhobar, double *slope) const {
  const MEAMElement &e = m_p.m_elements[static_cast<std::size_t>(element)];
  const double aec = e.asub * ec(element);
  if (rhobar > 0.0) {
    const double lrb = std::log(rhobar);
    *slope = aec * (1.0 + lrb);
    return aec * rhobar * lrb;
  }
  if (m_p.m_emb_lin_neg == 0) {
    *slope = 0.0;
    return 0.0;
  }
  *slope = -aec;
  return -aec * rhobar;
}

double MEAMModel::rose(double r, int i, int j) const {
  const std::size_t at = m_p.pair(i, j);
  const double reps = m_p.m_re[at];
  double result = 0.0;
  if (r > 0.0) {
    const double astar = m_p.m_alpha[at] * (r / reps - 1.0);
    double a3 = 0.0;
    if (astar >= 0) {
      a3 = m_p.m_attrac[at];
    } else if (astar < 0) {
      a3 = m_p.m_repuls[at];
    }
    const double ecij = m_p.m_ec[at];
    if (m_p.m_erose_form == 1) {
      result = -ecij *
               (1 + astar + (-m_p.m_attrac[at] + m_p.m_repuls[at] / r) * cube(astar)) *
               std::exp(-astar);
    } else if (m_p.m_erose_form == 2) {
      result = -ecij * (1 + astar + a3 * cube(astar)) * std::exp(-astar);
    } else {
      result = -ecij * (1 + astar + a3 * cube(astar) / (r / reps)) * std::exp(-astar);
    }
  }
  return result;
}

MEAMModel::MEAMModel(MEAMParameters parameters) : m_p(std::move(parameters)) {
  m_n = m_p.m_elements.size();
  if (!(m_p.m_rc > 0.0) || !std::isfinite(m_p.m_rc)) {
    bad("rc must be positive");
  }
  if (!(m_p.m_delr > 0.0) || !std::isfinite(m_p.m_delr)) {
    bad("delr must be positive");
  }
  for (std::size_t i = 0; i < m_n; ++i) {
    double g = 0.0;
    double dg = 0.0;
    if (!G(0.0, m_p.m_elements[i].ibar, &g, &dg)) {
      bad("ibar " + std::to_string(m_p.m_elements[i].ibar) + " of " +
          m_p.m_elements[i].name + " is not implemented");
    }
  }

  m_t.resize(m_n);
  for (std::size_t i = 0; i < m_n; ++i) {
    for (int l = 0; l < 4; ++l) {
      m_t[i][static_cast<std::size_t>(l)] = m_p.m_elements[i].t[l];
    }
    m_t[i][1] += m_p.m_augt1 * 3.0 / 5.0 * m_t[i][3];
  }

  mix_pairs();

  for (std::size_t i = 0; i < m_n; ++i) {
    for (std::size_t j = 0; j < m_n; ++j) {
      const std::size_t ij = m_p.pair(static_cast<int>(i), static_cast<int>(j));
      if (m_p.m_nn2[ij] == 1 &&
          static_cast<MEAMLattice>(m_p.m_lattice[ij]) == MEAMLattice::C11) {
        bad("nn2 = 1 is not defined for the c11 lattice");
      }
    }
  }

  std::size_t at = 0;
  m_eltind.assign(m_n * m_n, 0);
  for (std::size_t a = 0; a < m_n; ++a) {
    for (std::size_t b = a; b < m_n; ++b) {
      m_eltind[a * m_n + b] = at;
      m_eltind[b * m_n + a] = at;
      ++at;
    }
  }

  reference_density();
  build_pair_tables();

  Fnv1a h;
  h.u64(m_n);
  for (const MEAMElement &e : m_p.m_elements) {
    h.str(e.name);
    h.i64(static_cast<int>(e.lattice));
    h.i64(e.z);
    h.f64(e.alpha);
    for (double b : e.beta) {
      h.f64(b);
    }
    h.f64(e.alat);
    h.f64(e.esub);
    h.f64(e.asub);
    for (double t : e.t) {
      h.f64(t);
    }
    h.f64(e.rho0);
    h.i64(e.ibar);
  }
  for (double v : m_p.m_ec) h.f64(v);
  for (double v : m_p.m_re) h.f64(v);
  for (double v : m_p.m_alpha) h.f64(v);
  for (double v : m_p.m_delta) h.f64(v);
  for (double v : m_p.m_attrac) h.f64(v);
  for (double v : m_p.m_repuls) h.f64(v);
  for (double v : m_p.m_stheta) h.f64(v);
  for (double v : m_p.m_ctheta) h.f64(v);
  for (int v : m_p.m_lattice) h.i64(v);
  for (int v : m_p.m_nn2) h.i64(v);
  for (int v : m_p.m_zbl) h.i64(v);
  for (double v : m_p.m_cmin) h.f64(v);
  for (double v : m_p.m_cmax) h.f64(v);
  h.f64(m_p.m_rc);
  h.f64(m_p.m_delr);
  h.f64(m_p.m_gsmooth);
  h.i64(m_p.m_augt1);
  h.i64(m_p.m_ialloy);
  h.i64(m_p.m_mix_ref_t);
  h.i64(m_p.m_erose_form);
  h.i64(m_p.m_emb_lin_neg);
  h.i64(m_p.m_bkgd_dyn);
  m_fingerprint = h.h;
}

void MEAMModel::mix_pairs() {
  const int n = static_cast<int>(m_n);
  MEAMParameters &p = m_p;

  // Off-diagonal pairs: the i > j entries copy i < j, and a zero i < j
  // entry takes the mean of the elements or the lattice rule.
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const std::size_t ij = p.pair(i, j);
      const std::size_t ji = p.pair(j, i);
      if (i > j) {
        p.m_re[ij] = p.m_re[ji];
        p.m_ec[ij] = p.m_ec[ji];
        p.m_alpha[ij] = p.m_alpha[ji];
        p.m_lattice[ij] = p.m_lattice[ji];
        p.m_nn2[ij] = p.m_nn2[ji];
        p.m_stheta[ij] = p.m_stheta[ji];
        p.m_ctheta[ij] = p.m_ctheta[ji];
      } else if (j > i) {
        const std::size_t ii = p.pair(i, i);
        const std::size_t jj = p.pair(j, j);
        if (is_zero(p.m_ec[ij])) {
          const auto latt = static_cast<MEAMLattice>(p.m_lattice[ij]);
          if (latt == MEAMLattice::L12) {
            p.m_ec[ij] = (3 * p.m_ec[ii] + p.m_ec[jj]) / 4.0 - p.m_delta[ij];
          } else if (latt == MEAMLattice::C11) {
            if (static_cast<MEAMLattice>(p.m_lattice[ii]) == MEAMLattice::DIA) {
              p.m_ec[ij] = (2 * p.m_ec[ii] + p.m_ec[jj]) / 3.0 - p.m_delta[ij];
            } else {
              p.m_ec[ij] = (p.m_ec[ii] + 2 * p.m_ec[jj]) / 3.0 - p.m_delta[ij];
            }
          } else {
            p.m_ec[ij] = (p.m_ec[ii] + p.m_ec[jj]) / 2.0 - p.m_delta[ij];
          }
        }
        if (is_zero(p.m_alpha[ij])) {
          p.m_alpha[ij] = (p.m_alpha[ii] + p.m_alpha[jj]) / 2.0;
        }
        if (is_zero(p.m_re[ij])) {
          p.m_re[ij] = (p.m_re[ii] + p.m_re[jj]) / 2.0;
        }
      }
    }
  }

  // Cmin and Cmax are symmetric in the first two indices.
  for (int i = 1; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      for (int k = 0; k < n; ++k) {
        p.m_cmin[p.triple(i, j, k)] = p.m_cmin[p.triple(j, i, k)];
        p.m_cmax[p.triple(i, j, k)] = p.m_cmax[p.triple(j, i, k)];
      }
    }
  }

  // Squared-distance bound, in units of r_ij^2, outside which a third atom
  // cannot screen the pair i, j: the largest r_ik^2 on the ellipse
  // C = Cmax, maximised over the third element. Below Cmax = 2 the ellipse
  // never reaches past r_ik = r_ij. The LAMMPS package uses
  // Cmax^2 / (4 (Cmax - 1)) for every Cmax, which is the same for Cmax of 2
  // or more and only looser below, and unbounded at Cmax = 1.
  m_ebound.assign(m_n * m_n, 1.0);
  double widest = 1.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double &bound = m_ebound[p.pair(i, j)];
      for (int k = 0; k < n; ++k) {
        const double cmax = p.m_cmax[p.triple(i, j, k)];
        if (cmax >= 2.0) {
          bound = std::max(bound, (cmax * cmax) / (4.0 * (cmax - 1.0)));
        }
      }
      widest = std::max(widest, bound);
    }
  }
  m_radius = p.m_rc * std::sqrt(widest);
}

int MEAMModel::second_neighbours(MEAMLattice latt, double cmin, double cmax,
                                 double stheta, double *a, double *s) const {
  int zij2 = 0;
  int numscr = 0;
  switch (latt) {
  case MEAMLattice::FCC:
    zij2 = 6;
    *a = std::sqrt(2.0);
    numscr = 4;
    break;
  case MEAMLattice::BCC:
    zij2 = 6;
    *a = 2.0 / std::sqrt(3.0);
    numscr = 4;
    break;
  case MEAMLattice::HCP:
    zij2 = 6;
    *a = std::sqrt(2.0);
    numscr = 4;
    break;
  case MEAMLattice::B1:
  case MEAMLattice::SC:
    zij2 = 12;
    *a = std::sqrt(2.0);
    numscr = 2;
    break;
  case MEAMLattice::DIA:
    zij2 = 12;
    *a = std::sqrt(8.0 / 3.0);
    numscr = 1;
    break;
  case MEAMLattice::DIA3:
    zij2 = 12;
    *a = std::sqrt(11.0 / 3.0);
    numscr = 4;
    break;
  case MEAMLattice::CH4:
  case MEAMLattice::LIN:
  case MEAMLattice::DIM:
    *a = 1.0;
    *s = 0.0;
    return 0;
  case MEAMLattice::TRI:
    zij2 = 1;
    *a = 2.0 * stheta;
    numscr = 2;
    break;
  case MEAMLattice::ZIG:
    zij2 = 2;
    *a = 2.0 * stheta;
    numscr = 1;
    break;
  case MEAMLattice::L12:
    zij2 = 6;
    *a = std::sqrt(2.0);
    numscr = 4;
    break;
  case MEAMLattice::B2:
    zij2 = 6;
    *a = 2.0 / std::sqrt(3.0);
    numscr = 4;
    break;
  case MEAMLattice::C11:
    *a = 1.0;
    break;
  }
  const double c = latt == MEAMLattice::DIA3 ? 1.0 : 4.0 / (*a * *a) - 1.0;
  const double x = (c - cmin) / (cmax - cmin);
  *s = powint(fcut(x), numscr);
  return zij2;
}

int MEAMModel::second_neighbours_b2nn(MEAMLattice latt, double cmin,
                                      double cmax, double *s) const {
  int numscr = 0;
  int zij2 = 0;
  if (latt == MEAMLattice::ZIG || latt == MEAMLattice::TRI) {
    zij2 = 2;
    numscr = 1;
  }
  const double x = (1.0 - cmin) / (cmax - cmin);
  *s = powint(fcut(x), numscr);
  return zij2;
}

double MEAMModel::screening(int i, int j, int k, double c) const {
  const double x = (c - cmin(i, j, k)) / (cmax(i, j, k) - cmin(i, j, k));
  return fcut(x);
}

void MEAMModel::reference_density() {
  m_rho_ref.assign(m_n, 0.0);
  for (std::size_t ia = 0; ia < m_n; ++ia) {
    const int a = static_cast<int>(ia);
    const MEAMLattice latt = lattice(a, a);
    const int z = meam_coordination(latt);
    double gbar = 1.0;
    if (m_p.m_elements[ia].ibar > 0) {
      double shp[3];
      shape_for(latt, stheta(a, a), ctheta(a, a), shp);
      const double gam =
          (m_t[ia][1] * shp[0] + m_t[ia][2] * shp[1] + m_t[ia][3] * shp[2]) / (z * z);
      double dg = 0.0;
      static_cast<void>(G(gam, m_p.m_elements[ia].ibar, &gbar, &dg));
    }
    double rho0 = m_p.m_elements[ia].rho0 * z;
    if (m_p.m_nn2[m_p.pair(a, a)] == 1) {
      double arat = 0.0;
      double scrn = 0.0;
      const int z2 = second_neighbours(latt, cmin(a, a, a), cmax(a, a, a),
                                       stheta(a, a), &arat, &scrn);
      const double rho0_2nn =
          m_p.m_elements[ia].rho0 * std::exp(-m_p.m_elements[ia].beta[0] * (arat - 1.0));
      rho0 += z2 * rho0_2nn * scrn;
    }
    m_rho_ref[ia] = rho0 * gbar;
  }
}

void MEAMModel::reference_t(double *t11, double *t21, double *t31, double *t12,
                            double *t22, double *t32, double ta1, double ta2,
                            double ta3, double tb1, double tb2, double tb3,
                            double r, int a, int b, MEAMLattice latt) const {
  if (m_p.m_ialloy == 2) {
    *t11 = ta1;
    *t21 = ta2;
    *t31 = ta3;
    *t12 = tb1;
    *t22 = tb2;
    *t32 = tb3;
    return;
  }
  switch (latt) {
  case MEAMLattice::FCC:
  case MEAMLattice::BCC:
  case MEAMLattice::DIA:
  case MEAMLattice::DIA3:
  case MEAMLattice::HCP:
  case MEAMLattice::B1:
  case MEAMLattice::DIM:
  case MEAMLattice::B2:
  case MEAMLattice::CH4:
  case MEAMLattice::LIN:
  case MEAMLattice::ZIG:
  case MEAMLattice::TRI:
  case MEAMLattice::SC:
    // Every neighbour is of the other element.
    *t11 = tb1;
    *t21 = tb2;
    *t31 = tb3;
    *t12 = ta1;
    *t22 = ta2;
    *t32 = ta3;
    break;
  default: {
    const double a1 = r / re(a) - 1.0;
    const double a2 = r / re(b) - 1.0;
    const double rhoa01 = rho0(a) * std::exp(-beta(a, 0) * a1);
    const double rhoa02 = rho0(b) * std::exp(-beta(b, 0) * a2);
    if (latt == MEAMLattice::L12) {
      const double rho01 = 8 * rhoa01 + 4 * rhoa02;
      *t12 = ta1;
      *t22 = ta2;
      *t32 = ta3;
      if (rho01 > 0.0) {
        *t11 = (8 * ta1 * rhoa01 + 4 * tb1 * rhoa02) / rho01;
        *t21 = (8 * ta2 * rhoa01 + 4 * tb2 * rhoa02) / rho01;
        *t31 = (8 * ta3 * rhoa01 + 4 * tb3 * rhoa02) / rho01;
      } else {
        *t11 = (2.0 * ta1 + tb1) / 3.0;
        *t21 = (2.0 * ta2 + tb2) / 3.0;
        *t31 = (2.0 * ta3 + tb3) / 3.0;
      }
    }
    break;
  }
  }
}

void MEAMModel::reference_densities(double r, int a, int b, double *rho01,
                                    double *rho11, double *rho21, double *rho31,
                                    double *rho02, double *rho12, double *rho22,
                                    double *rho32) const {
  const double a1 = r / re(a) - 1.0;
  const double a2 = r / re(b) - 1.0;

  const double rhoa01 = rho0(a) * std::exp(-beta(a, 0) * a1);
  const double rhoa11 = rho0(a) * std::exp(-beta(a, 1) * a1);
  const double rhoa21 = rho0(a) * std::exp(-beta(a, 2) * a1);
  const double rhoa31 = rho0(a) * std::exp(-beta(a, 3) * a1);
  const double rhoa02 = rho0(b) * std::exp(-beta(b, 0) * a2);
  const double rhoa12 = rho0(b) * std::exp(-beta(b, 1) * a2);
  const double rhoa22 = rho0(b) * std::exp(-beta(b, 2) * a2);
  const double rhoa32 = rho0(b) * std::exp(-beta(b, 3) * a2);

  const MEAMLattice lat = lattice(a, b);
  const int zij = meam_coordination(lat);
  double s[3];

  *rho11 = *rho21 = *rho31 = 0.0;
  *rho12 = *rho22 = *rho32 = 0.0;

  switch (lat) {
  case MEAMLattice::FCC:
    *rho01 = 12.0 * rhoa02;
    *rho02 = 12.0 * rhoa01;
    break;
  case MEAMLattice::BCC:
    *rho01 = 8.0 * rhoa02;
    *rho02 = 8.0 * rhoa01;
    break;
  case MEAMLattice::B1:
  case MEAMLattice::SC:
    *rho01 = 6.0 * rhoa02;
    *rho02 = 6.0 * rhoa01;
    break;
  case MEAMLattice::DIA:
  case MEAMLattice::DIA3:
    *rho01 = 4.0 * rhoa02;
    *rho02 = 4.0 * rhoa01;
    *rho31 = 32.0 / 9.0 * rhoa32 * rhoa32;
    *rho32 = 32.0 / 9.0 * rhoa31 * rhoa31;
    break;
  case MEAMLattice::HCP:
    *rho01 = 12 * rhoa02;
    *rho02 = 12 * rhoa01;
    *rho31 = 1.0 / 3.0 * rhoa32 * rhoa32;
    *rho32 = 1.0 / 3.0 * rhoa31 * rhoa31;
    break;
  case MEAMLattice::DIM:
    shape_for(MEAMLattice::DIM, 0, 0, s);
    *rho01 = rhoa02;
    *rho02 = rhoa01;
    *rho11 = s[0] * rhoa12 * rhoa12;
    *rho12 = s[0] * rhoa11 * rhoa11;
    *rho21 = s[1] * rhoa22 * rhoa22;
    *rho22 = s[1] * rhoa21 * rhoa21;
    *rho31 = s[2] * rhoa32 * rhoa32;
    *rho32 = s[2] * rhoa31 * rhoa31;
    break;
  case MEAMLattice::C11:
    *rho01 = rhoa01;
    *rho02 = rhoa02;
    *rho11 = rhoa11;
    *rho12 = rhoa12;
    *rho21 = rhoa21;
    *rho22 = rhoa22;
    *rho31 = rhoa31;
    *rho32 = rhoa32;
    break;
  case MEAMLattice::L12:
    *rho01 = 8 * rhoa01 + 4 * rhoa02;
    *rho02 = 12 * rhoa01;
    if (m_p.m_ialloy == 1) {
      *rho21 = 8. / 3. * square(rhoa21 * m_t[static_cast<std::size_t>(a)][2] -
                                rhoa22 * m_t[static_cast<std::size_t>(b)][2]);
      const double denom =
          8 * rhoa01 * square(m_t[static_cast<std::size_t>(a)][2]) +
          4 * rhoa02 * square(m_t[static_cast<std::size_t>(b)][2]);
      if (denom > 0.) {
        *rho21 = *rho21 / denom * *rho01;
      }
    } else {
      *rho21 = 8. / 3. * (rhoa21 - rhoa22) * (rhoa21 - rhoa22);
    }
    break;
  case MEAMLattice::B2:
    *rho01 = 8.0 * rhoa02;
    *rho02 = 8.0 * rhoa01;
    break;
  case MEAMLattice::CH4:
    *rho01 = 4.0 * rhoa02;
    *rho02 = rhoa01;
    shape_for(MEAMLattice::DIM, 0, 0, s);
    *rho12 = s[0] * rhoa11 * rhoa11;
    *rho22 = s[1] * rhoa21 * rhoa21;
    *rho32 = s[2] * rhoa31 * rhoa31;
    shape_for(MEAMLattice::CH4, 0, 0, s);
    *rho11 = s[0] * rhoa12 * rhoa12;
    *rho21 = s[1] * rhoa22 * rhoa22;
    *rho31 = s[2] * rhoa32 * rhoa32;
    break;
  case MEAMLattice::LIN:
    *rho01 = rhoa02 * zij;
    *rho02 = rhoa01 * zij;
    shape_for(MEAMLattice::LIN, stheta(a, b), ctheta(a, b), s);
    *rho12 = s[0] * rhoa11 * rhoa11;
    *rho22 = s[1] * rhoa21 * rhoa21;
    *rho32 = s[2] * rhoa31 * rhoa31;
    *rho11 = s[0] * rhoa12 * rhoa12;
    *rho21 = s[1] * rhoa22 * rhoa22;
    *rho31 = s[2] * rhoa32 * rhoa32;
    break;
  case MEAMLattice::ZIG:
    *rho01 = rhoa02 * zij;
    *rho02 = rhoa01 * zij;
    shape_for(MEAMLattice::ZIG, stheta(a, b), ctheta(a, b), s);
    *rho12 = s[0] * rhoa11 * rhoa11;
    *rho22 = s[1] * rhoa21 * rhoa21;
    *rho32 = s[2] * rhoa31 * rhoa31;
    *rho11 = s[0] * rhoa12 * rhoa12;
    *rho21 = s[1] * rhoa22 * rhoa22;
    *rho31 = s[2] * rhoa32 * rhoa32;
    break;
  case MEAMLattice::TRI:
    *rho01 = rhoa02;
    *rho02 = rhoa01 * zij;
    shape_for(MEAMLattice::TRI, stheta(a, b), ctheta(a, b), s);
    *rho12 = s[0] * rhoa11 * rhoa11;
    *rho22 = s[1] * rhoa21 * rhoa21;
    *rho32 = s[2] * rhoa31 * rhoa31;
    s[0] = 1.0;
    s[1] = 2.0 / 3.0;
    s[2] = 1.0 - 0.6 * s[0];
    *rho11 = s[0] * rhoa12 * rhoa12;
    *rho21 = s[1] * rhoa22 * rhoa22;
    *rho31 = s[2] * rhoa32 * rhoa32;
    break;
  }

  if (m_p.m_nn2[m_p.pair(a, b)] == 1) {
    double arat = 0.0;
    double scrn = 0.0;
    int z2nn = second_neighbours(lat, cmin(a, a, b), cmax(a, a, b), stheta(a, b),
                                 &arat, &scrn);
    const double rhoa01nn = rho0(a) * std::exp(-beta(a, 0) * (arat * r / re(a) - 1.0));
    const double rhoa02nn = rho0(b) * std::exp(-beta(b, 0) * (arat * r / re(b) - 1.0));
    if (lat == MEAMLattice::L12) {
      // The second neighbours of an L12 pair are screened by a different
      // mix of first neighbours for each of the two elements.
      const double s111 = screening(a, a, a, 1.0);
      const double s112 = screening(a, a, b, 1.0);
      const double s221 = screening(b, b, a, 1.0);
      const double big_s11 = s111 * s111 * s112 * s112;
      const double big_s22 = s221 * s221 * s221 * s221;
      *rho01 += 6.0 * big_s11 * rhoa01nn;
      *rho02 += 6.0 * big_s22 * rhoa02nn;
    } else {
      *rho01 += z2nn * scrn * rhoa01nn;
      z2nn = second_neighbours(lat, cmin(b, b, a), cmax(b, b, a), stheta(a, b),
                               &arat, &scrn);
      *rho02 += z2nn * scrn * rhoa02nn;
    }
  }
}

double MEAMModel::phi_meam_series(double scrn, int z1, int z2, int a, int b,
                                  double r, double arat) const {
  double phi_sum = 0.0;
  if (scrn > 0.0) {
    const double b2nn = -z2 * scrn / z1;
    for (int n = 1; n <= 10; ++n) {
      const double phi_val = powint(b2nn, n) * phi_meam(r * powint(arat, n), a, b);
      if (is_zero(phi_val)) {
        break;
      }
      phi_sum += phi_val;
    }
  }
  return phi_sum;
}

double MEAMModel::phi_meam(double r, int a, int b) const {
  const MEAMParameters &p = m_p;
  const std::size_t sa = static_cast<std::size_t>(a);
  const std::size_t sb = static_cast<std::size_t>(b);

  double z1 = meam_coordination(lattice(a, a));
  double z2 = meam_coordination(lattice(b, b));
  const int z12 = meam_coordination(lattice(a, b));
  const MEAMLattice lab = lattice(a, b);

  double rho01, rho11, rho21, rho31, rho02, rho12, rho22, rho32;
  reference_densities(r, a, b, &rho01, &rho11, &rho21, &rho31, &rho02, &rho12,
                      &rho22, &rho32);

  // With both densities this small the pair term is numerically meaningless.
  if (rho01 <= 1e-14 && rho02 <= 1e-14) {
    return 0.0;
  }

  double t11av, t21av, t31av, t12av, t22av, t32av;
  if (lab == MEAMLattice::C11) {
    if (p.m_ialloy == 2) {
      t11av = m_t[sa][1];
      t12av = m_t[sb][1];
      t21av = m_t[sa][2];
      t22av = m_t[sb][2];
      t31av = m_t[sa][3];
      t32av = m_t[sb][3];
    } else {
      const double scalfac = 1.0 / (rho01 + rho02);
      t11av = scalfac * (m_t[sa][1] * rho01 + m_t[sb][1] * rho02);
      t12av = t11av;
      t21av = scalfac * (m_t[sa][2] * rho01 + m_t[sb][2] * rho02);
      t22av = t21av;
      t31av = scalfac * (m_t[sa][3] * rho01 + m_t[sb][3] * rho02);
      t32av = t31av;
    }
  } else {
    reference_t(&t11av, &t21av, &t31av, &t12av, &t22av, &t32av, m_t[sa][1],
                m_t[sa][2], m_t[sa][3], m_t[sb][1], m_t[sb][2], m_t[sb][3], r, a,
                b, lab);
  }

  double rhobar1, rhobar2;
  if (lab == MEAMLattice::C11) {
    if (lattice(a, a) == MEAMLattice::DIA) {
      rhobar1 = square((z12 / 2) * (rho02 + rho01)) +
                t11av * square(rho12 - rho11) +
                t21av / 6.0 * square(rho22 + rho21) +
                121.0 / 40.0 * t31av * square(rho32 - rho31);
      rhobar1 = std::sqrt(rhobar1);
      rhobar2 = square(z12 * rho01) + 2.0 / 3.0 * t21av * square(rho21);
      rhobar2 = std::sqrt(rhobar2);
    } else {
      rhobar2 = square((z12 / 2) * (rho01 + rho02)) +
                t12av * square(rho11 - rho12) +
                t22av / 6.0 * square(rho21 + rho22) +
                121.0 / 40.0 * t32av * square(rho31 - rho32);
      rhobar2 = std::sqrt(rhobar2);
      rhobar1 = square(z12 * rho02) + 2.0 / 3.0 * t22av * square(rho22);
      rhobar1 = std::sqrt(rhobar1);
    }
  } else {
    double g1;
    double g2;
    double rho0_1 = 0.0;
    double rho0_2 = 0.0;
    double dg;
    if (p.m_mix_ref_t == 1) {
      if (p.m_elements[sa].ibar <= 0) {
        g1 = 1.0;
      } else {
        double s1[3];
        shape_for(lattice(a, a), stheta(a, a), ctheta(a, a), s1);
        const double gam1 =
            (s1[0] * t11av + s1[1] * t21av + s1[2] * t31av) / (z1 * z1);
        static_cast<void>(G(gam1, p.m_elements[sa].ibar, &g1, &dg));
      }
      if (p.m_elements[sb].ibar <= 0) {
        g2 = 1.0;
      } else {
        double s2[3];
        shape_for(lattice(b, b), stheta(b, b), ctheta(b, b), s2);
        const double gam2 =
            (s2[0] * t12av + s2[1] * t22av + s2[2] * t32av) / (z2 * z2);
        static_cast<void>(G(gam2, p.m_elements[sb].ibar, &g2, &dg));
      }
      rho0_1 = rho0(a) * z1 * g1;
      rho0_2 = rho0(b) * z2 * g2;
    }

    double gam1 = t11av * rho11 + t21av * rho21 + t31av * rho31;
    gam1 = rho01 < 1.0e-14 ? 0.0 : gam1 / (rho01 * rho01);
    double gam2 = t12av * rho12 + t22av * rho22 + t32av * rho32;
    gam2 = rho02 < 1.0e-14 ? 0.0 : gam2 / (rho02 * rho02);

    static_cast<void>(G(gam1, p.m_elements[sa].ibar, &g1, &dg));
    static_cast<void>(G(gam2, p.m_elements[sb].ibar, &g2, &dg));
    double bkgd1, bkgd2;
    if (p.m_mix_ref_t == 1) {
      bkgd1 = rho0_1;
      bkgd2 = rho0_2;
    } else if (p.m_bkgd_dyn == 1) {
      bkgd1 = rho0(a) * z1;
      bkgd2 = rho0(b) * z2;
    } else {
      bkgd1 = m_rho_ref[sa];
      bkgd2 = m_rho_ref[sb];
    }
    rhobar1 = rho01 / bkgd1 * g1;
    rhobar2 = rho02 / bkgd2 * g2;
  }

  double df = 0.0;
  const double f1 = embedding(a, rhobar1, &df);
  const double f2 = embedding(b, rhobar2, &df);
  const double eu = rose(r, a, b);

  double phi_m = 0.0;
  double arat;
  double scrn;
  double scrn2;
  if (lab == MEAMLattice::C11) {
    if (lattice(a, a) == MEAMLattice::DIA) {
      const double phiaa = phi_meam(r, a, a);
      phi_m = (3 * eu - f2 - 2 * f1 - 5 * phiaa) / z12;
    } else {
      const double phibb = phi_meam(r, b, b);
      phi_m = (3 * eu - f1 - 2 * f2 - 5 * phibb) / z12;
    }
  } else if (lab == MEAMLattice::L12) {
    double phiaa = phi_meam(r, a, a);
    const int z1nn = meam_coordination(lattice(a, a));
    const int z2nn = second_neighbours(lattice(a, a), cmin(a, a, a), cmax(a, a, a),
                                       stheta(a, b), &arat, &scrn);
    phiaa += phi_meam_series(scrn, z1nn, z2nn, a, a, r, arat);
    phi_m = eu / 3.0 - f1 / 4.0 - f2 / 12.0 - phiaa;
  } else if (lab == MEAMLattice::CH4) {
    phi_m = (5 * eu - f1 - 4 * f2) / 4;
  } else if (lab == MEAMLattice::ZIG) {
    if (a == b) {
      phi_m = (2 * eu - f1 - f2) / z12;
    } else {
      z1 = meam_coordination(lab);
      z2 = second_neighbours_b2nn(lab, cmin(a, a, b), cmax(a, a, b), &scrn);
      const double b11s = -z2 / z1 * scrn;
      z2 = second_neighbours_b2nn(lab, cmin(b, b, a), cmax(b, b, a), &scrn2);
      const double b22s = -z2 / z1 * scrn2;
      const double phiaa = phi_meam(2.0 * stheta(a, b) * r, a, a);
      const double phibb = phi_meam(2.0 * stheta(a, b) * r, b, b);
      phi_m = (2.0 * eu - f1 - f2 + phiaa * b11s + phibb * b22s) / z12;
    }
  } else if (lab == MEAMLattice::TRI) {
    if (a == b) {
      phi_m = (3.0 * eu - 2.0 * f1 - f2) / z12;
    } else {
      z1 = meam_coordination(lab);
      z2 = second_neighbours_b2nn(lab, cmin(a, a, b), cmax(a, a, b), &scrn);
      const double b11s = -z2 / z1 * scrn;
      const double phiaa = phi_meam(2.0 * stheta(a, b) * r, a, a);
      phi_m = (3.0 * eu - 2.0 * f1 - f2 + phiaa * b11s) / z12;
    }
  } else {
    phi_m = (2 * eu - f1 - f2) / z12;
  }

  if (is_zero(r)) {
    phi_m = 0.0;
  }
  return phi_m;
}

double MEAMModel::second_neighbour_series(double r, int a, int b) const {
  const MEAMLattice lab = lattice(a, b);
  double arat = 0.0;
  double scrn = 0.0;
  double scrn2 = 0.0;
  int z1 = meam_coordination(lab);
  int z2 = second_neighbours(lab, cmin(a, a, b), cmax(a, a, b), stheta(a, b),
                             &arat, &scrn);
  const bool reference_pair = lab == MEAMLattice::B1 || lab == MEAMLattice::B2 ||
                              lab == MEAMLattice::L12 || lab == MEAMLattice::DIA;
  if (!reference_pair) {
    return phi_meam_series(scrn, z1, z2, a, b, r, arat);
  }

  // The second neighbours of these pairs are like atoms, so their pair
  // term is the pair term of the element with its own second neighbours.
  const double rarat = r * arat;
  double phiaa = phi_meam(rarat, a, a);
  z1 = meam_coordination(lattice(a, a));
  z2 = second_neighbours(lattice(a, a), cmin(a, a, a), cmax(a, a, a), stheta(a, a),
                         &arat, &scrn);
  phiaa += phi_meam_series(scrn, z1, z2, a, a, rarat, arat);

  double phibb = phi_meam(rarat, b, b);
  z1 = meam_coordination(lattice(b, b));
  z2 = second_neighbours(lattice(b, b), cmin(b, b, b), cmax(b, b, b), stheta(b, b),
                         &arat, &scrn);
  phibb += phi_meam_series(scrn, z1, z2, b, b, rarat, arat);

  if (lab == MEAMLattice::L12) {
    // 1-1 second neighbours are screened by two atoms of each element and
    // 2-2 second neighbours by four atoms of the first.
    const double s111 = screening(a, a, a, 1.0);
    const double s112 = screening(a, a, b, 1.0);
    const double s221 = screening(b, b, a, 1.0);
    const double big_s11 = s111 * s111 * s112 * s112;
    const double big_s22 = s221 * s221 * s221 * s221;
    return -0.75 * big_s11 * phiaa - 0.25 * big_s22 * phibb;
  }
  z1 = meam_coordination(lab);
  z2 = second_neighbours(lab, cmin(a, a, b), cmax(a, a, b), stheta(a, b), &arat,
                         &scrn);
  double sum = -z2 * scrn / (2.0 * z1) * phiaa;
  z2 = second_neighbours(lab, cmin(b, b, a), cmax(b, b, a), stheta(a, b), &arat,
                         &scrn2);
  sum -= z2 * scrn2 / (2.0 * z1) * phibb;
  return sum;
}

void MEAMModel::build_pair_tables() {
  const MEAMParameters &p = m_p;
  const std::size_t npairs = m_n * (m_n + 1) / 2;
  m_dr = 1.1 * p.m_rc / static_cast<double>(m_nr);
  m_rdr = 1.0 / m_dr;
  m_table.assign(npairs * m_nr, Knot{});

  std::vector<double> phir(m_nr);
  std::size_t at = 0;
  for (int a = 0; a < static_cast<int>(m_n); ++a) {
    for (int b = a; b < static_cast<int>(m_n); ++b) {
      for (std::size_t j = 0; j < m_nr; ++j) {
        const double r = static_cast<double>(j) * m_dr;
        double v = phi_meam(r, a, b);

        const std::size_t ab = p.pair(a, b);
        if (p.m_nn2[ab] == 1) {
          v += second_neighbour_series(r, a, b);
        }
        if (p.m_zbl[ab] == 1) {
          const double astar = p.m_alpha[ab] * (r / p.m_re[ab] - 1.0);
          const int za = p.m_elements[static_cast<std::size_t>(a)].z;
          const int zb = p.m_elements[static_cast<std::size_t>(b)].z;
          if (astar <= -3.0) {
            v = zbl(r, za, zb);
          } else if (astar > -3.0 && astar < -1.0) {
            const double frac = fcut(1 - (astar + 1.0) / (-3.0 + 1.0));
            v = frac * v + (1 - frac) * zbl(r, za, zb);
          }
        }
        phir[j] = v;
      }

      // Cubic through the five point slope, as the LAMMPS arrays.
      Knot *knots = &m_table[at * m_nr];
      const std::size_t n = m_nr;
      std::vector<double> d1(n);
      for (std::size_t j = 0; j < n; ++j) {
        knots[j].c[0] = phir[j];
      }
      d1[0] = phir[1] - phir[0];
      d1[1] = 0.5 * (phir[2] - phir[0]);
      d1[n - 2] = 0.5 * (phir[n - 1] - phir[n - 3]);
      d1[n - 1] = 0.0;
      for (std::size_t j = 2; j + 2 < n; ++j) {
        d1[j] = ((phir[j - 2] - phir[j + 2]) + 8.0 * (phir[j + 1] - phir[j - 1])) / 12.;
      }
      for (std::size_t j = 0; j < n; ++j) {
        knots[j].c[1] = d1[j];
      }
      for (std::size_t j = 0; j + 1 < n; ++j) {
        knots[j].c[2] = 3.0 * (phir[j + 1] - phir[j]) - 2.0 * d1[j] - d1[j + 1];
        knots[j].c[3] = d1[j] + d1[j + 1] - 2.0 * (phir[j + 1] - phir[j]);
      }
      knots[n - 1].c[2] = 0.0;
      knots[n - 1].c[3] = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        knots[j].c[4] = knots[j].c[1] / m_dr;
        knots[j].c[5] = 2.0 * knots[j].c[2] / m_dr;
        knots[j].c[6] = 3.0 * knots[j].c[3] / m_dr;
      }
      ++at;
    }
  }
}

} // namespace rgpot
