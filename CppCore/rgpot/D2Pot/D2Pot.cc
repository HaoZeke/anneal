// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/D2Pot/D2Pot.hpp"

#include "rgpot/D2Pot/D2Cell.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot {

namespace {

// dftd3 V3.1, subroutine loadoldpar. R0 is the paper's Table 1 radius
// divided by 1.1 (dftd3 applies rs6 = 1.1); C6 is in J nm^6 mol^-1.
// H to Xe are Grimme 2006, Table 1; Cs to Rn are dftd3's extension.
constexpr std::array<double, 86> kR0 = {
    0.91,  0.92,                                                   // H, He
    0.75,  1.28,  1.35,  1.32,  1.27,  1.22,  1.17,  1.13,         // Li-Ne
    1.04,  1.24,  1.49,  1.56,  1.55,  1.53,  1.49,  1.45,         // Na-Ar
    1.35,  1.34,                                                   // K, Ca
    1.42,  1.42,  1.42,  1.42,  1.42,  1.42,  1.42,  1.42,  1.42,  1.42, // Sc-Zn
    1.50,  1.57,  1.60,  1.61,  1.59,  1.57,                       // Ga-Kr
    1.48,  1.46,                                                   // Rb, Sr
    1.49,  1.49,  1.49,  1.49,  1.49,  1.49,  1.49,  1.49,  1.49,  1.49, // Y-Cd
    1.52,  1.64,  1.71,  1.72,  1.72,  1.71,                       // In-Xe
    1.638, 1.602, 1.564, 1.594, 1.594, 1.594, 1.594, 1.594, 1.594, // Cs-Gd
    1.594, 1.594, 1.594, 1.594, 1.594, 1.594, 1.594, 1.594,        // Tb-Lu
    1.625, 1.611, 1.611, 1.611, 1.611, 1.611, 1.611, 1.611,        // Hf-Au
    1.598, 1.805, 1.767, 1.725, 1.823, 1.810, 1.749,               // Hg-Rn
};

constexpr std::array<double, 86> kC6 = {
    0.14,    0.08,                                                  // H, He
    1.61,    1.61,    3.13,    1.75,    1.23,    0.70,    0.75,    0.63, // Li-Ne
    5.71,    5.71,    10.79,   9.23,    7.84,    5.57,    5.07,    4.61, // Na-Ar
    10.8,    10.8,                                                  // K, Ca
    10.8,    10.8,    10.8,    10.8,    10.8,    10.8,    10.8,    10.8, 10.8, 10.8, // Sc-Zn
    16.99,   17.10,   16.37,   12.64,   12.47,   12.01,             // Ga-Kr
    24.67,   24.67,                                                 // Rb, Sr
    24.67,   24.67,   24.67,   24.67,   24.67,   24.67,   24.67,   24.67, 24.67, 24.67, // Y-Cd
    37.32,   38.71,   38.44,   31.74,   31.50,   29.99,             // In-Xe
    315.275, 226.994, 176.252,                                      // Cs, Ba, La
    140.68,  140.68,  140.68,  140.68,  140.68,  140.68,  140.68,   // Ce-Gd
    140.68,  140.68,  140.68,  140.68,  140.68,  140.68,  140.68,   // Tb-Lu
    105.112,                                                        // Hf
    81.24,   81.24,   81.24,   81.24,   81.24,   81.24,   81.24,    // Ta-Au
    57.364,  57.254,  63.162,  63.540,  55.283,  57.171,  56.64,    // Hg-Rn
};

// dftd3's own conversions: autoang and c6conv (J/mol nm^6 to Ha bohr^6).
// They differ from CODATA 2018 in the eighth digit; the tables are only
// defined together with them.
constexpr double kD3AuToAng = 0.52917726;
const double kC6Conv = 1.0e-3 / 2625.4999 / std::pow(0.052917726, 6);

// Pairs pulled from the neighbour walk per call.
constexpr std::size_t kWalkChunk = 4096;

int checked(int z) {
  if (z < 1 || z > 86) {
    throw std::invalid_argument("D2 has parameters for atomic numbers 1 to 86, got " +
                                std::to_string(z));
  }
  return z - 1;
}

std::string lower(std::string_view s) {
  std::string out(s);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

} // namespace

double d2_c6(int z) { return kC6[checked(z)]; }
double d2_r0(int z) { return kR0[checked(z)]; }

D2Config d2_config_for(std::string_view functional) {
  // dftd3 setfuncpar, version 2. dsd-blyp also changes alp to 60.
  struct Row {
    const char *name;
    double s6;
    double d;
  };
  static constexpr Row kRows[] = {
      {"b-lyp", 1.2, 20.0},   {"blyp", 1.2, 20.0},     {"b-p", 1.05, 20.0},
      {"bp86", 1.05, 20.0},   {"b97-d", 1.25, 20.0},   {"revpbe", 1.25, 20.0},
      {"pbe", 0.75, 20.0},    {"tpss", 1.0, 20.0},     {"b3-lyp", 1.05, 20.0},
      {"b3lyp", 1.05, 20.0},  {"pbe0", 0.6, 20.0},     {"pw6b95", 0.5, 20.0},
      {"tpss0", 0.85, 20.0},  {"b2-plyp", 0.55, 20.0}, {"b2gp-plyp", 0.4, 20.0},
      {"dsd-blyp", 0.41, 60.0},
  };
  const std::string key = lower(functional);
  for (const Row &r : kRows) {
    if (key == r.name) {
      D2Config c;
      c.s6 = r.s6;
      c.d = r.d;
      return c;
    }
  }
  throw std::invalid_argument("D2 has no parameters for functional '" +
                              std::string(functional) + "'");
}

D2Pot::D2Pot(D2Config cfg) : Potential(PotType::D2), cfg_(cfg) {
  if (!(cfg_.cutoff > 0.0) || !std::isfinite(cfg_.cutoff)) {
    throw std::invalid_argument("D2 cutoff must be positive and finite");
  }
  if (!(cfg_.sr > 0.0) || !std::isfinite(cfg_.sr) || !std::isfinite(cfg_.s6) ||
      !std::isfinite(cfg_.d)) {
    throw std::invalid_argument("D2 needs positive finite sr and finite s6 and d");
  }
}

void D2Pot::forceImpl(const ForceInput &in, ForceOut *out) const {
  using units::BOHR_TO_ANGSTROM;
  using units::HARTREE_TO_EV;
  const std::size_t n = in.nAtoms;
  if (n > 0 && in.atmnrs == nullptr) {
    throw std::invalid_argument("D2 needs atomic numbers");
  }
  std::vector<int> zi(n);
  for (std::size_t i = 0; i < n; ++i) {
    zi[i] = checked(in.atmnrs[i]);
  }
  const d2::detail::Cell cell = d2::detail::Cell::from_box(in.box, BOHR_TO_ANGSTROM);
  if (cell.periodic) {
    cell.require_bounded_images(cfg_.cutoff / BOHR_TO_ANGSTROM);
  }

  std::vector<double> g(3 * n, 0.0);
  // The sum has millions of terms of one sign, so a plain running sum
  // would depend on the order the neighbour walk yields them.
  double disp = 0.0;
  double disp_lost = 0.0;
  std::array<double, 9> w{};

  auto pair = [&](int za, int zb, const double *dv, double weight, double *gi, double *gj) {
    const double r2 = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
    if (r2 == 0.0) {
      return;
    }
    const double r0 = (kR0[za] + kR0[zb]) / kD3AuToAng * cfg_.sr;
    const double c6 = std::sqrt(kC6[za] * kC6[zb]) * kC6Conv * cfg_.s6;
    const double r = std::sqrt(r2);
    const double r7 = r2 * r2 * r2 * r;
    const double damp6 = std::exp(-cfg_.d * (r / r0 - 1.0));
    const double damp1 = 1.0 + damp6;
    const double tmp1 = damp6 / (damp1 * damp1 * r7 * r0);
    const double tmp2 = 6.0 / (damp1 * r * r7);
    const double term = (cfg_.d * tmp1 - tmp2) * c6;
    const double term_e = weight * c6 / (damp1 * r2 * r2 * r2);
    const double total = disp + term_e;
    disp_lost += std::abs(disp) >= std::abs(term_e) ? (disp - total) + term_e
                                                    : (term_e - total) + disp;
    disp = total;
    if (gi != nullptr) {
      for (int a = 0; a < 3; ++a) {
        gi[a] -= term * dv[a];
        gj[a] += term * dv[a];
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        w[3 * a + b] -= weight * term * dv[a] * dv[b];
      }
    }
  };

  // Every pair, and every image of it within the cutoff, comes from the
  // neighbour layer. The half shell keeps each pair once and one of each
  // pair of opposite images of an atom with itself, so the self term
  // enters at weight one where the full set would give one half.
  nlist::Query query;
  query.cutoff = cfg_.cutoff;
  query.shape = nlist::Pairs{nlist::Shell::Half};
  query.cut = nlist::Cut::Strict;
  query.periodic = {cell.periodic, cell.periodic, cell.periodic};
  query.fields = {.shifts = false, .vectors = true, .distances = false};
  static constexpr std::array<double, 9> kOpenBox{};
  const double *box = cell.periodic ? in.box : kOpenBox.data();

  if (n > 0) {
    std::vector<std::int32_t> first(kWalkChunk);
    std::vector<std::int32_t> second(kWalkChunk);
    std::vector<double> vec(3 * kWalkChunk);
    nlist::Walk walk(in.pos, n, box, query);
    bool done = false;
    while (!done) {
      const std::size_t got = walk.pull({.first = first, .second = second, .vector = vec}, done);
      for (std::size_t p = 0; p < got; ++p) {
        const auto i = static_cast<std::size_t>(first[p]);
        const auto j = static_cast<std::size_t>(second[p]);
        // The vector runs from i to the image of j; the kernel takes i less j.
        const double dv[3] = {-vec[3 * p] / BOHR_TO_ANGSTROM, -vec[3 * p + 1] / BOHR_TO_ANGSTROM,
                              -vec[3 * p + 2] / BOHR_TO_ANGSTROM};
        if (i == j) {
          pair(zi[i], zi[i], dv, 1.0, nullptr, nullptr);
        } else {
          pair(zi[i], zi[j], dv, 1.0, &g[3 * i], &g[3 * j]);
        }
      }
    }
  }

  out->energy = -(disp + disp_lost) * HARTREE_TO_EV;
  const double fconv = HARTREE_TO_EV / BOHR_TO_ANGSTROM;
  for (std::size_t k = 0; k < 3 * n; ++k) {
    out->F[k] = -g[k] * fconv;
  }
  if (cell.periodic && cell.volume > 0.0) {
    for (int k = 0; k < 9; ++k) {
      out->stress[k] = w[k] / cell.volume * units::HARTREE_PER_BOHR3_TO_EV_PER_ANGSTROM3;
    }
    out->has_stress = 1;
  }
}

uint64_t D2Pot::paramsKey() const noexcept {
  Fnv1a h;
  h.str("rgpot::D2Pot");
  h.f64(cfg_.s6);
  h.f64(cfg_.d);
  h.f64(cfg_.sr);
  h.f64(cfg_.cutoff);
  return h.h;
}

} // namespace rgpot
