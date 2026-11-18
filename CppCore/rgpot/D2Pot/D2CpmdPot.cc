// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/D2Pot/D2CpmdPot.hpp"

#include "rgpot/D2Pot/D2Cell.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rgpot {

namespace {

// OpenCPMD 062582b, src/vdwin_utils.mod.F90 (empvdwin): vdwc6 in
// J nm^6 mol^-1 and vdwr0 in Angstrom, H to Rn. These are Grimme 2006,
// Table 1, as CPMD types them, B's C6 of 3.31 included; elements 87 to 99
// have no parameters and give nothing.
constexpr std::array<double, 86> kCpmdC6 = {
    0.14, 0.08, 1.61, 1.61, 3.31, 1.75, 1.23, 0.70,
    0.75, 0.63, 5.71, 5.71, 10.79, 9.23, 7.84, 5.57,
    5.07, 4.61, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80,
    10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 16.99, 17.10,
    16.37, 12.64, 12.47, 12.01, 24.67, 24.67, 24.67, 24.67,
    24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67,
    37.32, 38.71, 38.44, 31.74, 31.50, 29.99, 315.275, 226.994,
    176.252, 140.68, 140.68, 140.68, 140.68, 140.68, 140.68, 140.68,
    140.68, 140.68, 140.68, 140.68, 140.68, 140.68, 140.68, 105.112,
    81.24, 81.24, 81.24, 81.24, 81.24, 81.24, 81.24, 57.364,
    57.254, 63.162, 63.540, 55.283, 57.171, 56.64,
};
constexpr std::array<double, 86> kCpmdR0 = {
    1.001, 1.012, 0.825, 1.408, 1.485, 1.452, 1.397, 1.342,
    1.287, 1.243, 1.144, 1.364, 1.639, 1.716, 1.705, 1.683,
    1.639, 1.595, 1.485, 1.474, 1.562, 1.562, 1.562, 1.562,
    1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.650, 1.727,
    1.760, 1.771, 1.749, 1.727, 1.628, 1.606, 1.639, 1.639,
    1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639,
    1.672, 1.804, 1.881, 1.892, 1.892, 1.881, 1.802, 1.762,
    1.720, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753,
    1.753, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753, 1.788,
    1.772, 1.772, 1.772, 1.772, 1.772, 1.772, 1.772, 1.758,
    1.986, 1.944, 1.898, 2.005, 1.991, 1.924,
};

// convfac in empvdwin. kCpmdBohr is fbohr's reciprocal, defined once in
// the header so the input and the derivatives share it.
constexpr double kCpmdC6Conv = 1.733979e1;
constexpr double kCpmdFbohr = 1.0 / kCpmdBohr;
// reps in vdw: a pair closer than this many bohr contributes nothing.
constexpr double kCpmdMinPair = 1.0e-2;
constexpr int kCpmdMaxZ = 99;
constexpr double kCellTol = 1.0e-12;

std::string upper(std::string_view s) {
  std::string out(s);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return out;
}

/// CPMD's orthorhombic fold (pbc3 labels 5 and 15): dmod into one period,
/// then subtract the period times dint of that value over half a period.
double wrap_axis(double x, double length) {
  const double y = std::fmod(x, length);
  return y - length * std::trunc(y / (0.5 * length));
}

bool nearly(double a, double b) { return std::abs(a - b) <= kCellTol; }

// OpenCPMD 062582b, src/pbc_utils.mod.F90, subroutine pbc3, called from
// subroutine vdw in src/vdw_utils.mod.F90 with m = 1. The vectors are
// latgen's (src/latgen_utils.mod.F90, subroutine latgen) and apbc is the
// one setsc stores (src/setsc_utils.mod.F90, subroutine setsc). Symmetry
// 2 is face centred cubic in latgen's orientation. Symmetry 4 and
// symmetry 14 with alpha and beta at 90 degrees share pbc3's shear
// branch (label 25). A box that matches neither is the general branch,
// the nearest integer of each fractional coordinate (label 99), which is
// also what CELL VECTORS selects after setsc turns ibrav -2 into 14.

bool matches_fcc(const std::array<double, 9> &h, double *ap) {
  const double t = h[2];
  if (!(t > 0.0)) {
    return false;
  }
  const double expect[9] = {-t, 0.0, t, 0.0, t, t, -t, t, 0.0};
  for (int k = 0; k < 9; ++k) {
    if (!nearly(h[k], expect[k])) {
      return false;
    }
  }
  *ap = t * std::sqrt(2.0) * 0.5;
  return true;
}

bool matches_label25(const std::array<double, 9> &h, double ap[4]) {
  const double a = h[0];
  const double c = h[8];
  const double bx = h[3];
  const double by = h[4];
  if (!(a > 0.0 && c > 0.0 && by > 0.0)) {
    return false;
  }
  if (!(nearly(h[1], 0.0) && nearly(h[2], 0.0) && nearly(h[5], 0.0) &&
        nearly(h[6], 0.0) && nearly(h[7], 0.0))) {
    return false;
  }
  if (nearly(bx, 0.0)) {
    return false;
  }
  ap[0] = 0.5 * a;
  ap[1] = 0.5 * by;
  ap[2] = 0.5 * c;
  ap[3] = -bx / by;
  return true;
}

void pbc_fcc(double *r, double ap) {
  const double s2 = std::sqrt(2.0);
  const double ta = 2.0 * ap;
  double x2 = (r[0] + r[1]) / s2;
  double y2 = (-r[0] + r[1]) / s2;
  x2 = wrap_axis(x2, ta);
  y2 = wrap_axis(y2, ta);
  double z2 = wrap_axis(r[2], ta * s2);
  if (!(std::abs(x2) + std::abs(y2) + s2 * std::abs(z2) < ta)) {
    x2 -= std::copysign(ap, x2);
    y2 -= std::copysign(ap, y2);
    z2 -= std::copysign(s2 * ap, z2);
  }
  const double xt = (x2 - y2) / s2;
  r[0] = xt;
  r[1] = (x2 + y2) / s2;
  r[2] = z2;
}

void pbc_label25(double *r, const double ap[4]) {
  const double y0 = r[1];
  const double y2 = wrap_axis(y0, 2.0 * ap[1]);
  const double z2 = wrap_axis(r[2], 2.0 * ap[2]);
  const double dx = (y2 - y0) * ap[3];
  r[0] = wrap_axis(r[0] - dx, 2.0 * ap[0]);
  r[1] = y2;
  r[2] = z2;
}

void pbc_general(const d2::detail::Cell &cell, double *r) {
  double s[3];
  cell.fractional(r, s);
  for (double &v : s) {
    v -= std::round(v);
  }
  cell.cartesian(s, r);
}

// pbc3 reaches its monoclinic branch (label 20) through
//   go to (5,10,99,25,99,99,99,15,99,99,99,20,99,25) ibrav
// whose twelfth entry is 20, so only ibrav 12 takes it. The comment above
// the label calls that branch trigonal, but latgen builds ibrav 12 as the
// monoclinic cell (a1 along x, a2 in the xy plane at the angle with cosine
// celldm(4), a3 along z) and cry, called by setsc, is written for 12 alone.
// ibrav 4 is the hexagonal cell, a1 = (a, 0, 0) and a2 = (-a/2, a sqrt(3)/2,
// 0), and shares the shear branch (label 25) with ibrav 14, whose gamma
// cell has the same shape. A box shows the shape and not the code, so ibrav
// 12 is chosen only by D2CpmdConfig::symmetry.
//
// The fold takes the xy projection onto three lattice directions in turn. In
// setsc the four images a1 - k a2 (k = 0 to 3) are sorted by length with
// sort2, an insertion sort for n = 4 that keeps equal lengths in order. The
// shortest gives apbc(1) and v1crys, the next apbc(4) and v3crys, and a2
// gives apbc(2) and v2crys. Each v is the direction as (cos, sin), and apbc(3)
// is half of the c axis.
struct Monoclinic12 {
  std::array<double, 4> ap{};
  std::array<std::array<double, 2>, 3> dir{};
};

bool setup_monoclinic12(const std::array<double, 9> &h, Monoclinic12 *out) {
  if (!(h[0] > 0.0 && h[4] > 0.0 && h[8] > 0.0)) {
    return false;
  }
  if (!(nearly(h[1], 0.0) && nearly(h[2], 0.0) && nearly(h[5], 0.0) && nearly(h[6], 0.0) &&
        nearly(h[7], 0.0))) {
    return false;
  }
  const double a = h[0];
  const double bbbb = h[3];
  const double cccc = h[4];
  std::array<std::pair<double, int>, 4> images{};
  for (int k = 0; k < 4; ++k) {
    const double dx = a - k * bbbb;
    const double dy = k * cccc;
    images[static_cast<std::size_t>(k)] = {dx * dx + dy * dy, k};
  }
  std::stable_sort(images.begin(), images.end(),
                   [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
  auto direction = [](double x, double y) {
    const double aux = std::sqrt(x * x + y * y);
    return std::array<double, 2>{x / aux, y / aux};
  };
  out->ap = {0.5 * std::sqrt(images[0].first), 0.5 * std::sqrt(bbbb * bbbb + cccc * cccc),
             0.5 * h[8], 0.5 * std::sqrt(images[1].first)};
  const int k1 = images[0].second;
  const int k3 = images[1].second;
  out->dir[0] = direction(a - k1 * bbbb, -k1 * cccc);
  out->dir[1] = direction(bbbb, cccc);
  out->dir[2] = direction(a - k3 * bbbb, -k3 * cccc);
  return true;
}

// subroutine rot of src/pbc_utils.mod.F90, with isign = +1 for into the
// direction's frame and -1 for back out of it.
void rotate(const std::array<double, 2> &dir, double sign, double x1, double y1, double *x0,
            double *y0) {
  *x0 = x1 * dir[0] + y1 * dir[1] * sign;
  *y0 = -x1 * dir[1] * sign + y1 * dir[0];
}

void pbc_monoclinic12(double *r, const Monoclinic12 &g) {
  double x0 = 0.0;
  double y0 = 0.0;
  double x3 = 0.0;
  double y3 = 0.0;
  const double z2 = wrap_axis(r[2], 2.0 * g.ap[2]);
  rotate(g.dir[0], 1.0, r[0], r[1], &x0, &y0);
  x0 = wrap_axis(x0, 2.0 * g.ap[0]);
  rotate(g.dir[0], -1.0, x0, y0, &x3, &y3);
  rotate(g.dir[1], 1.0, x3, y3, &x0, &y0);
  x0 = wrap_axis(x0, 2.0 * g.ap[1]);
  rotate(g.dir[1], -1.0, x0, y0, &x3, &y3);
  rotate(g.dir[2], 1.0, x3, y3, &x0, &y0);
  x0 = wrap_axis(x0, 2.0 * g.ap[3]);
  rotate(g.dir[2], -1.0, x0, y0, &x3, &y3);
  rotate(g.dir[0], 1.0, x3, y3, &x0, &y0);
  x0 = wrap_axis(x0, 2.0 * g.ap[0]);
  rotate(g.dir[0], -1.0, x0, y0, &r[0], &r[1]);
  r[2] = z2;
}

enum class FoldKind { Open, Orthorhombic, FaceCentred, Shear, Monoclinic12, Nearest };

struct Fold {
  FoldKind kind = FoldKind::Open;
  double fcc_ap = 0.0;
  double shear_ap[4] = {};
  Monoclinic12 monoclinic;
};

bool is_orthorhombic(const std::array<double, 9> &h) {
  return h[1] == 0.0 && h[2] == 0.0 && h[3] == 0.0 && h[5] == 0.0 && h[6] == 0.0 && h[7] == 0.0;
}

[[noreturn]] void wrong_box(int symmetry, const char *shape) {
  throw std::invalid_argument("CPMD's D2 with symmetry " + std::to_string(symmetry) +
                              " needs a box with " + shape);
}

// The branch of pbc3 for the code, following the go to above. A code with a
// shape (1, 2, 4, 8, 12) needs a box of that shape: latgen cannot produce
// another one. 14 takes the shear branch for a gamma cell, as setsc does when
// alpha and beta are 90 degrees, and the nearest integer otherwise; the
// codes that go to label 99 (3, 5, 6, 7) take the nearest integer for any
// box. The default infers the branch from the box.
Fold select_fold(const d2::detail::Cell &cell, int symmetry) {
  Fold fold;
  if (!cell.periodic) {
    return fold;
  }
  const auto &h = cell.h;
  const bool orthorhombic = is_orthorhombic(h);
  const bool cubic = orthorhombic && nearly(h[0], h[4]) && nearly(h[4], h[8]);
  auto nearest = [&] { fold.kind = FoldKind::Nearest; };
  switch (symmetry) {
  case kCpmdInferSymmetry:
    if (orthorhombic) {
      fold.kind = FoldKind::Orthorhombic;
    } else if (matches_fcc(h, &fold.fcc_ap)) {
      fold.kind = FoldKind::FaceCentred;
    } else if (matches_label25(h, fold.shear_ap)) {
      fold.kind = FoldKind::Shear;
    } else {
      nearest();
    }
    break;
  case 1:
    if (!cubic) {
      wrong_box(symmetry, "equal orthogonal vectors");
    }
    fold.kind = FoldKind::Orthorhombic;
    break;
  case 2:
    if (!matches_fcc(h, &fold.fcc_ap)) {
      wrong_box(symmetry, "latgen's face centred cubic vectors");
    }
    fold.kind = FoldKind::FaceCentred;
    break;
  case 4:
    if (!(matches_label25(h, fold.shear_ap) && nearly(h[3], -0.5 * h[0]) &&
          nearly(h[4], 0.5 * std::sqrt(3.0) * h[0]))) {
      wrong_box(symmetry, "a1 = (a, 0, 0), a2 = (-a/2, a sqrt(3)/2, 0) and a3 along z");
    }
    fold.kind = FoldKind::Shear;
    break;
  case 8:
    if (!orthorhombic) {
      wrong_box(symmetry, "orthogonal vectors");
    }
    fold.kind = FoldKind::Orthorhombic;
    break;
  case 12:
    if (!setup_monoclinic12(h, &fold.monoclinic)) {
      wrong_box(symmetry, "a1 along x, a2 in the xy plane and a3 along z");
    }
    fold.kind = FoldKind::Monoclinic12;
    break;
  case 14:
    if (orthorhombic) {
      fold.kind = FoldKind::Orthorhombic;
    } else if (matches_label25(h, fold.shear_ap)) {
      fold.kind = FoldKind::Shear;
    } else {
      nearest();
    }
    break;
  default:
    nearest();
    break;
  }
  return fold;
}

void apply_fold(const Fold &fold, const d2::detail::Cell &cell, double *r) {
  switch (fold.kind) {
  case FoldKind::Open:
    break;
  case FoldKind::Orthorhombic:
    for (int a = 0; a < 3; ++a) {
      r[a] = wrap_axis(r[a], cell.h[4 * a]);
    }
    break;
  case FoldKind::FaceCentred:
    pbc_fcc(r, fold.fcc_ap);
    break;
  case FoldKind::Shear:
    pbc_label25(r, fold.shear_ap);
    break;
  case FoldKind::Monoclinic12:
    pbc_monoclinic12(r, fold.monoclinic);
    break;
  case FoldKind::Nearest:
    pbc_general(cell, r);
    break;
  }
}

bool symmetry_known(int symmetry) {
  switch (symmetry) {
  case kCpmdInferSymmetry:
  case 1:
  case 2:
  case 3:
  case 4:
  case 5:
  case 6:
  case 7:
  case 8:
  case 12:
  case 14:
    return true;
  default:
    return false;
  }
}

} // namespace

D2CpmdConfig d2_cpmd_config_for(std::string_view functional) {
  // set_s6 in src/vdwin_utils.mod.F90.
  struct Row {
    const char *name;
    double s6;
  };
  static constexpr Row kRows[] = {
      {"BLYP", 1.20}, {"B3LYP", 1.05}, {"BP86", 1.05}, {"TPSS", 1.00},
      {"PBE", 0.75},  {"REVPBE", 1.25}, {"PBE0", 0.60},
  };
  const std::string key = upper(functional);
  for (const Row &r : kRows) {
    if (key == r.name) {
      D2CpmdConfig c;
      c.s6 = r.s6;
      return c;
    }
  }
  throw std::invalid_argument("CPMD's D2 has no s6 for functional '" + std::string(functional) +
                              "'");
}

D2CpmdPot::D2CpmdPot(D2CpmdConfig cfg) : Potential(PotType::D2Cpmd), cfg_(cfg) {
  if (!std::isfinite(cfg_.s6) || !std::isfinite(cfg_.alpha) || !(cfg_.vdweps > 0.0) ||
      !std::isfinite(cfg_.vdweps)) {
    throw std::invalid_argument("CPMD's D2 needs finite s6 and alpha and a positive vdweps");
  }
  if (!symmetry_known(cfg_.symmetry)) {
    throw std::invalid_argument(
        "CPMD's D2 symmetry must be -1 or a latgen code (1 to 8, 12, 14), got " +
        std::to_string(cfg_.symmetry));
  }
}

void D2CpmdPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  using units::HARTREE_TO_EV;
  const double force_scale = HARTREE_TO_EV / kCpmdBohr;
  const double stress_scale =
      HARTREE_TO_EV / (kCpmdBohr * kCpmdBohr * kCpmdBohr);
  const std::size_t n = in.nAtoms;
  if (n > 0 && in.atmnrs == nullptr) {
    throw std::invalid_argument("CPMD's D2 needs atomic numbers");
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (in.atmnrs[i] < 1 || in.atmnrs[i] > kCpmdMaxZ) {
      throw std::invalid_argument("CPMD's D2 takes atomic numbers 1 to 99, got " +
                                  std::to_string(in.atmnrs[i]));
    }
  }
  std::vector<double> x(3 * n);
  for (std::size_t k = 0; k < 3 * n; ++k) {
    x[k] = in.pos[k] * kCpmdFbohr;
  }
  const d2::detail::Cell cell = d2::detail::Cell::from_box(in.box, kCpmdBohr);
  const Fold fold = select_fold(cell, cfg_.symmetry);
  const double rpow_max = 1.0 / cfg_.vdweps;

  std::vector<double> f(3 * n, 0.0);
  double evdw = 0.0;
  std::array<double, 9> w{};
  for (std::size_t l = 0; l < n; ++l) {
    for (std::size_t m = 0; m < l; ++m) {
      const int zl = in.atmnrs[l];
      const int zm = in.atmnrs[m];
      constexpr int kTabulated = static_cast<int>(kCpmdC6.size());
      if (zl > kTabulated || zm > kTabulated) {
        continue;
      }
      double rlp[3] = {x[3 * l] - x[3 * m], x[3 * l + 1] - x[3 * m + 1],
                       x[3 * l + 2] - x[3 * m + 2]};
      apply_fold(fold, cell, rlp);
      const double rlm2 = rlp[0] * rlp[0] + rlp[1] * rlp[1] + rlp[2] * rlp[2];
      const double rlm1 = std::sqrt(rlm2);
      if (rlm1 < kCpmdMinPair) {
        continue;
      }
      const double vdwst = std::sqrt(kCpmdC6[zl - 1] * kCpmdC6[zm - 1]) * kCpmdC6Conv;
      const double vdwrm = (kCpmdR0[zl - 1] + kCpmdR0[zm - 1]) * kCpmdFbohr;
      const double rlm6 = rlm2 * rlm2 * rlm2;
      const double rpow = cfg_.alpha * (rlm1 / vdwrm - 1.0);
      if (!(std::abs(rpow) < rpow_max)) {
        continue;
      }
      const double rexp = 1.0 + std::exp(-rpow);
      const double rphi = -cfg_.s6 * vdwst / (rlm6 * rexp);
      const double ffac = cfg_.s6 * vdwst / (rlm6 * rexp * rlm1) *
                          (6.0 / rlm1 - cfg_.alpha * (rexp - 1.0) / (vdwrm * rexp));
      evdw += rphi;
      for (int a = 0; a < 3; ++a) {
        f[3 * l + a] -= ffac * rlp[a];
        f[3 * m + a] += ffac * rlp[a];
        for (int b = 0; b < 3; ++b) {
          w[3 * a + b] += ffac * rlp[a] * rlp[b];
        }
      }
    }
  }

  out->energy = evdw * HARTREE_TO_EV;
  for (std::size_t k = 0; k < 3 * n; ++k) {
    out->F[k] = f[k] * force_scale;
  }
  if (cell.periodic) {
    for (int k = 0; k < 9; ++k) {
      out->stress[k] = w[k] / cell.volume * stress_scale;
    }
    out->has_stress = 1;
  }
}

uint64_t D2CpmdPot::paramsKey() const noexcept {
  Fnv1a h;
  h.str("rgpot::D2CpmdPot");
  h.f64(cfg_.s6);
  h.f64(cfg_.alpha);
  h.f64(cfg_.vdweps);
  h.i64(cfg_.symmetry);
  return h.h;
}

} // namespace rgpot
