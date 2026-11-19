// MIT License
// Copyright 2023--present rgpot developers
//
// Multi-image EAM checks against an independent supercell reference.

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAMTable/EAMTablePot.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/nlist/MultiImage.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::EAMTableFormat;

namespace {

const std::string kData = "CppCore/tests/data/";

rgpot::EAMTablePot cu_u3() {
  return rgpot::EAMTablePot({.path = kData + "eam_table/Cu_u3.eam",
                             .format = EAMTableFormat::Funcfl,
                             .periodic = true,
                             .stress = true});
}

std::string pair_table() {
  // phi(r) = (5 - r)^2 on knots one angstrom apart, stored as r phi.
  // Embedding and density are zero, so the energy is only the pair sum.
  std::ostringstream s;
  s << "pair\npair\npair\n1 Cu\n5 1 6 1 5\n29 63.546 1 fcc\n";
  for (int m = 0; m < 5; ++m) {
    s << "0 ";
  }
  s << '\n';
  for (int m = 0; m < 6; ++m) {
    s << "0 ";
  }
  s << '\n';
  for (int m = 0; m < 6; ++m) {
    const double r = static_cast<double>(m);
    s << (r * (5.0 - r) * (5.0 - r)) << ' ';
  }
  s << '\n';
  return s.str();
}

rgpot::EAMTablePot pair_only() {
  std::istringstream in(pair_table());
  return rgpot::EAMTablePot(rgpot::read_eam_table(in, EAMTableFormat::Setfl),
                            true, true);
}

struct Eval {
  double energy = 0.0;
  std::array<double, 9> stress{};
  int has_stress = 0;
  std::vector<double> forces;
  std::vector<double> eatom;
  std::vector<double> vatom;
};

Eval eval_full(const rgpot::EAMTablePot &pot, const std::vector<double> &pos,
               const std::vector<int> &z, const double *box) {
  Eval e;
  e.forces.assign(pos.size(), 0.0);
  e.eatom.assign(z.size(), 0.0);
  e.vatom.assign(6 * z.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = e.forces.data();
  pot.forcePerAtom(rgpot::ForceInput{z.size(), pos.data(), z.data(), box}, &out,
                   rgpot::PerAtomOut{e.eatom.data(), e.vatom.data()});
  e.energy = out.energy;
  e.has_stress = out.has_stress;
  std::copy(out.stress, out.stress + 9, e.stress.begin());
  return e;
}

double face_spacing(const double *box, int axis) {
  const double *u = box + 3 * ((axis + 1) % 3);
  const double *v = box + 3 * ((axis + 2) % 3);
  const double cx = u[1] * v[2] - u[2] * v[1];
  const double cy = u[2] * v[0] - u[0] * v[2];
  const double cz = u[0] * v[1] - u[1] * v[0];
  const double area = std::sqrt(cx * cx + cy * cy + cz * cz);
  const double vol = std::abs(box[0] * (box[4] * box[8] - box[5] * box[7]) -
                              box[1] * (box[3] * box[8] - box[5] * box[6]) +
                              box[2] * (box[3] * box[7] - box[4] * box[6]));
  REQUIRE(area > 0.0);
  REQUIRE(vol > 0.0);
  return vol / area;
}

std::array<int, 3> replication(const double *box, double cutoff) {
  std::array<int, 3> rep{1, 1, 1};
  for (int axis = 0; axis < 3; ++axis) {
    const double spacing = face_spacing(box, axis);
    while (!(2.0 * cutoff < rep[static_cast<std::size_t>(axis)] * spacing)) {
      ++rep[static_cast<std::size_t>(axis)];
    }
  }
  return rep;
}

struct Replicated {
  std::array<double, 9> box{};
  std::vector<double> pos;
  std::vector<int> z;
  std::size_t nrep = 1;
};

Replicated replicate(const std::vector<double> &pos, const std::vector<int> &z,
                     const double *box, const std::array<int, 3> &rep) {
  Replicated out;
  out.box = {rep[0] * box[0], rep[0] * box[1], rep[0] * box[2],
             rep[1] * box[3], rep[1] * box[4], rep[1] * box[5],
             rep[2] * box[6], rep[2] * box[7], rep[2] * box[8]};
  const std::size_t n = z.size();
  out.nrep = static_cast<std::size_t>(rep[0] * rep[1] * rep[2]);
  out.pos.reserve(3 * n * out.nrep);
  out.z.reserve(n * out.nrep);
  for (int h = 0; h < rep[0]; ++h) {
    for (int k = 0; k < rep[1]; ++k) {
      for (int l = 0; l < rep[2]; ++l) {
        const double sx = h * box[0] + k * box[3] + l * box[6];
        const double sy = h * box[1] + k * box[4] + l * box[7];
        const double sz = h * box[2] + k * box[5] + l * box[8];
        for (std::size_t i = 0; i < n; ++i) {
          out.pos.push_back(pos[3 * i] + sx);
          out.pos.push_back(pos[3 * i + 1] + sy);
          out.pos.push_back(pos[3 * i + 2] + sz);
          out.z.push_back(z[i]);
        }
      }
    }
  }
  return out;
}

Eval supercell_reference(const rgpot::EAMTablePot &pot,
                         const std::vector<double> &pos,
                         const std::vector<int> &z, const double *box,
                         const std::array<int, 3> &rep);

Eval supercell_reference(const rgpot::EAMTablePot &pot,
                         const std::vector<double> &pos,
                         const std::vector<int> &z, const double *box) {
  return supercell_reference(pot, pos, z, box,
                             replication(box, pot.locality().cutoff));
}

Eval supercell_reference(const rgpot::EAMTablePot &pot,
                         const std::vector<double> &pos,
                         const std::vector<int> &z, const double *box,
                         const std::array<int, 3> &rep) {
  const Replicated sc = replicate(pos, z, box, rep);
  const Eval big = eval_full(pot, sc.pos, sc.z, sc.box.data());
  const std::size_t n = z.size();
  const double inv = 1.0 / static_cast<double>(sc.nrep);
  Eval ref;
  ref.energy = big.energy * inv;
  ref.stress = big.stress;
  ref.has_stress = big.has_stress;
  ref.forces.assign(3 * n, 0.0);
  ref.eatom.assign(n, 0.0);
  ref.vatom.assign(6 * n, 0.0);
  for (std::size_t t = 0; t < sc.nrep; ++t) {
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t src = t * n + i;
      ref.eatom[i] += big.eatom[src] * inv;
      for (int c = 0; c < 3; ++c) {
        ref.forces[3 * i + static_cast<std::size_t>(c)] +=
            big.forces[3 * src + static_cast<std::size_t>(c)] * inv;
      }
      for (int v = 0; v < 6; ++v) {
        ref.vatom[6 * i + static_cast<std::size_t>(v)] +=
            big.vatom[6 * src + static_cast<std::size_t>(v)] * inv;
      }
    }
  }
  return ref;
}

std::array<double, 6> total_virial(const Eval &e) {
  std::array<double, 6> v{};
  for (std::size_t i = 0; i < e.vatom.size() / 6; ++i) {
    for (int k = 0; k < 6; ++k) {
      v[static_cast<std::size_t>(k)] +=
          e.vatom[6 * i + static_cast<std::size_t>(k)];
    }
  }
  return v;
}

void check(const rgpot::EAMTablePot &pot, const std::vector<double> &pos,
           const std::vector<int> &z, const double *box, double etol,
           double ftol, double stol, double atom_tol) {
  const Eval got = eval_full(pot, pos, z, box);
  const Eval want = supercell_reference(pot, pos, z, box);
  REQUIRE(got.has_stress == 1);
  REQUIRE(want.has_stress == 1);
  REQUIRE_THAT(got.energy, WithinAbs(want.energy, etol));
  for (std::size_t c = 0; c < got.forces.size(); ++c) {
    REQUIRE_THAT(got.forces[c], WithinAbs(want.forces[c], ftol));
  }
  for (int c = 0; c < 9; ++c) {
    REQUIRE_THAT(got.stress[static_cast<std::size_t>(c)],
                 WithinAbs(want.stress[static_cast<std::size_t>(c)], stol));
  }
  for (std::size_t i = 0; i < got.eatom.size(); ++i) {
    REQUIRE_THAT(got.eatom[i], WithinAbs(want.eatom[i], atom_tol));
  }
  const std::array<double, 6> gv = total_virial(got);
  const std::array<double, 6> wv = total_virial(want);
  for (int c = 0; c < 6; ++c) {
    REQUIRE_THAT(gv[static_cast<std::size_t>(c)],
                 WithinAbs(wv[static_cast<std::size_t>(c)], atom_tol));
  }
}

} // namespace

TEST_CASE("EAM multi-image matches an independent supercell reference",
          "[EAMTable][nlist]") {
  SECTION("pair-only one-atom cubic 4A cell") {
    const rgpot::EAMTablePot pot = pair_only();
    const std::vector<double> pos{0.4, 0.5, 0.6};
    const std::vector<int> z{29};
    const double box[9] = {4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
    const Eval got = eval_full(pot, pos, z, box);
    const Eval want = supercell_reference(pot, pos, z, box);
    REQUIRE_THAT(want.energy, WithinAbs(3.0, 1e-12));
    REQUIRE_THAT(got.energy, WithinAbs(3.0, 1e-12));
    const std::array<double, 6> got_v = total_virial(got);
    REQUIRE_THAT(got_v[0], WithinAbs(7.0, 1e-12));
    REQUIRE_THAT(got.stress[0], WithinAbs(-7.0 / 64.0, 1e-12));
    check(pot, pos, z, box, 1e-12, 1e-10, 1e-12, 1e-12);
  }

  SECTION("pair-only one-atom triclinic cell") {
    const rgpot::EAMTablePot pot = pair_only();
    const std::vector<double> pos{0.2, 0.3, 0.4};
    const std::vector<int> z{29};
    const double box[9] = {3.6, 0.0, 0.0, 1.8, 3.1, 0.0, 1.8, 1.0, 2.9};
    check(pot, pos, z, box, 1e-11, 1e-9, 1e-11, 1e-11);
  }

  SECTION("Cu_u3 one-atom cubic cell a=3.615A (3x3x3 supercell/N)") {
    const rgpot::EAMTablePot pot = cu_u3();
    const std::vector<double> pos{0.3, 0.5, 0.7};
    const std::vector<int> z{29};
    const double a = 3.615;
    const double box[9] = {a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
    REQUIRE(face_spacing(box, 0) < pot.locality().cutoff);
    const Eval got = eval_full(pot, pos, z, box);
    const Eval want = supercell_reference(pot, pos, z, box, {3, 3, 3});
    REQUIRE_THAT(want.energy, WithinAbs(-0.802510540848, 5e-10));
    REQUIRE_THAT(want.stress[0], WithinAbs(0.0321577087, 5e-10));
    REQUIRE(std::abs(want.energy) > 1e-6);
    REQUIRE(got.has_stress == 1);
    REQUIRE(want.has_stress == 1);
    REQUIRE_THAT(got.energy, WithinAbs(want.energy, 1e-8));
    for (std::size_t c = 0; c < got.forces.size(); ++c) {
      REQUIRE_THAT(got.forces[c], WithinAbs(want.forces[c], 1e-7));
    }
    for (int c = 0; c < 9; ++c) {
      REQUIRE_THAT(got.stress[static_cast<std::size_t>(c)],
                   WithinAbs(want.stress[static_cast<std::size_t>(c)], 1e-10));
    }
    for (std::size_t i = 0; i < got.eatom.size(); ++i) {
      REQUIRE_THAT(got.eatom[i], WithinAbs(want.eatom[i], 1e-8));
    }
    const std::array<double, 6> gv = total_virial(got);
    const std::array<double, 6> wv = total_virial(want);
    for (int c = 0; c < 6; ++c) {
      REQUIRE_THAT(gv[static_cast<std::size_t>(c)],
                   WithinAbs(wv[static_cast<std::size_t>(c)], 1e-8));
    }
  }

  SECTION("Cu_u3 one-atom cubic cell a=3.0A (4x4x4 supercell/N)") {
    const rgpot::EAMTablePot pot = cu_u3();
    const std::vector<double> pos{0.25, 0.5, 0.75};
    const std::vector<int> z{29};
    const double a = 3.0;
    const double box[9] = {a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
    REQUIRE(face_spacing(box, 0) < pot.locality().cutoff);
    const Eval got = eval_full(pot, pos, z, box);
    const Eval want = supercell_reference(pot, pos, z, box, {4, 4, 4});
    REQUIRE_THAT(want.energy, WithinAbs(-1.941358963766, 5e-10));
    REQUIRE_THAT(want.stress[0], WithinAbs(0.0855559087, 5e-10));
    REQUIRE(std::abs(want.energy) > 1e-6);
    REQUIRE(got.has_stress == 1);
    REQUIRE(want.has_stress == 1);
    REQUIRE_THAT(got.energy, WithinAbs(want.energy, 2e-8));
    for (std::size_t c = 0; c < got.forces.size(); ++c) {
      REQUIRE_THAT(got.forces[c], WithinAbs(want.forces[c], 2e-7));
    }
    for (int c = 0; c < 9; ++c) {
      REQUIRE_THAT(got.stress[static_cast<std::size_t>(c)],
                   WithinAbs(want.stress[static_cast<std::size_t>(c)], 2e-10));
    }
    for (std::size_t i = 0; i < got.eatom.size(); ++i) {
      REQUIRE_THAT(got.eatom[i], WithinAbs(want.eatom[i], 2e-8));
    }
    const std::array<double, 6> gv = total_virial(got);
    const std::array<double, 6> wv = total_virial(want);
    for (int c = 0; c < 6; ++c) {
      REQUIRE_THAT(gv[static_cast<std::size_t>(c)],
                   WithinAbs(wv[static_cast<std::size_t>(c)], 2e-8));
    }
  }

  SECTION("Cu_u3 vacancy cell") {
    const rgpot::EAMTablePot pot = cu_u3();
    const double a = 3.615;
    const double box[9] = {a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
    // One missing site in a conventional FCC cell.
    const std::vector<double> pos{0.0,     0.0,     0.0, 0.0,    0.5 * a,
                                  0.5 * a, 0.5 * a, 0.0, 0.5 * a};
    const std::vector<int> z{29, 29, 29};
    check(pot, pos, z, box, 2e-8, 2e-7, 2e-10, 2e-8);
  }
}

namespace {

constexpr double kEnergyTol = 1e-9;
constexpr double kForceTol = 1e-9;
constexpr double kStressTol = 1e-9;
constexpr double kMinSignal = 1e-3;

using Vec3 = std::array<double, 3>;

Vec3 cross(const Vec3 &u, const Vec3 &v) {
  return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
          u[0] * v[1] - u[1] * v[0]};
}

double dot(const Vec3 &u, const Vec3 &v) {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

/// One tabulated function: the cubic Hermite interpolant on each knot
/// interval, with the derivative estimates the table format fixes (a
/// five point rule inside, one and two point rules at the ends). Written
/// as a Hermite basis, not as the stored coefficient layout.
class Knots {
public:
  Knots(const std::vector<double> &f, double delta) : f_(f), delta_(delta) {
    const std::size_t n = f.size();
    REQUIRE(n >= 5);
    d_.assign(n, 0.0);
    d_[0] = f[1] - f[0];
    d_[1] = 0.5 * (f[2] - f[0]);
    d_[n - 2] = 0.5 * (f[n - 1] - f[n - 3]);
    d_[n - 1] = f[n - 1] - f[n - 2];
    for (std::size_t m = 2; m + 2 < n; ++m) {
      d_[m] = ((f[m - 2] - f[m + 2]) + 8.0 * (f[m + 1] - f[m - 1])) / 12.0;
    }
  }

  [[nodiscard]] double value(double x) const {
    const Spot s = locate(x);
    const double p = s.p;
    const double p2 = p * p;
    const double p3 = p2 * p;
    return (2.0 * p3 - 3.0 * p2 + 1.0) * f_[s.k] +
           (p3 - 2.0 * p2 + p) * d_[s.k] +
           (-2.0 * p3 + 3.0 * p2) * f_[s.k + 1] + (p3 - p2) * d_[s.k + 1];
  }

  [[nodiscard]] double slope(double x) const {
    const Spot s = locate(x);
    const double p = s.p;
    const double p2 = p * p;
    return ((6.0 * p2 - 6.0 * p) * f_[s.k] +
            (3.0 * p2 - 4.0 * p + 1.0) * d_[s.k] +
            (-6.0 * p2 + 6.0 * p) * f_[s.k + 1] +
            (3.0 * p2 - 2.0 * p) * d_[s.k + 1]) /
           delta_;
  }

private:
  struct Spot {
    std::size_t k;
    double p;
  };

  [[nodiscard]] Spot locate(double x) const {
    const double s = x / delta_;
    const std::size_t last = f_.size() - 2;
    std::size_t k = s < 0.0 ? 0 : static_cast<std::size_t>(s);
    if (k > last) {
      k = last;
    }
    double p = s - static_cast<double>(k);
    if (p > 1.0) {
      p = 1.0;
    }
    return {k, p};
  }

  std::vector<double> f_;
  std::vector<double> d_;
  double delta_;
};

/// A periodic image of atom j seen from atom i: d = x_i - x_j - S.
struct Image {
  std::size_t i;
  std::size_t j;
  Vec3 d;
  double r;
};

Vec3 lattice(const double *box, int axis) {
  return {box[3 * axis], box[3 * axis + 1], box[3 * axis + 2]};
}

/// Reciprocal lattice vectors, so rec[a] . x is the fractional
/// coordinate of x along axis a and 1 / |rec[a]| is the face spacing.
std::array<Vec3, 3> reciprocal(const double *box) {
  const Vec3 a = lattice(box, 0);
  const Vec3 b = lattice(box, 1);
  const Vec3 c = lattice(box, 2);
  const double vol = dot(a, cross(b, c));
  REQUIRE(std::abs(vol) > 0.0);
  std::array<Vec3, 3> rec{cross(b, c), cross(c, a), cross(a, b)};
  for (Vec3 &v : rec) {
    for (double &x : v) {
      x /= vol;
    }
  }
  return rec;
}

double reciprocal_spacing(const std::array<Vec3, 3> &rec, int axis) {
  return 1.0 / std::sqrt(dot(rec[static_cast<std::size_t>(axis)],
                             rec[static_cast<std::size_t>(axis)]));
}

int short_axes(const double *box, double cutoff) {
  const std::array<Vec3, 3> rec = reciprocal(box);
  int count = 0;
  for (int axis = 0; axis < 3; ++axis) {
    if (!(2.0 * cutoff < reciprocal_spacing(rec, axis))) {
      ++count;
    }
  }
  return count;
}

/// Every ordered (i, j, S) with 0 < |d| < cutoff. Atoms are folded into
/// the home cell and a generous cube of shifts is scanned around it, so
/// nothing here depends on the production image extent.
std::vector<Image> brute_images(const std::vector<double> &pos,
                                const double *box, double cutoff) {
  const std::array<Vec3, 3> rec = reciprocal(box);
  const std::size_t n = pos.size() / 3;
  const Vec3 lat[3] = {lattice(box, 0), lattice(box, 1), lattice(box, 2)};
  std::vector<Vec3> home(n);
  for (std::size_t i = 0; i < n; ++i) {
    const Vec3 x{pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
    Vec3 w{0.0, 0.0, 0.0};
    for (std::size_t a = 0; a < 3; ++a) {
      const double f = dot(rec[a], x);
      const double frac = f - std::floor(f);
      for (std::size_t c = 0; c < 3; ++c) {
        w[c] += frac * lat[a][c];
      }
    }
    home[i] = w;
  }
  int reach[3];
  for (int a = 0; a < 3; ++a) {
    reach[a] =
        static_cast<int>(std::ceil(cutoff / reciprocal_spacing(rec, a))) + 2;
  }
  std::vector<Image> out;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (int h = -reach[0]; h <= reach[0]; ++h) {
        for (int k = -reach[1]; k <= reach[1]; ++k) {
          for (int l = -reach[2]; l <= reach[2]; ++l) {
            Vec3 d;
            for (std::size_t c = 0; c < 3; ++c) {
              d[c] = home[i][c] - home[j][c] - h * lat[0][c] - k * lat[1][c] -
                     l * lat[2][c];
            }
            const double r = std::sqrt(dot(d, d));
            if (r > 0.0 && r < cutoff) {
              out.push_back({i, j, d, r});
            }
          }
        }
      }
    }
  }
  return out;
}

/// A single element table evaluated by summing brute_images.
class BruteForceEam {
public:
  explicit BruteForceEam(const rgpot::EAMTable &t)
      : frho_(single(t).frho[0], t.drho), rhor_(t.rhor[0], t.dr),
        z2r_(t.z2r[0], t.dr), cutoff_(t.cutoff), rhomax_(t.rhomax) {}

  [[nodiscard]] double cutoff() const { return cutoff_; }

  [[nodiscard]] Eval evaluate(const std::vector<double> &pos,
                              const double *box) const {
    const std::vector<Image> images = brute_images(pos, box, cutoff_);
    const std::size_t n = pos.size() / 3;
    const double volume =
        std::abs(dot(lattice(box, 0), cross(lattice(box, 1), lattice(box, 2))));
    std::vector<double> rho(n, 0.0);
    for (const Image &im : images) {
      rho[im.i] += rhor_.value(im.r);
    }
    Eval e;
    e.eatom.assign(n, 0.0);
    e.forces.assign(3 * n, 0.0);
    std::vector<double> slope(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      slope[i] = frho_.slope(rho[i]);
      double embed = frho_.value(rho[i]);
      if (rho[i] > rhomax_) {
        embed += slope[i] * (rho[i] - rhomax_);
      }
      e.eatom[i] = embed;
    }
    for (const Image &im : images) {
      const double phi = z2r_.value(im.r) / im.r;
      const double dphi = z2r_.slope(im.r) / im.r - phi / im.r;
      e.eatom[im.i] += 0.5 * phi;
      const double scale =
          -((slope[im.i] + slope[im.j]) * rhor_.slope(im.r) + dphi) / im.r;
      for (std::size_t c = 0; c < 3; ++c) {
        e.forces[3 * im.i + c] += scale * im.d[c];
        for (std::size_t b = 0; b < 3; ++b) {
          e.stress[3 * c + b] -= 0.5 * scale * im.d[c] * im.d[b] / volume;
        }
      }
    }
    e.has_stress = 1;
    for (const double v : e.eatom) {
      e.energy += v;
    }
    return e;
  }

private:
  static const rgpot::EAMTable &single(const rgpot::EAMTable &t) {
    REQUIRE(t.z.size() == 1);
    REQUIRE(t.frho.size() == 1);
    REQUIRE(t.rhor.size() == 1);
    REQUIRE(t.z2r.size() == 1);
    return t;
  }

  Knots frho_;
  Knots rhor_;
  Knots z2r_;
  double cutoff_;
  double rhomax_;
};

rgpot::EAMTable cu_table() {
  std::ifstream in(kData + "eam_table/Cu_u3.eam");
  REQUIRE(in.good());
  return rgpot::read_eam_table(in, EAMTableFormat::Funcfl);
}

rgpot::EAMTable pair_table_data() {
  std::istringstream in(pair_table());
  return rgpot::read_eam_table(in, EAMTableFormat::Setfl);
}

std::size_t multi_image_pairs(const std::vector<Image> &images, std::size_t n) {
  std::vector<std::size_t> count(n * n, 0);
  for (const Image &im : images) {
    if (im.i < im.j) {
      ++count[im.i * n + im.j];
    }
  }
  std::size_t pairs = 0;
  for (const std::size_t c : count) {
    if (c > 1) {
      ++pairs;
    }
  }
  return pairs;
}

void expect_same_eval(const Eval &got, const Eval &want) {
  REQUIRE_THAT(got.energy, WithinAbs(want.energy, kEnergyTol));
  REQUIRE(got.forces.size() == want.forces.size());
  for (std::size_t c = 0; c < got.forces.size(); ++c) {
    REQUIRE_THAT(got.forces[c], WithinAbs(want.forces[c], kForceTol));
  }
  REQUIRE(got.has_stress == 1);
  REQUIRE(want.has_stress == 1);
  for (std::size_t c = 0; c < got.stress.size(); ++c) {
    REQUIRE_THAT(got.stress[c], WithinAbs(want.stress[c], kStressTol));
  }
  REQUIRE(got.eatom.size() == want.eatom.size());
  for (std::size_t i = 0; i < got.eatom.size(); ++i) {
    REQUIRE_THAT(got.eatom[i], WithinAbs(want.eatom[i], kEnergyTol));
  }
}

void expect_matches_brute_force(const rgpot::EAMTable &table,
                                const std::vector<double> &pos,
                                const double *box) {
  const rgpot::EAMTablePot pot(table, true, true);
  const std::vector<int> z(pos.size() / 3, table.z[0]);
  const Eval got = eval_full(pot, pos, z, box);
  const Eval want = BruteForceEam(table).evaluate(pos, box);
  REQUIRE(std::abs(want.energy) > kMinSignal);
  if (z.size() > 1) {
    double largest = 0.0;
    for (const double f : want.forces) {
      largest = std::max(largest, std::abs(f));
    }
    REQUIRE(largest > kMinSignal);
  }
  expect_same_eval(got, want);
}

} // namespace

TEST_CASE("EAM neighbour lists match a brute force sum over every image",
          "[EAMTable][nlist]") {
  const rgpot::EAMTable cu = cu_table();
  const double cutoff = cu.cutoff;

  SECTION("control: orthogonal 12A cell, minimum image only") {
    const double box[9] = {12.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 12.0};
    REQUIRE(short_axes(box, cutoff) == 0);
    const std::vector<double> pos{1.0, 2.0, 3.0, 3.2, 2.4, 3.5,
                                  2.2, 4.5, 3.2, 4.0, 3.5, 5.0};
    expect_matches_brute_force(cu, pos, box);
  }

  SECTION("control: triclinic 12A cell, minimum image only") {
    const double box[9] = {12.0, 0.0, 0.0, 0.5, 12.4, 0.0, -0.4, 0.6, 13.0};
    REQUIRE(short_axes(box, cutoff) == 0);
    const std::vector<double> pos{1.0, 1.5, 2.0, 3.4, 1.8, 2.6,
                                  1.6, 3.6, 2.4, 3.0, 3.2, 4.2};
    expect_matches_brute_force(cu, pos, box);
  }

  SECTION("cubic 4A cell, every face short") {
    const double box[9] = {4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
    REQUIRE(short_axes(box, cutoff) == 3);
    const std::vector<double> pos{0.3, 0.4, 0.5, 1.9, 2.2, 0.9, 2.7, 1.1, 3.1};
    REQUIRE(multi_image_pairs(brute_images(pos, box, cutoff), 3) > 0);
    expect_matches_brute_force(cu, pos, box);
  }

  SECTION("cubic 2.9A cell, second shell needed") {
    // cutoff / spacing is 1.7, so an image two cells away is inside the
    // cutoff for a pair offset by 0.4 of a cell.
    const double box[9] = {2.9, 0.0, 0.0, 0.0, 2.9, 0.0, 0.0, 0.0, 2.9};
    REQUIRE(short_axes(box, cutoff) == 3);
    const std::vector<double> pos{0.2, 0.3, 0.4, 1.36, 0.5, 0.6, 2.0, 1.9, 1.1};
    REQUIRE(multi_image_pairs(brute_images(pos, box, cutoff), 3) > 0);
    expect_matches_brute_force(cu, pos, box);
  }

  SECTION("orthogonal slab, short along z only") {
    const double box[9] = {12.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 7.0};
    REQUIRE(short_axes(box, cutoff) == 1);
    REQUIRE(2.0 * cutoff < face_spacing(box, 0));
    REQUIRE(2.0 * cutoff < face_spacing(box, 1));
    REQUIRE(face_spacing(box, 2) < 2.0 * cutoff);
    const std::vector<double> pos{1.0, 2.0, 1.0, 3.2, 2.4, 4.5,
                                  2.2, 4.5, 3.2, 4.0, 3.5, 5.0};
    REQUIRE(multi_image_pairs(brute_images(pos, box, cutoff), 4) > 0);
    expect_matches_brute_force(cu, pos, box);
  }

  SECTION("orthogonal slab, short along x only") {
    const double box[9] = {7.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 12.0};
    REQUIRE(short_axes(box, cutoff) == 1);
    REQUIRE(face_spacing(box, 0) < 2.0 * cutoff);
    const std::vector<double> pos{1.0, 2.0, 1.0, 4.5, 2.4, 3.2,
                                  3.2, 4.5, 2.2, 5.0, 3.5, 4.0};
    REQUIRE(multi_image_pairs(brute_images(pos, box, cutoff), 4) > 0);
    expect_matches_brute_force(cu, pos, box);
  }

  SECTION("triclinic cell, every face short") {
    const double box[9] = {6.0, 0.0, 0.0, 2.0, 5.5, 0.0, -1.5, 1.2, 5.0};
    REQUIRE(short_axes(box, cutoff) == 3);
    const std::vector<double> pos{0.5, 0.7, 0.9, 3.1, 2.2, 1.4, 1.2, 3.9, 3.3};
    REQUIRE(multi_image_pairs(brute_images(pos, box, cutoff), 3) > 0);
    expect_matches_brute_force(cu, pos, box);
  }

  SECTION("triclinic cell, short along c only") {
    const double box[9] = {12.0, 0.0, 0.0, 0.5, 12.4, 0.0, -0.4, 0.6, 7.0};
    REQUIRE(short_axes(box, cutoff) == 1);
    REQUIRE(face_spacing(box, 2) < 2.0 * cutoff);
    const std::vector<double> pos{1.0, 1.5, 1.0, 3.4, 1.8, 4.6,
                                  1.6, 3.6, 2.4, 3.0, 3.2, 5.2};
    REQUIRE(multi_image_pairs(brute_images(pos, box, cutoff), 4) > 0);
    expect_matches_brute_force(cu, pos, box);
  }

  SECTION("pair-only table on a short triclinic cell") {
    const rgpot::EAMTable pairs = pair_table_data();
    const double box[9] = {3.6, 0.0, 0.0, 1.8, 3.1, 0.0, 1.8, 1.0, 2.9};
    REQUIRE(short_axes(box, pairs.cutoff) == 3);
    const std::vector<double> pos{0.2, 0.3, 0.4, 1.7, 1.1, 0.8};
    expect_matches_brute_force(pairs, pos, box);
  }

  SECTION("unwrapped positions on a short cell") {
    const double box[9] = {6.0, 0.0, 0.0, 2.0, 5.5, 0.0, -1.5, 1.2, 5.0};
    const std::vector<double> home{0.5, 0.7, 0.9, 3.1, 2.2, 1.4};
    std::vector<double> far = home;
    for (std::size_t c = 0; c < 3; ++c) {
      far[c] += 4.0 * box[c] - 3.0 * box[3 + c] + 2.0 * box[6 + c];
      far[3 + c] -= 5.0 * box[c] + box[3 + c] - 6.0 * box[6 + c];
    }
    const rgpot::EAMTablePot pot(cu, true, true);
    const std::vector<int> z(2, cu.z[0]);
    const Eval got = eval_full(pot, far, z, box);
    const Eval want = BruteForceEam(cu).evaluate(home, box);
    REQUIRE(std::abs(want.energy) > kMinSignal);
    expect_same_eval(got, want);
  }
}

TEST_CASE(
    "EAM multi-image counts a second image between cutoff and twice cutoff",
    "[EAMTable][nlist]") {
  const rgpot::EAMTablePot pot = cu_u3();
  const double a = 8.0;
  const double cutoff = pot.locality().cutoff;
  const double box[9] = {a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
  REQUIRE(face_spacing(box, 0) > cutoff);
  REQUIRE(face_spacing(box, 0) < 2.0 * cutoff);
  // Atom 1 sits 4 A from atom 0, so both the home image and the neighbour
  // cell image fall inside the 4.95 A cutoff.
  const std::vector<double> pos{0.2, 0.3, 0.4, 4.2, 0.3, 0.5, 0.3, 4.5, 0.4};
  const std::vector<int> z{29, 29, 29};
  REQUIRE(replication(box, cutoff) == std::array<int, 3>{2, 2, 2});
  check(pot, pos, z, box, 1e-8, 1e-7, 1e-10, 1e-8);
}

TEST_CASE("EAM multi-image recenters pairs outside the home cell",
          "[EAMTable][nlist]") {
  const rgpot::EAMTablePot pot = cu_u3();
  const double a = 8.0;
  const double cutoff = pot.locality().cutoff;
  const double box[9] = {a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
  REQUIRE(face_spacing(box, 0) < 2.0 * cutoff);
  // 80.2, 17.1, 24.4 and 184.5, 16.3, 25.2 are the images of
  // 0.2, 1.1, 0.4 and 0.5, 0.3, 1.2. The raw separation is many cells.
  const std::vector<double> unwrapped{80.2, 17.1, 24.4, 184.5, 16.3, 25.2};
  const std::vector<double> wrapped{0.2, 1.1, 0.4, 0.5, 0.3, 1.2};
  const std::vector<int> z{29, 29};
  const Eval got = eval_full(pot, unwrapped, z, box);
  const Eval want = supercell_reference(pot, wrapped, z, box);
  REQUIRE(std::abs(want.energy) > 1e-6);
  REQUIRE(got.has_stress == 1);
  REQUIRE(want.has_stress == 1);
  REQUIRE_THAT(got.energy, WithinAbs(want.energy, 1e-8));
  for (std::size_t c = 0; c < got.forces.size(); ++c) {
    REQUIRE_THAT(got.forces[c], WithinAbs(want.forces[c], 1e-7));
  }
  for (int c = 0; c < 9; ++c) {
    REQUIRE_THAT(got.stress[static_cast<std::size_t>(c)],
                 WithinAbs(want.stress[static_cast<std::size_t>(c)], 1e-10));
  }
  const std::array<double, 6> gv = total_virial(got);
  const std::array<double, 6> wv = total_virial(want);
  for (int c = 0; c < 6; ++c) {
    REQUIRE_THAT(gv[static_cast<std::size_t>(c)],
                 WithinAbs(wv[static_cast<std::size_t>(c)], 1e-8));
  }
}

TEST_CASE("EAM multi-image rejects unbounded image extent",
          "[EAMTable][nlist]") {
  const rgpot::EAMTablePot pot = pair_only();
  const std::vector<double> pos{0.4, 0.5, 0.6};
  const std::vector<int> z{29};
  const double bad_box[9] = {1.0e-6, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
  REQUIRE_THROWS_WITH(
      eval_full(pot, pos, z, bad_box),
      Catch::Matchers::ContainsSubstring("image extent on axis"));
}

TEST_CASE("EAM multi-image rejects a large image count", "[EAMTable][nlist]") {
  const rgpot::EAMTablePot pot = pair_only();
  const double cutoff = pot.locality().cutoff;
  // extent = ceil(cutoff / spacing) + 1. Stay under the per-axis cap and
  // over the product of the shift counts (2 * extent + 1).
  constexpr int extent = 40;
  REQUIRE(extent < rgpot::nlist::detail::kMaxImageExtent);
  const auto span = static_cast<std::int64_t>(2 * extent + 1);
  REQUIRE(span * span * span > rgpot::nlist::detail::kMaxImageCount);
  const double spacing = cutoff / (static_cast<double>(extent - 1) - 0.25);
  const double box[9] = {spacing, 0.0, 0.0, 0.0,    spacing,
                         0.0,     0.0, 0.0, spacing};
  const std::vector<double> pos{0.2 * spacing, 0.3 * spacing, 0.4 * spacing};
  const std::vector<int> z{29};
  REQUIRE_THROWS_WITH(eval_full(pot, pos, z, box),
                      Catch::Matchers::ContainsSubstring("image count"));
}
