// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/MEAM/MEAMPot.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rgpot/MEAM/MEAMReader.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/nlist/MultiImage.hpp"
#include "rgpot/stress.hpp"

namespace rgpot {

namespace {

constexpr double kTiny = 1e-20;

[[noreturn]] void bad(const std::string &what) {
  throw std::invalid_argument("MEAMPot: " + what);
}

bool is_zero(double f) { return std::fabs(f) < kTiny; }
bool is_one(double f) { return std::fabs(f - 1.0) < kTiny; }

double fdiv_zero(double n, double d) { return is_zero(d) ? 0.0 : n / d; }

/// The radial and screening cutoff, (1 - (1 - x)^4)^2 on 0 < x < 1.
double fcut(double x) {
  if (x >= 1.0) {
    return 1.0;
  }
  if (x <= 0.0) {
    return 0.0;
  }
  const double a = 1.0 - x;
  const double a4 = a * a * a * a;
  const double b = 1.0 - a4;
  return b * b;
}

/// fcut and its derivative in x.
double dfcut(double x, double *dfc) {
  if (x >= 1.0) {
    *dfc = 0.0;
    return 1.0;
  }
  if (x <= 0.0) {
    *dfc = 0.0;
    return 0.0;
  }
  const double a = 1.0 - x;
  const double a3 = a * a * a;
  const double a4 = a * a3;
  const double b = 1.0 - a4;
  *dfc = 8.0 * b * a3;
  return b * b;
}

// Symmetric tensor components in the order m <= n (<= p), with the number
// of full-tensor entries each stands for.
constexpr int kIdx2[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
constexpr double kMult2[6] = {1, 2, 2, 1, 2, 1};
constexpr int kIdx3[3][3][3] = {
    {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}},
    {{1, 3, 4}, {3, 6, 7}, {4, 7, 8}},
    {{2, 4, 5}, {4, 7, 8}, {5, 8, 9}},
};
constexpr double kMult3[10] = {1, 3, 3, 3, 6, 3, 1, 3, 3, 1};

/// Per-element numbers the density loops read.
struct Element {
  double rho0;
  double re;
  double beta[4];
  double t[4];
};

/// The partial density sums of one atom.
struct Density {
  double rho0 = 0.0;
  double tn[3] = {0, 0, 0};
  double tsq[3] = {0, 0, 0};
  double a1[3] = {0, 0, 0};
  double a2[6] = {0, 0, 0, 0, 0, 0};
  double a2b = 0.0;
  double a3[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double a3b[3] = {0, 0, 0};
};

/// Derivatives of the energy of one atom with respect to its density sums,
/// laid out so a bond reads them without further algebra. The sender b of a
/// bond enters the weight of rho0 through its t values.
struct Adjoint {
  bool active = false;
  double fp = 0.0;
  double l0 = 0.0;
  double l1[3] = {0, 0, 0};
  double l2[3] = {0, 0, 0};
  double g1[3] = {0, 0, 0};
  double g2[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  double g2b = 0.0;
  double g3[27] = {};
  double g3b[3] = {0, 0, 0};
};

/// One bond with its screening and the two density terms it carries.
struct Bond {
  std::int32_t i, j;
  std::int32_t entry; //!< Position in the neighbour list of i.
  double d[3];        //!< Vector from i to j.
  double r;
  double screen; //!< S_ij, the screening alone.
  double s;      //!< S_ij times the radial cutoff.
  double dfc;    //!< Derivative of the radial cutoff in its argument.
  double fc;     //!< The radial cutoff.
  double rr[2][4]; //!< R_l of the sender: [0] is j onto i, [1] is i onto j.
};

/// Every neighbour of every atom inside a radius, with the vector to it.
struct FullList {
  std::vector<std::int32_t> start;
  std::vector<std::int32_t> owner;
  std::vector<double> d;
  std::vector<double> r2;
};

struct Point {
  double x, y, z;
  std::int32_t owner;
};

void matvec_fractional(const double inv[9], const double *x, double *f) {
  for (int k = 0; k < 3; ++k) {
    f[k] = inv[k] * x[0] + inv[3 + k] * x[1] + inv[6 + k] * x[2];
  }
}

/// Neighbour list with a cell grid over the atoms and their periodic
/// images. The images are copies of each owned atom shifted by lattice
/// vectors, kept when they lie within one radius of the cell.
void build_list(const double *pos, std::size_t n, const double *box,
                bool periodic, double radius, FullList *list) {
  std::vector<Point> pts;
  pts.reserve(n);
  if (periodic) {
    double inv[9];
    if (!nlist::detail::invertColumns(box, inv)) {
      bad("periodic cell is singular");
    }
    double pad[3];
    int extent[3];
    for (int k = 0; k < 3; ++k) {
      const double h = nlist::faceSpacing(box, k);
      if (!(h > 0.0)) {
        bad("periodic cell is singular");
      }
      pad[k] = radius / h;
      const double e = std::ceil(pad[k]);
      if (!(e <= 64.0)) {
        bad("periodic image count is too large for the cell");
      }
      extent[k] = static_cast<int>(e);
    }
    for (std::size_t a = 0; a < n; ++a) {
      double f[3];
      matvec_fractional(inv, pos + 3 * a, f);
      for (int k = 0; k < 3; ++k) {
        f[k] -= std::floor(f[k]);
      }
      // The owned copy sits at fractional position f inside the cell.
      pts.push_back({f[0] * box[0] + f[1] * box[3] + f[2] * box[6],
                     f[0] * box[1] + f[1] * box[4] + f[2] * box[7],
                     f[0] * box[2] + f[1] * box[5] + f[2] * box[8],
                     static_cast<std::int32_t>(a)});
    }
    for (std::size_t a = 0; a < n; ++a) {
      double f[3];
      matvec_fractional(inv, pos + 3 * a, f);
      for (int k = 0; k < 3; ++k) {
        f[k] -= std::floor(f[k]);
      }
      for (int h0 = -extent[0]; h0 <= extent[0]; ++h0) {
        const double g0 = f[0] + h0;
        if (g0 < -pad[0] || g0 > 1.0 + pad[0]) {
          continue;
        }
        for (int h1 = -extent[1]; h1 <= extent[1]; ++h1) {
          const double g1 = f[1] + h1;
          if (g1 < -pad[1] || g1 > 1.0 + pad[1]) {
            continue;
          }
          for (int h2 = -extent[2]; h2 <= extent[2]; ++h2) {
            if (h0 == 0 && h1 == 0 && h2 == 0) {
              continue;
            }
            const double g2 = f[2] + h2;
            if (g2 < -pad[2] || g2 > 1.0 + pad[2]) {
              continue;
            }
            pts.push_back({g0 * box[0] + g1 * box[3] + g2 * box[6],
                           g0 * box[1] + g1 * box[4] + g2 * box[7],
                           g0 * box[2] + g1 * box[5] + g2 * box[8],
                           static_cast<std::int32_t>(a)});
          }
        }
      }
    }
  } else {
    for (std::size_t a = 0; a < n; ++a) {
      pts.push_back({pos[3 * a], pos[3 * a + 1], pos[3 * a + 2],
                     static_cast<std::int32_t>(a)});
    }
  }

  double lo[3] = {pts[0].x, pts[0].y, pts[0].z};
  double hi[3] = {lo[0], lo[1], lo[2]};
  for (const Point &p : pts) {
    const double c[3] = {p.x, p.y, p.z};
    for (int k = 0; k < 3; ++k) {
      lo[k] = std::min(lo[k], c[k]);
      hi[k] = std::max(hi[k], c[k]);
    }
  }
  int nb[3];
  double inv_bin[3];
  for (int k = 0; k < 3; ++k) {
    const double extent = hi[k] - lo[k];
    const double cells = std::floor(extent / radius);
    nb[k] = static_cast<int>(std::clamp(cells, 1.0, 256.0));
    inv_bin[k] = nb[k] / std::max(extent, radius);
  }
  const std::size_t nbins = static_cast<std::size_t>(nb[0]) *
                            static_cast<std::size_t>(nb[1]) *
                            static_cast<std::size_t>(nb[2]);
  auto bin_of = [&](const Point &p, int c[3]) {
    const double v[3] = {p.x, p.y, p.z};
    for (int k = 0; k < 3; ++k) {
      c[k] = std::clamp(static_cast<int>((v[k] - lo[k]) * inv_bin[k]), 0,
                        nb[k] - 1);
    }
  };
  auto flat = [&](const int c[3]) {
    return (static_cast<std::size_t>(c[0]) * static_cast<std::size_t>(nb[1]) +
            static_cast<std::size_t>(c[1])) *
               static_cast<std::size_t>(nb[2]) +
           static_cast<std::size_t>(c[2]);
  };
  std::vector<std::int32_t> bin_start(nbins + 1, 0);
  std::vector<std::int32_t> bin_id(pts.size());
  for (std::size_t p = 0; p < pts.size(); ++p) {
    int c[3];
    bin_of(pts[p], c);
    bin_id[p] = static_cast<std::int32_t>(flat(c));
    ++bin_start[static_cast<std::size_t>(bin_id[p]) + 1];
  }
  for (std::size_t b = 0; b < nbins; ++b) {
    bin_start[b + 1] += bin_start[b];
  }
  std::vector<std::int32_t> fill(bin_start.begin(), bin_start.end() - 1);
  std::vector<std::int32_t> order(pts.size());
  for (std::size_t p = 0; p < pts.size(); ++p) {
    order[static_cast<std::size_t>(fill[static_cast<std::size_t>(bin_id[p])]++)] =
        static_cast<std::int32_t>(p);
  }

  // A bin side may be larger than the radius, never smaller, because the
  // count of bins is rounded down. Neighbouring bins then hold every point
  // within the radius.
  const double r2max = radius * radius;
  list->start.assign(n + 1, 0);
  list->owner.clear();
  list->d.clear();
  list->r2.clear();
  list->owner.reserve(n * 48);
  list->d.reserve(n * 48 * 3);
  list->r2.reserve(n * 48);
  for (std::size_t a = 0; a < n; ++a) {
    const Point &c = pts[a];
    int cc[3];
    bin_of(c, cc);
    for (int x = std::max(cc[0] - 1, 0); x <= std::min(cc[0] + 1, nb[0] - 1); ++x) {
      for (int y = std::max(cc[1] - 1, 0); y <= std::min(cc[1] + 1, nb[1] - 1); ++y) {
        for (int z = std::max(cc[2] - 1, 0); z <= std::min(cc[2] + 1, nb[2] - 1); ++z) {
          const int at[3] = {x, y, z};
          const std::size_t b = flat(at);
          for (std::int32_t q = bin_start[b]; q < bin_start[b + 1]; ++q) {
            const std::int32_t pi = order[static_cast<std::size_t>(q)];
            if (static_cast<std::size_t>(pi) == a) {
              continue;
            }
            const Point &o = pts[static_cast<std::size_t>(pi)];
            const double dx = o.x - c.x;
            const double dy = o.y - c.y;
            const double dz = o.z - c.z;
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < r2max && r2 > 0.0) {
              list->owner.push_back(o.owner);
              list->d.push_back(dx);
              list->d.push_back(dy);
              list->d.push_back(dz);
              list->r2.push_back(r2);
            }
          }
        }
      }
    }
    list->start[a + 1] = static_cast<std::int32_t>(list->owner.size());
  }
}

} // namespace

MEAMPot::MEAMPot(const MEAMConfig &config)
    : MEAMPot(load_meam_model(config.library, config.parameters, config.elements),
              config.periodic, config.stress, config.atomic_numbers) {}

MEAMPot::MEAMPot(MEAMModel model, bool periodic, bool stress,
                 std::vector<int> atomic_numbers)
    : Potential(PotType::MEAM), m_model{std::move(model)}, m_periodic{periodic},
      m_stress{stress} {
  const std::size_t n = m_model.size();
  if (!atomic_numbers.empty() && atomic_numbers.size() != n) {
    bad("atomic_numbers needs one entry per element");
  }
  for (std::size_t a = 0; a < n; ++a) {
    const MEAMElement &el = m_model.element(static_cast<int>(a));
    int z = atomic_numbers.empty() ? meam_atomic_number(el.name) : atomic_numbers[a];
    if (z == 0) {
      z = el.z;
    }
    if (z < 0) {
      bad("negative atomic number " + std::to_string(z));
    }
    if (static_cast<std::size_t>(z) >= m_element.size()) {
      m_element.resize(static_cast<std::size_t>(z) + 1, -1);
    }
    // Atoms take the first of two entries with one atomic number.
    if (m_element[static_cast<std::size_t>(z)] < 0) {
      m_element[static_cast<std::size_t>(z)] = static_cast<int>(a);
    }
  }
  Fnv1a fp;
  fp.u64(kKernelVersion);
  fp.u64(m_model.fingerprint());
  fp.u64(m_periodic ? 1 : 0);
  fp.u64(m_stress ? 1 : 0);
  for (std::size_t z = 0; z < m_element.size(); ++z) {
    fp.u64(static_cast<std::uint64_t>(m_element[z] + 1));
  }
  m_paramsKey = fp.h;
}

void MEAMPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (out == nullptr) {
    bad("null output buffer");
  }
  if (in.nAtoms > 0 && out->F == nullptr) {
    bad("null force buffer");
  }
  out->energy = 0.0;
  out->variance = 0.0;
  out->has_stress = 0;
  for (double &s : out->stress) {
    s = 0.0;
  }
  const std::size_t n = in.nAtoms;
  if (n == 0) {
    return;
  }
  if (in.pos == nullptr || in.atmnrs == nullptr ||
      (in.box == nullptr && (m_periodic || m_stress))) {
    bad("null input buffer");
  }
  std::fill(out->F, out->F + 3 * n, 0.0);

  const MEAMModel &m = m_model;
  const int ne = static_cast<int>(m.size());
  const double rc = m.rc();
  const double delr = m.delr();
  const double rc2 = rc * rc;
  const int ialloy = m.ialloy();
  const bool mix_ref = m.mixture_ref_t() == 1;
  const bool bkgd_dyn = m.bkgd_dyn() == 1;

  std::vector<Element> elem(static_cast<std::size_t>(ne));
  std::vector<std::array<double, 3>> shape(static_cast<std::size_t>(ne));
  std::vector<double> zref(static_cast<std::size_t>(ne));
  for (int a = 0; a < ne; ++a) {
    Element &e = elem[static_cast<std::size_t>(a)];
    e.rho0 = m.rho0(a);
    e.re = m.re(a);
    for (int l = 0; l < 4; ++l) {
      e.beta[l] = m.beta(a, l);
      e.t[l] = m.t(a, l);
    }
    const MEAMLattice latt = m.lattice(a, a);
    zref[static_cast<std::size_t>(a)] = meam_coordination(latt);
    MEAMModel::shape_factors(latt, m.stheta(a, a), m.ctheta(a, a),
                             shape[static_cast<std::size_t>(a)].data());
  }

  std::vector<int> type(n);
  for (std::size_t a = 0; a < n; ++a) {
    const int z = in.atmnrs[a];
    const int t = (z >= 0 && static_cast<std::size_t>(z) < m_element.size())
                      ? m_element[static_cast<std::size_t>(z)]
                      : -1;
    if (t < 0) {
      bad("atomic number " + std::to_string(z) + " is not in the library");
    }
    type[a] = t;
  }

  FullList list;
  build_list(in.pos, n, in.box, m_periodic, m.screening_radius(), &list);

  // Pairs, their screening and the radial cutoff.
  std::vector<Bond> bonds;
  bonds.reserve(list.owner.size() / 2 + 1);
  for (std::size_t i = 0; i < n; ++i) {
    const int ti = type[i];
    for (std::int32_t e = list.start[i]; e < list.start[i + 1]; ++e) {
      const std::size_t ue = static_cast<std::size_t>(e);
      const std::int32_t j = list.owner[ue];
      const double *d = &list.d[3 * ue];
      const double r2 = list.r2[ue];
      if (r2 >= rc2) {
        continue;
      }
      if (static_cast<std::size_t>(j) < i) {
        continue;
      }
      if (static_cast<std::size_t>(j) == i &&
          !(d[0] > 0.0 || (d[0] == 0.0 && (d[1] > 0.0 || (d[1] == 0.0 && d[2] > 0.0))))) {
        continue;
      }
      const int tj = type[static_cast<std::size_t>(j)];
      const double r = std::sqrt(r2);
      const double rbound = m.ebound(ti, tj) * r2;
      double screen = 1.0;
      for (std::int32_t k = list.start[i]; k < list.start[i + 1]; ++k) {
        if (k == e) {
          continue;
        }
        const std::size_t uk = static_cast<std::size_t>(k);
        const double rik2 = list.r2[uk];
        if (rik2 > rbound) {
          continue;
        }
        const double *dk = &list.d[3 * uk];
        const double jx = dk[0] - d[0];
        const double jy = dk[1] - d[1];
        const double jz = dk[2] - d[2];
        const double rjk2 = jx * jx + jy * jy + jz * jz;
        if (rjk2 > rbound) {
          continue;
        }
        const double xik = rik2 / r2;
        const double xjk = rjk2 / r2;
        const double a = 1.0 - (xik - xjk) * (xik - xjk);
        if (a <= 0.0) {
          continue;
        }
        const int tk = type[static_cast<std::size_t>(list.owner[uk])];
        const double cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
        const double cmax = m.cmax(ti, tj, tk);
        const double cmin = m.cmin(ti, tj, tk);
        if (cikj >= cmax) {
          continue;
        }
        if (cikj <= cmin) {
          screen = 0.0;
          break;
        }
        screen *= fcut((cikj - cmin) / (cmax - cmin));
      }
      if (is_zero(screen)) {
        continue;
      }
      Bond b{};
      b.i = static_cast<std::int32_t>(i);
      b.j = j;
      b.entry = e;
      b.d[0] = d[0];
      b.d[1] = d[1];
      b.d[2] = d[2];
      b.r = r;
      b.screen = screen;
      double dfc = 0.0;
      b.fc = dfcut((rc - r) / delr, &dfc);
      b.dfc = dfc;
      b.s = screen * b.fc;
      bonds.push_back(b);
    }
  }

  // Partial densities.
  std::vector<Density> dens(n);
  auto accumulate = [&](Density &rec, int tb, const double *u, double r, double s,
                        const double rr[4]) {
    const Element &eb = elem[static_cast<std::size_t>(tb)];
    const double q0 = rr[0] * s;
    rec.rho0 += q0;
    if (ialloy != 2) {
      for (int l = 0; l < 3; ++l) {
        rec.tn[l] += eb.t[l + 1] * q0;
      }
    }
    if (ialloy == 1) {
      for (int l = 0; l < 3; ++l) {
        rec.tsq[l] += eb.t[l + 1] * eb.t[l + 1] * q0;
      }
    }
    const double inv = 1.0 / r;
    const double ux = u[0] * inv, uy = u[1] * inv, uz = u[2] * inv;
    const double v[3] = {ux, uy, uz};
    const double p1 = rr[1] * s;
    const double p2 = rr[2] * s;
    const double p3 = rr[3] * s;
    for (int a = 0; a < 3; ++a) {
      rec.a1[a] += p1 * v[a];
      rec.a3b[a] += p3 * v[a];
    }
    rec.a2b += p2;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        rec.a2[kIdx2[a][b]] += p2 * v[a] * v[b];
        for (int c = b; c < 3; ++c) {
          rec.a3[kIdx3[a][b][c]] += p3 * v[a] * v[b] * v[c];
        }
      }
    }
  };
  auto radial = [&](int tb, double r, double rr[4], double drr[4]) {
    const Element &eb = elem[static_cast<std::size_t>(tb)];
    const double x = r / eb.re - 1.0;
    for (int l = 0; l < 4; ++l) {
      double v = eb.rho0 * std::exp(-eb.beta[l] * x);
      if (ialloy == 1 && l > 0) {
        v *= eb.t[l];
      }
      rr[l] = v;
      drr[l] = -eb.beta[l] / eb.re * v;
    }
  };
  std::vector<std::array<double, 8>> dr_store(bonds.size());
  for (std::size_t bi = 0; bi < bonds.size(); ++bi) {
    Bond &b = bonds[bi];
    const int ti = type[static_cast<std::size_t>(b.i)];
    const int tj = type[static_cast<std::size_t>(b.j)];
    double drj[4], dri[4];
    // Sender j onto i, then sender i onto j.
    radial(tj, b.r, b.rr[0], drj);
    radial(ti, b.r, b.rr[1], dri);
    for (int l = 0; l < 4; ++l) {
      dr_store[bi][l] = drj[l];
      dr_store[bi][4 + l] = dri[l];
    }
    const double neg[3] = {-b.d[0], -b.d[1], -b.d[2]};
    accumulate(dens[static_cast<std::size_t>(b.i)], tj, b.d, b.r, b.s, b.rr[0]);
    accumulate(dens[static_cast<std::size_t>(b.j)], ti, neg, b.r, b.s, b.rr[1]);
  }

  // Embedding energy and the derivatives of every atom's energy with
  // respect to its density sums.
  std::vector<Adjoint> adj(n);
  double energy = 0.0;
  for (std::size_t a = 0; a < n; ++a) {
    const Density &rec = dens[a];
    if (!(rec.rho0 > 0.0)) {
      continue;
    }
    const int ta = type[a];
    const Element &ea = elem[static_cast<std::size_t>(ta)];
    const int ibar = m.ibar(ta);
    const double rho0 = rec.rho0;

    double rho1 = 0.0;
    double rho2 = -rec.a2b * rec.a2b / 3.0;
    double rho3 = 0.0;
    for (int k = 0; k < 3; ++k) {
      rho1 += rec.a1[k] * rec.a1[k];
      rho3 -= 3.0 / 5.0 * rec.a3b[k] * rec.a3b[k];
    }
    for (int k = 0; k < 6; ++k) {
      rho2 += kMult2[k] * rec.a2[k] * rec.a2[k];
    }
    for (int k = 0; k < 10; ++k) {
      rho3 += kMult3[k] * rec.a3[k] * rec.a3[k];
    }

    double tave[3];
    if (ialloy == 1) {
      for (int l = 0; l < 3; ++l) {
        tave[l] = fdiv_zero(rec.tn[l], rec.tsq[l]);
      }
    } else if (ialloy == 2) {
      for (int l = 0; l < 3; ++l) {
        tave[l] = ea.t[l + 1];
      }
    } else {
      for (int l = 0; l < 3; ++l) {
        tave[l] = rec.tn[l] / rho0;
      }
    }
    const double rhol[3] = {rho1, rho2, rho3};
    const double gamma =
        (tave[0] * rho1 + tave[1] * rho2 + tave[2] * rho3) / (rho0 * rho0);

    double g = 0.0, dg = 0.0;
    if (!m.G(gamma, ibar, &g, &dg)) {
      bad("unsupported ibar " + std::to_string(ibar));
    }

    const double z = zref[static_cast<std::size_t>(ta)];
    const std::array<double, 3> &shp = shape[static_cast<std::size_t>(ta)];
    double bkgd;
    double gbar = 1.0, dgbar = 0.0;
    if (mix_ref) {
      if (ibar > 0) {
        const double gam = (tave[0] * shp[0] + tave[1] * shp[1] + tave[2] * shp[2]) / (z * z);
        static_cast<void>(m.G(gam, ibar, &gbar, &dgbar));
      }
      bkgd = ea.rho0 * z * gbar;
    } else if (bkgd_dyn) {
      bkgd = ea.rho0 * z;
    } else {
      bkgd = m.rho_ref(ta);
    }
    const double rhobar = rho0 * g / bkgd;
    double fp = 0.0;
    energy += m.embedding(ta, rhobar, &fp);

    Adjoint &ad = adj[a];
    ad.active = true;
    ad.fp = fp;
    const double dgamma1 = (g - 2.0 * dg * gamma) / bkgd;
    const double dgamma2 = dg / rho0 / bkgd;

    double tau[3];
    for (int l = 0; l < 3; ++l) {
      tau[l] = fp * dgamma2 * rhol[l];
      if (mix_ref && ibar > 0) {
        tau[l] -= fp * rho0 * g * dgbar * shp[static_cast<std::size_t>(l)] /
                  (gbar * z * z * bkgd);
      }
    }
    const double w[3] = {fp * dgamma2 * tave[0], fp * dgamma2 * tave[1],
                         fp * dgamma2 * tave[2]};

    ad.l0 = fp * dgamma1;
    if (ialloy == 0) {
      for (int l = 0; l < 3; ++l) {
        ad.l0 -= tau[l] * tave[l] / rho0;
        ad.l1[l] = tau[l] / rho0;
      }
    } else if (ialloy == 1) {
      for (int l = 0; l < 3; ++l) {
        const double c = fdiv_zero(tau[l], rec.tsq[l]);
        ad.l1[l] = c;
        ad.l2[l] = -c * tave[l];
      }
    }

    for (int k = 0; k < 3; ++k) {
      ad.g1[k] = 2.0 * w[0] * rec.a1[k];
      ad.g3b[k] = -6.0 / 5.0 * w[2] * rec.a3b[k];
    }
    ad.g2b = -2.0 / 3.0 * w[1] * rec.a2b;
    for (int p = 0; p < 3; ++p) {
      for (int q = 0; q < 3; ++q) {
        ad.g2[3 * p + q] = 2.0 * w[1] * rec.a2[kIdx2[p][q]];
        for (int r = 0; r < 3; ++r) {
          ad.g3[9 * p + 3 * q + r] = 2.0 * w[2] * rec.a3[kIdx3[p][q][r]];
        }
      }
    }
  }

  // Forces and the strain derivative.
  double strain[6] = {0, 0, 0, 0, 0, 0};
  double *F = out->F;
  auto side = [&](const Adjoint &ad, int tb, const double *u, double r,
                  const double rr[4], const double drr[4], double s, double grad[3],
                  double *deds) {
    if (!ad.active) {
      grad[0] = grad[1] = grad[2] = 0.0;
      *deds = 0.0;
      return;
    }
    const Element &eb = elem[static_cast<std::size_t>(tb)];
    double lam = ad.l0;
    if (ialloy != 2) {
      for (int l = 0; l < 3; ++l) {
        lam += ad.l1[l] * eb.t[l + 1] + ad.l2[l] * eb.t[l + 1] * eb.t[l + 1];
      }
    }
    const double inv = 1.0 / r;
    const double v[3] = {u[0] * inv, u[1] * inv, u[2] * inv};
    const double gv = ad.g1[0] * v[0] + ad.g1[1] * v[1] + ad.g1[2] * v[2];
    double g2v[3], g3vv[3];
    double q2 = 0.0, c3 = 0.0;
    for (int p = 0; p < 3; ++p) {
      g2v[p] = ad.g2[3 * p] * v[0] + ad.g2[3 * p + 1] * v[1] + ad.g2[3 * p + 2] * v[2];
      q2 += g2v[p] * v[p];
      double acc = 0.0;
      for (int q = 0; q < 3; ++q) {
        for (int w = 0; w < 3; ++w) {
          acc += ad.g3[9 * p + 3 * q + w] * v[q] * v[w];
        }
      }
      g3vv[p] = acc;
      c3 += acc * v[p];
    }
    const double hb = ad.g3b[0] * v[0] + ad.g3b[1] * v[1] + ad.g3b[2] * v[2];

    const double t1 = gv;
    const double t2 = q2 + ad.g2b;
    const double t3 = c3 + hb;
    *deds = lam * rr[0] + rr[1] * t1 + rr[2] * t2 + rr[3] * t3;

    const double radial_part =
        lam * drr[0] + drr[1] * t1 + drr[2] * t2 + drr[3] * t3;
    for (int p = 0; p < 3; ++p) {
      grad[p] = s * (radial_part * v[p] +
                     inv * (rr[1] * (ad.g1[p] - gv * v[p]) +
                            rr[2] * (2.0 * g2v[p] - 2.0 * q2 * v[p]) +
                            rr[3] * (3.0 * g3vv[p] - 3.0 * c3 * v[p] +
                                     ad.g3b[p] - hb * v[p])));
    }
  };

  for (std::size_t bi = 0; bi < bonds.size(); ++bi) {
    const Bond &b = bonds[bi];
    const std::size_t i = static_cast<std::size_t>(b.i);
    const std::size_t j = static_cast<std::size_t>(b.j);
    const int ti = type[i];
    const int tj = type[j];
    double phi = 0.0, dphi = 0.0;
    m.phi(m.pair_index(ti, tj), b.r, &phi, &dphi);
    energy += phi * b.s;

    const double neg[3] = {-b.d[0], -b.d[1], -b.d[2]};
    double gi[3], gj[3], ds_i, ds_j;
    side(adj[i], tj, b.d, b.r, b.rr[0], &dr_store[bi][0], b.s, gi, &ds_i);
    side(adj[j], ti, neg, b.r, b.rr[1], &dr_store[bi][4], b.s, gj, &ds_j);

    const double inv = 1.0 / b.r;
    double gij[3];
    for (int p = 0; p < 3; ++p) {
      gij[p] = gi[p] - gj[p] + dphi * b.s * b.d[p] * inv;
    }
    const double deds = phi + ds_i + ds_j;

    if (!is_zero(deds)) {
      // Radial cutoff.
      const double dr = deds * (-b.screen * b.dfc / delr) * inv;
      for (int p = 0; p < 3; ++p) {
        gij[p] += dr * b.d[p];
      }
      // Screening by third atoms.
      if (!is_one(b.screen)) {
        const double r2 = b.r * b.r;
        const double rbound = m.ebound(ti, tj) * r2;
        for (std::int32_t k = list.start[i]; k < list.start[i + 1]; ++k) {
          if (k == b.entry) {
            continue;
          }
          const std::size_t uk = static_cast<std::size_t>(k);
          const double rik2 = list.r2[uk];
          if (rik2 > rbound) {
            continue;
          }
          const double *dk = &list.d[3 * uk];
          const double djk[3] = {dk[0] - b.d[0], dk[1] - b.d[1], dk[2] - b.d[2]};
          const double rjk2 = djk[0] * djk[0] + djk[1] * djk[1] + djk[2] * djk[2];
          if (rjk2 > rbound) {
            continue;
          }
          const double xik = rik2 / r2;
          const double xjk = rjk2 / r2;
          const double a = 1.0 - (xik - xjk) * (xik - xjk);
          if (a <= 0.0) {
            continue;
          }
          const std::size_t kk = static_cast<std::size_t>(list.owner[uk]);
          const int tk = type[kk];
          const double cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
          const double cmax = m.cmax(ti, tj, tk);
          const double cmin = m.cmin(ti, tj, tk);
          if (cikj >= cmax || cikj <= cmin) {
            continue;
          }
          const double delc = cmax - cmin;
          double dfc = 0.0;
          const double sikj = dfcut((cikj - cmin) / delc, &dfc);
          if (is_zero(sikj)) {
            continue;
          }
          const double kappa = deds * b.fc * b.screen * dfc / (delc * sikj);

          const double dd = r2;
          const double pp = rik2;
          const double qq = rjk2;
          const double df = pp - qq;
          const double nn = 2.0 * (pp + qq) * dd - df * df - dd * dd;
          const double mm = dd * dd - df * df;
          const double nd = 2.0 * (pp + qq) - 2.0 * dd;
          const double np = 2.0 * dd - 2.0 * df;
          const double nq = 2.0 * dd + 2.0 * df;
          const double md = 2.0 * dd;
          const double mp = -2.0 * df;
          const double mq = 2.0 * df;
          const double im2 = 1.0 / (mm * mm);
          const double cd = (nd * mm - nn * md) * im2;
          const double cp = (np * mm - nn * mp) * im2;
          const double cq = (nq * mm - nn * mq) * im2;

          double gik[3], gjk[3];
          for (int p = 0; p < 3; ++p) {
            gij[p] += kappa * cd * 2.0 * b.d[p];
            gik[p] = kappa * cp * 2.0 * dk[p];
            gjk[p] = kappa * cq * 2.0 * djk[p];
          }
          F[3 * i + 0] += gik[0];
          F[3 * i + 1] += gik[1];
          F[3 * i + 2] += gik[2];
          F[3 * kk + 0] -= gik[0] + gjk[0];
          F[3 * kk + 1] -= gik[1] + gjk[1];
          F[3 * kk + 2] -= gik[2] + gjk[2];
          F[3 * j + 0] += gjk[0];
          F[3 * j + 1] += gjk[1];
          F[3 * j + 2] += gjk[2];
          accumulateBondStrain(strain, dk[0], dk[1], dk[2], gik[0], gik[1], gik[2]);
          accumulateBondStrain(strain, djk[0], djk[1], djk[2], gjk[0], gjk[1], gjk[2]);
        }
      }
    }
    for (int p = 0; p < 3; ++p) {
      F[3 * i + p] += gij[p];
      F[3 * j + p] -= gij[p];
    }
    accumulateBondStrain(strain, b.d[0], b.d[1], b.d[2], gij[0], gij[1], gij[2]);
  }

  out->energy = energy;
  if (m_stress) {
    publishCauchyStress(out, strain, in.box != nullptr ? cellVolume(in.box) : 0.0);
  }
}

} // namespace rgpot
