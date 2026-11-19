// MIT License
// Copyright 2023--present rgpot developers
//
// Staged three body evaluation on emulated hosts. A periodic cell is split
// into slabs. Each host holds the atoms of one slab and, as copies, every
// periodic image within Locality::ghostReach plus a skin. The neighbour
// list is the full list nlist::build stores, including the rows of the
// copies, so a centre can read each neighbour's list. The kernel evaluates
// owned centres. Copies out to twice the cutoff are a superset of the
// interaction cutoff, and the kernel keeps pairs inside its own cutoff.
// Summing the owned energies and reverse summing the forces written on
// copies reproduces the whole cell. An empty extra host is included.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <set>
#include <tuple>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/Staged.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/stress.hpp"

namespace {

constexpr int kNeighMask = 0x1FFFFFFF;
constexpr double kListSkin = 0.5;
constexpr double kLatticeA = 5.431;
/// Relative gap of energy, forces and virial against the whole cell.
/// Documented with the staged force stage in the C++ reference.
constexpr double kStagedThreeBodyRel = 1e-10;

enum class Width { ThickerThanReach, ThinnerThanReach };

struct Box {
  std::vector<double> pos;
  std::vector<int> z;
  double cell[9] = {};
};

double length3(double x, double y, double z) {
  return std::sqrt(x * x + y * y + z * z);
}

double edge_a(const double *h) { return length3(h[0], h[1], h[2]); }

/// n cells of diamond along each axis, side a, rattled, optionally sheared.
/// The first four basis sites are silicon. The tetrahedral four are carbon
/// when @p sic is set and silicon otherwise.
Box diamond(double a, int nx, int ny, int nz, bool triclinic, bool sic) {
  const double basis[8][3] = {{0, 0, 0},
                              {0.5, 0.5, 0},
                              {0.5, 0, 0.5},
                              {0, 0.5, 0.5},
                              {0.25, 0.25, 0.25},
                              {0.75, 0.75, 0.25},
                              {0.75, 0.25, 0.75},
                              {0.25, 0.75, 0.75}};
  const int carbon = sic ? 6 : 14;
  const double lx = a * nx;
  const double ly = a * ny;
  const double lz = a * nz;
  Box b;
  const double h[9] = {lx,
                       0,
                       0,
                       triclinic ? 0.1 * lx : 0.0,
                       ly,
                       0,
                       triclinic ? -0.08 * lx : 0.0,
                       triclinic ? 0.05 * ly : 0.0,
                       lz};
  std::copy(h, h + 9, b.cell);
  int k = 0;
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int l = 0; l < nz; ++l) {
        for (int s = 0; s < 8; ++s) {
          const double f0 = (i + basis[s][0]) / nx;
          const double f1 = (j + basis[s][1]) / ny;
          const double f2 = (l + basis[s][2]) / nz;
          for (int d = 0; d < 3; ++d) {
            const double r = f0 * h[d] + f1 * h[3 + d] + f2 * h[6 + d];
            b.pos.push_back(r + 0.06 * a * std::sin(1.37 * k + 0.9 * d));
          }
          b.z.push_back(s < 4 ? 14 : carbon);
          ++k;
        }
      }
    }
  }
  return b;
}

struct Whole {
  double energy = 0.0;
  double stress[9] = {};
  std::vector<double> forces;
};

template <typename P> Whole evaluate(const P &pot, const Box &b) {
  Whole r;
  r.forces.assign(3 * b.z.size(), 0.0);
  rgpot::ForceOut out{r.forces.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl({b.z.size(), b.pos.data(), b.z.data(), b.cell}, &out);
  r.energy = out.energy;
  std::copy(out.stress, out.stress + 9, r.stress);
  REQUIRE(out.has_stress != 0);
  return r;
}

struct Host {
  std::size_t nOwned = 0;
  std::vector<double> pos;
  std::vector<int> z;
  std::vector<std::size_t> source;
  std::vector<int> rows;
  std::vector<int> count;
  std::vector<std::vector<int>> neighbours;
  std::vector<const int *> first;

  [[nodiscard]] std::size_t size() const { return z.size(); }

  [[nodiscard]] rgpot::HalfList list() const {
    rgpot::HalfList l;
    l.nAtoms = size();
    l.nOwned = nOwned;
    l.pos = size() == 0 ? nullptr : pos.data();
    l.atmnrs = size() == 0 ? nullptr : z.data();
    l.nRows = rows.size();
    l.rows = rows.empty() ? nullptr : rows.data();
    l.count = size() == 0 ? nullptr : count.data();
    l.neighbours = size() == 0 ? nullptr : first.data();
    l.indexMask = kNeighMask;
    l.newton = true;
    return l;
  }
};

double dist2(const double *a, const double *b) {
  const double dx = a[0] - b[0];
  const double dy = a[1] - b[1];
  const double dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

/// Images needed along each axis so a search of @p reach cannot miss a
/// periodic copy. The heights are those of nlist::build.
void image_spans(const double *h, double reach, int span[3]) {
  const double a[3] = {h[0], h[1], h[2]};
  const double b[3] = {h[3], h[4], h[5]};
  const double c[3] = {h[6], h[7], h[8]};
  const double bx[3] = {b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2],
                        b[0] * c[1] - b[1] * c[0]};
  const double cx[3] = {c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2],
                        c[0] * a[1] - c[1] * a[0]};
  const double ax[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0]};
  const double vol = std::abs(rgpot::cellVolume(h));
  const double face[3] = {length3(bx[0], bx[1], bx[2]), length3(cx[0], cx[1], cx[2]),
                          length3(ax[0], ax[1], ax[2])};
  for (int k = 0; k < 3; ++k) {
    const double height = face[k] > 0.0 ? vol / face[k] : 0.0;
    span[k] = height > 0.0 ? static_cast<int>(std::ceil(reach / height)) : 1;
  }
}

/// Hosts for @p slabs slabs of the first fractional coordinate, plus one
/// host that owns nothing. Each occupied host builds a full list out to
/// @p list_cut plus the skin, with a row for every atom including copies.
std::vector<Host> split(const Box &b, int slabs, double gather, double list_cut) {
  const double *h = b.cell;
  const std::size_t n = b.z.size();
  std::vector<int> slab(n);
  for (std::size_t j = 0; j < n; ++j) {
    const double s2 = b.pos[3 * j + 2] / h[8];
    const double s1 = (b.pos[3 * j + 1] - s2 * h[7]) / h[4];
    double s0 = (b.pos[3 * j] - s1 * h[3] - s2 * h[6]) / h[0];
    s0 -= std::floor(s0);
    slab[j] = std::min(slabs - 1, static_cast<int>(s0 * slabs));
  }
  int span[3] = {};
  image_spans(h, gather, span);
  std::mt19937 shuffle(1234U + static_cast<unsigned>(slabs));
  std::vector<Host> hosts(static_cast<std::size_t>(slabs) + 1);
  const double open[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int s = 0; s < slabs; ++s) {
    Host &host = hosts[static_cast<std::size_t>(s)];
    auto add = [&](std::size_t j, const std::array<double, 3> &r) {
      host.pos.insert(host.pos.end(), r.begin(), r.end());
      host.z.push_back(b.z[j]);
      host.source.push_back(j);
    };
    for (std::size_t j = 0; j < n; ++j) {
      if (slab[j] == s) {
        add(j, {b.pos[3 * j], b.pos[3 * j + 1], b.pos[3 * j + 2]});
      }
    }
    host.nOwned = host.size();
    std::set<std::tuple<std::size_t, int, int, int>> seen;
    for (std::size_t j = 0; j < n; ++j) {
      for (int i0 = -span[0]; i0 <= span[0]; ++i0) {
        for (int i1 = -span[1]; i1 <= span[1]; ++i1) {
          for (int i2 = -span[2]; i2 <= span[2]; ++i2) {
            if (i0 == 0 && i1 == 0 && i2 == 0 && slab[j] == s) {
              continue;
            }
            const auto key = std::make_tuple(j, i0, i1, i2);
            if (!seen.insert(key).second) {
              continue;
            }
            std::array<double, 3> r{};
            for (int d = 0; d < 3; ++d) {
              r[static_cast<std::size_t>(d)] = b.pos[3 * j + static_cast<std::size_t>(d)] +
                                               i0 * h[d] + i1 * h[3 + d] + i2 * h[6 + d];
            }
            bool near = false;
            for (std::size_t a = 0; a < host.nOwned && !near; ++a) {
              near = dist2(r.data(), host.pos.data() + 3 * a) < gather * gather;
            }
            if (near) {
              add(j, r);
            }
          }
        }
      }
    }
    const std::size_t m = host.size();
    rgpot::nlist::Query query;
    query.cutoff = list_cut;
    query.skin = kListSkin;
    query.cut = rgpot::nlist::Cut::Loose;
    query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
    query.periodic = {false, false, false};
    query.fields = {false, false, false};
    rgpot::nlist::Table table;
    rgpot::nlist::build(table, host.pos.data(), m, open, query);
    host.neighbours.assign(m, {});
    for (std::size_t p = 0; p < table.first.size(); ++p) {
      host.neighbours[static_cast<std::size_t>(table.first[p])].push_back(table.second[p]);
    }
    host.count.assign(m, 0);
    for (std::size_t a = 0; a < m; ++a) {
      std::vector<int> &row = host.neighbours[a];
      std::shuffle(row.begin(), row.end(), shuffle);
      for (std::size_t q = 0; q < row.size(); q += 3) {
        row[q] |= 1 << 30;
      }
      host.count[a] = static_cast<int>(row.size());
      if (a < host.nOwned) {
        host.rows.push_back(static_cast<int>(a));
      }
    }
    std::shuffle(host.rows.begin(), host.rows.end(), shuffle);
    host.first.reserve(m);
    for (const auto &row : host.neighbours) {
      host.first.push_back(row.data());
    }
  }
  return hosts;
}

double max_abs(const std::vector<double> &v) {
  double m = 0.0;
  for (double x : v) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

double max_gap(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    m = std::max(m, std::abs(a[k] - b[k]));
  }
  return m;
}

struct Result {
  double energy = 0.0;
  double virial[6] = {};
  std::vector<double> forces;
};

template <typename List> void mark_full(List &view) { view.full = true; }

template <typename P> Result run(const P &pot, const Box &b, const std::vector<Host> &hosts) {
  Result r;
  r.forces.assign(3 * b.z.size(), 0.0);
  for (const Host &host : hosts) {
    const std::size_t m = host.size();
    rgpot::HalfList view = host.list();
    mark_full(view);
    if (m == 0) {
      rgpot::StageTally tally;
      pot.densityStage(view, nullptr);
      pot.forceStage(view, nullptr, nullptr, &tally);
      r.energy += tally.energy;
      continue;
    }
    std::vector<double> densities(m, 1.0);
    pot.densityStage(view, densities.data());
    for (double value : densities) {
      CHECK(value == 1.0);
    }
    std::vector<double> forces(3 * m, 0.0);
    rgpot::StageTally tally;
    pot.forceStage(view, nullptr, forces.data(), &tally);
    std::vector<double> bare(3 * m, 0.0);
    pot.forceStage(view, nullptr, bare.data(), nullptr);
    CHECK(max_gap(bare, forces) == 0.0);
    r.energy += tally.energy;
    for (int c = 0; c < 6; ++c) {
      r.virial[c] += tally.virial[c];
    }
    for (std::size_t a = 0; a < m; ++a) {
      const std::size_t j = host.source[a];
      for (int d = 0; d < 3; ++d) {
        r.forces[3 * j + d] += forces[3 * a + d];
      }
    }
  }
  return r;
}

template <typename P>
void check_staged_3body(const P &pot, const Box &box, int slabs, Width expect) {
  const auto range = pot.locality();
  REQUIRE(range.finite());
  REQUIRE(range.full);
  REQUIRE(range.reach == 2);
  REQUIRE(range.densities == 0);
  REQUIRE_FALSE(range.pairwise);
  REQUIRE(range.ghostReach() == Catch::Approx(2.0 * range.cutoff));
  REQUIRE(range.forceRange() == Catch::Approx(2.0 * range.cutoff));
  const double width = edge_a(box.cell) / static_cast<double>(slabs);
  if (expect == Width::ThinnerThanReach) {
    REQUIRE(width < range.ghostReach());
  } else {
    REQUIRE(width > range.ghostReach());
  }
  const double gather = range.ghostReach() + kListSkin;
  std::vector<Host> hosts = split(box, slabs, gather, range.ghostReach());
  REQUIRE(hosts.size() == static_cast<std::size_t>(slabs) + 1);
  REQUIRE(hosts.back().size() == 0);
  int copies = 0;
  int shifted = 0;
  int ghost_rows = 0;
  int ghost_neighbours = 0;
  for (std::size_t s = 0; s + 1 < hosts.size(); ++s) {
    const Host &host = hosts[s];
    REQUIRE(host.nOwned > 0);
    copies += static_cast<int>(host.size() - host.nOwned);
    for (std::size_t a = host.nOwned; a < host.size(); ++a) {
      const std::size_t j = host.source[a];
      const double dx = host.pos[3 * a] - box.pos[3 * j];
      const double dy = host.pos[3 * a + 1] - box.pos[3 * j + 1];
      const double dz = host.pos[3 * a + 2] - box.pos[3 * j + 2];
      if (dx != 0.0 || dy != 0.0 || dz != 0.0) {
        ++shifted;
      }
      if (host.count[a] > 0) {
        ++ghost_rows;
      }
    }
    for (int row : host.rows) {
      const int n = host.count[static_cast<std::size_t>(row)];
      const int *nb = host.neighbours[static_cast<std::size_t>(row)].data();
      for (int q = 0; q < n; ++q) {
        if ((nb[q] & kNeighMask) >= static_cast<int>(host.nOwned)) {
          ++ghost_neighbours;
        }
      }
    }
  }
  REQUIRE(copies > 0);
  REQUIRE(shifted > 0);
  REQUIRE(ghost_rows > 0);
  REQUIRE(ghost_neighbours > 0);
  const Whole whole = evaluate(pot, box);
  const Result got = run(pot, box, hosts);
  REQUIRE(std::abs(whole.energy) > 0.0);
  CHECK(std::abs(got.energy - whole.energy) < kStagedThreeBodyRel * std::abs(whole.energy));
  const double fscale = max_abs(whole.forces);
  REQUIRE(fscale > 0.0);
  CHECK(max_gap(got.forces, whole.forces) < kStagedThreeBodyRel * fscale);
  const double volume = rgpot::cellVolume(box.cell);
  const int voigt[6] = {0, 4, 8, 5, 2, 1};
  double whole_w[6] = {};
  double vscale = 0.0;
  double vgap = 0.0;
  for (int c = 0; c < 6; ++c) {
    whole_w[c] = -volume * whole.stress[voigt[c]];
    vscale = std::max(vscale, std::abs(whole_w[c]));
    vgap = std::max(vgap, std::abs(got.virial[c] - whole_w[c]));
  }
  REQUIRE(vscale > 0.0);
  CHECK(vgap < kStagedThreeBodyRel * vscale);
}

template <typename P>
void check_or_fail(const P &pot, bool sic, int nx, int ny, int nz, int slabs, bool triclinic,
                   Width expect) {
  if constexpr (!rgpot::StagedPotential<P>) {
    FAIL("the potential has no staged evaluation");
  } else {
    const Box box = diamond(kLatticeA, nx, ny, nz, triclinic, sic);
    check_staged_3body(pot, box, slabs, expect);
  }
}

} // namespace

TEST_CASE("Staged Stillinger Weber matches two subdomains", "[Staged][3body]") {
  check_or_fail(rgpot::StillingerWeberSi{}, false, 12, 2, 2, 2, false, Width::ThickerThanReach);
}

TEST_CASE("Staged Stillinger Weber matches four subdomains", "[Staged][3body]") {
  check_or_fail(rgpot::StillingerWeberSi{}, false, 12, 2, 2, 4, false, Width::ThickerThanReach);
}

TEST_CASE("Staged Stillinger Weber matches eight subdomains", "[Staged][3body]") {
  check_or_fail(rgpot::StillingerWeberSi{}, false, 12, 2, 2, 8, false, Width::ThickerThanReach);
}

TEST_CASE("Staged Tersoff matches two subdomains", "[Staged][3body]") {
  check_or_fail(rgpot::TersoffSiC{}, true, 9, 2, 2, 2, false, Width::ThickerThanReach);
}

TEST_CASE("Staged Tersoff matches four subdomains", "[Staged][3body]") {
  check_or_fail(rgpot::TersoffSiC{}, true, 9, 2, 2, 4, false, Width::ThickerThanReach);
}

TEST_CASE("Staged Tersoff matches eight subdomains", "[Staged][3body]") {
  check_or_fail(rgpot::TersoffSiC{}, true, 9, 2, 2, 8, false, Width::ThickerThanReach);
}

TEST_CASE("Staged Stillinger Weber matches a triclinic cell", "[Staged][3body]") {
  check_or_fail(rgpot::StillingerWeberSi{}, false, 4, 4, 4, 2, true, Width::ThickerThanReach);
}

TEST_CASE("Staged Tersoff matches a triclinic cell", "[Staged][3body]") {
  check_or_fail(rgpot::TersoffSiC{}, true, 4, 4, 4, 2, true, Width::ThickerThanReach);
}

TEST_CASE("Staged Stillinger Weber matches a thin subdomain", "[Staged][3body]") {
  check_or_fail(rgpot::StillingerWeberSi{}, false, 2, 2, 2, 2, false, Width::ThinnerThanReach);
}

TEST_CASE("Staged Tersoff matches a thin subdomain", "[Staged][3body]") {
  check_or_fail(rgpot::TersoffSiC{}, true, 2, 2, 2, 2, false, Width::ThinnerThanReach);
}
