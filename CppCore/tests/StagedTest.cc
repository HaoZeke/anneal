// MIT License
// Copyright 2023--present rgpot developers
//
// The staged evaluation of Staged.hpp on emulated hosts. A periodic box is
// split into slabs, each host holds the atoms of one slab and, as copies,
// every periodic image within the cutoff plus a skin of them, and builds
// a half list the way LAMMPS does with newton_pair on or off. The two
// stages with the exchange between them give the owners the forces, site
// energies and per-atom virials of the periodic box, and the hosts' tallies
// sum to its energy and virial. This is what pair_style rgpot's domain
// path relies on.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Locality.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/Staged.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/SuttonChen/SuttonChenScales.hpp"
#ifdef RGPOT_HAS_EAM_TABLE
#include "rgpot/EAMTable/EAMTablePot.hpp"
#endif

namespace {

constexpr int kNeighMask = 0x1FFFFFFF;
constexpr double kSkin = 1.0;

struct Box {
  std::vector<double> pos;
  std::vector<int> z;
  double cell[9] = {};
};

/// n^3 fcc cells of side a, displaced by a function of the index and
/// optionally sheared. Every second atom takes z2.
Box fcc(double a, int n, bool triclinic, int z1, int z2) {
  Box b;
  const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  const double side = a * n;
  const double h[9] = {side, 0, 0, triclinic ? 0.1 * side : 0.0, side, 0,
                       triclinic ? -0.08 * side : 0.0, triclinic ? 0.05 * side : 0.0, side};
  std::copy(h, h + 9, b.cell);
  int k = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int l = 0; l < n; ++l) {
        for (const auto &s : basis) {
          const double f[3] = {(i + s[0]) / n, (j + s[1]) / n, (l + s[2]) / n};
          for (int d = 0; d < 3; ++d) {
            const double r = f[0] * h[d] + f[1] * h[3 + d] + f[2] * h[6 + d];
            b.pos.push_back(r + 0.06 * a * std::sin(1.37 * k + 0.9 * d));
          }
          b.z.push_back(k % 2 == 0 ? z1 : z2);
          ++k;
        }
      }
    }
  }
  return b;
}

struct Whole {
  double energy = 0.0;
  std::vector<double> forces, atomEnergy, atomVirial;
};

template <typename P> Whole evaluate(const P &pot, const Box &b) {
  const std::size_t n = b.z.size();
  Whole r;
  r.forces.assign(3 * n, 0.0);
  r.atomEnergy.assign(n, 0.0);
  r.atomVirial.assign(6 * n, 0.0);
  rgpot::ForceOut out{r.forces.data(), 0.0, 0.0, {}, 0};
  const rgpot::ForceInput in{n, b.pos.data(), b.z.data(), b.cell};
  pot.forcePerAtom(in, &out, {r.atomEnergy.data(), r.atomVirial.data()});
  r.energy = out.energy;
  return r;
}

/// One host: its owned atoms, then copies; source[a] is the atom of the
/// box that atom a stands for.
struct Host {
  std::size_t nOwned = 0;
  std::vector<double> pos;
  std::vector<int> z;
  std::vector<std::size_t> source;
  std::vector<int> rows;
  std::vector<int> count;
  std::vector<std::vector<int>> neighbours;
  std::vector<const int *> first;
  bool newton = false;

  [[nodiscard]] std::size_t size() const { return z.size(); }

  [[nodiscard]] rgpot::HalfList list() const {
    rgpot::HalfList l;
    l.nAtoms = size();
    l.nOwned = nOwned;
    l.pos = pos.data();
    l.atmnrs = z.data();
    l.nRows = rows.size();
    l.rows = rows.data();
    l.count = count.data();
    l.neighbours = first.data();
    l.indexMask = kNeighMask;
    l.newton = newton;
    return l;
  }
};

double dist2(const double *a, const double *b) {
  const double dx = a[0] - b[0];
  const double dy = a[1] - b[1];
  const double dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

/// Under newton_pair on, one of the two hosts that hold a pair of an owned
/// atom and a copy lists it: the one that sees the copy above its atom in
/// z, then y, then x, as LAMMPS breaks ties within a bin.
bool above(const double *copy, const double *atom) {
  for (int d = 2; d >= 0; --d) {
    if (copy[d] != atom[d]) {
      return copy[d] > atom[d];
    }
  }
  return false;
}

/// Hosts for @p slabs slabs of the first fractional coordinate and one more
/// host that owns nothing. Rows and neighbours come in a shuffled order,
/// every third entry carries a bit above the index, as LAMMPS marks
/// special neighbours, and the list holds pairs out to cutoff + skin.
std::vector<Host> split(const Box &b, int slabs, double cutoff, bool newton) {
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
  const double reach = cutoff + kSkin;
  std::mt19937 shuffle(1234U + static_cast<unsigned>(slabs) + (newton ? 99U : 0U));
  std::vector<Host> hosts(static_cast<std::size_t>(slabs) + 1);
  for (int s = 0; s < slabs; ++s) {
    Host &host = hosts[static_cast<std::size_t>(s)];
    host.newton = newton;
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
    for (std::size_t j = 0; j < n; ++j) {
      for (int i0 = -1; i0 <= 1; ++i0) {
        for (int i1 = -1; i1 <= 1; ++i1) {
          for (int i2 = -1; i2 <= 1; ++i2) {
            if (i0 == 0 && i1 == 0 && i2 == 0 && slab[j] == s) {
              continue;
            }
            std::array<double, 3> r{};
            for (int d = 0; d < 3; ++d) {
              r[static_cast<std::size_t>(d)] =
                  b.pos[3 * j + static_cast<std::size_t>(d)] + i0 * h[d] + i1 * h[3 + d] +
                  i2 * h[6 + d];
            }
            bool near = false;
            for (std::size_t a = 0; a < host.nOwned && !near; ++a) {
              near = dist2(r.data(), host.pos.data() + 3 * a) < reach * reach;
            }
            if (near) {
              add(j, r);
            }
          }
        }
      }
    }
    const std::size_t m = host.size();
    host.neighbours.assign(m, {});
    host.count.assign(m, 0);
    for (std::size_t a = 0; a < host.nOwned; ++a) {
      const double *ra = host.pos.data() + 3 * a;
      std::vector<int> &row = host.neighbours[a];
      for (std::size_t c = a + 1; c < m; ++c) {
        const double *rc = host.pos.data() + 3 * c;
        if (dist2(ra, rc) >= reach * reach) {
          continue;
        }
        if (newton && c >= host.nOwned && !above(rc, ra)) {
          continue;
        }
        row.push_back(static_cast<int>(c));
      }
      std::shuffle(row.begin(), row.end(), shuffle);
      for (std::size_t q = 0; q < row.size(); q += 3) {
        row[q] |= 1 << 30;
      }
      host.count[a] = static_cast<int>(row.size());
      host.rows.push_back(static_cast<int>(a));
    }
    std::shuffle(host.rows.begin(), host.rows.end(), shuffle);
    for (const auto &row : host.neighbours) {
      host.first.push_back(row.data());
    }
  }
  hosts.back().newton = newton;
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
  std::vector<double> forces, atomEnergy, atomVirial;
};

/// Both stages on every host, with the exchange in between and the
/// returns of a newton host's copies to their owners after the force stage.
template <typename P> Result run(const P &pot, const Box &b, std::vector<Host> &hosts) {
  const std::size_t n = b.z.size();
  const auto width = static_cast<std::size_t>(pot.locality().densities);
  std::vector<std::size_t> ownerHost(n);
  std::vector<std::size_t> ownerIndex(n);
  for (std::size_t s = 0; s < hosts.size(); ++s) {
    for (std::size_t a = 0; a < hosts[s].nOwned; ++a) {
      ownerHost[hosts[s].source[a]] = s;
      ownerIndex[hosts[s].source[a]] = a;
    }
  }
  std::vector<std::vector<double>> dens(hosts.size());
  for (std::size_t s = 0; s < hosts.size(); ++s) {
    dens[s].assign(width * hosts[s].size(), 0.0);
    pot.densityStage(hosts[s].list(), dens[s].data());
  }
  for (std::size_t s = 0; s < hosts.size(); ++s) {
    if (!hosts[s].newton) {
      continue;
    }
    for (std::size_t a = hosts[s].nOwned; a < hosts[s].size(); ++a) {
      const std::size_t j = hosts[s].source[a];
      for (std::size_t c = 0; c < width; ++c) {
        dens[ownerHost[j]][width * ownerIndex[j] + c] += dens[s][width * a + c];
      }
    }
  }
  for (std::size_t s = 0; s < hosts.size(); ++s) {
    for (std::size_t a = hosts[s].nOwned; a < hosts[s].size(); ++a) {
      const std::size_t j = hosts[s].source[a];
      for (std::size_t c = 0; c < width; ++c) {
        dens[s][width * a + c] = dens[ownerHost[j]][width * ownerIndex[j] + c];
      }
    }
  }

  Result r;
  r.forces.assign(3 * n, 0.0);
  r.atomEnergy.assign(n, 0.0);
  r.atomVirial.assign(6 * n, 0.0);
  for (std::size_t s = 0; s < hosts.size(); ++s) {
    const Host &host = hosts[s];
    const std::size_t m = host.size();
    std::vector<double> forces(3 * m, 0.0);
    std::vector<double> energy(m, 0.0);
    std::vector<double> virial(6 * m, 0.0);
    rgpot::StageTally tally;
    tally.atoms = {energy.data(), virial.data()};
    pot.forceStage(host.list(), dens[s].data(), forces.data(), &tally);

    std::vector<double> bare(3 * m, 0.0);
    pot.forceStage(host.list(), dens[s].data(), bare.data(), nullptr);
    CHECK(max_gap(bare, forces) <= 1e-14 * max_abs(forces));

    r.energy += tally.energy;
    for (int c = 0; c < 6; ++c) {
      r.virial[c] += tally.virial[c];
    }
    const std::size_t kept = host.newton ? m : host.nOwned;
    for (std::size_t a = 0; a < m; ++a) {
      if (a >= kept) {
        for (int d = 0; d < 3; ++d) {
          REQUIRE(forces[3 * a + static_cast<std::size_t>(d)] == 0.0);
        }
        REQUIRE(energy[a] == 0.0);
        continue;
      }
      const std::size_t j = host.source[a];
      for (std::size_t d = 0; d < 3; ++d) {
        r.forces[3 * j + d] += forces[3 * a + d];
      }
      r.atomEnergy[j] += energy[a];
      for (std::size_t c = 0; c < 6; ++c) {
        r.atomVirial[6 * j + c] += virial[6 * a + c];
      }
    }
  }
  return r;
}

template <typename P> void check_staged(const P &pot, const Box &b, int slabs, bool newton) {
  static_assert(rgpot::StagedPotential<P>);
  const rgpot::Locality range = pot.locality();
  REQUIRE(range.finite());
  const Whole whole = evaluate(pot, b);
  std::vector<Host> hosts = split(b, slabs, range.cutoff, newton);
  std::size_t copies = 0;
  for (const Host &host : hosts) {
    copies += host.size() - host.nOwned;
  }
  REQUIRE(copies > 0);
  REQUIRE(hosts.back().size() == 0);
  const Result got = run(pot, b, hosts);

  CHECK(max_gap(got.forces, whole.forces) < 1e-12 * max_abs(whole.forces));
  CHECK(max_gap(got.atomEnergy, whole.atomEnergy) < 1e-12 * max_abs(whole.atomEnergy));
  CHECK(max_gap(got.atomVirial, whole.atomVirial) < 1e-12 * max_abs(whole.atomVirial));
  CHECK(got.energy == Catch::Approx(whole.energy).epsilon(1e-12));
  std::vector<double> virial(6, 0.0);
  for (std::size_t a = 0; a < b.z.size(); ++a) {
    for (std::size_t c = 0; c < 6; ++c) {
      virial[c] += whole.atomVirial[6 * a + c];
    }
  }
  const double scale = max_abs(virial);
  for (std::size_t c = 0; c < 6; ++c) {
    CHECK(std::abs(got.virial[c] - virial[c]) < 1e-12 * scale);
  }
}

rgpot::SuttonChenPot sutton_chen() {
  rgpot::SuttonChenConfig c = rgpot::sutton_chen_platinum();
  c.cutoff = 7.0;
  c.periodic = true;
  c.stress = true;
  return rgpot::SuttonChenPot(c);
}

#ifdef RGPOT_HAS_EAM_TABLE
rgpot::EAMTablePot cu_ni() {
  rgpot::EAMTableConfig c;
  c.path = "CppCore/tests/data/eam_table/CuNi.eam.alloy";
  c.format = rgpot::EAMTableFormat::Setfl;
  c.stress = true;
  return rgpot::EAMTablePot(c);
}
#endif

} // namespace

TEST_CASE("Potentials with stages declare what they exchange", "[Staged]") {
  STATIC_REQUIRE(rgpot::StagedPotential<rgpot::LJPot>);
  STATIC_REQUIRE(rgpot::StagedPotential<rgpot::MorsePot>);
  STATIC_REQUIRE(rgpot::StagedPotential<rgpot::SuttonChenPot>);
  struct Local {
    [[nodiscard]] rgpot::Locality locality() const noexcept { return {.cutoff = 1.0}; }
  };
  STATIC_REQUIRE(rgpot::LocalPotential<Local>);
  STATIC_REQUIRE_FALSE(rgpot::StagedPotential<Local>);
  CHECK(rgpot::LJPot({.u0 = 0.0104, .cutoff = 8.5, .psi = 3.40}).locality().densities == 0);
  CHECK(rgpot::MorsePot({.cutoff = 7.0}).locality().densities == 0);
  CHECK(sutton_chen().locality().densities == 1);
#ifdef RGPOT_HAS_EAM_TABLE
  STATIC_REQUIRE(rgpot::StagedPotential<rgpot::EAMTablePot>);
  CHECK(cu_ni().locality().densities == 1);
#endif
}

TEST_CASE("Owners of staged hosts match the periodic box", "[Staged]") {
  const bool triclinic = GENERATE(false, true);
  const bool newton = GENERATE(false, true);
  const int slabs = GENERATE(1, 2, 3);
  CAPTURE(triclinic, newton, slabs);
  SECTION("LJ") {
    check_staged(rgpot::LJPot({.u0 = 0.0104, .cutoff = 8.5, .psi = 3.40}),
                 fcc(5.26, 4, triclinic, 18, 18), slabs, newton);
  }
  SECTION("Morse") {
    check_staged(rgpot::MorsePot({.cutoff = 7.0}), fcc(3.92, 4, triclinic, 78, 78), slabs,
                 newton);
  }
  SECTION("SuttonChen") {
    check_staged(sutton_chen(), fcc(3.92, 4, triclinic, 78, 78), slabs, newton);
  }
#ifdef RGPOT_HAS_EAM_TABLE
  SECTION("EAMTable") { check_staged(cu_ni(), fcc(3.6, 4, triclinic, 29, 28), slabs, newton); }
#endif
}

TEST_CASE("A pair potential's density stage writes nothing", "[Staged]") {
  const Box b = fcc(5.26, 4, false, 18, 18);
  const rgpot::LJPot lj({.u0 = 0.0104, .cutoff = 8.5, .psi = 3.40});
  std::vector<Host> hosts = split(b, 2, 8.5, true);
  std::vector<double> dens(hosts[0].size(), 0.0);
  lj.densityStage(hosts[0].list(), dens.data());
  CHECK(max_abs(dens) == 0.0);
}

TEST_CASE("Many-body forces need the exchanged densities", "[Staged]") {
  // Copies that keep only the densities summed on their own host lack the
  // neighbours beyond it, so the forces of the owners near them are off.
  const Box b = fcc(3.92, 4, false, 78, 78);
  const rgpot::SuttonChenPot sc = sutton_chen();
  const Whole whole = evaluate(sc, b);
  std::vector<Host> hosts = split(b, 2, 7.0, false);
  const Host &host = hosts[0];
  std::vector<double> dens(host.size(), 0.0);
  sc.densityStage(host.list(), dens.data());
  std::vector<double> forces(3 * host.size(), 0.0);
  sc.forceStage(host.list(), dens.data(), forces.data(), nullptr);
  double gap = 0.0;
  for (std::size_t a = 0; a < host.nOwned; ++a) {
    for (std::size_t d = 0; d < 3; ++d) {
      gap = std::max(gap, std::abs(forces[3 * a + d] - whole.forces[3 * host.source[a] + d]));
    }
  }
  CHECK(gap > 1e-3 * max_abs(whole.forces));
}
