// MIT License
// Copyright 2023--present rgpot developers
//
// MEAM checks. The reference runs are the Large-scale Atomic/Molecular
// Massively Parallel Simulator (LAMMPS) with pair_style meam, written by
// data/meam/make_lammps_ref.py and stored beside the potential files they
// used. The rest are properties of the model itself: finite differences of
// the energy against the forces and the stress, the Rose equation of state
// of the reference crystals, and consistency under replication of the cell.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/MEAM/MEAMPot.hpp"
#include "rgpot/MEAM/MEAMReader.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

const std::string kData = "CppCore/tests/data/meam/";

/// One LAMMPS run. Virial and stress are xx yy zz yz xz xy, the virial in
/// eV and the stress in eV/Angstrom^3. Rows are Z x y z fx fy fz.
struct LammpsRef {
  double energy = 0.0, box[9] = {}, virial[6] = {}, stress[6] = {};
  std::vector<std::string> elements;
  std::vector<double> pos, forces;
  std::vector<int> z;
};

LammpsRef load_ref(const std::string &name) {
  std::ifstream in(kData + name + ".ref");
  REQUIRE(in.good());
  LammpsRef ref;
  std::string key;
  std::size_t n = 0;
  while (in >> key) {
    if (key == "energy") {
      in >> ref.energy;
    } else if (key == "box") {
      for (double &v : ref.box)
        in >> v;
    } else if (key == "virial") {
      for (double &v : ref.virial)
        in >> v;
    } else if (key == "stress") {
      for (double &v : ref.stress)
        in >> v;
    } else if (key == "elements") {
      std::string line;
      std::getline(in, line);
      std::istringstream words(line);
      for (std::string w; words >> w;)
        ref.elements.push_back(w);
    } else if (key == "atoms" && in >> n) {
      for (std::size_t i = 0; i < n; ++i) {
        int z = 0;
        in >> z;
        ref.z.push_back(z);
        for (int k = 0; k < 6; ++k) {
          double v = 0.0;
          in >> v;
          (k < 3 ? ref.pos : ref.forces).push_back(v);
        }
      }
    } else {
      std::getline(in, key);
    }
  }
  return ref;
}

/// Row-major indices of the Voigt order xx yy zz yz xz xy.
constexpr int kVoigt[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

struct Eval {
  double energy = 0.0, stress[9] = {};
  int has_stress = 0;
  std::vector<double> forces;
};

Eval eval(const rgpot::MEAMPot &pot, const std::vector<double> &pos,
          const std::vector<int> &z, const double *box) {
  Eval e;
  e.forces.assign(pos.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = e.forces.data();
  std::fill(out.stress, out.stress + 9, 7.0);
  pot.forceImpl(rgpot::ForceInput{z.size(), pos.data(), z.data(), box}, &out);
  e.energy = out.energy;
  std::copy(out.stress, out.stress + 9, e.stress);
  e.has_stress = out.has_stress;
  return e;
}

rgpot::MEAMPot make_pot(const std::string &library, const std::string &parameters,
                        std::vector<std::string> elements, bool periodic = true,
                        bool stress = true) {
  rgpot::MEAMConfig c;
  c.library = kData + library;
  c.parameters = parameters.empty() ? "" : kData + parameters;
  c.elements = std::move(elements);
  c.periodic = periodic;
  c.stress = stress;
  return rgpot::MEAMPot(c);
}

double det3(const double *b) {
  return b[0] * (b[4] * b[8] - b[5] * b[7]) - b[1] * (b[3] * b[8] - b[5] * b[6]) +
         b[2] * (b[3] * b[7] - b[4] * b[6]);
}

void check_against(const rgpot::MEAMPot &pot, const LammpsRef &ref, double etol,
                   double ftol, double stol) {
  const Eval e = eval(pot, ref.pos, ref.z, ref.box);
  REQUIRE_THAT(e.energy, WithinAbs(ref.energy, etol));
  for (std::size_t k = 0; k < ref.forces.size(); ++k) {
    INFO("force component " << k);
    REQUIRE_THAT(e.forces[k], WithinAbs(ref.forces[k], ftol));
  }
  REQUIRE(e.has_stress == 1);
  const double volume = std::abs(det3(ref.box));
  for (int v = 0; v < 6; ++v) {
    const double got = e.stress[3 * kVoigt[v][0] + kVoigt[v][1]];
    INFO("stress component " << v);
    REQUIRE_THAT(got, WithinAbs(ref.stress[v], stol));
    REQUIRE_THAT(-got * volume, WithinAbs(ref.virial[v], stol * volume));
  }
}

/// Positions of a jittered crystal in a triclinic cell.
struct Config {
  std::vector<double> pos;
  std::vector<int> z;
  double box[9];
};

Config jittered(const std::vector<std::array<double, 3>> &basis, double a, int cells,
                const std::vector<int> &basis_z, double jitter, std::uint32_t seed,
                bool sheared) {
  Config c{};
  const double L = cells * a;
  const double box[9] = {L, 0, 0, sheared ? 0.9 : 0.0, L, 0, sheared ? 0.5 : 0.0,
                         sheared ? -0.7 : 0.0, L};
  std::copy(box, box + 9, c.box);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-jitter, jitter);
  for (int i = 0; i < cells; ++i)
    for (int j = 0; j < cells; ++j)
      for (int k = 0; k < cells; ++k)
        for (std::size_t b = 0; b < basis.size(); ++b) {
          const double f[3] = {(i + basis[b][0]) / cells, (j + basis[b][1]) / cells,
                               (k + basis[b][2]) / cells};
          for (int d = 0; d < 3; ++d)
            c.pos.push_back(f[0] * box[d] + f[1] * box[3 + d] + f[2] * box[6 + d] +
                            u(rng));
          c.z.push_back(basis_z[b]);
        }
  return c;
}

const std::vector<std::array<double, 3>> kFcc = {
    {0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
const std::vector<std::array<double, 3>> kBcc = {{0, 0, 0}, {0.5, 0.5, 0.5}};

/// Central difference gradient of the energy.
std::vector<double> numeric_forces(const rgpot::MEAMPot &pot, std::vector<double> pos,
                                   const std::vector<int> &z, const double *box,
                                   double h) {
  std::vector<double> f(pos.size());
  for (std::size_t k = 0; k < pos.size(); ++k) {
    const double x = pos[k];
    pos[k] = x + h;
    const double ep = eval(pot, pos, z, box).energy;
    pos[k] = x - h;
    const double em = eval(pot, pos, z, box).energy;
    pos[k] = x;
    f[k] = -(ep - em) / (2.0 * h);
  }
  return f;
}

} // namespace

TEST_CASE("MEAM matches LAMMPS pair_style meam", "[meam][lammps]") {
  SECTION("Ni, cubic cell") {
    check_against(make_pot("library.meam", "Ni.meam", {"Ni"}), load_ref("ni_cubic"),
                  1e-8, 1e-8, 1e-9);
  }
  SECTION("Ni, small sheared cell with images beyond half the cell") {
    check_against(make_pot("library.meam", "Ni.meam", {"Ni"}),
                  load_ref("ni_sheared_small"), 1e-8, 1e-8, 1e-9);
  }
  SECTION("Fe, no parameter file") {
    check_against(make_pot("library.meam", "", {"Fe"}), load_ref("fe_bcc"), 1e-8, 1e-8,
                  1e-9);
  }
  SECTION("Si-C, sheared zincblende with a mixed pair") {
    check_against(make_pot("library.meam", "SiC.meam", {"Si", "C"}),
                  load_ref("sic_zincblende"), 1e-8, 1e-8, 1e-9);
  }
  // The Lee-Baskes second-neighbour sets: bcc Fe and fcc Ni, a seven element
  // alloy with the B1, B2 and L12 reference pairs, and the five element
  // Al-Si-Mg-Cu-Fe set with dynamic background density.
  const std::vector<std::string> hea = {"Mo", "Co", "Ni", "V", "Fe", "Al", "Cr"};
  SECTION("2NN Fe, bcc") {
    check_against(make_pot("library_2nn.meam", "MoCoNiVFeAlCr_2nn.meam", hea),
                  load_ref("fe_2nn"), 1e-8, 1e-8, 1e-9);
  }
  SECTION("2NN Ni, sheared fcc") {
    check_against(make_pot("library_2nn.meam", "MoCoNiVFeAlCr_2nn.meam", hea),
                  load_ref("ni_2nn_sheared"), 1e-8, 1e-8, 1e-9);
  }
  SECTION("2NN seven element alloy") {
    check_against(make_pot("library_2nn.meam", "MoCoNiVFeAlCr_2nn.meam", hea),
                  load_ref("hea_2nn"), 1e-8, 1e-8, 1e-9);
  }
  SECTION("2NN Al-Si-Mg-Cu-Fe") {
    check_against(make_pot("library.meam", "AlSiMgCuFe.meam", {"AlS", "SiS", "MgS", "CuS", "FeS"}),
                  load_ref("alsimgcufe_2nn"), 1e-8, 1e-8, 1e-9);
  }
  // Five elements with every G(gamma) selector the library offers and the
  // options of the data/meam/opts_*.meam files.
  const std::vector<std::string> five = {"Ni", "AlS", "Cu", "Fe", "WL"};
  SECTION("alloy, linear embedding, dynamic background and mixed reference pairs") {
    check_against(make_pot("library.meam", "opts_a.meam", five), load_ref("alloy_opts_a"),
                  1e-7, 1e-7, 1e-8);
  }
  SECTION("alloy, ialloy 1 and the Rose form 2") {
    check_against(make_pot("library.meam", "opts_b.meam", five), load_ref("alloy_opts_b"),
                  1e-7, 1e-7, 1e-8);
  }
  SECTION("alloy, ialloy 2 and the mixed reference t") {
    check_against(make_pot("library.meam", "opts_c.meam", five), load_ref("alloy_opts_c"),
                  1e-7, 1e-7, 1e-8);
  }
}

TEST_CASE("MEAM forces are the gradient of the energy", "[meam][fd]") {
  SECTION("Ni, sheared cell") {
    const rgpot::MEAMPot pot = make_pot("library.meam", "Ni.meam", {"Ni"});
    const Config c = jittered(kFcc, 3.52, 2, {28, 28, 28, 28}, 0.15, 7, true);
    const Eval e = eval(pot, c.pos, c.z, c.box);
    const std::vector<double> f = numeric_forces(pot, c.pos, c.z, c.box, 1e-5);
    for (std::size_t k = 0; k < f.size(); ++k) {
      REQUIRE_THAT(e.forces[k], WithinAbs(f[k], 2e-7 + 1e-9 * std::abs(f[k])));
    }
  }
  SECTION("Si-C, sheared cell") {
    const rgpot::MEAMPot pot = make_pot("library.meam", "SiC.meam", {"Si", "C"});
    std::vector<std::array<double, 3>> zb = kFcc;
    for (std::array<double, 3> p : kFcc) {
      zb.push_back({p[0] + 0.25, p[1] + 0.25, p[2] + 0.25});
    }
    const Config c =
        jittered(zb, 4.36, 2, {14, 14, 14, 14, 6, 6, 6, 6}, 0.05, 8, true);
    const Eval e = eval(pot, c.pos, c.z, c.box);
    const std::vector<double> f = numeric_forces(pot, c.pos, c.z, c.box, 1e-5);
    for (std::size_t k = 0; k < f.size(); ++k) {
      REQUIRE_THAT(e.forces[k], WithinAbs(f[k], 2e-7 + 1e-9 * std::abs(f[k])));
    }
  }
  SECTION("Fe, bcc") {
    const rgpot::MEAMPot pot = make_pot("library.meam", "", {"Fe"});
    const Config c = jittered(kBcc, 2.87, 3, std::vector<int>(2, 26), 0.1, 9, false);
    const Eval e = eval(pot, c.pos, c.z, c.box);
    const std::vector<double> f = numeric_forces(pot, c.pos, c.z, c.box, 1e-5);
    for (std::size_t k = 0; k < f.size(); ++k) {
      REQUIRE_THAT(e.forces[k], WithinAbs(f[k], 2e-7 + 1e-9 * std::abs(f[k])));
    }
  }
}

TEST_CASE("MEAM forces of the parameter sets are the gradient of the energy",
          "[meam][fd]") {
  struct Set {
    const char *library, *parameters, *ref;
    std::vector<std::string> elements;
  };
  const std::vector<std::string> five = {"Ni", "AlS", "Cu", "Fe", "WL"};
  const std::vector<std::string> hea = {"Mo", "Co", "Ni", "V", "Fe", "Al", "Cr"};
  const Set sets[] = {
      {"library.meam", "opts_a.meam", "alloy_opts_a", five},
      {"library.meam", "opts_b.meam", "alloy_opts_b", five},
      {"library.meam", "opts_c.meam", "alloy_opts_c", five},
      {"library_2nn.meam", "MoCoNiVFeAlCr_2nn.meam", "hea_2nn", hea},
      {"library.meam", "AlSiMgCuFe.meam", "alsimgcufe_2nn", {"AlS", "SiS", "MgS", "CuS", "FeS"}},
  };
  for (const Set &set : sets) {
    DYNAMIC_SECTION(set.ref) {
      const rgpot::MEAMPot pot = make_pot(set.library, set.parameters, set.elements);
      const LammpsRef ref = load_ref(set.ref);
      const Eval e = eval(pot, ref.pos, ref.z, ref.box);
      // The first four atoms are enough: every term of the energy reaches them.
      std::vector<double> pos = ref.pos;
      for (std::size_t k = 0; k < 12; ++k) {
        const double h = 2e-6;
        const double x = pos[k];
        pos[k] = x + h;
        const double ep = eval(pot, pos, ref.z, ref.box).energy;
        pos[k] = x - h;
        const double em = eval(pot, pos, ref.z, ref.box).energy;
        pos[k] = x;
        const double f = -(ep - em) / (2.0 * h);
        REQUIRE_THAT(e.forces[k], WithinAbs(f, 1e-6 + 2e-7 * std::abs(f)));
      }
    }
  }
}

TEST_CASE("MEAM stress is the strain derivative of the energy", "[meam][fd]") {
  const rgpot::MEAMPot pot = make_pot("library.meam", "Ni.meam", {"Ni"});
  const Config c = jittered(kFcc, 3.52, 2, {28, 28, 28, 28}, 0.1, 11, true);
  const Eval e = eval(pot, c.pos, c.z, c.box);
  const double volume = std::abs(det3(c.box));
  const double h = 1e-6;
  for (int v = 0; v < 6; ++v) {
    const int a = kVoigt[v][0], b = kVoigt[v][1];
    double energy[2];
    for (int side = 0; side < 2; ++side) {
      // A symmetric strain epsilon sends the row r to r (I + epsilon).
      const double s = side == 0 ? h : -h;
      double m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      m[3 * a + b] += s / (a == b ? 1.0 : 2.0);
      if (a != b) {
        m[3 * b + a] += s / 2.0;
      }
      auto strain = [&](const double *r, double *out) {
        for (int j = 0; j < 3; ++j)
          out[j] = r[0] * m[j] + r[1] * m[3 + j] + r[2] * m[6 + j];
      };
      Config d = c;
      for (std::size_t i = 0; i < c.z.size(); ++i)
        strain(&c.pos[3 * i], &d.pos[3 * i]);
      for (int r = 0; r < 3; ++r)
        strain(&c.box[3 * r], &d.box[3 * r]);
      energy[side] = eval(pot, d.pos, d.z, d.box).energy;
    }
    const double expected = (energy[0] - energy[1]) / (2.0 * h) / volume;
    INFO("stress component " << v);
    REQUIRE_THAT(e.stress[3 * a + b], WithinAbs(expected, 1e-7));
  }
}

TEST_CASE("MEAM reference crystals follow the Rose equation of state", "[meam][rose]") {
  // The pair term is built so that the energy per atom of the reference
  // structure is the Rose energy of its nearest neighbour distance. The
  // second-neighbour sets reach past the second shell, so under
  // compression a third shell leaks in and the match is looser.
  struct Case {
    const char *name;
    const char *library;
    const char *parameters;
    int z;
    const std::vector<std::array<double, 3>> *basis;
    double nn_per_a; //!< Nearest neighbour distance over the cube edge.
  };
  const std::vector<std::array<double, 3>> dia = {
      {0, 0, 0},      {0.5, 0.5, 0},  {0.5, 0, 0.5},  {0, 0.5, 0.5},
      {0.25, 0.25, 0.25}, {0.75, 0.75, 0.25}, {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};
  const Case cases[] = {{"Ni", "library.meam", "Ni.meam", 28, &kFcc, std::sqrt(0.5)},
                        {"Cu", "library.meam", "", 29, &kFcc, std::sqrt(0.5)},
                        {"Si", "library.meam", "", 14, &dia, std::sqrt(3.0) / 4.0},
                        {"Ni", "library_2nn.meam", "MoCoNiVFeAlCr_2nn.meam", 28, &kFcc,
                         std::sqrt(0.5)},
                        {"Fe", "library_2nn.meam", "MoCoNiVFeAlCr_2nn.meam", 26, &kBcc,
                         std::sqrt(3.0) / 2.0},
                        {"Al", "library_2nn.meam", "MoCoNiVFeAlCr_2nn.meam", 13, &kFcc,
                         std::sqrt(0.5)}};
  for (const Case &k : cases) {
    DYNAMIC_SECTION(k.name << " from " << k.library << " " << k.parameters) {
      // The parameter file names seven elements; the potential takes one of
      // them, so the file's indices need all seven.
      const bool second = std::string(k.library) == "library_2nn.meam";
      const std::vector<std::string> names =
          second ? std::vector<std::string>{"Mo", "Co", "Ni", "V", "Fe", "Al", "Cr"}
                 : std::vector<std::string>{k.name};
      const rgpot::MEAMPot pot = make_pot(k.library, k.parameters, names);
      const int self = static_cast<int>(
          std::find(names.begin(), names.end(), std::string(k.name)) - names.begin());
      const rgpot::MEAMModel &m = pot.model();
      const double nn0 = m.re(self);
      for (double scale : {0.94, 0.98, 1.0, 1.03, 1.08}) {
        const double a = scale * nn0 / k.nn_per_a;
        const Config c = jittered(*k.basis, a, 3, std::vector<int>(k.basis->size(), k.z),
                                  0.0, 1, false);
        const Eval e = eval(pot, c.pos, c.z, c.box);
        const double per_atom = e.energy / static_cast<double>(c.z.size());
        INFO("scale " << scale);
        REQUIRE_THAT(per_atom, WithinAbs(m.rose(scale * nn0, self, self), second ? 5e-4 : 1e-6));
      }
    }
  }
}

TEST_CASE("MEAM energy is extensive under replication of the cell", "[meam]") {
  // A cell narrower than the cutoff sums images; doubling it must give
  // twice the energy and the same forces and stress.
  const rgpot::MEAMPot pot = make_pot("library.meam", "Ni.meam", {"Ni"});
  const Config c = jittered(kFcc, 3.52, 2, {28, 28, 28, 28}, 0.1, 13, true);
  const Eval one = eval(pot, c.pos, c.z, c.box);
  Config big{};
  const int rep[3] = {2, 1, 3};
  for (int d = 0; d < 9; ++d)
    big.box[d] = c.box[d] * rep[d / 3];
  for (int i = 0; i < rep[0]; ++i)
    for (int j = 0; j < rep[1]; ++j)
      for (int k = 0; k < rep[2]; ++k)
        for (std::size_t a = 0; a < c.z.size(); ++a) {
          for (int d = 0; d < 3; ++d)
            big.pos.push_back(c.pos[3 * a + d] + i * c.box[d] + j * c.box[3 + d] +
                              k * c.box[6 + d]);
          big.z.push_back(c.z[a]);
        }
  const std::size_t nrep = static_cast<std::size_t>(rep[0] * rep[1] * rep[2]);
  const Eval many = eval(pot, big.pos, big.z, big.box);
  REQUIRE_THAT(many.energy, WithinRel(one.energy * static_cast<double>(nrep), 1e-12));
  for (std::size_t a = 0; a < big.z.size(); ++a) {
    for (int d = 0; d < 3; ++d) {
      REQUIRE_THAT(many.forces[3 * a + static_cast<std::size_t>(d)],
                   WithinAbs(one.forces[3 * (a % c.z.size()) + static_cast<std::size_t>(d)],
                             1e-10));
    }
  }
  for (int v = 0; v < 9; ++v) {
    REQUIRE_THAT(many.stress[v], WithinAbs(one.stress[v], 1e-12));
  }
}

TEST_CASE("MEAM cluster energy matches the periodic cell far from images", "[meam]") {
  const rgpot::MEAMPot open = make_pot("library.meam", "Ni.meam", {"Ni"}, false, false);
  const rgpot::MEAMPot closed = make_pot("library.meam", "Ni.meam", {"Ni"}, true, false);
  std::vector<double> pos;
  std::vector<int> z;
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> u(-0.2, 0.2);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 2; ++k) {
        pos.push_back(40.0 + 1.76 * i + u(rng));
        pos.push_back(40.0 + 1.76 * j + u(rng));
        pos.push_back(40.0 + 1.76 * k + u(rng));
        z.push_back(28);
      }
  const double box[9] = {90, 0, 0, 0, 90, 0, 0, 0, 90};
  const Eval a = eval(open, pos, z, box);
  const Eval b = eval(closed, pos, z, box);
  REQUIRE_THAT(a.energy, WithinRel(b.energy, 1e-12));
  for (std::size_t k = 0; k < a.forces.size(); ++k) {
    REQUIRE_THAT(a.forces[k], WithinAbs(b.forces[k], 1e-10));
  }
  REQUIRE(a.has_stress == 0);
}

TEST_CASE("MEAM reader", "[meam][reader]") {
  SECTION("library entries may wrap lines and carry quotes") {
    std::istringstream in(
        "# a comment\n"
        "'Ni' 'fcc' 12. 28 58.71   # trailing comment\n"
        "  4.99 2.45 2.7 3.0 1.5\n"
        "3.52 4.45 0.94 1.0 3.0 1.5\n"
        "-1.0 1.0 1\n");
    const std::vector<rgpot::MEAMElement> e = rgpot::read_meam_library(in, {});
    REQUIRE(e.size() == 1);
    REQUIRE(e[0].name == "Ni");
    REQUIRE(e[0].lattice == rgpot::MEAMLattice::FCC);
    REQUIRE(e[0].z == 28);
    REQUIRE_THAT(e[0].alat, WithinAbs(3.52, 1e-15));
    REQUIRE_THAT(e[0].t[3], WithinAbs(-1.0, 1e-15));
    REQUIRE(e[0].ibar == 1);
  }
  SECTION("element order follows the request") {
    std::ifstream in(kData + "library.meam");
    REQUIRE(in.good());
    const std::vector<rgpot::MEAMElement> e = rgpot::read_meam_library(in, {"Si", "C", "Ni"});
    REQUIRE(e.size() == 3);
    REQUIRE(e[0].name == "Si");
    REQUIRE(e[1].lattice == rgpot::MEAMLattice::DIA);
    REQUIRE(e[2].z == 28);
  }
  SECTION("a missing element is an error") {
    std::ifstream in(kData + "library.meam");
    REQUIRE_THROWS_AS(rgpot::read_meam_library(in, {"Ni", "Xx"}), std::invalid_argument);
  }
  SECTION("a short entry is an error") {
    std::istringstream in("'Ni' 'fcc' 12. 28 58.71\n");
    REQUIRE_THROWS_AS(rgpot::read_meam_library(in, {}), std::invalid_argument);
  }
  SECTION("z must match the lattice") {
    std::istringstream in("'Ni' 'fcc' 8. 28 58.71 4.99 2.45 2.7 3.0 1.5 3.52 4.45 0.94 "
                          "1.0 3.0 1.5 -1.0 1.0 1\n");
    REQUIRE_THROWS_AS(rgpot::read_meam_library(in, {}), std::invalid_argument);
  }
  SECTION("t0 must be 1 and rho0 positive") {
    std::istringstream t0("'Ni' 'fcc' 12. 28 58.71 4.99 2.45 2.7 3.0 1.5 3.52 4.45 0.94 "
                          "2.0 3.0 1.5 -1.0 1.0 1\n");
    REQUIRE_THROWS_AS(rgpot::read_meam_library(t0, {}), std::invalid_argument);
    std::istringstream rho0("'Ni' 'fcc' 12. 28 58.71 4.99 2.45 2.7 3.0 1.5 3.52 4.45 0.94 "
                            "1.0 3.0 1.5 -1.0 0.0 1\n");
    REQUIRE_THROWS_AS(rgpot::read_meam_library(rho0, {}), std::invalid_argument);
  }
  SECTION("parameter files set the pair keywords") {
    std::ifstream lib(kData + "library.meam");
    rgpot::MEAMParameters p(rgpot::read_meam_library(lib, {"Si", "C"}));
    std::ifstream in(kData + "SiC.meam");
    REQUIRE(in.good());
    rgpot::read_meam_parameters(in, p);
    const rgpot::MEAMModel m(p);
    REQUIRE(m.lattice(0, 1) == rgpot::MEAMLattice::DIA);
    REQUIRE(m.lattice(1, 0) == rgpot::MEAMLattice::DIA);
    REQUIRE_THAT(m.rc(), WithinAbs(4.0, 1e-15));
    REQUIRE_THAT(m.cmax(0, 1, 1), WithinAbs(4.0, 1e-15));
    REQUIRE_THAT(m.cmax(0, 0, 1), WithinAbs(4.0, 1e-15));
    REQUIRE_THAT(m.cmax(1, 1, 0), WithinAbs(4.0, 1e-15));
    REQUIRE_THAT(m.cmax(0, 0, 0), WithinAbs(2.8, 1e-15));
  }
  SECTION("bad parameter lines carry the line number") {
    std::ifstream lib(kData + "library.meam");
    rgpot::MEAMParameters p(rgpot::read_meam_library(lib, {"Ni"}));
    std::istringstream unknown("rc = 4.0\nbogus = 1\n");
    REQUIRE_THROWS_WITH(rgpot::read_meam_parameters(unknown, p),
                        Catch::Matchers::ContainsSubstring("line 2"));
    std::istringstream range("Ec(1,3) = 1.0\n");
    REQUIRE_THROWS_AS(rgpot::read_meam_parameters(range, p), std::invalid_argument);
    std::istringstream few("Cmin(1,1) = 1.0\n");
    REQUIRE_THROWS_AS(rgpot::read_meam_parameters(few, p), std::invalid_argument);
    std::istringstream lattice("lattce(1,1) = 'nope'\n");
    REQUIRE_THROWS_AS(rgpot::read_meam_parameters(lattice, p), std::invalid_argument);
    std::istringstream rho0("rho0(1) = -1.0\n");
    REQUIRE_THROWS_AS(rgpot::read_meam_parameters(rho0, p), std::invalid_argument);
  }
  SECTION("a missing file is an error") {
    REQUIRE_THROWS_AS(rgpot::load_meam_model(kData + "absent.meam", "", {}),
                      std::invalid_argument);
  }
  SECTION("atomic numbers come from the leading symbol") {
    REQUIRE(rgpot::meam_atomic_number("Fe") == 26);
    REQUIRE(rgpot::meam_atomic_number("FeS") == 26);
    REQUIRE(rgpot::meam_atomic_number("Co") == 27);
    REQUIRE(rgpot::meam_atomic_number("C") == 6);
    REQUIRE(rgpot::meam_atomic_number("") == 0);
  }
}

TEST_CASE("MEAMPot rejects bad input and keys its parameters", "[meam]") {
  const rgpot::MEAMPot ni = make_pot("library.meam", "Ni.meam", {"Ni"});
  const rgpot::MEAMPot fe = make_pot("library.meam", "", {"Fe"});
  REQUIRE(ni.paramsKey() != fe.paramsKey());
  REQUIRE(ni.paramsKey() == make_pot("library.meam", "Ni.meam", {"Ni"}).paramsKey());
  REQUIRE(ni.paramsKey() !=
          make_pot("library.meam", "Ni.meam", {"Ni"}, false, true).paramsKey());
  const std::vector<double> pos = {0, 0, 0, 1.7, 0, 0};
  const std::vector<int> z = {28, 28};
  const std::vector<int> bad_z = {28, 79};
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  REQUIRE_THROWS_WITH(ni.forceImpl(rgpot::ForceInput{z.size(), pos.data(), z.data(), box}, nullptr),
                      Catch::Matchers::ContainsSubstring("null output buffer"));
  rgpot::ForceOut missing{};
  missing.energy = 4.0;
  REQUIRE_THROWS_WITH(
      ni.forceImpl(rgpot::ForceInput{z.size(), pos.data(), z.data(), box},
                   &missing),
      Catch::Matchers::ContainsSubstring("null force buffer"));
  REQUIRE(missing.energy == 4.0);
  rgpot::ForceOut none{};
  REQUIRE_NOTHROW(
      ni.forceImpl(rgpot::ForceInput{0, nullptr, nullptr, nullptr}, &none));
  REQUIRE(none.energy == 0.0);
  REQUIRE_THROWS_AS(eval(ni, pos, bad_z, box), std::invalid_argument);
  REQUIRE_THROWS_AS(make_pot("library.meam", "", {"Ni", "Ni", "Xx"}), std::invalid_argument);
}
