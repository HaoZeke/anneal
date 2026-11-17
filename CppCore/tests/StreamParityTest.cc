// MIT License
// Copyright 2023--present rgpot developers
//
// Stream and stored evaluations of the classical potentials. The skin
// case fails while a potential keeps a strict pair table and reuses it:
// a pair that sat between the cutoff and the skin is missing after the
// atoms move inside the cutoff.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include <catch2/catch_all.hpp>

#include "TestFixtures.hpp"
#include "lsan_libomp.hpp"
#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace {

void setThreads(const char *value) {
  rgpot::test_fixtures::setOmpThreads(value);
}

bool relClose(double a, double b, double rel = 1e-12) {
  const double scale = std::max(std::abs(a), std::abs(b));
  return std::abs(a - b) <= rel * std::max(1.0, scale);
}

std::vector<double> fccAtoms(int nx, int ny, int nz, double a) {
  const double b[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  std::vector<double> pos;
  pos.reserve(static_cast<std::size_t>(nx * ny * nz * 4 * 3));
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        for (const auto &p : b) {
          pos.push_back((ix + p[0]) * a);
          pos.push_back((iy + p[1]) * a);
          pos.push_back((iz + p[2]) * a);
        }
      }
    }
  }
  return pos;
}

std::vector<double> diamondAtoms(int n, double a) {
  const double b[8][3] = {{0, 0, 0},          {0.5, 0.5, 0},     {0.5, 0, 0.5},
                          {0, 0.5, 0.5},      {0.25, 0.25, 0.25},{0.75, 0.75, 0.25},
                          {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};
  std::vector<double> pos;
  for (int ix = 0; ix < n; ++ix) {
    for (int iy = 0; iy < n; ++iy) {
      for (int iz = 0; iz < n; ++iz) {
        for (const auto &p : b) {
          pos.push_back((ix + p[0]) * a);
          pos.push_back((iy + p[1]) * a);
          pos.push_back((iz + p[2]) * a);
        }
      }
    }
  }
  return pos;
}

rgpot::types::AtomMatrix matrixOf(const std::vector<double> &pos) {
  rgpot::types::AtomMatrix xyz(pos.size() / 3, 3);
  for (std::size_t i = 0; i < pos.size() / 3; ++i) {
    xyz(i, 0) = pos[3 * i];
    xyz(i, 1) = pos[3 * i + 1];
    xyz(i, 2) = pos[3 * i + 2];
  }
  return xyz;
}

std::array<std::array<double, 3>, 3> ortho(double lx, double ly, double lz) {
  return {{{{lx, 0, 0}}, {{0, ly, 0}}, {{0, 0, lz}}}};
}

std::array<std::array<double, 3>, 3> triclinic(double lx, double ly, double lz) {
  return {{{{lx, 0.15 * ly, 0.05 * lz}}, {{0, ly, 0.1 * lz}}, {{0, 0, lz}}}};
}

void nudge(rgpot::types::AtomMatrix &xyz) {
  if (xyz.rows() == 0) {
    return;
  }
  xyz(0, 0) += 0.07;
  xyz(xyz.rows() / 2, 1) -= 0.04;
  xyz(xyz.rows() - 1, 2) += 0.03;
}

template <typename Pot>
void checkParity(Pot &stored, Pot &stream, const rgpot::types::AtomMatrix &xyz,
                 const std::vector<int> &z,
                 const std::array<std::array<double, 3>, 3> &box, double rel = 1e-12) {
  const auto a = stored(xyz, z, box);
  const auto b = stream(xyz, z, box);
  REQUIRE(relClose(std::get<0>(a), std::get<0>(b), rel));
  const auto &fa = std::get<1>(a);
  const auto &fb = std::get<1>(b);
  REQUIRE(fa.rows() == fb.rows());
  for (std::size_t i = 0; i < fa.rows(); ++i) {
    for (std::size_t k = 0; k < 3; ++k) {
      REQUIRE(relClose(fa(i, k), fb(i, k), rel));
    }
  }
}

template <typename Pot>
void checkForce(Pot &pot, rgpot::types::AtomMatrix xyz, const std::vector<int> &z,
                const std::array<std::array<double, 3>, 3> &box) {
  const auto got = pot(xyz, z, box);
  const std::size_t atom = std::min<std::size_t>(1, xyz.rows() - 1);
  constexpr int axis = 0;
  // Copy before the perturbed calls. A later evaluation may reuse the
  // force buffer the first result still views.
  const double analytic = -std::get<1>(got)(atom, axis);
  constexpr double h = 1e-5;
  const double base = xyz(atom, axis);
  xyz(atom, axis) = base + h;
  const double ep = std::get<0>(pot(xyz, z, box));
  xyz(atom, axis) = base - h;
  const double em = std::get<0>(pot(xyz, z, box));
  const double numeric = (ep - em) / (2.0 * h);
  const double scale = std::max(1.0, std::max(std::abs(numeric), std::abs(analytic)));
  REQUIRE(std::abs(numeric - analytic) <= 2e-4 * scale);
}

} // namespace

TEST_CASE("builder and the cell list share the half-skin reuse", "[nlist][stream]") {
  setThreads("");
  const auto pos = fccAtoms(2, 2, 2, 4.0);
  double box[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
  rgpot::nlist::Query query;
  query.cutoff = 3.0;
  query.skin = 1.0;
  query.cut = rgpot::nlist::Cut::Loose;
  rgpot::nlist::Builder builder;
  builder.update(pos.data(), pos.size() / 3, box, query);
  REQUIRE_FALSE(builder.reused());

  rgpot::nlist::Neighbours cells;
  rgpot::nlist::updateCells(cells, pos.data(), pos.size() / 3, box, 3.0, 1.0,
                            rgpot::nlist::Image::Round, rgpot::nlist::Mode::Stored,
                            {{true, true, true}});
  REQUIRE_FALSE(cells.reused());

  auto moved = pos;
  moved[0] += 0.2;
  REQUIRE(rgpot::nlist::withinSkin(pos.data(), moved.data(), pos.size() / 3, 1.0));
  builder.update(moved.data(), moved.size() / 3, box, query);
  rgpot::nlist::updateCells(cells, moved.data(), moved.size() / 3, box, 3.0, 1.0,
                            rgpot::nlist::Image::Round, rgpot::nlist::Mode::Stored,
                            {{true, true, true}});
  REQUIRE(builder.reused());
  REQUIRE(cells.reused());

  moved[0] += 2.0;
  REQUIRE_FALSE(rgpot::nlist::withinSkin(pos.data(), moved.data(), pos.size() / 3, 1.0));
  builder.update(moved.data(), moved.size() / 3, box, query);
  REQUIRE_FALSE(builder.reused());
}

TEST_CASE("stored LJ keeps a pair that crosses the cutoff inside the skin",
          "[nlist][stream][lj]") {
  setThreads("");
  // Skin is 1 A. The pair starts past the cutoff and inside the skin, then
  // moves inside the cutoff by less than half the skin. A strict table
  // reused from the first call drops the pair and the energy stays zero.
  rgpot::LJConfig cfg;
  cfg.cutoff = 3.0;
  cfg.psi = 1.0;
  cfg.u0 = 1.0;
  rgpot::LJPot pot(cfg);
  pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
  rgpot::types::AtomMatrix xyz(2, 3);
  xyz(0, 0) = 5.0;
  xyz(0, 1) = 5.0;
  xyz(0, 2) = 5.0;
  xyz(1, 0) = 8.2;
  xyz(1, 1) = 5.0;
  xyz(1, 2) = 5.0;
  const std::vector<int> z{18, 18};
  const auto box = ortho(20, 20, 20);
  const auto far = pot(xyz, z, box);
  REQUIRE(std::abs(std::get<0>(far)) < 1e-12);
  REQUIRE_FALSE(pot.listReused());

  xyz(1, 0) = 7.9;
  const auto near = pot(xyz, z, box);
  REQUIRE(pot.listReused());
  REQUIRE(std::abs(std::get<0>(near)) > 1e-6);

  rgpot::LJPot fresh(cfg);
  fresh.setNeighbourMode(rgpot::nlist::Mode::Stored);
  const auto again = fresh(xyz, z, box);
  REQUIRE(relClose(std::get<0>(near), std::get<0>(again)));
  REQUIRE(relClose(std::get<1>(near)(0, 0), std::get<1>(again)(0, 0)));
}

TEST_CASE("stored Morse keeps a pair that crosses the cutoff inside the skin",
          "[nlist][stream][morse]") {
  setThreads("");
  rgpot::MorseConfig cfg;
  cfg.cutoff = 5.0;
  rgpot::MorsePot pot(cfg);
  pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
  const double skin = rgpot::MorsePot::neighbourSkin(cfg.cutoff);
  rgpot::types::AtomMatrix xyz(2, 3);
  xyz(0, 0) = 2.0;
  xyz(0, 1) = 2.0;
  xyz(0, 2) = 2.0;
  xyz(1, 0) = 2.0 + cfg.cutoff + 0.2 * skin;
  xyz(1, 1) = 2.0;
  xyz(1, 2) = 2.0;
  const std::vector<int> z{78, 78};
  const auto box = ortho(30, 30, 30);
  const auto far = pot(xyz, z, box);
  REQUIRE(std::abs(std::get<0>(far)) < 1e-12);

  xyz(1, 0) = 2.0 + cfg.cutoff - 0.2 * skin;
  REQUIRE(std::abs(xyz(1, 0) - (2.0 + cfg.cutoff + 0.2 * skin)) < 0.5 * skin);
  const auto near = pot(xyz, z, box);
  REQUIRE(pot.listReused());
  REQUIRE(std::abs(std::get<0>(near)) > 1e-8);
  rgpot::MorsePot fresh(cfg);
  fresh.setNeighbourMode(rgpot::nlist::Mode::Stored);
  REQUIRE(relClose(std::get<0>(near), std::get<0>(fresh(xyz, z, box))));
}

TEST_CASE("stream copies no pair table", "[nlist][stream]") {
  setThreads("");
  const auto pos = fccAtoms(4, 4, 4, 4.05);
  auto xyz = matrixOf(pos);
  nudge(xyz);
  const std::vector<int> al(pos.size() / 3, 13);
  const auto box = ortho(4 * 4.05, 4 * 4.05, 4 * 4.05);

  rgpot::LJConfig ljcfg;
  ljcfg.cutoff = 2.5;
  rgpot::LJPot lj(ljcfg);
  lj.setNeighbourMode(rgpot::nlist::Mode::Stream);
  (void)lj(xyz, al, box);
  REQUIRE(lj.retainedPairs() == 0);

  rgpot::MorseConfig mcfg;
  mcfg.cutoff = 6.0;
  rgpot::MorsePot morse(mcfg);
  morse.setNeighbourMode(rgpot::nlist::Mode::Stream);
  (void)morse(xyz, al, box);
  REQUIRE(morse.retainedPairs() == 0);

  rgpot::ZBLPot zbl;
  zbl.setNeighbourMode(rgpot::nlist::Mode::Stream);
  (void)zbl(xyz, al, box);
  REQUIRE(zbl.retainedPairs() == 0);

  rgpot::EAMPot eam;
  eam.setNeighbourMode(rgpot::nlist::Mode::Stream);
  (void)eam(xyz, al, box);
  REQUIRE(eam.neighborSlots() == 0);
  REQUIRE(eam.streamPartners() == 0);
}

TEST_CASE("stream matches the stored list on a small cell", "[nlist][stream]") {
  setThreads("");
  const auto metal = fccAtoms(3, 3, 3, 4.05);
  auto xyz = matrixOf(metal);
  nudge(xyz);
  const std::vector<int> al(metal.size() / 3, 13);
  const auto box = ortho(3 * 4.05, 3 * 4.05, 3 * 4.05);

  rgpot::EAMPot eam_s;
  rgpot::EAMPot eam_w;
  eam_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  eam_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkParity(eam_s, eam_w, xyz, al, box);

  rgpot::LJConfig ljcfg;
  ljcfg.cutoff = 2.5;
  ljcfg.psi = 1.0;
  rgpot::LJPot lj_s(ljcfg);
  rgpot::LJPot lj_w(ljcfg);
  lj_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  lj_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  const double a_lj = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0);
  auto ljxyz = matrixOf(fccAtoms(3, 3, 3, a_lj));
  nudge(ljxyz);
  const std::vector<int> ar(ljxyz.rows(), 18);
  const auto ljbox = ortho(3 * a_lj, 3 * a_lj, 3 * a_lj);
  checkParity(lj_s, lj_w, ljxyz, ar, ljbox);

  rgpot::MorseConfig mcfg;
  mcfg.cutoff = 6.0;
  rgpot::MorsePot mo_s(mcfg);
  rgpot::MorsePot mo_w(mcfg);
  mo_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  mo_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkParity(mo_s, mo_w, xyz, al, box);

  rgpot::ZBLPot zb_s;
  rgpot::ZBLPot zb_w;
  zb_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  zb_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkParity(zb_s, zb_w, xyz, al, box);

  rgpot::SuttonChenConfig sc;
  sc.cutoff = 3.0;
  sc.periodic = true;
  rgpot::SuttonChenPot sc_s(sc);
  rgpot::SuttonChenPot sc_w(sc);
  sc_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  sc_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  auto scxyz = matrixOf(fccAtoms(3, 3, 3, std::sqrt(2.0)));
  nudge(scxyz);
  const std::vector<int> pt(scxyz.rows(), 78);
  const double lsc = 3 * std::sqrt(2.0);
  checkParity(sc_s, sc_w, scxyz, pt, ortho(lsc, lsc, lsc));

  auto si = matrixOf(diamondAtoms(2, 5.431));
  nudge(si);
  const std::vector<int> zsi(si.rows(), 14);
  const auto sibox = ortho(2 * 5.431, 2 * 5.431, 2 * 5.431);
  rgpot::StillingerWeberSi sw_s;
  rgpot::StillingerWeberSi sw_w;
  sw_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  sw_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkParity(sw_s, sw_w, si, zsi, sibox);
  rgpot::TersoffSiC te_s;
  rgpot::TersoffSiC te_w;
  te_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  te_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkParity(te_s, te_w, si, zsi, sibox);

  rgpot::LJClusterConfig clcfg;
  clcfg.cutoff = 2.5;
  clcfg.psi = 1.0;
  rgpot::LJClusterPot cl_s(clcfg);
  rgpot::LJClusterPot cl_w(clcfg);
  cl_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  cl_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkParity(cl_s, cl_w, ljxyz, ar, ljbox);
  REQUIRE(cl_w.retainedPairs() == 0);
}

TEST_CASE("finite differences match analytic forces inside the cutoff",
          "[nlist][stream]") {
  setThreads("");
  const auto box = ortho(30, 30, 30);
  rgpot::types::AtomMatrix pair(2, 3);
  pair(0, 0) = 10;
  pair(0, 1) = 10;
  pair(0, 2) = 10;
  pair(1, 0) = 11.3;
  pair(1, 1) = 10;
  pair(1, 2) = 10;
  const std::vector<int> ar{18, 18};
  rgpot::LJConfig ljcfg;
  ljcfg.cutoff = 4.0;
  ljcfg.psi = 1.0;
  rgpot::LJPot lj(ljcfg);
  lj.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkForce(lj, pair, ar, box);

  const std::vector<int> pt{78, 78};
  rgpot::MorsePot morse;
  morse.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkForce(morse, pair, pt, box);

  const std::vector<int> al{13, 13};
  pair(1, 0) = 11.2;
  rgpot::ZBLPot zbl;
  zbl.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkForce(zbl, pair, al, box);

  pair(1, 0) = 12.2;
  rgpot::SuttonChenConfig sc;
  sc.cutoff = 6.0;
  sc.periodic = true;
  rgpot::SuttonChenPot sutton(sc);
  sutton.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkForce(sutton, pair, pt, box);
}

TEST_CASE("stream matches the stored list on a triclinic cell", "[nlist][stream]") {
  setThreads("");
  const double a = 4.05;
  auto xyz = matrixOf(fccAtoms(3, 3, 3, a));
  nudge(xyz);
  const std::vector<int> al(xyz.rows(), 13);
  const auto box = triclinic(3 * a, 3 * a, 3 * a);

  rgpot::LJConfig ljcfg;
  ljcfg.cutoff = 3.0;
  rgpot::LJPot lj_s(ljcfg);
  rgpot::LJPot lj_w(ljcfg);
  lj_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  lj_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkParity(lj_s, lj_w, xyz, al, box);

  rgpot::MorseConfig mcfg;
  mcfg.cutoff = 6.0;
  rgpot::MorsePot mo_s(mcfg);
  rgpot::MorsePot mo_w(mcfg);
  mo_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  mo_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkParity(mo_s, mo_w, xyz, al, box);

  rgpot::SuttonChenConfig sc;
  sc.cutoff = 4.0;
  sc.periodic = true;
  rgpot::SuttonChenPot sc_s(sc);
  rgpot::SuttonChenPot sc_w(sc);
  sc_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  sc_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkParity(sc_s, sc_w, xyz, al, box);

  rgpot::EAMPot eam_s;
  rgpot::EAMPot eam_w;
  eam_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  eam_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkParity(eam_s, eam_w, xyz, al, box);
}

TEST_CASE("stream matches the stored list past 100k atoms", "[nlist][stream]") {
  setThreads("");
  const double a = 4.05;
  constexpr int n = 30;
  auto xyz = matrixOf(fccAtoms(n, n, n, a));
  REQUIRE(xyz.rows() >= 100000);
  nudge(xyz);
  const std::vector<int> al(xyz.rows(), 13);
  const auto box = ortho(n * a, n * a, n * a);

  rgpot::LJConfig ljcfg;
  ljcfg.cutoff = 2.5;
  ljcfg.psi = 1.0;
  rgpot::LJPot lj_s(ljcfg);
  rgpot::LJPot lj_w(ljcfg);
  lj_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  lj_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkParity(lj_s, lj_w, xyz, al, box);
  REQUIRE(lj_w.retainedPairs() == 0);
  REQUIRE(lj_s.retainedPairs() > 0);

  rgpot::MorseConfig mcfg;
  mcfg.cutoff = 5.0;
  rgpot::MorsePot mo_s(mcfg);
  rgpot::MorsePot mo_w(mcfg);
  mo_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  mo_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkParity(mo_s, mo_w, xyz, al, box);
  REQUIRE(mo_w.retainedPairs() == 0);

  rgpot::EAMPot eam_s;
  rgpot::EAMPot eam_w;
  eam_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  eam_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  // The embedding sums about fifty terms per atom. Two orders differ by
  // a few ulps per atom, and those add across 1e5 atoms.
  checkParity(eam_s, eam_w, xyz, al, box, 1e-9);
  REQUIRE(eam_w.streamPartners() == 0);

  rgpot::SuttonChenConfig sc;
  sc.cutoff = 3.0;
  sc.periodic = true;
  const double asc = std::sqrt(2.0);
  auto scxyz = matrixOf(fccAtoms(n, n, n, asc));
  nudge(scxyz);
  const std::vector<int> pt(scxyz.rows(), 78);
  rgpot::SuttonChenPot sc_s(sc);
  rgpot::SuttonChenPot sc_w(sc);
  sc_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  sc_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkParity(sc_s, sc_w, scxyz, pt, ortho(n * asc, n * asc, n * asc));
}

TEST_CASE("ZBL stream matches a direct pair sum on a short cutoff", "[nlist][stream][zbl]") {
  setThreads("1");
  const double a = 4.05;
  auto pos = fccAtoms(10, 4, 4, a);
  // Pull one neighbour inside the 2.5 A cutoff. The crystal spacing is longer.
  pos[3] = 1.2;
  pos[4] = 0.0;
  pos[5] = 0.0;
  auto xyz = matrixOf(pos);
  const std::vector<int> au(xyz.rows(), 79);
  const auto box = ortho(10 * a, 4 * a, 4 * a);
  const double cutoff = 2.5;
  const double cut2 = cutoff * cutoff;

  rgpot::ZBLPot direct_pot;
  double energy = 0.0;
  std::vector<double> force(static_cast<std::size_t>(xyz.rows()) * 3, 0.0);
  int pairs = 0;
  for (int i = 0; i < xyz.rows(); ++i) {
    for (int j = i + 1; j < xyz.rows(); ++j) {
      const double dx = xyz(i, 0) - xyz(j, 0);
      const double dy = xyz(i, 1) - xyz(j, 1);
      const double dz = xyz(i, 2) - xyz(j, 2);
      if (dx * dx + dy * dy + dz * dz > cut2) {
        continue;
      }
      rgpot::types::AtomMatrix pair(2, 3);
      pair(0, 0) = xyz(i, 0);
      pair(0, 1) = xyz(i, 1);
      pair(0, 2) = xyz(i, 2);
      pair(1, 0) = xyz(j, 0);
      pair(1, 1) = xyz(j, 1);
      pair(1, 2) = xyz(j, 2);
      const std::vector<int> z{au[static_cast<std::size_t>(i)], au[static_cast<std::size_t>(j)]};
      const auto one = direct_pot(pair, z, box);
      energy += std::get<0>(one);
      const auto &f = std::get<1>(one);
      for (int k = 0; k < 3; ++k) {
        force[static_cast<std::size_t>(i) * 3 + static_cast<std::size_t>(k)] += f(0, k);
        force[static_cast<std::size_t>(j) * 3 + static_cast<std::size_t>(k)] += f(1, k);
      }
      ++pairs;
    }
  }
  REQUIRE(pairs >= 1);
  REQUIRE(std::abs(energy) > 1.0);

  rgpot::ZBLPot streamed;
  streamed.setNeighbourMode(rgpot::nlist::Mode::Stream);
  const auto got = streamed(xyz, au, box);
  REQUIRE(streamed.retainedPairs() == 0);
  REQUIRE(relClose(std::get<0>(got), energy, 1e-12));
  const auto &gf = std::get<1>(got);
  for (int i = 0; i < xyz.rows(); ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE(relClose(gf(i, k), force[static_cast<std::size_t>(i) * 3 + static_cast<std::size_t>(k)],
                       1e-12));
    }
  }

  rgpot::ZBLPot stored;
  stored.setNeighbourMode(rgpot::nlist::Mode::Stored);
  checkParity(stored, streamed, xyz, au, box);
}
