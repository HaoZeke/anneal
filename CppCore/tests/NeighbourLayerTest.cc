// MIT License
// Copyright 2023--present rgpot developers

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/nlist/Neighbours.hpp"

namespace {

void setThreads(const char *value) {
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", value == nullptr ? "" : value);
#else
  if (value == nullptr || value[0] == '\0') {
    unsetenv("OMP_NUM_THREADS");
  } else {
    setenv("OMP_NUM_THREADS", value, 1);
  }
#endif
}

bool close(double a, double b) {
  const double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
  return std::abs(a - b) <= 1e-9 * scale;
}

std::uint64_t bits(double v) {
  std::uint64_t out = 0;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}

std::vector<double> fcc(int n, double a) {
  const double b[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
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

std::vector<double> diamond(int n, double a) {
  const double b[8][3] = {{0, 0, 0},         {0.5, 0.5, 0},     {0.5, 0, 0.5},
                          {0, 0.5, 0.5},     {0.25, 0.25, 0.25},{0.75, 0.75, 0.25},
                          {0.75, 0.25, 0.75},{0.25, 0.75, 0.75}};
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

template <typename Pot>
void checkModes(Pot &stored, Pot &stream, const std::vector<double> &pos,
                const std::vector<int> &z, double length) {
  rgpot::types::AtomMatrix xyz(z.size(), 3);
  for (std::size_t i = 0; i < z.size(); ++i) {
    xyz(i, 0) = pos[3 * i];
    xyz(i, 1) = pos[3 * i + 1];
    xyz(i, 2) = pos[3 * i + 2];
  }
  const std::array<std::array<double, 3>, 3> box{{
      {length, 0, 0},
      {0, length, 0},
      {0, 0, length},
  }};
  const auto a = stored(xyz, z, box);
  const auto b = stream(xyz, z, box);
  const auto c = stream(xyz, z, box);
  REQUIRE(close(std::get<0>(a), std::get<0>(b)));
  REQUIRE(bits(std::get<0>(b)) == bits(std::get<0>(c)));
  const auto &fa = std::get<1>(a);
  const auto &fb = std::get<1>(b);
  const auto &fc = std::get<1>(c);
  for (std::size_t i = 0; i < z.size() * 3; ++i) {
    REQUIRE(close(fa.data()[i], fb.data()[i]));
    REQUIRE(bits(fb.data()[i]) == bits(fc.data()[i]));
  }
}

} // namespace

TEST_CASE("neighbour modes agree for the classical potentials", "[nlist]") {
  setThreads("");
  const auto metal = fcc(3, 4.05);
  const std::vector<int> al(metal.size() / 3, 13);
  rgpot::EAMPot eam_s;
  rgpot::EAMPot eam_w;
  eam_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  eam_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkModes(eam_s, eam_w, metal, al, 3 * 4.05);

  rgpot::LJConfig ljcfg;
  ljcfg.cutoff = 2.5;
  ljcfg.psi = 1.0;
  rgpot::LJPot lj_s(ljcfg);
  rgpot::LJPot lj_w(ljcfg);
  lj_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  lj_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  const double a_lj = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0);
  const auto ljpos = fcc(3, a_lj);
  const std::vector<int> ar(ljpos.size() / 3, 18);
  checkModes(lj_s, lj_w, ljpos, ar, 3 * a_lj);

  rgpot::SuttonChenConfig sc;
  sc.cutoff = 3.0;
  sc.periodic = true;
  rgpot::SuttonChenPot sc_s(sc);
  rgpot::SuttonChenPot sc_w(sc);
  sc_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  sc_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  const auto scpos = fcc(3, std::sqrt(2.0));
  const std::vector<int> pt(scpos.size() / 3, 78);
  checkModes(sc_s, sc_w, scpos, pt, 3 * std::sqrt(2.0));

  const auto si = diamond(2, 5.431);
  const std::vector<int> zsi(si.size() / 3, 14);
  rgpot::StillingerWeberSi sw_s;
  rgpot::StillingerWeberSi sw_w;
  sw_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  sw_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkModes(sw_s, sw_w, si, zsi, 2 * 5.431);

  rgpot::TersoffSiC te_s;
  rgpot::TersoffSiC te_w;
  te_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  te_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkModes(te_s, te_w, si, zsi, 2 * 5.431);
}

template <typename Pot>
void checkThreads(Pot &&serial_pot, Pot &&parallel_pot,
                  const std::vector<double> &pos, const std::vector<int> &z,
                  double length) {
  rgpot::types::AtomMatrix xyz(z.size(), 3);
  for (std::size_t i = 0; i < z.size(); ++i) {
    xyz(i, 0) = pos[3 * i];
    xyz(i, 1) = pos[3 * i + 1];
    xyz(i, 2) = pos[3 * i + 2];
  }
  const std::array<std::array<double, 3>, 3> box{{
      {length, 0, 0},
      {0, length, 0},
      {0, 0, length},
  }};
  setThreads("");
  const auto serial = serial_pot(xyz, z, box);
  setThreads("2");
  const auto a = parallel_pot(xyz, z, box);
  const auto b = parallel_pot(xyz, z, box);
  REQUIRE(bits(std::get<0>(a)) == bits(std::get<0>(b)));
  REQUIRE(close(std::get<0>(serial), std::get<0>(a)));
  const auto &fs = std::get<1>(serial);
  const auto &fa = std::get<1>(a);
  const auto &fb = std::get<1>(b);
  for (std::size_t i = 0; i < z.size() * 3; ++i) {
    REQUIRE(close(fs.data()[i], fa.data()[i]));
    REQUIRE(bits(fa.data()[i]) == bits(fb.data()[i]));
  }
  setThreads("");
}

TEST_CASE("threaded classical forces repeat and stay near one thread",
          "[nlist]") {
  const auto metal = fcc(3, 4.05);
  const std::vector<int> al(metal.size() / 3, 13);
  rgpot::EAMPot eam_one;
  rgpot::EAMPot eam_two;
  eam_one.setNeighbourMode(rgpot::nlist::Mode::Stored);
  eam_two.setNeighbourMode(rgpot::nlist::Mode::Stored);
  checkThreads(std::move(eam_one), std::move(eam_two), metal, al, 3 * 4.05);
  rgpot::EAMPot eam_one_s;
  rgpot::EAMPot eam_two_s;
  eam_one_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  eam_two_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkThreads(std::move(eam_one_s), std::move(eam_two_s), metal, al, 3 * 4.05);

  rgpot::LJConfig ljcfg;
  ljcfg.cutoff = 2.5;
  ljcfg.psi = 1.0;
  const double a_lj = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0);
  const auto ljpos = fcc(3, a_lj);
  const std::vector<int> ar(ljpos.size() / 3, 18);
  rgpot::LJPot lj_one(ljcfg);
  rgpot::LJPot lj_two(ljcfg);
  lj_one.setNeighbourMode(rgpot::nlist::Mode::Stored);
  lj_two.setNeighbourMode(rgpot::nlist::Mode::Stored);
  checkThreads(std::move(lj_one), std::move(lj_two), ljpos, ar, 3 * a_lj);
  rgpot::LJPot lj_one_s(ljcfg);
  rgpot::LJPot lj_two_s(ljcfg);
  lj_one_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  lj_two_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkThreads(std::move(lj_one_s), std::move(lj_two_s), ljpos, ar, 3 * a_lj);

  rgpot::SuttonChenConfig sc;
  sc.cutoff = 3.0;
  sc.periodic = true;
  const auto scpos = fcc(3, std::sqrt(2.0));
  const std::vector<int> pt(scpos.size() / 3, 78);
  rgpot::SuttonChenPot sc_one(sc);
  rgpot::SuttonChenPot sc_two(sc);
  sc_one.setNeighbourMode(rgpot::nlist::Mode::Stored);
  sc_two.setNeighbourMode(rgpot::nlist::Mode::Stored);
  checkThreads(std::move(sc_one), std::move(sc_two), scpos, pt,
               3 * std::sqrt(2.0));
  rgpot::SuttonChenPot sc_one_s(sc);
  rgpot::SuttonChenPot sc_two_s(sc);
  sc_one_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  sc_two_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkThreads(std::move(sc_one_s), std::move(sc_two_s), scpos, pt,
               3 * std::sqrt(2.0));

  const auto si = diamond(2, 5.431);
  const std::vector<int> zsi(si.size() / 3, 14);
  rgpot::StillingerWeberSi sw_one;
  rgpot::StillingerWeberSi sw_two;
  sw_one.setNeighbourMode(rgpot::nlist::Mode::Stored);
  sw_two.setNeighbourMode(rgpot::nlist::Mode::Stored);
  checkThreads(std::move(sw_one), std::move(sw_two), si, zsi, 2 * 5.431);
  rgpot::StillingerWeberSi sw_one_s;
  rgpot::StillingerWeberSi sw_two_s;
  sw_one_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  sw_two_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkThreads(std::move(sw_one_s), std::move(sw_two_s), si, zsi, 2 * 5.431);

  rgpot::TersoffSiC te_one;
  rgpot::TersoffSiC te_two;
  te_one.setNeighbourMode(rgpot::nlist::Mode::Stored);
  te_two.setNeighbourMode(rgpot::nlist::Mode::Stored);
  checkThreads(std::move(te_one), std::move(te_two), si, zsi, 2 * 5.431);
  rgpot::TersoffSiC te_one_s;
  rgpot::TersoffSiC te_two_s;
  te_one_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  te_two_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkThreads(std::move(te_one_s), std::move(te_two_s), si, zsi, 2 * 5.431);
}

namespace {

double roundVisit(double t) {
  if (!std::isfinite(t)) {
    return 0.0;
  }
  const double shifted = t + std::copysign(0.5, t);
  if (!std::isfinite(shifted) || shifted >= 9223372036854775808.0 ||
      shifted <= -9223372036854775808.0) {
    return 0.0;
  }
  return static_cast<double>(static_cast<long long>(shifted));
}

void foldDelta(double &dx, double &dy, double &dz, double lx, double ly, double lz, bool periodic) {
  if (!periodic) {
    return;
  }
  dx -= lx * roundVisit(dx / lx);
  dy -= ly * roundVisit(dy / ly);
  dz -= lz * roundVisit(dz / lz);
}

using PairMap = std::map<std::pair<int, int>, double>;

PairMap directPairs(const std::vector<double> &pos, double lx, double ly, double lz, double cutoff,
                    bool periodic) {
  const double cut2 = cutoff * cutoff;
  const int n = static_cast<int>(pos.size() / 3);
  PairMap out;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double dx = pos[3 * i] - pos[3 * j];
      double dy = pos[3 * i + 1] - pos[3 * j + 1];
      double dz = pos[3 * i + 2] - pos[3 * j + 2];
      foldDelta(dx, dy, dz, lx, ly, lz, periodic);
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 <= cut2) {
        out[{i, j}] = r2;
      }
    }
  }
  return out;
}

std::vector<double> fccBlock(int nx, int ny, int nz, double a) {
  const double b[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  std::vector<double> pos;
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

PairMap walkedPairs(rgpot::nlist::Neighbours &cells, std::size_t n, double cutoff) {
  const double cut2 = cutoff * cutoff;
  PairMap out;
  cells.forHalf(0, static_cast<std::int64_t>(n),
                [&](std::int32_t i, std::int32_t j, double, double, double, double r2) {
                  if (r2 <= cut2) {
                    const int lo = std::min(i, j);
                    const int hi = std::max(i, j);
                    out[{lo, hi}] = r2;
                  }
                });
  return out;
}

void requireSamePairs(const PairMap &got, const PairMap &ref) {
  REQUIRE(got.size() == ref.size());
  for (const auto &item : ref) {
    const auto it = got.find(item.first);
    REQUIRE(it != got.end());
    const double scale = std::max(1.0, std::abs(item.second));
    REQUIRE(std::abs(it->second - item.second) <= 1e-12 * scale);
  }
}

void loadCells(rgpot::nlist::Neighbours &cells, const std::vector<double> &pos, double lx,
               double ly, double lz, double cutoff, bool periodic, rgpot::nlist::Mode mode) {
  const double box[9] = {lx, 0, 0, 0, ly, 0, 0, 0, lz};
  rgpot::nlist::Spec spec;
  spec.cutoff = cutoff;
  spec.skin = 0;
  spec.periodic = {periodic, periodic, periodic};
  spec.image = rgpot::nlist::Image::Round;
  spec.mode = mode;
  cells.update(pos.data(), pos.size() / 3, box, spec);
}

// fcc(n, a) mapped into `cell`, which should be about n * a wide, with a
// jitter that keeps pairs off bin and image boundaries.
std::vector<double> shearedFcc(int n, double a, const double cell[9]) {
  const double side = n * a;
  const auto cube = fcc(n, a);
  const std::size_t count = cube.size() / 3;
  std::vector<double> pos(cube.size());
  for (std::size_t i = 0; i < count; ++i) {
    double f[3];
    for (std::size_t k = 0; k < 3; ++k) {
      const auto slot = static_cast<double>(3 * i + k);
      f[k] = cube[3 * i + k] / side + 0.002 * std::sin(1.3 * slot + 0.7);
    }
    for (int c = 0; c < 3; ++c) {
      pos[3 * i + c] = f[0] * cell[c] + f[1] * cell[3 + c] + f[2] * cell[6 + c];
    }
  }
  return pos;
}

} // namespace

TEST_CASE("a short cutoff stream matches the direct pair sum", "[nlist][stream][direct]") {
  setThreads("1");
  const double a = 4.05;
  const double cutoff = 2.5;
  auto pos = fccBlock(10, 4, 4, a);
  const double lx = 10 * a;
  const double ly = 4 * a;
  const double lz = 4 * a;
  const auto none = directPairs(pos, lx, ly, lz, cutoff, false);
  REQUIRE(none.empty());

  rgpot::nlist::Neighbours streamed;
  loadCells(streamed, pos, lx, ly, lz, cutoff, false, rgpot::nlist::Mode::Stream);
  REQUIRE(streamed.streaming());
  REQUIRE(streamed.slots() == 0);
  requireSamePairs(walkedPairs(streamed, pos.size() / 3, cutoff), none);

  // One pair inside the cutoff. The rest of the lattice stays outside it.
  pos[3] = 1.2;
  pos[4] = 0;
  pos[5] = 0;
  const auto close = directPairs(pos, lx, ly, lz, cutoff, false);
  REQUIRE(close.count({0, 1}) == 1);

  rgpot::nlist::Neighbours again;
  loadCells(again, pos, lx, ly, lz, cutoff, false, rgpot::nlist::Mode::Stream);
  requireSamePairs(walkedPairs(again, pos.size() / 3, cutoff), close);

  rgpot::nlist::Neighbours stored;
  loadCells(stored, pos, lx, ly, lz, cutoff, false, rgpot::nlist::Mode::Stored);
  REQUIRE(stored.mode() == rgpot::nlist::Mode::Stored);
  requireSamePairs(walkedPairs(stored, pos.size() / 3, cutoff), close);

  rgpot::nlist::Neighbours automatic;
  loadCells(automatic, pos, lx, ly, lz, cutoff, false, rgpot::nlist::Mode::Auto);
  REQUIRE(automatic.mode() == rgpot::nlist::Mode::Stored);
  REQUIRE(!automatic.streaming());
  requireSamePairs(walkedPairs(automatic, pos.size() / 3, cutoff), close);

  setThreads("4");
  std::vector<PairMap> parts(4);
  again.forCellTeam(true, 4,
                    [&](int thread, std::int32_t i, std::int32_t j, double, double, double,
                        double r2) {
                      if (r2 <= cutoff * cutoff) {
                        const int lo = std::min(i, j);
                        const int hi = std::max(i, j);
                        parts[static_cast<std::size_t>(thread)][{lo, hi}] = r2;
                      }
                    });
  PairMap merged;
  for (const auto &part : parts) {
    for (const auto &item : part) {
      REQUIRE(merged.insert(item).second);
    }
  }
  requireSamePairs(merged, close);
}

TEST_CASE("periodic and long-cutoff streams match the direct pair sum", "[nlist][stream][direct]") {
  setThreads("1");
  const double a = 4.05;
  auto pos = fccBlock(10, 4, 4, a);
  const double lx = 10 * a;
  const double ly = 4 * a;
  const double lz = 4 * a;
  // Image of atom 0 across the x boundary, inside a 2.5 A cutoff.
  pos[3] = lx - 0.4;
  pos[4] = 0;
  pos[5] = 0;
  const double cutoff = 2.5;
  const auto boundary = directPairs(pos, lx, ly, lz, cutoff, true);
  REQUIRE(boundary.count({0, 1}) == 1);
  rgpot::nlist::Neighbours cells;
  loadCells(cells, pos, lx, ly, lz, cutoff, true, rgpot::nlist::Mode::Stream);
  REQUIRE(cells.streaming());
  requireSamePairs(walkedPairs(cells, pos.size() / 3, cutoff), boundary);

  const auto wide = fccBlock(4, 4, 4, a);
  const double side = 4 * a;
  const double long_cut = 6.0;
  const auto many = directPairs(wide, side, side, side, long_cut, true);
  REQUIRE(many.size() > 100);
  rgpot::nlist::Neighbours tiles;
  loadCells(tiles, wide, side, side, side, long_cut, true, rgpot::nlist::Mode::Stream);
  requireSamePairs(walkedPairs(tiles, wide.size() / 3, long_cut), many);
}

TEST_CASE("a sheared cell wider than twice the search needs no image search",
          "[nlist][triclinic]") {
  // The kernel cell of the ASE morse-bulk check: a jittered copper cell,
  // sheared as tests/python/test_ase_dropin.py shears it and tiled 3x3x3 by
  // the calculator, 864 atoms. Every lattice plane spacing is above twice
  // the cutoff plus the MorsePot skin, so the image bounds settle all 745k
  // candidate pairs and neither the build nor the walks search over images.
  setThreads("1");
  const double a = 3.6;
  const double side = 6 * a;
  const double cell[9] = {
      side, 0, 0, 0.08 * side, side, 0, 0.04 * side, -0.06 * side, side};
  const auto pos = shearedFcc(6, a, cell);
  const std::size_t n = pos.size() / 3;
  const double cutoff = 2.7 * 2.866;
  const rgpot::nlist::MinimumImage image(cell, {true, true, true});
  PairMap direct;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      double dx = pos[3 * i] - pos[3 * j];
      double dy = pos[3 * i + 1] - pos[3 * j + 1];
      double dz = pos[3 * i + 2] - pos[3 * j + 2];
      image.fold(dx, dy, dz);
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 <= cutoff * cutoff) {
        direct[{static_cast<int>(i), static_cast<int>(j)}] = r2;
      }
    }
  }
  REQUIRE(direct.size() > 10000);
  for (const auto mode :
       {rgpot::nlist::Mode::Stored, rgpot::nlist::Mode::Stream}) {
    rgpot::nlist::Spec spec;
    spec.cutoff = cutoff;
    spec.skin = mode == rgpot::nlist::Mode::Stored ? cutoff / 5.0 : 0.0;
    spec.image = rgpot::nlist::Image::Round;
    spec.mode = mode;
    rgpot::nlist::Neighbours cells;
    const auto before = rgpot::nlist::MinimumImage::searches();
    cells.update(pos.data(), n, cell, spec);
    REQUIRE(cells.streaming() == (mode == rgpot::nlist::Mode::Stream));
    const auto walked = walkedPairs(cells, n, cutoff);
    REQUIRE(rgpot::nlist::MinimumImage::searches() == before);
    requireSamePairs(walked, direct);
  }
}
