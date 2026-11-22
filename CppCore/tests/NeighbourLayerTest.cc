// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <catch2/catch_all.hpp>

#include "lsan_libomp.hpp"
#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/nlist/Neighbours.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

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

#if defined(_OPENMP) && _OPENMP >= 200805
/// While this lives, no parallel region starts more than one thread, as
/// inside another region or under OMP_THREAD_LIMIT=1.
class OneThreadTeams {
public:
  OneThreadTeams() { omp_set_max_active_levels(0); }
  ~OneThreadTeams() { omp_set_max_active_levels(levels_); }
  OneThreadTeams(const OneThreadTeams &) = delete;
  OneThreadTeams &operator=(const OneThreadTeams &) = delete;

private:
  int levels_ = omp_get_max_active_levels();
};
#endif

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
#if defined(_OPENMP) && _OPENMP >= 200805
  // Both slices on one thread give the bits of two threads.
  {
    const OneThreadTeams one;
    const auto c = parallel_pot(xyz, z, box);
    REQUIRE(bits(std::get<0>(c)) == bits(std::get<0>(a)));
    const auto &fc = std::get<1>(c);
    for (std::size_t i = 0; i < z.size() * 3; ++i) {
      REQUIRE(bits(fc.data()[i]) == bits(fa.data()[i]));
    }
  }
#endif
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
  rgpot::LJClusterConfig clcfg;
  clcfg.cutoff = 2.5;
  checkThreads(rgpot::LJClusterPot(clcfg), rgpot::LJClusterPot(clcfg), ljpos,
               ar, 3 * a_lj);

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

TEST_CASE("a stored half walk folds only the pairs it visits",
          "[nlist][triclinic]") {
  // The cell is wider than twice the cutoff but not twice the cutoff plus
  // the skin, so folds of stored pairs near half its width search over
  // images. A pair and its reverse search alike, so a half walk that folds
  // only j > i searches half as often as the directed walk.
  setThreads("1");
  const double a = 3.6;
  const double side = 3 * a;
  const double cell[9] = {
      side, 0, 0, 0.08 * side, side, 0, 0.04 * side, -0.06 * side, side};
  const auto pos = shearedFcc(3, a, cell);
  const auto n = static_cast<std::int64_t>(pos.size() / 3);
  rgpot::nlist::Spec spec;
  spec.cutoff = 5.0;
  spec.skin = 1.0;
  spec.image = rgpot::nlist::Image::Round;
  spec.mode = rgpot::nlist::Mode::Stored;
  rgpot::nlist::Neighbours cells;
  cells.update(pos.data(), pos.size() / 3, cell, spec);
  REQUIRE(cells.mode() == rgpot::nlist::Mode::Stored);

  std::uint64_t directed = 0;
  auto start = rgpot::nlist::MinimumImage::searches();
  cells.forDirected(0, n,
                    [&](std::int32_t, std::int32_t, double, double, double,
                        double) { ++directed; });
  const auto directed_searches = rgpot::nlist::MinimumImage::searches() - start;

  std::uint64_t half = 0;
  bool ordered = true;
  start = rgpot::nlist::MinimumImage::searches();
  cells.forHalf(
      0, n,
      [&](std::int32_t i, std::int32_t j, double, double, double, double) {
        ordered = ordered && j > i;
        ++half;
      });
  const auto half_searches = rgpot::nlist::MinimumImage::searches() - start;

  REQUIRE(ordered);
  REQUIRE(directed == 2 * half);
  REQUIRE(directed_searches > 0);
  REQUIRE(2 * half_searches == directed_searches);
}

namespace {

/// Pairs within `cutoff` from one walk, sorted, with the bits of the
/// image vector the walk gave.
using PairBits = std::vector<std::array<std::uint64_t, 5>>;

PairBits sortedPairs(PairBits pairs) {
  std::sort(pairs.begin(), pairs.end());
  return pairs;
}

struct Walks {
  PairBits half, team, atom;
};

Walks walk(const rgpot::nlist::Neighbours &cells, std::size_t n, double cutoff) {
  const double cut2 = cutoff * cutoff;
  Walks out;
  auto into = [&](PairBits &dst, bool once) {
    return [&dst, cut2, once](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                              double r2) {
      if (r2 < cut2 && (!once || i < j)) {
        dst.push_back({static_cast<std::uint64_t>(i), static_cast<std::uint64_t>(j), bits(dx),
                       bits(dy), bits(dz)});
      }
    };
  };
  cells.forHalf(0, static_cast<std::int64_t>(n), into(out.half, false));
  auto team = into(out.team, false);
  cells.forCellTeam(true, 1,
                    [&](int, std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                        double r2) { team(i, j, dx, dy, dz, r2); });
  auto atom = into(out.atom, true);
  for (std::size_t i = 0; i < n; ++i) {
    cells.forAtom(static_cast<std::int32_t>(i), atom);
  }
  out.half = sortedPairs(out.half);
  out.team = sortedPairs(out.team);
  out.atom = sortedPairs(out.atom);
  return out;
}

/// Wrap every coordinate of an orthorhombic cell into [0, side).
void wrapInto(std::vector<double> &pos, double side) {
  for (double &x : pos) {
    x -= side * std::floor(x / side);
  }
}

} // namespace

TEST_CASE("a stored list survives atoms wrapped through the faces", "[nlist][skin]") {
  // Five bins per side, so the tile walk takes the image of each pair from
  // the bins of its atoms.
  const double a = 3.6;
  const double side = 9 * a;
  const double cutoff = 5.0;
  const double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  auto pos = fcc(9, a);
  for (std::size_t k = 0; k < pos.size(); ++k) {
    pos[k] += 0.05 * std::sin(1.7 * static_cast<double>(k));
  }
  wrapInto(pos, side);
  const std::size_t n = pos.size() / 3;

  rgpot::nlist::Spec spec;
  spec.cutoff = cutoff;
  spec.skin = 1.0;
  spec.image = rgpot::nlist::Image::Round;
  spec.mode = rgpot::nlist::Mode::Stored;
  rgpot::nlist::Neighbours warm;

  auto check = [&](const std::vector<double> &xyz, const double *cell, bool reused,
                   std::uint64_t builds) {
    warm.update(xyz.data(), xyz.size() / 3, cell, spec);
    REQUIRE(warm.mode() == rgpot::nlist::Mode::Stored);
    REQUIRE(warm.reused() == reused);
    REQUIRE(warm.builds() == builds);
    rgpot::nlist::Neighbours fresh;
    fresh.update(xyz.data(), xyz.size() / 3, cell, spec);
    const Walks got = walk(warm, xyz.size() / 3, cutoff);
    const Walks want = walk(fresh, xyz.size() / 3, cutoff);
    REQUIRE(!want.half.empty());
    REQUIRE(got.half == want.half);
    REQUIRE(got.team == want.team);
    REQUIRE(got.atom == want.atom);
  };

  check(pos, box, false, 1);

  // Under half the skin, inside the cell.
  for (std::size_t k = 0; k < pos.size(); ++k) {
    pos[k] += 0.1 * std::cos(0.9 * static_cast<double>(k));
  }
  wrapInto(pos, side);
  check(pos, box, true, 1);

  // A step along x that takes the atoms near both x faces through them.
  std::size_t crossed = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double step = (i % 2 == 0 ? 0.25 : -0.25);
    const double x = pos[3 * i] + step;
    crossed += (x < 0.0 || x >= side) ? 1 : 0;
    pos[3 * i] = x;
  }
  wrapInto(pos, side);
  REQUIRE(crossed > 0);
  check(pos, box, true, 1);

  // One atom past half the skin.
  pos[0] += 0.6;
  wrapInto(pos, side);
  check(pos, box, false, 2);

  // Another cell.
  const double grown = side * 1.002;
  const double big[9] = {grown, 0, 0, 0, grown, 0, 0, 0, grown};
  for (double &x : pos) {
    x *= 1.002;
  }
  check(pos, big, false, 3);

  // Another atom count.
  pos.resize(pos.size() - 3);
  check(pos, big, false, 4);
}

TEST_CASE("forCellTeam on a stored list walks the stored index across threads and Verlet reuse",
          "[nlist][stored][threads]") {
  const double a = 3.6;
  const double side = 8 * a;
  const double cutoff = 5.0;
  const double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  auto pos = fcc(8, a);
  for (std::size_t k = 0; k < pos.size(); ++k) {
    pos[k] += 0.04 * std::sin(1.9 * static_cast<double>(k));
  }
  wrapInto(pos, side);
  const std::size_t n = pos.size() / 3;

  rgpot::nlist::Spec spec;
  spec.cutoff = cutoff;
  spec.skin = 1.0;
  spec.image = rgpot::nlist::Image::Round;
  spec.mode = rgpot::nlist::Mode::Stored;
  rgpot::nlist::Neighbours stored;
  stored.update(pos.data(), n, box, spec);
  REQUIRE_FALSE(stored.streaming());

  for (int step = 0; step < 2; ++step) {
    if (step == 1) {
      for (std::size_t k = 0; k < pos.size(); ++k) {
        pos[k] += 0.12 * std::cos(1.1 * static_cast<double>(k));
      }
      stored.update(pos.data(), n, box, spec);
      REQUIRE(stored.reused());
    }
    PairBits want_half;
    stored.forHalf(0, static_cast<std::int64_t>(n),
                   [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
                     if (r2 <= cutoff * cutoff) {
                       want_half.push_back({static_cast<std::uint64_t>(i),
                                            static_cast<std::uint64_t>(j), bits(dx), bits(dy),
                                            bits(dz)});
                     }
                   });
    // forHalf walks rows in atom-index order. A stored forCellTeam gives
    // slice s only the central atoms in partition(n, s, nt) and walks those
    // rows the same way, so the slices joined in order are the half walk.
    // Sorting that join compared the pairs as a set and let a tile walk,
    // which owns cells rather than those atom runs, pass.
    REQUIRE_FALSE(want_half.empty());

    struct TeamWalk {
      int nt = 0;
      std::vector<char> owns;
      PairBits got;
    };
    std::vector<TeamWalk> walks;
    for (const int nt : {1, 2, 4}) {
      TeamWalk walk;
      walk.nt = nt;
      std::vector<PairBits> per_slice(static_cast<std::size_t>(nt));
      // One byte per slice. forEachSlice may run the slices on different
      // threads, and each slice is written by only one of them.
      walk.owns.assign(static_cast<std::size_t>(nt), 1);
      stored.forCellTeam(
          true, nt,
          [&](int slice, std::int32_t i, std::int32_t j, double dx, double dy, double dz,
              double r2) {
            const auto run =
                rgpot::nlist::partition(static_cast<std::int64_t>(n), slice, nt);
            const auto s = static_cast<std::size_t>(slice);
            if (static_cast<std::int64_t>(i) < run.begin ||
                static_cast<std::int64_t>(i) >= run.end) {
              walk.owns[s] = 0;
            }
            if (r2 <= cutoff * cutoff) {
              per_slice[s].push_back({static_cast<std::uint64_t>(i), static_cast<std::uint64_t>(j),
                                      bits(dx), bits(dy), bits(dz)});
            }
          });
      for (const auto &part : per_slice) {
        walk.got.insert(walk.got.end(), part.begin(), part.end());
      }
      walks.push_back(std::move(walk));
    }
    // Ownership first. One slice owns every atom, so a cell walk and a row
    // walk agree there, and the split is what a tile stencil changes.
    for (const TeamWalk &walk : walks) {
      for (int slice = 0; slice < walk.nt; ++slice) {
        INFO("step=" << step << " nt=" << walk.nt << " slice=" << slice);
        REQUIRE(walk.owns[static_cast<std::size_t>(slice)] != 0);
      }
    }
    for (const TeamWalk &walk : walks) {
      // Report the first row that disagrees. Catch would otherwise print
      // both walks.
      std::size_t mismatch = walk.got.size();
      const bool same_length = walk.got.size() == want_half.size();
      if (same_length) {
        for (std::size_t k = 0; k < walk.got.size(); ++k) {
          if (walk.got[k] != want_half[k]) {
            mismatch = k;
            break;
          }
        }
      }
      INFO("step=" << step << " nt=" << walk.nt << " got=" << walk.got.size()
                   << " want=" << want_half.size() << " first_mismatch=" << mismatch);
      REQUIRE(same_length);
      REQUIRE(mismatch == walk.got.size());
    }
  }
  setThreads("");
}

namespace {

/// Every visit of a walk in order, with the bits of its image vector.
using Visits = std::vector<std::array<std::uint64_t, 6>>;

auto into(Visits &dst) {
  return [&dst](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
    dst.push_back({static_cast<std::uint64_t>(i), static_cast<std::uint64_t>(j), bits(dx),
                   bits(dy), bits(dz), bits(r2)});
  };
}

Visits halfVisits(const rgpot::nlist::Neighbours &cells, std::size_t n) {
  Visits out;
  cells.forHalf(0, static_cast<std::int64_t>(n), into(out));
  return out;
}

Visits directedVisits(const rgpot::nlist::Neighbours &cells, std::size_t n) {
  Visits out;
  cells.forDirected(0, static_cast<std::int64_t>(n), into(out));
  return out;
}

/// fcc(n, a) centred on the origin, with a jitter.
std::vector<double> jitteredCluster(int n, double a) {
  auto pos = fcc(n, a);
  const double centre = 0.5 * (n - 0.5) * a;
  for (std::size_t k = 0; k < pos.size(); ++k) {
    pos[k] += 0.07 * std::sin(2.3 * static_cast<double>(k)) - centre;
  }
  return pos;
}

constexpr double kInf = std::numeric_limits<double>::infinity();

// The box a cluster potential passes when it has no cell.
constexpr double kFreeBox[9] = {1.0e6, 0, 0, 0, 1.0e6, 0, 0, 0, 1.0e6};

/// The directed walk of an unbounded search split into `nslices`, joined
/// in slice order. A slice holds the n - 1 pairs of each atom in its run,
/// so a slice past the atom count holds none.
Visits teamVisits(const rgpot::nlist::Neighbours &cells, std::size_t n, int nslices) {
  std::vector<Visits> parts(static_cast<std::size_t>(nslices));
  cells.forCellTeam(false, nslices,
                    [&](int t, std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                        double r2) {
                      into(parts[static_cast<std::size_t>(t)])(i, j, dx, dy, dz, r2);
                    });
  Visits joined;
  for (int t = 0; t < nslices; ++t) {
    const auto run = rgpot::nlist::partition(static_cast<std::int64_t>(n), t, nslices);
    const auto &part = parts[static_cast<std::size_t>(t)];
    REQUIRE(part.size() == static_cast<std::size_t>(run.end - run.begin) * (n - 1));
    joined.insert(joined.end(), part.begin(), part.end());
  }
  return joined;
}

} // namespace

TEST_CASE("an unbounded search visits every pair in the order of one bin",
          "[nlist][allpairs]") {
  setThreads("1");
  const auto pos = jitteredCluster(3, 3.9);
  const std::size_t n = pos.size() / 3;
  // A search wider than the cluster puts every atom in one bin. The stored
  // walk of that bin is the order of the unbounded walk.
  rgpot::nlist::Spec wide;
  wide.cutoff = 1.0e6;
  wide.skin = 2.0e5;
  wide.image = rgpot::nlist::Image::Round;
  wide.mode = rgpot::nlist::Mode::Stored;
  rgpot::nlist::Neighbours one_bin;
  one_bin.update(pos.data(), n, kFreeBox, wide);
  REQUIRE(one_bin.mode() == rgpot::nlist::Mode::Stored);
  REQUIRE(one_bin.slots() == n * (n - 1));
  const Visits half_ref = halfVisits(one_bin, n);
  const Visits directed_ref = directedVisits(one_bin, n);

  for (const auto mode :
       {rgpot::nlist::Mode::Auto, rgpot::nlist::Mode::Stored, rgpot::nlist::Mode::Stream}) {
    INFO("mode " << static_cast<int>(mode));
    rgpot::nlist::Spec spec = wide;
    spec.cutoff = kInf;
    spec.skin = mode == rgpot::nlist::Mode::Stream ? 0.0 : kInf;
    spec.mode = mode;
    rgpot::nlist::Neighbours all;
    all.update(pos.data(), n, kFreeBox, spec);
    REQUIRE(all.allPairs());
    REQUIRE(all.streaming());
    REQUIRE(all.mode() == rgpot::nlist::Mode::Stream);
    REQUIRE(all.slots() == 0);
    REQUIRE(all.builds() == 0);
    REQUIRE(halfVisits(all, n) == half_ref);
    const Visits directed = directedVisits(all, n);
    REQUIRE(directed == directed_ref);

    Visits by_atom;
    for (std::size_t i = 0; i < n; ++i) {
      all.forAtom(static_cast<std::int32_t>(i), into(by_atom));
    }
    REQUIRE(by_atom == directed);

    // A team splits the atoms into one run per slice. Every slice is
    // walked, also when the runtime starts fewer threads than slices.
    REQUIRE(teamVisits(all, n, 3) == directed);
#if defined(_OPENMP) && _OPENMP >= 200805
    {
      const OneThreadTeams one;
      REQUIRE(teamVisits(all, n, 3) == directed);
    }
#endif
  }

  // More slices than atoms leaves some slices without atoms.
  const auto few = jitteredCluster(1, 3.9);
  const std::size_t m = few.size() / 3;
  rgpot::nlist::Spec unbounded = wide;
  unbounded.cutoff = kInf;
  unbounded.skin = kInf;
  rgpot::nlist::Neighbours small;
  small.update(few.data(), m, kFreeBox, unbounded);
  REQUIRE(small.allPairs());
  REQUIRE(teamVisits(small, m, static_cast<int>(m) + 2) == directedVisits(small, m));

  // A finite search past the cluster still bins.
  rgpot::nlist::Neighbours finite;
  finite.update(pos.data(), n, kFreeBox, wide);
  REQUIRE(!finite.allPairs());
}

TEST_CASE("an unbounded search keeps its update until the atoms, the cell or the search change",
          "[nlist][allpairs][skin]") {
  setThreads("1");
  auto pos = jitteredCluster(2, 3.9);
  const std::size_t n = pos.size() / 3;
  rgpot::nlist::Spec spec;
  spec.cutoff = kInf;
  spec.skin = kInf;
  spec.image = rgpot::nlist::Image::Round;
  spec.mode = rgpot::nlist::Mode::Auto;
  rgpot::nlist::Neighbours all;
  all.update(pos.data(), n, kFreeBox, spec);
  REQUIRE(all.allPairs());
  REQUIRE(!all.reused());

  // No move takes a pair out of an unbounded search, however far.
  for (std::size_t k = 0; k < pos.size(); ++k) {
    pos[k] += 3.0 * std::cos(1.3 * static_cast<double>(k));
  }
  all.update(pos.data(), n, kFreeBox, spec);
  REQUIRE(all.reused());
  rgpot::nlist::Neighbours fresh;
  fresh.update(pos.data(), n, kFreeBox, spec);
  REQUIRE(halfVisits(all, n) == halfVisits(fresh, n));
  REQUIRE(directedVisits(all, n) == directedVisits(fresh, n));

  const double other_box[9] = {2.0e6, 0, 0, 0, 2.0e6, 0, 0, 0, 2.0e6};
  all.update(pos.data(), n, other_box, spec);
  REQUIRE(!all.reused());
  REQUIRE(all.allPairs());

  all.update(pos.data(), n - 1, other_box, spec);
  REQUIRE(!all.reused());
  REQUIRE(halfVisits(all, n - 1).size() == (n - 1) * (n - 2) / 2);

  rgpot::nlist::Spec bounded = spec;
  bounded.cutoff = 4.0;
  bounded.skin = 0.8;
  all.update(pos.data(), n - 1, other_box, bounded);
  REQUIRE(!all.reused());
  REQUIRE(!all.allPairs());
  REQUIRE(all.mode() == rgpot::nlist::Mode::Stored);
  REQUIRE(all.slots() < (n - 1) * (n - 2));

  all.update(pos.data(), n - 1, other_box, spec);
  REQUIRE(!all.reused());
  REQUIRE(all.allPairs());
  REQUIRE(all.slots() == 0);
}

TEST_CASE("an unbounded periodic search takes the nearest image of every pair",
          "[nlist][allpairs][direct]") {
  setThreads("1");
  const double a = 4.05;
  auto pos = fccBlock(3, 2, 2, a);
  for (std::size_t k = 0; k < pos.size(); ++k) {
    pos[k] += 0.05 * std::cos(1.1 * static_cast<double>(k));
  }
  const std::size_t n = pos.size() / 3;
  const auto direct = directPairs(pos, 3 * a, 2 * a, 2 * a, kInf, true);
  REQUIRE(direct.size() == n * (n - 1) / 2);
  rgpot::nlist::Neighbours all;
  loadCells(all, pos, 3 * a, 2 * a, 2 * a, kInf, true, rgpot::nlist::Mode::Auto);
  REQUIRE(all.allPairs());
  requireSamePairs(walkedPairs(all, n, kInf), direct);
}

TEST_CASE("a directed walk gives a lower partner the negated vector of the half walk",
          "[nlist][triclinic][allpairs]") {
  // A team adds the force of a pair at both of its rows, one thread at
  // both ends from the half walk, and the bits agree only if the second
  // row sees the negated vector. Sites without a jitter put pairs on the
  // half cell, where a sheared cell ties two images.
  setThreads("1");
  const double a = 3.6;
  const double side = 3 * a;
  const double cubic[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  const double sheared[9] = {side, 0, 0, 0.08 * side, side, 0, 0.04 * side, -0.06 * side, side};
  const auto cube = fcc(3, a);
  const std::size_t n = cube.size() / 3;
  const auto rows = static_cast<std::int64_t>(n);
  for (const double *cell : {cubic, sheared}) {
    std::vector<double> pos(cube.size());
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t c = 0; c < 3; ++c) {
        pos[3 * i + c] = (cube[3 * i] * cell[c] + cube[3 * i + 1] * cell[3 + c] +
                          cube[3 * i + 2] * cell[6 + c]) /
                         side;
      }
    }
    for (const auto image : {rgpot::nlist::Image::Round, rgpot::nlist::Image::General}) {
      for (const double cutoff : {5.0, kInf}) {
        INFO("sheared " << (cell == sheared) << " image " << static_cast<int>(image)
                        << " cutoff " << cutoff);
        rgpot::nlist::Spec spec;
        spec.cutoff = cutoff;
        spec.skin = 1.0;
        spec.image = image;
        spec.mode = rgpot::nlist::Mode::Stored;
        rgpot::nlist::Neighbours cells;
        cells.update(pos.data(), n, cell, spec);
        std::map<std::pair<std::int32_t, std::int32_t>, std::array<double, 4>> half;
        cells.forHalf(0, rows,
                      [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                          double r2) { half[{i, j}] = {dx, dy, dz, r2}; });
        std::size_t lower = 0;
        std::size_t apart = 0;
        cells.forDirected(0, rows, [&](std::int32_t i, std::int32_t j, double dx, double dy,
                                       double dz, double r2) {
          if (j > i) {
            return;
          }
          ++lower;
          const auto &h = half.at({j, i});
          if (dx != -h[0] || dy != -h[1] || dz != -h[2] || bits(r2) != bits(h[3])) {
            ++apart;
          }
        });
        REQUIRE(lower == half.size());
        REQUIRE(apart == 0);
      }
    }
  }
}

namespace {

// Magnitude at which the cofactor determinant of the tilted cell below
// overflows double, including after scaling by 1e-3 and by 1e3.
constexpr double kDetOverflowScale = 1e105;

void cellPoint(const double cell[9], double f0, double f1, double f2, double out[3]) {
  out[0] = f0 * cell[0] + f1 * cell[3] + f2 * cell[6];
  out[1] = f0 * cell[1] + f1 * cell[4] + f2 * cell[7];
  out[2] = f0 * cell[2] + f1 * cell[5] + f2 * cell[8];
}

void tiltedCell(double scale, double cell[9]) {
  const double side = 10.0 * scale;
  const double tilt = 6.0 * scale;
  cell[0] = side;
  cell[1] = 0.0;
  cell[2] = 0.0;
  cell[3] = 0.0;
  cell[4] = side;
  cell[5] = 0.0;
  cell[6] = tilt;
  cell[7] = 0.0;
  cell[8] = side;
}

// w*x and y*z round to one double, so the naive minor w*x - y*z is 0
// while the fused minor is not. The cell is skewed and ill conditioned.
void cancellingCell(double scale, double cell[9]) {
  const double w = 3560.0;
  const double x = 4432.0;
  const double y = 628.0;
  const double z = (w * x) / y;
  cell[0] = scale;
  cell[1] = 0.0;
  cell[2] = 0.0;
  cell[3] = 0.0;
  cell[4] = w * scale;
  cell[5] = y * scale;
  cell[6] = 0.0;
  cell[7] = z * scale;
  cell[8] = x * scale;
}

/// Every lattice image with index coordinates in [-4, 4] along periodic
/// axes (and 0 along non-periodic axes). The minimum image of each pair
/// must lie strictly inside that box on periodic axes.
PairMap bruteAllImages(const std::vector<double> &pos, const double cell[9],
                       double cutoff,
                       std::array<bool, 3> periodic = {true, true, true}) {
  constexpr int kRad = 4;
  const double cut2 = cutoff * cutoff;
  const int n = static_cast<int>(pos.size() / 3);
  const int ra = periodic[0] ? kRad : 0;
  const int rb = periodic[1] ? kRad : 0;
  const int rc = periodic[2] ? kRad : 0;
  PairMap out;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double best = 0.0;
      int ba = 0;
      int bb = 0;
      int bc = 0;
      bool any = false;
      for (int a = -ra; a <= ra; ++a) {
        for (int b = -rb; b <= rb; ++b) {
          for (int c = -rc; c <= rc; ++c) {
            double shift[3];
            cellPoint(cell, static_cast<double>(a), static_cast<double>(b),
                      static_cast<double>(c), shift);
            const double dx = pos[3 * i] - pos[3 * j] - shift[0];
            const double dy = pos[3 * i + 1] - pos[3 * j + 1] - shift[1];
            const double dz = pos[3 * i + 2] - pos[3 * j + 2] - shift[2];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (!any || r2 < best) {
              best = r2;
              ba = a;
              bb = b;
              bc = c;
              any = true;
            }
          }
        }
      }
      REQUIRE(any);
      REQUIRE(std::abs(ba) < kRad);
      REQUIRE(std::abs(bb) < kRad);
      REQUIRE(std::abs(bc) < kRad);
      if (best <= cut2) {
        out[{i, j}] = best;
      }
    }
  }
  return out;
}

void requireSameKeys(const PairMap &got, const PairMap &ref) {
  REQUIRE(got.size() == ref.size());
  for (const auto &item : ref) {
    REQUIRE(got.find(item.first) != got.end());
  }
}

std::vector<double> facePair(const double cell[9]) {
  std::vector<double> pos(6);
  cellPoint(cell, 0.05, 0.5, 0.02, pos.data());
  cellPoint(cell, 0.05, 0.5, 0.98, pos.data() + 3);
  return pos;
}

void requireTiltPairs(double scale) {
  double cell[9];
  tiltedCell(scale, cell);
  const double cutoff = 2.5 * scale;
  const auto pos = facePair(cell);
  const auto ref = bruteAllImages(pos, cell, cutoff);
  REQUIRE(ref.size() == 1);
  for (const auto mode : {rgpot::nlist::Mode::Stream, rgpot::nlist::Mode::Stored}) {
    rgpot::nlist::Spec spec;
    spec.cutoff = cutoff;
    spec.skin = 0.0;
    spec.periodic = {true, true, true};
    spec.image = rgpot::nlist::Image::Round;
    spec.mode = mode;
    rgpot::nlist::Neighbours cells;
    cells.update(pos.data(), pos.size() / 3, cell, spec);
    requireSameKeys(walkedPairs(cells, pos.size() / 3, cutoff), ref);
  }
}

void requireCancellingThrows(double scale) {
  double cell[9];
  cancellingCell(scale, cell);
  const auto pos = facePair(cell);
  rgpot::nlist::Spec spec;
  spec.cutoff = std::abs(scale);
  spec.skin = 0.0;
  spec.periodic = {true, true, true};
  spec.image = rgpot::nlist::Image::Round;
  spec.mode = rgpot::nlist::Mode::Stream;
  rgpot::nlist::Neighbours cells;
  REQUIRE_THROWS_WITH(cells.update(pos.data(), pos.size() / 3, cell, spec),
                      Catch::Matchers::ContainsSubstring("too ill conditioned"));
}

} // namespace

TEST_CASE("a strongly skewed triclinic cell matches brute force images",
          "[nlist][triclinic][skew]") {
  setThreads("1");
  requireTiltPairs(kDetOverflowScale);
}

TEST_CASE("a strongly skewed cell scaled by 1e-3 matches brute force images",
          "[nlist][triclinic][skew]") {
  setThreads("1");
  requireTiltPairs(kDetOverflowScale * 1e-3);
}

TEST_CASE("a strongly skewed cell scaled by 1e3 matches brute force images",
          "[nlist][triclinic][skew]") {
  setThreads("1");
  requireTiltPairs(kDetOverflowScale * 1e3);
}

TEST_CASE("naive cofactor cancellation on a skewed cell is rejected",
          "[nlist][triclinic][singular]") {
  setThreads("1");
  requireCancellingThrows(1.0);
}

TEST_CASE("naive cofactor cancellation scaled by 1e-3 is rejected",
          "[nlist][triclinic][singular]") {
  setThreads("1");
  requireCancellingThrows(1e-3);
}

TEST_CASE("naive cofactor cancellation scaled by 1e3 is rejected",
          "[nlist][triclinic][singular]") {
  setThreads("1");
  requireCancellingThrows(1e3);
}

TEST_CASE("naive cofactor cancellation scaled by 1e-110 is rejected",
          "[nlist][triclinic][singular]") {
  setThreads("1");
  requireCancellingThrows(1e-110);
}

TEST_CASE("mixed periodicity on a skewed cell keeps the 0.466 pair and matches "
          "brute force",
          "[nlist][triclinic][mixed]") {
  setThreads("1");
  const double cutoff = 1.0;

  // Exact regression case: a skewed cell with periodicity {true, true, false}
  // and a pair across the periodic b face at distance 0.04 * hypot(6, 10) =
  // 0.466476... with cutoff 1.0.
  {
    const double cell[9] = {10.0, 0.0, 0.0, 6.0, 10.0, 0.0, 0.0, 0.0, 10.0};
    const std::array<bool, 3> periodic{true, true, false};
    std::vector<double> pos(6);
    cellPoint(cell, 0.05, 0.02, 0.5, pos.data());
    cellPoint(cell, 0.05, 0.98, 0.5, pos.data() + 3);
    const auto ref = bruteAllImages(pos, cell, cutoff, periodic);
    REQUIRE(ref.size() == 1);
    const double dist = std::sqrt(ref.begin()->second);
    REQUIRE(std::abs(dist - 0.04 * std::hypot(6.0, 10.0)) < 1e-12);
    REQUIRE(std::abs(dist - 0.466476151587624) < 1e-12);
    for (const auto mode :
         {rgpot::nlist::Mode::Stream, rgpot::nlist::Mode::Stored}) {
      rgpot::nlist::Spec spec;
      spec.cutoff = cutoff;
      spec.skin = 0.0;
      spec.periodic = periodic;
      spec.image = rgpot::nlist::Image::Round;
      spec.mode = mode;
      rgpot::nlist::Neighbours cells;
      cells.update(pos.data(), pos.size() / 3, cell, spec);
      requireSamePairs(walkedPairs(cells, pos.size() / 3, cutoff), ref);
    }
  }

  // All six mixed periodicity patterns on a fully skewed triclinic cell, with
  // pairs wrapping across each periodic face (at distance ~0.466) and pairs
  // along non-periodic axes that cross fractional 0 and 1 and extend outside
  // [0, 1).
  const double cell[9] = {10.0, 0.0, 0.0, 6.0, 10.0, 0.0, 6.0, 6.0, 10.0};
  const double fracs[][3] = {
      // Face-wrapping pairs along axis 0, axis 1, and axis 2.
      {0.02, 0.25, 0.50},
      {0.98, 0.25, 0.50},
      {0.25, 0.02, 0.50},
      {0.25, 0.98, 0.50},
      {0.25, 0.50, 0.02},
      {0.25, 0.50, 0.98},
      // Non-periodic pairs straddling fractional 0 and 1, and outside [0, 1).
      {-0.02, 0.50, 0.50},
      {0.02, 0.50, 0.50},
      {0.50, -0.02, 0.50},
      {0.50, 0.02, 0.50},
      {0.50, 0.50, -0.02},
      {0.50, 0.50, 0.02},
      {1.35, -0.35, 1.40},
      {1.38, -0.35, 1.40},
      {-0.45, 1.45, -0.40},
      {-0.45, 1.48, -0.40},
      {0.40, 0.40, -0.55},
      {0.40, 0.40, -0.52},
  };
  const std::size_t n = sizeof(fracs) / sizeof(fracs[0]);
  std::vector<double> pos(3 * n);
  for (std::size_t a = 0; a < n; ++a) {
    cellPoint(cell, fracs[a][0], fracs[a][1], fracs[a][2], pos.data() + 3 * a);
  }
  const std::array<bool, 3> patterns[] = {
      {true, true, false},  {true, false, true},  {false, true, true},
      {true, false, false}, {false, true, false}, {false, false, true},
  };
  for (const auto &periodic : patterns) {
    INFO("p0=" << periodic[0] << " p1=" << periodic[1] << " p2=" << periodic[2]);
    const auto ref = bruteAllImages(pos, cell, cutoff, periodic);
    REQUIRE(!ref.empty());
    for (const auto mode :
         {rgpot::nlist::Mode::Stream, rgpot::nlist::Mode::Stored}) {
      rgpot::nlist::Spec spec;
      spec.cutoff = cutoff;
      spec.skin = 0.0;
      spec.periodic = periodic;
      spec.image = rgpot::nlist::Image::Round;
      spec.mode = mode;
      rgpot::nlist::Neighbours cells;
      cells.update(pos.data(), n, cell, spec);
      requireSamePairs(walkedPairs(cells, n, cutoff), ref);
    }
  }
}

TEST_CASE("ulp cutoff boundaries and subnormal off-diagonal offsets match brute force",
          "[nlist][triclinic][boundary]") {
  setThreads("1");
  const double cutoff = 1.0;
  const double d_below = std::nextafter(cutoff, 0.0);
  const double d_exact = cutoff;
  const double d_above =
      std::nextafter(cutoff, std::numeric_limits<double>::infinity());
  const double sub = std::numeric_limits<double>::denorm_min();

  // 1. Direct and periodic-wrapped pairs at 1-ulp below, at, and 1-ulp above
  // the cutoff in a skewed triclinic cell.
  {
    const double cell[9] = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 6.0, 0.0, 10.0};
    for (const double d : {d_below, d_exact, d_above}) {
      // Atom 0 at (0, 0.5, 5) and Atom 1 at (d, 10.5, 5) wrap across b =
      // (0, 10, 0) to exact displacement (-d, 0, 0), and Atom 2 at (0, 5, 5)
      // with Atom 3 at (d, 5, 5) has direct displacement (-d, 0, 0).
      const std::vector<double> pos = {
          0.0, 0.5, 5.0, d, 10.5, 5.0, 0.0, 5.0, 5.0, d, 5.0, 5.0,
      };
      for (const auto &periodic :
           {std::array<bool, 3>{true, true, true},
            std::array<bool, 3>{true, true, false}}) {
        const auto ref = bruteAllImages(pos, cell, cutoff, periodic);
        if (d <= cutoff) {
          REQUIRE(ref.count({0, 1}) == 1);
          REQUIRE(ref.count({2, 3}) == 1);
        } else {
          REQUIRE(ref.count({0, 1}) == 0);
          REQUIRE(ref.count({2, 3}) == 0);
        }
        for (const auto mode :
             {rgpot::nlist::Mode::Stream, rgpot::nlist::Mode::Stored}) {
          rgpot::nlist::Spec spec;
          spec.cutoff = cutoff;
          spec.skin = 0.0;
          spec.periodic = periodic;
          spec.image = rgpot::nlist::Image::Round;
          spec.mode = mode;
          rgpot::nlist::Neighbours cells;
          cells.update(pos.data(), pos.size() / 3, cell, spec);
          requireSamePairs(walkedPairs(cells, pos.size() / 3, cutoff), ref);
        }
      }
    }
  }

  // 2. Zero, subnormal, and small nonzero off-diagonal entries, with
  // subnormal offsets on both the cell and the coordinates.
  const double scale = 10.0;
  const double tiny_shear =
      4096.0 * std::numeric_limits<double>::epsilon() * scale;
  const double offsets[] = {
      0.0,
      sub,
      -sub,
      0.5 * std::numeric_limits<double>::min(),
      std::nextafter(tiny_shear, 0.0),
      tiny_shear,
      std::nextafter(tiny_shear, std::numeric_limits<double>::infinity()),
      -std::nextafter(tiny_shear, std::numeric_limits<double>::infinity()),
  };
  for (const double off : offsets) {
    const double cell[9] = {10.0, 0.0, 0.0, off, 8.0, 0.0, 0.0, -off, 8.0};
    const std::vector<double> pos = {
        0.0, sub,     -sub, d_below,    sub, -sub,      0.0,        4.0,
        4.0, d_exact, 4.0,  4.0,        0.0, 2.0,       6.0,        d_above,
        2.0, 6.0,     0.1,  0.02 * 8.0, 4.0, 0.1 + off, 0.98 * 8.0, 4.0,
    };
    const auto ref = bruteAllImages(pos, cell, cutoff);
    REQUIRE(ref.count({0, 1}) == 1);
    REQUIRE(ref.count({2, 3}) == 1);
    REQUIRE(ref.count({4, 5}) == 0);
    REQUIRE(ref.count({6, 7}) == 1);
    for (const auto mode :
         {rgpot::nlist::Mode::Stream, rgpot::nlist::Mode::Stored}) {
      rgpot::nlist::Spec spec;
      spec.cutoff = cutoff;
      spec.skin = 0.0;
      spec.periodic = {true, true, true};
      spec.image = rgpot::nlist::Image::Round;
      spec.mode = mode;
      rgpot::nlist::Neighbours cells;
      cells.update(pos.data(), pos.size() / 3, cell, spec);
      requireSamePairs(walkedPairs(cells, pos.size() / 3, cutoff), ref);
    }
  }
}

TEST_CASE("non-finite fractional coordinates throw in skewed cell binning",
          "[nlist][triclinic][nonfinite]") {
  setThreads("1");
  double cell[9];
  tiltedCell(1e-3, cell);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const double huge = std::numeric_limits<double>::max();

  // NaN coordinate, infinite coordinate, and finite coordinate whose product
  // with frac_inv (~100 for scale 1e-3) overflows double to infinity.
  const std::vector<double> bad_cases[] = {
      {nan, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, inf, 0.0, 0.0, 0.0, 0.0},
      {huge, 0.0, 0.0, 0.0, 0.0, 0.0},
  };
  for (const auto &pos : bad_cases) {
    for (const auto &periodic : {std::array<bool, 3>{true, true, true},
                                 std::array<bool, 3>{true, true, false}}) {
      for (const double cutoff : {2.5e-3, 1.0}) {
        rgpot::nlist::Spec spec;
        spec.cutoff = cutoff;
        spec.skin = 0.0;
        spec.periodic = periodic;
        spec.image = rgpot::nlist::Image::Round;
        spec.mode = rgpot::nlist::Mode::Stream;
        rgpot::nlist::Neighbours cells;
        REQUIRE_THROWS_AS(cells.update(pos.data(), pos.size() / 3, cell, spec),
                          std::invalid_argument);
        REQUIRE_THROWS_WITH(
            cells.update(pos.data(), pos.size() / 3, cell, spec),
            Catch::Matchers::ContainsSubstring("fractional coordinate"));
      }
    }
  }
}

TEST_CASE("disproportionate open rows with a small shear match brute force",
          "[nlist][triclinic][mixed]") {
  setThreads("1");
  const double cutoff = 1.0;
  const double open_a = 5.0e8;
  const double open_b = 4.0e8;
  const double shear = 3.0e-4;
  const double cross = -2.0e-4;

  // 2D periodic slabs with a huge open row (5e8 vs periodic 10.0) and a small
  // real shear (3e-4). Atom 0 and Atom 1 sit across two Cartesian bin
  // boundaries (1.9999 in bin 1 vs 3.0001 in bin 3) while their periodic
  // minimum image across the sheared row has distance ~0.9999 < cutoff.
  struct SlabCase {
    double cell[9];
    std::array<bool, 3> periodic;
    std::vector<double> pos;
  };
  const SlabCase slabs[] = {
      {
          {10.0, 0.0, 0.0, shear, 10.0, 0.0, 0.0, 0.0, open_a},
          {true, true, false},
          {
              1.9999,
              0.0001,
              0.0,
              3.0001,
              9.9999,
              0.0,
              0.05,
              5.0,
              -1.2,
              9.85,
              5.0,
              -0.8,
              5.0,
              0.1,
              2.5,
              5.0 + shear,
              9.8,
              2.7,
          },
      },
      {
          {10.0, 0.0, 0.0, 0.0, open_a, 0.0, shear, 0.0, 10.0},
          {true, false, true},
          {
              1.9999,
              0.0,
              0.0001,
              3.0001,
              0.0,
              9.9999,
              0.05,
              -1.2,
              5.0,
              9.85,
              -0.8,
              5.0,
              5.0,
              2.5,
              0.1,
              5.0 + shear,
              2.7,
              9.8,
          },
      },
      {
          {open_a, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, shear, 10.0},
          {false, true, true},
          {
              0.0,
              1.9999,
              0.0001,
              0.0,
              3.0001,
              9.9999,
              -1.2,
              0.05,
              5.0,
              -0.8,
              9.85,
              5.0,
              2.5,
              5.0,
              0.1,
              2.7,
              5.0 + shear,
              9.8,
          },
      },
  };
  for (const auto &tc : slabs) {
    const auto ref = bruteAllImages(tc.pos, tc.cell, cutoff, tc.periodic);
    REQUIRE(ref.count({0, 1}) == 1);
    REQUIRE(ref.size() >= 3);
    for (const auto mode :
         {rgpot::nlist::Mode::Stream, rgpot::nlist::Mode::Stored,
          rgpot::nlist::Mode::Auto}) {
      rgpot::nlist::Spec spec;
      spec.cutoff = cutoff;
      spec.skin = 0.0;
      spec.periodic = tc.periodic;
      spec.image = rgpot::nlist::Image::Round;
      spec.mode = mode;
      rgpot::nlist::Neighbours cells;
      cells.update(tc.pos.data(), tc.pos.size() / 3, tc.cell, spec);
      requireSamePairs(walkedPairs(cells, tc.pos.size() / 3, cutoff), ref);
    }
  }

  // 1D periodic wires with two disproportionate open rows (5e8 and 4e8) and a
  // small real shear along the open axes. Anchor atoms establish a 5.0-wide
  // span along an open axis so Atom 0 (1.9999) and Atom 1 (3.0001) land two
  // Cartesian 1.0-wide bins apart while wrapping along the tilted periodic
  // vector brings them within 0.9999 < cutoff.
  const SlabCase wires[] = {
      {
          {10.0, shear, cross, 0.0, open_a, 0.0, 0.0, 0.0, open_b},
          {true, false, false},
          {
              0.05,
              1.9999,
              0.2,
              10.05,
              3.0001,
              0.2 + cross,
              5.0,
              0.0,
              -1.0,
              5.0,
              5.0,
              2.0,
              0.10,
              2.5,
              0.5,
              9.85,
              2.5 - shear,
              0.5 - cross,
          },
      },
      {
          {open_a, 0.0, 0.0, shear, 10.0, cross, 0.0, 0.0, open_b},
          {false, true, false},
          {
              1.9999,
              0.05,
              0.2,
              3.0001,
              10.05,
              0.2 + cross,
              0.0,
              5.0,
              -1.0,
              5.0,
              5.0,
              2.0,
              2.5,
              0.10,
              0.5,
              2.5 - shear,
              9.85,
              0.5 - cross,
          },
      },
      {
          {open_a, 0.0, 0.0, 0.0, open_b, 0.0, shear, cross, 10.0},
          {false, false, true},
          {
              1.9999,
              0.2,
              0.05,
              3.0001,
              0.2 + cross,
              10.05,
              0.0,
              -1.0,
              5.0,
              5.0,
              2.0,
              5.0,
              2.5,
              0.5,
              0.10,
              2.5 - shear,
              0.5 - cross,
              9.85,
          },
      },
  };
  for (const auto &tc : wires) {
    const auto ref = bruteAllImages(tc.pos, tc.cell, cutoff, tc.periodic);
    REQUIRE(ref.count({0, 1}) == 1);
    REQUIRE(ref.count({4, 5}) == 1);
    for (const auto mode :
         {rgpot::nlist::Mode::Stream, rgpot::nlist::Mode::Stored,
          rgpot::nlist::Mode::Auto}) {
      rgpot::nlist::Spec spec;
      spec.cutoff = cutoff;
      spec.skin = 0.0;
      spec.periodic = tc.periodic;
      spec.image = rgpot::nlist::Image::Round;
      spec.mode = mode;
      rgpot::nlist::Neighbours cells;
      cells.update(tc.pos.data(), tc.pos.size() / 3, tc.cell, spec);
      requireSamePairs(walkedPairs(cells, tc.pos.size() / 3, cutoff), ref);
    }
  }
}

TEST_CASE("fractional bin edges keep nextafter cutoff pairs across exact bin boundaries",
          "[nlist][triclinic][boundary]") {
  setThreads("1");
  const double cutoff = 1.0;
  const double d_below = std::nextafter(cutoff, 0.0);
  // Column 0 of cell^-1 is (0.1, -0.075, 0) with exact Euclidean norm 0.125 =
  // 1/8. Without an ulp cushion on frac_edge, axisCount(1.0, 0.125) selects 8
  // fractional bins and rounding in r * frac_inv places pairs at distance
  // nextafter(cutoff, 0) two bins apart.
  const double cell[9] = {10.0, 0.0, 0.0, 7.5, 10.0, 0.0, 0.0, 0.0, 10.0};

  const std::vector<double> direct_pair = {
      1.2499999999999996, 0.0, 5.0,
      2.0499999999999994, -0.60000000000000009, 5.0,
  };
  const std::vector<double> wrapped_pair = {
      1.2499999999999996, 0.0, 0.25,
      2.0499999999999994, -0.60000000000000009, 10.25,
  };
  for (const auto &periodic :
       {std::array<bool, 3>{true, true, true},
        std::array<bool, 3>{true, true, false}}) {
    const auto ref = bruteAllImages(direct_pair, cell, cutoff, periodic);
    REQUIRE(ref.size() == 1);
    REQUIRE(std::sqrt(ref.at({0, 1})) == d_below);
    for (const auto mode :
         {rgpot::nlist::Mode::Stream, rgpot::nlist::Mode::Stored,
          rgpot::nlist::Mode::Auto}) {
      rgpot::nlist::Spec spec;
      spec.cutoff = cutoff;
      spec.skin = 0.0;
      spec.periodic = periodic;
      spec.image = rgpot::nlist::Image::Round;
      spec.mode = mode;
      rgpot::nlist::Neighbours cells;
      cells.update(direct_pair.data(), direct_pair.size() / 3, cell, spec);
      requireSamePairs(walkedPairs(cells, direct_pair.size() / 3, cutoff), ref);
    }
  }
  {
    const std::array<bool, 3> periodic{true, true, true};
    const auto ref = bruteAllImages(wrapped_pair, cell, cutoff, periodic);
    REQUIRE(ref.size() == 1);
    REQUIRE(std::sqrt(ref.at({0, 1})) <= d_below);
    for (const auto mode :
         {rgpot::nlist::Mode::Stream, rgpot::nlist::Mode::Stored,
          rgpot::nlist::Mode::Auto}) {
      rgpot::nlist::Spec spec;
      spec.cutoff = cutoff;
      spec.skin = 0.0;
      spec.periodic = periodic;
      spec.image = rgpot::nlist::Image::Round;
      spec.mode = mode;
      rgpot::nlist::Neighbours cells;
      cells.update(wrapped_pair.data(), wrapped_pair.size() / 3, cell, spec);
      requireSamePairs(walkedPairs(cells, wrapped_pair.size() / 3, cutoff),
                       ref);
    }
  }

  // Open axis 0 with anchor atoms spanning fractional [0, 1] so the open axis
  // hits the same 1 / 0.125 = 8 bin boundary without the ulp cushion.
  const std::vector<double> open_axis_pos = {
      1.2499999999999996, 0.0, 5.0,
      2.0499999999999994, -0.60000000000000009, 5.0,
      0.0, 0.0, 0.0,
      10.0, 0.0, 8.5,
  };
  const std::array<bool, 3> open_a{false, true, true};
  const auto open_ref = bruteAllImages(open_axis_pos, cell, cutoff, open_a);
  REQUIRE(open_ref.count({0, 1}) == 1);
  REQUIRE(std::sqrt(open_ref.at({0, 1})) == d_below);
  for (const auto mode :
       {rgpot::nlist::Mode::Stream, rgpot::nlist::Mode::Stored,
        rgpot::nlist::Mode::Auto}) {
    rgpot::nlist::Spec spec;
    spec.cutoff = cutoff;
    spec.skin = 0.0;
    spec.periodic = open_a;
    spec.image = rgpot::nlist::Image::Round;
    spec.mode = mode;
    rgpot::nlist::Neighbours cells;
    cells.update(open_axis_pos.data(), open_axis_pos.size() / 3, cell, spec);
    requireSamePairs(walkedPairs(cells, open_axis_pos.size() / 3, cutoff),
                     open_ref);
  }
}
