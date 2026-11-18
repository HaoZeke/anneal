// MIT License
// Copyright 2023--present rgpot developers
//
// Tersoff and ZBL store a neighbour list on the same atom slices as the
// other classical potentials. The energy, forces and stress match one
// thread at 1, 2, 4 and 8 threads, below and above kTeamMinAtoms, on a
// fresh list, a Verlet reuse and a rebuild. An open search in a huge
// box keeps the pairs of a direct scan.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "lsan_libomp.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/Slices.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
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

struct RestoreThreads {
  ~RestoreThreads() { setThreads(nullptr); }
};

std::uint64_t bits(double v) {
  std::uint64_t out = 0;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}

const double kBasis[8][3] = {
    {0, 0, 0},          {0.5, 0.5, 0},      {0.5, 0, 0.5},
    {0, 0.5, 0.5},      {0.25, 0.25, 0.25}, {0.75, 0.75, 0.25},
    {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75},
};

// Diamond sites with a fixed jitter, so no force cancels by symmetry.
std::vector<double> diamond(int side, double a) {
  std::vector<double> pos;
  for (int ix = 0; ix < side; ++ix) {
    for (int iy = 0; iy < side; ++iy) {
      for (int iz = 0; iz < side; ++iz) {
        for (const auto &p : kBasis) {
          const double cell[3] = {ix + p[0], iy + p[1], iz + p[2]};
          for (const double c : cell) {
            const auto k = static_cast<double>(pos.size());
            pos.push_back(c * a + 0.01 * a * std::sin(1.7 * k + 0.3));
          }
        }
      }
    }
  }
  return pos;
}

struct Snapshot {
  double energy = 0;
  std::vector<double> force;
  double stress[9]{};
};

std::size_t apart(const Snapshot &a, const Snapshot &b) {
  if (a.force.size() != b.force.size()) {
    return a.force.size() + b.force.size() + 1;
  }
  std::size_t n = bits(a.energy) == bits(b.energy) ? 0 : 1;
  for (std::size_t i = 0; i < a.force.size(); ++i) {
    n += bits(a.force[i]) == bits(b.force[i]) ? 0 : 1;
  }
  for (int k = 0; k < 9; ++k) {
    n += bits(a.stress[k]) == bits(b.stress[k]) ? 0 : 1;
  }
  return n;
}

template <typename Pot>
Snapshot evaluate(Pot &pot, const std::vector<double> &pos,
                  const std::vector<int> &z, const double *box) {
  const std::size_t n = z.size();
  const rgpot::ForceInput in{n, pos.data(), z.data(), box};
  Snapshot s;
  s.force.assign(3 * n, 0.0);
  rgpot::ForceOut out{};
  out.F = s.force.data();
  pot.forceImpl(in, &out);
  s.energy = out.energy;
  std::memcpy(s.stress, out.stress, sizeof(s.stress));
  return s;
}

std::vector<double> shifted(const std::vector<double> &pos, double dx) {
  auto out = pos;
  for (std::size_t i = 0; i < out.size(); i += 3) {
    out[i] += dx;
  }
  return out;
}

const char *kThreads[] = {"1", "2", "4", "8"};

// Half a skin is 0.30 for Tersoff and 0.25 for ZBL. 0.05 stays inside
// both. 0.45 is past both, and far from either boundary.
constexpr double kInside = 0.05;
constexpr double kPast = 0.45;

template <typename Pot>
void agree(const char *label, const std::vector<double> &pos,
           const double *box) {
  const std::size_t n = pos.size() / 3;
  const std::vector<int> z(n, 14);
  const bool large = n >= rgpot::kTeamMinAtoms;
  const auto inside = shifted(pos, kInside);
  const auto past = shifted(pos, kPast);

  setThreads("1");
  Pot serial;
  serial.setNeighbourMode(rgpot::nlist::Mode::Stored);
  const Snapshot ref = evaluate(serial, pos, z, box);
  Pot fresh_inside;
  fresh_inside.setNeighbourMode(rgpot::nlist::Mode::Stored);
  const Snapshot ref_inside = evaluate(fresh_inside, inside, z, box);
  Pot fresh_past;
  fresh_past.setNeighbourMode(rgpot::nlist::Mode::Stored);
  const Snapshot ref_past = evaluate(fresh_past, past, z, box);

  for (const char *th : kThreads) {
    INFO(label << " threads " << th << " atoms " << n);
    setThreads(th);
    const auto before = rgpot::teamsOpened();
    Pot pot;
    pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
    const Snapshot got = evaluate(pot, pos, z, box);
    REQUIRE(apart(got, ref) == 0);
    if (std::string(th) == "1" || !large) {
      REQUIRE(rgpot::teamsOpened() == before);
    } else {
      REQUIRE(rgpot::teamsOpened() > before);
    }

    const Snapshot reused = evaluate(pot, inside, z, box);
    REQUIRE(pot.listReused());
    REQUIRE(apart(reused, ref_inside) == 0);

    const Snapshot rebuilt = evaluate(pot, past, z, box);
    REQUIRE_FALSE(pot.listReused());
    REQUIRE(apart(rebuilt, ref_past) == 0);
  }
}

rgpot::nlist::Spec storedSpec(bool periodic, double cutoff, double skin) {
  rgpot::nlist::Spec spec;
  spec.cutoff = cutoff;
  spec.skin = skin;
  spec.periodic = {{periodic, periodic, periodic}};
  spec.image = rgpot::nlist::Image::Round;
  spec.mode = rgpot::nlist::Mode::Stored;
  return spec;
}

rgpot::nlist::Neighbours buildList(const std::vector<double> &pos,
                                   const double *box,
                                   const rgpot::nlist::Spec &spec) {
  rgpot::nlist::Neighbours cells;
  cells.update(pos.data(), pos.size() / 3, box, spec);
  return cells;
}

void directOpen(const std::vector<double> &pos, double search,
                std::vector<std::int32_t> &off,
                std::vector<std::int32_t> &nbr) {
  const std::size_t n = pos.size() / 3;
  const double search2 = search * search;
  off.assign(n + 1, 0);
  nbr.clear();
  for (std::size_t i = 0; i < n; ++i) {
    off[i] = static_cast<std::int32_t>(nbr.size());
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      const double dx = pos[3 * i] - pos[3 * j];
      const double dy = pos[3 * i + 1] - pos[3 * j + 1];
      const double dz = pos[3 * i + 2] - pos[3 * j + 2];
      if (dx * dx + dy * dy + dz * dz <= search2) {
        nbr.push_back(static_cast<std::int32_t>(j));
      }
    }
  }
  off[n] = static_cast<std::int32_t>(nbr.size());
}

} // namespace

TEST_CASE("Tersoff and ZBL agree across thread counts",
          "[threads][nlist][tersoff][zbl]") {
  const RestoreThreads restore;
  const double a = 5.431;
  // 512 atoms is below kTeamMinAtoms. 10648 is above it.
  for (const int side : {4, 11}) {
    const auto pos = diamond(side, a);
    const double length = side * a;
    const double box[9] = {length, 0, 0, 0, length, 0, 0, 0, length};
    agree<rgpot::TersoffSiC>("Tersoff", pos, box);
    agree<rgpot::ZBLPot>("ZBL", pos, box);
    agree<rgpot::ZBLPot>("ZBL free", pos, nullptr);
  }
}

TEST_CASE("a stored neighbour index matches one thread", "[threads][nlist]") {
  const RestoreThreads restore;
  const double a = 5.431;
  const auto small = diamond(4, a);
  const auto large = diamond(11, a);
  const double small_box[9] = {4 * a, 0, 0, 0, 4 * a, 0, 0, 0, 4 * a};
  const double large_box[9] = {11 * a, 0, 0, 0, 11 * a, 0, 0, 0, 11 * a};
  const double huge[9] = {1.0e6, 0, 0, 0, 1.0e6, 0, 0, 0, 1.0e6};
  const auto periodic = storedSpec(true, 3.0, 0.6);
  const auto open = storedSpec(false, 3.0, 0.6);

  setThreads("1");
  const auto periodic_ref = buildList(large, large_box, periodic);
  const auto open_ref = buildList(large, huge, open);
  REQUIRE(periodic_ref.mode() == rgpot::nlist::Mode::Stored);
  REQUIRE(open_ref.mode() == rgpot::nlist::Mode::Stored);
  REQUIRE(open_ref.slots() < large.size() / 3 * (large.size() / 3 - 1));
  std::vector<std::int32_t> off;
  std::vector<std::int32_t> nbr;
  directOpen(small, 3.6, off, nbr);
  setThreads("1");
  const auto open_small = buildList(small, huge, open);
  REQUIRE(open_small.offsets() == off);
  REQUIRE(open_small.neighbours() == nbr);

  for (const char *th : kThreads) {
    INFO("threads " << th);
    setThreads(th);
    const bool one = std::string(th) == "1";
    const auto before = rgpot::teamsOpened();
    const auto periodic_got = buildList(large, large_box, periodic);
    if (one) {
      REQUIRE(rgpot::teamsOpened() == before);
    } else {
      REQUIRE(rgpot::teamsOpened() > before);
    }
    REQUIRE(periodic_got.offsets() == periodic_ref.offsets());
    REQUIRE(periodic_got.neighbours() == periodic_ref.neighbours());

    const auto open_before = rgpot::teamsOpened();
    const auto open_got = buildList(large, huge, open);
    if (one) {
      REQUIRE(rgpot::teamsOpened() == open_before);
    } else {
      REQUIRE(rgpot::teamsOpened() > open_before);
    }
    REQUIRE(open_got.offsets() == open_ref.offsets());
    REQUIRE(open_got.neighbours() == open_ref.neighbours());

    const auto small_before = rgpot::teamsOpened();
    const auto small_got = buildList(small, small_box, periodic);
    REQUIRE(rgpot::teamsOpened() == small_before);
    setThreads("1");
    const auto small_ref = buildList(small, small_box, periodic);
    REQUIRE(small_got.offsets() == small_ref.offsets());
    REQUIRE(small_got.neighbours() == small_ref.neighbours());
    setThreads(th);
  }
}
