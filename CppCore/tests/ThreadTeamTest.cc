// MIT License
// Copyright 2023--present rgpot developers
//
// A potential gives the bits of one thread on any thread count, and on any
// team the runtime starts: under OMP_THREAD_LIMIT, inside another region,
// with dynamic teams, or below a TeamCap. A call on fewer than
// rgpot::kTeamMinAtoms atoms opens no region. A neighbour table has the
// order of its slice count on any team. meson runs this file again under
// OMP_THREAD_LIMIT=1 and 2.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <catch2/catch_all.hpp>

#include "lsan_libomp.hpp"
#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/Slices.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/nlist/NeighbourList.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

// Against one slice, a table distance may move only by rounding. A lost or
// repeated pair moves a sorted distance by a whole shell. 1e-9 of the
// scale, with a floor of 1 for values near zero, separates the two.
constexpr double kTolerance = 1e-9;

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

double scaledGap(double a, double b) {
  return std::abs(a - b) / std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

std::uint64_t bits(double v) {
  std::uint64_t out = 0;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}

struct System {
  std::vector<double> pos;
  std::vector<int> z;
  double length = 0;
};

// Lattice points with a fixed jitter, so no force cancels by symmetry.
System lattice(const std::vector<std::array<double, 3>> &basis, int n, double a, int z) {
  System s;
  for (int ix = 0; ix < n; ++ix) {
    for (int iy = 0; iy < n; ++iy) {
      for (int iz = 0; iz < n; ++iz) {
        for (const auto &p : basis) {
          const double cell[3] = {ix + p[0], iy + p[1], iz + p[2]};
          for (const double c : cell) {
            const auto k = static_cast<double>(s.pos.size());
            s.pos.push_back(c * a + 0.01 * a * std::sin(1.7 * k + 0.3));
          }
          s.z.push_back(z);
        }
      }
    }
  }
  s.length = n * a;
  return s;
}

System fcc(int n, double a, int z) {
  return lattice({{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}}, n, a, z);
}

System diamond(int n, double a, int z) {
  return lattice({{0, 0, 0},
                  {0.5, 0.5, 0},
                  {0.5, 0, 0.5},
                  {0, 0.5, 0.5},
                  {0.25, 0.25, 0.25},
                  {0.75, 0.75, 0.25},
                  {0.75, 0.25, 0.75},
                  {0.25, 0.75, 0.75}},
                 n, a, z);
}

// The `n` points of an fcc block nearest its centre, in a box a thousand
// lattice constants wide.
System cluster(std::size_t n, double a, int z) {
  const System block =
      fcc(static_cast<int>(std::cbrt(static_cast<double>(n) / 4.0)) + 2, a, z);
  const double centre = 0.5 * block.length;
  const auto dist2 = [&](std::size_t i) {
    double d2 = 0;
    for (std::size_t c = 0; c < 3; ++c) {
      const double d = block.pos[3 * i + c] - centre;
      d2 += d * d;
    }
    return d2;
  };
  std::vector<std::size_t> order(block.z.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t p, std::size_t q) { return dist2(p) < dist2(q); });
  System s;
  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t i = order[k];
    s.pos.insert(s.pos.end(), block.pos.begin() + 3 * i, block.pos.begin() + 3 * i + 3);
    s.z.push_back(z);
  }
  s.length = 1000 * a;
  return s;
}

// Everything a call reports: the energy, forces and stress of forceImpl,
// then, with `split` and where the potential splits its sums over the
// atoms, those of forcePerAtom with the per-atom energies and virials.
using Result = std::vector<double>;

template <typename Pot>
Result evaluate(Pot &pot, const System &s, bool split = true) {
  const std::size_t n = s.z.size();
  const double box[9] = {s.length, 0, 0, 0, s.length, 0, 0, 0, s.length};
  const rgpot::ForceInput in{n, s.pos.data(), s.z.data(), box};
  std::vector<double> forces(3 * n, 0.0);
  Result r;
  const auto keep = [&](const rgpot::ForceOut &out) {
    r.push_back(out.energy);
    r.insert(r.end(), forces.begin(), forces.end());
    r.insert(r.end(), out.stress, out.stress + 9);
  };
  rgpot::ForceOut out{};
  out.F = forces.data();
  pot.forceImpl(in, &out);
  keep(out);
  if constexpr (rgpot::PerAtomPotential<Pot>) {
    if (split) {
      std::vector<double> energy(n, 0.0);
      std::vector<double> virial(6 * n, 0.0);
      std::fill(forces.begin(), forces.end(), 0.0);
      rgpot::ForceOut atoms{};
      atoms.F = forces.data();
      pot.forcePerAtom(in, &atoms, rgpot::PerAtomOut{energy.data(), virial.data()});
      keep(atoms);
      r.insert(r.end(), energy.begin(), energy.end());
      r.insert(r.end(), virial.begin(), virial.end());
    }
  }
  return r;
}

// Values whose bits differ. Results of two lengths differ everywhere.
std::size_t bitsApart(const Result &a, const Result &b) {
  if (a.size() != b.size()) {
    return std::max(a.size(), b.size());
  }
  std::size_t apart = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    apart += bits(a[i]) == bits(b[i]) ? 0 : 1;
  }
  return apart;
}

#if defined(_OPENMP) && _OPENMP >= 200805
// `eval` on two objects from an outer region of two threads, with `levels`
// active levels allowed. At 1 every inner region gets one thread; at 2 the
// inner regions get their own teams. The objects are made before the
// region, and the outer region shares out its two calls the same way the
// helpers share out slices, since it may get one thread too.
template <typename Make, typename Eval>
auto fromOuterTeam(Make &make, Eval &eval, int levels) {
  auto first = make();
  auto second = make();
  std::vector<decltype(eval(first))> out(2);
  const int saved = omp_get_max_active_levels();
  omp_set_max_active_levels(levels);
#pragma omp parallel num_threads(2)
  {
    const int team = omp_get_num_threads();
    for (int t = omp_get_thread_num(); t < 2; t += team) {
      out[static_cast<std::size_t>(t)] = eval(t == 0 ? first : second);
    }
  }
  omp_set_max_active_levels(saved);
  return out;
}
#endif

// `eval` on a fresh object from `make` under every short team this file
// can force, each paired with its name.
template <typename Make, typename Eval>
auto underShortTeams(Make make, Eval eval) {
  using Out = decltype(eval(std::declval<decltype(make()) &>()));
  std::vector<std::pair<std::string, Out>> out;
  for (const int cap : {1, 2, 3}) {
    const rgpot::TeamCap team(cap);
    auto obj = make();
    out.emplace_back("TeamCap(" + std::to_string(cap) + ")", eval(obj));
  }
#if defined(_OPENMP)
  {
    const int dynamic = omp_get_dynamic();
    omp_set_dynamic(1);
    auto obj = make();
    out.emplace_back("dynamic teams", eval(obj));
    omp_set_dynamic(dynamic);
  }
#if _OPENMP >= 200805
  for (const int levels : {1, 2}) {
    for (auto &r : fromOuterTeam(make, eval, levels)) {
      out.emplace_back("inside a region, max active levels " + std::to_string(levels),
                       std::move(r));
    }
  }
#endif
#endif
  return out;
}

// Everything below, or the thread counts on forceImpl alone.
enum class Depth { Full, Counts };

// One thread against 1, 2, 3 and 8 threads, bit for bit, and with
// Depth::Full 8 threads on every short team as well. Below
// rgpot::kTeamMinAtoms no call opens a region. From there up, with OpenMP,
// a second call at 8 threads opens one: it reuses the neighbour list, so
// the region is the potential's own sum. `make` returns a pointer, since a
// potential need not move.
template <typename Make>
void checkPotential(const char *name, Make make, const System &s, Depth depth = Depth::Full) {
  const bool large = s.z.size() >= rgpot::kTeamMinAtoms;
  const bool split = depth == Depth::Full;
  INFO(name << " on " << s.z.size() << " atoms");
  setThreads("");
  const Result serial = evaluate(*make(), s, split);
  for (const int threads : {1, 2, 3, 8}) {
    INFO(threads << " threads");
    const std::string count = std::to_string(threads);
    setThreads(count.c_str());
    const std::uint64_t start = rgpot::teamsOpened();
    auto pot = make();
    CHECK(bitsApart(evaluate(*pot, s, split), serial) == 0);
    if (!large || threads == 1) {
      CHECK(rgpot::teamsOpened() == start);
    } else if (threads == 8) {
      const std::uint64_t built = rgpot::teamsOpened();
      CHECK(bitsApart(evaluate(*pot, s, split), serial) == 0);
#if defined(_OPENMP)
      CHECK(rgpot::teamsOpened() > built);
#endif
    }
  }
  if (depth == Depth::Full) {
    setThreads("8");
    const auto eval = [&s](auto &p) { return evaluate(*p, s); };
    for (const auto &[shape, r] : underShortTeams(make, eval)) {
      INFO(shape);
      CHECK(bitsApart(r, serial) == 0);
    }
  }
  setThreads("");
}

template <typename Pot>
auto mode(rgpot::nlist::Mode m) {
  return [m] {
    auto p = std::make_unique<Pot>();
    p->setNeighbourMode(m);
    return p;
  };
}

template <typename Pot, typename Config>
auto mode(const Config &c, rgpot::nlist::Mode m) {
  return [c, m] {
    auto p = std::make_unique<Pot>(c);
    p->setNeighbourMode(m);
    return p;
  };
}

constexpr rgpot::nlist::Mode kStored = rgpot::nlist::Mode::Stored;
constexpr rgpot::nlist::Mode kStream = rgpot::nlist::Mode::Stream;

} // namespace

TEST_CASE("partition covers its range once, in slice order", "[threads][slices]") {
  for (const std::int64_t n : {0, 1, 5, 64, 1001}) {
    for (const int nslices : {1, 2, 3, 8, 13}) {
      std::int64_t next = 0;
      for (int s = 0; s < nslices; ++s) {
        const auto r = rgpot::partition(n, s, nslices);
        CHECK(r.begin == next);
        CHECK(r.end >= r.begin);
        next = r.end;
      }
      CHECK(next == n);
      CHECK(rgpot::partition(n, nslices, nslices).end == 0);
      CHECK(rgpot::partition(n, -1, nslices).end == 0);
    }
  }
}

TEST_CASE("each slice runs once on any team, and phases stay in order",
          "[threads][slices]") {
  for (const int nslices : {1, 2, 3, 4, 7, 16}) {
    for (const int cap : {0, 1, 2, 3}) {
      INFO(nslices << " slices under TeamCap(" << cap << ")");
      const rgpot::TeamCap team(cap);
      std::vector<int> hits(static_cast<std::size_t>(nslices), 0);
      rgpot::forEachSlice(nslices, [&](int s) { ++hits[static_cast<std::size_t>(s)]; });
      CHECK(std::count(hits.begin(), hits.end(), 1) == nslices);

      constexpr int kPhases = 3;
      std::array<std::atomic<int>, kPhases> done{};
      std::atomic<int> early{0};
      std::vector<int> visits(static_cast<std::size_t>(kPhases * nslices), 0);
      rgpot::forEachSlicePhase(kPhases, nslices, [&](int phase, int s) {
        if (phase > 0 && done[static_cast<std::size_t>(phase - 1)].load() != nslices) {
          early.fetch_add(1);
        }
        ++visits[static_cast<std::size_t>(phase * nslices + s)];
        done[static_cast<std::size_t>(phase)].fetch_add(1);
      });
      CHECK(early.load() == 0);
      CHECK(std::count(visits.begin(), visits.end(), 1) == kPhases * nslices);
    }
  }
}

namespace {

// Ranks that wait for each other, with a deadline so a team that runs them
// one after another fails instead of hanging.
struct Meeting {
  std::atomic<int> arrived{0};
  std::atomic<int> runs{0};
  std::atomic<bool> stuck{false};

  void operator()(int /*rank*/) {
    runs.fetch_add(1);
    arrived.fetch_add(1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (arrived.load() < 2) {
      if (std::chrono::steady_clock::now() > deadline) {
        stuck.store(true);
        return;
      }
      std::this_thread::yield();
    }
  }
};

#if defined(_OPENMP) && _OPENMP >= 200805
struct Inside {
  int outer_team = 0;
  bool together = false;
  int runs = 0;
  bool stuck = false;
};

// onFullTeam from the first thread of an outer region of two.
Inside fullTeamInside(int levels) {
  Inside got;
  const int saved = omp_get_max_active_levels();
  omp_set_max_active_levels(levels);
#pragma omp parallel num_threads(2)
  {
    if (omp_get_thread_num() == 0) {
      Meeting meet;
      got.outer_team = omp_get_num_threads();
      got.together = rgpot::onFullTeam(2, [&meet](int rank) { meet(rank); });
      got.runs = meet.runs.load();
      got.stuck = meet.stuck.load();
    }
  }
  omp_set_max_active_levels(saved);
  return got;
}
#endif

} // namespace

TEST_CASE("onFullTeam runs every rank at once or none", "[threads][slices]") {
  int ran = 0;
  CHECK(rgpot::onFullTeam(1, [&](int rank) { ran += 1 + rank; }));
  CHECK(ran == 1);
  {
    const rgpot::TeamCap one(1);
    Meeting meet;
    CHECK_FALSE(rgpot::onFullTeam(2, [&meet](int rank) { meet(rank); }));
    CHECK(meet.runs.load() == 0);
  }
  Meeting meet;
  const bool together = rgpot::onFullTeam(2, [&meet](int rank) { meet(rank); });
  CHECK_FALSE(meet.stuck.load());
  CHECK(meet.runs.load() == (together ? 2 : 0));
#if !defined(_OPENMP)
  CHECK_FALSE(together);
#elif _OPENMP >= 200805
  if (omp_get_thread_limit() >= 2 && omp_get_dynamic() == 0 && omp_get_max_active_levels() >= 1) {
    CHECK(together);
  }
  for (const int levels : {1, 2}) {
    INFO("max active levels " << levels);
    const Inside got = fullTeamInside(levels);
    CHECK_FALSE(got.stuck);
    CHECK(got.runs == (got.together ? 2 : 0));
    if (levels == 1 && got.outer_team == 2) {
      CHECK_FALSE(got.together);
    }
  }
#endif
}

TEST_CASE("a call below kTeamMinAtoms takes one slice", "[threads][slices]") {
  setThreads("8");
  CHECK(rgpot::teamSize(0) == 1);
  CHECK(rgpot::teamSize(55) == 1);
  CHECK(rgpot::teamSize(rgpot::kTeamMinAtoms - 1) == 1);
#if defined(_OPENMP)
  CHECK(rgpot::teamSize(rgpot::kTeamMinAtoms) == 8);
#else
  CHECK(rgpot::teamSize(rgpot::kTeamMinAtoms) == 1);
#endif
  setThreads("");
  CHECK(rgpot::teamSize(rgpot::kTeamMinAtoms) == 1);

  const std::uint64_t start = rgpot::teamsOpened();
  int runs = 0;
  rgpot::forEachSlice(1, [&](int) { ++runs; });
  CHECK(rgpot::teamsOpened() == start);
  std::vector<int> hits(4, 0);
  rgpot::forEachSlice(4, [&](int s) { ++hits[static_cast<std::size_t>(s)]; });
#if defined(_OPENMP)
  CHECK(rgpot::teamsOpened() == start + 1);
#else
  CHECK(rgpot::teamsOpened() == start);
#endif
  CHECK(runs == 1);
  CHECK(std::count(hits.begin(), hits.end(), 1) == 4);
}

// Each potential below on a system under rgpot::kTeamMinAtoms atoms and on
// one above it.
TEST_CASE("EAM gives the one-thread bits on any team", "[threads][eam]") {
  for (const int side : {5, 13}) {
    const auto al = fcc(side, 4.05, 13);
    checkPotential("stored EAM", mode<rgpot::EAMPot>(kStored), al);
    checkPotential("streamed EAM", mode<rgpot::EAMPot>(kStream), al);
  }
}

TEST_CASE("pair potentials give the one-thread bits on any team", "[threads][pair]") {
  // The large cells keep two neighbour shells, so a debug build checks
  // them in seconds.
  const double a_lj = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0);
  rgpot::LJConfig lj;
  lj.psi = 1.0;
  rgpot::LJClusterConfig cluster;
  for (const int side : {4, 13}) {
    lj.cutoff = side == 4 ? 2.5 : 1.8;
    cluster.cutoff = lj.cutoff;
    const auto ar = fcc(side, a_lj, 18);
    checkPotential("stored LJ", mode<rgpot::LJPot>(lj, kStored), ar);
    checkPotential("streamed LJ", mode<rgpot::LJPot>(lj, kStream), ar);
    checkPotential("stored LJ cluster", mode<rgpot::LJClusterPot>(cluster, kStored), ar);
    checkPotential("streamed LJ cluster", mode<rgpot::LJClusterPot>(cluster, kStream), ar);
  }

  rgpot::MorseConfig morse;
  for (const int side : {5, 13}) {
    morse.cutoff = side == 5 ? 9.5 : 4.5;
    const auto cu = fcc(side, 4.1, 29);
    checkPotential("stored Morse", mode<rgpot::MorsePot>(morse, kStored), cu);
    checkPotential("streamed Morse", mode<rgpot::MorsePot>(morse, kStream), cu);
  }

  for (const int side : {3, 11}) {
    const auto si = diamond(side, 5.431, 14);
    checkPotential("stored ZBL", mode<rgpot::ZBLPot>(rgpot::ZBLConfig{}, kStored), si);
    checkPotential("streamed ZBL", mode<rgpot::ZBLPot>(rgpot::ZBLConfig{}, kStream), si);
  }
}

TEST_CASE("many-body potentials give the one-thread bits on any team",
          "[threads][manybody]") {
  rgpot::SuttonChenConfig sc;
  sc.periodic = true;
  sc.stress = true;
  for (const int side : {4, 13}) {
    // Three nearest-neighbour distances on the small cell, one and a half
    // on the large one, where three would cost seconds in a debug build.
    sc.cutoff = side == 4 ? 3.0 : 1.5;
    const auto pt = fcc(side, std::sqrt(2.0), 78);
    checkPotential("stored Sutton-Chen", mode<rgpot::SuttonChenPot>(sc, kStored), pt);
    checkPotential("streamed Sutton-Chen", mode<rgpot::SuttonChenPot>(sc, kStream), pt);
  }
  // A cutoff of zero walks every pair of an open cluster. Each call on
  // kTeamMinAtoms atoms adds 33 million pairs twice, so that size takes
  // the thread counts alone.
  rgpot::SuttonChenConfig open;
  open.cutoff = 0.0;
  open.stress = true;
  const auto every_pair = [open] { return std::make_unique<rgpot::SuttonChenPot>(open); };
  checkPotential("Sutton-Chen on every pair", every_pair, cluster(55, std::sqrt(2.0), 78));
  checkPotential("Sutton-Chen on every pair", every_pair,
                 cluster(rgpot::kTeamMinAtoms, std::sqrt(2.0), 78), Depth::Counts);

  for (const int side : {3, 11}) {
    const auto si = diamond(side, 5.431, 14);
    checkPotential("stored Stillinger-Weber", mode<rgpot::StillingerWeberSi>(kStored), si);
    checkPotential("streamed Stillinger-Weber", mode<rgpot::StillingerWeberSi>(kStream), si);
    checkPotential("stored Tersoff", mode<rgpot::TersoffSiC>(kStored), si);
    checkPotential("streamed Tersoff", mode<rgpot::TersoffSiC>(kStream), si);
  }
}

namespace {

template <typename T>
bool sameArray(const std::vector<T> &a, const std::vector<T> &b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

bool sameTable(const rgpot::nlist::Table &a, const rgpot::nlist::Table &b) {
  return sameArray(a.first, b.first) && sameArray(a.second, b.second) &&
         sameArray(a.shift, b.shift) && sameArray(a.vector, b.vector) &&
         sameArray(a.distance, b.distance) && sameArray(a.mask, b.mask);
}

struct Pair {
  std::array<std::int32_t, 5> key{};
  double distance = 0;
};

// Pairs keyed by atoms and image, in key order. A build on several slices
// emits them in cell order, one slice in atom order.
std::vector<Pair> sortedPairs(const rgpot::nlist::Table &t) {
  std::vector<Pair> out(t.size());
  for (std::size_t p = 0; p < t.size(); ++p) {
    out[p].key = {t.first[p], t.second[p], t.shift[3 * p], t.shift[3 * p + 1],
                  t.shift[3 * p + 2]};
    out[p].distance = t.distance[p];
  }
  std::sort(out.begin(), out.end(),
            [](const Pair &a, const Pair &b) { return a.key < b.key; });
  return out;
}

void checkTable(const char *name, const System &s, const rgpot::nlist::Query &query) {
  INFO(name << " on " << s.z.size() << " atoms");
  const double box9[9] = {s.length, 0, 0, 0, s.length, 0, 0, 0, s.length};
  const auto make = [] { return rgpot::nlist::Table{}; };
  const auto eval = [&](rgpot::nlist::Table &t) {
    rgpot::nlist::build(t, s.pos.data(), s.z.size(), box9, query);
    return t;
  };
  setThreads("");
  auto serial = make();
  eval(serial);
  setThreads("4");
  auto full = make();
  eval(full);
  REQUIRE(full.size() > s.z.size());
  if (std::holds_alternative<rgpot::nlist::Pairs>(query.shape)) {
    const auto a = sortedPairs(serial);
    const auto b = sortedPairs(full);
    REQUIRE(a.size() == b.size());
    std::size_t keys_apart = 0;
    double gap = 0;
    for (std::size_t p = 0; p < a.size(); ++p) {
      keys_apart += a[p].key == b[p].key ? 0 : 1;
      gap = std::max(gap, scaledGap(a[p].distance, b[p].distance));
    }
    CHECK(keys_apart == 0);
    CHECK(gap <= kTolerance);
  }
  for (const auto &[shape, t] : underShortTeams(make, eval)) {
    INFO(shape);
    CHECK(sameTable(t, full));
  }
  setThreads("");
}

} // namespace

TEST_CASE("neighbour tables keep the order of their slice count on any team",
          "[threads][nlist]") {
  rgpot::nlist::Query query;
  query.cutoff = 5.0;
  const auto cube = fcc(13, 4.05, 13);
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  checkTable("half shell", cube, query);
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
  checkTable("full shell", cube, query);
  query.shape = rgpot::nlist::Cluster4{};
  checkTable("clusters of four", cube, query);
}
