// MIT License
// Copyright 2023--present rgpot developers
//
// The runtime may start fewer threads than a region asks for: under
// OMP_THREAD_LIMIT, inside another region, with dynamic teams, or below a
// TeamCap. A result then has the bits of its slice count. meson runs this
// file again under OMP_THREAD_LIMIT=1 and 2.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
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

// Against one slice, a result may move only by the order of its sums. The
// longest here, the Morse energy, adds about 52000 pair terms of mostly one
// sign, and any two orders of that sum differ by at most about 1e-11 of its
// size. A lost or repeated slice moves the energy by a share of the whole,
// near an eV, and a force by a whole bond. 1e-9 of the scale, with a floor
// of 1 for values near zero, separates the two.
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

struct Result {
  double energy = 0;
  std::vector<double> forces;
};

template <typename Pot>
Result evaluate(Pot &pot, const System &s) {
  rgpot::types::AtomMatrix xyz(s.z.size(), 3);
  for (std::size_t i = 0; i < s.z.size(); ++i) {
    xyz(i, 0) = s.pos[3 * i];
    xyz(i, 1) = s.pos[3 * i + 1];
    xyz(i, 2) = s.pos[3 * i + 2];
  }
  const std::array<std::array<double, 3>, 3> box{{
      {s.length, 0, 0},
      {0, s.length, 0},
      {0, 0, s.length},
  }};
  const auto out = pot(xyz, s.z, box);
  const auto &f = std::get<1>(out);
  return {std::get<0>(out), std::vector<double>(f.data(), f.data() + 3 * s.z.size())};
}

double scaledGap(const Result &a, const Result &b) {
  double gap = scaledGap(a.energy, b.energy);
  for (std::size_t i = 0; i < a.forces.size(); ++i) {
    gap = std::max(gap, scaledGap(a.forces[i], b.forces[i]));
  }
  return gap;
}

// Values whose bits differ, the energy included.
std::size_t bitsApart(const Result &a, const Result &b) {
  std::size_t apart = bits(a.energy) == bits(b.energy) ? 0 : 1;
  for (std::size_t i = 0; i < a.forces.size(); ++i) {
    apart += bits(a.forces[i]) == bits(b.forces[i]) ? 0 : 1;
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

// One slice against `slices` slices within the tolerance, then `slices`
// slices on every short team against the full team bit for bit. `make`
// returns a pointer, since a potential need not move.
template <typename Make>
void checkPotential(const char *name, Make make, const System &s, int slices) {
  INFO(name << " on " << s.z.size() << " atoms, " << slices << " slices");
  setThreads("");
  const Result serial = evaluate(*make(), s);
  const std::string count = std::to_string(slices);
  setThreads(count.c_str());
  const Result full = evaluate(*make(), s);
  CHECK(scaledGap(serial, full) <= kTolerance);
  const auto eval = [&s](auto &p) { return evaluate(*p, s); };
  for (const auto &[shape, r] : underShortTeams(make, eval)) {
    INFO(shape);
    CHECK(bitsApart(r, full) == 0);
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

TEST_CASE("EAM keeps the bits of its slice count on any team", "[threads][eam]") {
  const auto al = fcc(5, 4.05, 13);
  checkPotential("stored EAM", mode<rgpot::EAMPot>(kStored), al, 2);
  checkPotential("stored EAM", mode<rgpot::EAMPot>(kStored), al, 4);
  checkPotential("streamed EAM", mode<rgpot::EAMPot>(kStream), al, 4);
}

TEST_CASE("pair potentials keep the bits of their slice count on any team",
          "[threads][pair]") {
  const double a_lj = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0);
  const auto ar = fcc(4, a_lj, 18);
  rgpot::LJConfig lj;
  lj.cutoff = 2.5;
  lj.psi = 1.0;
  checkPotential("stored LJ", mode<rgpot::LJPot>(lj, kStored), ar, 4);
  checkPotential("streamed LJ", mode<rgpot::LJPot>(lj, kStream), ar, 4);
  rgpot::LJClusterConfig cluster;
  cluster.cutoff = 2.5;
  checkPotential(
      "LJ cluster", [cluster] { return std::make_unique<rgpot::LJClusterPot>(cluster); }, ar, 4);

  const auto cu = fcc(5, 4.1, 29);
  checkPotential("stored Morse", mode<rgpot::MorsePot>(rgpot::MorseConfig{}, kStored), cu, 4);
  checkPotential("streamed Morse", mode<rgpot::MorsePot>(rgpot::MorseConfig{}, kStream), cu, 4);

  const auto si = diamond(3, 5.431, 14);
  checkPotential("stored ZBL", mode<rgpot::ZBLPot>(rgpot::ZBLConfig{}, kStored), si, 4);
}

TEST_CASE("many-body potentials keep the bits of their slice count on any team",
          "[threads][manybody]") {
  rgpot::SuttonChenConfig sc;
  sc.cutoff = 3.0;
  sc.periodic = true;
  const auto pt = fcc(4, std::sqrt(2.0), 78);
  checkPotential("stored Sutton-Chen", mode<rgpot::SuttonChenPot>(sc, kStored), pt, 4);
  checkPotential("streamed Sutton-Chen", mode<rgpot::SuttonChenPot>(sc, kStream), pt, 4);

  const auto si = diamond(3, 5.431, 14);
  checkPotential("stored Stillinger-Weber", mode<rgpot::StillingerWeberSi>(kStored), si, 4);
  checkPotential("streamed Stillinger-Weber", mode<rgpot::StillingerWeberSi>(kStream), si, 4);
  checkPotential("stored Tersoff", mode<rgpot::TersoffSiC>(kStored), si, 4);
  checkPotential("streamed Tersoff", mode<rgpot::TersoffSiC>(kStream), si, 4);
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
