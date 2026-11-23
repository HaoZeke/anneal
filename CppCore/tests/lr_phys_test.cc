// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/CudaPath.hpp"
#include "rgpot/LongRange/Dispersion.hpp"
#include "rgpot/LongRange/FastD3.hpp"
#include "rgpot/LongRange/FastD4.hpp"
#include "rgpot/LongRange/Electro.hpp"
#include "rgpot/LongRange/Lattice.hpp"
#include "rgpot/LongRange/Nlh.hpp"
#include "rgpot/LongRange/SumPot.hpp"
#include "rgpot/LongRange/Team.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using rgpot::ForceInput;
using rgpot::ForceOut;
using rgpot::lr::CellGeom;
using rgpot::lr::CoulombConfig;
using rgpot::lr::CoulombMethod;
using rgpot::lr::CoulombPot;
using rgpot::lr::DispersionConfig;
using rgpot::lr::DispersionKind;
using rgpot::lr::DispersionPot;
using rgpot::lr::DualList;
using rgpot::lr::FixedCharges;
using rgpot::lr::FixedRatios;
using rgpot::lr::NlhPot;
using rgpot::lr::SumPot;
using rgpot::lr::cell_from_box;
using rgpot::lr::so3lr_erf_pair;

namespace {

int g_fails = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    std::cerr << "FAIL " << what << '\n';
    ++g_fails;
  }
}

void check_close(double got, double ref, double tol, const char *what) {
  const double err = std::abs(got - ref);
  const double scale = std::max(1.0, std::abs(ref));
  if (!(err <= tol * scale) || !std::isfinite(got)) {
    std::cerr << "FAIL " << what << " got " << got << " ref " << ref << " err " << err << '\n';
    ++g_fails;
  }
}

std::vector<std::tuple<int, int, double, double, double, double>>
pair_signature(const std::vector<rgpot::lr::Pair> &pairs) {
  std::vector<std::tuple<int, int, double, double, double, double>> out;
  out.reserve(pairs.size());
  for (const auto &p : pairs) {
    out.emplace_back(p.i, p.j, p.x, p.y, p.z, p.r);
  }
  return out;
}

struct Eval {
  double energy = 0;
  std::vector<double> f;
  double stress[9]{};
  bool has_stress = false;
};

template <class Pot>
Eval run(Pot &pot, const std::vector<double> &pos, const std::vector<int> &z, const double *box) {
  Eval e;
  e.f.assign(pos.size(), 0.0);
  ForceOut out{};
  out.F = e.f.data();
  ForceInput in{pos.size() / 3, pos.data(), z.data(), box};
  pot.forceImpl(in, &out);
  e.energy = out.energy;
  e.has_stress = out.has_stress != 0;
  std::copy(out.stress, out.stress + 9, e.stress);
  return e;
}

template <class Pot>
Eval run_batch(Pot &pot, const std::vector<double> &pos,
               const std::vector<int> &z, const double *box) {
  Eval e;
  e.f.assign(pos.size(), 0.0);
  ForceInput in{pos.size() / 3, pos.data(), z.data(), box};
  ForceOut out{};
  out.F = e.f.data();
  rgpot::ForceBatch batch{1, &in, &out};
  pot.forceBatch(batch);
  e.energy = out.energy;
  e.has_stress = out.has_stress != 0;
  std::copy(out.stress, out.stress + 9, e.stress);
  return e;
}

#if !defined(RGPOT_HAS_DFTD3) || !defined(RGPOT_HAS_DFTD4)
void check_unlinked(DispersionPot &pot, const std::vector<double> &pos, const std::vector<int> &z,
                    const char *what) {
  bool threw = false;
  try {
    run(pot, pos, z, nullptr);
  } catch (const std::runtime_error &ex) {
    threw = std::string(ex.what()).find("not linked") != std::string::npos;
  }
  check(threw, what);
}
#endif

double fd_force(auto &&energy, std::vector<double> pos, std::size_t i, double step) {
  const double x = pos[i];
  pos[i] = x + step;
  const double ep = energy(pos);
  pos[i] = x - step;
  const double em = energy(pos);
  return -(ep - em) / (2.0 * step);
}

void check_periodic_dispersion_self_images(DispersionKind kind, const char *name,
                                           bool check_stress) {
  DispersionConfig cfg;
  cfg.kind = kind;
  cfg.functional = "pbe";
  cfg.atm = false;
  cfg.periodic = true;
  cfg.cutoff = 7.4;
  cfg.short_cutoff = 2.0;
  cfg.lib_disp2 = 12.0;
  cfg.lib_disp3 = 12.0;
  cfg.lib_cn = 12.0;
  const double box[9] = {5, 0, 0, 0, 5, 0, 0, 0, 5};
  std::vector<double> pos = {0.4, 0.5, 0.6, 1.6, 0.7, 0.8};
  const std::vector<int> z = {6, 8};
  DispersionPot pot(cfg);
  const Eval cell = run(pot, pos, z, box);
  std::cout << "CASE " << name << "_periodic_self_images " << cell.energy << '\n';

  constexpr int repeat = 3;
  std::vector<double> super_pos;
  std::vector<int> super_z;
  for (int ix = 0; ix < repeat; ++ix) {
    for (int iy = 0; iy < repeat; ++iy) {
      for (int iz = 0; iz < repeat; ++iz) {
        for (std::size_t atom = 0; atom < z.size(); ++atom) {
          super_pos.insert(super_pos.end(),
                           {pos[3 * atom] + 5.0 * ix, pos[3 * atom + 1] + 5.0 * iy,
                            pos[3 * atom + 2] + 5.0 * iz});
          super_z.push_back(z[atom]);
        }
      }
    }
  }
  const double super_box[9] = {15, 0, 0, 0, 15, 0, 0, 0, 15};
  DispersionPot super(cfg);
  const Eval explicit_cell = run(super, super_pos, super_z, super_box);
  const std::string energy_what =
      std::string(name) + " periodic self-image energy matches an explicit supercell";
  check_close(cell.energy, explicit_cell.energy / (repeat * repeat * repeat), 2e-9,
              energy_what.c_str());

  for (int axis = 0; axis < 3; ++axis) {
    double net = 0.0;
    for (std::size_t atom = 0; atom < z.size(); ++atom) {
      net += cell.f[3 * atom + static_cast<std::size_t>(axis)];
    }
    const std::string net_what = std::string(name) + " periodic self-image net force";
    check_close(net, 0.0, 2e-10, net_what.c_str());
  }
  for (std::size_t i = 0; i < pos.size(); ++i) {
    const double force = fd_force(
        [&](const std::vector<double> &p) { return run(pot, p, z, box).energy; }, pos, i, 1e-6);
    const std::string force_what = std::string(name) + " periodic self-image force";
    check_close(cell.f[i], force, 3e-5, force_what.c_str());
  }

  if (check_stress) {
    check(cell.has_stress, (std::string(name) + " periodic self-image stress is set").c_str());
    const double eps = 1e-6;
    auto strained_energy = [&](double strain) {
      std::vector<double> moved = pos;
      double strained_box[9];
      std::copy(box, box + 9, strained_box);
      strained_box[0] *= 1.0 + strain;
      for (std::size_t i = 0; i < moved.size(); i += 3) {
        moved[i] *= 1.0 + strain;
      }
      return run(pot, moved, z, strained_box).energy;
    };
    const double dedstrain = (strained_energy(eps) - strained_energy(-eps)) / (2.0 * eps);
    const std::string stress_what = std::string(name) + " periodic self-image xx stress";
    check_close(cell.stress[0], dedstrain / 125.0, 5e-5, stress_what.c_str());
  }
}

void test_lists() {
  DualList lists(3.0, 6.0, 0.4);
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> u(0.2, 9.8);
  std::vector<double> pos;
  for (int i = 0; i < 40; ++i) {
    pos.push_back(u(rng));
    pos.push_back(u(rng));
    pos.push_back(u(rng));
  }
  const double box[9] = {10, 0.4, 0.1, 0.2, 11, 0.3, 0.0, -0.2, 9};
  const int n = static_cast<int>(pos.size() / 3);
  auto brute_pairs = [](const std::vector<double> &p, const double *b, double cut) {
    std::vector<rgpot::lr::Pair> out;
    const int count = static_cast<int>(p.size() / 3);
    const CellGeom g = cell_from_box(b);
    const double cut2 = cut * cut;
    for (int i = 0; i < count; ++i) {
      for (int j = i; j < count; ++j) {
        for (int sa = -2; sa <= 2; ++sa) {
          for (int sb = -2; sb <= 2; ++sb) {
            for (int sc = -2; sc <= 2; ++sc) {
              if (i == j) {
                const int sum = sa + sb + sc;
                if (sum < 0 || (sum == 0 && (sc < 0 || (sc == 0 && sb < 0))) ||
                    (sa == 0 && sb == 0 && sc == 0)) {
                  continue;
                }
              }
              const double dx =
                  p[3 * j] - p[3 * i] + sa * g.H[0][0] + sb * g.H[1][0] + sc * g.H[2][0];
              const double dy =
                  p[3 * j + 1] - p[3 * i + 1] + sa * g.H[0][1] + sb * g.H[1][1] + sc * g.H[2][1];
              const double dz =
                  p[3 * j + 2] - p[3 * i + 2] + sa * g.H[0][2] + sb * g.H[1][2] + sc * g.H[2][2];
              const double r2 = dx * dx + dy * dy + dz * dz;
              if (r2 > 0.0 && r2 < cut2) {
                out.push_back(rgpot::lr::Pair{i, j, dx, dy, dz, std::sqrt(r2)});
              }
            }
          }
        }
      }
    }
    return out;
  };
  auto check_matches_brute = [&](const std::vector<double> &p, const double *b,
                                 const char *what) {
    // Pairs of one (i, j) differ by a whole lattice vector, so the rounded
    // shift orders them exactly. A tolerance on the vector would not be a
    // strict weak ordering.
    const CellGeom g = cell_from_box(b);
    auto key = [&](const rgpot::lr::Pair &pr) {
      const auto i = static_cast<std::size_t>(pr.i);
      const auto j = static_cast<std::size_t>(pr.j);
      double r[3] = {pr.x - (p[3 * j] - p[3 * i]), pr.y - (p[3 * j + 1] - p[3 * i + 1]),
                     pr.z - (p[3 * j + 2] - p[3 * i + 2])};
      double s[3];
      g.fractional(r, s);
      return std::make_tuple(pr.i, pr.j, std::lround(s[0]), std::lround(s[1]),
                             std::lround(s[2]));
    };
    auto less = [&](const rgpot::lr::Pair &lhs, const rgpot::lr::Pair &rhs) {
      return key(lhs) < key(rhs);
    };
    for (bool is_long : {false, true}) {
      auto got = is_long ? lists.long_pairs() : lists.short_pairs();
      auto ref = brute_pairs(p, b, is_long ? 6.0 : 3.0);
      std::sort(got.begin(), got.end(), less);
      std::sort(ref.begin(), ref.end(), less);
      check(got.size() == ref.size(), what);
      if (got.size() != ref.size()) {
        continue;
      }
      for (std::size_t k = 0; k < got.size(); ++k) {
        check(got[k].i == ref[k].i && got[k].j == ref[k].j, what);
        check_close(got[k].x, ref[k].x, 1e-12, what);
        check_close(got[k].y, ref[k].y, 1e-12, what);
        check_close(got[k].z, ref[k].z, 1e-12, what);
        check_close(got[k].r, ref[k].r, 1e-12, what);
      }
    }
  };

  lists.update(pos.data(), n, box, true);
  check(lists.short_rebuilt() && lists.long_rebuilt(), "first list build rebuilds");
  check(lists.short_pairs().size() < lists.long_pairs().size(), "long list is the larger cutoff");
  check_matches_brute(pos, box, "initial build matches brute force");

  // Compare the long list to a minimum-image brute force at the same cutoff.
  std::size_t brute_n = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double dx = pos[3 * j] - pos[3 * i];
      double dy = pos[3 * j + 1] - pos[3 * i + 1];
      double dz = pos[3 * j + 2] - pos[3 * i + 2];
      const CellGeom g = cell_from_box(box);
      double s[3];
      double r[3] = {dx, dy, dz};
      g.fractional(r, s);
      for (int c = 0; c < 3; ++c) {
        s[c] -= std::round(s[c]);
      }
      double w[3] = {};
      for (int c = 0; c < 3; ++c) {
        w[c] = s[0] * g.H[0][c] + s[1] * g.H[1][c] + s[2] * g.H[2][c];
      }
      if (w[0] * w[0] + w[1] * w[1] + w[2] * w[2] < 36.0) {
        ++brute_n;
      }
    }
  }
  check(lists.long_pairs().size() >= brute_n, "long list covers the minimum image");
  for (double &x : pos) {
    x += 0.01;
  }
  lists.update(pos.data(), n, box, true);
  check(!lists.short_rebuilt() && !lists.long_rebuilt(), "a small move stays inside the skin");
  check_matches_brute(pos, box, "small move matches brute force");

  const double small_box[9] = {
      box[0] * 1.002, box[1] * 1.002 + 0.01, box[2] * 1.002,
      box[3] * 1.002, box[4] * 1.002 - 0.01, box[5] * 1.002,
      box[6] * 1.002, box[7] * 1.002 + 0.01, box[8] * 1.002,
  };
  for (double &x : pos) {
    x *= 1.002;
  }
  const auto t0 = std::chrono::steady_clock::now();
  lists.update(pos.data(), n, small_box, true);
  const auto reuse_us =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  check(reuse_us < 1e6, "small cell change smoke time");
  check(!lists.short_rebuilt() && !lists.long_rebuilt(),
        "a small cell change stays inside the skin");
  check_matches_brute(pos, small_box, "small cell change matches brute force");

  const auto t_steady = std::chrono::steady_clock::now();
  lists.update(pos.data(), n, small_box, true);
  const auto steady_us =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_steady)
          .count();
  check(steady_us < 1e6, "steady repeat smoke time");
  check(!lists.short_rebuilt() && !lists.long_rebuilt(),
        "a steady repeat reuses the list");
  check_matches_brute(pos, small_box, "steady repeat matches brute force");

  const double large_box[9] = {
      box[0] * 1.06, box[1] * 1.06, box[2] * 1.06,
      box[3] * 1.06, box[4] * 1.06, box[5] * 1.06,
      box[6] * 1.06, box[7] * 1.06, box[8] * 1.06,
  };
  for (double &x : pos) {
    x *= 1.06 / 1.002;
  }
  lists.update(pos.data(), n, large_box, true);
  check(lists.short_rebuilt() && lists.long_rebuilt(),
        "a large cell change rebuilds the skin");
  check_matches_brute(pos, large_box, "large cell change matches brute force");

  pos[0] += 2.0;
  lists.update(pos.data(), n, large_box, true);
  check(lists.short_rebuilt() && lists.long_rebuilt(), "a large move rebuilds the skin");
  check_matches_brute(pos, large_box, "large move matches brute force");
}

void test_duallist_skin_sequences() {
  const double box[9] = {8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0};

  {
    DualList lists(1.0, 1.0, 1.0);
    const std::vector<double> A = {1.0, 1.0, 1.0, 1.9, 1.0, 1.0};
    const std::vector<double> B = {1.0, 1.0, 1.0, 2.2, 1.0, 1.0};
    const int n = 2;

    lists.update(A.data(), n, box, true);
    const auto a_ref = pair_signature(lists.short_pairs());
    check(!a_ref.empty(), "A has short-range pairs");
    lists.update(B.data(), n, box, true);
    check(lists.short_pairs().empty(), "B leaves the short cutoff");
    lists.update(A.data(), n, box, true);
    check(pair_signature(lists.short_pairs()) == a_ref, "A->B->A restores A pairs");
  }

  {
    DualList lists(1.0, 1.0, 1.0);
    const std::vector<double> A = {1.0, 1.0, 1.0, 1.9, 1.0, 1.0};
    const int n = 2;
    lists.update(A.data(), n, box, true);
    const auto a_ref = pair_signature(lists.short_pairs());
    lists.update(nullptr, 0, box, true);
    check(lists.short_pairs().empty(), "n=0 clears the pair list");
    lists.update(A.data(), n, box, true);
    check(pair_signature(lists.short_pairs()) == a_ref, "A->(n=0)->A restores A pairs");
  }

  {
    DualList lists(1.0, 1.0, 1.0);
    const std::vector<double> A = {1.0, 1.0, 1.0, 2.2, 1.0, 1.0};
    const std::vector<double> B = {1.0, 1.0, 1.0, 1.9, 1.0, 1.0};
    const int n = 2;
    lists.update(A.data(), n, box, true);
    lists.update(B.data(), n, box, true);
    check(!lists.short_pairs().empty(), "B enters the short cutoff");
    auto &poison = const_cast<std::vector<rgpot::lr::Pair> &>(lists.short_pairs());
    poison[0].r = -123.0;
    lists.update(B.data(), n, box, true);
    check(!lists.short_rebuilt() && !lists.long_rebuilt(), "A->B->B keeps the skin candidates");
    check(lists.short_pairs()[0].r == -123.0, "A->B->B skips refiltering");
  }

  {
    const double periodic_box[9] = {4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
    const std::vector<double> A = {0.1, 0.0, 0.0, 3.9, 0.0, 0.0};
    const int n = 2;
    DualList lists(1.0, 1.0, 0.5);
    DualList fresh(1.0, 1.0, 0.5);
    lists.update(A.data(), n, periodic_box, true);
    lists.update(nullptr, 0, nullptr, false);
    check(lists.short_pairs().empty() && lists.long_pairs().empty(),
          "periodic->non-periodic clears the emitted lists");
    lists.update(A.data(), n, periodic_box, true);
    fresh.update(A.data(), n, periodic_box, true);
    check(lists.short_pairs().size() == fresh.short_pairs().size(),
          "periodic->non-periodic->periodic short count matches a fresh build");
    check(lists.long_pairs().size() == fresh.long_pairs().size(),
          "periodic->non-periodic->periodic long count matches a fresh build");
  }
}

void test_list_edges() {
  {
    // 18.5 is 1.5 from the image of the origin two cells back.
    const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    const std::vector<double> pos = {0, 0, 0, 18.5, 0, 0};
    DualList lists(1.0, 2.0, 0.3);
    lists.update(pos.data(), 2, box, true);
    check(lists.long_pairs().size() == 1, "an atom two cells out keeps its pair");
    if (lists.long_pairs().size() == 1) {
      const rgpot::lr::Pair &pr = lists.long_pairs()[0];
      check(pr.i == 0 && pr.j == 1, "the unwrapped pair joins atoms 0 and 1");
      check_close(pr.x, -1.5, 1e-12, "the unwrapped pair vector is r_j + S H - r_i");
      check_close(pr.r, 1.5, 1e-12, "the unwrapped pair distance");
    }
    check(lists.short_pairs().empty(), "the unwrapped pair is past the short cutoff");
  }
  {
    const double box[9] = {5, 0, 0, 0, 6, 0, 0, 0, 7};
    const std::vector<double> wrapped = {
        0.4, 0.5, 0.6, 1.2, 2.0, 1.1, 2.5, 5.7, 6.8,
    };
    DualList lists(6.01, 6.01, 0.4);
    lists.update(wrapped.data(), 3, box, true);
    auto expected = lists.long_pairs();
    std::vector<double> drifted = wrapped;
    drifted[0] += 3.0 * box[0];
    drifted[1] += 3.0 * box[1];
    drifted[2] += 3.0 * box[2];
    lists.update(drifted.data(), 3, box, true);
    check(!lists.short_rebuilt() && !lists.long_rebuilt(),
          "whole-cell drift stays inside the DualList skin");
    const auto &got = lists.long_pairs();
    check(got.size() == expected.size(), "whole-cell drift keeps the DualList pair count");
    if (got.size() == expected.size()) {
      auto less = [](const rgpot::lr::Pair &a, const rgpot::lr::Pair &b) {
        return std::tie(a.i, a.j, a.x, a.y, a.z) <
               std::tie(b.i, b.j, b.x, b.y, b.z);
      };
      std::sort(expected.begin(), expected.end(), less);
      auto sorted = got;
      std::sort(sorted.begin(), sorted.end(), less);
      for (std::size_t k = 0; k < sorted.size(); ++k) {
        check(sorted[k].i == expected[k].i && sorted[k].j == expected[k].j,
              "whole-cell drift keeps the DualList atom pairs");
        check_close(sorted[k].x, expected[k].x, 1e-12,
                    "whole-cell drift keeps the DualList x vector");
        check_close(sorted[k].y, expected[k].y, 1e-12,
                    "whole-cell drift keeps the DualList y vector");
        check_close(sorted[k].z, expected[k].z, 1e-12,
                    "whole-cell drift keeps the DualList z vector");
      }
    }
  }
  {
    // Clearing the lists for no atoms must not leave a builder that
    // reports the next geometry as unchanged.
    const double box[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
    const std::vector<double> pos = {0.5, 0.5, 0.5, 1.6, 0.7, 0.4, 7.6, 0.4, 0.6};
    DualList lists(2.0, 3.0, 0.2);
    lists.update(pos.data(), 3, box, true);
    const std::size_t n_short = lists.short_pairs().size();
    const std::size_t n_long = lists.long_pairs().size();
    check(n_short > 0 && n_long > n_short, "three atoms have short and long pairs");
    lists.update(nullptr, 0, box, true);
    check(lists.short_pairs().empty() && lists.long_pairs().empty(), "no atoms have no pairs");
    lists.update(pos.data(), 3, box, true);
    check(lists.short_pairs().size() == n_short && lists.long_pairs().size() == n_long,
          "the same atoms after an empty update get their pairs back");
    check(lists.short_rebuilt() && lists.long_rebuilt(),
          "the update after an empty one rebuilds");
    lists.update(pos.data(), 1, nullptr, false);
    check(lists.long_pairs().empty(), "one molecular atom has no pairs");
    lists.update(pos.data(), 3, box, true);
    check(lists.short_pairs().size() == n_short && lists.long_pairs().size() == n_long,
          "the same atoms after a one-atom update get their pairs back");
  }
  for (const double skin : {std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN(), -0.1}) {
    std::string message;
    try {
      DualList bad(1.0, 2.0, skin);
    } catch (const std::invalid_argument &ex) {
      message = ex.what();
    }
    check(message == "skin must be finite and non-negative",
          "a non-finite or negative skin is rejected by name");
  }
}

// Every periodic image of the cell as an atom of a supercell, m cells on
// each side.
void supercell(const std::vector<double> &pos, const std::vector<int> &z, const double *box,
               int m, std::vector<double> *spos, std::vector<int> *sz, double *sbox) {
  spos->clear();
  sz->clear();
  for (int a = 0; a < m; ++a) {
    for (int b = 0; b < m; ++b) {
      for (int c = 0; c < m; ++c) {
        for (std::size_t i = 0; i < z.size(); ++i) {
          for (int k = 0; k < 3; ++k) {
            spos->push_back(pos[3 * i + static_cast<std::size_t>(k)] + a * box[k] +
                            b * box[3 + k] + c * box[6 + k]);
          }
          sz->push_back(z[i]);
        }
      }
    }
  }
  for (int k = 0; k < 9; ++k) {
    sbox[k] = m * box[k];
  }
}

struct Native {
  double energy = 0;
  std::vector<double> f;
  double strain[6]{};
};

// A cell shorter than the cutoffs, so an atom sees its own images. A
// supercell of m cells a side is wider than twice every cutoff and has no
// self image: each image is an atom and its pairs are counted directly. The
// cell energy is then the supercell energy over m^3, which counts +S and -S
// for every self image.
void test_native_dispersion_images() {
  constexpr double cut2 = 7.3;
  constexpr double cut3 = 5.1;
  constexpr double cut_cn = 5.1;
  const double reach = std::max(cut2, std::max(cut3, cut_cn));
  auto d3 = [&](const std::vector<double> &pos, const std::vector<int> &z, const double *box) {
    rgpot::lr::D3BjParam param;
    rgpot::lr::d3bj_param("pbe", &param);
    DualList lists(2.0, reach, 0.0);
    lists.update(pos.data(), z.size(), box, true);
    Native out;
    out.f.assign(pos.size(), 0.0);
    rgpot::lr::d3bj_evaluate(pos.data(), z.data(), z.size(), box, true, lists.long_pairs(), param,
                             cut2, cut3, cut_cn, &out.energy, out.f.data(), out.strain);
    return out;
  };
  auto d4 = [&](const std::vector<double> &pos, const std::vector<int> &z, const double *box) {
    rgpot::lr::D4BjParam param;
    rgpot::lr::d4bj_param("pbe", &param);
    Native out;
    out.f.assign(pos.size(), 0.0);
    rgpot::lr::d4_evaluate(pos.data(), z.data(), z.size(), box, true, 0.0, param, cut2, cut3,
                           cut_cn, &out.energy, out.f.data(), out.strain);
    return out;
  };
  auto net_force = [](const Native &e) {
    double s[3] = {};
    for (std::size_t i = 0; i < e.f.size(); ++i) {
      s[i % 3] += e.f[i];
    }
    return std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
  };
  struct Model {
    const char *name;
    std::function<Native(const std::vector<double> &, const std::vector<int> &, const double *)>
        eval;
    int m;
    double energy_tol;
  };
  // D4 solves the supercell charges by conjugate gradient on a real-space
  // sum cut at about 0.45 of the supercell, and the cell charges by a dense
  // lattice factorization. Nine cells a side puts that cut far enough out
  // that the two agree to 1e-8.
  const Model models[] = {{"D3", d3, 6, 1e-10}, {"D4", d4, 9, 1e-8}};
  const double cell[9] = {2.6, 0.0, 0.0, 0.4, 2.7, 0.0, 0.3, 0.2, 2.8};
  const std::vector<double> one_pos = {0.1, 0.2, 0.3};
  const std::vector<int> one_z = {6};
  const std::vector<double> two_pos = {0.1, 0.2, 0.3, 1.2, 0.5, 0.45};
  const std::vector<int> two_z = {6, 1};
  for (const Model &model : models) {
    for (const auto *system : {&one_pos, &two_pos}) {
      const std::vector<double> &pos = *system;
      const std::vector<int> &z = system == &one_pos ? one_z : two_z;
      const std::string tag = std::string(model.name) + (system == &one_pos ? " one" : " two") +
                              "-atom periodic cell";
      const Native e = model.eval(pos, z, cell);
      std::vector<double> spos;
      std::vector<int> sz;
      double sbox[9];
      supercell(pos, z, cell, model.m, &spos, &sz, sbox);
      const Native ref = model.eval(spos, sz, sbox);
      const double cells = static_cast<double>(model.m * model.m * model.m);
      std::cout << "CASE " << tag << ' ' << e.energy << ' ' << ref.energy / cells << '\n';
      check(e.energy < 0.0, (tag + " energy is negative").c_str());
      check_close(e.energy * cells, ref.energy, model.energy_tol,
                  (tag + " energy matches the supercell sum").c_str());
      check(net_force(e) < 1e-10, (tag + " net force is zero").c_str());
      for (std::size_t i = 0; i < pos.size(); ++i) {
        const double fd = fd_force(
            [&](const std::vector<double> &p) { return model.eval(p, z, cell).energy; }, pos, i,
            1e-4);
        check_close(e.f[i], fd, 1e-6, (tag + " force matches a finite difference").c_str());
        // Atom i of the cell is atom i of the first supercell cell.
        check_close(e.f[i], ref.f[i], 1e-6, (tag + " force matches the supercell").c_str());
      }
      if (std::string(model.name) == "D3") {
        for (int k = 0; k < 6; ++k) {
          check_close(e.strain[k] * cells, ref.strain[k], 1e-9,
                      (tag + " virial matches the supercell").c_str());
        }
      }
    }
  }
}

void test_madelung() {
  const double ke = rgpot::units::HARTREE_TO_EV * rgpot::units::BOHR_TO_ANGSTROM;
  const double a = 5.64;
  const double cubic[9] = {a, 0, 0, 0, a, 0, 0, 0, a};
  const double triclinic[9] = {a, 0, 0, a, a, 0, 0, a, a};
  std::vector<double> pos;
  std::vector<int> z(8, 11);
  std::vector<double> q;
  const double na[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  const double cl[4][3] = {{0.5, 0, 0}, {0, 0.5, 0}, {0, 0, 0.5}, {0.5, 0.5, 0.5}};
  for (const auto &p : na) {
    pos.insert(pos.end(), {p[0] * a, p[1] * a, p[2] * a});
    q.push_back(1.0);
  }
  for (const auto &p : cl) {
    pos.insert(pos.end(), {p[0] * a, p[1] * a, p[2] * a});
    q.push_back(-1.0);
    z[q.size() - 1] = 17;
  }
  CoulombConfig cfg;
  cfg.method = CoulombMethod::Ewald;
  cfg.cutoff = 5.5;
  cfg.alpha = 1.0;
  cfg.kmax = 16;
  cfg.short_cutoff = 3.0;
  cfg.skin = 0.2;
  cfg.periodic = true;
  std::vector<double> hq;
  double htotal = 1.0;
  rgpot::ForceInput hin{q.size(), pos.data(), z.data(), cubic};
  rgpot::lr::FnCharges hooked(
      [](const rgpot::ForceInput &in, double *qq, double *total, void *) {
        if (in.nAtoms >= 2) {
          qq[0] = 1.0;
          qq[1] = -1.0;
        }
        *total = 0.0;
      },
      nullptr);
  hooked.load(hin, &hq, &htotal);
  check(hq.size() == q.size() && htotal == 0.0, "a charge callback fills the mesh");
  // M. L. Glasser and I. J. Zucker, Theoretical Chemistry: Advances and
  // Perspectives, volume 5, 1980, pages 67 to 139, gives 1.747565 for NaCl.
  constexpr double madelung_nacl = 1.747565;
  const double expect = -4.0 * madelung_nacl * ke / (0.5 * a);
  auto check_nacl = [&](const double *box, double cutoff, double alpha, const char *what) {
    cfg.cutoff = cutoff;
    cfg.alpha = alpha;
    CoulombPot pot(cfg, std::make_shared<FixedCharges>(q, 0.0));
    check(pot.caps().periodic && pot.caps().stress, "coulomb caps report periodic stress");
    const auto key = pot.paramsKey();
    check(key == pot.paramsKey(), "coulomb params key is stable");
    const Eval e = run(pot, pos, z, box);
    std::cout << "CASE " << what << ' ' << e.energy << '\n';
    check_close(e.energy, expect, 2e-6, what);
  };
  check_nacl(cubic, 5.5, 1.0, "NaCl Madelung below cell edge");
  check_nacl(cubic, 18.0, 0.25, "NaCl Madelung above cell edge");
  check_nacl(triclinic, 18.0, 0.25, "NaCl Madelung triclinic cell");

  std::vector<double> cs = {0, 0, 0, 0.5 * a, 0.5 * a, 0.5 * a};
  std::vector<int> z2 = {55, 17};
  cfg.cutoff = 18.0;
  cfg.alpha = 0.25;
  CoulombPot cscl(cfg, std::make_shared<FixedCharges>(std::vector<double>{1.0, -1.0}, 0.0));
  const Eval e2 = run(cscl, cs, z2, cubic);
  const double Mcs = 1.762674773070988;
  const double rnn = a * std::sqrt(3.0) / 2.0;
  check_close(e2.energy, -Mcs * ke / rnn, 2e-5, "CsCl Madelung");
}

void test_dispersion_lattice_sum() {
  const double a = 3.0;
  const double primitive_box[9] = {a, 0, 0, 0, a, 0, 0, 0, a};
  const std::vector<double> primitive_pos = {0, 0, 0};
  const std::vector<int> primitive_z = {6};
  DispersionConfig cfg;
  cfg.kind = DispersionKind::QDO;
  cfg.cutoff = 7.1;
  cfg.short_cutoff = 3.0;
  cfg.periodic = true;
  cfg.qdo_damped = false;
  DispersionPot primitive(cfg);
  const Eval lattice = run(primitive, primitive_pos, primitive_z, primitive_box);

  constexpr int repeat = 5;
  std::vector<double> explicit_pos;
  std::vector<int> explicit_z;
  for (int i = 0; i < repeat; ++i) {
    for (int j = 0; j < repeat; ++j) {
      for (int k = 0; k < repeat; ++k) {
        explicit_pos.insert(explicit_pos.end(),
                            {a * static_cast<double>(i), a * static_cast<double>(j),
                             a * static_cast<double>(k)});
        explicit_z.push_back(6);
      }
    }
  }
  const double extent = a * static_cast<double>(repeat);
  const double explicit_box[9] = {extent, 0, 0, 0, extent, 0, 0, 0, extent};
  DispersionPot explicit_supercell(cfg);
  const Eval direct = run(explicit_supercell, explicit_pos, explicit_z, explicit_box);
  const double per_cell = direct.energy / static_cast<double>(explicit_z.size());
  check_close(lattice.energy, per_cell, 2e-12,
              "QDO lattice sum matches an explicit supercell");

  const double triclinic[9] = {5.0, 0.2, 0.1, 1.1, 4.8, 0.3, 0.4, 0.7, 5.2};
  std::vector<double> pair_pos = {0.4, 0.7, 0.9, 2.1, 1.4, 1.8};
  const std::vector<int> pair_z = {6, 8};
  DispersionPot pair(cfg);
  const Eval pair_eval = run(pair, pair_pos, pair_z, triclinic);
  for (std::size_t i = 0; i < pair_pos.size(); ++i) {
    const double force = fd_force(
        [&](const std::vector<double> &p) { return run(pair, p, pair_z, triclinic).energy; },
        pair_pos, i, 1e-6);
    check_close(pair_eval.f[i], force, 2e-5, "periodic QDO force");
  }
}

void test_forces_and_stress() {
  const double box[9] = {6, 0.3, 0.1, -0.2, 5.5, 0.2, 0.1, -0.4, 6.2};
  std::vector<double> pos = {0.4, 0.5, 0.6, 2.2, 1.1, 0.8, 1.0, 3.3, 2.4, 4.2, 2.2, 3.5};
  std::vector<int> z = {1, 1, 8, 8};
  std::vector<double> q = {0.4, -0.2, 0.5, -0.7};
  CoulombConfig cfg;
  cfg.method = CoulombMethod::Ewald;
  cfg.cutoff = 8.0;
  cfg.alpha = 0.45;
  cfg.kmax = 8;
  cfg.short_cutoff = 2.5;
  cfg.periodic = true;
  auto pot = std::make_shared<CoulombPot>(cfg, std::make_shared<FixedCharges>(q, 0.0));
  const Eval e0 = run(*pot, pos, z, box);
  for (std::size_t i = 0; i < pos.size(); ++i) {
    const double f = fd_force(
        [&](const std::vector<double> &p) { return run(*pot, p, z, box).energy; }, pos, i, 1e-5);
    check_close(e0.f[i], f, 2e-5, "Ewald force");
  }
  check(e0.has_stress, "Ewald stress is set");
  const double eps = 1e-6;
  auto strain_xx = [&](double eps_xx) {
    std::vector<double> moved = pos;
    double box2[9];
    std::copy(box, box + 9, box2);
    for (int row = 0; row < 3; ++row) {
      box2[3 * row] *= (1.0 + eps_xx);
    }
    for (std::size_t i = 0; i < moved.size(); i += 3) {
      moved[i] *= (1.0 + eps_xx);
    }
    return run(*pot, moved, z, box2).energy;
  };
  const double dE = (strain_xx(eps) - strain_xx(-eps)) / (2.0 * eps);
  const double volume = std::abs(cell_from_box(box).volume);
  check_close(e0.stress[0], dE / volume, 5e-4, "Ewald xx stress");

  cfg.method = CoulombMethod::SPME;
  cfg.grid[0] = cfg.grid[1] = cfg.grid[2] = 128;
  cfg.spline_order = 6;
  CoulombPot spme(cfg, std::make_shared<FixedCharges>(q, 0.0));
  const Eval es = run(spme, pos, z, box);
  std::cout << "CASE ewald_cluster " << e0.energy << '\n';
  std::cout << "CASE spme_cluster " << es.energy << '\n';
  check_close(es.energy, e0.energy, 2e-3, "SPME matches Ewald");
  for (std::size_t i = 0; i < pos.size(); ++i) {
    const double f = fd_force(
        [&](const std::vector<double> &p) { return run(spme, p, z, box).energy; }, pos, i, 1e-5);
    check_close(es.f[i], f, 5e-4, "SPME force");
  }
}

void test_dsf() {
  std::vector<double> pos = {0, 0, 0, 1.2, 0.2, -0.1, 2.5, 1.0, 0.4};
  std::vector<int> z = {1, 1, 8};
  std::vector<double> q = {0.3, -0.4, 0.1};
  CoulombConfig cfg;
  cfg.method = CoulombMethod::DSF;
  cfg.periodic = false;
  cfg.cutoff = 8.0;
  cfg.alpha = 0.25;
  cfg.short_cutoff = 3.0;
  CoulombPot pot(cfg, std::make_shared<FixedCharges>(q, 0.0));
  const Eval e0 = run(pot, pos, z, nullptr);
  for (std::size_t i = 0; i < pos.size(); ++i) {
    const double f = fd_force(
        [&](const std::vector<double> &p) { return run(pot, p, z, nullptr).energy; }, pos, i, 1e-6);
    check_close(e0.f[i], f, 2e-6, "DSF force");
  }
  double ene = 0, dedr = 0;
  so3lr_erf_pair(0.2, -0.3, 1.4, 2.0, 6.0, 9.0, 14.399645351950548, &ene, &dedr);
  check(std::isfinite(ene) && dedr != 0.0, "SO3LR erf pair is finite");
  const double step = 1e-6;
  double ep = 0, em = 0, dummy = 0;
  so3lr_erf_pair(0.2, -0.3, 1.4 + step, 2.0, 6.0, 9.0, 14.399645351950548, &ep, &dummy);
  so3lr_erf_pair(0.2, -0.3, 1.4 - step, 2.0, 6.0, 9.0, 14.399645351950548, &em, &dummy);
  check_close(dedr, (ep - em) / (2.0 * step), 1e-6, "SO3LR erf derivative");
}

void test_public_null_cell_calls() {
  const std::vector<double> pos = {0, 0, 0, 1.2, 0.2, -0.1, 2.5, 1.0, 0.4};
  const std::vector<int> z = {1, 1, 8};

  DispersionConfig dcfg;
  dcfg.kind = DispersionKind::QDO;
  dcfg.periodic = false;
  dcfg.cutoff = 12;
  dcfg.short_cutoff = 3;
  DispersionPot dispersion(dcfg);
  check(std::isfinite(run_batch(dispersion, pos, z, nullptr).energy),
        "public dispersion batch accepts a molecule");

  NlhPot nlh;
  check(std::isfinite(run_batch(nlh, pos, z, nullptr).energy),
        "public NLH batch accepts a molecule");

  CoulombConfig ccfg;
  ccfg.method = CoulombMethod::DSF;
  ccfg.periodic = false;
  ccfg.cutoff = 8;
  ccfg.alpha = 0.25;
  ccfg.short_cutoff = 3;
  const auto charges =
      std::make_shared<FixedCharges>(std::vector<double>{0.3, -0.4, 0.1}, 0.0);
  CoulombPot coulomb(ccfg, charges);
  check(std::isfinite(run_batch(coulomb, pos, z, nullptr).energy),
        "public Coulomb batch accepts a molecule");

  ccfg.method = CoulombMethod::Ewald;
  ccfg.periodic = true;
  CoulombPot ewald(ccfg, charges);
  bool threw = false;
  try {
    run_batch(ewald, pos, z, nullptr);
  } catch (const std::invalid_argument &ex) {
    threw = std::string(ex.what()) == "invalid parameter: box is null";
  }
  check(threw, "public periodic Ewald batch requires a cell");
}

void test_qdo_nlh_mbd() {
  std::vector<double> pos = {0, 0, 0, 1.4, 0.1, 0.0, 0.2, 1.6, 0.3};
  std::vector<int> z = {6, 1, 8};
  DispersionConfig cfg;
  cfg.kind = DispersionKind::QDO;
  cfg.cutoff = 12;
  cfg.short_cutoff = 3;
  cfg.periodic = false;
  cfg.qdo_damped = true;
  DispersionPot qdo(cfg, std::make_shared<FixedRatios>(std::vector<double>{1.0, 0.8, 1.1},
                                                           std::vector<double>{}));
  const Eval e0 = run(qdo, pos, z, nullptr);
  std::cout << "CASE qdo_cho " << e0.energy << '\n';
  check(e0.energy < 0.0, "QDO dispersion is attractive");
  for (std::size_t i = 0; i < pos.size(); ++i) {
    const double f = fd_force(
        [&](const std::vector<double> &p) { return run(qdo, p, z, nullptr).energy; }, pos, i, 1e-6);
    check_close(e0.f[i], f, 2e-5, "QDO force");
  }
  DispersionPot bare(cfg, std::make_shared<FixedRatios>(std::vector<double>{1, 1, 1},
                                                            std::vector<double>{1, 1, 1}));
  check(std::abs(run(bare, pos, z, nullptr).energy - e0.energy) > 1e-6, "C6 ratios change the energy");

  NlhPot nlh;
  const Eval en = run(nlh, pos, z, nullptr);
  check(en.energy > 0.0, "NLH is repulsive");
  for (std::size_t i = 0; i < 3; ++i) {
    const double f = fd_force(
        [&](const std::vector<double> &p) { return run(nlh, p, z, nullptr).energy; }, pos, i, 1e-6);
    check_close(en.f[i], f, 2e-5, "NLH force");
  }

  DispersionConfig mcfg;
  mcfg.kind = DispersionKind::MBD;
  mcfg.periodic = false;
  mcfg.cutoff = 20;
  mcfg.short_cutoff = 4;
  mcfg.mbd_beta = 0.83;
  DispersionPot mbd(mcfg);
  const Eval em = run(mbd, pos, z, nullptr);
  std::cout << "CASE mbd_cho " << em.energy << '\n';
  check(std::isfinite(em.energy), "MBD energy is finite");
  for (std::size_t i = 0; i < 3; ++i) {
    const double f = fd_force(
        [&](const std::vector<double> &p) { return run(mbd, p, z, nullptr).energy; }, pos, i, 2e-4);
    check_close(em.f[i], f, 5e-3, "MBD force");
  }
}

class Bias final : public rgpot::Potential<Bias> {
public:
  static constexpr bool kCellOptional = true;

  Bias() : rgpot::Potential<Bias>(rgpot::PotType::UNKNOWN) {}
  void forceImpl(const ForceInput &in, ForceOut *out) const override {
    out->energy = 0.5;
    for (std::size_t i = 0; i < in.nAtoms; ++i) {
      out->F[3 * i] += 0.2;
    }
    out->has_stress = 1;
    out->variance = 0.1;
  }
  [[nodiscard]] rgpot::PotCaps caps() const noexcept override {
    return {.stress = true};
  }
};

void test_library_dispersion() {
#if defined(RGPOT_HAS_DFTD3)
  {
  std::vector<double> pos = {0, 0, 0, 0.96, 0, 0, -0.24, 0.93, 0};
  std::vector<int> z = {8, 1, 1};
  rgpot::lr::DispersionConfig cfg;
  cfg.kind = rgpot::lr::DispersionKind::D3BJ;
  cfg.functional = "pbe";
  cfg.atm = true;
  cfg.periodic = false;
  cfg.cutoff = 20;
  cfg.short_cutoff = 4;
  rgpot::lr::DispersionPot d3(cfg);
  const Eval e = run(d3, pos, z, nullptr);
  std::cout << "CASE d3_water " << e.energy;
  for (double f : e.f) {
    std::cout << ' ' << f;
  }
  std::cout << '\n';
  check(e.energy < 0.0, "D3 energy is negative");
  rgpot::lr::D3BjParam blyp;
  rgpot::lr::D3BjParam b3lyp;
  check(rgpot::lr::d3bj_param("blyp", &blyp) && rgpot::lr::d3bj_param("b3lyp", &b3lyp),
        "BLYP and B3LYP D3 parameters");
  check(!rgpot::lr::d3bj_param(nullptr, &blyp), "a null D3 functional is rejected");
  cfg.functional = "blyp";
  cfg.atm = false;
  rgpot::lr::DispersionPot blyp_pot(cfg);
  const Eval blyp_e = run(blyp_pot, pos, z, nullptr);
  check(std::isfinite(blyp_e.energy), "BLYP D3 without ATM is finite");
  cfg.functional = "pbe";
  cfg.atm = true;
  bool threw = false;
  try {
    rgpot::lr::DispersionPot badz(cfg);
    std::vector<double> one = {0, 0, 0};
    std::vector<int> zz = {200};
    run(badz, one, zz, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "D3 rejects an atomic number outside the table");
  {
    // Hydrogen first, so the centre of the pair has the larger atomic number
    // and the van der Waals table swaps its arguments. A short cell adds a
    // self image, which is the repeated-index three-body weight.
    rgpot::lr::DispersionConfig img;
    img.kind = rgpot::lr::DispersionKind::D3BJ;
    img.functional = "pbe";
    img.atm = true;
    img.periodic = true;
    img.cutoff = 8;
    img.short_cutoff = 2;
    img.lib_disp2 = 12;
    img.lib_disp3 = 12;
    img.lib_cn = 12;
    const double ibox[9] = {6, 0, 0, 0, 6, 0, 0, 0, 6};
    std::vector<double> ipos = {1, 1, 1, 3.2, 1.2, 1.1};
    std::vector<int> iz = {1, 8};
    rgpot::lr::DispersionPot ipot(img);
    const Eval ie = run(ipot, ipos, iz, ibox);
    // Three cells a side are wider than twice every cutoff, so the
    // supercell counts each image of the cell as an atom.
    std::vector<double> spos;
    std::vector<int> sz;
    double sbox[9];
    supercell(ipos, iz, ibox, 3, &spos, &sz, sbox);
    rgpot::lr::DispersionPot spot(img);
    const Eval se = run(spot, spos, sz, sbox);
    std::cout << "CASE d3_self_image " << ie.energy << ' ' << se.energy / 27.0 << '\n';
    check(ie.energy < 0.0, "periodic D3 with a self image is negative");
    check_close(ie.energy * 27.0, se.energy, 1e-10,
                "periodic D3 with a self image matches the supercell sum");
    double net[3] = {};
    for (std::size_t i = 0; i < ipos.size(); ++i) {
      net[i % 3] += ie.f[i];
      const double fd = fd_force(
          [&](const std::vector<double> &p) { return run(ipot, p, iz, ibox).energy; }, ipos, i,
          1e-4);
      check_close(ie.f[i], fd, 1e-6, "periodic D3 with a self image force");
      check_close(ie.f[i], se.f[i], 1e-8, "periodic D3 with a self image force matches the supercell");
    }
    check(std::abs(net[0]) + std::abs(net[1]) + std::abs(net[2]) < 1e-10,
          "periodic D3 with a self image has no net force");
  }
  for (std::size_t i = 0; i < 3; ++i) {
    const double f = fd_force(
        [&](const std::vector<double> &p) { return run(d3, p, z, nullptr).energy; }, pos, i, 1e-6);
    check_close(e.f[i], f, 2e-5, "D3 force");
  }
  check_periodic_dispersion_self_images(DispersionKind::D3BJ, "d3", true);
  }
#else
  std::cout << "CASE d3_water skipped\n";
#endif
#if defined(RGPOT_HAS_DFTD4)
  {
  std::vector<double> pos = {0, 0, 0, 0.96, 0, 0, -0.24, 0.93, 0};
  std::vector<int> z = {8, 1, 1};
  rgpot::lr::DispersionConfig cfg;
  cfg.kind = rgpot::lr::DispersionKind::D4;
  cfg.functional = "pbe";
  cfg.atm = true;
  cfg.charge = 0.0;
  cfg.periodic = false;
  cfg.cutoff = 20;
  cfg.short_cutoff = 4;
  rgpot::lr::DispersionPot d4(cfg);
  const Eval e = run(d4, pos, z, nullptr);
  std::cout << "CASE d4_water " << e.energy;
  for (double f : e.f) {
    std::cout << ' ' << f;
  }
  std::cout << '\n';
  check(e.energy < 0.0, "D4 energy is negative");
  rgpot::lr::D4BjParam bj;
  check(rgpot::lr::d4bj_param("pbe", &bj), "PBE D4 parameters");
  check(!rgpot::lr::d4bj_param(nullptr, &bj) && !rgpot::lr::d4bj_param("nope", &bj),
        "an unknown D4 parameter name is rejected");
  double native = 0.0;
  double nF[9] = {};
  double nstrain[6] = {};
  const double bohr = 0.529177210903;
  rgpot::lr::d4_evaluate(pos.data(), z.data(), pos.size() / 3, nullptr, false, 0.0, bj, 60.0 * bohr,
                         40.0 * bohr, 30.0 * bohr, &native, nF, nstrain);
  check_close(native, e.energy, 1e-8, "native D4 matches dftd4 on water");
  // The native periodic D4 path does not currently calculate strain.
  check_periodic_dispersion_self_images(DispersionKind::D4, "d4", false);
  }
#else
  std::cout << "CASE d4_water skipped\n";
#endif
#if defined(RGPOT_HAS_DFTD4)
  {
    rgpot::lr::DispersionConfig cfg;
    cfg.kind = rgpot::lr::DispersionKind::D4;
    cfg.functional = "pbe";
    cfg.atm = true;
    cfg.periodic = true;
    cfg.cutoff = 8.0;
    cfg.short_cutoff = 3.0;
    cfg.lib_disp2 = 15.0;
    cfg.lib_disp3 = 10.0;
    cfg.lib_cn = 10.0;
    const double box[9] = {12, 0, 0, 0, 12, 0, 0, 0, 12};
    std::vector<double> pos;
    std::vector<int> z;
    for (int i = 0; i < 8; ++i) {
      pos.push_back(1.0 + (i % 2) * 4.0);
      pos.push_back(1.0 + ((i / 2) % 2) * 4.0);
      pos.push_back(1.0 + (i / 4) * 4.0);
      z.push_back(i % 2 == 0 ? 8 : 1);
    }
    rgpot::lr::DispersionPot pot(cfg);
    const Eval e = run(pot, pos, z, box);
    check(e.energy < 0.0, "periodic D4 energy is negative");
    // Two cells a side hold 64 atoms, so both use the dense lattice charge
    // solve, and the supercell is wider than twice every cutoff.
    std::vector<double> spos;
    std::vector<int> sz;
    double sbox[9];
    supercell(pos, z, box, 2, &spos, &sz, sbox);
    rgpot::lr::DispersionPot spot(cfg);
    const Eval se = run(spot, spos, sz, sbox);
    std::cout << "CASE d4_periodic " << e.energy << ' ' << se.energy / 8.0 << '\n';
    check_close(e.energy * 8.0, se.energy, 1e-9, "periodic D4 matches the supercell sum");
    double net[3] = {};
    for (std::size_t i = 0; i < pos.size(); ++i) {
      net[i % 3] += e.f[i];
      check_close(e.f[i], se.f[i], 1e-8, "periodic D4 force matches the supercell");
    }
    check(std::abs(net[0]) + std::abs(net[1]) + std::abs(net[2]) < 1e-9,
          "periodic D4 has no net force");
    for (std::size_t i : {std::size_t{0}, std::size_t{4}, std::size_t{11}}) {
      const double fd = fd_force(
          [&](const std::vector<double> &p) { return run(pot, p, z, box).energy; }, pos, i, 1e-4);
      check_close(e.f[i], fd, 1e-6, "periodic D4 force matches a finite difference");
    }
    cfg.atm = false;
    rgpot::lr::DispersionPot noatm(cfg);
    const Eval bare = run(noatm, pos, z, box);
    rgpot::lr::DispersionPot snoatm(cfg);
    const Eval sbare = run(snoatm, spos, sz, sbox);
    check_close(bare.energy * 8.0, sbare.energy, 1e-9,
                "periodic D4 without the three-body term matches the supercell sum");
    bool threw = false;
    try {
      std::vector<int> badz = z;
      badz[0] = 200;
      run(pot, pos, badz, box);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    check(threw, "periodic D4 rejects an atomic number outside the table");
    threw = false;
    try {
      rgpot::lr::DispersionConfig bad = cfg;
      bad.atm = true;
      bad.functional = "not-a-functional";
      rgpot::lr::DispersionPot badpot(bad);
      run(badpot, pos, z, box);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    check(threw, "periodic D4 rejects an unknown functional");
    std::vector<double> many(3 * 100);
    std::vector<int> mz(100, 6);
    for (int i = 0; i < 100; ++i) {
      many[static_cast<std::size_t>(3 * i)] = 1.2 * i;
      many[static_cast<std::size_t>(3 * i + 1)] = 0.1;
      many[static_cast<std::size_t>(3 * i + 2)] = 0.2;
    }
    double genergy = 0.0;
    std::vector<double> gF(many.size());
    double gstrain[6] = {};
    rgpot::lr::D4BjParam gp;
    rgpot::lr::d4bj_param("pbe", &gp);
    rgpot::lr::d4_evaluate(many.data(), mz.data(), 100, nullptr, false, 0.0, gp, 20, 15, 15, &genergy,
                           gF.data(), gstrain);
    check(std::isfinite(genergy), "a gas-phase D4 above the dense cutoff is finite");
  }
#endif
}

void test_sum_and_charge() {
  std::vector<double> pos = {0, 0, 0, 2, 0, 0};
  std::vector<int> z = {1, 1};
  CoulombConfig cfg;
  cfg.method = CoulombMethod::DSF;
  cfg.periodic = false;
  cfg.cutoff = 6;
  cfg.alpha = 0.2;
  cfg.short_cutoff = 3;
  auto coul = std::make_unique<CoulombPot>(cfg, std::make_shared<FixedCharges>(std::vector<double>{0.5, -0.5}, 0.0));
  const Eval alone = run(*coul, pos, z, nullptr);
  SumPot sum;
  sum.add(std::move(coul));
  sum.add(std::make_unique<Bias>());
  const Eval both = run(sum, pos, z, nullptr);
  check_close(both.energy, alone.energy + 0.5, 1e-12, "sum energy");
  check_close(both.f[0], alone.f[0] + 0.2, 1e-12, "sum force");
  check_close(both.f[1], alone.f[1], 1e-12, "sum leaves the other component");
  bool threw = false;
  try {
    CoulombPot bad(cfg, std::make_shared<FixedCharges>(std::vector<double>{0.5, -0.2}, 0.0));
    run(bad, pos, z, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "total charge is checked");
}

void test_nve() {
  const double a = 5.64;
  const double box[9] = {a, 0, 0, 0, a, 0, 0, 0, a};
  std::vector<double> pos;
  std::vector<double> q;
  std::vector<int> z;
  const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  for (const auto &p : basis) {
    pos.insert(pos.end(), {p[0] * a, p[1] * a, p[2] * a});
    q.push_back(1);
    z.push_back(11);
  }
  for (const auto &p : basis) {
    pos.insert(pos.end(), {(p[0] + 0.5) * a, p[1] * a, p[2] * a});
    q.push_back(-1);
    z.push_back(17);
  }
  // The second loop is not CsCl/NaCl. Use the NaCl decoration instead.
  pos.clear();
  q.clear();
  z.clear();
  const double na[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  const double cl[4][3] = {{0.5, 0, 0}, {0, 0.5, 0}, {0, 0, 0.5}, {0.5, 0.5, 0.5}};
  for (const auto &p : na) {
    pos.insert(pos.end(), {p[0] * a, p[1] * a, p[2] * a});
    q.push_back(1);
    z.push_back(23);
  }
  for (const auto &p : cl) {
    pos.insert(pos.end(), {p[0] * a, p[1] * a, p[2] * a});
    q.push_back(-1);
    z.push_back(17);
  }
  CoulombConfig cfg;
  cfg.cutoff = 8;
  cfg.alpha = 0.35;
  cfg.kmax = 6;
  cfg.short_cutoff = 3;
  cfg.skin = 0.5;
  CoulombPot pot(cfg, std::make_shared<FixedCharges>(q, 0.0));
  std::vector<double> vel(pos.size(), 0.0);
  vel[0] = 0.002;
  vel[4] = -0.001;
  const double mass = 22.99;
  const double conv = 0.009648533212;
  const double dt = 1.0;
  auto energy = [&]() {
    const Eval e = run(pot, pos, z, box);
    double ke = 0.0;
    for (std::size_t i = 0; i < vel.size(); ++i) {
      ke += 0.5 * mass * vel[i] * vel[i] / conv;
    }
    return e.energy + ke;
  };
  // velocity unit: Angstrom/fs. ke = 0.5 m v^2 / conv, with conv = a per (F/m).
  double e1 = 0;
  for (int step = 0; step < 40; ++step) {
    const Eval e = run(pot, pos, z, box);
    if (step == 0) {
      double kin = 0;
      for (double v : vel) {
        kin += 0.5 * mass * v * v / conv;
      }
      e1 = e.energy + kin;
    }
    for (std::size_t i = 0; i < pos.size(); ++i) {
      vel[i] += 0.5 * dt * e.f[i] / mass * conv;
      pos[i] += dt * vel[i];
    }
    const Eval e2 = run(pot, pos, z, box);
    for (std::size_t i = 0; i < pos.size(); ++i) {
      vel[i] += 0.5 * dt * e2.f[i] / mass * conv;
    }
  }
  double kin = 0;
  for (double v : vel) {
    kin += 0.5 * mass * v * v / conv;
  }
  const Eval ef = run(pot, pos, z, box);
  check_close(ef.energy + kin, e1, 2e-4, "NVE energy");
  (void)energy;
}

struct Hook final : rgpot::lr::ChargeResponse {
  void accumulate(const ForceInput &in, const double *dE_dq, double *force) const override {
    for (std::size_t i = 0; i < in.nAtoms; ++i) {
      force[3 * i] += 0.01 * dE_dq[i];
    }
  }
};

struct Ratios final : rgpot::lr::RatioResponse {
  void jacobians(const ForceInput &, const double **dh, const double **dc6) const override {
    static double hder[9] = {0.05, 0, 0, 0, 0, 0, 0, 0, 0};
    static double cder[9] = {0, 0.02, 0, 0, 0, 0, 0, 0, 0};
    *dh = hder;
    *dc6 = cder;
  }
};

void fill_charges(const ForceInput &, double *q, double *total, void *) {
  q[0] = 0.25;
  q[1] = -0.25;
  *total = 0.0;
}

void test_edges() {
  bool threw = false;
  try {
    DualList bad(0.0, 1.0, 0.1);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "non-positive cutoff is rejected");
  threw = false;
  try {
    DualList bad(5.0, 1.0, 0.1);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "short cutoff cannot exceed the long cutoff");
  DualList empty(1.0, 2.0, 0.1);
  empty.update(nullptr, 1, nullptr, false);
  check(empty.long_pairs().empty(), "one atom has no pairs");
  threw = false;
  try {
    std::vector<double> pos = {0, 0, 0, 1, 0, 0};
    empty.update(nullptr, 2, nullptr, false);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "null positions are rejected");

  CoulombConfig cfg;
  cfg.method = CoulombMethod::DSF;
  cfg.periodic = false;
  cfg.cutoff = 5;
  cfg.alpha = 0.2;
  cfg.short_cutoff = 2;
  std::vector<double> pos = {0, 0, 0, 1.1, 0.2, 0};
  std::vector<int> z = {1, 1};
  threw = false;
  try {
    CoulombPot pot(cfg, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a charge source is required");
  threw = false;
  try {
    rgpot::lr::FnCharges(nullptr, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a null charge callback is rejected");
  CoulombPot hooked(cfg, std::make_shared<rgpot::lr::FnCharges>(&fill_charges, nullptr),
                    std::make_shared<Hook>());
  const Eval eh = run(hooked, pos, z, nullptr);
  check(std::isfinite(eh.energy), "backend charges evaluate");

  threw = false;
  try {
    CoulombPot pot(cfg, std::make_shared<FixedCharges>(std::vector<double>{1.0}, 1.0));
    run(pot, pos, z, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "charge count is checked");

  double ene = 0, dedr = 0;
  so3lr_erf_pair(1, -1, 12, 2, 4, 8, 1, &ene, &dedr);
  check(ene == 0.0 && dedr == 0.0, "erf pair is zero past the cutoff");
  so3lr_erf_pair(1, -1, 6, 2, 4, 8, 1, &ene, &dedr);
  check(std::isfinite(ene), "erf pair is finite inside the switch");

  cfg.method = CoulombMethod::SPME;
  cfg.periodic = true;
  cfg.spline_order = 2;
  const double box[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
  threw = false;
  try {
    CoulombPot pot(cfg, std::make_shared<FixedCharges>(std::vector<double>{0.2, -0.2}, 0.0));
    run(pot, pos, z, box);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "SPME order below 3 is rejected");
  const double badbox[9] = {1, 0, 0, 2, 0, 0, 3, 0, 0};
  cfg.spline_order = 4;
  cfg.method = CoulombMethod::Ewald;
  cfg.kmax = 2;
  threw = false;
  try {
    CoulombPot pot(cfg, std::make_shared<FixedCharges>(std::vector<double>{0.2, -0.2}, 0.0));
    run(pot, pos, z, badbox);
  } catch (const std::exception &) {
    threw = true;
  }
  check(threw, "a singular cell is rejected");

  auto qdo = rgpot::lr::make_dispersion(rgpot::lr::DispersionKind::QDO);
  auto mbd = rgpot::lr::make_dispersion(rgpot::lr::DispersionKind::MBD);
  auto d3 = rgpot::lr::make_dispersion(rgpot::lr::DispersionKind::D3BJ);
  auto d4 = rgpot::lr::make_dispersion(rgpot::lr::DispersionKind::D4);
  check(std::string(qdo->name()) == "QDO" && qdo->kind() == rgpot::lr::DispersionKind::QDO, "QDO name");
  check(std::string(mbd->name()) == "MBD@rsSCS", "MBD name");
#if defined(RGPOT_HAS_DFTD3)
  check(std::string(d3->name()) == "D3BJ", "D3 library name");
#else
  check(std::string(d3->name()) == "unavailable", "D3 without s-dftd3 is unavailable");
#endif
#if defined(RGPOT_HAS_DFTD4)
  check(std::string(d4->name()) == "D4", "D4 library name");
#else
  check(std::string(d4->name()) == "unavailable", "D4 without dftd4 is unavailable");
#endif

  DispersionConfig dc;
  dc.kind = DispersionKind::QDO;
  dc.qdo_legacy = true;
  dc.qdo_damped = false;
  dc.switch_width = 1.0;
  dc.cutoff = 6;
  dc.short_cutoff = 3;
  dc.periodic = false;
  std::vector<double> tri = {0, 0, 0, 1.5, 0.2, 0.1, 0.3, 1.8, 0.4};
  std::vector<int> z3 = {1, 6, 8};
  DispersionPot legacy(dc, std::make_shared<FixedRatios>(std::vector<double>{1, 1, 1}, std::vector<double>{1, 1, 1}),
                       std::make_shared<Ratios>());
  const Eval el = run(legacy, tri, z3, nullptr);
  check(std::isfinite(el.energy), "legacy undamped QDO with a ratio response");
  const double cell[9] = {12, 0, 0, 0, 12, 0, 0, 0, 12};
  dc.periodic = true;
  DispersionPot periodic(dc);
  const Eval ep = run(periodic, tri, z3, cell);
  check(ep.has_stress, "periodic QDO publishes stress");

  NlhPot nlh;
  const Eval en = run(nlh, tri, z3, cell);
  check(en.has_stress, "NLH publishes stress in a cell");
  threw = false;
  try {
    NlhPot bad(rgpot::lr::NlhConfig{1.0, 0.2, 1.5});
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "NLH cutoff must clear the switch");

  SumPot sum;
  sum.add(std::make_unique<CoulombPot>(cfg, std::make_shared<FixedCharges>(std::vector<double>{0.2, -0.2, 0.0}, 0.0)));
  sum.add(std::make_unique<Bias>());
  DispersionConfig scfg;
  scfg.kind = DispersionKind::QDO;
  scfg.cutoff = 6;
  scfg.short_cutoff = 2;
  sum.add(std::make_unique<DispersionPot>(scfg));
  sum.add(std::make_unique<NlhPot>());
  auto shared = std::make_shared<DualList>(2.0, 5.0, 0.4);
  sum.shareNeighbours(shared);
  const Eval es = run(sum, tri, z3, box);
  check(es.has_stress, "the sum keeps child stress");
  threw = false;
  try {
    sum.add(nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a null summand is rejected");

  DispersionConfig d3cfg;
  d3cfg.kind = DispersionKind::D3BJ;
  d3cfg.functional = "pbe";
  d3cfg.cutoff = 12;
  d3cfg.short_cutoff = 3;
  DispersionPot d3pot(d3cfg);
  check(mbd->kind() == rgpot::lr::DispersionKind::MBD, "MBD kind");
  check(d3->kind() == rgpot::lr::DispersionKind::D3BJ && d4->kind() == rgpot::lr::DispersionKind::D4,
        "library kinds");
  threw = false;
  try {
    DispersionConfig badz;
    badz.kind = DispersionKind::MBD;
    badz.cutoff = 8;
    badz.short_cutoff = 3;
    DispersionPot pot(badz);
    std::vector<int> zz = {0, 1, 1};
    run(pot, tri, zz, nullptr);
  } catch (const std::exception &) {
    threw = true;
  }
  check(threw, "MBD rejects an unknown element");
  threw = false;
  try {
    DispersionConfig badh;
    badh.kind = DispersionKind::MBD;
    badh.cutoff = 8;
    badh.short_cutoff = 3;
    DispersionPot pot(badh, std::make_shared<FixedRatios>(std::vector<double>{0, 1, 1}, std::vector<double>{}));
    run(pot, tri, z3, nullptr);
  } catch (const std::exception &) {
    threw = true;
  }
  check(threw, "MBD rejects a zero ratio");
  threw = false;
  try {
    DispersionConfig qcfg;
    qcfg.kind = DispersionKind::QDO;
    qcfg.cutoff = 8;
    qcfg.short_cutoff = 3;
    DispersionPot pot(qcfg);
    std::vector<int> zz = {200, 1, 1};
    run(pot, tri, zz, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "QDO rejects an unknown element");
  threw = false;
  try {
    DispersionPot pot(dc, std::make_shared<FixedRatios>(std::vector<double>{1.0}, std::vector<double>{}));
    run(pot, tri, z3, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "ratio length is checked");
  threw = false;
  try {
    DispersionPot pot(dc);
    ForceOut out{};
    std::vector<double> f(6, 0);
    out.F = f.data();
    ForceInput in{2, pos.data(), nullptr, nullptr};
    pot.forceImpl(in, &out);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "dispersion requires atomic numbers");
#if defined(RGPOT_HAS_DFTD3)
  threw = false;
  try {
    DispersionConfig badm = d3cfg;
    badm.functional = "not-a-functional";
    DispersionPot pot(badm);
    run(pot, pos, z, nullptr);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  check(threw, "an unknown D3 functional is rejected");
#endif
  threw = false;
  try {
    NlhPot pot;
    ForceOut out{};
    std::vector<double> f(6, 0);
    out.F = f.data();
    ForceInput in{2, pos.data(), nullptr, nullptr};
    pot.forceImpl(in, &out);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "NLH requires atomic numbers");
  threw = false;
  try {
    NlhPot pot;
    std::vector<int> zz = {99, 1};
    run(pot, pos, zz, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "NLH rejects Z beyond uranium");

  std::vector<double> dedq(2);
  rgpot::lr::ElecOut er;
  std::vector<double> ff(6, 0);
  rgpot::lr::CoulombConfig ec = cfg;
  ec.method = CoulombMethod::Ewald;
  ec.kmax = 3;
  ec.cutoff = 6;
  ec.alpha = 0.4;
  std::vector<rgpot::lr::Pair> nopairs;
  rgpot::lr::evaluate_electrostatics(CoulombMethod::Ewald, ec, pos.data(),
                                    std::vector<double>{0.3, -0.3}.data(), 2, 0.0, box, nopairs, ff.data(),
                                    dedq.data(), &er);
  check(std::isfinite(dedq[0]), "Ewald charge derivative");
  ec.method = CoulombMethod::SPME;
  ec.grid[0] = ec.grid[1] = ec.grid[2] = 16;
  ec.spline_order = 4;
  rgpot::lr::evaluate_electrostatics(CoulombMethod::SPME, ec, pos.data(),
                                    std::vector<double>{0.3, -0.3}.data(), 2, 0.0, box, nopairs, ff.data(),
                                    dedq.data(), &er);
  check(std::isfinite(er.energy), "SPME charge derivative");
  threw = false;
  try {
    rgpot::lr::evaluate_electrostatics(CoulombMethod::Ewald, ec, pos.data(), nullptr, 2, 0, box, nopairs,
                                      ff.data(), nullptr, &er);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "null charges are rejected");
  threw = false;
  try {
    std::vector<double> nq = {std::nan(""), 0};
    rgpot::lr::evaluate_electrostatics(CoulombMethod::DSF, ec, pos.data(), nq.data(), 2, 0, nullptr, nopairs,
                                      ff.data(), nullptr, &er);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a non-finite charge is rejected");
  threw = false;
  try {
    ec.alpha = -1;
    rgpot::lr::evaluate_electrostatics(CoulombMethod::DSF, ec, pos.data(),
                                      std::vector<double>{0.1, -0.1}.data(), 2, 0, nullptr, nopairs, ff.data(),
                                      nullptr, &er);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a negative alpha is rejected");
  threw = false;
  try {
    std::vector<double> badq = {0.2, -0.2};
    ec.alpha = 0.3;
    ec.method = CoulombMethod::Ewald;
    rgpot::lr::evaluate_electrostatics(CoulombMethod::Ewald, ec, pos.data(), badq.data(), 2,
                                      std::numeric_limits<double>::infinity(), box, nopairs, ff.data(),
                                      nullptr, &er);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a non-finite total charge is rejected");
  threw = false;
  try {
    ec.kmax = 0;
    rgpot::lr::evaluate_electrostatics(CoulombMethod::Ewald, ec, pos.data(),
                                      std::vector<double>{0.2, -0.2}.data(), 2, 0, box, nopairs,
                                      ff.data(), nullptr, &er);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "Ewald kmax of zero is rejected");
  threw = false;
  try {
    std::vector<double> badpos = {0, 0, std::numeric_limits<double>::quiet_NaN(), 1, 0, 0};
    ec.kmax = 2;
    rgpot::lr::evaluate_electrostatics(CoulombMethod::DSF, ec, badpos.data(),
                                      std::vector<double>{0.2, -0.2}.data(), 2, 0, nullptr, nopairs,
                                      ff.data(), nullptr, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a non-finite position is rejected");
  threw = false;
  try {
    DispersionConfig badc6;
    badc6.kind = DispersionKind::QDO;
    badc6.cutoff = 6;
    badc6.short_cutoff = 2;
    DispersionPot pot(badc6, std::make_shared<FixedRatios>(std::vector<double>{1, 1},
                                                          std::vector<double>{1}));
    run(pot, pos, z, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a short C6 ratio vector is rejected");
#if defined(RGPOT_HAS_DFTD4)
  threw = false;
  try {
    DispersionConfig bad4;
    bad4.kind = DispersionKind::D4;
    bad4.functional = "not-a-functional";
    bad4.cutoff = 8;
    bad4.short_cutoff = 2;
    DispersionPot pot(bad4);
    run(pot, pos, z, nullptr);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  check(threw, "an unknown D4 functional is rejected");
#endif
  {
    DispersionConfig close_cfg;
    close_cfg.kind = DispersionKind::MBD;
    close_cfg.cutoff = 8;
    close_cfg.short_cutoff = 2;
    close_cfg.mbd_beta = 0.1;
    DispersionPot pot(close_cfg);
    std::vector<double> piled = {0, 0, 0, 0.05, 0, 0, 0.08, 0.02, 0};
    try {
      run(pot, piled, z3, nullptr);
    } catch (const std::exception &) {
    }
  }

#if defined(RGPOT_HAS_DFTD3)
  run(d3pot, pos, z, nullptr);
  run(d3pot, pos, z, nullptr);
#else
  check_unlinked(d3pot, pos, z, "D3 without s-dftd3 refuses to evaluate");
#endif
  DispersionConfig d4cfg = d3cfg;
  d4cfg.kind = DispersionKind::D4;
  d4cfg.charge = 0;
  DispersionPot d4pot(d4cfg);
#if defined(RGPOT_HAS_DFTD4)
  run(d4pot, pos, z, nullptr);
  run(d4pot, pos, z, nullptr);
#else
  check_unlinked(d4pot, pos, z, "D4 without dftd4 refuses to evaluate");
#endif
}

void test_cuda_and_periodic_mbd() {
  const double pos[] = {0.1, 0.2, 0.3, 1.4, 0.2, 0.5, 0.4, 1.6, 0.2, 1.2, 1.1, 1.3};
  const double q[] = {0.4, -0.2, 0.5, -0.7};
  const double box[] = {6, 0.3, 0.1, -0.2, 5.5, 0.2, 0.1, -0.4, 6.2};
  DualList lists(3, 8, 0.5);
  lists.update(pos, 4, box, true);
  const auto &pairs = lists.long_pairs();
  std::vector<int> pi, pj;
  std::vector<double> px, py, pz, pr;
  for (const auto &p : pairs) {
    pi.push_back(p.i);
    pj.push_back(p.j);
    px.push_back(p.x);
    py.push_back(p.y);
    pz.push_back(p.z);
    pr.push_back(p.r);
  }
  CoulombConfig cfg;
  cfg.cutoff = 8;
  cfg.alpha = 0.35;
  cfg.kmax = 4;
  cfg.periodic = true;
  cfg.short_cutoff = 3;
  cfg.skin = 0.5;
  std::vector<double> serialF(12, 0.0);
  rgpot::lr::ElecOut serial;
  rgpot::lr::evaluate_electrostatics(CoulombMethod::Ewald, cfg, pos, q, 4, 0.0, box, pairs,
                                    serialF.data(), nullptr, &serial);
  rgpot::lr::CudaCoulombIn in;
  in.pos = pos;
  in.q = q;
  in.n = 4;
  in.box = box;
  in.pair_i = pi.data();
  in.pair_j = pj.data();
  in.pair_x = px.data();
  in.pair_y = py.data();
  in.pair_z = pz.data();
  in.pair_r = pr.data();
  in.npair = pairs.size();
  in.alpha = 0.35;
  in.cutoff = 8;
  in.kmax = 4;
  in.method = 0;
  double ene = 0.0;
  std::vector<double> Fc(12, 0.0);
  int used = 1;
  rgpot::lr::coulomb_cuda(in, &ene, Fc.data(), &used);
  check(used == 0 || used == 1, "cuda reports whether a device ran");
  check_close(ene, serial.energy, 1e-8, "cuda Ewald matches the serial energy");
  for (int i = 0; i < 12; ++i) {
    check_close(Fc[i], serialF[i], 1e-8, "cuda Ewald matches the serial force");
  }
  cfg.method = CoulombMethod::SPME;
  cfg.grid[0] = cfg.grid[1] = cfg.grid[2] = 16;
  cfg.spline_order = 4;
  std::fill(serialF.begin(), serialF.end(), 0.0);
  rgpot::lr::evaluate_electrostatics(CoulombMethod::SPME, cfg, pos, q, 4, 0.0, box, pairs,
                                    serialF.data(), nullptr, &serial);
  in.method = 1;
  in.grid = 16;
  in.order = 4;
  rgpot::lr::coulomb_cuda(in, &ene, Fc.data(), &used);
  check_close(ene, serial.energy, 1e-8, "cuda PME matches the serial energy");
  for (int i = 0; i < 12; ++i) {
    check_close(Fc[i], serialF[i], 1e-8, "cuda PME matches the serial force");
  }
  check(rgpot::lr::cuda_device_count() > 0 || used == 0, "without a device the host kernel runs");
  const double c6[] = {12.0, 4.0};
  const double r0[] = {3.0, 2.5};
  const double rr[] = {3.4, 2.8};
  const double damp = rgpot::lr::damp_sum_cuda(c6, r0, rr, 2, 0.4, 1.2, &used);
  check_close(damp, rgpot::lr::damp_sum_host(c6, r0, rr, 2, 0.4, 1.2), 1e-15, "BJ pair sum");
  const double mat[] = {0.2, -0.1, 0.05};
  check_close(rgpot::lr::pair_sum_cuda(mat, 3, &used), 0.15, 1e-15, "pairwise dispersion sum");
  bool threw = false;
  try {
    rgpot::lr::coulomb_cuda(in, &ene, Fc.data(), nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "cuda status pointer is required");
  threw = false;
  try {
    rgpot::lr::coulomb_host(in, nullptr, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "cuda energy pointer is required");
  threw = false;
  in.alpha = -1;
  try {
    rgpot::lr::coulomb_host(in, &ene, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "cuda alpha must be positive");
  in.alpha = 0.35;
  in.method = 3;
  threw = false;
  try {
    rgpot::lr::coulomb_host(in, &ene, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "cuda method must be known");
  in.method = 0;
  in.kmax = 0;
  threw = false;
  try {
    rgpot::lr::coulomb_host(in, &ene, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "cuda Ewald kmax must be positive");
  in.kmax = 4;
  in.method = 1;
  in.order = 2;
  threw = false;
  try {
    rgpot::lr::coulomb_host(in, &ene, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "cuda PME order must be at least 3");
  in.order = 4;
  in.n = 0;
  threw = false;
  try {
    rgpot::lr::coulomb_host(in, &ene, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "cuda coulomb needs atoms");
  in.n = 4;
  double badq[] = {0.4, std::numeric_limits<double>::quiet_NaN(), 0.5, -0.7};
  in.q = badq;
  threw = false;
  try {
    rgpot::lr::coulomb_host(in, &ene, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "cuda charges must be finite");
  in.q = q;
  threw = false;
  try {
    rgpot::lr::pair_sum_host(nullptr, 2);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "pair sum pointer is required");
  double badp[] = {1.0, std::numeric_limits<double>::infinity()};
  threw = false;
  try {
    rgpot::lr::pair_sum_host(badp, 2);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "pair sum entries must be finite");
  threw = false;
  try {
    rgpot::lr::damp_sum_host(c6, r0, rr, 2, 0.4, 0.0);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "BJ a2 must be positive");
  double badr[] = {0.0, 1.0};
  threw = false;
  try {
    rgpot::lr::damp_sum_host(c6, r0, badr, 2, 0.4, 1.0);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "BJ distances must be positive");
  threw = false;
  try {
    rgpot::lr::pair_sum_cuda(mat, 3, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "pair-sum status pointer is required");
  threw = false;
  try {
    rgpot::lr::damp_sum_cuda(c6, r0, rr, 2, 0.4, 1.0, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "damped-sum status pointer is required");

  DispersionConfig mcfg;
  mcfg.kind = DispersionKind::MBD;
  mcfg.periodic = true;
  mcfg.cutoff = 20;
  mcfg.short_cutoff = 4;
  mcfg.mbd_k[0] = mcfg.mbd_k[1] = mcfg.mbd_k[2] = 4;
  const double apos[] = {0.3, 0.1, 0.2, 4.1, -0.2, -0.1};
  const int az[] = {18, 18};
  const double abox[] = {8.1, 0.1, -0.2, 0.3, 3.9, -0.1, -0.1, 0.2, 4.2};
  auto model = rgpot::lr::make_dispersion(DispersionKind::MBD);
  std::vector<rgpot::lr::Pair> none;
  std::vector<double> h(2, 1.0);
  double energy = 0.0;
  double strain[6] = {};
  int has = 0;
  rgpot::ForceInput fin{2, apos, az, abox};
  model->evaluate(fin, none, h, nullptr, mcfg, &energy, nullptr, strain, &has);
  const double ha = energy * rgpot::units::EV_TO_HARTREE;
  check_close(ha, -0.0021037562496878173, 1e-8, "periodic MBD matches pymbd argon");
  mcfg.mbd_k[0] = 0;
  threw = false;
  try {
    model->evaluate(fin, none, h, nullptr, mcfg, &energy, nullptr, strain, &has);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "MBD k-mesh must be positive");
  mcfg.mbd_k[0] = 4;
  fin.box = nullptr;
  threw = false;
  try {
    model->evaluate(fin, none, h, nullptr, mcfg, &energy, nullptr, strain, &has);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "periodic MBD needs a cell");

  const double nanpos[] = {std::numeric_limits<double>::quiet_NaN(), 0, 0};
  const double oneq[] = {0.0};
  std::vector<rgpot::lr::Pair> nopairs;
  rgpot::lr::ElecOut eout;
  threw = false;
  try {
    rgpot::lr::evaluate_electrostatics(CoulombMethod::Ewald, cfg, nanpos, oneq, 1, 0.0, box, nopairs,
                                      nullptr, nullptr, &eout);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a non-finite position is rejected");

  {
    DispersionConfig cut;
    cut.kind = DispersionKind::D3BJ;
    cut.functional = "pbe";
    cut.cutoff = 12;
    cut.short_cutoff = 3;
    cut.lib_disp2 = 15;
    cut.lib_disp3 = 10;
    cut.lib_cn = 10;
    DispersionPot d3(cut);
    std::vector<double> wpos = {0, 0, 0, 0.96, 0, 0, -0.24, 0.93, 0};
    std::vector<int> wz = {8, 1, 1};
#if defined(RGPOT_HAS_DFTD3)
    run(d3, wpos, wz, nullptr);
#else
    check_unlinked(d3, wpos, wz, "D3 with library cutoffs refuses to evaluate without s-dftd3");
#endif
    cut.kind = DispersionKind::D4;
    DispersionPot d4(cut);
#if defined(RGPOT_HAS_DFTD4)
    run(d4, wpos, wz, nullptr);
#else
    check_unlinked(d4, wpos, wz, "D4 with library cutoffs refuses to evaluate without dftd4");
#endif
  }
  threw = false;
  try {
    rgpot::lr::make_dispersion(static_cast<DispersionKind>(99));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "an unknown dispersion kind is rejected");

  {
    DispersionConfig sw;
    sw.kind = DispersionKind::QDO;
    sw.cutoff = 2.0;
    sw.switch_width = 1.5;
    sw.short_cutoff = 0.5;
    DispersionPot pot(sw);
    std::vector<double> wpos = {0, 0, 0, 1.2, 0.1, 0, 0.2, 1.3, 0.2};
    std::vector<int> wz = {6, 1, 8};
    const Eval e = run(pot, wpos, wz, nullptr);
    check(std::isfinite(e.energy), "switched QDO energy is finite");
  }
  {
    rgpot::lr::NlhConfig ncfg;
    ncfg.cutoff = 2;
    NlhPot pot(ncfg);
    std::vector<double> one = {0, 0, 0};
    std::vector<int> z1 = {6};
    run(pot, one, z1, nullptr);
    std::vector<double> same = {0, 0, 0, 0, 0, 0};
    std::vector<int> z2 = {6, 6};
    run(pot, same, z2, nullptr);
  }
  {
    mcfg.mbd_gamma = true;
    fin.box = abox;
    model->evaluate(fin, none, h, nullptr, mcfg, &energy, nullptr, strain, &has);
    check(std::isfinite(energy), "Gamma-point MBD energy is finite");
    mcfg.mbd_gamma = false;
  }
  {
    const double badbox[] = {0, 0, 0, 0, 4, 0, 0, 0, 4};
    fin.box = badbox;
    threw = false;
    try {
      model->evaluate(fin, none, h, nullptr, mcfg, &energy, nullptr, strain, &has);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    check(threw, "a singular MBD cell is rejected");
    fin.box = abox;
  }
  {
    mcfg.mbd_beta = 0.05;
    const double piled[] = {0, 0, 0, 0.02, 0, 0};
    const double pbox[] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
    fin.pos = piled;
    fin.box = pbox;
    try {
      model->evaluate(fin, none, h, nullptr, mcfg, &energy, nullptr, strain, &has);
    } catch (const std::runtime_error &) {
    }
    mcfg.mbd_beta = 0.83;
  }
  in.pos = nullptr;
  in.q = q;
  in.n = 4;
  in.method = 0;
  in.kmax = 4;
  in.order = 4;
  in.alpha = 0.35;
  threw = false;
  try {
    rgpot::lr::coulomb_host(in, &ene, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "cuda coulomb needs a position pointer");
  in.pos = pos;
  std::vector<int> far_i = {0}, far_j = {1};
  std::vector<double> far_x = {20}, far_y = {0}, far_z = {0}, far_r = {20};
  in.pair_i = far_i.data();
  in.pair_j = far_j.data();
  in.pair_x = far_x.data();
  in.pair_y = far_y.data();
  in.pair_z = far_z.data();
  in.pair_r = far_r.data();
  in.npair = 1;
  rgpot::lr::coulomb_host(in, &ene, nullptr);
  check(std::isfinite(ene), "a pair outside the cutoff is skipped");
  threw = false;
  try {
    rgpot::lr::damp_sum_host(nullptr, nullptr, nullptr, 1, 0.4, 1.0);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a damped pair sum needs its coefficients");
}

void test_worker_cap() {
  using rgpot::lr::atom_workers;
  using rgpot::lr::axis_workers;
  // A 25-point axis: one worker below four threads, four workers after that.
  // Eight threads must not be given more work than four.
  check(axis_workers(25, 1) == 1, "one thread stays one worker");
  check(axis_workers(25, 2) == 1, "two threads do not split a 25-point axis");
  check(axis_workers(25, 4) == 4, "four threads fit a 25-point axis");
  check(axis_workers(25, 8) == 4, "eight threads cap a 25-point axis at four");
  check(axis_workers(32, 8) == 4, "a 32-point axis stays at four workers");
  check(axis_workers(40, 8) == 5, "40 points feed five workers at eight planes each");
  check(axis_workers(80, 8) == 8, "80 points feed eight workers");
  check(axis_workers(80, 8) <= 8, "workers never exceed the team");
  // More threads never reduce the worker count.
  for (int n : {25, 32, 40, 63, 80}) {
    int prev = 0;
    for (int team : {1, 2, 4, 8}) {
      const int got = axis_workers(n, team);
      check(got >= prev, "axis workers do not shrink as the team grows");
      check(got <= team, "axis workers stay inside the team");
      prev = got;
    }
  }
  check(atom_workers(128, 8) == 4, "128 atoms feed four spread workers");
  check(atom_workers(128, 8) <= atom_workers(4096, 8), "more atoms can feed more workers");
  check(atom_workers(31, 8) == 1, "fewer than 32 atoms stay on one worker");
  check(rgpot::lr::pme_team(128, 25, 8) == 8, "a request of 8 keeps the spare real-space threads");
  check(rgpot::lr::beside_real_workers(128, 25, 8) == 4, "four spare threads run real space");
  check(rgpot::lr::beside_real_workers(128, 25, 4) == 0, "four threads have no spare real team");
  check(rgpot::lr::beside_real_workers(256, 32, 8) == 4, "a 32-point mesh still hides real space");
  check(rgpot::lr::beside_real_workers(512, 40, 8) == 3, "a 40-point mesh hides real space on three threads");
  check(rgpot::lr::beside_real_workers(1024, 50, 8) == 0, "a 50-point mesh keeps one team");
  check(rgpot::lr::beside_real_workers(128, 25, 16) == 4, "a spare team above four stays at four");
  check(rgpot::lr::pme_team(128, 25, 4) == 4, "a request of 4 on 128 atoms is a team of 4");
  check(rgpot::lr::pme_team(128, 25, 2) == 2, "a request of 2 keeps the real-space pair");
  check(rgpot::lr::pme_team(4096, 80, 8) == 8, "a long mesh keeps the whole request");
}

// N=128 on the 25-point mesh. A request of 8 must run a team of 4 in
// every phase, and must not leave the other four workers in the pool.
void test_capped_phase_team() {
  constexpr int n = 128;
  const double side = 24.0;
  const double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  std::vector<double> pos(3 * n);
  std::vector<int> z(n, 1);
  std::vector<double> q(n);
  int k = 0;
  for (int ix = 0; ix < 4; ++ix) {
    for (int iy = 0; iy < 4; ++iy) {
      for (int iz = 0; iz < 8; ++iz) {
        pos[3 * k] = (static_cast<double>(ix) + 0.5) * side / 4.0;
        pos[3 * k + 1] = (static_cast<double>(iy) + 0.5) * side / 4.0;
        pos[3 * k + 2] = (static_cast<double>(iz) + 0.5) * side / 8.0;
        q[k] = (k % 2 == 0) ? 0.1 : -0.1;
        ++k;
      }
    }
  }
#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(1);
#endif
  rgpot::lr::caller_team() = 1;
  CoulombConfig serial_cfg;
  serial_cfg.method = CoulombMethod::SPME;
  serial_cfg.periodic = true;
  serial_cfg.cutoff = 8.0;
  serial_cfg.alpha = 0.364803;
  serial_cfg.grid[0] = serial_cfg.grid[1] = serial_cfg.grid[2] = 25;
  serial_cfg.spline_order = 5;
  serial_cfg.stress = false;
  CoulombPot serial_pot(serial_cfg, std::make_shared<FixedCharges>(q, 0.0));
  const Eval serial = run(serial_pot, pos, z, box);
#ifdef _OPENMP
  omp_set_num_threads(4);
#endif
  rgpot::lr::caller_team() = 4;
  CoulombPot four(serial_cfg, std::make_shared<FixedCharges>(q, 0.0));
  const Eval team4 = run(four, pos, z, box);
#ifdef _OPENMP
  omp_set_num_threads(8);
#pragma omp parallel
  {
    (void)omp_get_thread_num();
  }
  check(omp_get_max_threads() == 8, "the request starts as a team of 8");
#endif
  rgpot::lr::caller_team() = 8;
  CoulombConfig cfg;
  cfg.method = CoulombMethod::SPME;
  cfg.periodic = true;
  cfg.cutoff = 8.0;
  cfg.alpha = 0.364803;
  cfg.grid[0] = cfg.grid[1] = cfg.grid[2] = 25;
  cfg.spline_order = 5;
  cfg.stress = false;
  CoulombPot pot(cfg, std::make_shared<FixedCharges>(q, 0.0));
  run(pot, pos, z, box);
  const Eval wide = run(pot, pos, z, box);
  const rgpot::lr::PmeTrace tr = rgpot::lr::pme_trace();
  check(tr.spread_team == 4, "spread runs the capped team");
  check(tr.fft_team == 4, "the transform runs the capped team");
  check(tr.kernel_team == 4, "the kernel runs the capped team");
  check(tr.gather_team == 4, "gather runs the capped team");
  check(tr.real_team == 4, "real space runs on the spare threads");
  check(wide.energy == team4.energy && wide.f == team4.f,
        "beside real space matches the four-thread mesh bit for bit");
  check_close(wide.energy, serial.energy, 1e-12, "beside energy matches the serial energy");
  double versus_serial = 0.0;
  double versus_four = 0.0;
  for (std::size_t i = 0; i < wide.f.size(); ++i) {
    versus_serial = std::max(versus_serial, std::abs(wide.f[i] - serial.f[i]));
    versus_four = std::max(versus_four, std::abs(wide.f[i] - team4.f[i]));
  }
  // The float slab transform already moves forces by about 2e-10 against
  // one thread. The beside run adds nothing on top of that.
  check(versus_four == 0.0, "beside forces match the four-thread forces bit for bit");
  check(versus_serial <= 1e-9, "beside forces stay inside the float-transform gap");
#ifdef _OPENMP
  check(omp_get_max_threads() == 8, "the pool is the mesh team plus real space");
#endif
  rgpot::lr::caller_team() = 1;
}

// Paths the ladder does not take: a spline order other than 5, a team that
// has to grow, a non-finite coordinate, and a D4 system past the dense
// cutoff so the conjugate gradient runs.
void test_edge_paths() {
  bool threw = false;
  try {
    const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    std::vector<double> pos = {1, 1, 1, 2, 2, std::numeric_limits<double>::infinity()};
    std::vector<int> z = {1, 8};
    std::vector<double> q = {0.2, -0.2};
    CoulombConfig cfg;
    cfg.method = CoulombMethod::SPME;
    cfg.periodic = true;
    cfg.cutoff = 4.0;
    cfg.alpha = 0.3;
    cfg.grid[0] = cfg.grid[1] = cfg.grid[2] = 16;
    cfg.spline_order = 5;
    cfg.stress = false;
    CoulombPot pot(cfg, std::make_shared<FixedCharges>(q, 0.0));
    run(pot, pos, z, box);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a non-finite position is rejected");

  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  constexpr int n = 32;
  std::vector<double> pos(3 * n);
  std::vector<int> z(n, 1);
  std::vector<double> q(n);
  for (int i = 0; i < n; ++i) {
    pos[3 * i] = 1.0 + (i % 4) * 4.0;
    pos[3 * i + 1] = 1.0 + ((i / 4) % 4) * 4.0;
    pos[3 * i + 2] = 1.0 + (i / 16) * 8.0;
    q[i] = (i % 2 == 0) ? 0.1 : -0.1;
    z[i] = (i % 2 == 0) ? 8 : 1;
  }
#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(1);
#endif
  rgpot::lr::caller_team() = 8;
  CoulombConfig cfg;
  cfg.method = CoulombMethod::SPME;
  cfg.periodic = true;
  cfg.cutoff = 6.0;
  cfg.alpha = 0.3;
  cfg.grid[0] = cfg.grid[1] = cfg.grid[2] = 40;
  cfg.spline_order = 6;
  cfg.stress = false;
  CoulombPot pot(cfg, std::make_shared<FixedCharges>(q, 0.0));
  const Eval grown = run(pot, pos, z, box);
  check(std::isfinite(grown.energy), "order-6 PME energy is finite");
#ifdef _OPENMP
  omp_set_num_threads(4);
#endif
  setenv("LR_PME_PROFILE", "1", 1);
  const Eval profiled = run(pot, pos, z, box);
  unsetenv("LR_PME_PROFILE");
  check(std::isfinite(profiled.energy), "a profiled PME call stays finite");
  rgpot::lr::caller_team() = 1;

  // A pair closer than the screen table's first node uses the direct erfc.
  // One atom sits on the box edge so a stencil index wraps through zero.
  std::vector<double> close_pos = {0.02, 0.2, 0.2, 0.02005, 0.2, 0.2};
  std::vector<int> close_z = {1, 1};
  std::vector<double> close_q = {0.3, -0.3};
  CoulombConfig near_cfg;
  near_cfg.method = CoulombMethod::SPME;
  near_cfg.periodic = true;
  near_cfg.cutoff = 8.0;
  near_cfg.alpha = 0.25;
  near_cfg.grid[0] = near_cfg.grid[1] = near_cfg.grid[2] = 16;
  near_cfg.spline_order = 5;
  near_cfg.stress = false;
  const double near_box[9] = {12, 0, 0, 0, 12, 0, 0, 0, 12};
  CoulombPot near_pot(near_cfg, std::make_shared<FixedCharges>(close_q, 0.0));
  const Eval near = run(near_pot, close_pos, close_z, near_box);
  check(std::isfinite(near.energy), "a very short pair stays finite");

  // Orders other than 5 and 6 use the recursive cardinal spline. A second
  // box rebuilds the mesh kernel, and a stress call drops the float buffer.
  for (int order : {3, 4}) {
    CoulombConfig alt = cfg;
    alt.spline_order = order;
    alt.grid[0] = alt.grid[1] = alt.grid[2] = 24;
    CoulombPot alt_pot(alt, std::make_shared<FixedCharges>(q, 0.0));
    const Eval a = run(alt_pot, pos, z, box);
    double box2[9];
    std::copy(box, box + 9, box2);
    box2[0] = box2[4] = box2[8] = 24.0;
    std::vector<double> edge = pos;
    edge[2] = 24.0;
    const Eval b = run(alt_pot, edge, z, box2);
    check(std::isfinite(a.energy) && std::isfinite(b.energy), "lower-order PME stays finite");
    alt.stress = true;
    CoulombPot stress_pot(alt, std::make_shared<FixedCharges>(q, 0.0));
    const Eval c = run(stress_pot, pos, z, box);
    check(c.has_stress && std::isfinite(c.energy), "stress PME publishes a stress");
  }

#if defined(RGPOT_HAS_DFTD4)
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  setenv("LR_EEQ_DIRECT", "8", 1);
  rgpot::lr::DispersionConfig d4;
  d4.kind = rgpot::lr::DispersionKind::D4;
  d4.functional = "pbe";
  d4.atm = false;
  d4.periodic = true;
  d4.cutoff = 8.0;
  d4.short_cutoff = 3.0;
  constexpr int nd4 = 80;
  const double side = 18.0;
  const double dbox[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  std::vector<double> dpos(3 * nd4);
  std::vector<int> dz(nd4);
  for (int i = 0; i < nd4; ++i) {
    dpos[3 * i] = 0.8 + (i % 5) * 3.2;
    dpos[3 * i + 1] = 0.8 + ((i / 5) % 4) * 4.0;
    dpos[3 * i + 2] = 0.8 + (i / 20) * 4.0;
    dz[i] = (i % 2 == 0) ? 6 : 1;
  }
  rgpot::lr::DispersionPot d4pot(d4);
  const Eval ed4 = run(d4pot, dpos, dz, dbox);
  check(std::isfinite(ed4.energy), "periodic D4 above the dense cutoff is finite");
  unsetenv("LR_EEQ_DIRECT");
#endif
}

} // namespace

int main() {
  std::cout << std::setprecision(17);
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  try {
    test_lists();
    test_duallist_skin_sequences();
    test_list_edges();
    test_native_dispersion_images();
    test_madelung();
    test_dispersion_lattice_sum();
    test_forces_and_stress();
    test_dsf();
    test_public_null_cell_calls();
    test_qdo_nlh_mbd();
    test_library_dispersion();
    test_sum_and_charge();
    test_edges();
    test_nve();
    test_cuda_and_periodic_mbd();
    test_worker_cap();
    test_edge_paths();
    test_capped_phase_team();
  } catch (const std::exception &ex) {
    std::cerr << "FAIL exception " << ex.what() << '\n';
    return 1;
  }
  if (g_fails != 0) {
    std::cerr << g_fails << " checks failed\n";
    return 1;
  }
  std::cout << "lr_phys_test ok\n";
  return 0;
}
