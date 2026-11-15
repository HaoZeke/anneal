// MIT License
// Copyright 2023--present rgpot developers
//
// One timing for bench_dropin.py: frames evaluated either by the pot_spec
// potential directly or through librgpot_gromacs with the calls GROMACS
// makes each step.
//
//   rgpot-gromacs-bench bare Name [key value ...] < frames
//   rgpot-gromacs-bench shim input.inp < frames
//
// frames is rgpot-eval's geometry (n, nine cell values, z x y z lines, in
// Angstrom) followed by any number of further frames of 3n coordinates.
// bare passes each frame as it is; shim converts it to Bohr as GROMACS
// does, so the shim wraps it. For shim, input.inp must name a PDB file with
// the same atoms, all QM. Five untimed calls go first, then each frame is
// timed once in order (a single frame is timed 200 times). It prints
// "mean <us> median <us> min <us> calls <k> energy_ev <E>". Kernels with a
// Verlet skin rebuild their list on frames that move an atom too far, so
// the mean over a trajectory includes those rebuilds.

#include "libcp2k.h"
#include "pot_spec.hpp"
#include "qmmm_units.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kWarmup = 5;
constexpr long kSingleFrameCalls = 200;

struct Frames {
  std::vector<int> z;
  std::vector<std::vector<double>> xyz;
  double box[9]{};
};

bool readFrames(Frames *f) {
  int n = 0;
  if (std::scanf("%d", &n) != 1 || n <= 0) {
    return false;
  }
  for (double &v : f->box) {
    if (std::scanf("%lf", &v) != 1) {
      return false;
    }
  }
  const auto count = static_cast<std::size_t>(n);
  f->z.resize(count);
  std::vector<double> first(3 * count);
  for (std::size_t i = 0; i < count; ++i) {
    if (std::scanf("%d %lf %lf %lf", &f->z[i], &first[3 * i], &first[3 * i + 1],
                   &first[3 * i + 2]) != 4) {
      return false;
    }
  }
  f->xyz.push_back(std::move(first));
  for (;;) {
    std::vector<double> next(3 * count);
    for (std::size_t k = 0; k < next.size(); ++k) {
      if (std::scanf("%lf", &next[k]) != 1) {
        return k == 0;
      }
    }
    f->xyz.push_back(std::move(next));
  }
}

template <typename F> void report(const Frames &f, F &&call) {
  const std::size_t nframes = f.xyz.size();
  const long calls = nframes == 1 ? kSingleFrameCalls : static_cast<long>(nframes);
  double energy = 0.0;
  for (int i = 0; i < kWarmup; ++i) {
    energy = call(f.xyz[static_cast<std::size_t>(i) % nframes]);
  }
  std::vector<double> us(static_cast<std::size_t>(calls));
  for (long i = 0; i < calls; ++i) {
    const std::vector<double> &x = f.xyz[static_cast<std::size_t>(i) % nframes];
    const auto t0 = std::chrono::steady_clock::now();
    energy = call(x);
    const auto t1 = std::chrono::steady_clock::now();
    us[static_cast<std::size_t>(i)] = std::chrono::duration<double, std::micro>(t1 - t0).count();
  }
  const double mean = std::accumulate(us.begin(), us.end(), 0.0) / static_cast<double>(calls);
  std::sort(us.begin(), us.end());
  std::printf("mean %.3f median %.3f min %.3f calls %ld energy_ev %.17g\n", mean,
              us[us.size() / 2], us.front(), calls, energy);
}

} // namespace

int main(int argc, char **argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if ((mode != "bare" || argc < 3 || argc % 2 != 1) && (mode != "shim" || argc != 3)) {
    std::fprintf(stderr, "usage: rgpot-gromacs-bench bare Name [key value ...] < frames\n"
                         "       rgpot-gromacs-bench shim input.inp < frames\n");
    return 2;
  }
  Frames f;
  if (!readFrames(&f)) {
    std::fprintf(stderr, "bad frames on stdin\n");
    return 2;
  }
  const std::size_t n = f.z.size();
  std::vector<double> forces(3 * n, 0.0);

  if (mode == "bare") {
    rgpot::spec::Options options;
    for (int k = 3; k + 1 < argc; k += 2) {
      options.emplace_back(argv[k], argv[k + 1]);
    }
    rgpot::spec::Built built = rgpot::spec::build(argv[2], options, 0.0);
    if (!built.pot) {
      std::fprintf(stderr, "%s\n", built.error.c_str());
      return 1;
    }
    report(f, [&](const std::vector<double> &x) {
      std::fill(forces.begin(), forces.end(), 0.0);
      rgpot::ForceInput in{n, x.data(), f.z.data(), f.box};
      rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
      built.eval(in, &out);
      return out.energy;
    });
    return 0;
  }

  namespace units = rgpot::gromacs;
  // GROMACS 2026 layout: box_d[3 * i + j] = box[i][j], rows are cell vectors.
  double cell[9];
  for (int i = 0; i < 9; ++i) {
    cell[i] = f.box[i] / units::kAngstromPerBohr;
  }
  const int nel = static_cast<int>(3 * n);
  std::vector<double> bohr(3 * n);
  std::vector<double> back(3 * n);
  cp2k_init_without_mpi();
  force_env_t env = 0;
  cp2k_create_force_env(&env, argv[2], "bench_cp2k.out");
  report(f, [&](const std::vector<double> &x) {
    for (std::size_t i = 0; i < 3 * n; ++i) {
      bohr[i] = x[i] / units::kAngstromPerBohr;
    }
    double hartree = 0.0;
    cp2k_set_positions(env, bohr.data(), nel);
    cp2k_set_cell(env, cell);
    cp2k_calc_energy_force(env);
    cp2k_get_potential_energy(env, &hartree);
    cp2k_get_forces(env, forces.data(), nel);
    cp2k_get_positions(env, back.data(), nel);
    return hartree / units::kHartreePerEv;
  });
  cp2k_destroy_force_env(env);
  cp2k_finalize_without_mpi();
  return 0;
}
