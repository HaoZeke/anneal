// MIT License
// Copyright 2023--present rgpot developers

// The direct pair loop below is the energy and force evaluation this
// neighbour list replaced. One thread must reproduce it bit for bit,
// including the wrap into the cell and the order of the pair sums.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef EAM_STANDALONE
#define EAM_STANDALONE
#endif
#include "lsan_libomp.hpp"
#include "rgpot/EAM/EAM.h"
#include "rgpot/EAM/EAMPot.hpp"

#include <catch2/catch_all.hpp>

namespace {

struct Elem {
  int Z;
  double Dm;
  double alphaM;
  double Rm;
  double beta1;
  double beta2;
  double r_cut;
  double coeff[9];
};

// Parameters.h, copied so this oracle does not link the kernel's table.
const Elem kElem[] = {
    {13,
     3.63262,
     1.49303,
     2.12396,
     3.45762,
     3.54208,
     5.555,
     {-0.0847305, 17.0887, 60.7909, -75.8412, 39.9948, -6.21563, -3.07993,
      1.51365, -0.196034}},
    {46,
     1.6054,
     1.55083,
     2.35604,
     3.35148,
     3.34715,
     5.4120,
     {0.0992627, 4.8356, -0.368647, 17.2739, -18.450, 9.5018, -2.69876,
      0.405341, -0.0251768}},
    {79,
     0.669134,
     1.89707,
     2.57021,
     3.69785,
     3.69674,
     5.5155,
     {-0.198791, -27.7663, 224.068, -594.101, 1316.31, -2070.62, 1973.14,
      -1012.02, 214.373}},
};

const Elem &elemOf(int z) {
  for (const Elem &e : kElem) {
    if (e.Z == z) {
      return e;
    }
  }
  throw 14324;
}

double embed(const double *c, double rho) {
  double result = c[8];
  for (int i = 7; i >= 0; i--) {
    result = result * rho + c[i];
  }
  return result;
}

double embedForce(const double *c, double rho) {
  double result = c[8] * 8;
  for (int i = 7; i >= 1; i--) {
    result = result * rho + i * c[i];
  }
  return -result;
}

struct Outcome {
  double energy;
  std::vector<double> force;
};

std::uint64_t bitsOf(double v) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

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

// IEEE 754 leaves the sign of a NaN result unspecified: an optimiser may
// swap the operands of an add or a multiply, and given two NaNs the
// hardware returns the first. A NaN has to match a NaN, and any other
// value its bits.
void expectSame(double got, double ref) {
  if (std::isnan(ref)) {
    REQUIRE(std::isnan(got));
  } else {
    REQUIRE(bitsOf(got) == bitsOf(ref));
  }
}

void expectBits(const Outcome &got, const Outcome &ref) {
  expectSame(got.energy, ref.energy);
  REQUIRE(got.force.size() == ref.force.size());
  for (std::size_t i = 0; i < got.force.size(); ++i) {
    expectSame(got.force[i], ref.force[i]);
  }
}

Outcome direct(const std::vector<double> &pos, const std::vector<int> &z,
               double length) {
  const long n = static_cast<long>(z.size());
  const double box[3] = {length, length, length};
  std::vector<double> wrapped(static_cast<std::size_t>(3 * n));
  double xmin = 1e300;
  double ymin = xmin;
  double zmin = xmin;
  for (long i = 0; i < 3 * n; i += 3) {
    wrapped[static_cast<std::size_t>(i)] = pos[static_cast<std::size_t>(i)];
    wrapped[static_cast<std::size_t>(i + 1)] =
        pos[static_cast<std::size_t>(i + 1)];
    wrapped[static_cast<std::size_t>(i + 2)] =
        pos[static_cast<std::size_t>(i + 2)];
    xmin = std::min(xmin, pos[static_cast<std::size_t>(i)]);
    ymin = std::min(ymin, pos[static_cast<std::size_t>(i + 1)]);
    zmin = std::min(zmin, pos[static_cast<std::size_t>(i + 2)]);
  }
  if (xmin > 0) {
    xmin = 0;
  }
  if (ymin > 0) {
    ymin = 0;
  }
  if (zmin > 0) {
    zmin = 0;
  }
  for (long i = 0; i < 3 * n; i += 3) {
    wrapped[static_cast<std::size_t>(i)] += std::abs(xmin);
    wrapped[static_cast<std::size_t>(i + 1)] += std::abs(ymin);
    wrapped[static_cast<std::size_t>(i + 2)] += std::abs(zmin);
  }
  for (long i = 0; i < 3 * n; ++i) {
    while (wrapped[static_cast<std::size_t>(i)] > box[i % 3]) {
      wrapped[static_cast<std::size_t>(i)] -= box[i % 3];
    }
  }

  Outcome out;
  out.force.assign(static_cast<std::size_t>(3 * n), 0.0);
  out.energy = 0.0;
  if (n < 1) {
    return out;
  }
  double *F = out.force.data();
  double *U = &out.energy;
  std::vector<double> drho(static_cast<std::size_t>(3 * n));
  const double half0 = box[0] * 0.5;
  const double half1 = box[1] * 0.5;
  const double half2 = box[2] * 0.5;
  for (long i = 0; i < n; ++i) {
    std::fill(drho.begin(), drho.end(), 0.0);
    const Elem &epar = elemOf(z[static_cast<std::size_t>(i)]);
    const double cutoff2 = epar.r_cut * epar.r_cut;
    double dens = 0.0;
    const double xi = wrapped[static_cast<std::size_t>(3 * i)];
    const double yi = wrapped[static_cast<std::size_t>(3 * i + 1)];
    const double zi = wrapped[static_cast<std::size_t>(3 * i + 2)];
    for (long j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      double dx = xi - wrapped[static_cast<std::size_t>(3 * j)];
      double dy = yi - wrapped[static_cast<std::size_t>(3 * j + 1)];
      double dz = zi - wrapped[static_cast<std::size_t>(3 * j + 2)];
      if (dx > half0) {
        dx -= box[0];
      } else if (dx < -half0) {
        dx += box[0];
      }
      if (dy > half1) {
        dy -= box[1];
      } else if (dy < -half1) {
        dy += box[1];
      }
      if (dz > half2) {
        dz -= box[2];
      } else if (dz < -half2) {
        dz += box[2];
      }
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > cutoff2) {
        continue;
      }
      const double r = std::sqrt(r2);
      const double r6 = r2 * r2 * r2;
      const double exp_b1 = std::exp(-epar.beta1 * r);
      const double exp_b2 = std::exp(-2.0 * epar.beta2 * r);
      const double rho_pair = exp_b1 + 512.0 * exp_b2;
      dens += r6 * rho_pair;
      const double r5 = r2 * r2 * r;
      const double mag_den =
          6.0 * r5 * rho_pair +
          r6 * (-epar.beta1 * exp_b1 - 1024.0 * epar.beta2 * exp_b2);
      const double invR = 1.0 / r;
      const double fscale = -mag_den * invR;
      drho[static_cast<std::size_t>(3 * i)] += fscale * dx;
      drho[static_cast<std::size_t>(3 * i + 1)] += fscale * dy;
      drho[static_cast<std::size_t>(3 * i + 2)] += fscale * dz;
      drho[static_cast<std::size_t>(3 * j)] -= fscale * dx;
      drho[static_cast<std::size_t>(3 * j + 1)] -= fscale * dy;
      drho[static_cast<std::size_t>(3 * j + 2)] -= fscale * dz;
      if (j > i) {
        const double expArg = std::exp(-epar.alphaM * (r - epar.Rm));
        const double d = 1.0 - expArg;
        const double phi = epar.Dm * d * d - epar.Dm;
        const double mag = 2.0 * epar.alphaM * epar.Dm * d * (d - 1.0);
        const double scale = mag * invR;
        F[3 * i] += scale * dx;
        F[3 * i + 1] += scale * dy;
        F[3 * i + 2] += scale * dz;
        F[3 * j] -= scale * dx;
        F[3 * j + 1] -= scale * dy;
        F[3 * j + 2] -= scale * dz;
        *U += phi;
      }
    }
    *U += embed(epar.coeff, dens);
    const double dF = embedForce(epar.coeff, dens);
    for (long k = 0; k < 3 * n; ++k) {
      F[k] -= dF * drho[static_cast<std::size_t>(k)];
    }
  }
  return out;
}

Outcome evaluate(rgpot::EAMPot &pot, const std::vector<double> &pos,
                 const std::vector<int> &z, double length) {
  const auto n = z.size();
  rgpot::types::AtomMatrix xyz(n, 3);
  for (std::size_t i = 0; i < n; ++i) {
    xyz(i, 0) = pos[3 * i];
    xyz(i, 1) = pos[3 * i + 1];
    xyz(i, 2) = pos[3 * i + 2];
  }
  const std::array<std::array<double, 3>, 3> box{{
      {length, 0.0, 0.0},
      {0.0, length, 0.0},
      {0.0, 0.0, length},
  }};
  const auto [energy, forces, variance] = pot(xyz, z, box);
  (void)variance;
  Outcome out;
  out.energy = energy;
  out.force.assign(forces.data(), forces.data() + 3 * n);
  return out;
}

std::vector<double> fcc(int nrep, double a, std::vector<int> &z, int atomic) {
  const double basis[4][3] = {
      {0.0, 0.0, 0.0},
      {0.0, 0.5, 0.5},
      {0.5, 0.0, 0.5},
      {0.5, 0.5, 0.0},
  };
  std::vector<double> pos;
  z.clear();
  for (int ix = 0; ix < nrep; ++ix) {
    for (int iy = 0; iy < nrep; ++iy) {
      for (int iz = 0; iz < nrep; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * a);
          pos.push_back((iy + b[1]) * a);
          pos.push_back((iz + b[2]) * a);
          z.push_back(atomic);
        }
      }
    }
  }
  return pos;
}

} // namespace

TEST_CASE("EAM matches the direct pair loop bit for bit", "[eam]") {
  setThreads("");
  rgpot::EAMPot pot;
  std::vector<double> pos = {0.0, 0.0, 0.0, 2.5, 0.0, 0.0};
  std::vector<int> z = {13, 13};
  expectBits(evaluate(pot, pos, z, 20.0), direct(pos, z, 20.0));

  pos = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  expectBits(evaluate(pot, pos, z, 20.0), direct(pos, z, 20.0));

  pos = {-1.25, 0.2, 3.0, 4.5, -0.4, 19.2};
  z = {13, 46};
  expectBits(evaluate(pot, pos, z, 20.0), direct(pos, z, 20.0));

  pos = {0.05, 10.0, 10.0, 19.9, 10.0, 10.0};
  z = {13, 13};
  expectBits(evaluate(pot, pos, z, 20.0), direct(pos, z, 20.0));

  pos = {0.2, 0.2, 0.2, 2.2, 0.3, 0.4, 0.4, 3.1, 1.2, 5.5, 6.6, 7.1};
  z = {13, 46, 79, 13};
  expectBits(evaluate(pot, pos, z, 8.0), direct(pos, z, 8.0));

  pos = fcc(3, 4.05, z, 13);
  expectBits(evaluate(pot, pos, z, 3 * 4.05), direct(pos, z, 3 * 4.05));
  pos[0] += 0.37;
  pos[4] -= 0.11;
  expectBits(evaluate(pot, pos, z, 3 * 4.05), direct(pos, z, 3 * 4.05));

  pos = fcc(2, 4.05, z, 13);
  const int species[] = {13, 46, 79};
  for (std::size_t i = 0; i < z.size(); ++i) {
    z[i] = species[i % 3];
    pos[3 * i] -= 0.8;
  }
  expectBits(evaluate(pot, pos, z, 2 * 4.05), direct(pos, z, 2 * 4.05));

  pos = {1.0, 2.0, 3.0};
  z = {79};
  expectBits(evaluate(pot, pos, z, 10.0), direct(pos, z, 10.0));
  REQUIRE(pot.threadsUsed() == 1);
}

TEST_CASE("EAM keeps a short neighbour list and reuses it inside the skin",
          "[eam]") {
  setThreads("");
  // 8^3 fcc cells. The search cells are about 8 Angstrom wide, so an
  // atom that starts near 2 Angstrom can move by 1 Angstrom and stay in
  // the same cell. A cell-crossing rule would not rebuild; the skin does.
  std::vector<int> z;
  std::vector<double> pos = fcc(8, 4.05, z, 13);
  const double length = 8 * 4.05;
  const auto n = z.size();
  rgpot::EAMPot pot;
  const Outcome first = evaluate(pot, pos, z, length);
  REQUIRE(pot.listBuilds() == 1);
  REQUIRE(pot.neighborSlots() > n);
  REQUIRE(pot.neighborSlots() < n * 96);
  const Outcome second = evaluate(pot, pos, z, length);
  REQUIRE(pot.listBuilds() == 1);
  expectBits(second, first);

  // Aluminium has the widest cutoff in the element table.
  const double half_skin = 0.5 * kEamSkinOverCutoff * 5.555;
  std::size_t atom = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (pos[3 * i] > 1.5 && pos[3 * i] < 3.0) {
      atom = i;
      break;
    }
  }
  pos[3 * atom] += 0.25 * half_skin;
  const Outcome inside = evaluate(pot, pos, z, length);
  REQUIRE(pot.listBuilds() == 1);
  expectBits(inside, direct(pos, z, length));

  pos[3 * atom] += 3.0 * half_skin;
  const Outcome outside = evaluate(pot, pos, z, length);
  REQUIRE(pot.listBuilds() == 2);
  expectBits(outside, direct(pos, z, length));
}

TEST_CASE("EAM gives the one-thread bits on any thread count", "[eam]") {
  struct Crystal {
    int nrep;
    bool team; // OMP_NUM_THREADS=n gives n slices rather than one.
  };
  // 500 atoms stay on one thread. 8788 atoms reach rgpot::kTeamMinAtoms.
  for (const Crystal c : {Crystal{5, false}, Crystal{13, true}}) {
    std::vector<int> z;
    std::vector<double> pos = fcc(c.nrep, 4.05, z, 13);
    pos[0] += 0.37;
    pos[4] -= 0.11;
    const double length = c.nrep * 4.05;
    setThreads("");
    rgpot::EAMPot serial;
    const Outcome one = evaluate(serial, pos, z, length);
    REQUIRE(serial.threadsUsed() == 1);
    for (const int threads : {1, 2, 3, 8}) {
      const std::string count = std::to_string(threads);
      setThreads(count.c_str());
      rgpot::EAMPot parallel;
      expectBits(evaluate(parallel, pos, z, length), one);
      expectBits(evaluate(parallel, pos, z, length), one);
#if defined(_OPENMP)
      REQUIRE(parallel.threadsUsed() == (c.team ? threads : 1));
#else
      REQUIRE(parallel.threadsUsed() == 1);
#endif
    }
  }

  std::vector<double> pos = {0.2, 0.2, 0.2, 2.2, 0.3, 0.4,
                             0.4, 3.1, 1.2, 5.5, 6.6, 7.1};
  std::vector<int> z = {13, 46, 79, 13};
  setThreads("");
  rgpot::EAMPot mixedSerial;
  const Outcome mixedOne = evaluate(mixedSerial, pos, z, 12.0);
  setThreads("2");
  rgpot::EAMPot mixedParallel;
  expectBits(evaluate(mixedParallel, pos, z, 12.0), mixedOne);
  REQUIRE(mixedParallel.threadsUsed() == 1);
  setThreads("");
}
