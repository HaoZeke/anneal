// MIT License
// Copyright 2023--present rgpot developers
//
// Finite-difference forces for the many-body potentials. The EAM pair
// term used to point the wrong way: at 2.8 A the energy derivative is
// about 0.76 and the force was about +4.27. This test fails on that
// sign and passes once the pair force is -(d phi/dr) (r_i - r_j) / r.

#include <array>
#include <cmath>
#include <cstdlib>
#include <vector>

#include <catch2/catch_all.hpp>

#include "lsan_libomp.hpp"
#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

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

template <typename Pot>
void checkDerivative(Pot &pot, rgpot::types::AtomMatrix xyz, const std::vector<int> &z,
                     const std::array<std::array<double, 3>, 3> &box) {
  const auto got = pot(xyz, z, box);
  const double analytic = -std::get<1>(got)(1, 0);
  constexpr double h = 1e-5;
  const double base = xyz(1, 0);
  xyz(1, 0) = base + h;
  const double ep = std::get<0>(pot(xyz, z, box));
  xyz(1, 0) = base - h;
  const double em = std::get<0>(pot(xyz, z, box));
  const double numeric = (ep - em) / (2.0 * h);
  const double scale = std::max(1.0, std::max(std::abs(numeric), std::abs(analytic)));
  REQUIRE(std::abs(numeric - analytic) <= 2e-4 * scale);
}

} // namespace

TEST_CASE("EAM pair force matches the energy derivative", "[eam][force]") {
  setThreads("");
  rgpot::types::AtomMatrix xyz(2, 3);
  xyz(0, 0) = 10.0;
  xyz(0, 1) = 10.0;
  xyz(0, 2) = 10.0;
  xyz(1, 0) = 12.8;
  xyz(1, 1) = 10.0;
  xyz(1, 2) = 10.0;
  const std::vector<int> z{13, 13};
  const std::array<std::array<double, 3>, 3> box{{
      {{30.0, 0.0, 0.0}},
      {{0.0, 30.0, 0.0}},
      {{0.0, 0.0, 30.0}},
  }};
  rgpot::EAMPot stored;
  stored.setNeighbourMode(rgpot::nlist::Mode::Stored);
  checkDerivative(stored, xyz, z, box);
  rgpot::EAMPot stream;
  stream.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkDerivative(stream, xyz, z, box);
}

TEST_CASE("Sutton-Chen force matches the energy derivative", "[suttonchen][force]") {
  setThreads("");
  rgpot::SuttonChenConfig cfg;
  cfg.cutoff = 6.0;
  cfg.periodic = true;
  rgpot::types::AtomMatrix xyz(2, 3);
  xyz(0, 0) = 4.0;
  xyz(0, 1) = 4.0;
  xyz(0, 2) = 4.0;
  xyz(1, 0) = 6.4;
  xyz(1, 1) = 4.0;
  xyz(1, 2) = 4.0;
  const std::vector<int> z{78, 78};
  const std::array<std::array<double, 3>, 3> box{{
      {{20.0, 0.0, 0.0}},
      {{0.0, 20.0, 0.0}},
      {{0.0, 0.0, 20.0}},
  }};
  rgpot::SuttonChenPot stored(cfg);
  stored.setNeighbourMode(rgpot::nlist::Mode::Stored);
  checkDerivative(stored, xyz, z, box);
  rgpot::SuttonChenPot stream(cfg);
  stream.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkDerivative(stream, xyz, z, box);
}
