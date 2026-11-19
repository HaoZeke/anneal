// MIT License
// Copyright 2023--present rgpot developers
//
// PotentialHandle::eam_table, the C++ face of rgpot_potential_new_eam_table:
// the copper vacancy cell against LAMMPS pair_style eam (see
// data/eam_table/make_vacancy_ref.py) and the refusals that throw.

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/potential.hpp"

using rgpot::PotentialHandle;

namespace {

const std::string kData = "CppCore/tests/data/eam_table/";

// The largest gaps of the run are 4e-13 eV and 6e-15 eV/Angstrom.
constexpr double kEnergyTolerance = 1.0e-11;
constexpr double kForceTolerance = 1.0e-12;

struct Reference {
  std::array<double, 9> cell{};
  double energy = 0.0;
  std::vector<double> positions;
  std::vector<std::int32_t> numbers;
  std::vector<double> forces;
};

Reference loadVacancy() {
  std::ifstream in(kData + "cu_u3_vacancy.ref");
  REQUIRE(in.good());
  Reference ref;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream fields(line);
    std::string key;
    if (line.empty() || line.front() == '#') {
      continue;
    }
    if (line.rfind("box ", 0) == 0) {
      fields >> key;
      for (double &v : ref.cell) {
        fields >> v;
      }
    } else if (line.rfind("energy ", 0) == 0) {
      fields >> key >> ref.energy;
    } else if (line.rfind("atoms ", 0) == 0 || line.rfind("stress ", 0) == 0) {
      continue;
    } else {
      int z = 0;
      double v[6];
      fields >> z;
      for (double &x : v) {
        fields >> x;
      }
      ref.numbers.push_back(z);
      ref.positions.insert(ref.positions.end(), v, v + 3);
      ref.forces.insert(ref.forces.end(), v + 3, v + 6);
    }
  }
  return ref;
}

} // namespace

TEST_CASE("eam_table evaluates the copper vacancy like LAMMPS", "[EAMTable]") {
  Reference ref = loadVacancy();
  REQUIRE(ref.numbers.size() == 107);

  auto handle =
      PotentialHandle::eam_table(kData + "Cu_u3.eam", RGPOT_EAM_TABLE_FUNCFL);
  REQUIRE(handle.raw() != nullptr);

  rgpot::InputSpec input(ref.numbers.size(), ref.positions.data(),
                         ref.numbers.data(), ref.cell.data());
  const auto result = handle.calculate(input);

  REQUIRE(result.has_forces());
  REQUIRE_THAT(result.energy(),
               Catch::Matchers::WithinAbs(ref.energy, kEnergyTolerance));
  const std::vector<double> forces = result.forces_vec();
  REQUIRE(forces.size() == ref.forces.size());
  double worst = 0.0;
  for (std::size_t k = 0; k < forces.size(); ++k) {
    worst = std::max(worst, std::abs(forces[k] - ref.forces[k]));
  }
  REQUIRE(worst < kForceTolerance);
}

TEST_CASE("eam_table throws rgpot::Error on a bad path or table",
          "[EAMTable]") {
  const std::vector<std::string> refused = {"", kData + "no_such_table.eam",
                                            kData + "malformed_funcfl.eam"};
  for (const std::string &path : refused) {
    CAPTURE(path);
    try {
      (void)PotentialHandle::eam_table(path, RGPOT_EAM_TABLE_FUNCFL);
      FAIL("a handle came back");
    } catch (const rgpot::Error &e) {
      REQUIRE(e.status() == RGPOT_INVALID_PARAMETER);
      REQUIRE(std::string(e.what()).find("EAM table") != std::string::npos);
    }
  }
  REQUIRE_THROWS_AS(PotentialHandle::eam_table(kData + "Cu_u3.eam", 17),
                    rgpot::Error);
}
