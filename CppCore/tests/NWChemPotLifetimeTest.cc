// MIT License

#include "rgpot/NWChemPot/NWChemPot.hpp"
#include "rgpot/units.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {
void call(rgpot::NWChemPot &pot, int expected_call) {
  if (!pot.available())
    throw std::runtime_error("fixture engine unavailable");
  const double positions[3] = {0.0, 0.0, 0.0};
  const int numbers[1] = {1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  double forces[3] = {};
  const rgpot::ForceInput input{1, positions, numbers, box};
  rgpot::ForceOut output{forces, 0.0, 0.0, {}, 0};
  pot.forceImpl(input, &output);
  const double expected =
      0.01 * expected_call * rgpot::units::HARTREE_TO_EV;
  std::printf("call %d energy %.17g expected %.17g\n", expected_call,
              output.energy, expected);
  if (std::abs(output.energy - expected) > 1e-12)
    throw std::runtime_error(
        "engine process state reset between object lifetimes");
  for (int i = 0; i < 3; ++i)
    if (std::abs(forces[i] - 0.001 * (i + 1) *
                               rgpot::units::NEG_GRAD_TO_FORCE) > 1e-12)
      throw std::runtime_error("engine force conversion changed");
}

void record_objects_destroyed() {
  const char *path = std::getenv("RGPOT_NWCHEM_LIFETIME_LOG");
  if (!path)
    throw std::runtime_error("fixture event path missing");
  FILE *file = std::fopen(path, "a");
  if (!file)
    throw std::runtime_error("fixture event log unavailable");
  std::fprintf(file, "objects destroyed\n");
  std::fclose(file);
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc > 1 && std::string(argv[1]) == "probe") {
      if (!rgpot::NWChemPot::probeAvailable() ||
          !rgpot::NWChemPot::abiAvailable())
        throw std::runtime_error("fixture probe failed");
    }
    {
      rgpot::NWChemPot first;
      call(first, 1);
      {
        rgpot::NWChemPot second;
        call(second, 2);
      }
      call(first, 3);
    }
    {
      rgpot::NWChemPot third;
      call(third, 4);
    }
    record_objects_destroyed();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
  return 0;
}
