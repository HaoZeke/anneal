// MIT License — generic rgpot engine table for the classical potentials.
// BuiltinParams names the potential: lj, ljcluster, morse, zbl and, when the
// water option is on, tip4p, spce, tip4p_pt. When the Fortran kernels are
// built: sw, edip, lenosky, tersoff, eamal, fehe, cuh2, waterh. Every
// potential takes its default parameters, except muller_brown, which reads
// BuiltinParams.config.mullerBrown. An arm that names another potential is
// an error.

#include "rgpot/engine/BuiltinParams.hpp"
#include "rgpot/engine/EngineTable.hpp"

#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/Models/ModelPots.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#ifdef RGPOT_HAS_CBE
#include "rgpot/CBEPot/CBEPot.hpp"
#endif
#ifdef RGPOT_HAS_WATER
#include "rgpot/Water/WaterPots.hpp"
#endif
#ifdef RGPOT_HAS_FORTRAN_POTS
#include "rgpot/fortran/FortranPots.hpp"
#endif

namespace rgpot::engine {

const char *backend_name() { return "classical"; }

Created create(const void *config, std::size_t config_len) {
  Created out;
  try {
    const Builtin builtin = read_builtin(config, config_len);
    std::unique_ptr<PotentialBase> pot;
    const std::string &n = builtin.name;
    if (builtin.mullerBrown && n != "muller_brown") {
      throw std::invalid_argument(
          "BuiltinParams.config arm mullerBrown does not match potential '" +
          n + "'");
    }
    if (builtin.xtb) {
      throw std::invalid_argument(
          "BuiltinParams.config arm xtb does not match potential '" + n + "'");
    }
    if (n == "lj") {
      pot = std::make_unique<LJPot>();
    } else if (n == "ljcluster") {
      pot = std::make_unique<LJClusterPot>();
    } else if (n == "morse") {
      pot = std::make_unique<MorsePot>();
    } else if (n == "zbl") {
      pot = std::make_unique<ZBLPot>();
    } else if (n == "eckart") {
      pot = std::make_unique<EckartPot>();
    } else if (n == "muller_brown") {
      pot = std::make_unique<MullerBrownPot>(
          builtin.mullerBrown.value_or(MullerBrownConfig{}));
    } else if (n == "modified_muller_brown") {
      pot = std::make_unique<ModifiedMullerBrownPot>();
    } else if (n == "leps_hh2") {
      pot = std::make_unique<LepsPot>(LepsSystem::HH2);
    } else if (n == "leps_hcl2") {
      pot = std::make_unique<LepsPot>(LepsSystem::HCl2);
    } else if (n == "leps_fh2") {
      pot = std::make_unique<LepsPot>(LepsSystem::FH2);
    } else if (n == "double_well") {
      pot = std::make_unique<DoubleWellPot>(1, 1.0, 5.0, 0.0, 0.0);
    } else if (n == "coupled_proton_well") {
      pot = std::make_unique<CoupledProtonWellPot>(1.0, 5.0, 0.32);
#ifdef RGPOT_HAS_CBE
    } else if (n == "cbe") {
      pot = std::make_unique<CBEPot>();
#endif
#ifdef RGPOT_HAS_WATER
    } else if (n == "tip4p") {
      pot = std::make_unique<TIP4PPot>();
    } else if (n == "spce") {
      pot = std::make_unique<SPCEPot>();
    } else if (n == "tip4p_pt") {
      pot = std::make_unique<TIP4PPtPot>();
#endif
#ifdef RGPOT_HAS_FORTRAN_POTS
    } else if (n == "sw") {
      pot = std::make_unique<fortranpots::SWPot>();
    } else if (n == "edip") {
      pot = std::make_unique<fortranpots::EDIPPot>();
    } else if (n == "lenosky") {
      pot = std::make_unique<fortranpots::LenoskyPot>();
    } else if (n == "tersoff") {
      // All-silicon stays on the Fortran gather. Any other Z is the
      // 1989 kernel, inside TersoffPot::forceImpl.
      pot = std::make_unique<fortranpots::TersoffPot>();
    } else if (n == "eamal") {
      pot = std::make_unique<fortranpots::EAMAlPot>();
    } else if (n == "fehe") {
      pot = std::make_unique<fortranpots::FeHePot>();
    } else if (n == "cuh2") {
      pot = std::make_unique<fortranpots::CuH2Pot>();
    } else if (n == "waterh") {
      pot = std::make_unique<fortranpots::WaterHPot>();
#endif
    } else {
      out.error = "rgpot_engine_create(classical): unknown potential '" + n +
                  "'";
      return out;
    }
    auto backend = std::make_unique<Backend>();
    backend->pot = std::move(pot);
    out.backend = std::move(backend);
  } catch (const std::exception &e) {
    out.error = e.what();
  } catch (...) {
    out.error = "rgpot_engine_create(classical): unknown exception";
  }
  return out;
}

} // namespace rgpot::engine
