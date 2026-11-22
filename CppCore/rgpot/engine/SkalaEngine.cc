// MIT License — generic rgpot engine table for Skala XC on NWChem. The
// configuration is the NWChemParams message; the engine reads basis, charge,
// multiplicity, enginePath and nwchemRoot from it and keeps Skala's XC.

#include "rgpot/engine/EngineTable.hpp"
#include "rgpot/SkalaPot/SkalaPot.hpp"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>

namespace rgpot::engine {

const char *backend_name() { return "skala"; }

Created create(const void *config, std::size_t config_len) {
  Created out;
  try {
    const kj::ArrayPtr<const capnp::word> words(
        reinterpret_cast<const capnp::word *>(config),
        config_len / sizeof(capnp::word));
    capnp::FlatArrayMessageReader reader(words);
    const auto params = reader.getRoot<::NWChemParams>();
    SkalaConfig c;
    if (params.getBasis().size() > 0) {
      c.basis = params.getBasis().cStr();
    }
    c.charge = params.getCharge();
    c.multiplicity = params.getMultiplicity();
    c.engine_path = params.getEnginePath().cStr();
    c.nwchem_root = params.getNwchemRoot().cStr();
    auto backend = std::make_unique<Backend>();
    auto pot = std::make_unique<SkalaPot>(c);
    SkalaPot *raw = pot.get();
    backend->pot = std::move(pot);
    backend->set_charge_spin = [raw](int charge, int spin) {
      if (spin < 1) {
        throw Unsupported("skala spin multiplicity must be >= 1; got " +
                          std::to_string(spin));
      }
      raw->setChargeMultiplicity(charge, spin);
    };
    out.backend = std::move(backend);
  } catch (const std::exception &e) {
    out.error = e.what();
  } catch (...) {
    out.error = "rgpot_engine_create(skala): unknown exception";
  }
  return out;
}

} // namespace rgpot::engine
