// MIT License — generic rgpot engine table for the XTB engine. BuiltinParams
// carries the name "xtb"; the potential takes XTBConfig's defaults (GFN2-xTB).
// The XTB-specific table (xtb_c_abi.h) stays in XTBEngineAbi.cc.

#include "rgpot/engine/BuiltinParams.hpp"
#include "rgpot/engine/EngineTable.hpp"
#include "rgpot/XTBPot/XTBPot.hpp"

namespace rgpot::engine {

const char *backend_name() { return "xtb"; }

Created create(const void *config, std::size_t config_len) {
  Created out;
  try {
    const Builtin builtin = read_builtin(config, config_len);
    if (!builtin.name.empty() && builtin.name != "xtb") {
      out.error = "rgpot_engine_create(xtb): unknown potential '" +
                  builtin.name + "'";
      return out;
    }
    if (builtin.mullerBrown) {
      throw std::invalid_argument(
          "BuiltinParams.config arm mullerBrown does not match potential "
          "'xtb'");
    }
    auto backend = std::make_unique<Backend>();
    backend->pot =
        std::make_unique<XTBPot>(builtin.xtb.value_or(XTBConfig{}));
    out.backend = std::move(backend);
  } catch (const std::exception &e) {
    out.error = e.what();
  } catch (...) {
    out.error = "rgpot_engine_create(xtb): unknown exception";
  }
  return out;
}

} // namespace rgpot::engine
