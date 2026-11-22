// MIT License — generic rgpot engine table for MOPAC (libmopacc through the
// dlopen frontend). BuiltinParams carries the name "mopac" and, optionally,
// the path of libmopacc.

#include "rgpot/engine/BuiltinParams.hpp"
#include "rgpot/engine/EngineTable.hpp"
#include "rgpot/MOPACPot/MOPACPot.hpp"

namespace rgpot::engine {

const char *backend_name() { return "mopac"; }

Created create(const void *config, std::size_t config_len) {
  Created out;
  try {
    const Builtin builtin = read_builtin(config, config_len);
    if (!builtin.name.empty() && builtin.name != "mopac") {
      out.error = "rgpot_engine_create(mopac): unknown potential '" +
                  builtin.name + "'";
      return out;
    }
    if (builtin.xtb) {
      throw std::invalid_argument(
          "BuiltinParams.config arm xtb does not match potential 'mopac'");
    }
    if (builtin.mullerBrown) {
      throw std::invalid_argument(
          "BuiltinParams.config arm mullerBrown does not match potential "
          "'mopac'");
    }
    MOPACPot::Config c;
    c.engine_path = builtin.library;
    auto backend = std::make_unique<Backend>();
    backend->pot = std::make_unique<MOPACPot>(c);
    out.backend = std::move(backend);
  } catch (const std::exception &e) {
    out.error = e.what();
  } catch (...) {
    out.error = "rgpot_engine_create(mopac): unknown exception";
  }
  return out;
}

} // namespace rgpot::engine
