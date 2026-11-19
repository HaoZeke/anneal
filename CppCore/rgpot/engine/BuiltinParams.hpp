#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file BuiltinParams.hpp
 * @brief Reads the `BuiltinParams` arm (a name and an optional library path)
 * that selects a backend with no parameters of its own.
 */

#include <algorithm>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>

#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/XTBPot/XTBConfig.hpp"

namespace rgpot::engine {

struct Builtin {
  std::string name;    ///< Lowercased.
  std::string library; ///< Optional explicit engine library path.
  std::optional<rgpot::XTBConfig> xtb; ///< Present for the XTB config arm.
};

inline Builtin read_builtin(const void *config, std::size_t config_len) {
  const kj::ArrayPtr<const capnp::word> words(
      reinterpret_cast<const capnp::word *>(config),
      config_len / sizeof(capnp::word));
  capnp::FlatArrayMessageReader reader(words);
  const auto params = reader.getRoot<::BuiltinParams>();
  Builtin out;
  out.name = params.getName().cStr();
  std::transform(out.name.begin(), out.name.end(), out.name.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  out.library = params.getLibrary().cStr();
  const auto native = params.getConfig();
  if (native.which() == ::PotentialConfig::XTB) {
    const auto source = native.getXtb();
    rgpot::XTBConfig target;
    std::string method = source.getMethod().cStr();
    std::transform(method.begin(), method.end(), method.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (method == "gfnff") {
      target.method = rgpot::GFNMethod::GFNFF;
    } else if (method == "gfn0") {
      target.method = rgpot::GFNMethod::GFN0xTB;
    } else if (method == "gfn1") {
      target.method = rgpot::GFNMethod::GFN1xTB;
    } else if (method == "gfn2") {
      target.method = rgpot::GFNMethod::GFN2xTB;
    } else {
      throw std::invalid_argument("BuiltinParams.config.xtb: unknown method '" +
                                  method + "'");
    }
    target.accuracy = source.getAccuracy();
    target.electronic_temperature = source.getElectronicTemperature();
    target.max_iterations = source.getMaxIterations();
    target.charge = source.getCharge();
    target.uhf = source.getUhf();
    out.xtb = target;
  }
  return out;
}

} // namespace rgpot::engine
