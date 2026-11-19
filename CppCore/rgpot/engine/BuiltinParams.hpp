#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file BuiltinParams.hpp
 * @brief Reads the `BuiltinParams` arm (a name and an optional library path)
 * that selects a backend with no parameters of its own.
 */

#include <algorithm>
#include <array>
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
  struct MullerBrown {
    std::array<double, 4> amplitude{-200.0, -100.0, -170.0, 15.0};
    std::array<double, 4> a{-1.0, -1.0, -6.5, 0.7};
    std::array<double, 4> b{0.0, 0.0, 11.0, 0.6};
    std::array<double, 4> c{-10.0, -10.0, -6.5, 0.7};
    std::array<double, 4> x0{1.0, 0.0, -0.5, -1.0};
    std::array<double, 4> y0{0.0, 0.5, 1.5, 1.0};
  };

  std::string name;    ///< Lowercased.
  std::string library; ///< Optional explicit engine library path.
  std::optional<rgpot::XTBConfig> xtb; ///< Present for the XTB config arm.
  std::optional<MullerBrown> mullerBrown;
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
  if (native.which() == ::PotentialConfig::MULLER_BROWN) {
    out.mullerBrown.emplace();
    const auto wire = native.getMullerBrown();
    const auto copy = [](const auto &source, auto &target,
                         const char *name) {
      if (source.size() == 0)
        return;
      if (source.size() != target.size())
        throw std::invalid_argument(std::string("Muller-Brown ") + name +
                                    " must contain four values");
      std::copy(source.begin(), source.end(), target.begin());
    };
    copy(wire.getAmplitude(), out.mullerBrown->amplitude, "amplitude");
    copy(wire.getA(), out.mullerBrown->a, "a");
    copy(wire.getB(), out.mullerBrown->b, "b");
    copy(wire.getC(), out.mullerBrown->c, "c");
    copy(wire.getX0(), out.mullerBrown->x0, "x0");
    copy(wire.getY0(), out.mullerBrown->y0, "y0");
  }
  return out;
}

} // namespace rgpot::engine
