#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Name of the PotentialConfig arm a configure() call carries.
 *
 * A peer built against a newer potentials-schema can send an arm this
 * build does not know. which() then returns the raw discriminant, so the
 * name comes from the compiled schema and an unknown arm is reported by
 * its discriminant instead of indexing past the known ones.
 */

#include <cstdint>
#include <string>

#include <capnp/schema.h>

#include "rgpot/rpc/Potentials.capnp.h"

namespace rgpot::rpc {

[[nodiscard]] inline std::string potentialConfigArm(const ::PotentialConfig::Reader &cfg) {
  const auto which = static_cast<std::uint16_t>(cfg.which());
  for (auto field : capnp::Schema::from<::PotentialConfig>().getUnionFields()) {
    if (field.getProto().getDiscriminantValue() == which) {
      return field.getProto().getName().cStr();
    }
  }
  return "discriminant " + std::to_string(which) + " (not in this build's schema)";
}

} // namespace rgpot::rpc
