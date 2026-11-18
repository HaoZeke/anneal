// MIT License
// Copyright 2023--present rgpot developers

#pragma once

#include <string>

namespace rgpot {

inline std::string formatCpmdConfigMessage(const std::string &summary, bool ok,
                                           bool engine_loaded,
                                           const std::string &reason) {
  if (ok)
    return "rgpot params applied (cpmd arm): " + summary;
  if (!engine_loaded)
    return "rgpot params cpmd arm failed (engine not loaded): " + summary;
  std::string message = "rgpot params cpmd arm rejected by embed: " + summary;
  if (!reason.empty())
    message += " (" + reason + ")";
  return message;
}

} // namespace rgpot
