#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief OpenMOPAC frontend. Pure consumer of libmopacc.so.
 *
 * Packed MopacCParams. Default model is AM1.
 */

#include "rgpot/Potential.hpp"

#include <string>

namespace rgpot {

class MOPACPot : public Potential<MOPACPot> {
public:
  struct Config {
    int charge = 0;
    int spin = 0;
    int model = 4; // AM1
    double tolerance = 1.0;
    int max_time = 3600;
    std::string engine_path;
  };

  MOPACPot();
  explicit MOPACPot(const Config &cfg);
  ~MOPACPot() override;

  MOPACPot(const MOPACPot &) = delete;
  MOPACPot &operator=(const MOPACPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::ProcessSerial, .periodic = false};
  }

  bool available() const;
  static bool probeAvailable();
  static bool abiAvailable();

private:
  struct Impl;
  Impl *impl_;
};

} // namespace rgpot
