#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief CP2K Quickstep potential through the dlopen'd libcp2k shim.
 */

#include "rgpot/Potential.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace rgpot {

/// The SCF did not converge. The shim agrees on this outcome over the
/// calculator's communicator, so every rank of the calculator throws it
/// for the same call and no rank aborts the world. A caller may reject
/// the geometry and continue.
class CP2KScfError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class CP2KPot : public Potential<CP2KPot> {
public:
  CP2KPot();
  explicit CP2KPot(const ::CP2KParams::Reader &params);
  ~CP2KPot() override;

  CP2KPot(const CP2KPot &) = delete;
  CP2KPot &operator=(const CP2KPot &) = delete;

  /// Every rank of the calling calculator enters this call together.
  /// Positions and the box are Angstrom; the energy is eV and the forces
  /// eV/Angstrom. An unconverged SCF throws CP2KScfError on every rank.
  /// Any other failure is printed on every rank of the calculator, then
  /// MPI_Abort runs on MPI_COMM_WORLD, and then the exception leaves the
  /// call.
  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// libcp2k keeps global state: serialize process-wide. The SCF and the
  /// convergence exchange are collectives on the calculator's
  /// communicator.
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::ProcessSerial, .groupCollective = true};
  }

  /// FNV-1a over the serialized CP2KParams bytes plus kKernelVersion.
  [[nodiscard]] uint64_t paramsKey() const noexcept override;

  /// Bump whenever the engine-facing numerics change.
  static constexpr uint64_t kKernelVersion = 1;

  /// Stores the params. A session built from other params is destroyed;
  /// the next force renders the deck again. Returns false when the shim
  /// cannot be loaded or inputTemplate lacks a geometry token.
  bool setParams(const ::CP2KParams::Reader &params);

  void getParams(::CP2KParams::Builder out) const;

  bool setPotentialConfig(const ::PotentialConfig::Reader &cfg,
                          std::string *message_out = nullptr);

  /// Names the calculation the following forces belong to (a band image,
  /// a ring-polymer bead). The shim keeps one CP2K force environment per
  /// key, so each keeps its own wavefunction. Every rank of a calculator
  /// names the same key before the same force.
  void selectOrbitals(int64_t key);

  /// True when the loaded shim keeps a force environment per key.
  [[nodiscard]] bool keepsOrbitalsPerKey() const;

  /// True when the shim is loaded and the params are usable.
  bool available() const;
  /// Why available() is false; empty otherwise.
  std::string loadError() const;
  /// Directory holding input.inp and the CP2K output of the live session;
  /// empty before the first force.
  std::string runDirectory() const;

  static bool probe_available();

  /// Collective on MPI_COMM_WORLD, as CPMDPot::bindCalculators: splits
  /// into calculators of ranks_per_calc ranks and hands the shim the
  /// calculator communicator. Returns the group index, or -1 when the
  /// shim has no communicator-adoption symbol.
  static int bindCalculators(int ranks_per_calc);

private:
  struct Impl;
  Impl *impl_;

  void forceImplOrThrow(const ForceInput &in, ForceOut *out) const;
};

} // namespace rgpot
