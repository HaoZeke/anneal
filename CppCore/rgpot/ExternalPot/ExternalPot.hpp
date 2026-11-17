#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief LAMMPS, GROMACS and OpenMM constructors.
 *
 * Each one is a @c Potential. The call surface is Angstrom and eV.
 * @c rgpot_lammps_potential dlopens the engine plugin
 * (@c liblj_engine.so or @c liblammps_engine.so) and leaves that
 * plugin as the backend. @c rgpot_gromacs_potential and
 * @c rgpot_openmm_potential store the nanometre and kJ/mol factors
 * once. An empty engine path evaluates the shifted Lennard-Jones
 * potential in process. A path is a library whose
 * @c rgpot_md_native_force speaks those units.
 */

#include "rgpot/ExternalPot/external_potentials.h"
#include "rgpot/Potential.hpp"

#include <memory>
#include <string>

namespace rgpot {

/**
 * @brief Host-unit parameters of the shifted 12-6 potential the
 * engine drivers compare.
 */
struct ExternalLjConfig {
  double u0 = 1.0;     //!< Well depth (eV).
  double cutoff = 8.0; //!< Truncation (Angstrom).
  double psi = 1.0;    //!< Zero crossing (Angstrom).
  std::string engine_path;
};

class LammpsPot : public Potential<LammpsPot> {
public:
  explicit LammpsPot(const ExternalLjConfig &config);
  ~LammpsPot() override;
  LammpsPot(const LammpsPot &) = delete;
  LammpsPot &operator=(const LammpsPot &) = delete;
  LammpsPot(LammpsPot &&) noexcept;
  LammpsPot &operator=(LammpsPot &&) noexcept;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .periodic = true};
  }

  [[nodiscard]] const ExternalLjConfig &config() const noexcept {
    return config_;
  }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  ExternalLjConfig config_;
};

class GromacsPot : public Potential<GromacsPot> {
public:
  explicit GromacsPot(const ExternalLjConfig &config);
  ~GromacsPot() override;
  GromacsPot(const GromacsPot &) = delete;
  GromacsPot &operator=(const GromacsPot &) = delete;
  GromacsPot(GromacsPot &&) noexcept;
  GromacsPot &operator=(GromacsPot &&) noexcept;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;
  [[nodiscard]] PotCaps caps() const noexcept override;

  [[nodiscard]] const ExternalLjConfig &config() const noexcept {
    return config_;
  }
  /// Well depth stored for the engine, kJ/mol.
  [[nodiscard]] double nativeU0() const noexcept { return native_u0_; }
  /// Cutoff stored for the engine, nanometres.
  [[nodiscard]] double nativeCutoff() const noexcept { return native_cutoff_; }
  /// Zero crossing stored for the engine, nanometres.
  [[nodiscard]] double nativePsi() const noexcept { return native_psi_; }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  ExternalLjConfig config_;
  double native_u0_ = 0.0;
  double native_cutoff_ = 0.0;
  double native_psi_ = 0.0;
};

class OpenMMPot : public Potential<OpenMMPot> {
public:
  explicit OpenMMPot(const ExternalLjConfig &config);
  ~OpenMMPot() override;
  OpenMMPot(const OpenMMPot &) = delete;
  OpenMMPot &operator=(const OpenMMPot &) = delete;
  OpenMMPot(OpenMMPot &&) noexcept;
  OpenMMPot &operator=(OpenMMPot &&) noexcept;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;
  [[nodiscard]] PotCaps caps() const noexcept override;

  [[nodiscard]] const ExternalLjConfig &config() const noexcept {
    return config_;
  }
  [[nodiscard]] double nativeU0() const noexcept { return native_u0_; }
  [[nodiscard]] double nativeCutoff() const noexcept { return native_cutoff_; }
  [[nodiscard]] double nativePsi() const noexcept { return native_psi_; }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  ExternalLjConfig config_;
  double native_u0_ = 0.0;
  double native_cutoff_ = 0.0;
  double native_psi_ = 0.0;
};

std::unique_ptr<LammpsPot>
rgpot_lammps_potential(const ExternalLjConfig &config);
std::unique_ptr<GromacsPot>
rgpot_gromacs_potential(const ExternalLjConfig &config);
std::unique_ptr<OpenMMPot>
rgpot_openmm_potential(const ExternalLjConfig &config);

} // namespace rgpot
