#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// Electrostatics as an additive Potential.
//   Ewald: tinfoil (conducting) boundary, orthorhombic and triclinic.
//   SPME:  smooth particle-mesh Ewald, same split, B-spline order >= 2.
//   DSF:   Fennell-Gezelter damped shifted force, gas phase.
// Per-atom charges come from a ChargeSource (fixed, or filled by a
// backend each call). The net charge is a separate input and must match
// the sum of the per-atom charges. It is the G=0 background charge.

#include <memory>

#include "rgpot/LongRange/Pairs.hpp"
#include "rgpot/Potential.hpp"

namespace rgpot::lr {

enum class CoulombMethod { Ewald, SPME, DSF };

struct CoulombConfig {
  CoulombMethod method = CoulombMethod::Ewald;
  double cutoff = 9.0;     // Angstrom
  double alpha = 0.30;     // 1/Angstrom
  int kmax = 8;            // Ewald reciprocal shell
  int grid[3] = {32, 32, 32};
  int spline_order = 6;
  double short_cutoff = 4.0;
  double skin = 1.0;
  bool periodic = true;
  // The benchmark asks for energy and force. Stress is optional.
  bool stress = true;
};

class ChargeSource {
public:
  virtual ~ChargeSource() = default;
  // q has one elementary charge per atom. total is the net charge.
  virtual void load(const ForceInput &in, std::vector<double> *q, double *total) const = 0;
};

class FixedCharges final : public ChargeSource {
public:
  FixedCharges(std::vector<double> q, double total);
  void load(const ForceInput &in, std::vector<double> *q, double *total) const override;

private:
  std::vector<double> q_;
  double total_;
};

// Backend hook. fn writes q[0, n) and *total.
class FnCharges final : public ChargeSource {
public:
  using Fn = void (*)(const ForceInput &, double *, double *, void *);
  FnCharges(Fn fn, void *user);
  void load(const ForceInput &in, std::vector<double> *q, double *total) const override;

private:
  Fn fn_;
  void *user_;
};

// Chain rule for charges that depend on coordinates.
// dE_dq[i] is the partial of the electrostatic energy at fixed positions.
class ChargeResponse {
public:
  virtual ~ChargeResponse() = default;
  virtual void accumulate(const ForceInput &in, const double *dE_dq, double *force) const = 0;
};

struct ElecOut {
  double energy = 0.0;
  double strain[6]{}; // xx, yy, zz, yz, xz, xy of dE/d epsilon
  bool has_stress = false;
};

// Writes energy, adds forces into F (length 3 n, caller zeroes), and
// optionally dE/dq (length n). pairs is the long half list. For DSF the
// list cutoff is the DSF cutoff. For Ewald and SPME it is the real-space
// cutoff. box is null only for DSF.
void evaluate_electrostatics(CoulombMethod method, const CoulombConfig &cfg,
                             const double *pos, const double *q, std::size_t n,
                             double total_charge, const double *box,
                             const std::vector<Pair> &pairs, double *F, double *dE_dq,
                             ElecOut *out);

// SO3LR real-space kernel: erf(r/sigma)/r with shifted force and the
// smooth switch between cuton and cutoff. Energy of one unordered pair,
// in eV, and dE/dr in eV/Angstrom.
void so3lr_erf_pair(double qi, double qj, double r, double sigma, double cuton,
                    double cutoff, double ke, double *energy, double *dEdr);

class CoulombPot final : public Potential<CoulombPot> {
public:
  CoulombPot(CoulombConfig cfg, std::shared_ptr<ChargeSource> charges,
             std::shared_ptr<ChargeResponse> response = nullptr);

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .periodic = true, .stress = true};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override { return params_; }

  void set_neighbours(std::shared_ptr<DualList> lists) { lists_ = std::move(lists); }

  [[nodiscard]] const DualList &neighbours() const { return *lists_; }

private:
  CoulombConfig cfg_;
  std::shared_ptr<ChargeSource> charges_;
  std::shared_ptr<ChargeResponse> response_;
  std::shared_ptr<DualList> lists_;
  uint64_t params_ = 0;
};

} // namespace rgpot::lr
