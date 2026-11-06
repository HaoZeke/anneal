#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// One dispersion model. D3(BJ) and D4 wrap s-dftd3 and dftd4 when those
// libraries are linked. MBD@rsSCS and the SO3LR QDO term are in-tree.
// Environment-dependent ratios scale the free-atom C6 and polarizability.

#include <memory>
#include <string>

#include "rgpot/LongRange/Pairs.hpp"
#include "rgpot/Potential.hpp"

namespace rgpot::lr {

enum class DispersionKind { D3BJ, D4, MBD, QDO };

struct DispersionConfig {
  DispersionKind kind = DispersionKind::QDO;
  std::string functional = "pbe";
  bool atm = true;
  double charge = 0.0; // D4 total charge
  double mbd_beta = 0.83;
  double mbd_a = 6.0;
  double qdo_scale = 1.0;
  bool qdo_legacy = false;
  bool qdo_damped = true;
  double cutoff = 15.0;
  double switch_width = 0.0; // 0 keeps a hard list cutoff
  double short_cutoff = 4.0;
  double skin = 1.0;
  bool periodic = false;
};

// h is the Hirshfeld (volume) ratio. c6_ratio scales the free-atom C6.
// A null c6 vector means c6_ratio = h^2.
class RatioSource {
public:
  virtual ~RatioSource() = default;
  virtual void load(const ForceInput &in, std::vector<double> *h,
                    std::vector<double> *c6) const = 0;
};

class UnitRatios final : public RatioSource {
public:
  void load(const ForceInput &in, std::vector<double> *h, std::vector<double> *c6) const override;
};

class FixedRatios final : public RatioSource {
public:
  FixedRatios(std::vector<double> h, std::vector<double> c6);
  void load(const ForceInput &in, std::vector<double> *h, std::vector<double> *c6) const override;

private:
  std::vector<double> h_;
  std::vector<double> c6_;
  bool have_c6_ = false;
};

// dh and dc6 are n*3 derivatives of each atom's ratio w.r.t. its own
// coordinates (d h_i / d r_i). Null skips that chain.
class RatioResponse {
public:
  virtual ~RatioResponse() = default;
  virtual void jacobians(const ForceInput &in, const double **dh, const double **dc6) const = 0;
};

class DispersionModel {
public:
  virtual ~DispersionModel() = default;
  [[nodiscard]] virtual const char *name() const = 0;
  [[nodiscard]] virtual DispersionKind kind() const = 0;
  virtual void evaluate(const ForceInput &in, const std::vector<Pair> &pairs,
                        const std::vector<double> &h, const std::vector<double> *c6,
                        const DispersionConfig &cfg, double *energy, double *F,
                        double strain[6], int *has_stress) const = 0;
};

std::unique_ptr<DispersionModel> make_dispersion(DispersionKind kind);

class DispersionPot final : public Potential<DispersionPot> {
public:
  explicit DispersionPot(DispersionConfig cfg, std::shared_ptr<RatioSource> ratios = nullptr,
                         std::shared_ptr<RatioResponse> response = nullptr);

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .stress = true};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override { return params_; }

  void set_neighbours(std::shared_ptr<DualList> lists) { lists_ = std::move(lists); }

private:
  DispersionConfig cfg_;
  std::shared_ptr<RatioSource> ratios_;
  std::shared_ptr<RatioResponse> response_;
  std::shared_ptr<DualList> lists_;
  std::unique_ptr<DispersionModel> model_;
  uint64_t params_ = 0;
};

} // namespace rgpot::lr
