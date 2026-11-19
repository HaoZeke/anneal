#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/Potential.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace rgpot {

struct MullerBrownConfig {
  std::array<double, 4> amplitude{-200.0, -100.0, -170.0, 15.0};
  std::array<double, 4> a{-1.0, -1.0, -6.5, 0.7};
  std::array<double, 4> b{0.0, 0.0, 11.0, 0.6};
  std::array<double, 4> c{-10.0, -10.0, -6.5, 0.7};
  std::array<double, 4> x0{1.0, 0.0, -0.5, -1.0};
  std::array<double, 4> y0{0.0, 0.5, 1.5, 1.0};
};

struct ModelEvaluation {
  double energy;
  std::vector<double> gradient;
  std::vector<double> product;
};

class AnalyticModelPot : public Potential<AnalyticModelPot> {
public:
  ~AnalyticModelPot() override;
  AnalyticModelPot(AnalyticModelPot &&) noexcept;
  AnalyticModelPot &operator=(AnalyticModelPot &&) noexcept;
  AnalyticModelPot(const AnalyticModelPot &) = delete;
  AnalyticModelPot &operator=(const AnalyticModelPot &) = delete;

  [[nodiscard]] std::size_t dimensions() const noexcept;
  [[nodiscard]] ModelEvaluation evaluateCoordinates(
      std::span<const double> coordinates,
      std::span<const double> direction) const;
  void forceImpl(const ForceInput &in, ForceOut *out) const;
  void hessianVectorImpl(const ForceInput &in, const double *direction,
                         double *product) const override;
  [[nodiscard]] PotCaps caps() const noexcept override;

protected:
  struct Impl;
  explicit AnalyticModelPot(PotType type, std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

class EckartPot final : public AnalyticModelPot {
public:
  explicit EckartPot(double v0 = 1.0, double width = 1.0);
};

class MullerBrownPot final : public AnalyticModelPot {
public:
  explicit MullerBrownPot(MullerBrownConfig config = {});
  [[nodiscard]] const MullerBrownConfig &config() const noexcept;
};

class ModifiedMullerBrownPot final : public AnalyticModelPot {
public:
  explicit ModifiedMullerBrownPot(double amplitude = 9.0,
                                  double waveNumber = 5.0);
};

enum class LepsSystem { HH2, HCl2, FH2 };

class LepsPot final : public AnalyticModelPot {
public:
  explicit LepsPot(LepsSystem system);
};

class DoubleWellPot final : public AnalyticModelPot {
public:
  DoubleWellPot(std::size_t dimensions, double v0, double x0, double omega,
                double coupling);
};

class CoupledProtonWellPot final : public AnalyticModelPot {
public:
  CoupledProtonWellPot(double v0, double x0, double spring);
};

} // namespace rgpot
