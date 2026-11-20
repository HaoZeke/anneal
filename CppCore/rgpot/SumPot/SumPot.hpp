#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Weighted sum of child potentials.
 *
 * Each child is a Potential evaluated through ForceInput. The sum adds
 * weight * energy, weight * forces and weight * stress over the enabled
 * children and records what each child returned, so a host can read the
 * components of the last call. A DFT engine is one child and a dispersion
 * correction another; a weight of -1 subtracts a term.
 */

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "rgpot/Potential.hpp"

namespace rgpot {

/// How one child enters the sum.
struct SumTerm {
  std::string name;    //!< Label in the component record and the log.
  double weight = 1.0; //!< Any finite value; the sign is allowed.
  bool enabled = true; //!< A disabled child is never evaluated.
};

/// What one child returned in the last force call, before its weight.
struct SumComponent {
  std::string name;
  double weight = 1.0;
  bool enabled = true;
  /// False for a disabled child and before the first call.
  bool evaluated = false;
  double energy = 0.0;        //!< eV, the child's own energy.
  std::vector<double> forces; //!< 3 * nAtoms, eV/Angstrom. Empty when not evaluated.
  double variance = 0.0;
  bool has_stress = false;
  std::array<double, 9> stress{}; //!< eV/Angstrom^3, row-major.
};

/**
 * @class SumPot
 * @brief @f$E = \sum_i w_i E_i@f$ over the enabled children.
 * @ingroup rgpot_potentials
 *
 * Forces and variances follow the same weights (variance with
 * @f$w_i^2@f$). The stress is summed when every evaluated child reports
 * one; otherwise @c has_stress is 0. With no enabled child the sum
 * reports no stress, and @c caps() says so. @c caps() is otherwise the
 * intersection of the enabled children's capabilities, and its reentrancy
 * is the weakest of theirs and @c PerInstance, because the component
 * record is per instance.
 *
 * A child that throws stops the call: the exception reaches the caller
 * unchanged, after the failure is logged with the child's name.
 *
 * Setting @c RGPOT_SUM_LOG in the environment writes one line per child
 * and call to stderr; @c setLogSink replaces that destination.
 */
class SumPot : public Potential<SumPot> {
public:
  using LogSink = std::function<void(std::string_view)>;

  SumPot();

  /// Appends @p term as "term<i>" with weight 1, enabled.
  void add(std::unique_ptr<PotentialBase> term);
  /// Appends @p term under @p spec. @p term may be null only for a
  /// disabled term, which is then recorded and never built.
  void add(SumTerm spec, std::unique_ptr<PotentialBase> term);

  void setEnabled(std::size_t index, bool enabled);
  void setWeight(std::size_t index, double weight);

  [[nodiscard]] std::size_t size() const noexcept { return terms_.size(); }
  [[nodiscard]] const SumTerm &term(std::size_t index) const;
  /// The potential of term @p index; null for a term added disabled
  /// without one.
  [[nodiscard]] PotentialBase *child(std::size_t index) const;

  /// One entry per child, in the order they were added, from the last
  /// force call on this instance.
  [[nodiscard]] const std::vector<SumComponent> &lastComponents() const noexcept {
    return components_;
  }

  /// Receives the log lines instead of stderr. An empty sink restores the
  /// environment-controlled default.
  void setLogSink(LogSink sink);

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override;
  [[nodiscard]] uint64_t paramsKey() const noexcept override;

private:
  struct Entry {
    SumTerm spec;
    std::unique_ptr<PotentialBase> pot;
  };

  void log(std::string_view line) const;
  void resetComponents() const;

  std::vector<Entry> terms_;
  mutable std::vector<SumComponent> components_;
  LogSink sink_;
  bool env_log_ = false;
};

} // namespace rgpot
