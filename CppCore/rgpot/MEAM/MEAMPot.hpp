#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file MEAMPot.hpp
 * @brief Modified embedded-atom method energy, forces and stress.
 *
 * The energy of MEAMModel for a configuration of atoms. Atoms are matched
 * to library elements by atomic number. The forces are the exact gradient
 * of the energy, including the many-body screening, so the result agrees
 * with LAMMPS pair_style meam to rounding and finite differences of the
 * energy reproduce the forces.
 */

// clang-format off
#include <cstdint>
#include <string>
#include <vector>
// clang-format on
#include "rgpot/MEAM/MEAMModel.hpp"
#include "rgpot/Potential.hpp"

namespace rgpot {

/**
 * @brief Parameters for MEAMPot.
 *
 * @c library is a LAMMPS meam library file and @c parameters the optional
 * parameter file of pair_style meam. @c elements picks and orders the
 * library entries, as the element list of pair_coeff does. An empty list
 * takes every entry of the library in file order.
 */
struct MEAMConfig {
  std::string library;               //!< Library file.
  std::string parameters;            //!< Parameter file, empty for none.
  std::vector<std::string> elements; //!< Library entries to use.
  /// Atomic number of each entry of `elements`. Empty takes it from the
  /// leading chemical symbol of the name.
  std::vector<int> atomic_numbers;
  /// Periodic boundaries. Every image inside the cutoff is counted, also
  /// when the cutoff exceeds half the cell.
  bool periodic = true;
  /// Write the Cauchy stress of stress.hpp when the cell volume is positive.
  bool stress = false;
};

/**
 * @class MEAMPot
 * @brief MEAM and 2NN MEAM potential.
 * @ingroup rgpot_potentials
 */
class MEAMPot : public Potential<MEAMPot> {
public:
  explicit MEAMPot(const MEAMConfig &config);
  /// @param atomic_numbers One per element of the model, or empty to read
  ///        each from the leading symbol of its library name (and from the
  ///        library column when the name has none).
  MEAMPot(MEAMModel model, bool periodic, bool stress,
          std::vector<int> atomic_numbers = {});

  [[nodiscard]] PotCaps caps() const noexcept override {
    // forceImpl keeps nothing between calls, so one instance may serve
    // concurrent callers.
    return {.periodic = m_periodic, .stress = m_stress};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  /// Radial cutoff rc of the model, in Angstrom.
  [[nodiscard]] double cutoff() const noexcept { return m_model.rc(); }

  [[nodiscard]] const MEAMModel &model() const noexcept { return m_model; }

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  static constexpr uint64_t kKernelVersion = 1;

  MEAMModel m_model;
  bool m_periodic;
  bool m_stress;
  uint64_t m_paramsKey{0};
  /// Element index of each atomic number, -1 for one the model lacks.
  std::vector<int> m_element;
};

} // namespace rgpot
