#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Neighbor lists use vesin. Source is compatible with both vesin 0.3.x
// (enum VesinDevice, no Options.algorithm) and 0.5+ (struct VesinDevice
// {type, device_id}, Options.algorithm) via type traits in vesin_compat.hpp.

#include <mutex>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wold-style-cast"

#include <torch/script.h>

#include "metatensor/torch.hpp"
#include "metatensor/torch/module.hpp"
#include "metatomic/torch.hpp"

#pragma GCC diagnostic pop

#include "rgpot/MetatomicPot/MetatomicConfig.hpp"
#include "rgpot/Potential.hpp"

namespace rgpot {

class MetatomicPot : public Potential<MetatomicPot> {
public:
  explicit MetatomicPot(const MetatomicConfig &config);
  ~MetatomicPot() = default;

  MetatomicPot(const MetatomicPot &) = delete;
  MetatomicPot &operator=(const MetatomicPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// One model forward over the batch when no orientation average is on.
  /// Rotations stay on the per-system path, which owns that average.
  void forceBatchImpl(const ForceBatch &batch) const override;

  void setChargeSpin(int charge, int spin) {
    m_config.charge = charge;
    m_config.spin = spin;
  }

  /// Whether the model reads the charge (rgpot::charge) and the multiplicity
  /// (rgpot::spin_multiplicity or rgpot::spin): it declares them, or
  /// attach_system_extras is set.
  [[nodiscard]] bool readsCharge() const noexcept { return m_attach_charge; }
  [[nodiscard]] bool readsSpin() const noexcept {
    return m_attach_spin_multiplicity || m_attach_spin;
  }

  /// Internal mutex serializes the torch session, so a shared instance is
  /// safe; multi-image callers still prefer clones for throughput.
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::SharedInstance,
            .perImageInstances = true,
            .batched = true};
  }


private:
  MetatomicConfig m_config;

  mutable metatensor_torch::Module m_model;
  metatomic_torch::ModelCapabilities m_capabilities;
  std::vector<metatomic_torch::NeighborListOptions> m_nl_requests;
  metatomic_torch::ModelEvaluationOptions m_eval_options;

  torch::ScalarType m_dtype;
  torch::Device m_device;
  bool m_check_consistency;
  std::string m_energy_key;
  std::string m_energy_uncertainty_key;
  double m_uncertainty_threshold = -1.0;
  bool m_attach_charge = false;
  bool m_attach_spin_multiplicity = false;
  bool m_attach_spin = false;

  mutable std::mutex m_mutex;
  mutable torch::Tensor m_cached_types;
  mutable size_t m_cached_natoms = 0;

  metatensor_torch::TensorBlock
  computeNeighbors(metatomic_torch::NeighborListOptions request, long nAtoms,
                   const double *positions, const double *box,
                   const bool periodic[3]) const;

  void attachSystemExtras(const metatomic_torch::System &system) const;
};

} // namespace rgpot
