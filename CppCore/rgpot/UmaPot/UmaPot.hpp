#pragma once
// MIT License — UMA / OMol via the neighbour query + AOTInductor .pt2

#include "rgpot/ParamHash.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/UmaPot/UmaConfig.hpp"
#include "rgpot/UmaPot/UmaContract.hpp"

#include <cstdint>
#include <memory>

namespace rgpot {

/**
 * @brief UMA / OMol calculator: neighbour query + AOTI package.
 *
 * Loads an AOTInductor ``.pt2`` exported by ``scripts/export_uma_aoti.py``.
 * Each force call builds a fairchem-convention edge list with the neighbour query and
 * runs the compiled graph. Charge and spin are per-call tensor inputs.
 *
 * A package is exported for one charge, spin and task, and either one
 * composition (merged MoLE) or any composition up to ``z_max``. A
 * dynamic package accepts a range of atom and edge counts. Every force
 * call checks the config and the input against the metadata embedded
 * in the package and throws UmaContractError or UmaShapeError on a
 * mismatch (see UmaContract).
 */
class UmaPot : public Potential<UmaPot> {
public:
  explicit UmaPot(const UmaConfig &config);
  ~UmaPot() override;

  UmaPot(const UmaPot &) = delete;
  UmaPot &operator=(const UmaPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// One AOTI call per chunk of up to the package's batch_max
  /// same-composition systems (a NEB band). Band packages carry only
  /// the batched graph, so a lone system is padded to two.
  void forceBatchImpl(const ForceBatch &batch) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::SharedInstance,
            .perImageInstances = true};
  }

  [[nodiscard]] const UmaConfig &config() const noexcept { return m_config; }

  void setChargeSpin(int charge, int spin);

  /// Input dtype used for positions, the cell and cell offsets.
  /// ``float32`` or ``float64``. Loads the package on the first call.
  [[nodiscard]] const char *posDtypeName() const;
  /// Weight dtype recorded in the package. Same lifetime rules as
  /// posDtypeName().
  [[nodiscard]] const char *modelDtypeName() const;

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

private:
  static constexpr uint64_t kKernelVersion = 5;

  struct Impl;

  UmaConfig m_config;
  uint64_t m_paramsKey{0};
  std::unique_ptr<Impl> m_impl;

  void recomputeParamsKey();
  void ensureLoaded() const;
  /// Throws UmaContractError when the package's embedded charge, spin,
  /// task or composition disagrees with the config or this input.
  void checkContract(const ForceInput &in) const;
};

} // namespace rgpot
