#pragma once
// MIT License — config POD for UmaPot (vesin neighbor list + AOTI .pt2)

#include <stdexcept>
#include <string>
#include <string_view>

namespace rgpot {

/// Upper bound of fairchem's SPIN_RANGE (`[0, 100]`).
inline constexpr int kUmaSpinMax = 100;

/**
 * Validate a UMA spin value for @p task_name.
 *
 * Every task (including `omol`, `omat`, `oc20`, `oc25`, `odac`, `omc`) accepts
 * integer spin values in `[0, kUmaSpinMax]`. Accepted values are not
 * rewritten. Throws `std::invalid_argument` on violation.
 */
inline void validate_uma_spin_for_task(std::string_view task_name, int spin,
                                       std::string_view where = "UmaPot");

inline void validate_uma_spin_for_task(std::string_view task_name, int spin,
                                       std::string_view where) {
  const std::string prefix =
      where.empty() ? std::string{} : std::string(where) + ": ";
  if (spin < 0 || spin > kUmaSpinMax) {
    throw std::invalid_argument(prefix + "UMA task '" +
                                (task_name.empty() ? std::string{"<unset>"}
                                                   : std::string(task_name)) +
                                "' spin must be an integer from 0 to " +
                                std::to_string(kUmaSpinMax) + "; got " +
                                std::to_string(spin));
  }
}

/**
 * UMA / OMol calculator configuration.
 *
 * model_path is an AOTInductor package (.pt2) produced by
 * scripts/export_uma_aoti.py. Neighbor lists are built with vesin and
 * passed as tensors; the compiled graph does not call fairchem.
 */
struct UmaConfig {
  std::string model_path;
  std::string task_name = "omol";
  std::string device = "cpu";
  int charge = 0;
  int spin = 1;
  /// Cutoff in angstrom. The package's embedded `cutoff` metadata wins
  /// when present; no sidecar file is read.
  double cutoff = 6.0;
  int max_neighbors = 300;
  /// Root of the persistent cache the package is unpacked into, under
  /// `aoti/<sha256 of the .pt2>.model/`. Empty means `$RGPOT_CACHE_DIR`,
  /// then `$XDG_CACHE_HOME/rgpot`, then `.rgpot-cache` in the working
  /// directory. Nothing is written to the system temporary directory.
  std::string cache_dir;
};

} // namespace rgpot
