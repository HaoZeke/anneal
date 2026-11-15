#pragma once
// MIT License — config POD for UmaPot (vesin neighbor list + AOTI .pt2)

#include <string>

namespace rgpot {

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
