#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file EngineTable.hpp
 * @brief One implementation of the generic engine C ABI over any
 * @c PotentialBase.
 *
 * `EngineTable.cc` exports every `rgpot_engine_*` symbol of
 * `engine_c_abi.h` for an engine library except `rgpot_engine_abi_stamp`,
 * which `EngineAbiStamp.cc` defines. The library supplies what only the
 * backend knows by defining `rgpot::engine::create` and
 * `rgpot::engine::backend_name`; the adapter does argument validation, the
 * coordinate transform, batching, the error text, the thread-safety policy,
 * the identity queries and the Python GIL release.
 *
 * Thread safety of a handle follows the backend's @c PotCaps::reentrancy:
 * - @c SharedInstance: any number of threads may call force and force_batch on
 *   one handle at once.
 * - @c PerInstance: the adapter admits one call at a time per handle, so a
 *   handle is safe from any thread and calls on it are serialized.
 * - @c ProcessSerial: the adapter admits one call at a time in the process
 *   across all handles of the library.
 * Setting calls (threads, charge and spin) and destroy must not run
 * concurrently with a call on the same handle.
 */

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rgpot/Potential.hpp"
#include "rgpot/SumPot/SumPot.hpp"

namespace rgpot::engine {

/// Thrown by a setting hook for a value the backend cannot take; the adapter
/// returns 3 with the reason in last_error.
struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// What a backend hands the adapter at create.
struct Backend {
  std::unique_ptr<PotentialBase> pot;
  /// File whose SHA-256 identifies the loaded model; empty when the backend
  /// has no model file.
  std::string model_path;
  /// Empty when the backend has no charge and spin to change. Throws
  /// Unsupported, before changing anything, for a value it cannot take.
  std::function<void(int charge, int spin)> set_charge_spin;
  /// Empty when the backend has no tensor runtime.
  std::function<void(int intra_op, int inter_op)> set_num_threads;
  /// Empty when the backend keeps one wavefunction for every call.
  /// The argument is the caller's key: an image of a band, a bead of a ring.
  std::function<void(long long key)> select_orbitals;
  /// Empty when the backend cannot copy orbitals between keys.
  std::function<void(long long from_key, long long to_key)> copy_orbitals;
  /// Empty when the backend is not a sum; otherwise the per-term record of
  /// its last force call (rgpot::SumPot::lastComponents).
  std::function<const std::vector<SumComponent> &()> components;
};

struct Created {
  std::unique_ptr<Backend> backend;
  std::string error; ///< Reason when @c backend is null.
};

/// Defined by the engine library: builds the backend from the Cap'n Proto
/// flat-array message `config` (`config_len` bytes).
Created create(const void *config, std::size_t config_len);

/// Defined by the engine library: short name of the backend ("uma", "lj").
const char *backend_name();

} // namespace rgpot::engine
