#pragma once
// MIT License
// Copyright 2023--present rgpot developers
#include "ForceStructs.hpp"
#include <atomic>
#include <cstddef>

/**
 * @brief Utility templates and functions for potential management.
 *
 * Defines a static registry of atomic instance and force-call counters.
 * It also provides utility functions for structure initialization and
 * validation.
 */

namespace rgpot {

/**
 * @class registry
 * @brief Static registry for instance tracking and statistics.
 *
 * The registry counts active instances of a derived potential class
 * and how many times those instances have been evaluated.
 */
template <typename T> class registry {
public:
  /// Active-instance and force-call counters. Both are atomic so a batch
  /// worker can construct or evaluate a potential while another worker does
  /// the same (F2). There is no process-global list of instances: each DSO
  /// that includes this header would otherwise keep its own `head`, and the
  /// destructor in librgpot would write through a node freed by the Python
  /// extension.
  static std::atomic<size_t> count; //!< Total number of active instances.
  static std::atomic<size_t>
      forceCalls; //!< Global counter for force evaluations.
  T *prev = nullptr; //!< Unused. Kept so the potential object layout stays.
  T *next = nullptr; //!< Unused. Kept so the potential object layout stays.

protected:
  /**
   * @brief Default constructor.
   */
  registry() { count.fetch_add(1, std::memory_order_relaxed); }

  /**
   * @brief Copy constructor.
   * @param other The instance to copy.
   */
  registry(const registry &) { count.fetch_add(1, std::memory_order_relaxed); }

  /**
   * @brief Destructor.
   */
  ~registry() { count.fetch_sub(1, std::memory_order_relaxed); }

public:
  /**
   * @brief Increments the force call counter.
   * @return Void.
   */
  static void incrementForceCalls() {
    forceCalls.fetch_add(1, std::memory_order_relaxed);
  }
};

template <typename T> std::atomic<size_t> registry<T>::count{0};
template <typename T> std::atomic<size_t> registry<T>::forceCalls{0};

/**
 * @brief Zeroes the members of a ForceOut structure.
 * @param nAtoms The number of atoms.
 * @param efvd The results structure to reset.
 * @return Void.
 */
void zeroForceOut(const size_t &nAtoms, ForceOut *efvd);

/**
 * @brief Validates the input parameters for a potential calculation.
 * @param params The configuration structure to check.
 * @return Void.
 */
void checkParams(const ForceInput &params);

/// Same numeric value as @c RGPOT_PLUGIN_INVALID_PARAM and
/// @c RGPOT_INVALID_PARAMETER.
constexpr int kInvalidParameter = 1;

/**
 * @brief Store the reason of a failed force call on this thread.
 *
 * A null or empty @a reason does nothing. The plugin ABI describes
 * @c last_error as the message of the last failure and does not say a
 * later success clears it, so a successful call must not pass null here.
 * The pointer from @c forceLastError stays valid until the next
 * non-empty call on this thread.
 */
void setForceLastError(const char *reason) noexcept;

/// Reason stored by the last failed force call on this thread.
/// Empty until the first failure. Not cleared by a later success.
const char *forceLastError() noexcept;

/**
 * @brief Reject a force call that would read a null buffer.
 *
 * A system with no atoms is rejected first, with its own message, before
 * the force pointer is read. An empty force allocation has a null data
 * pointer, and that call must not be reported as a null force buffer.
 * After that, a null @a out or force array, a null position or species
 * pointer, or a null cell when @a periodic is set, stores a message and
 * returns @c kInvalidParameter. A usable call returns 0 and leaves the
 * stored message unchanged.
 */
int rejectForceCall(const ForceInput &in, const ForceOut *out, bool periodic);

/// Throw @c std::invalid_argument with the stored message when
/// @c rejectForceCall rejects @a in and @a out.
void guardForceCall(const ForceInput &in, const ForceOut *out, bool periodic);

/// @c pot.caps().periodic when the kernel reports it, otherwise true,
/// which is the default of @c PotCaps.
template <class Pot> bool kernelPeriodic(const Pot &pot) {
  if constexpr (requires { pot.caps().periodic; }) {
    return pot.caps().periodic;
  } else {
    return true;
  }
}

} // namespace rgpot
