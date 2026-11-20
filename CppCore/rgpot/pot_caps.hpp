#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Capability descriptors for potential implementations.
 *
 * Consumers that evaluate potentials from multiple threads (NEB images,
 * dimer endpoints, batched searches) read these instead of maintaining
 * per-potential blocklists.
 */

#include <cstdint>

namespace rgpot {

/**
 * @brief How a potential tolerates concurrent evaluation.
 */
enum class Reentrancy : uint8_t {
  /// One instance may be called from many threads concurrently.
  SharedInstance,
  /// Concurrent evaluation needs one instance per thread (per-instance
  /// native handles or scratch state).
  PerInstance,
  /// Global or COMMON-block state: all evaluations in the process must be
  /// serialized regardless of instance count.
  ProcessSerial,
};

/**
 * @brief Capability set a potential advertises to its callers.
 */
struct PotCaps {
  Reentrancy reentrancy = Reentrancy::SharedInstance;
  /// Callers running multi-image methods (NEB) should clone one instance
  /// per image even when reentrancy alone would allow sharing.
  bool perImageInstances = false;
  /// Native multi-system batch evaluation exists (forceBatch-style hook).
  bool batched = false;
  /// Kernel supports periodic boundary conditions.
  bool periodic = true;
  /// Every rank of a calculator group enters each force call together (an
  /// MPI engine whose call runs collectives on the group's communicator).
  /// The result cache then takes its hit-or-miss decision jointly over the
  /// group: a hit on one rank and a miss on another would leave the missing
  /// rank alone in a collective. Other groups take no part, so a host may
  /// hand groups uneven batches.
  bool groupCollective = false;
  /// forceImpl writes Cauchy stress for a positive cell volume:
  /// sigma = (1/V) dE/dstrain in energy/length^3, row-major in ForceOut.
  /// IPIPot also keeps the tensor for copyStress.
  bool stress = false;
  /// An exact Hessian-vector product is available without differencing.
  bool analyticHvp = false;
  /// hessianImpl writes the dense Hessian from closed-form derivatives.
  bool analyticHessian = false;
  /// hessianAdImpl writes the dense Hessian from dual-number evaluation of
  /// the energy kernel.
  bool adHessian = false;
  /// hessianVectorAdImpl writes the Hessian-vector product from dual-number
  /// evaluation of the energy kernel.
  bool adHvp = false;
};

} // namespace rgpot
