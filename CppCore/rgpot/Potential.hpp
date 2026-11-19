#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Base classes and templates for chemical potentials.
 *
 * Provides the abstract interface and CRTP template for all potential energy
 * surfaces. Handles the high-level logic for caching, hashing, and force call
 * registration.
 */

// clang-format off
#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
// clang-format on

#ifdef RGPOT_HAS_CACHE
#define XXH_INLINE_ALL
#include "rgpot/CalculatorGroup.hpp"
#include "rgpot/PotentialCache.hpp"
#include <xxhash.h>
#endif

#include "rgpot/ForceStructs.hpp"
#include "rgpot/PotHelpers.hpp"
#include "rgpot/pot_caps.hpp"
#include "rgpot/pot_types.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace rgpot {

/**
 * @class PotentialBase
 * @brief Abstract base class for all potential energy surfaces.
 */
class PotentialBase {
public:
  /**
   * @brief Constructor for PotentialBase.
   * @param inp_type The type of the potential.
   */
  explicit PotentialBase(PotType inp_type) : m_type(inp_type) {}

  /**
   * @brief Virtual destructor.
   */
  virtual ~PotentialBase() = default;

  /**
   * @brief Main interface for potential and force calculation.
   * @param positions The atomic coordinates.
   * @param atmtypes The atomic numbers.
   * @param box The simulation cell vectors.
   * @return A pair containing the energy and the force matrix.
   */
  /// Energy, forces, and optional uncertainty scale (eV). See derived
  /// Potential::operator() for variance semantics.
  virtual std::tuple<double, types::AtomMatrix, double>
  operator()(const types::AtomMatrix &positions,
             const std::vector<int> &atmtypes,
             const std::array<std::array<double, 3>, 3> &box) = 0;

  /**
   * @brief Concurrency and evaluation capabilities of this potential.
   *
   * Multi-threaded callers (NEB images, dimer endpoints) derive their
   * sharing and cloning policy from this instead of hard-coded lists.
   */
  [[nodiscard]] virtual PotCaps caps() const noexcept { return {}; }

  /// Exact energy-Hessian product. Implementations advertising
  /// `caps().analyticHvp` override this hook; callers guarantee that the
  /// direction and output buffers (`3 * in.nAtoms` doubles each) do not
  /// overlap.
  virtual void hessianVectorImpl(const ForceInput &, const double *,
                                 double *) const {
    throw std::runtime_error("potential has no analytic Hessian-vector product");
  }

  /**
   * @brief Evaluate a batch of independent systems in one call.
   *
   * Callers running multi-system methods (NEB images, dimer endpoints)
   * use this instead of a loop over @c operator(), so a kernel that can
   * evaluate many systems at once -- a GPU model batching into a single
   * forward pass, say -- gets the chance to.
   *
   * Every kernel supports this call. @c caps().batched reports whether it
   * is served natively or by the default per-system loop, which callers
   * read when deciding whether batching is worth the marshalling.
   *
   * Results land in @c batch.out; see ForceBatch for the buffer contract.
   * Every slot is zeroed before any kernel runs, so a reused buffer never
   * carries an earlier call's results, even when this call throws.
   */
  virtual void forceBatch(const ForceBatch & /*batch*/) {
    throw std::runtime_error("PotentialBase::forceBatch called directly");
  }

  /**
   * @brief Fingerprint of the potential's parameter set.
   *
   * Mixed into the result-cache key so results never cross parameter
   * sets. Implementations hash their config field by field (see
   * ParamHash.hpp) plus a kernel-version salt that changes whenever the
   * numerics change. The default (0) suits parameter-free potentials
   * whose numerics never changed.
   */
  [[nodiscard]] virtual uint64_t paramsKey() const noexcept { return 0; }

#ifdef RGPOT_HAS_CACHE
  /**
   * @brief Sets the computation cache.
   * @param c Pointer to a PotentialCache instance.
   * @return Void.
   */
  virtual void setCache(rgpot::cache::PotentialCache * /*c*/) {
    throw std::runtime_error("PotentialBase::setCache called directly");
  }
#endif

  /**
   * @brief Fetches the potential type.
   * @return The potential type.
   */
  [[nodiscard]] PotType getType() const { return m_type; }

protected:
  PotType m_type; //!< The type of the potential energy surface.
};

/**
 * @class Potential
 * @brief Template class for specific potential implementations.
 *
 * Uses the Curiously Recurring Template Pattern to provide static
 * polymorphism for the internal @c forceImpl call.
 */
template <typename Derived>
class Potential : public PotentialBase, public registry<Derived> {
public:
  using PotentialBase::PotentialBase;

#ifdef RGPOT_HAS_CACHE
  /**
   * @brief Sets the computation cache for the specific implementation.
   * @param c Pointer to a PotentialCache instance.
   * @return Void.
   */
  void setCache(rgpot::cache::PotentialCache *c) override { _cache = c; }
#endif

  /**
   * @brief Implements the potential and force calculation logic.
   *
   * This method manages the transformation of @c Eigen matrices into
   * flat @c double arrays.
   *
   * # Caching Logic
   * If @c RGPOT_HAS_CACHE is defined, the method:
   * 1. Generates a @c XXH3_64bits hash of positions, types, and box.
   * 2. Checks the @c rocksdb backend for a hit.
   * 3. Returns cached values if present, otherwise computes and stores results.
   *
   * @param positions The atomic coordinates.
   * @param atmtypes The atomic numbers.
   * @param box The simulation cell vectors.
   * @return Energy, force matrix, and optional model/orientation variance
   *         (e.g. metatomic energy_uncertainty mean or multi-rotation energy
   *         sample variance). Variance is 0 when unused.
   */
  std::tuple<double, types::AtomMatrix, double>
  operator()(const types::AtomMatrix &positions,
             const std::vector<int> &atmtypes,
             const std::array<std::array<double, 3>, 3> &box) override {
    size_t nAtoms = positions.rows();
    types::AtomMatrix forces = types::AtomMatrix::Zero(nAtoms, 3);

    double flatBox[9];
    static_assert(sizeof(box) == 9 * sizeof(double));
    // Nested std::array is contiguous; &box[0][0] is invalid on MSVC (C2676)
    // and box.data() can hit incomplete-type issues depending on include order.
    std::memcpy(flatBox, static_cast<const void *>(&box), sizeof(flatBox));

    ForceInput fi{.nAtoms = nAtoms,
                  .pos = positions.data(),
                  .atmnrs = atmtypes.data(),
                  .box = flatBox};
    ForceOut fo{.F = forces.data(),
                .energy = 0.0,
                .variance = 0.0,
                .stress = {},
                .has_stress = 0};

#ifdef RGPOT_HAS_CACHE
    rgpot::cache::KeyHash key = cacheKey(fi);

    // Cache Read
    if (_cache) {
      auto hit = _cache->find(key);
      if (caps().groupCollective) {
        // Every rank must agree before any skips the collective call.
        unsigned char have = hit ? 1 : 0;
        calculatorAgree(&have, 1);
        if (!have) {
          hit.reset();
        }
      }
      if (hit) {
        _cache->deserialize_hit(*hit, fo, nAtoms);
        _cache->note_served();
        return {fo.energy, std::move(forces), fo.variance};
      }
    }

    // Computation
    static_cast<Derived *>(this)->forceImpl(fi, &fo);
    registry<Derived>::incrementForceCalls();

    // Cache Write
    if (_cache) {
      _cache->note_computed();
      _cache->add_serialized(key, fo, nAtoms);
    }
#else
    // Fallback when caching is disabled
    static_cast<Derived *>(this)->forceImpl(fi, &fo);
    registry<Derived>::incrementForceCalls();
#endif

    return {fo.energy, std::move(forces), fo.variance};
  }

  /**
   * @brief Abstract hook for the actual implementation.
   *
   * @c operator() and @c forceBatch hand over @c out zeroed: all
   * @c 3*in.nAtoms entries of @c F, @c energy, @c variance, @c stress and
   * @c has_stress. An override may assume that, but must not rely on what
   * an earlier call left in the buffer. It may throw with @c out partly
   * written.
   *
   * @param in Structure containing coordinates and cell info.
   * @param out Pointer to the results structure.
   * @return Void.
   */
  virtual void forceImpl(const ForceInput &in, ForceOut *out) const = 0;

  /**
   * @brief Batched counterpart of @c forceImpl.
   *
   * The default evaluates the systems one at a time, so every kernel
   * answers a batch call correctly without writing anything. Override it
   * where the kernel can do better than a loop -- one model forward pass
   * over all systems, one device transfer, one neighbour-list build
   * across a shared topology -- and set @c caps().batched so callers know
   * the difference.
   *
   * An override must fill @c batch.out[i] for every i from @c batch.in[i]
   * and treat the systems as independent: no ordering, no shared state,
   * and no assumption that they have equal atom counts. Each
   * @c batch.out[i] arrives zeroed, as for @c forceImpl.
   */
  virtual void forceBatchImpl(const ForceBatch &batch) const {
    for (size_t i = 0; i < batch.nSystems; ++i) {
      static_cast<const Derived *>(this)->forceImpl(batch.in[i], &batch.out[i]);
    }
  }

  /**
   * @brief Cache-aware batch entry point.
   *
   * Keeps the per-system semantics of @c operator(): each system is
   * hashed and looked up on its own, only the misses reach the kernel,
   * and each computed result is written back. Batching therefore composes
   * with caching rather than defeating it -- a band where two images have
   * not moved sends only the rest to the kernel.
   */
  void forceBatch(const ForceBatch &batch) override {
    if (batch.nSystems == 0) {
      return;
    }
    for (size_t i = 0; i < batch.nSystems; ++i) {
      ForceOut &out = batch.out[i];
      if (out.F) {
        std::fill_n(out.F, 3 * batch.in[i].nAtoms, 0.0);
      }
      out.energy = 0.0;
      out.variance = 0.0;
      out.has_stress = 0;
      std::fill(out.stress, out.stress + 9, 0.0);
    }
#ifdef RGPOT_HAS_CACHE
    if (_cache) {
      std::vector<size_t> misses;
      std::vector<rgpot::cache::KeyHash> keys;
      std::vector<std::optional<std::string>> hits;
      misses.reserve(batch.nSystems);
      keys.reserve(batch.nSystems);
      hits.reserve(batch.nSystems);
      for (size_t i = 0; i < batch.nSystems; ++i) {
        keys.push_back(cacheKey(batch.in[i]));
        hits.push_back(_cache->find(keys[i]));
        if (hits.back() && caps().stress &&
            !cache::PotentialCache::has_result_metadata(*hits.back(),
                                                        batch.in[i].nAtoms)) {
          hits.back().reset();
        }
      }
      if (caps().groupCollective) {
        // One joint decision per system, so a system any rank misses is
        // computed on every rank.
        std::vector<unsigned char> have(batch.nSystems);
        for (size_t i = 0; i < batch.nSystems; ++i) {
          have[i] = hits[i] ? 1 : 0;
        }
        calculatorAgree(have.data(), have.size());
        for (size_t i = 0; i < batch.nSystems; ++i) {
          if (!have[i]) {
            hits[i].reset();
          }
        }
      }

      for (size_t i = 0; i < batch.nSystems; ++i) {
        auto &hit = hits[i];
        if (hit) {
          _cache->deserialize_hit(*hit, batch.out[i], batch.in[i].nAtoms);
          _cache->note_served();
        } else {
          misses.push_back(i);
        }
      }

      if (misses.empty()) {
        return;
      }

      // Compact the misses so the kernel sees one contiguous batch.
      std::vector<ForceInput> missIn;
      std::vector<ForceOut> missOut;
      missIn.reserve(misses.size());
      missOut.reserve(misses.size());
      for (size_t idx : misses) {
        missIn.push_back(batch.in[idx]);
        missOut.push_back(batch.out[idx]);
      }
      ForceBatch missBatch{.nSystems = missIn.size(),
                           .in = missIn.data(),
                           .out = missOut.data()};
      static_cast<Derived *>(this)->forceBatchImpl(missBatch);

      for (size_t j = 0; j < misses.size(); ++j) {
        const size_t idx = misses[j];
        batch.out[idx].energy = missOut[j].energy;
        batch.out[idx].variance = missOut[j].variance;
        batch.out[idx].has_stress = missOut[j].has_stress;
        std::copy_n(missOut[j].stress, 9, batch.out[idx].stress);
        registry<Derived>::incrementForceCalls();
        _cache->note_computed();
        _cache->add_serialized(keys[idx], batch.out[idx], batch.in[idx].nAtoms);
      }
      return;
    }
#endif
    static_cast<Derived *>(this)->forceBatchImpl(batch);
    for (size_t i = 0; i < batch.nSystems; ++i) {
      registry<Derived>::incrementForceCalls();
    }
  }

private:
#ifdef RGPOT_HAS_CACHE
  /**
   * @brief Result-cache key for one system.
   *
   * Shared by the single and batched entry points so the two can never
   * disagree about what identifies a result. Mixes the geometry, the
   * species, the cell (with a null box hashed as a non-periodic marker
   * distinct from any 9-double box), the potential type and the parameter
   * fingerprint, so neither a different config nor a different kernel can
   * collide.
   */
  [[nodiscard]] rgpot::cache::KeyHash cacheKey(const ForceInput &fi) const {
    size_t hash_val = 0;
    hash_val ^= XXH3_64bits(fi.pos, fi.nAtoms * 3 * sizeof(double));
    hash_val ^= XXH3_64bits(fi.atmnrs, fi.nAtoms * sizeof(int));
    if (fi.box != nullptr) {
      hash_val ^= XXH3_64bits(fi.box, 9 * sizeof(double));
    } else {
      static constexpr char kNonPeriodic[] = "rgpot::non-periodic";
      hash_val ^= XXH3_64bits(kNonPeriodic, sizeof(kNonPeriodic));
    }
    size_t type_val = static_cast<size_t>(m_type);
    hash_val ^= XXH3_64bits(&type_val, sizeof(size_t));
    // Parameter fingerprint: two instances of one PotType with different
    // configs must never share cache entries.
    uint64_t params_val = paramsKey();
    hash_val ^= XXH3_64bits(&params_val, sizeof(params_val));
    return rgpot::cache::KeyHash(hash_val);
  }

  rgpot::cache::PotentialCache *_cache =
      nullptr; //!< Pointer to the optional calculation cache.
#endif
};

} // namespace rgpot
