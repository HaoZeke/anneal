#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file potential.hpp
 * @author rgpot Developers
 * @date 2026-02-15
 * @brief Move-only RAII handle for callback-backed potential calculations.
 *
 * Provides @c PotentialHandle, the primary C++ entry point for performing
 * force/energy calculations through the Rust core.  The handle owns an
 * opaque @c rgpot_potential_t pointer obtained from @c rgpot_potential_new()
 * or @c rgpot_potential_new_eindir() and releases it via @c
 * rgpot_potential_free() on destruction.
 *
 * Two factory functions are offered:
 *
 * - @c from_impl<Impl>() — wraps an existing C++ potential whose
 *   @c forceImpl() uses the legacy @c ForceInput / @c ForceOut structs
 *   (defined in @c ForceStructs.hpp).  A template trampoline extracts
 *   raw CPU pointers from the DLPack tensors and creates an owning DLPack
 *   forces tensor from the legacy output.
 * - @c from_callback() — registers a bare C function pointer directly.
 *
 * # Example
 * @code
 * rgpot::LJPot lj;
 * auto pot = rgpot::PotentialHandle::from_impl(lj);
 *
 * double pos[] = {0,0,0, 1,0,0};
 * int    atm[] = {1, 1};
 * double box[] = {10,0,0, 0,10,0, 0,0,10};
 * rgpot::InputSpec input(2, pos, atm, box);
 *
 * auto result = pot.calculate(input);
 * double energy = result.energy();
 * @endcode
 *
 * @ingroup rgpot_cpp
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "c_api.hpp"
#include "errors.hpp"
#include "rgpot/FusedSettings.hpp"
#include "rgpot/HessianVector.hpp"
#include "types.hpp"

namespace rgpot {

/**
 * @class PotentialHandle
 * @brief Move-only RAII handle around an opaque @c rgpot_potential_t pointer.
 * @ingroup rgpot_cpp
 *
 * The handle is non-copyable.  Move semantics transfer ownership of the
 * underlying Rust allocation.  On destruction the Rust core frees all
 * resources associated with the potential via @c rgpot_potential_free().
 */
class PotentialHandle final {
public:
  /**
   * @brief Adopt an existing raw handle.
   * @pre @a handle was obtained from @c rgpot_potential_new() or
   *      @c rgpot_potential_new_eindir() and has not been freed.
   * @param handle Raw pointer to the Rust-side potential object.
   */
  explicit PotentialHandle(rgpot_potential_t *handle) : handle_(handle) {}

  /**
   * @brief Destructor — releases the Rust-side allocation.
   *
   * Safe to call on a moved-from handle (internal pointer is @c nullptr).
   */
  ~PotentialHandle() {
    if (handle_) {
      rgpot_potential_free(handle_);
    }
  }

  /**
   * @brief Move constructor — transfers ownership.
   * @param other Handle to move from; left in a null state.
   */
  PotentialHandle(PotentialHandle &&other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  /**
   * @brief Move assignment — releases current handle, transfers ownership.
   * @param other Handle to move from; left in a null state.
   * @return Reference to @c *this.
   */
  PotentialHandle &operator=(PotentialHandle &&other) noexcept {
    if (this != &other) {
      if (handle_) {
        rgpot_potential_free(handle_);
      }
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  PotentialHandle(const PotentialHandle &) = delete;
  PotentialHandle &operator=(const PotentialHandle &) = delete;

  /**
   * @brief Perform a force/energy calculation.
   *
   * Creates a @c CalcResult, delegates to @c rgpot_potential_calculate(),
   * and checks the returned status code.  The callback sets the forces
   * tensor in the output struct.
   *
   * @param input The atomic configuration to evaluate.
   * @return A @c CalcResult containing energy, variance, and forces.
   * @throws rgpot::Error when the underlying callback reports failure.
   */
  CalcResult calculate(const InputSpec &input) {
    CalcResult result;
    auto status = rgpot_potential_calculate(handle_, &input.c_struct(),
                                            &result.c_struct());
    detail::check_status(status);
    return result;
  }

  /**
   * @brief Write @f$H v@f$ at @a input to @a hv_out.
   *
   * See @c rgpot_potential_hvp. @a direction and @a hv_out hold
   * @c 3*n_atoms doubles and must not overlap. With @c RGPOT_HVP_FORWARD,
   * @c k products at one geometry cost @c k+1 force calls.
   *
   * @param step Displacement length; 0 selects @c RGPOT_HVP_DEFAULT_STEP.
   * @throws rgpot::Error on a malformed request or a failing callback.
   */
  void hvp(const InputSpec &input, const double *direction, double *hv_out,
           double step = 0.0,
           rgpot_hvp_scheme_t scheme = RGPOT_HVP_AUTO) const {
    detail::check_status(
        rgpot_potential_hvp(handle_, &input.c_struct(), direction, step,
                            static_cast<int32_t>(scheme), hv_out));
  }

  /// @c hvp() into a new vector of @c 3*n_atoms doubles.
  std::vector<double> hvp(const InputSpec &input,
                          const std::vector<double> &direction,
                          double step = 0.0,
                          rgpot_hvp_scheme_t scheme = RGPOT_HVP_AUTO) const {
    if (direction.size() != 3 * input.n_atoms()) {
      throw Error("invalid parameter: direction must hold 3 * n_atoms values");
    }
    std::vector<double> out(direction.size());
    hvp(input, direction.data(), out.data(), step, scheme);
    return out;
  }

  /**
   * @brief Energy and forces at @a input, from the centre the products use.
   *
   * See @c rgpot_potential_hvp_centre. A search that needs the centre
   * forces asks here instead of @c calculate(), so the forward products
   * that follow make one force call each.
   */
  CalcResult hvp_centre(const InputSpec &input) const {
    CalcResult result;
    detail::check_status(rgpot_potential_hvp_centre(handle_, &input.c_struct(),
                                                    &result.c_struct()));
    return result;
  }

  /// Register or clear (@c nullptr) the analytic product of this handle.
  void set_hvp_callback(HessianVectorCallback callback) const {
    detail::check_status(rgpot_potential_set_hvp_callback(handle_, callback));
  }

  /// Drop the cached centre, for a potential that changed in place.
  void invalidate_hvp() const { rgpot_potential_hvp_invalidate(handle_); }

  /// Products, force calls, centre reuses and analytic products so far.
  rgpot_hvp_counts_t hvp_counts() const {
    rgpot_hvp_counts_t counts{};
    detail::check_status(rgpot_potential_hvp_counts(handle_, &counts));
    return counts;
  }

  void reset_hvp_counts() const { rgpot_potential_reset_hvp_counts(handle_); }

  /**
   * @brief Create a handle from a legacy C++ potential implementation.
   *
   * Registers a template trampoline that extracts raw CPU pointers from
   * the DLPack tensors in @c rgpot_force_input_t, constructs legacy
   * @c ForceInput / @c ForceOut types for @a Impl::forceImpl(), and
   * wraps the resulting forces in an owning DLPack tensor.
   *
   * The caller retains ownership of @a impl and @b must keep it alive
   * for the lifetime of the returned handle.
   *
   * @tparam Impl A type with
   *         @c void @c forceImpl(const @c ForceInput&, @c ForceOut*).
   * @param impl Reference to the C++ potential object.
   * @return A new @c PotentialHandle wrapping @a impl.
   *
   * The handle is one @c rgpot_potential_t and therefore one
   * @c eindir_objective_t (first-member IS-A). Child terms stay inside
   * @a impl; this factory does not build a list of objectives.
   *
   * The analytic callback is registered only when this specific instance
   * advertises @c caps().analyticHvp. This gate is done at registration time
   * on @a impl because @c caps() is an instance query.
   */
  template <typename Impl> static PotentialHandle from_impl(Impl &impl) {
    auto *handle = rgpot_potential_new_eindir(
        &trampoline_callback<Impl>, static_cast<void *>(&impl), nullptr, 0,
        nullptr, nullptr, nullptr, nullptr);
    return with_analytic_hvp<Impl>(
        with_fused_settings<Impl>(PotentialHandle(handle), impl), impl);
  }

  /**
   * @brief Wrap @a impl as one eindir objective with molecular context.
   *
   * Same trampoline as @c from_impl(Impl&), but calls
   * @c rgpot_potential_new_eindir with atom count, species, and cell so
   * hosts that eval through @c eindir_objective_t* have bounds and
   * species. @c rgpot_potential_calculate still uses the input tensors.
   */
  template <typename Impl>
  static PotentialHandle
  from_impl(Impl &impl, size_t n_atoms, const int32_t *atomic_numbers,
            const double *box_matrix, const double *bounds_low = nullptr,
            const double *bounds_high = nullptr) {
    auto *handle = rgpot_potential_new_eindir(
        &trampoline_callback<Impl>, static_cast<void *>(&impl), nullptr,
        n_atoms, atomic_numbers, box_matrix, bounds_low, bounds_high);
    return with_analytic_hvp<Impl>(
        with_fused_settings<Impl>(PotentialHandle(handle), impl), impl);
  }

  /**
   * @brief Create a handle from an explicit C callback and user data.
   *
   * This is the low-level factory; prefer @c from_impl() for C++
   * potentials.
   *
   * @param callback Function pointer matching the @c rgpot_potential_t
   *                 callback signature.
   * @param user_data Opaque pointer forwarded to every @a callback
   *                  invocation.
   * @param free_fn  Optional destructor for @a user_data (may be
   *                 @c nullptr if the caller manages the lifetime).
   * @return A new @c PotentialHandle.
   */
  static PotentialHandle
  from_callback(rgpot_status_t (*callback)(void *, const rgpot_force_input_t *,
                                           rgpot_force_out_t *),
                void *user_data, void (*free_fn)(void *) = nullptr) {
    auto *handle = rgpot_potential_new(callback, user_data, free_fn);
    return PotentialHandle(handle);
  }

  /**
   * @brief Install the engine settings that key the fused cache.
   *
   * @c from_impl does this once for a potential with @c fusedSettings().
   * Call it again after changing a setting of the potential: the changed
   * setting is a new key, so the next request runs the engine even at
   * positions whose outcome, a failure included, is stored. Without the
   * call the handle keeps answering a stored failure.
   *
   * @throws rgpot::Error when the core refuses the settings.
   */
  void set_fused_settings(const FusedSettingsSpec &spec) const {
    rgpot_fused_settings_t c{};
    rgpot_fused_settings_default(&c);
    c.periodic = spec.periodic;
    c.spin = spec.spin;
    c.scc_max_iterations = spec.sccMaxIterations;
    c.charge = spec.charge;
    c.accuracy = spec.accuracy;
    c.electronic_temperature = spec.electronicTemperature;
    c.scc_mixer = spec.sccMixer.c_str();
    c.engine_build = spec.engineBuild.c_str();
    detail::check_status(rgpot_potential_set_fused_settings(handle_, &c));
  }

  /// @c set_fused_settings(impl.fusedSettings()).
  template <ProvidesFusedSettings Impl>
  void sync_fused_settings(const Impl &impl) const {
    set_fused_settings(impl.fusedSettings());
  }

  /// Kind of the failure the fused cache of this handle stores, or
  /// @c RGPOT_FUSED_OK when it stores none.
  [[nodiscard]] rgpot_fused_error_kind_t fused_error_kind() const {
    return rgpot_potential_fused_error_kind(handle_);
  }

  /**
   * @brief Access the raw opaque handle for C interop.
   * @return Pointer to the underlying @c rgpot_potential_t.
   */
  rgpot_potential_t *raw() const { return handle_; }

private:
  rgpot_potential_t *handle_; //!< Owned opaque handle (may be @c nullptr
                              //!< after move).

  /// Raw CPU pointers of the DLPack tensors in @a input, as a legacy
  /// @c ForceInput. The positions tensor must be present.
  static ::rgpot::ForceInput legacy_input(const rgpot_force_input_t *input) {
    auto *pos_tensor = &input->positions->dl_tensor;
    return ::rgpot::ForceInput{
        .nAtoms = static_cast<size_t>(pos_tensor->shape[0]),
        .pos = static_cast<const double *>(pos_tensor->data),
        .atmnrs = input->atomic_numbers
                      ? static_cast<const int *>(
                            input->atomic_numbers->dl_tensor.data)
                      : nullptr,
        .box = input->box_matrix ? static_cast<const double *>(
                                       input->box_matrix->dl_tensor.data)
                                 : nullptr};
  }

  /**
   * @brief Template trampoline bridging DLPack-based C ABI types to legacy
   *        C++ types.
   *
   * 1. Extracts raw CPU pointers from DLPack tensors in the input.
   * 2. Constructs legacy @c ForceInput and @c ForceOut.
   * 3. Calls @c Impl::forceImpl().
   * 4. Wraps the output forces in an owning DLPack tensor via
   *    @c rgpot_tensor_owned_cpu_f64_2d() (since the local
   *    @c std::vector goes out of scope).
   *
   * @tparam Impl The concrete potential type.
   * @param user_data Pointer to the @a Impl instance (cast from @c void*).
   * @param input     Pointer to the C input struct (DLPack tensors).
   * @param output    Pointer to the C output struct.
   * @return @c RGPOT_SUCCESS or @c RGPOT_INTERNAL_ERROR.
   */
  template <typename Impl>
  static rgpot_status_t trampoline_callback(void *user_data,
                                            const rgpot_force_input_t *input,
                                            rgpot_force_out_t *output) {
    try {
      auto *self = static_cast<Impl *>(user_data);
      if (!input->positions) {
        return RGPOT_INVALID_PARAMETER;
      }
      const ::rgpot::ForceInput fi = legacy_input(input);
      const size_t n_atoms = fi.nAtoms;

      // Temporary buffer for forces
      std::vector<double> forces(n_atoms * 3, 0.0);
      ::rgpot::ForceOut fo{.F = forces.data(),
                           .energy = 0.0,
                           .variance = 0.0,
                           .stress = {},
                           .has_stress = 0};

      self->forceImpl(fi, &fo);

      output->energy = fo.energy;
      output->variance = fo.variance;

      // Create an owning DLPack tensor that copies the forces data.
      // This is necessary because `forces` is a local vector that goes
      // out of scope when this function returns.
      output->forces = rgpot_tensor_owned_cpu_f64_2d(
          forces.data(), static_cast<int64_t>(n_atoms), 3);

      return RGPOT_SUCCESS;
    } catch (const ::rgpot::SccNotConverged &e) {
      rgpot_set_last_error(e.what());
      return RGPOT_SCC_NOT_CONVERGED;
    } catch (const std::exception &e) {
      rgpot_set_last_error(e.what());
      return RGPOT_INTERNAL_ERROR;
    } catch (...) {
      rgpot_set_last_error("unknown exception in the potential");
      return RGPOT_INTERNAL_ERROR;
    }
  }

  /// @c HessianVectorCallback over @c Impl::hessianVectorImpl.
  template <typename Impl>
  static rgpot_status_t
  hvp_trampoline(void *user_data, const rgpot_force_input_t *input,
                 const double *direction, double *hv_out) {
    try {
      if (!input->positions) {
        return RGPOT_INVALID_PARAMETER;
      }
      const auto *self = static_cast<const Impl *>(user_data);
      self->hessianVectorImpl(legacy_input(input), direction, hv_out);
      return RGPOT_SUCCESS;
    } catch (const ::rgpot::SccNotConverged &e) {
      rgpot_set_last_error(e.what());
      return RGPOT_SCC_NOT_CONVERGED;
    } catch (const std::exception &e) {
      rgpot_set_last_error(e.what());
      return RGPOT_INTERNAL_ERROR;
    } catch (...) {
      rgpot_set_last_error("unknown exception in the potential HVP");
      return RGPOT_INTERNAL_ERROR;
    }
  }

  /// Install the settings @a impl reports, when it reports any.
  template <typename Impl>
  static PotentialHandle with_fused_settings(PotentialHandle handle,
                                             const Impl &impl) {
    if constexpr (::rgpot::ProvidesFusedSettings<Impl>) {
      if (handle.raw() != nullptr) {
        handle.set_fused_settings(impl.fusedSettings());
      }
    }
    return handle;
  }

  template <typename Impl>
  static PotentialHandle with_analytic_hvp(PotentialHandle handle,
                                           const Impl &impl) {
    if constexpr (::rgpot::AnalyticHessianVector<Impl>) {
      if (handle.raw() != nullptr && impl.caps().analyticHvp) {
        if constexpr (requires(const Impl &self, const ::rgpot::ForceInput &in,
                               const double *direction, double *product) {
                        self.hessianVectorImpl(in, direction, product);
                      }) {
          handle.set_hvp_callback(&hvp_trampoline<Impl>);
        } else {
          throw std::invalid_argument("from_impl: caps().analyticHvp is true "
                                      "but hessianVectorImpl is missing");
        }
      }
    }
    return handle;
  }
};

} // namespace rgpot
