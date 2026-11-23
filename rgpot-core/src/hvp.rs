// MIT License
// Copyright 2023--present rgpot developers

//! Hessian-vector products of a potential handle.
//!
//! The product is `H v`, where `H` is the Hessian of the energy with respect
//! to the `3N` Cartesian positions and `v` is a direction laid out like the
//! positions (`[n_atoms, 3]`, row-major). A minimum-mode search (Lanczos,
//! dimer) applies `H` to many directions at one geometry. With only a force
//! callback each product is a difference of two force evaluations, and one
//! of them is the centre the search has not moved from.
//!
//! A handle keeps the forces of the last centre it evaluated in its fused
//! cache ([`crate::fused`]), keyed on the bits of the positions, the atomic
//! numbers, the cell and the engine settings. `k` forward products at one
//! centre therefore cost `k + 1` force calls instead of `2k`. A centre that
//! failed is stored as well, so asking for it again returns the failure
//! without a second call, unless the engine refused the request as
//! invalid. That failure is not stored, and the next request runs the
//! engine again. Another geometry, species list, cell or engine setting
//! is a new centre, and
//! [`rgpot_potential_t::invalidate_hvp_centre`] makes the next one new when
//! the potential itself changes. A potential that knows its Hessian
//! registers a [`HessianVectorCallback`] and the products make no force
//! call at all; one that differentiates its energy with dual numbers
//! registers its product as a second, dual-number callback.
//!
//! | Scheme | Force calls per product | Error |
//! |--------|-------------------------|-------|
//! | `RGPOT_HVP_FORWARD` | 1, plus 1 for each new centre | `O(h)` |
//! | `RGPOT_HVP_CENTRAL` | 2 | `O(h^2)` |
//! | `RGPOT_HVP_ANALYTIC` | 0 | that of the callback |
//! | `RGPOT_HVP_AD` | 0 | that of the callback |
//!
//! `RGPOT_HVP_AUTO` takes the analytic callback and falls back to forward
//! differences without one. `RGPOT_HVP_EXACT` takes the analytic callback,
//! else the dual-number one, and is refused rather than differenced. Either
//! way [`rgpot_potential_t::hessian_vector_resolved`] returns the scheme
//! that ran and [`rgpot_potential_t::derivative_counts`] counts it, so a
//! fallback is never silent.
//!
//! `h` is the length of the displacement along the unit direction
//! `v / |v|`, in the units of the positions. The difference is scaled back
//! by `|v|`, so the product is homogeneous in `|v|`; the forward scheme
//! is not linear in `v`, because its `O(h)` term keeps its sign when `v`
//! does. Neither finite difference can
//! be `O(h^2)` and cost one call per product: the forward error is
//! `(h / 2) T(u, u) |v|`, with `T` the third derivative, and cancelling it
//! takes the second evaluation that the central scheme makes.

use std::os::raw::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::c_api::potential::force_input_defect;
use crate::eindir::rgpot_potential_t;
use crate::fused::{lock, FusedError, FusedSource};
use crate::status::rgpot_status_t;
use crate::tensor::{copy_f64_tensor, copy_i32_tensor, rgpot_tensor_cpu_f64_2d, rgpot_tensor_free};
use crate::types::rgpot_force_input_t;

/// Analytic Hessian-vector product of a potential.
///
/// The callback receives the `user_data` of the potential handle, the
/// geometry, a direction of `3 * n_atoms` doubles and room for as many
/// doubles of output. It writes `H direction` to `hv_out` and returns
/// `RGPOT_SUCCESS`, or an error status code. `direction` and `hv_out` do
/// not overlap.
pub type HessianVectorCallback = unsafe extern "C" fn(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    hv_out: *mut f64,
) -> rgpot_status_t;

/// Displacement length used when a caller passes a step of 0.
pub const RGPOT_HVP_DEFAULT_STEP: f64 = 1.0e-3;

/// How a Hessian-vector product is computed.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum rgpot_hvp_scheme_t {
    /// The analytic callback when one is registered, otherwise forward.
    /// The scheme that ran is reported and counted.
    RGPOT_HVP_AUTO = 0,
    /// One displaced force call against the cached centre forces.
    RGPOT_HVP_FORWARD = 1,
    /// Two displaced force calls, one on each side of the centre.
    RGPOT_HVP_CENTRAL = 2,
    /// The registered callback. Refused when none is registered.
    RGPOT_HVP_ANALYTIC = 3,
    /// The registered dual-number callback. Refused when none is registered.
    RGPOT_HVP_AD = 4,
    /// The analytic callback, else the dual-number one. Refused when neither
    /// is registered; never a finite difference.
    RGPOT_HVP_EXACT = 5,
}

impl rgpot_hvp_scheme_t {
    /// The scheme with this value, or `None` for a value the enum lacks.
    pub fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            0 => Some(Self::RGPOT_HVP_AUTO),
            1 => Some(Self::RGPOT_HVP_FORWARD),
            2 => Some(Self::RGPOT_HVP_CENTRAL),
            3 => Some(Self::RGPOT_HVP_ANALYTIC),
            4 => Some(Self::RGPOT_HVP_AD),
            5 => Some(Self::RGPOT_HVP_EXACT),
            _ => None,
        }
    }
}

/// Work done by the Hessian-vector entry points of one handle.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct rgpot_hvp_counts_t {
    /// Products answered, a zero direction's zero product included.
    pub products: u64,
    /// Force callback invocations made for products and centres.
    pub force_calls: u64,
    /// Requests answered from the cached centre without a force call.
    pub centre_reused: u64,
    /// Products answered by the analytic callback.
    pub analytic: u64,
}

/// Why a Hessian-vector request failed.
#[derive(Debug, Clone, PartialEq)]
pub enum HvpError {
    /// A malformed argument or callback result. C callers see
    /// `RGPOT_INVALID_PARAMETER` and this sentence in `rgpot_last_error`.
    Invalid(String),
    /// A callback returned this status. Its own `rgpot_last_error`
    /// message, if it set one, is left in place.
    Callback {
        status: rgpot_status_t,
        message: String,
    },
}

impl HvpError {
    /// The status a C caller receives.
    pub fn status(&self) -> rgpot_status_t {
        match self {
            Self::Invalid(_) => rgpot_status_t::RGPOT_INVALID_PARAMETER,
            Self::Callback { status, .. } => *status,
        }
    }
}

impl std::fmt::Display for HvpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(why) => f.write_str(why),
            Self::Callback { status, message } => {
                write!(f, "{message} (callback returned {status:?})")
            }
        }
    }
}

impl std::error::Error for HvpError {}

pub(crate) fn invalid(why: impl Into<String>) -> HvpError {
    HvpError::Invalid(why.into())
}

/// Energy, variance and forces (`3 * n_atoms`, row-major) at a centre.
pub use crate::fused::FusedValue as HvpCentre;

/// Per-handle Hessian-vector state, kept in the handle's opaque cache.
#[derive(Default)]
pub(crate) struct HvpState {
    callback: Mutex<Option<HessianVectorCallback>>,
    ad_callback: Mutex<Option<HessianVectorCallback>>,
    products: AtomicU64,
    force_calls: AtomicU64,
    centre_reused: AtomicU64,
    analytic: AtomicU64,
    /// Products by the scheme they resolved to, zero directions included.
    pub(crate) by_forward: AtomicU64,
    pub(crate) by_central: AtomicU64,
    pub(crate) by_analytic: AtomicU64,
    pub(crate) by_ad: AtomicU64,
}

impl From<FusedError> for HvpError {
    fn from(error: FusedError) -> Self {
        match error {
            FusedError::Invalid { message } => Self::Invalid(message),
            other => Self::Callback {
                status: other.status(),
                message: other.to_string(),
            },
        }
    }
}

pub(crate) struct Geometry {
    pub(crate) n_atoms: usize,
    pub(crate) positions: Vec<f64>,
    atomic_numbers: Option<Vec<i32>>,
    cell: Option<Vec<f64>>,
}

/// Copy the positions, species and cell of `input` after checking them.
///
/// The input must pass `force_input_defect`, the check that
/// `rgpot_potential_calculate` makes, so a missing species buffer and an
/// empty system are refused here as they are on the C++ side. A missing
/// cell is left to the callback, so molecular potentials stay callable.
pub(crate) unsafe fn read_geometry(input: &rgpot_force_input_t) -> Result<Geometry, HvpError> {
    if let Some(why) = unsafe { force_input_defect(input) } {
        return Err(invalid(why));
    }
    let n_atoms = unsafe { input.n_atoms() }.ok_or_else(|| invalid("positions: tensor is NULL"))?;
    let dim = n_atoms
        .checked_mul(3)
        .ok_or_else(|| invalid("n_atoms * 3 overflows"))?;
    let positions = copy_f64_tensor(input.positions, "positions", dim, true).map_err(invalid)?;
    let atomic_numbers = if input.atomic_numbers.is_null() {
        None
    } else {
        Some(copy_i32_tensor(input.atomic_numbers, "atomic_numbers", n_atoms).map_err(invalid)?)
    };
    let cell = if input.box_matrix.is_null() {
        None
    } else {
        Some(copy_f64_tensor(input.box_matrix, "box_matrix", 9, false).map_err(invalid)?)
    };
    Ok(Geometry {
        n_atoms,
        positions,
        atomic_numbers,
        cell,
    })
}

/// `step`, or [`RGPOT_HVP_DEFAULT_STEP`] for 0, refused unless positive and
/// finite.
pub(crate) fn checked_step(step: f64) -> Result<f64, HvpError> {
    let h = if step == 0.0 {
        RGPOT_HVP_DEFAULT_STEP
    } else {
        step
    };
    if h.is_finite() && h > 0.0 {
        Ok(h)
    } else {
        Err(invalid(format!(
            "step must be positive and finite, got {step}"
        )))
    }
}

/// Call a derivative callback and check that the `out` it wrote is finite.
/// A failing status keeps the message the callback set with
/// `rgpot_set_last_error` during the call, or is described as
/// `{caller} returned <status>` when it set none; a
/// non-finite value is `{writer}: non-finite value at element <i>`. A
/// successful call leaves the earlier message in place.
pub(crate) fn run_callback(
    caller: &str,
    writer: &str,
    out: &mut [f64],
    call: impl FnOnce(*mut f64) -> rgpot_status_t,
) -> Result<(), HvpError> {
    let earlier = crate::status::last_error_message();
    crate::status::set_last_error("");
    let status = call(out.as_mut_ptr());
    if status != rgpot_status_t::RGPOT_SUCCESS {
        let message = crate::status::last_error_message();
        return Err(HvpError::Callback {
            status,
            message: if message.is_empty() {
                format!("{caller} returned {status:?}")
            } else {
                message
            },
        });
    }
    crate::status::set_last_error(&earlier);
    if let Some(i) = out.iter().position(|v| !v.is_finite()) {
        return Err(invalid(format!(
            "{writer}: non-finite value at element {i}"
        )));
    }
    Ok(())
}

/// The scheme a request runs: `AUTO` and `EXACT` resolved against the
/// registered callbacks, and the callback schemes refused without theirs.
fn resolve(
    scheme: rgpot_hvp_scheme_t,
    analytic: bool,
    ad: bool,
) -> Result<rgpot_hvp_scheme_t, HvpError> {
    use rgpot_hvp_scheme_t::*;
    match scheme {
        RGPOT_HVP_AUTO if analytic => Ok(RGPOT_HVP_ANALYTIC),
        RGPOT_HVP_AUTO => Ok(RGPOT_HVP_FORWARD),
        RGPOT_HVP_ANALYTIC if !analytic => Err(invalid(
            "RGPOT_HVP_ANALYTIC: no Hessian-vector callback is registered",
        )),
        RGPOT_HVP_AD if !ad => Err(invalid(
            "RGPOT_HVP_AD: no dual-number Hessian-vector callback is registered",
        )),
        RGPOT_HVP_EXACT if analytic => Ok(RGPOT_HVP_ANALYTIC),
        RGPOT_HVP_EXACT if ad => Ok(RGPOT_HVP_AD),
        RGPOT_HVP_EXACT => Err(invalid(
            "RGPOT_HVP_EXACT: neither an analytic nor a dual-number \
             Hessian-vector callback is registered",
        )),
        other => Ok(other),
    }
}

/// Euclidean norm, scaled so that large or tiny entries neither overflow
/// nor underflow the sum of squares.
fn norm(v: &[f64]) -> f64 {
    let scale = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if scale == 0.0 {
        return 0.0;
    }
    scale
        * v.iter()
            .map(|x| (x / scale) * (x / scale))
            .sum::<f64>()
            .sqrt()
}

impl rgpot_potential_t {
    /// Register `callback` as the analytic Hessian-vector product of this
    /// handle, or clear it with `None`. It is called with the handle's
    /// `user_data`.
    pub fn set_hessian_vector_callback(&self, callback: Option<HessianVectorCallback>) {
        *lock(&self.hvp_state().callback) = callback;
    }

    /// Whether an analytic Hessian-vector callback is registered.
    pub fn has_hessian_vector_callback(&self) -> bool {
        lock(&self.hvp_state().callback).is_some()
    }

    /// Register `callback` as the dual-number Hessian-vector product of
    /// this handle, or clear it with `None`. `RGPOT_HVP_AD` calls it, and
    /// `RGPOT_HVP_EXACT` when there is no analytic callback; `AUTO` never
    /// does.
    pub fn set_ad_hessian_vector_callback(&self, callback: Option<HessianVectorCallback>) {
        *lock(&self.hvp_state().ad_callback) = callback;
    }

    /// Whether a dual-number Hessian-vector callback is registered.
    pub fn has_ad_hessian_vector_callback(&self) -> bool {
        lock(&self.hvp_state().ad_callback).is_some()
    }

    /// Drop the cached centre. Call it when the potential behind the handle
    /// changes while the geometry does not.
    pub fn invalidate_hvp_centre(&self) {
        self.fused().invalidate();
    }

    /// The counters of the Hessian-vector entry points.
    pub fn hvp_counts(&self) -> rgpot_hvp_counts_t {
        let state = self.hvp_state();
        rgpot_hvp_counts_t {
            products: state.products.load(Ordering::Relaxed),
            force_calls: state.force_calls.load(Ordering::Relaxed),
            centre_reused: state.centre_reused.load(Ordering::Relaxed),
            analytic: state.analytic.load(Ordering::Relaxed),
        }
    }

    /// Zero the counters, those of [`Self::derivative_counts`] included.
    /// The cached centre stays.
    pub fn reset_hvp_counts(&self) {
        let state = self.hvp_state();
        for counter in [
            &state.products,
            &state.force_calls,
            &state.centre_reused,
            &state.analytic,
            &state.by_forward,
            &state.by_central,
            &state.by_analytic,
            &state.by_ad,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        self.reset_hessian_counts();
    }

    /// Energy and forces at the geometry of `input`.
    ///
    /// Served from the cached centre when its positions, species and cell
    /// carry the same bits, otherwise computed with one force call and
    /// cached. A search that needs the centre forces asks here instead of
    /// `rgpot_potential_calculate`, so the products that follow reuse them.
    ///
    /// # Safety
    ///
    /// The tensors of `input` must be valid for the call, and the handle's
    /// callback safe to call with its `user_data`.
    pub unsafe fn hvp_centre(&self, input: &rgpot_force_input_t) -> Result<HvpCentre, HvpError> {
        self.refuse_if_peer_refused()?;
        let geometry = unsafe { read_geometry(input) }?;
        let centre = unsafe { self.centre_of(input, &geometry) }?;
        Ok((*centre).clone())
    }

    /// Write `H direction` to `hv_out`; see the module documentation.
    ///
    /// [`Self::hessian_vector_resolved`] without the scheme that ran, which
    /// is still counted.
    ///
    /// # Safety
    ///
    /// As for [`Self::hessian_vector_resolved`].
    pub unsafe fn hessian_vector(
        &self,
        input: &rgpot_force_input_t,
        direction: &[f64],
        step: f64,
        scheme: rgpot_hvp_scheme_t,
        hv_out: &mut [f64],
    ) -> Result<(), HvpError> {
        unsafe { self.hessian_vector_resolved(input, direction, step, scheme, hv_out) }.map(|_| ())
    }

    /// Write `H direction` to `hv_out` and return the scheme that computed
    /// it: `AUTO` resolves to `ANALYTIC` or `FORWARD`, `EXACT` to
    /// `ANALYTIC` or `AD`, and any other scheme to itself.
    ///
    /// `direction` and `hv_out` hold `3 * n_atoms` doubles. `step` is the
    /// displacement length, and 0 selects [`RGPOT_HVP_DEFAULT_STEP`]. A zero
    /// direction gives a zero product without any call, counted under the
    /// scheme it resolved to. `hv_out` is only written on success by the
    /// finite differences; a callback may leave it partly written when it
    /// fails. A refused scheme makes no call.
    ///
    /// # Safety
    ///
    /// As for [`Self::hvp_centre`]; registered callbacks must also be safe
    /// to call with the handle's `user_data`.
    pub unsafe fn hessian_vector_resolved(
        &self,
        input: &rgpot_force_input_t,
        direction: &[f64],
        step: f64,
        scheme: rgpot_hvp_scheme_t,
        hv_out: &mut [f64],
    ) -> Result<rgpot_hvp_scheme_t, HvpError> {
        self.refuse_if_peer_refused()?;
        let geometry = unsafe { read_geometry(input) }?;
        let dim = geometry.positions.len();
        if direction.len() != dim {
            return Err(invalid(format!(
                "direction: expected {dim} elements, got {}",
                direction.len()
            )));
        }
        if hv_out.len() != dim {
            return Err(invalid(format!(
                "hv_out: expected {dim} elements, got {}",
                hv_out.len()
            )));
        }
        if let Some(i) = direction.iter().position(|v| !v.is_finite()) {
            return Err(invalid(format!(
                "direction: non-finite value at element {i}"
            )));
        }
        let h = checked_step(step)?;
        let state = self.hvp_state();
        let analytic = *lock(&state.callback);
        let ad = *lock(&state.ad_callback);
        let scheme = resolve(scheme, analytic.is_some(), ad.is_some())?;
        let length = norm(direction);
        if length == 0.0 {
            hv_out.fill(0.0);
            state.count(scheme);
            return Ok(scheme);
        }
        let displaced = |sign: f64| -> Vec<f64> {
            geometry
                .positions
                .iter()
                .zip(direction)
                .map(|(x, v)| x + sign * h * (v / length))
                .collect()
        };
        let call = |callback: HessianVectorCallback, hv: *mut f64| unsafe {
            callback(self.pot_user_data, input, direction.as_ptr(), hv)
        };
        match scheme {
            rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC => {
                let callback = analytic.expect("resolved against it");
                run_callback(
                    "analytic HVP callback",
                    "Hessian-vector callback",
                    hv_out,
                    |out| call(callback, out),
                )?;
                state.analytic.fetch_add(1, Ordering::Relaxed);
            }
            rgpot_hvp_scheme_t::RGPOT_HVP_AD => {
                let callback = ad.expect("resolved against it");
                run_callback(
                    "dual-number HVP callback",
                    "dual-number Hessian-vector callback",
                    hv_out,
                    |out| call(callback, out),
                )?;
            }
            rgpot_hvp_scheme_t::RGPOT_HVP_CENTRAL => {
                let calls = &state.force_calls;
                let plus =
                    unsafe { self.forces_at(input, displaced(1.0), geometry.n_atoms, calls) }?;
                let minus =
                    unsafe { self.forces_at(input, displaced(-1.0), geometry.n_atoms, calls) }?;
                let scale = length / (2.0 * h);
                for ((out, fm), fp) in hv_out.iter_mut().zip(&minus.forces).zip(&plus.forces) {
                    *out = (fm - fp) * scale;
                }
            }
            _ => {
                let centre = unsafe { self.centre_of(input, &geometry) }?;
                let plus = unsafe {
                    self.forces_at(input, displaced(1.0), geometry.n_atoms, &state.force_calls)
                }?;
                let scale = length / h;
                for ((out, f0), fp) in hv_out.iter_mut().zip(&centre.forces).zip(&plus.forces) {
                    *out = (f0 - fp) * scale;
                }
            }
        }
        state.count(scheme);
        Ok(scheme)
    }

    pub(crate) fn refuse_if_peer_refused(&self) -> Result<(), HvpError> {
        match self.fused_cache_peer_refusal() {
            Some(why) => Err(HvpError::Invalid(why)),
            None => Ok(()),
        }
    }

    unsafe fn centre_of(
        &self,
        input: &rgpot_force_input_t,
        geometry: &Geometry,
    ) -> Result<Arc<HvpCentre>, HvpError> {
        let state = self.hvp_state();
        let key = self.fused().key(
            &geometry.positions,
            geometry.atomic_numbers.as_deref(),
            geometry.cell.as_deref(),
        );
        let (outcome, source) = self.fused().evaluate(key, || {
            state.force_calls.fetch_add(1, Ordering::Relaxed);
            unsafe { self.run_engine(input, geometry.n_atoms) }
        });
        if source == FusedSource::Stored {
            state.centre_reused.fetch_add(1, Ordering::Relaxed);
        }
        Ok(outcome?)
    }

    /// Forces at `positions`, with the species and cell of `input`, counted
    /// in `calls`.
    pub(crate) unsafe fn forces_at(
        &self,
        input: &rgpot_force_input_t,
        mut positions: Vec<f64>,
        n_atoms: usize,
        calls: &AtomicU64,
    ) -> Result<HvpCentre, HvpError> {
        let moved = rgpot_force_input_t {
            positions: unsafe {
                rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), n_atoms as i64, 3)
            },
            atomic_numbers: input.atomic_numbers,
            box_matrix: input.box_matrix,
        };
        calls.fetch_add(1, Ordering::Relaxed);
        let result = unsafe { self.run_engine(&moved, n_atoms) }
            .map(|outcome| (*outcome).clone())
            .map_err(HvpError::from);
        unsafe { rgpot_tensor_free(moved.positions) };
        result
    }
}

impl HvpState {
    pub(crate) fn has_analytic(&self) -> bool {
        lock(&self.callback).is_some()
    }

    pub(crate) fn has_ad(&self) -> bool {
        lock(&self.ad_callback).is_some()
    }

    fn count(&self, scheme: rgpot_hvp_scheme_t) {
        self.products.fetch_add(1, Ordering::Relaxed);
        let by = match scheme {
            rgpot_hvp_scheme_t::RGPOT_HVP_FORWARD => &self.by_forward,
            rgpot_hvp_scheme_t::RGPOT_HVP_CENTRAL => &self.by_central,
            rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC => &self.by_analytic,
            rgpot_hvp_scheme_t::RGPOT_HVP_AD => &self.by_ad,
            rgpot_hvp_scheme_t::RGPOT_HVP_AUTO | rgpot_hvp_scheme_t::RGPOT_HVP_EXACT => {
                unreachable!("resolved before counting")
            }
        };
        by.fetch_add(1, Ordering::Relaxed);
    }
}
