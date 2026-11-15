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
//! A handle keeps the forces of the last centre it evaluated, keyed on the
//! bits of the positions, the atomic numbers and the cell. `k` forward
//! products at one centre therefore cost `k + 1` force calls instead of
//! `2k`. Another geometry, species list or cell is a new centre, and
//! [`rgpot_potential_t::invalidate_hvp_centre`] drops the cached one when
//! the potential itself changes. A potential that knows its Hessian
//! registers a [`HessianVectorCallback`] and the products make no force
//! call at all.
//!
//! | Scheme | Force calls per product | Error |
//! |--------|-------------------------|-------|
//! | `RGPOT_HVP_FORWARD` | 1, plus 1 for each new centre | `O(h)` |
//! | `RGPOT_HVP_CENTRAL` | 2 | `O(h^2)` |
//! | `RGPOT_HVP_ANALYTIC` | 0 | that of the callback |
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
use std::sync::{Arc, Mutex, MutexGuard};

use crate::eindir::{rgpot_potential_t, validated_result};
use crate::status::rgpot_status_t;
use crate::tensor::{copy_f64_tensor, copy_i32_tensor, rgpot_tensor_cpu_f64_2d, rgpot_tensor_free};
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

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
    RGPOT_HVP_AUTO = 0,
    /// One displaced force call against the cached centre forces.
    RGPOT_HVP_FORWARD = 1,
    /// Two displaced force calls, one on each side of the centre.
    RGPOT_HVP_CENTRAL = 2,
    /// The registered callback. Refused when none is registered.
    RGPOT_HVP_ANALYTIC = 3,
}

impl rgpot_hvp_scheme_t {
    /// The scheme with this value, or `None` for a value the enum lacks.
    pub fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            0 => Some(Self::RGPOT_HVP_AUTO),
            1 => Some(Self::RGPOT_HVP_FORWARD),
            2 => Some(Self::RGPOT_HVP_CENTRAL),
            3 => Some(Self::RGPOT_HVP_ANALYTIC),
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
    Callback(rgpot_status_t),
}

impl HvpError {
    /// The status a C caller receives.
    pub fn status(&self) -> rgpot_status_t {
        match self {
            Self::Invalid(_) => rgpot_status_t::RGPOT_INVALID_PARAMETER,
            Self::Callback(status) => *status,
        }
    }
}

impl std::fmt::Display for HvpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(why) => f.write_str(why),
            Self::Callback(status) => write!(f, "callback returned {status:?}"),
        }
    }
}

impl std::error::Error for HvpError {}

fn invalid(why: impl Into<String>) -> HvpError {
    HvpError::Invalid(why.into())
}

/// Energy, variance and forces (`3 * n_atoms`, row-major) at a centre.
#[derive(Debug, Clone, PartialEq)]
pub struct HvpCentre {
    pub energy: f64,
    pub variance: f64,
    pub forces: Vec<f64>,
}

/// `f64::to_bits` of every coordinate and cell entry: float equality would
/// serve the forces of `+0.0` at `-0.0`.
#[derive(Clone, PartialEq)]
struct GeometryKey {
    positions: Vec<u64>,
    atomic_numbers: Option<Vec<i32>>,
    box_matrix: Option<Vec<u64>>,
}

struct CachedCentre {
    key: GeometryKey,
    centre: Arc<HvpCentre>,
}

/// Per-handle Hessian-vector state, kept in the handle's opaque cache.
#[derive(Default)]
pub(crate) struct HvpState {
    callback: Mutex<Option<HessianVectorCallback>>,
    centre: Mutex<Option<CachedCentre>>,
    products: AtomicU64,
    force_calls: AtomicU64,
    centre_reused: AtomicU64,
    analytic: AtomicU64,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl HvpState {
    pub(crate) fn invalidate(&self) {
        *lock(&self.centre) = None;
    }
}

struct Geometry {
    n_atoms: usize,
    positions: Vec<f64>,
    key: GeometryKey,
}

/// Copy the positions, species and cell of `input` after checking them.
/// Species and cell are optional, as for the force callback.
unsafe fn read_geometry(input: &rgpot_force_input_t) -> Result<Geometry, HvpError> {
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
    let box_matrix = if input.box_matrix.is_null() {
        None
    } else {
        let cell = copy_f64_tensor(input.box_matrix, "box_matrix", 9, false).map_err(invalid)?;
        Some(cell.iter().map(|v| v.to_bits()).collect())
    };
    let key = GeometryKey {
        positions: positions.iter().map(|v| v.to_bits()).collect(),
        atomic_numbers,
        box_matrix,
    };
    Ok(Geometry {
        n_atoms,
        positions,
        key,
    })
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

    /// Drop the cached centre. Call it when the potential behind the handle
    /// changes while the geometry does not.
    pub fn invalidate_hvp_centre(&self) {
        self.hvp_state().invalidate();
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

    /// Zero the counters. The cached centre stays.
    pub fn reset_hvp_counts(&self) {
        let state = self.hvp_state();
        state.products.store(0, Ordering::Relaxed);
        state.force_calls.store(0, Ordering::Relaxed);
        state.centre_reused.store(0, Ordering::Relaxed);
        state.analytic.store(0, Ordering::Relaxed);
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
    /// `direction` and `hv_out` hold `3 * n_atoms` doubles. `step` is the
    /// displacement length, and 0 selects [`RGPOT_HVP_DEFAULT_STEP`]. A zero
    /// direction gives a zero product without any call. `hv_out` is only
    /// written on success by the finite differences; the analytic callback
    /// may leave it partly written when it fails.
    ///
    /// # Safety
    ///
    /// As for [`Self::hvp_centre`]; a registered analytic callback must also
    /// be safe to call with the handle's `user_data`.
    pub unsafe fn hessian_vector(
        &self,
        input: &rgpot_force_input_t,
        direction: &[f64],
        step: f64,
        scheme: rgpot_hvp_scheme_t,
        hv_out: &mut [f64],
    ) -> Result<(), HvpError> {
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
        let h = if step == 0.0 {
            RGPOT_HVP_DEFAULT_STEP
        } else {
            step
        };
        if !(h.is_finite() && h > 0.0) {
            return Err(invalid(format!(
                "step must be positive and finite, got {step}"
            )));
        }
        let state = self.hvp_state();
        let analytic = *lock(&state.callback);
        let scheme = match scheme {
            rgpot_hvp_scheme_t::RGPOT_HVP_AUTO if analytic.is_some() => {
                rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC
            }
            rgpot_hvp_scheme_t::RGPOT_HVP_AUTO => rgpot_hvp_scheme_t::RGPOT_HVP_FORWARD,
            other => other,
        };
        if scheme == rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC && analytic.is_none() {
            return Err(invalid(
                "RGPOT_HVP_ANALYTIC: no Hessian-vector callback is registered",
            ));
        }
        let length = norm(direction);
        if length == 0.0 {
            hv_out.fill(0.0);
            state.products.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        let displaced = |sign: f64| -> Vec<f64> {
            geometry
                .positions
                .iter()
                .zip(direction)
                .map(|(x, v)| x + sign * h * (v / length))
                .collect()
        };
        match scheme {
            rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC => {
                let callback = analytic.expect("checked above");
                let status = unsafe {
                    callback(
                        self.pot_user_data,
                        input,
                        direction.as_ptr(),
                        hv_out.as_mut_ptr(),
                    )
                };
                if status != rgpot_status_t::RGPOT_SUCCESS {
                    return Err(HvpError::Callback(status));
                }
                if let Some(i) = hv_out.iter().position(|v| !v.is_finite()) {
                    return Err(invalid(format!(
                        "Hessian-vector callback: non-finite value at element {i}"
                    )));
                }
                state.analytic.fetch_add(1, Ordering::Relaxed);
            }
            rgpot_hvp_scheme_t::RGPOT_HVP_CENTRAL => {
                let plus = unsafe { self.forces_at(input, displaced(1.0), geometry.n_atoms) }?;
                let minus = unsafe { self.forces_at(input, displaced(-1.0), geometry.n_atoms) }?;
                let scale = length / (2.0 * h);
                for ((out, fm), fp) in hv_out.iter_mut().zip(&minus.forces).zip(&plus.forces) {
                    *out = (fm - fp) * scale;
                }
            }
            _ => {
                let centre = unsafe { self.centre_of(input, &geometry) }?;
                let plus = unsafe { self.forces_at(input, displaced(1.0), geometry.n_atoms) }?;
                let scale = length / h;
                for ((out, f0), fp) in hv_out.iter_mut().zip(&centre.forces).zip(&plus.forces) {
                    *out = (f0 - fp) * scale;
                }
            }
        }
        state.products.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn refuse_if_peer_refused(&self) -> Result<(), HvpError> {
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
        if let Some(cached) = lock(&state.centre).as_ref() {
            if cached.key == geometry.key {
                state.centre_reused.fetch_add(1, Ordering::Relaxed);
                return Ok(Arc::clone(&cached.centre));
            }
        }
        let centre = Arc::new(unsafe { self.evaluate(input, geometry.n_atoms) }?);
        *lock(&state.centre) = Some(CachedCentre {
            key: geometry.key.clone(),
            centre: Arc::clone(&centre),
        });
        Ok(centre)
    }

    /// Forces at `positions`, with the species and cell of `input`.
    unsafe fn forces_at(
        &self,
        input: &rgpot_force_input_t,
        mut positions: Vec<f64>,
        n_atoms: usize,
    ) -> Result<HvpCentre, HvpError> {
        let moved = rgpot_force_input_t {
            positions: unsafe {
                rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), n_atoms as i64, 3)
            },
            atomic_numbers: input.atomic_numbers,
            box_matrix: input.box_matrix,
        };
        let result = unsafe { self.evaluate(&moved, n_atoms) };
        unsafe { rgpot_tensor_free(moved.positions) };
        result
    }

    unsafe fn evaluate(
        &self,
        input: &rgpot_force_input_t,
        n_atoms: usize,
    ) -> Result<HvpCentre, HvpError> {
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        self.hvp_state().force_calls.fetch_add(1, Ordering::Relaxed);
        let status = unsafe { (self.callback)(self.pot_user_data, input, &mut output) };
        let result = if status == rgpot_status_t::RGPOT_SUCCESS {
            validated_result(&output, 3 * n_atoms)
                .map(|forces| HvpCentre {
                    energy: output.energy,
                    variance: output.variance,
                    forces,
                })
                .map_err(invalid)
        } else {
            Err(HvpError::Callback(status))
        };
        unsafe { rgpot_tensor_free(output.forces) };
        result
    }
}
