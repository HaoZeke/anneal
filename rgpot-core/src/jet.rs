// MIT License
// Copyright 2023--present rgpot developers

//! Validated Taylor jets of a potential along a line.
//!
//! For a geometry `x`, a direction `v` laid out like the positions
//! (`[n_atoms, 3]`, row-major, in the potential's length unit) and an order
//! `N`, the jet is the list of intervals `[lo_k, hi_k]`, `k = 0..=N`, with
//!
//! ```text
//! lo_k <= D^k V(x)[v, ..., v] / k! <= hi_k
//! ```
//!
//! where `V` is the real function the potential's source defines when every
//! literal is read as the value its compiler assigns it and every operation
//! is exact. The statement is about that function, not about the
//! floating-point program, so a caller can cite it as a bound on the
//! surface.
//!
//! A potential that cannot enclose its coefficients this way has no jet and
//! answers [`JetError::NoJet`] (`RGPOT_NO_JET`). It never returns a
//! differenced or sampled jet in its place.
//!
//! Branches of the source are decided on the enclosure of the tested
//! quantity: an enclosure strictly on one side takes that side, and one that
//! contains the threshold fails with [`JetError::Branch`]
//! (`RGPOT_JET_BRANCH`). A function evaluated where its Taylor recurrence
//! divides by an enclosure of zero, or outside its domain, fails with
//! [`JetError::Domain`] (`RGPOT_JET_DOMAIN`). Neither failure is retried by
//! another method.
//!
//! A handle answers jets through a registered [`rgpot_taylor_jet_fn`].
//! [`crate::taylor`] carries the interval series that implement one.

use std::fmt;
use std::os::raw::c_void;
use std::sync::{Mutex, MutexGuard};

use crate::eindir::rgpot_potential_t;
use crate::status::{rgpot_last_error, rgpot_status_t, set_last_error};
use crate::tensor::copy_f64_tensor;
use crate::types::rgpot_force_input_t;

/// One coefficient enclosure, `lo <= c_k <= hi`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct rgpot_interval_t {
    pub lo: f64,
    pub hi: f64,
}

/// Why a potential returns no jet.
#[derive(Debug, Clone, PartialEq)]
pub enum JetError {
    /// The potential has no exact jet.
    NoJet,
    /// A branch test's enclosure contains its threshold.
    Branch {
        test: &'static str,
        lo: f64,
        hi: f64,
    },
    /// An elementary function or division left its domain.
    Domain {
        function: &'static str,
        lo: f64,
        hi: f64,
    },
    /// Positions, direction or order do not fit the potential.
    Shape(&'static str),
}

impl JetError {
    /// The status a C caller receives.
    pub fn status(&self) -> rgpot_status_t {
        match self {
            Self::NoJet => rgpot_status_t::RGPOT_NO_JET,
            Self::Branch { .. } => rgpot_status_t::RGPOT_JET_BRANCH,
            Self::Domain { .. } => rgpot_status_t::RGPOT_JET_DOMAIN,
            Self::Shape(_) => rgpot_status_t::RGPOT_INVALID_PARAMETER,
        }
    }
}

impl fmt::Display for JetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoJet => f.write_str("the potential has no exact Taylor jet"),
            Self::Branch { test, lo, hi } => {
                write!(f, "branch {test}: [{lo:e}, {hi:e}] contains the threshold")
            }
            Self::Domain { function, lo, hi } => {
                write!(f, "{function} at [{lo:e}, {hi:e}] is outside its domain")
            }
            Self::Shape(what) => write!(f, "Taylor jet shape: {what}"),
        }
    }
}

impl std::error::Error for JetError {}

/// A potential with an exact jet.
pub trait TaylorJet {
    /// Enclosures of the coefficients `0..=order` of `V(positions + t direction)`.
    fn taylor_jet(
        &self,
        positions: &[f64],
        direction: &[f64],
        order: usize,
    ) -> Result<Vec<rgpot_interval_t>, JetError>;
}

/// Taylor jet callback of a potential handle.
///
/// It receives the handle's `user_data`, the geometry, a direction of
/// `3 * n_atoms` doubles and room for `order + 1` lower and `order + 1`
/// upper bounds; `lower` and `upper` do not overlap. It writes the
/// enclosures and returns `RGPOT_SUCCESS`, or returns `RGPOT_NO_JET`,
/// `RGPOT_JET_BRANCH`, `RGPOT_JET_DOMAIN` or another error status after
/// describing the failure in `rgpot_last_error`.
pub type rgpot_taylor_jet_fn = unsafe extern "C" fn(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    order: usize,
    lower: *mut f64,
    upper: *mut f64,
) -> rgpot_status_t;

/// A failed jet request on a handle: the status a C caller receives and the
/// message left in `rgpot_last_error`.
#[derive(Debug, Clone, PartialEq)]
pub struct JetFailure {
    pub status: rgpot_status_t,
    pub message: String,
}

impl JetFailure {
    fn new(status: rgpot_status_t, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new(rgpot_status_t::RGPOT_INVALID_PARAMETER, message)
    }
}

impl fmt::Display for JetFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.status, self.message)
    }
}

impl std::error::Error for JetFailure {}

/// Per-handle jet state, kept in the handle's opaque cache.
#[derive(Default)]
pub(crate) struct JetState {
    callback: Mutex<Option<rgpot_taylor_jet_fn>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The number of coefficients, `order + 1`, once the `lower` and `upper`
/// output ranges of that many doubles are known to be disjoint.
///
/// Both ranges are written through mutable slices, which must not alias, so
/// this runs on the raw addresses before any reference to them exists. A
/// count whose byte size overflows is `RGPOT_BUFFER_SIZE_ERROR`; ranges
/// that overlap or wrap the address space are `RGPOT_INVALID_PARAMETER`.
pub(crate) fn disjoint_outputs(
    lower: *const f64,
    upper: *const f64,
    order: usize,
) -> Result<usize, JetFailure> {
    let overflow = || {
        JetFailure::new(
            rgpot_status_t::RGPOT_BUFFER_SIZE_ERROR,
            "order + 1 coefficients overflow",
        )
    };
    let count = order.checked_add(1).ok_or_else(overflow)?;
    let bytes = count
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(overflow)?;
    let (lo, hi) = (lower as usize, upper as usize);
    let (Some(lo_end), Some(hi_end)) = (lo.checked_add(bytes), hi.checked_add(bytes)) else {
        return Err(JetFailure::invalid(
            "lower or upper wraps the address space",
        ));
    };
    if lo < hi_end && hi < lo_end {
        return Err(JetFailure::invalid("lower and upper overlap"));
    }
    Ok(count)
}

fn last_error() -> String {
    let ptr = unsafe { rgpot_last_error() };
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

impl rgpot_potential_t {
    /// Register `jet` as the Taylor jet of this handle, or clear it with
    /// `None`. It is called with the handle's `user_data`.
    pub fn set_taylor_jet(&self, jet: Option<rgpot_taylor_jet_fn>) {
        *lock(&self.jet_state().callback) = jet;
    }

    /// Whether a Taylor jet callback is registered.
    pub fn has_taylor_jet(&self) -> bool {
        lock(&self.jet_state().callback).is_some()
    }

    /// The jet of the potential at the geometry of `input` along `direction`
    /// through `order`; see the module documentation.
    ///
    /// Without a registered callback the answer is `RGPOT_NO_JET`. A
    /// callback's failure is returned with its status and the message it
    /// left in `rgpot_last_error`. A success whose bounds are NaN or
    /// reversed is refused as `RGPOT_INVALID_PARAMETER`.
    ///
    /// # Safety
    ///
    /// The tensors of `input` must be valid for the call, and a registered
    /// callback safe to call with the handle's `user_data`.
    pub unsafe fn taylor_jet(
        &self,
        input: &rgpot_force_input_t,
        direction: &[f64],
        order: usize,
    ) -> Result<Vec<rgpot_interval_t>, JetFailure> {
        if let Some(why) = self.fused_cache_peer_refusal() {
            return Err(JetFailure::invalid(why));
        }
        let dim = unsafe { input.n_atoms() }
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| JetFailure::invalid("positions: tensor is NULL or too large"))?;
        if direction.len() != dim {
            return Err(JetFailure::invalid(format!(
                "direction: expected {dim} elements, got {}",
                direction.len()
            )));
        }
        if let Some(i) = direction.iter().position(|v| !v.is_finite()) {
            return Err(JetFailure::invalid(format!(
                "direction: non-finite value at element {i}"
            )));
        }
        let count = order.checked_add(1).ok_or_else(|| {
            JetFailure::new(
                rgpot_status_t::RGPOT_BUFFER_SIZE_ERROR,
                "order + 1 coefficients overflow",
            )
        })?;
        let Some(callback) = *lock(&self.jet_state().callback) else {
            return Err(JetFailure::new(
                rgpot_status_t::RGPOT_NO_JET,
                "no Taylor jet is registered on this handle",
            ));
        };
        let mut lower = vec![f64::NAN; count];
        let mut upper = vec![f64::NAN; count];
        let status = unsafe {
            callback(
                self.pot_user_data,
                input,
                direction.as_ptr(),
                order,
                lower.as_mut_ptr(),
                upper.as_mut_ptr(),
            )
        };
        if status != rgpot_status_t::RGPOT_SUCCESS {
            return Err(JetFailure::new(status, last_error()));
        }
        lower
            .into_iter()
            .zip(upper)
            .enumerate()
            .map(|(k, (lo, hi))| {
                if lo <= hi {
                    Ok(rgpot_interval_t { lo, hi })
                } else {
                    Err(JetFailure::invalid(format!(
                        "jet callback returned [{lo:e}, {hi:e}] for coefficient {k}"
                    )))
                }
            })
            .collect()
    }
}

/// Answer a jet callback from a [`TaylorJet`]: read the positions of
/// `input`, call `jet`, and write the bounds or the failure.
///
/// # Safety
///
/// NULL pointers and overlapping `lower` and `upper` return
/// `RGPOT_INVALID_PARAMETER` before `jet` is called. Otherwise `input` must
/// point to a valid `rgpot_force_input_t`, `direction` to `3 * n_atoms`
/// readable doubles, and `lower` and `upper` to `order + 1` writable doubles
/// each.
pub unsafe fn serve_taylor_jet<J: TaylorJet>(
    jet: &J,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    order: usize,
    lower: *mut f64,
    upper: *mut f64,
) -> rgpot_status_t {
    if input.is_null() || direction.is_null() || lower.is_null() || upper.is_null() {
        set_last_error("Taylor jet: NULL argument");
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let count = match disjoint_outputs(lower, upper, order) {
        Ok(count) => count,
        Err(failure) => {
            set_last_error(&format!("Taylor jet: {}", failure.message));
            return failure.status;
        }
    };
    let input = unsafe { &*input };
    let Some(dim) = unsafe { input.n_atoms() }.and_then(|n| n.checked_mul(3)) else {
        set_last_error("Taylor jet: positions tensor is NULL or too large");
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    };
    let positions = match copy_f64_tensor(input.positions, "positions", dim, true) {
        Ok(positions) => positions,
        Err(why) => {
            set_last_error(&why);
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
    };
    let direction = unsafe { std::slice::from_raw_parts(direction, dim) };
    match jet.taylor_jet(&positions, direction, order) {
        Ok(coefficients) if coefficients.len() == count => {
            let lower = unsafe { std::slice::from_raw_parts_mut(lower, coefficients.len()) };
            let upper = unsafe { std::slice::from_raw_parts_mut(upper, coefficients.len()) };
            for (k, c) in coefficients.iter().enumerate() {
                lower[k] = c.lo;
                upper[k] = c.hi;
            }
            rgpot_status_t::RGPOT_SUCCESS
        }
        Ok(coefficients) => {
            set_last_error(&format!(
                "Taylor jet: {} coefficients for order {order}",
                coefficients.len()
            ));
            rgpot_status_t::RGPOT_INTERNAL_ERROR
        }
        Err(err) => {
            set_last_error(&err.to_string());
            err.status()
        }
    }
}
