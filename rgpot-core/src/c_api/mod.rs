// MIT License
// Copyright 2023--present rgpot developers

//! Public C API entry points.
//!
//! Each submodule exposes `extern "C"` functions that cbindgen collects into
//! `rgpot-core/include/rgpot.h`. All functions in this module follow three
//! invariants:
//!
//! 1. **Return [`rgpot_status_t`](crate::status::rgpot_status_t)** (or a
//!    pointer / void for constructors and destructors).
//! 2. **Wrap the body in [`catch_unwind`](crate::status::catch_unwind)** to
//!    prevent panics from crossing the FFI boundary.
//! 3. **Validate pointer arguments** and call
//!    [`set_last_error`](crate::status::set_last_error) before returning a
//!    non-success status.
//!
//! ## Submodules
//!
//! - [`types`] — Convenience constructors for
//!   [`rgpot_force_input_t`](crate::types::rgpot_force_input_t) and
//!   [`rgpot_force_out_t`](crate::types::rgpot_force_out_t).
//! - [`potential`] — Lifecycle functions for
//!   [`rgpot_potential_t`](crate::potential::rgpot_potential_t): create,
//!   calculate, free.
//! - [`hvp`] — Hessian-vector products of a potential handle: the product,
//!   the cached centre, the analytic callback and the counters.
//! - [`jet`] — Validated Taylor jets of a potential handle and their callback.
//! - [`rpc`] — RPC client functions (feature-gated on `rpc`): connect,
//!   calculate, disconnect.

pub mod hvp;
pub mod jet;
pub mod potential;
pub mod types;

#[cfg(feature = "rpc")]
pub mod rpc;

// Re-export eindir lifecycle functions at the c_api level so cbindgen picks them up.
#[cfg(feature = "schema")]
pub use crate::eindir::rgpot_potential_set_peer_capabilities;
pub use crate::eindir::{
    rgpot_eval_counts_t, rgpot_fused_settings_default, rgpot_fused_settings_t,
    rgpot_potential_eval_counts, rgpot_potential_free_eindir, rgpot_potential_fused_error_kind,
    rgpot_potential_new_eindir, rgpot_potential_reset_eval_counts,
    rgpot_potential_set_fused_settings,
};
pub use crate::fused::rgpot_fused_error_kind_t;
