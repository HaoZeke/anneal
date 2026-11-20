// MIT License
// Copyright 2023--present rgpot developers

//! One staticlib for the rgpot C ABI and the fast pair list.
//!
//! The crates are named so the linker keeps the `#[no_mangle]` entry points
//! of both; the archive then exports what `librgpot_core.a` and
//! `librgpot_fastnlist.a` export, with a single copy of the Rust runtime.

pub use rgpot_core;
pub use rgpot_fastnlist;
