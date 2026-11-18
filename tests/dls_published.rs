// Copyright (c) anneal developers
//! Tests of the published dynamic-lattice baseline.
//!
//! The baseline lives with its example, outside the library, because it
//! reproduces a competitor method and reads its size table from experiment
//! input. This target compiles that module so its behaviour tests run with
//! `cargo test`.

#![allow(dead_code)]

#[path = "../examples/lj_dls/dls_published.rs"]
mod dls_published;
