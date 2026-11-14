// Copyright (c) anneal developers
//! `anneal._core`: anneal-core's Python module with the rgpot kernels.

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    anneal_core::install_rgpot_kernels!();
    anneal_core::python::register(m)
}
