//! pyo3 entry point for `anneal._core`. Re-exports the package version;
//! typed bindings land with Spec 2.

use pyo3::prelude::*;

/// pyo3 module initialiser. Exposed to Python as `anneal._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
