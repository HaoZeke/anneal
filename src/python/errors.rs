// Copyright (c) anneal developers

//! Typed exceptions for `anneal`.
//!
//! Every exception derives from `ValueError`, so code that catches
//! `ValueError` keeps working. The classes tell a caller what to fix:
//! a bad argument, a bad box, a configuration the library rejects, or a
//! search that found no finite start.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::methods::cluster_hopping::{RemovedConfigKeyError as RemovedKey, SearchError};

pyo3::create_exception!(
    anneal,
    AnnealError,
    PyValueError,
    "Base class of every error that `anneal` raises for a rejected input or a failed search."
);
pyo3::create_exception!(
    anneal,
    InvalidArgumentError,
    AnnealError,
    "An argument has a value, length or combination that the call cannot use."
);
pyo3::create_exception!(
    anneal,
    BoundsError,
    InvalidArgumentError,
    "The box is empty, has mismatched sides, or has a bound that is not finite or not ordered."
);
pyo3::create_exception!(
    anneal,
    ConfigError,
    AnnealError,
    "A configuration payload could not be read. The `key` attribute names the offending key."
);
pyo3::create_exception!(
    anneal,
    RemovedConfigKeyError,
    ConfigError,
    "A configuration sets a key that a past release removed."
);
pyo3::create_exception!(
    anneal,
    NoFiniteStartError,
    AnnealError,
    "No initial quench was finite. The `attempts` attribute counts the quenches tried."
);
pyo3::create_exception!(
    anneal,
    FeatureUnavailableError,
    AnnealError,
    "The call needs a Cargo feature that this build of `anneal` does not include."
);

pub(crate) fn invalid_argument(msg: impl Into<String>) -> PyErr {
    InvalidArgumentError::new_err(msg.into())
}

pub(crate) fn anneal_error(msg: impl Into<String>) -> PyErr {
    AnnealError::new_err(msg.into())
}

pub(crate) fn bounds_error(msg: impl Into<String>) -> PyErr {
    BoundsError::new_err(msg.into())
}

#[cfg(not(feature = "graphkey"))]
pub(crate) fn feature_unavailable(msg: impl Into<String>) -> PyErr {
    FeatureUnavailableError::new_err(msg.into())
}

/// Map a rejected configuration payload to its exception, with `key` set.
pub(crate) fn config_error(err: &RemovedKey) -> PyErr {
    let parse_failure = err.key == "<json>" || err.key == "<toml>";
    let msg = err.message.clone();
    let py_err = if parse_failure {
        ConfigError::new_err(msg)
    } else {
        RemovedConfigKeyError::new_err(msg)
    };
    Python::attach(|py| {
        let _ = py_err.value(py).setattr("key", err.key.as_str());
    });
    py_err
}

/// Map a failed search to its exception.
pub(crate) fn search_error(err: &SearchError) -> PyErr {
    match err {
        SearchError::NoFiniteStart { attempts } => {
            let py_err = NoFiniteStartError::new_err(err.to_string());
            Python::attach(|py| {
                let _ = py_err.value(py).setattr("attempts", *attempts);
            });
            py_err
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("AnnealError", py.get_type::<AnnealError>())?;
    m.add(
        "InvalidArgumentError",
        py.get_type::<InvalidArgumentError>(),
    )?;
    m.add("BoundsError", py.get_type::<BoundsError>())?;
    m.add("ConfigError", py.get_type::<ConfigError>())?;
    m.add(
        "RemovedConfigKeyError",
        py.get_type::<RemovedConfigKeyError>(),
    )?;
    m.add("NoFiniteStartError", py.get_type::<NoFiniteStartError>())?;
    m.add(
        "FeatureUnavailableError",
        py.get_type::<FeatureUnavailableError>(),
    )?;
    Ok(())
}
