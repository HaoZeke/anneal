// Copyright (c) anneal developers

//! `Bounds` and `PyObjective` of `anneal._core`.
//!
//! Both hold `eindir_core` types. The classes that `eindir-core`'s
//! `py-objective` module exports build a box with `Bounds::new`, which
//! asserts, and score a Python callback with `expect`, so a bad box or a
//! callback that raises reaches Python as `PanicException`. These check
//! the box first and run callbacks through the drivers' own adapters.

use ndarray::ArrayView1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use eindir_core::Bounds;

use super::errors::{bounds_error, invalid_argument};
use super::{CallableObjective, CallablePyGradient, CallbackErrors, as_float_vector};

/// A box `[low, high]` with a `slack` tolerance on `contains`.
///
/// `low` and `high` must have the same length, be finite, have finite
/// widths, and satisfy `low <= high`; `slack` must be finite and not
/// negative. Any other box raises `BoundsError`.
#[pyclass(name = "Bounds", frozen)]
#[derive(Clone)]
pub struct PyBounds {
    pub(super) inner: Bounds<f64>,
}

#[pymethods]
impl PyBounds {
    #[new]
    #[pyo3(signature = (low, high, slack = 1e-6))]
    fn new(
        low: PyReadonlyArray1<'_, f64>,
        high: PyReadonlyArray1<'_, f64>,
        slack: f64,
    ) -> PyResult<Self> {
        if !(slack.is_finite() && slack >= 0.0) {
            return Err(bounds_error(format!(
                "slack must be finite and not negative, got {slack}"
            )));
        }
        let low = low.as_slice()?.to_vec();
        let high = high.as_slice()?.to_vec();
        Ok(Self {
            inner: super::checked_bounds(low, high, slack)?,
        })
    }

    #[getter]
    fn dims(&self) -> usize {
        self.inner.dims
    }

    #[getter]
    fn slack(&self) -> f64 {
        self.inner.slack
    }

    /// Whether `x` has the box's length and lies within `slack` of it.
    fn contains(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<bool> {
        Ok(self.inner.contains(ArrayView1::from(x.as_slice()?)))
    }

    /// `x` with every coordinate moved into the box. An `x` whose length is
    /// not the box's raises `InvalidArgumentError`.
    fn clip<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = x.as_slice()?;
        check_length("x", slice.len(), self.inner.dims)?;
        let clipped = self.inner.clip(ArrayView1::from(slice));
        Ok(PyArray1::from_vec(py, clipped.to_vec()))
    }
}

fn check_length(name: &str, len: usize, dims: usize) -> PyResult<()> {
    if len == dims {
        Ok(())
    } else {
        Err(invalid_argument(format!(
            "{name} has length {len} but the box has {dims} dimensions"
        )))
    }
}

/// A Python objective, with an optional gradient, over a `Bounds`.
///
/// The `*_objective` drivers call `fn_` and `grad_fn` as the other drivers
/// call `obj_fn` and `grad_fn`: an `Exception` is scored as the worst
/// value with a `RuntimeWarning`, and any other `BaseException` ends the
/// run and is raised from the driver.
#[pyclass(name = "PyObjective")]
pub struct PyObjective {
    fn_: Py<PyAny>,
    grad: Option<Py<PyAny>>,
    bounds: Bounds<f64>,
}

impl PyObjective {
    pub(super) fn dims(&self) -> usize {
        self.bounds.dims
    }

    pub(super) fn box_bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    /// The objective as the drivers' callback adapter.
    pub(super) fn callable(&self, py: Python<'_>) -> CallableObjective {
        CallableObjective::new(self.fn_.clone_ref(py), self.bounds.clone())
    }

    /// The gradient adapter, sharing `errors` with the objective. A driver
    /// that needs a gradient raises `InvalidArgumentError` without one.
    pub(super) fn gradient(
        &self,
        py: Python<'_>,
        errors: &std::sync::Arc<CallbackErrors>,
        driver: &str,
    ) -> PyResult<CallablePyGradient> {
        self.optional_gradient(py, errors).ok_or_else(|| {
            invalid_argument(format!(
                "{driver} integrates forces: build the PyObjective with grad_fn"
            ))
        })
    }

    pub(super) fn optional_gradient(
        &self,
        py: Python<'_>,
        errors: &std::sync::Arc<CallbackErrors>,
    ) -> Option<CallablePyGradient> {
        self.grad
            .as_ref()
            .map(|grad| CallablePyGradient::new(grad.clone_ref(py), self.bounds.dims, errors))
    }
}

#[pymethods]
impl PyObjective {
    #[new]
    #[pyo3(signature = (fn_, bounds, grad_fn = None))]
    fn new(fn_: Py<PyAny>, bounds: PyRef<'_, PyBounds>, grad_fn: Option<Py<PyAny>>) -> Self {
        Self {
            fn_,
            grad: grad_fn,
            bounds: bounds.inner.clone(),
        }
    }

    /// `fn_(x)` as a float. An `x` of the wrong length raises
    /// `InvalidArgumentError`; an exception of `fn_` propagates.
    fn eval(&self, py: Python<'_>, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let slice = x.as_slice()?;
        check_length("x", slice.len(), self.bounds.dims)?;
        self.fn_
            .call1(py, (PyArray1::from_slice(py, slice),))?
            .extract::<f64>(py)
    }

    /// `grad_fn(x)` as a float array of the box's length. Without a
    /// `grad_fn`, or for an `x` of the wrong length, raises
    /// `InvalidArgumentError`.
    fn grad<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let Some(grad) = &self.grad else {
            return Err(invalid_argument(
                "PyObjective was constructed without a grad_fn; pass grad_fn=... for a gradient",
            ));
        };
        let slice = x.as_slice()?;
        let dims = self.bounds.dims;
        check_length("x", slice.len(), dims)?;
        let value = grad.call1(py, (PyArray1::from_slice(py, slice),))?;
        match as_float_vector(py, &value) {
            Some(values) if values.len() == dims => Ok(PyArray1::from_vec(py, values)),
            _ => Err(invalid_argument(format!(
                "grad_fn must return {dims} numbers"
            ))),
        }
    }

    #[getter]
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    #[getter]
    fn bounds(&self) -> PyBounds {
        PyBounds {
            inner: self.bounds.clone(),
        }
    }
}
