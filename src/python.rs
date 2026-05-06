//! pyo3 entry point for `anneal._core`. Exposes the three preset variants
//! (Boltzmann / Fast / Gsa), a `History` wrapper, and a `run` function
//! that dispatches on the preset type. The Python objective is a callable
//! `f: ndarray -> float` plus a `Bounds` for sampling and dimensionality.
//!
//! Internally, `run` wraps the Python callable in a thin `Objective<f64>`
//! adapter that re-acquires the GIL per `eval`. This is acceptable when
//! evaluation cost dominates the per-call GIL overhead (~hundreds of ns),
//! which is true for any non-toy objective.

use ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use eindir_core::{Bounds, Objective};

use crate::history::History;
use crate::variant::{boltzmann, fast, gsa};

// ---------------------------------------------------------------------------
// Preset parameter holders.
// ---------------------------------------------------------------------------

/// Boltzmann preset parameters: initial temperature and Gaussian step size.
#[pyclass(name = "Boltzmann")]
#[derive(Clone, Copy, Debug)]
pub struct PyBoltzmann {
    /// Initial temperature `T_0`.
    #[pyo3(get, set)]
    pub t_init: f64,
    /// Gaussian per-component standard deviation.
    #[pyo3(get, set)]
    pub sigma: f64,
}

#[pymethods]
impl PyBoltzmann {
    #[new]
    #[pyo3(signature = (t_init = 1.0, sigma = 0.5))]
    fn new(t_init: f64, sigma: f64) -> Self {
        Self { t_init, sigma }
    }

    fn __repr__(&self) -> String {
        format!(
            "Boltzmann(t_init={:?}, sigma={:?})",
            self.t_init, self.sigma
        )
    }
}

/// Fast SA preset parameters: initial temperature and Cauchy scale.
#[pyclass(name = "Fast")]
#[derive(Clone, Copy, Debug)]
pub struct PyFast {
    /// Initial temperature `T_0`.
    #[pyo3(get, set)]
    pub t_init: f64,
    /// Cauchy per-component scale.
    #[pyo3(get, set)]
    pub gamma: f64,
}

#[pymethods]
impl PyFast {
    #[new]
    #[pyo3(signature = (t_init = 1.0, gamma = 0.5))]
    fn new(t_init: f64, gamma: f64) -> Self {
        Self { t_init, gamma }
    }

    fn __repr__(&self) -> String {
        format!("Fast(t_init={:?}, gamma={:?})", self.t_init, self.gamma)
    }
}

/// GSA preset parameters: initial temperature and Tsallis indices.
#[pyclass(name = "Gsa")]
#[derive(Clone, Copy, Debug)]
pub struct PyGsa {
    /// Initial temperature `T_0`.
    #[pyo3(get, set)]
    pub t_init: f64,
    /// Visiting index `q_v in (1, 3)`.
    #[pyo3(get, set)]
    pub q_v: f64,
    /// Acceptance index `q_a` (`1.0` collapses to Metropolis).
    #[pyo3(get, set)]
    pub q_a: f64,
}

#[pymethods]
impl PyGsa {
    #[new]
    #[pyo3(signature = (t_init = 1.0, q_v = 2.62, q_a = 1.7))]
    fn new(t_init: f64, q_v: f64, q_a: f64) -> Self {
        Self { t_init, q_v, q_a }
    }

    fn __repr__(&self) -> String {
        format!(
            "Gsa(t_init={:?}, q_v={:?}, q_a={:?})",
            self.t_init, self.q_v, self.q_a
        )
    }
}

// ---------------------------------------------------------------------------
// History wrapper.
// ---------------------------------------------------------------------------

/// Per-epoch summary line exposed to Python.
#[pyclass(name = "EpochLine")]
#[derive(Clone, Debug)]
pub struct PyEpochLine {
    /// Zero-based epoch index.
    #[pyo3(get)]
    pub epoch: usize,
    /// Temperature at this epoch.
    #[pyo3(get)]
    pub temp: f64,
    /// Proposals accepted in this epoch.
    #[pyo3(get)]
    pub accepted: usize,
    /// Proposals rejected in this epoch.
    #[pyo3(get)]
    pub rejected: usize,
    /// Best objective value seen up to and including this epoch.
    #[pyo3(get)]
    pub best_val: f64,
}

/// Run history exposed to Python.
#[pyclass(name = "History")]
#[derive(Clone, Debug)]
pub struct PyHistory {
    /// Per-epoch summary lines, in epoch order.
    #[pyo3(get)]
    pub epochs: Vec<PyEpochLine>,
    /// Best position seen across the entire run.
    #[pyo3(get)]
    pub best_pos: Vec<f64>,
    /// Best objective value seen across the entire run.
    #[pyo3(get)]
    pub best_val: f64,
}

#[pymethods]
impl PyHistory {
    /// Total proposals accepted across all epochs.
    #[getter]
    fn total_accepted(&self) -> usize {
        self.epochs.iter().map(|e| e.accepted).sum()
    }

    /// Total proposals rejected across all epochs.
    #[getter]
    fn total_rejected(&self) -> usize {
        self.epochs.iter().map(|e| e.rejected).sum()
    }
}

impl From<History> for PyHistory {
    fn from(h: History) -> Self {
        Self {
            epochs: h
                .epochs
                .into_iter()
                .map(|e| PyEpochLine {
                    epoch: e.epoch,
                    temp: e.temp,
                    accepted: e.accepted,
                    rejected: e.rejected,
                    best_val: e.best_val,
                })
                .collect(),
            best_pos: h.best.pos.to_vec(),
            best_val: h.best.val,
        }
    }
}

// ---------------------------------------------------------------------------
// Objective adapter wrapping a Python callable.
// ---------------------------------------------------------------------------

/// Internal objective adapter: a Python callable plus a `Bounds`.
/// Holds the GIL per `eval` call. `Py<PyAny>` is `Send + Sync` because
/// pyo3 ref-counts atomically and gates dereferencing on the GIL.
struct CallableObjective {
    fn_: Py<PyAny>,
    bounds: Bounds<f64>,
}

impl Objective<f64> for CallableObjective {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        Python::attach(|py| {
            let owned: Vec<f64> = x.iter().copied().collect();
            let py_arr = PyArray1::from_vec(py, owned);
            let r = self
                .fn_
                .call1(py, (py_arr,))
                .expect("anneal.run: objective callable raised");
            r.extract::<f64>(py)
                .expect("anneal.run: objective callable returned non-float")
        })
    }
}

/// Internal gradient adapter: a Python callable that takes a numpy
/// array `x` and returns a numpy array `grad f(x)` of the same shape.
/// Ceres-style plug-in: PyTorch users pass a callable that does
/// `tensor.requires_grad_(True); f(tensor).backward(); return tensor.grad.numpy()`,
/// JAX users pass `jax.grad(f)`, and analytic users pass a hand-coded
/// gradient. The pyo3 surface remains agnostic.
struct CallablePyGradient {
    fn_: Py<PyAny>,
    dim: usize,
}

impl crate::grad::Gradient<f64> for CallablePyGradient {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        Python::attach(|py| {
            let owned: Vec<f64> = x.iter().copied().collect();
            let py_arr = PyArray1::from_vec(py, owned);
            let r = self
                .fn_
                .call1(py, (py_arr,))
                .expect("anneal.run_hmc: gradient callable raised");
            let arr = r
                .extract::<numpy::PyReadonlyArray1<f64>>(py)
                .expect("anneal.run_hmc: gradient callable returned non-array");
            Array1::from_vec(arr.as_slice().expect("contiguous").to_vec())
        })
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Runs HMC-driven SA with a user-supplied gradient and returns a `History`.
///
/// Args:
///   obj_fn: Python callable `f(numpy.ndarray) -> float`.
///   grad_fn: Python callable `g(numpy.ndarray) -> numpy.ndarray`. Pass
///            `jax.grad(f)`, a torch `.backward()`-based wrapper, or
///            a hand-coded analytic gradient. Ceres-style plug-in.
///   low, high: numpy box bounds.
///   t_init: initial temperature for the log-cooling schedule.
///   epsilon: Omelyan integrator step size.
///   l_steps: number of Omelyan steps per HMC trajectory.
///   q: Tsallis momentum index. Default `1.0` (Gaussian). Values
///      `q in (1, 1 + 2/dim)` enable q-Gaussian momentum -- heavy-tailed
///      draws that help HMC-SA escape local cups on multimodal
///      objectives. For `dim=5`, `q < 1.4`; for `dim=10`, `q < 1.2`.
///   n_epochs, steps_per_epoch, seed.
#[pyfunction]
#[pyo3(signature = (obj_fn, grad_fn, low, high, t_init = 5.0, epsilon = 0.05, l_steps = 5, q = 1.0, n_epochs = 100, steps_per_epoch = 50, seed = 42))]
#[allow(clippy::too_many_arguments)]
fn run_hmc(
    obj_fn: Py<PyAny>,
    grad_fn: Py<PyAny>,
    low: PyReadonlyArray1<'_, f64>,
    high: PyReadonlyArray1<'_, f64>,
    t_init: f64,
    epsilon: f64,
    l_steps: usize,
    q: f64,
    n_epochs: usize,
    steps_per_epoch: usize,
    seed: u64,
) -> PyResult<PyHistory> {
    use crate::cool::LogCool;
    use crate::hmc::{GaussianMomentum, HmcSaSampler, OmelyanIntegrator, QGaussianMomentum};

    let low_vec = low.as_slice()?.to_vec();
    let high_vec = high.as_slice()?.to_vec();
    if low_vec.len() != high_vec.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "low and high must have the same length",
        ));
    }
    let dim = low_vec.len();
    let q_max = 1.0 + 2.0 / dim as f64;
    if q > 1.0 && q >= q_max {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "q-Gaussian momentum requires q < 1 + 2/dim = {} for dim = {}; got q = {}",
            q_max, dim, q
        )));
    }
    let bounds = Bounds::new(Array1::from_vec(low_vec), Array1::from_vec(high_vec), 1e-9);
    let obj = CallableObjective {
        fn_: obj_fn,
        bounds,
    };
    let grad = CallablePyGradient { fn_: grad_fn, dim };
    let cool = LogCool::new(t_init, 2.0_f64);
    let integrator = OmelyanIntegrator::new(epsilon, l_steps, t_init);
    let history = if q <= 1.0 + 1e-9 {
        let sampler =
            HmcSaSampler::with_momentum(obj, grad, cool.clone(), GaussianMomentum, integrator);
        crate::runner::run_rs(sampler, &cool, n_epochs, steps_per_epoch, seed)
    } else {
        let sampler = HmcSaSampler::with_momentum(
            obj,
            grad,
            cool.clone(),
            QGaussianMomentum::new(q),
            integrator,
        );
        crate::runner::run_rs(sampler, &cool, n_epochs, steps_per_epoch, seed)
    };
    Ok(PyHistory::from(history))
}

// ---------------------------------------------------------------------------
// Preset enum + dispatch.
// ---------------------------------------------------------------------------

/// Preset selector accepted by `run`. Matches one of the three
/// IISE-manuscript points.
#[derive(FromPyObject)]
enum Preset {
    #[pyo3(transparent)]
    Boltzmann(PyBoltzmann),
    #[pyo3(transparent)]
    Fast(PyFast),
    #[pyo3(transparent)]
    Gsa(PyGsa),
}

/// Runs the SA driver and returns a `History`.
///
/// Args:
///   obj_fn: Python callable `f(numpy.ndarray) -> float` evaluated at every
///           proposal. Held via the GIL.
///   low, high: numpy arrays defining the box bounds used to draw the
///              initial position uniformly. Same length defines the
///              objective dimensionality.
///   preset: one of `Boltzmann()`, `Fast()`, `Gsa()` from `anneal`.
///   n_epochs, steps_per_epoch: SA loop dimensions.
///   seed: u64 seed for the StdRng.
#[pyfunction]
#[pyo3(signature = (obj_fn, low, high, preset, n_epochs = 100, steps_per_epoch = 200, seed = 42))]
fn run(
    obj_fn: Py<PyAny>,
    low: PyReadonlyArray1<'_, f64>,
    high: PyReadonlyArray1<'_, f64>,
    preset: Preset,
    n_epochs: usize,
    steps_per_epoch: usize,
    seed: u64,
) -> PyResult<PyHistory> {
    let low_vec = low.as_slice()?.to_vec();
    let high_vec = high.as_slice()?.to_vec();
    if low_vec.len() != high_vec.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "low and high must have the same length",
        ));
    }
    let bounds = Bounds::new(Array1::from_vec(low_vec), Array1::from_vec(high_vec), 1e-9);
    let obj = CallableObjective {
        fn_: obj_fn,
        bounds,
    };
    let history = match preset {
        Preset::Boltzmann(p) => {
            let v = boltzmann(obj, p.t_init, p.sigma)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
            crate::runner::run_rs_variant(v, n_epochs, steps_per_epoch, seed)
        }
        Preset::Fast(p) => {
            let v = fast(obj, p.t_init, p.gamma)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
            crate::runner::run_rs_variant(v, n_epochs, steps_per_epoch, seed)
        }
        Preset::Gsa(p) => {
            let v = gsa(obj, p.t_init, p.q_v, p.q_a)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
            crate::runner::run_rs_variant(v, n_epochs, steps_per_epoch, seed)
        }
    };
    Ok(PyHistory::from(history))
}

// ---------------------------------------------------------------------------
// Module entry point.
// ---------------------------------------------------------------------------

/// pyo3 module initialiser. Exposed to Python as `anneal._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyBoltzmann>()?;
    m.add_class::<PyFast>()?;
    m.add_class::<PyGsa>()?;
    m.add_class::<PyEpochLine>()?;
    m.add_class::<PyHistory>()?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(run_hmc, m)?)?;
    Ok(())
}
