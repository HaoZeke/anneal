//! Native preset control with backend-owned Array API storage and random draws.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::Preset;
use crate::accept::{Metropolis, ProbabilityArithmetic, TsallisAccept};
use crate::cool::{Cooling, LogCool, ReciprocalCool, TsallisCool};
use crate::movekernel::{Cauchy, Gaussian, TsallisVisit};

enum DevicePolicy {
    Boltzmann(LogCool<f64>, Gaussian),
    Fast(ReciprocalCool<f64>, Cauchy),
    Gsa(TsallisCool<f64>, TsallisVisit, TsallisAccept<f64>),
}

impl DevicePolicy {
    fn new(preset: Preset) -> PyResult<Self> {
        fn positive(value: f64, name: &str) -> PyResult<()> {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!("{name} must be positive and finite")))
            }
        }
        Ok(match preset {
            Preset::Boltzmann(p) => {
                positive(p.t_init, "t_init")?;
                positive(p.sigma, "sigma")?;
                Self::Boltzmann(LogCool::new(p.t_init, 2.0), Gaussian::new(p.sigma))
            }
            Preset::Fast(p) => {
                positive(p.t_init, "t_init")?;
                positive(p.gamma, "gamma")?;
                Self::Fast(ReciprocalCool::new(p.t_init), Cauchy::new(p.gamma))
            }
            Preset::Gsa(p) => {
                positive(p.t_init, "t_init")?;
                if !(p.q_v > 1.0 && p.q_v < 3.0) || !p.q_a.is_finite() {
                    return Err(PyValueError::new_err("q_v must lie in (1, 3) and q_a must be finite"));
                }
                Self::Gsa(TsallisCool::new(p.t_init, p.q_v), TsallisVisit::new(p.q_v), TsallisAccept::new(p.q_a))
            }
        })
    }

    fn temperature(&self, epoch: usize) -> f64 {
        match self {
            Self::Boltzmann(cool, _) => cool.temperature(epoch),
            Self::Fast(cool, _) => cool.temperature(epoch),
            Self::Gsa(cool, _, _) => cool.temperature(epoch),
        }
    }

    fn probability<'py>(&self, delta: &Bound<'py, PyAny>, temp: f64, arrays: &DeviceArrays<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Gsa(_, _, accept) => accept.probabilities_with(delta, temp, arrays),
            _ => Metropolis.probabilities_with(delta, temp, arrays),
        }
    }

    fn proposal<'py>(&self, current: &Bound<'py, PyAny>, temp: f64, random: &Bound<'py, PyAny>, shape: &Bound<'py, PyTuple>, arrays: &DeviceArrays<'py>) -> PyResult<Bound<'py, PyAny>> {
        let step = match self {
            Self::Boltzmann(_, mover) => arrays.scale(&random.call_method1("normal", (shape,))?, mover.sigma)?,
            Self::Fast(_, mover) => {
                let uniform = random.call_method1("uniform", (shape,))?;
                let centered = arrays.offset(&uniform, -0.5)?;
                let angle = arrays.scale(&centered, std::f64::consts::PI)?;
                arrays.scale(&arrays.unary("tan", &angle)?, mover.gamma)?
            }
            Self::Gsa(_, mover, _) => {
                let parameters = mover.parameters(temp);
                let x = random.call_method1("normal", (shape,))?;
                let y = random.call_method1("normal", (shape,))?;
                let numerator = arrays.scale(&x, parameters.scale)?;
                let denominator = arrays.powf(&arrays.unary("abs", &y)?, parameters.exponent)?;
                let visit = arrays.binary("divide", &numerator, &denominator)?;
                // Fixed-shape draws keep tail handling on the device without a
                // host reduction to discover which coordinates exceed the cap.
                let tail = arrays.scale(&random.call_method1("uniform", (shape,))?, parameters.tail_limit)?;
                let limit = arrays.constant(parameters.tail_limit)?;
                let positive = arrays.binary("greater", &visit, &limit)?;
                let negative = arrays.binary("less", &visit, &arrays.scale(&limit, -1.0)?)?;
                let clipped = arrays.select(&positive, &tail, &visit)?;
                arrays.select(&negative, &arrays.scale(&tail, -1.0)?, &clipped)?
            }
        };
        arrays.binary("add", current, &step)
    }
}

struct DeviceArrays<'py> {
    helpers: Bound<'py, PyModule>,
    xp: Bound<'py, PyAny>,
    device: Bound<'py, PyAny>,
    options: Bound<'py, PyDict>,
    location: Bound<'py, PyDict>,
}

impl<'py> DeviceArrays<'py> {
    fn new(py: Python<'py>, reference: &Bound<'py, PyAny>) -> PyResult<Self> {
        let helpers = PyModule::import(py, "anneal.device")?;
        let xp = helpers.call_method1("_array_namespace", (reference,))?;
        let device = helpers.call_method1("_device", (reference,))?;
        let dtype = reference.getattr("dtype")?;
        let location = PyDict::new(py);
        location.set_item("xp", &xp)?;
        location.set_item("device", &device)?;
        let options = location.copy()?;
        options.set_item("dtype", &dtype)?;
        Ok(Self { helpers, xp, device, options, location })
    }

    fn array(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.helpers.call_method("_asarray", (value,), Some(&self.options))
    }

    fn integer(&self, value: usize) -> PyResult<Bound<'py, PyAny>> {
        let options = self.location.copy()?;
        options.set_item("dtype", self.xp.getattr("int64")?)?;
        self.helpers.call_method("_asarray", (value,), Some(&options))
    }

    fn unary(&self, name: &str, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.xp.call_method1(name, (value,))
    }

    fn binary(&self, name: &str, left: &Bound<'py, PyAny>, right: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.xp.call_method1(name, (left, right))
    }

    fn select(&self, mask: &Bound<'py, PyAny>, yes: &Bound<'py, PyAny>, no: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.xp.call_method1("where", (mask, yes, no))
    }

    fn position_mask(&self, mask: &Bound<'py, PyAny>, batched: bool) -> PyResult<Bound<'py, PyAny>> {
        if batched {
            self.xp.call_method1("expand_dims", (mask, -1))
        } else {
            Ok(mask.clone())
        }
    }

    fn clip(&self, value: &Bound<'py, PyAny>, low: &Bound<'py, PyAny>, high: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary("minimum", &self.binary("maximum", value, low)?, high)
    }

    fn evaluate(&self, objective: &Bound<'py, PyAny>, position: &Bound<'py, PyAny>, n_chains: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        let value = objective.call1((position,))?;
        match n_chains {
            Some(n) => self.helpers.call_method("_ensemble_objective_value", (value, n), Some(&self.options)),
            None => self.helpers.call_method("_objective_value", (value,), Some(&self.options)),
        }
    }

    fn count(&self, mask: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let counts = self.helpers.call_method("_count_from_bool", (mask,), Some(&self.location))?;
        self.unary("sum", &counts)
    }

    fn stack(&self, values: &[Bound<'py, PyAny>]) -> PyResult<Bound<'py, PyAny>> {
        self.xp.call_method1("stack", (PyList::new(self.xp.py(), values)?,))
    }
}

impl<'py> ProbabilityArithmetic<f64> for DeviceArrays<'py> {
    type Value = Bound<'py, PyAny>;
    type Error = PyErr;

    fn constant(&self, value: f64) -> PyResult<Self::Value> {
        self.helpers.call_method("_asarray", (value,), Some(&self.options))
    }
    fn scale(&self, value: &Self::Value, factor: f64) -> PyResult<Self::Value> {
        value.call_method1("__mul__", (factor,))
    }
    fn divide(&self, value: &Self::Value, divisor: f64) -> PyResult<Self::Value> {
        value.call_method1("__truediv__", (divisor,))
    }
    fn offset(&self, value: &Self::Value, addend: f64) -> PyResult<Self::Value> {
        value.call_method1("__add__", (addend,))
    }
    fn exp(&self, value: &Self::Value) -> PyResult<Self::Value> {
        self.unary("exp", value)
    }
    fn powf(&self, value: &Self::Value, exponent: f64) -> PyResult<Self::Value> {
        value.call_method1("__pow__", (exponent,))
    }
    fn select_nonpositive(&self, condition: &Self::Value, nonpositive: &Self::Value, positive: &Self::Value) -> PyResult<Self::Value> {
        self.select(&self.binary("less_equal", condition, &self.constant(0.0)?)?, nonpositive, positive)
    }
}

fn run_native<'py>(
    py: Python<'py>,
    objective: &Bound<'py, PyAny>,
    low: &Bound<'py, PyAny>,
    high: &Bound<'py, PyAny>,
    preset: Preset,
    n_chains: Option<usize>,
    n_epochs: usize,
    steps_per_epoch: usize,
    seed: u64,
    start: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if n_epochs == 0 || steps_per_epoch == 0 || n_chains == Some(0) {
        return Err(PyValueError::new_err("n_epochs, steps_per_epoch and n_chains must be positive"));
    }
    let chains = n_chains.unwrap_or(1);
    n_epochs.checked_mul(steps_per_epoch).and_then(|n| n.checked_add(1)).and_then(|n| n.checked_mul(chains))
        .ok_or_else(|| PyValueError::new_err("evaluation count overflows usize"))?;
    let policy = DevicePolicy::new(preset)?;
    let arrays = DeviceArrays::new(py, low)?;
    let low = arrays.array(low)?;
    let high = arrays.array(high)?;
    let low_shape: Vec<usize> = low.getattr("shape")?.extract()?;
    let high_shape: Vec<usize> = high.getattr("shape")?.extract()?;
    if low_shape != high_shape || low_shape.len() != 1 {
        return Err(PyValueError::new_err("low and high must be one-dimensional arrays of equal shape"));
    }
    let batched = n_chains.is_some();
    let shape = if batched { vec![chains, low_shape[0]] } else { low_shape.clone() };
    let shape = PyTuple::new(py, shape)?;
    let accept_shape = PyTuple::new(py, n_chains)?;
    let random_options = arrays.options.copy()?;
    random_options.set_item("seed", seed)?;
    let random = arrays.helpers.call_method("_Random", (&low,), Some(&random_options))?;
    let initial = match start {
        Some(start) => {
            let start = arrays.array(start)?;
            if start.getattr("shape")?.extract::<Vec<usize>>()? != low_shape {
                return Err(PyValueError::new_err("start must have the same shape as low and high"));
            }
            start
        }
        None => arrays.binary("add", &low, &arrays.binary("multiply", &random.call_method1("uniform", (&shape,))?, &arrays.binary("subtract", &high, &low)?)?)?,
    };
    // Clipping is the declared device-box policy, not the unconstrained domain
    // of the classical native presets or a manifold retraction.
    let mut current = arrays.clip(&initial, &low, &high)?;
    let mut current_val = arrays.evaluate(objective, &current, n_chains)?;
    let mut n_evals = 1usize;
    let mut best_pos = arrays.offset(&current, 0.0)?;
    let mut best_val = arrays.offset(&current_val, 0.0)?;
    let mut epochs = Vec::new();
    let mut temps = Vec::new();
    let mut accepted_history = Vec::new();
    let mut rejected_history = Vec::new();
    let mut best_vals = Vec::new();
    let mut accepted_total = arrays.integer(0)?;
    let mut rejected_total = arrays.integer(0)?;

    for epoch in 0..n_epochs {
        let temp = policy.temperature(epoch);
        let mut accepted_epoch = arrays.integer(0)?;
        let mut rejected_epoch = arrays.integer(0)?;
        for _ in 0..steps_per_epoch {
            let candidate = arrays.clip(&policy.proposal(&current, temp, &random, &shape, &arrays)?, &low, &high)?;
            let candidate_val = arrays.evaluate(objective, &candidate, n_chains)?;
            n_evals += 1;

            // The incumbent is the best finite evaluated point, independent of
            // whether the stochastic transition accepts the candidate.
            let finite = arrays.unary("isfinite", &candidate_val)?;
            let absent = arrays.unary("logical_not", &arrays.unary("isfinite", &best_val)?)?;
            let lower = arrays.binary("less", &candidate_val, &best_val)?;
            let improved = arrays.binary("logical_and", &finite, &arrays.binary("logical_or", &absent, &lower)?)?;
            best_pos = arrays.select(&arrays.position_mask(&improved, batched)?, &candidate, &best_pos)?;
            best_val = arrays.select(&improved, &candidate_val, &best_val)?;

            let delta = arrays.binary("subtract", &candidate_val, &current_val)?;
            let probability = policy.probability(&delta, temp, &arrays)?;
            let accepted = arrays.binary("less", &random.call_method1("uniform", (&accept_shape,))?, &probability)?;
            accepted_epoch = arrays.binary("add", &accepted_epoch, &arrays.count(&accepted)?)?;
            rejected_epoch = arrays.binary("add", &rejected_epoch, &arrays.count(&arrays.unary("logical_not", &accepted)?)?)?;
            current = arrays.select(&arrays.position_mask(&accepted, batched)?, &candidate, &current)?;
            current_val = arrays.select(&accepted, &candidate_val, &current_val)?;
        }
        if batched {
            accepted_total = arrays.binary("add", &accepted_total, &accepted_epoch)?;
            rejected_total = arrays.binary("add", &rejected_total, &rejected_epoch)?;
        } else {
            epochs.push(arrays.integer(epoch)?);
            temps.push(arrays.constant(temp)?);
            accepted_history.push(accepted_epoch);
            rejected_history.push(rejected_epoch);
            best_vals.push(best_val.clone());
        }
    }

    let result = PyDict::new(py);
    result.set_item("best_pos", &best_pos)?;
    result.set_item("best_val", arrays.array(&best_val)?)?;
    result.set_item("namespace", &arrays.xp)?;
    result.set_item("device", &arrays.device)?;
    result.set_item("n_evals", n_evals)?;
    result.set_item("evaluated_points", n_evals * chains)?;
    let history = if batched {
        let finite_values = arrays.select(&arrays.unary("isfinite", &best_val)?, &best_val, &arrays.constant(f64::INFINITY)?)?;
        // Selecting a result index synchronizes one scalar; population arrays
        // and all transition decisions remain in the backend namespace.
        let index: usize = arrays.unary("argmin", &finite_values)?.call_method0("__int__")?.extract()?;
        result.set_item("global_best_pos", best_pos.get_item(index)?)?;
        result.set_item("global_best_val", arrays.array(&best_val.get_item(index)?)?)?;
        result.set_item("accepted", accepted_total)?;
        result.set_item("rejected", rejected_total)?;
        "EnsembleHistory"
    } else {
        result.set_item("epochs", arrays.stack(&epochs)?)?;
        result.set_item("temps", arrays.stack(&temps)?)?;
        result.set_item("accepted", arrays.stack(&accepted_history)?)?;
        result.set_item("rejected", arrays.stack(&rejected_history)?)?;
        result.set_item("best_vals", arrays.stack(&best_vals)?)?;
        result.set_item("current_pos", current)?;
        result.set_item("current_val", current_val)?;
        "DeviceHistory"
    };
    arrays.helpers.getattr(history)?.call((), Some(&result))
}

#[pyfunction(name = "_run_device")]
#[pyo3(signature = (obj_fn, low, high, preset, *, n_epochs = 100, steps_per_epoch = 200, seed = 42, start = None))]
fn run_device<'py>(py: Python<'py>, obj_fn: Bound<'py, PyAny>, low: Bound<'py, PyAny>, high: Bound<'py, PyAny>, preset: Preset, n_epochs: usize, steps_per_epoch: usize, seed: u64, start: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    run_native(py, &obj_fn, &low, &high, preset, None, n_epochs, steps_per_epoch, seed, start.as_ref())
}

#[pyfunction(name = "_run_device_ensemble")]
#[pyo3(signature = (obj_fn, low, high, preset, *, n_chains, n_epochs = 100, steps_per_epoch = 200, seed = 42))]
fn run_ensemble<'py>(py: Python<'py>, obj_fn: Bound<'py, PyAny>, low: Bound<'py, PyAny>, high: Bound<'py, PyAny>, preset: Preset, n_chains: usize, n_epochs: usize, steps_per_epoch: usize, seed: u64) -> PyResult<Bound<'py, PyAny>> {
    run_native(py, &obj_fn, &low, &high, preset, Some(n_chains), n_epochs, steps_per_epoch, seed, None)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_device, module)?)?;
    module.add_function(wrap_pyfunction!(run_ensemble, module)?)?;
    Ok(())
}
