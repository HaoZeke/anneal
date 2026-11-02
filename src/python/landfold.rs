//! `landfold` and `landfold_book`: the sketch-map embedding of
//! [`crate::landfold`] for Python.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::landfold as lf;
use crate::landfold::{Landfold, LandfoldConfig, LandfoldError, LandfoldMap, LandfoldMetric};

fn py_err(err: LandfoldError) -> PyErr {
    match err {
        LandfoldError::Read(msg) => PyOSError::new_err(msg),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn rows_of(rows: &PyReadonlyArray2<'_, f64>) -> Vec<Vec<f64>> {
    rows.as_array()
        .outer_iter()
        .map(|row| row.iter().copied().collect())
        .collect()
}

fn xy_array<'py>(py: Python<'py>, xy: &[[f64; 2]]) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = xy.iter().flatten().copied().collect();
    PyArray1::from_vec(py, flat)
        .reshape([xy.len(), 2])
        .expect("an n x 2 map")
}

/// A fitted landmark map. `project` places rows that were not in the fit.
#[pyclass(name = "LandfoldMap", module = "anneal._core", frozen)]
pub struct PyLandfoldMap {
    map: LandfoldMap,
}

#[pymethods]
impl PyLandfoldMap {
    /// Place each row of an `(k, D)` array against the fixed landmarks.
    fn project<'py>(
        &self,
        py: Python<'py>,
        rows: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let rows = rows_of(&rows);
        let map = &self.map;
        let xy: Vec<[f64; 2]> = py.detach(|| rows.iter().map(|r| map.project(r)).collect());
        xy_array(py, &xy)
    }

    /// Landmark positions on the plane, `(m, 2)`.
    #[getter]
    fn landmark_xy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        xy_array(py, self.map.landmark_xy())
    }

    /// Landmark weights: each Voronoi population to the weight power.
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.map.weights())
    }
}

fn config(
    landmarks: usize,
    seed: u64,
    sigma: Option<f64>,
    sigma_scale: f64,
    sigma_low: Option<f64>,
    high: (f64, f64),
    low: (f64, f64),
    grid: usize,
    iterations: usize,
    mix: Option<Vec<f64>>,
    weight_power: f64,
    metric: &str,
) -> PyResult<LandfoldConfig> {
    let metric = match metric {
        "l1" => LandfoldMetric::L1,
        "l2" => LandfoldMetric::L2,
        other => {
            return Err(PyValueError::new_err(format!(
                "metric must be 'l1' or 'l2', not {other:?}"
            )));
        }
    };
    Ok(LandfoldConfig {
        landmarks,
        seed,
        sigma,
        sigma_scale,
        sigma_low,
        high: [high.0, high.1],
        low: [low.0, low.1],
        grid,
        iterations,
        mix: mix.unwrap_or_else(|| lf::DEFAULT_MIX.to_vec()),
        weight_power,
        metric,
    })
}

/// Run `work` on a private pool of `threads` workers, or on the global pool.
fn on_pool<T: Send>(threads: Option<usize>, work: impl FnOnce() -> T + Send) -> PyResult<T> {
    match threads {
        None => Ok(work()),
        Some(0) => Err(PyValueError::new_err("threads must be at least 1")),
        Some(t) => rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .map(|pool| pool.install(work))
            .map_err(|err| PyValueError::new_err(err.to_string())),
    }
}

fn fold_dict<'py>(py: Python<'py>, fold: Landfold) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("xy", xy_array(py, &fold.xy))?;
    out.set_item("landmarks", PyArray1::from_slice(py, &fold.landmarks))?;
    out.set_item("sigma", fold.map.high().sigma())?;
    out.set_item("sigma_low", fold.map.low().sigma())?;
    out.set_item("high", fold.map.high().exponents().to_vec())?;
    out.set_item("low", fold.map.low().exponents().to_vec())?;
    out.set_item("stress", fold.stress)?;
    out.set_item("spectrum", fold.spectrum.to_vec())?;
    out.set_item("map", PyLandfoldMap { map: fold.map })?;
    Ok(out)
}

/// Fold an `(n, D)` array of descriptor rows onto the plane.
///
/// Returns a dict with `xy` `(n, 2)`, `landmarks` (row indices), `sigma`,
/// `sigma_low`, `high`, `low`, `stress`, `spectrum` (lambda1, lambda2 of the
/// landmark MDS start) and `map`, a `LandfoldMap` for held-out rows.
#[pyfunction(name = "landfold")]
#[pyo3(signature = (
    rows, *, landmarks = lf::DEFAULT_LANDMARKS, seed = 0, sigma = None,
    sigma_scale = lf::DEFAULT_SIGMA_SCALE, sigma_low = None,
    high = (lf::DEFAULT_HIGH[0], lf::DEFAULT_HIGH[1]),
    low = (lf::DEFAULT_LOW[0], lf::DEFAULT_LOW[1]),
    grid = lf::DEFAULT_GRID, iterations = lf::DEFAULT_ITERATIONS,
    mix = None, weight_power = lf::DEFAULT_WEIGHT_POWER, metric = "l1",
    threads = None,
))]
fn fold_rows<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray2<'py, f64>,
    landmarks: usize,
    seed: u64,
    sigma: Option<f64>,
    sigma_scale: f64,
    sigma_low: Option<f64>,
    high: (f64, f64),
    low: (f64, f64),
    grid: usize,
    iterations: usize,
    mix: Option<Vec<f64>>,
    weight_power: f64,
    metric: &str,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let config = config(
        landmarks,
        seed,
        sigma,
        sigma_scale,
        sigma_low,
        high,
        low,
        grid,
        iterations,
        mix,
        weight_power,
        metric,
    )?;
    let rows = rows_of(&rows);
    let fold = py
        .detach(|| on_pool(threads, || lf::landfold(&rows, &config)))?
        .map_err(py_err)?;
    fold_dict(py, fold)
}

/// Read minima books, describe each minimum by its DECAF class histogram,
/// and fold the rows with the L1 distance.
///
/// `paths` are `.min` dumps or multi-frame `.xyz` files, concatenated in
/// order. The dict of `landfold` gains `energy`, `rows` (the descriptors),
/// `family` and `community` (-1 where a minimum has no community).
#[pyfunction]
#[pyo3(signature = (
    paths, *, landmarks = lf::DEFAULT_LANDMARKS, seed = 0, sigma = None,
    sigma_scale = lf::DEFAULT_SIGMA_SCALE, sigma_low = None,
    high = (lf::DEFAULT_HIGH[0], lf::DEFAULT_HIGH[1]),
    low = (lf::DEFAULT_LOW[0], lf::DEFAULT_LOW[1]),
    grid = lf::DEFAULT_GRID, iterations = lf::DEFAULT_ITERATIONS,
    mix = None, weight_power = lf::DEFAULT_WEIGHT_POWER, threads = None,
))]
fn landfold_book<'py>(
    py: Python<'py>,
    paths: Vec<std::path::PathBuf>,
    landmarks: usize,
    seed: u64,
    sigma: Option<f64>,
    sigma_scale: f64,
    sigma_low: Option<f64>,
    high: (f64, f64),
    low: (f64, f64),
    grid: usize,
    iterations: usize,
    mix: Option<Vec<f64>>,
    weight_power: f64,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let config = config(
        landmarks,
        seed,
        sigma,
        sigma_scale,
        sigma_low,
        high,
        low,
        grid,
        iterations,
        mix,
        weight_power,
        "l1",
    )?;
    let work = || -> Result<_, LandfoldError> {
        let mut minima = Vec::new();
        for path in &paths {
            minima.extend(lf::read_book(path)?);
        }
        let coordinates: Vec<Vec<f64>> = minima.iter().map(|m| m.coordinates.clone()).collect();
        let energy: Vec<f64> = minima.iter().map(|m| m.energy).collect();
        let descriptors = lf::book_descriptors(&coordinates)?;
        let fold = lf::landfold(&descriptors.rows, &config)?;
        Ok((energy, descriptors, fold))
    };
    let (energy, descriptors, fold) = py.detach(|| on_pool(threads, work))?.map_err(py_err)?;
    let out = fold_dict(py, fold)?;
    out.set_item("energy", PyArray1::from_vec(py, energy))?;
    out.set_item(
        "rows",
        PyArray2::from_vec2(py, &descriptors.rows)
            .map_err(|err| PyValueError::new_err(err.to_string()))?,
    )?;
    out.set_item("family", PyArray1::from_slice(py, &descriptors.family))?;
    let community: Vec<i64> = descriptors
        .community
        .iter()
        .map(|&c| if c == usize::MAX { -1 } else { c as i64 })
        .collect();
    out.set_item("community", PyArray1::from_vec(py, community))?;
    Ok(out)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLandfoldMap>()?;
    module.add_function(wrap_pyfunction!(fold_rows, module)?)?;
    module.add_function(wrap_pyfunction!(landfold_book, module)?)?;
    Ok(())
}
