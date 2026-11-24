// Copyright (c) anneal developers

//! Python bindings for built-in cluster potentials and coordinate helpers.

use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use eindir_core::py_objective::{PyBounds as EindirPyBounds, PyObjective};
use ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::potentials::{
    BinaryLennardJones, COPPER_Z, GOLD_Z, GminTip4p, Gupta, PairKind, PairPotential, SuttonChen,
    SuttonChenFamily, Tip4pCluster, block_from_interleaved_gmin, ccd_tip4p_minimum, energy_ohh,
    gmin_tip4p_sites, interleaved_gmin_from_block, rigid_from_ohh,
};

const PLATINUM_Z: u32 = 78;

fn py_bounds_from(py: Python<'_>, bounds: &Bounds<f64>) -> PyResult<EindirPyBounds> {
    let low = PyArray1::from_slice(py, bounds.low.as_slice().unwrap_or(&[]));
    let high = PyArray1::from_slice(py, bounds.high.as_slice().unwrap_or(&[]));
    let cls = py.get_type::<EindirPyBounds>();
    cls.call1((low, high, bounds.slack))?.extract()
}

fn to_py_objective(
    py: Python<'_>,
    self_obj: &Bound<'_, PyAny>,
    bounds: &Bounds<f64>,
) -> PyResult<Py<PyObjective>> {
    let py_bounds = py_bounds_from(py, bounds)?;
    let eval_fn = self_obj.getattr("eval")?.unbind();
    let grad_fn = self_obj.getattr("grad")?.unbind();
    let cls = py.get_type::<PyObjective>();
    let inst = cls.call1((eval_fn, py_bounds, grad_fn))?;
    Ok(inst.cast_into::<PyObjective>()?.unbind())
}

fn check_len(slice: &[f64], expected: usize, label: &str) -> PyResult<()> {
    if slice.len() != expected {
        return Err(PyValueError::new_err(format!(
            "{label} requires a vector of length {expected}, got {}",
            slice.len()
        )));
    }
    Ok(())
}

/// Pairwise Lennard-Jones or Morse cluster potential over `n_points` atoms.
#[pyclass(name = "PairPotential", module = "anneal._core")]
#[derive(Clone)]
pub struct PyPairPotential {
    pub(crate) inner: PairPotential,
}

#[pymethods]
impl PyPairPotential {
    /// Construct a pair potential (`rho=None` selects Lennard-Jones; setting `rho` selects Morse).
    #[new]
    #[pyo3(signature = (n_points, rho = None, extent = None))]
    fn new(n_points: usize, rho: Option<f64>, extent: Option<f64>) -> PyResult<Self> {
        if n_points < 2 {
            return Err(PyValueError::new_err(
                "a pair potential needs at least two points",
            ));
        }
        if let Some(r) = rho
            && (!r.is_finite() || r <= 0.0)
        {
            return Err(PyValueError::new_err("rho must be finite and positive"));
        }
        if let Some(ext) = extent
            && (!ext.is_finite() || ext <= 0.0)
        {
            return Err(PyValueError::new_err("extent must be finite and positive"));
        }
        let inner = match (rho, extent) {
            (None, None) => PairPotential::lennard_jones(n_points),
            (Some(r), None) => PairPotential::morse(n_points, r),
            (None, Some(ext)) => PairPotential::new(n_points, PairKind::LennardJones, ext),
            (Some(r), Some(ext)) => PairPotential::new(n_points, PairKind::Morse { rho: r }, ext),
        };
        Ok(Self { inner })
    }

    /// Untruncated Lennard-Jones cluster over `n_points` atoms.
    #[staticmethod]
    fn lennard_jones(n_points: usize) -> PyResult<Self> {
        Self::new(n_points, None, None)
    }

    /// Untruncated Morse cluster over `n_points` atoms with range `rho`.
    #[staticmethod]
    fn morse(n_points: usize, rho: f64) -> PyResult<Self> {
        Self::new(n_points, Some(rho), None)
    }

    /// Cambridge Cluster Database global minimum for Morse `(n_points, rho)`, if tabulated.
    #[staticmethod]
    fn ccd_morse_minimum(n_points: usize, rho: f64) -> Option<f64> {
        PairPotential::ccd_morse_minimum(n_points, rho)
    }

    #[getter]
    fn n_points(&self) -> usize {
        self.inner.n_points
    }

    #[getter]
    fn dim(&self) -> usize {
        Objective::dim(&self.inner)
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind() {
            PairKind::LennardJones => "lj",
            PairKind::Morse { .. } => "morse",
        }
    }

    #[getter]
    fn rho(&self) -> Option<f64> {
        match self.inner.kind() {
            PairKind::LennardJones => None,
            PairKind::Morse { rho } => Some(rho),
        }
    }

    #[getter]
    fn r_min(&self) -> f64 {
        self.inner.kind().r_min()
    }

    #[getter]
    fn length_scale(&self) -> f64 {
        self.inner.kind().length_scale()
    }

    #[getter]
    fn second_neighbour_energy(&self) -> f64 {
        self.inner.kind().second_neighbour_energy()
    }

    #[getter]
    fn bounds(&self, py: Python<'_>) -> PyResult<EindirPyBounds> {
        py_bounds_from(py, Objective::bounds(&self.inner))
    }

    fn eval(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let slice = x.as_slice()?;
        check_len(slice, 3 * self.inner.n_points, "PairPotential")?;
        Ok(Objective::eval(&self.inner, ArrayView1::from(slice)))
    }

    fn grad<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = x.as_slice()?;
        check_len(slice, 3 * self.inner.n_points, "PairPotential")?;
        let g = Gradient::grad(&self.inner, ArrayView1::from(slice));
        Ok(super::py_view1(py, g.view()))
    }

    fn value_and_gradient<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let slice = x.as_slice()?;
        check_len(slice, 3 * self.inner.n_points, "PairPotential")?;
        let (e, g) = self.inner.value_and_gradient(ArrayView1::from(slice));
        Ok((e, super::py_view1(py, g.view())))
    }

    fn to_objective(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyObjective>> {
        let bounds = Objective::bounds(&slf.borrow().inner).clone();
        to_py_objective(py, slf.as_any(), &bounds)
    }

    fn __call__(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        self.eval(x)
    }

    fn __repr__(&self) -> String {
        match self.inner.kind() {
            PairKind::LennardJones => {
                format!("PairPotential.lennard_jones({})", self.inner.n_points)
            }
            PairKind::Morse { rho } => {
                format!(
                    "PairPotential.morse({}, rho={:?})",
                    self.inner.n_points, rho
                )
            }
        }
    }
}

/// Two-species untruncated Lennard-Jones cluster potential.
#[pyclass(name = "BinaryLennardJones", module = "anneal._core")]
#[derive(Clone)]
pub struct PyBinaryLennardJones {
    pub(crate) inner: BinaryLennardJones,
    n_a: usize,
    n_b: usize,
    epsilon_aa: f64,
    epsilon_ab: f64,
    epsilon_bb: f64,
    sigma_aa: f64,
    sigma_ab: f64,
    sigma_bb: f64,
}

#[pymethods]
impl PyBinaryLennardJones {
    #[new]
    #[pyo3(signature = (
        n_a,
        n_b,
        epsilon_aa = 1.0,
        epsilon_ab = 1.0,
        epsilon_bb = 1.0,
        sigma_aa = 1.0,
        sigma_ab = 1.0,
        sigma_bb = 1.0,
    ))]
    fn new(
        n_a: usize,
        n_b: usize,
        epsilon_aa: f64,
        epsilon_ab: f64,
        epsilon_bb: f64,
        sigma_aa: f64,
        sigma_ab: f64,
        sigma_bb: f64,
    ) -> PyResult<Self> {
        if n_a + n_b < 2 {
            return Err(PyValueError::new_err(
                "a binary cluster needs at least two atoms",
            ));
        }
        if ![
            epsilon_aa, epsilon_ab, epsilon_bb, sigma_aa, sigma_ab, sigma_bb,
        ]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0)
        {
            return Err(PyValueError::new_err(
                "Lennard-Jones parameters must be finite and positive",
            ));
        }
        Ok(Self {
            inner: BinaryLennardJones::new(
                n_a, n_b, epsilon_aa, epsilon_ab, epsilon_bb, sigma_aa, sigma_ab, sigma_bb,
            ),
            n_a,
            n_b,
            epsilon_aa,
            epsilon_ab,
            epsilon_bb,
            sigma_aa,
            sigma_ab,
            sigma_bb,
        })
    }

    /// Doye-Meyer binary Lennard-Jones (`eps = 1`, `sigma_aa = 1`, `sigma_ab = (1 + sigma_bb)/2`).
    #[staticmethod]
    fn doye_meyer(n_a: usize, n_b: usize, sigma_bb: f64) -> PyResult<Self> {
        if !sigma_bb.is_finite() || sigma_bb <= 0.0 {
            return Err(PyValueError::new_err(
                "sigma_bb must be finite and positive",
            ));
        }
        Self::new(
            n_a,
            n_b,
            1.0,
            1.0,
            1.0,
            1.0,
            0.5 * (1.0 + sigma_bb),
            sigma_bb,
        )
    }

    #[getter]
    fn n_a(&self) -> usize {
        self.n_a
    }

    #[getter]
    fn n_b(&self) -> usize {
        self.n_b
    }

    #[getter]
    fn n_points(&self) -> usize {
        self.inner.n_points()
    }

    #[getter]
    fn dim(&self) -> usize {
        Objective::dim(&self.inner)
    }

    #[getter]
    fn epsilon_aa(&self) -> f64 {
        self.epsilon_aa
    }

    #[getter]
    fn epsilon_ab(&self) -> f64 {
        self.epsilon_ab
    }

    #[getter]
    fn epsilon_bb(&self) -> f64 {
        self.epsilon_bb
    }

    #[getter]
    fn sigma_aa(&self) -> f64 {
        self.sigma_aa
    }

    #[getter]
    fn sigma_ab(&self) -> f64 {
        self.sigma_ab
    }

    #[getter]
    fn sigma_bb(&self) -> f64 {
        self.sigma_bb
    }

    #[getter]
    fn bounds(&self, py: Python<'_>) -> PyResult<EindirPyBounds> {
        py_bounds_from(py, Objective::bounds(&self.inner))
    }

    fn eval(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let slice = x.as_slice()?;
        check_len(slice, 3 * self.inner.n_points(), "BinaryLennardJones")?;
        Ok(Objective::eval(&self.inner, ArrayView1::from(slice)))
    }

    fn grad<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = x.as_slice()?;
        check_len(slice, 3 * self.inner.n_points(), "BinaryLennardJones")?;
        let g = Gradient::grad(&self.inner, ArrayView1::from(slice));
        Ok(super::py_view1(py, g.view()))
    }

    fn value_and_gradient<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let slice = x.as_slice()?;
        check_len(slice, 3 * self.inner.n_points(), "BinaryLennardJones")?;
        let (e, g) = self.inner.value_and_gradient(ArrayView1::from(slice));
        Ok((e, super::py_view1(py, g.view())))
    }

    fn to_objective(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyObjective>> {
        let bounds = Objective::bounds(&slf.borrow().inner).clone();
        to_py_objective(py, slf.as_any(), &bounds)
    }

    fn __call__(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        self.eval(x)
    }

    fn __repr__(&self) -> String {
        format!(
            "BinaryLennardJones(n_a={}, n_b={}, sigma_bb={:?})",
            self.n_a, self.n_b, self.sigma_bb
        )
    }
}

/// Sutton-Chen metal cluster potential (`12-6`, `9-6`, or `10-8`).
#[pyclass(name = "SuttonChen", module = "anneal._core")]
#[derive(Clone)]
pub struct PySuttonChen {
    pub(crate) inner: SuttonChen,
}

#[pymethods]
impl PySuttonChen {
    #[new]
    fn new(n_points: usize, family: &str) -> PyResult<Self> {
        if n_points < 2 {
            return Err(PyValueError::new_err(
                "a Sutton-Chen cluster needs at least two points",
            ));
        }
        let Some(fam) = SuttonChenFamily::parse(family) else {
            return Err(PyValueError::new_err(format!(
                "family must be \"12-6\", \"9-6\", or \"10-8\", got {family:?}"
            )));
        };
        Ok(Self {
            inner: SuttonChen::new(n_points, fam),
        })
    }

    #[staticmethod]
    fn sc_12_6(n_points: usize) -> PyResult<Self> {
        Self::new(n_points, "12-6")
    }

    #[staticmethod]
    fn sc_9_6(n_points: usize) -> PyResult<Self> {
        Self::new(n_points, "9-6")
    }

    #[staticmethod]
    fn sc_10_8(n_points: usize) -> PyResult<Self> {
        Self::new(n_points, "10-8")
    }

    #[getter]
    fn n_points(&self) -> usize {
        self.inner.n_points
    }

    #[getter]
    fn dim(&self) -> usize {
        Objective::dim(&self.inner)
    }

    #[getter]
    fn family(&self) -> &'static str {
        self.inner.family().label()
    }

    #[getter]
    fn n_exp(&self) -> i32 {
        self.inner.family().n_exp()
    }

    #[getter]
    fn m_exp(&self) -> i32 {
        self.inner.family().m_exp()
    }

    #[getter]
    fn c(&self) -> f64 {
        self.inner.family().c()
    }

    #[getter]
    fn basin_hopping_temperature(&self) -> f64 {
        self.inner.family().basin_hopping_temperature()
    }

    #[getter]
    fn bounds(&self, py: Python<'_>) -> PyResult<EindirPyBounds> {
        py_bounds_from(py, Objective::bounds(&self.inner))
    }

    fn eval(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let slice = x.as_slice()?;
        check_len(slice, 3 * self.inner.n_points, "SuttonChen")?;
        Ok(Objective::eval(&self.inner, ArrayView1::from(slice)))
    }

    fn grad<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = x.as_slice()?;
        check_len(slice, 3 * self.inner.n_points, "SuttonChen")?;
        let g = Gradient::grad(&self.inner, ArrayView1::from(slice));
        Ok(super::py_view1(py, g.view()))
    }

    fn value_and_gradient<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let slice = x.as_slice()?;
        check_len(slice, 3 * self.inner.n_points, "SuttonChen")?;
        let (e, g) = self.inner.value_and_gradient(ArrayView1::from(slice));
        Ok((e, super::py_view1(py, g.view())))
    }

    fn to_objective(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyObjective>> {
        let bounds = Objective::bounds(&slf.borrow().inner).clone();
        to_py_objective(py, slf.as_any(), &bounds)
    }

    fn __call__(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        self.eval(x)
    }

    fn __repr__(&self) -> String {
        format!(
            "SuttonChen({}, {:?})",
            self.inner.n_points,
            self.inner.family().label()
        )
    }
}

/// Cleri-Rosato Gupta embedded-atom cluster potential.
#[pyclass(name = "Gupta", module = "anneal._core")]
#[derive(Clone)]
pub struct PyGupta {
    pub(crate) inner: Gupta,
}

#[pymethods]
impl PyGupta {
    #[staticmethod]
    fn zinc(n_points: usize) -> PyResult<Self> {
        if n_points < 2 {
            return Err(PyValueError::new_err(
                "a Gupta cluster needs at least two atoms",
            ));
        }
        Ok(Self {
            inner: Gupta::zinc(n_points),
        })
    }

    #[staticmethod]
    fn platinum(n_points: usize) -> PyResult<Self> {
        if n_points < 2 {
            return Err(PyValueError::new_err(
                "a Gupta cluster needs at least two atoms",
            ));
        }
        Ok(Self {
            inner: Gupta::platinum(n_points),
        })
    }

    #[staticmethod]
    fn copper_gold(n_copper: usize, n_gold: usize) -> PyResult<Self> {
        if n_copper + n_gold < 2 {
            return Err(PyValueError::new_err(
                "a Gupta cluster needs at least two atoms",
            ));
        }
        Ok(Self {
            inner: Gupta::copper_gold(n_copper, n_gold),
        })
    }

    #[staticmethod]
    fn platinum_copper(n_platinum: usize, n_copper: usize) -> PyResult<Self> {
        if n_platinum + n_copper < 2 {
            return Err(PyValueError::new_err(
                "a Gupta cluster needs at least two atoms",
            ));
        }
        Ok(Self {
            inner: Gupta::platinum_copper(n_platinum, n_copper),
        })
    }

    #[staticmethod]
    fn elements(labels: Vec<u32>) -> PyResult<Self> {
        if labels.len() < 2 {
            return Err(PyValueError::new_err(
                "a Gupta cluster needs at least two atoms",
            ));
        }
        if !labels.iter().all(|&z| z == COPPER_Z || z == GOLD_Z) {
            return Err(PyValueError::new_err(
                "Gupta.elements labels must be COPPER_Z (29) or GOLD_Z (79)",
            ));
        }
        Ok(Self {
            inner: Gupta::elements(&labels),
        })
    }

    #[staticmethod]
    fn copper_gold_species(n_copper: usize, n_gold: usize) -> Vec<u32> {
        Gupta::copper_gold_species(n_copper, n_gold)
    }

    #[staticmethod]
    fn platinum_copper_species(n_platinum: usize, n_copper: usize) -> Vec<u32> {
        Gupta::platinum_copper_species(n_platinum, n_copper)
    }

    #[staticmethod]
    fn homonuclear_r0(label: u32) -> Option<f64> {
        Gupta::homonuclear_r0(label)
    }

    #[getter]
    fn n_points(&self) -> usize {
        self.inner.n_points()
    }

    #[getter]
    fn dim(&self) -> usize {
        Objective::dim(&self.inner)
    }

    #[getter]
    fn bounds(&self, py: Python<'_>) -> PyResult<EindirPyBounds> {
        py_bounds_from(py, Objective::bounds(&self.inner))
    }

    fn eval(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        if crate::potentials::cluster_kernels().is_none() {
            return Err(PyRuntimeError::new_err("Gupta requires rgpot kernels"));
        }
        let slice = x.as_slice()?;
        check_len(slice, 3 * self.inner.n_points(), "Gupta")?;
        Ok(Objective::eval(&self.inner, ArrayView1::from(slice)))
    }

    fn grad<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if crate::potentials::cluster_kernels().is_none() {
            return Err(PyRuntimeError::new_err("Gupta requires rgpot kernels"));
        }
        let slice = x.as_slice()?;
        check_len(slice, 3 * self.inner.n_points(), "Gupta")?;
        let g = Gradient::grad(&self.inner, ArrayView1::from(slice));
        Ok(super::py_view1(py, g.view()))
    }

    fn value_and_gradient<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        if crate::potentials::cluster_kernels().is_none() {
            return Err(PyRuntimeError::new_err("Gupta requires rgpot kernels"));
        }
        let slice = x.as_slice()?;
        check_len(slice, 3 * self.inner.n_points(), "Gupta")?;
        let (e, g) = self.inner.value_and_gradient(ArrayView1::from(slice));
        Ok((e, super::py_view1(py, g.view())))
    }

    fn to_objective(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyObjective>> {
        let bounds = Objective::bounds(&slf.borrow().inner).clone();
        to_py_objective(py, slf.as_any(), &bounds)
    }

    fn __call__(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        self.eval(x)
    }

    fn __repr__(&self) -> String {
        format!("Gupta(n_points={})", self.inner.n_points())
    }
}

/// Rigid Jorgensen TIP4P water cluster in block `(centres, rotations)` coordinates.
#[pyclass(name = "Tip4pCluster", module = "anneal._core")]
#[derive(Clone)]
pub struct PyTip4pCluster {
    pub(crate) inner: Tip4pCluster,
}

#[pymethods]
impl PyTip4pCluster {
    #[new]
    fn new(n_molecules: usize) -> PyResult<Self> {
        if n_molecules < 2 {
            return Err(PyValueError::new_err(
                "a TIP4P cluster needs at least two molecules",
            ));
        }
        Ok(Self {
            inner: Tip4pCluster::new(n_molecules),
        })
    }

    /// Putative Cambridge Cluster Database minimum in kJ/mol for 2..=21 molecules.
    #[staticmethod]
    fn ccd_minimum(n_molecules: usize) -> Option<f64> {
        ccd_tip4p_minimum(n_molecules)
    }

    #[getter]
    fn n_molecules(&self) -> usize {
        self.inner.n_molecules
    }

    #[getter]
    fn dim(&self) -> usize {
        Objective::dim(&self.inner)
    }

    #[getter]
    fn bounds(&self, py: Python<'_>) -> PyResult<EindirPyBounds> {
        py_bounds_from(py, Objective::bounds(&self.inner))
    }

    /// Return `(4, 3)` lab-frame site coordinates `(O, H, H, M)` for molecule `index`.
    fn molecule_sites<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        index: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let slice = x.as_slice()?;
        check_len(slice, 6 * self.inner.n_molecules, "Tip4pCluster")?;
        if index >= self.inner.n_molecules {
            return Err(PyValueError::new_err(format!(
                "molecule index {index} out of range for {} molecules",
                self.inner.n_molecules
            )));
        }
        let sites = self.inner.molecule_sites(ArrayView1::from(slice), index);
        let flat: Vec<f64> = sites.iter().flatten().copied().collect();
        Ok(PyArray1::from_vec(py, flat)
            .reshape([4, 3])
            .expect("4 x 3 sites"))
    }

    /// Fold rotation vectors into the ball of radius `pi`, returning a new vector.
    fn fold_rotations<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = x.as_slice()?;
        check_len(slice, 6 * self.inner.n_molecules, "Tip4pCluster")?;
        let mut folded = slice.to_vec();
        self.inner.fold_rotations(&mut folded);
        Ok(PyArray1::from_vec(py, folded))
    }

    fn eval(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let slice = x.as_slice()?;
        check_len(slice, 6 * self.inner.n_molecules, "Tip4pCluster")?;
        Ok(Objective::eval(&self.inner, ArrayView1::from(slice)))
    }

    fn grad<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = x.as_slice()?;
        check_len(slice, 6 * self.inner.n_molecules, "Tip4pCluster")?;
        let g = Gradient::grad(&self.inner, ArrayView1::from(slice));
        Ok(super::py_view1(py, g.view()))
    }

    fn value_and_gradient<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let slice = x.as_slice()?;
        check_len(slice, 6 * self.inner.n_molecules, "Tip4pCluster")?;
        let (e, g) = self.inner.value_and_gradient(ArrayView1::from(slice));
        Ok((e, super::py_view1(py, g.view())))
    }

    fn to_objective(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyObjective>> {
        let bounds = Objective::bounds(&slf.borrow().inner).clone();
        to_py_objective(py, slf.as_any(), &bounds)
    }

    fn __call__(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        self.eval(x)
    }

    fn __repr__(&self) -> String {
        format!("Tip4pCluster({})", self.inner.n_molecules)
    }
}

/// Rigid TIP4P cluster on the GMIN `NTIP` parameter set, in the block Rodrigues chart.
#[pyclass(name = "GminTip4p", module = "anneal._core")]
#[derive(Clone)]
pub struct PyGminTip4p {
    pub(crate) inner: GminTip4p,
}

#[pymethods]
impl PyGminTip4p {
    #[new]
    fn new(n_molecules: usize) -> PyResult<Self> {
        if n_molecules < 1 {
            return Err(PyValueError::new_err("TIP4P needs at least one molecule"));
        }
        Ok(Self {
            inner: GminTip4p::new(n_molecules),
        })
    }

    #[getter]
    fn n_molecules(&self) -> usize {
        self.inner.n_molecules()
    }

    #[getter]
    fn dim(&self) -> usize {
        Objective::dim(&self.inner)
    }

    #[getter]
    fn bounds(&self, py: Python<'_>) -> PyResult<EindirPyBounds> {
        py_bounds_from(py, Objective::bounds(&self.inner))
    }

    fn eval(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let slice = x.as_slice()?;
        check_len(slice, 6 * self.inner.n_molecules(), "GminTip4p")?;
        Ok(Objective::eval(&self.inner, ArrayView1::from(slice)))
    }

    fn grad<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = x.as_slice()?;
        check_len(slice, 6 * self.inner.n_molecules(), "GminTip4p")?;
        let g = Gradient::grad(&self.inner, ArrayView1::from(slice));
        Ok(super::py_view1(py, g.view()))
    }

    fn value_and_gradient<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let slice = x.as_slice()?;
        check_len(slice, 6 * self.inner.n_molecules(), "GminTip4p")?;
        let (e, g) = self.inner.value_and_gradient(ArrayView1::from(slice));
        Ok((e, super::py_view1(py, g.view())))
    }

    fn to_objective(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyObjective>> {
        let bounds = Objective::bounds(&slf.borrow().inner).clone();
        to_py_objective(py, slf.as_any(), &bounds)
    }

    fn __call__(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        self.eval(x)
    }

    fn __repr__(&self) -> String {
        format!("GminTip4p({})", self.inner.n_molecules())
    }
}

/// Convert Cartesian `(O, H, H)` triples (`9 * n` numbers) to block Rodrigues coordinates (`6 * n`).
#[pyfunction(name = "rigid_from_ohh")]
fn py_rigid_from_ohh<'py>(
    py: Python<'py>,
    cartesian: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = cartesian.as_slice()?;
    if slice.is_empty() || !slice.len().is_multiple_of(9) {
        return Err(PyValueError::new_err(
            "OHH coordinates must have non-zero length divisible by 9",
        ));
    }
    let state = rigid_from_ohh(slice);
    Ok(super::py_view1(py, state.view()))
}

/// Evaluate rigid TIP4P energy directly on Cartesian `(O, H, H)` triples (`9 * n` numbers).
#[pyfunction(name = "energy_ohh")]
fn py_energy_ohh(cartesian: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let slice = cartesian.as_slice()?;
    if slice.is_empty() || !slice.len().is_multiple_of(9) {
        return Err(PyValueError::new_err(
            "OHH coordinates must have non-zero length divisible by 9",
        ));
    }
    Ok(energy_ohh(slice))
}

/// Reorder interleaved GMIN `(c, omega)` rows (`6 * n`) to block `(centres, omega)`.
#[pyfunction(name = "block_from_interleaved_gmin")]
fn py_block_from_interleaved_gmin<'py>(
    py: Python<'py>,
    interleaved: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = interleaved.as_slice()?;
    if !slice.len().is_multiple_of(6) {
        return Err(PyValueError::new_err(
            "interleaved TIP4P state must have length divisible by 6",
        ));
    }
    Ok(PyArray1::from_vec(py, block_from_interleaved_gmin(slice)))
}

/// Reorder block `(centres, omega)` (`6 * n`) to interleaved GMIN `(c, omega)` rows.
#[pyfunction(name = "interleaved_gmin_from_block")]
fn py_interleaved_gmin_from_block<'py>(
    py: Python<'py>,
    block: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = block.as_slice()?;
    if !slice.len().is_multiple_of(6) {
        return Err(PyValueError::new_err(
            "block TIP4P state must have length divisible by 6",
        ));
    }
    Ok(PyArray1::from_vec(py, interleaved_gmin_from_block(slice)))
}

/// Reference `(4, 3)` body-frame site coordinates `(O, H, H, M)` in angstrom.
#[pyfunction(name = "gmin_tip4p_sites")]
fn py_gmin_tip4p_sites(py: Python<'_>) -> Bound<'_, PyArray2<f64>> {
    let sites = gmin_tip4p_sites();
    let flat: Vec<f64> = sites.iter().flatten().copied().collect();
    PyArray1::from_vec(py, flat)
        .reshape([4, 3])
        .expect("4 x 3 reference sites")
}

/// Extracted native differentiable potential for zero-GIL-overhead Rust dispatch.
#[derive(Clone)]
pub(crate) enum NativePotential {
    Pair(PairPotential),
    BinaryLj(BinaryLennardJones),
    SuttonChen(SuttonChen),
    Gupta(Gupta),
    Tip4p(Tip4pCluster),
    GminTip4p(GminTip4p),
}

impl NativePotential {
    pub(crate) fn extract_bound(obj: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        if let Ok(p) = obj.extract::<PyRef<'_, PyPairPotential>>() {
            return Ok(Some(Self::Pair(p.inner.clone())));
        }
        if let Ok(p) = obj.extract::<PyRef<'_, PyBinaryLennardJones>>() {
            return Ok(Some(Self::BinaryLj(p.inner.clone())));
        }
        if let Ok(p) = obj.extract::<PyRef<'_, PySuttonChen>>() {
            return Ok(Some(Self::SuttonChen(p.inner.clone())));
        }
        if let Ok(p) = obj.extract::<PyRef<'_, PyGupta>>() {
            if crate::potentials::cluster_kernels().is_none() {
                return Err(PyRuntimeError::new_err("Gupta requires rgpot kernels"));
            }
            return Ok(Some(Self::Gupta(p.inner.clone())));
        }
        if let Ok(p) = obj.extract::<PyRef<'_, PyTip4pCluster>>() {
            return Ok(Some(Self::Tip4p(p.inner.clone())));
        }
        if let Ok(p) = obj.extract::<PyRef<'_, PyGminTip4p>>() {
            return Ok(Some(Self::GminTip4p(p.inner.clone())));
        }
        Ok(None)
    }

    pub(crate) fn is_rigid_body(&self) -> bool {
        matches!(self, Self::Tip4p(_) | Self::GminTip4p(_))
    }

    pub(crate) fn n_points(&self) -> usize {
        match self {
            Self::Pair(p) => p.n_points,
            Self::BinaryLj(p) => p.n_points(),
            Self::SuttonChen(p) => p.n_points,
            Self::Gupta(p) => p.n_points(),
            Self::Tip4p(p) => p.n_molecules,
            Self::GminTip4p(p) => p.n_molecules(),
        }
    }
}

impl Objective<f64> for NativePotential {
    fn dim(&self) -> usize {
        match self {
            Self::Pair(p) => Objective::dim(p),
            Self::BinaryLj(p) => Objective::dim(p),
            Self::SuttonChen(p) => Objective::dim(p),
            Self::Gupta(p) => Objective::dim(p),
            Self::Tip4p(p) => Objective::dim(p),
            Self::GminTip4p(p) => Objective::dim(p),
        }
    }

    fn bounds(&self) -> &Bounds<f64> {
        match self {
            Self::Pair(p) => Objective::bounds(p),
            Self::BinaryLj(p) => Objective::bounds(p),
            Self::SuttonChen(p) => Objective::bounds(p),
            Self::Gupta(p) => Objective::bounds(p),
            Self::Tip4p(p) => Objective::bounds(p),
            Self::GminTip4p(p) => Objective::bounds(p),
        }
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        match self {
            Self::Pair(p) => Objective::eval(p, x),
            Self::BinaryLj(p) => Objective::eval(p, x),
            Self::SuttonChen(p) => Objective::eval(p, x),
            Self::Gupta(p) => Objective::eval(p, x),
            Self::Tip4p(p) => Objective::eval(p, x),
            Self::GminTip4p(p) => Objective::eval(p, x),
        }
    }
}

impl Gradient<f64> for NativePotential {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        match self {
            Self::Pair(p) => Gradient::grad(p, x),
            Self::BinaryLj(p) => Gradient::grad(p, x),
            Self::SuttonChen(p) => Gradient::grad(p, x),
            Self::Gupta(p) => Gradient::grad(p, x),
            Self::Tip4p(p) => Gradient::grad(p, x),
            Self::GminTip4p(p) => Gradient::grad(p, x),
        }
    }

    fn dim(&self) -> usize {
        Objective::dim(self)
    }
}

impl DifferentiableObjective<f64> for NativePotential {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        match self {
            Self::Pair(p) => p.value_and_gradient(x),
            Self::BinaryLj(p) => p.value_and_gradient(x),
            Self::SuttonChen(p) => p.value_and_gradient(x),
            Self::Gupta(p) => p.value_and_gradient(x),
            Self::Tip4p(p) => p.value_and_gradient(x),
            Self::GminTip4p(p) => p.value_and_gradient(x),
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("COPPER_Z", COPPER_Z)?;
    m.add("GOLD_Z", GOLD_Z)?;
    m.add("PLATINUM_Z", PLATINUM_Z)?;
    m.add_class::<PyPairPotential>()?;
    m.add_class::<PyBinaryLennardJones>()?;
    m.add_class::<PySuttonChen>()?;
    m.add_class::<PyGupta>()?;
    m.add_class::<PyTip4pCluster>()?;
    m.add_class::<PyGminTip4p>()?;
    m.add_function(wrap_pyfunction!(py_rigid_from_ohh, m)?)?;
    m.add_function(wrap_pyfunction!(py_energy_ohh, m)?)?;
    m.add_function(wrap_pyfunction!(py_block_from_interleaved_gmin, m)?)?;
    m.add_function(wrap_pyfunction!(py_interleaved_gmin_from_block, m)?)?;
    m.add_function(wrap_pyfunction!(py_gmin_tip4p_sites, m)?)?;
    Ok(())
}
