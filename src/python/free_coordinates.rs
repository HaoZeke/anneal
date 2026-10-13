//! The free coordinates of a box with fixed ones.
//!
//! A coordinate whose two ends are equal is fixed. `run`, `run_qmc` and
//! `global_optimize` search the other coordinates, as `minimize` does, and
//! complete each point with the fixed values before the callable sees it, so
//! a search on such a box makes the calls of the same search on its free
//! coordinates.

use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// The free coordinates of a box and the values its fixed ones keep.
#[derive(Clone)]
pub(super) struct FixedCoordinates {
    /// A point of the box; every completed point has its fixed coordinates.
    template: Array1<f64>,
    /// The indices of the free coordinates, in order.
    free: Vec<usize>,
    /// The box of the free coordinates.
    bounds: Bounds<f64>,
}

impl FixedCoordinates {
    /// The fixed coordinates of `bounds` with the values of `template`, a
    /// point of the box, or `None` when no coordinate is fixed.
    pub(super) fn of(bounds: &Bounds<f64>, template: Array1<f64>) -> Option<Self> {
        let free: Vec<usize> = (0..bounds.dims)
            .filter(|&i| bounds.low[i] < bounds.high[i])
            .collect();
        if free.len() == bounds.dims {
            return None;
        }
        let reduced = Bounds::new(
            free.iter().map(|&i| bounds.low[i]).collect(),
            free.iter().map(|&i| bounds.high[i]).collect(),
            bounds.slack,
        );
        Some(Self {
            template,
            free,
            bounds: reduced,
        })
    }

    /// Whether every coordinate is fixed.
    pub(super) fn all(&self) -> bool {
        self.free.is_empty()
    }

    /// The template, which is the box's only point when every coordinate is
    /// fixed.
    pub(super) fn template(&self) -> &Array1<f64> {
        &self.template
    }

    /// `z`, a point of the free coordinates, completed with the fixed values.
    pub(super) fn full(&self, z: ArrayView1<f64>) -> Array1<f64> {
        let mut x = self.template.clone();
        for (&i, &value) in self.free.iter().zip(z.iter()) {
            x[i] = value;
        }
        x
    }

    /// The free coordinates of `x`, a point of the whole box.
    pub(super) fn free_part(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.free.iter().map(|&i| x[i]).collect()
    }

    /// `pos`, a point of the free coordinates of `fixed` when it is given,
    /// completed with the fixed values; `pos` itself otherwise.
    pub(super) fn complete(fixed: Option<&Self>, pos: Vec<f64>) -> Vec<f64> {
        match fixed {
            Some(fixed) => fixed.full(ArrayView1::from(&pos)).to_vec(),
            None => pos,
        }
    }

    /// `inner`, an objective on the whole box, seen from the free coordinates.
    pub(super) fn objective<'a, O>(&self, inner: &'a O) -> FreeObjective<'a, O> {
        FreeObjective {
            inner,
            fixed: self.clone(),
        }
    }

    /// `inner`, a gradient read on the whole box, seen from the free
    /// coordinates.
    pub(super) fn gradient<'a, G>(&'a self, inner: &'a G) -> FreeGradient<'a, G> {
        FreeGradient { inner, fixed: self }
    }
}

/// An objective on a whole box seen from its free coordinates. A batch goes
/// to the objective's own batch evaluation whole.
pub(super) struct FreeObjective<'a, O> {
    inner: &'a O,
    fixed: FixedCoordinates,
}

impl<O: Objective<f64>> Objective<f64> for FreeObjective<'_, O> {
    fn dim(&self) -> usize {
        self.fixed.free.len()
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.fixed.bounds
    }

    fn eval(&self, z: ArrayView1<f64>) -> f64 {
        self.inner.eval(self.fixed.full(z).view())
    }

    fn eval_batch(&self, rows: ArrayView2<f64>) -> Array1<f64> {
        let mut full = Array2::zeros((rows.nrows(), self.fixed.template.len()));
        for (mut point, row) in full.outer_iter_mut().zip(rows.outer_iter()) {
            point.assign(&self.fixed.full(row));
        }
        self.inner.eval_batch(full.view())
    }
}

/// A gradient read at full length on the whole box, of which the free
/// coordinates' part is kept.
pub(super) struct FreeGradient<'a, G> {
    inner: &'a G,
    fixed: &'a FixedCoordinates,
}

impl<G: Gradient<f64>> Gradient<f64> for FreeGradient<'_, G> {
    fn grad(&self, z: ArrayView1<f64>) -> Array1<f64> {
        let g = self.inner.grad(self.fixed.full(z).view());
        self.fixed.free_part(g.view())
    }

    fn dim(&self) -> usize {
        self.fixed.free.len()
    }
}
