//! GMRF residual on the floor graph.
//!
//! Nodes are energy classes. Edges are observed hops between classes. The
//! prior precision is the graph Laplacian plus a nugget, so neighbouring
//! floors share a level and disconnected components do not. The likelihood
//! pins each node to its observed `e_min`.
//!
//! The residual score of a class is the posterior *variance* (unknown floor
//! depth). An unassigned point — the residual cell `U` — scores the prior
//! variance, which is larger than any observed node. That is the LGCP-style
//! "hole" without a mesh on R^{3N} and without R-INLA.
//!
//! Preferential effort: `effort[i]` counts full quenches spent on class `i`.
//! The score is variance / (1 + effort), so a well-sampled uncertain node
//! loses to an empty hole.

use std::cell::RefCell;

use ndarray::{Array1, Array2};

/// Field on the class graph.
#[derive(Debug, Clone, Default)]
pub struct ResidualField {
    n: usize,
    edges: Vec<(usize, usize)>,
    /// Observed floor energies, `nan` if a node has no observation yet.
    y: Vec<f64>,
    /// Full quenches spent on each node.
    effort: Vec<f64>,
    /// Prior precision nugget. One: the Laplacian scale.
    nugget: f64,
    /// Likelihood precision.
    noise: f64,
    /// Cached Cholesky factor of the graph posterior precision.
    factor: RefCell<Option<Array2<f64>>>,
}

impl ResidualField {
    /// Empty field.
    pub fn new() -> Self {
        Self {
            nugget: 1.0,
            noise: 4.0,
            ..Self::default()
        }
    }

    /// Ensure nodes `0..n` exist.
    pub fn resize(&mut self, n: usize) {
        if n <= self.n {
            return;
        }
        self.y.resize(n, f64::NAN);
        self.effort.resize(n, 0.0);
        self.n = n;
        self.invalidate();
    }

    /// Observe class `i` at energy `e` and count one unit of effort.
    pub fn observe(&mut self, i: usize, e: f64) {
        self.resize(i + 1);
        if !self.y[i].is_finite() || e < self.y[i] {
            self.y[i] = e;
        }
        self.effort[i] += 1.0;
        self.invalidate();
    }

    /// Record a hop between classes `a` and `b`.
    pub fn edge(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        self.resize(a.max(b) + 1);
        let (u, v) = if a < b { (a, b) } else { (b, a) };
        if !self.edges.contains(&(u, v)) {
            self.edges.push((u, v));
            self.invalidate();
        }
    }

    /// Posterior mean and variance at each node. Empty if there are no nodes.
    pub fn posterior(&self) -> Option<(Array1<f64>, Array1<f64>)> {
        let n = self.n;
        if n == 0 {
            return None;
        }
        self.ensure_factor()?;
        let factor = self.factor.borrow();
        let factor = factor.as_ref()?;
        let mut rhs = Array1::<f64>::zeros(n);
        for i in 0..n {
            if self.y[i].is_finite() {
                rhs[i] = self.noise * self.y[i];
            }
        }
        let mean = solve_factor(factor, &rhs)?;
        // Marginal variances: diagonal of Q^{-1}, by solving Q e_i.
        let mut var = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut e = Array1::<f64>::zeros(n);
            e[i] = 1.0;
            let z = solve_factor(factor, &e)?;
            var[i] = z[i].max(1e-12);
        }
        Some((mean, var))
    }

    /// Residual score of class `i` (high = worth a start).
    pub fn score(&self, i: usize) -> f64 {
        if i >= self.n {
            return self.prior_var();
        }
        let Some(variance) = self.marginal_variance(i) else {
            return self.prior_var();
        };
        variance / (1.0 + self.effort.get(i).copied().unwrap_or(0.0))
    }

    /// Score of the unassigned residual cell `U`.
    pub fn residual_score(&self) -> f64 {
        self.prior_var()
    }

    fn prior_var(&self) -> f64 {
        1.0 / self.nugget.max(1e-12)
    }

    fn marginal_variance(&self, i: usize) -> Option<f64> {
        self.ensure_factor()?;
        let factor = self.factor.borrow();
        let factor = factor.as_ref()?;
        let mut basis = Array1::<f64>::zeros(self.n);
        basis[i] = 1.0;
        solve_factor(factor, &basis).map(|solution| solution[i].max(1e-12))
    }

    fn ensure_factor(&self) -> Option<()> {
        if self.factor.borrow().is_none() {
            let factor = cholesky(&self.precision())?;
            *self.factor.borrow_mut() = Some(factor);
        }
        Some(())
    }

    fn precision(&self) -> Array2<f64> {
        let mut precision = Array2::<f64>::zeros((self.n, self.n));
        for i in 0..self.n {
            precision[[i, i]] = self.nugget;
        }
        for &(a, b) in &self.edges {
            precision[[a, a]] += 1.0;
            precision[[b, b]] += 1.0;
            precision[[a, b]] -= 1.0;
            precision[[b, a]] -= 1.0;
        }
        for i in 0..self.n {
            if self.y[i].is_finite() {
                precision[[i, i]] += self.noise;
            }
        }
        precision
    }

    fn invalidate(&mut self) {
        *self.factor.get_mut() = None;
    }

    /// Class with the highest residual score, or `None` if `U` wins.
    pub fn best_node(&self) -> Option<usize> {
        if self.n == 0 {
            return None;
        }
        let u = self.residual_score();
        let mut best_i = 0usize;
        let mut best_s = self.score(0);
        for i in 1..self.n {
            let s = self.score(i);
            if s > best_s {
                best_s = s;
                best_i = i;
            }
        }
        if best_s > u { Some(best_i) } else { None }
    }
}

fn cholesky(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return None;
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    Some(l)
}

fn solve_factor(l: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = b.len();
    if l.nrows() != n || l.ncols() != n {
        return None;
    }
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[[i, k]] * y[k];
        }
        y[i] = s / l[[i, i]];
    }
    let mut z = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[[k, i]] * z[k];
        }
        z[i] = s / l[[i, i]];
    }
    Some(z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unobserved_scores_above_a_pinned_node() {
        let mut f = ResidualField::new();
        f.observe(0, -10.0);
        f.observe(0, -10.0);
        f.observe(0, -10.0);
        let s0 = f.score(0);
        let u = f.residual_score();
        assert!(
            u > s0,
            "residual {u} should outrank a thrice-observed node {s0}"
        );
        assert!(
            f.best_node().is_none(),
            "U should win over one mapped floor"
        );
    }

    #[test]
    fn an_edge_ties_neighbours() {
        let mut f = ResidualField::new();
        f.observe(0, -10.0);
        f.observe(1, -10.2);
        f.edge(0, 1);
        let (mean, _) = f.posterior().unwrap();
        assert!(
            (mean[0] - mean[1]).abs() < 0.3,
            "neighbours should share a level, got {} and {}",
            mean[0],
            mean[1]
        );
    }

    #[test]
    fn repeated_observations_retain_the_lowest_floor_and_count_effort() {
        let mut f = ResidualField::new();
        f.observe(0, -10.0);
        f.observe(0, -8.0);

        assert_eq!(f.y[0], -10.0);
        assert_eq!(f.effort[0], 2.0);
        let (mean, _) = f.posterior().expect("observed node has a posterior");
        assert!((mean[0] + 8.0).abs() < 1e-12);
    }
}
