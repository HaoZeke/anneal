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
    }

    /// Observe class `i` at energy `e` and count one unit of effort.
    pub fn observe(&mut self, i: usize, e: f64) {
        self.resize(i + 1);
        self.y[i] = e;
        self.effort[i] += 1.0;
    }

    /// Record a hop between classes `a` and `b`.
    pub fn edge(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        self.resize(a.max(b) + 1);
        let (u, v) = if a < b { (a, b) } else { (b, a) };
        if !self.edges.iter().any(|&e| e == (u, v)) {
            self.edges.push((u, v));
        }
    }

    /// Posterior mean and variance at each node. Empty if there are no nodes.
    pub fn posterior(&self) -> Option<(Array1<f64>, Array1<f64>)> {
        let n = self.n;
        if n == 0 {
            return None;
        }
        let mut q = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            q[[i, i]] = self.nugget;
        }
        for &(a, b) in &self.edges {
            q[[a, a]] += 1.0;
            q[[b, b]] += 1.0;
            q[[a, b]] -= 1.0;
            q[[b, a]] -= 1.0;
        }
        let mut rhs = Array1::<f64>::zeros(n);
        for i in 0..n {
            if self.y[i].is_finite() {
                q[[i, i]] += self.noise;
                rhs[i] = self.noise * self.y[i];
            }
        }
        let mean = chol_solve(&q, &rhs)?;
        // Marginal variances: diagonal of Q^{-1}, by solving Q e_i.
        let mut var = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut e = Array1::<f64>::zeros(n);
            e[i] = 1.0;
            let z = chol_solve(&q, &e)?;
            var[i] = z[i].max(1e-12);
        }
        Some((mean, var))
    }

    /// Residual score of class `i` (high = worth a start).
    pub fn score(&self, i: usize) -> f64 {
        let (_, var) = match self.posterior() {
            Some(p) => p,
            None => return 1.0,
        };
        if i >= var.len() {
            return self.prior_var();
        }
        var[i] / (1.0 + self.effort.get(i).copied().unwrap_or(0.0))
    }

    /// Score of the unassigned residual cell `U`.
    pub fn residual_score(&self) -> f64 {
        self.prior_var()
    }

    fn prior_var(&self) -> f64 {
        1.0 / self.nugget.max(1e-12)
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
        if best_s > u {
            Some(best_i)
        } else {
            None
        }
    }
}

fn chol_solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = b.len();
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
        assert!(f.best_node().is_none(), "U should win over one mapped floor");
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
}
