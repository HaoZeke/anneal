//! Limited-memory quasi-Newton relaxation whose curvature survives between calls.
//!
//! A hopping chain relaxes thousands of times, and each relaxation starts from
//! a perturbation of a structure that was already relaxed. The curvature at the
//! new start is close to the curvature at the old minimum, and a minimiser that
//! discards its memory between calls rediscovers it every time.
//!
//! Measured on a 75-point Lennard-Jones cluster with a library minimiser: a
//! relaxation costs about 270 evaluations from a random start and 270 from a
//! perturbed minimum, so nothing is being reused. Since a work ledger charges
//! evaluations rather than seconds, halving that count doubles the number of
//! hops a budget buys, which is the only lever on solve rate that does not
//! require more budget.
//!
//! The memory is dropped when a proposal is accepted from far away, because
//! curvature carried across a structural change is worse than none.

use ndarray::{Array1, ArrayView1};

/// Stored curvature pair from one accepted step.
struct Pair {
    s: Array1<f64>,
    y: Array1<f64>,
    rho: f64,
}

/// L-BFGS with memory that persists across relaxations.
pub struct WarmLbfgs {
    memory: Vec<Pair>,
    /// Pairs retained; the usual choice is between five and ten.
    pub max_pairs: usize,
    /// Gradient infinity norm below which a relaxation is converged.
    pub gtol: f64,
    /// Backtracking contraction factor.
    pub backtrack: f64,
    /// Armijo sufficient-decrease constant.
    pub armijo: f64,
    /// Line-search steps attempted before the direction is abandoned.
    pub max_backtracks: usize,
}

impl Default for WarmLbfgs {
    fn default() -> Self {
        Self {
            memory: Vec::new(),
            max_pairs: 8,
            gtol: 1e-6,
            backtrack: 0.5,
            armijo: 1e-4,
            max_backtracks: 20,
        }
    }
}

impl WarmLbfgs {
    /// Discards the stored curvature.
    ///
    /// Called when the chain moves somewhere structurally different, where the
    /// retained pairs describe a Hessian that no longer applies.
    pub fn forget(&mut self) {
        self.memory.clear();
    }

    /// Pairs currently held.
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    /// True when no curvature is stored.
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// Two-loop recursion: applies the inverse-Hessian approximation to `g`.
    fn direction(&self, g: ArrayView1<f64>) -> Array1<f64> {
        let mut q = g.to_owned();
        let m = self.memory.len();
        let mut alpha = vec![0.0; m];
        for (i, p) in self.memory.iter().enumerate().rev() {
            let a = p.rho * p.s.dot(&q);
            alpha[i] = a;
            q.scaled_add(-a, &p.y);
        }
        // Scale by the most recent pair's curvature, which is what makes the
        // first step of a warm start the right size rather than a guess.
        if let Some(p) = self.memory.last() {
            let yy = p.y.dot(&p.y);
            if yy > 0.0 {
                q *= p.s.dot(&p.y) / yy;
            }
        }
        for (i, p) in self.memory.iter().enumerate() {
            let b = p.rho * p.y.dot(&q);
            q.scaled_add(alpha[i] - b, &p.s);
        }
        q.mapv_inplace(|v| -v);
        q
    }

    fn push(&mut self, s: Array1<f64>, y: Array1<f64>) {
        let sy = s.dot(&y);
        // Curvature condition: a non-positive pair would make the
        // approximation indefinite and the direction an ascent one.
        if sy <= 1e-12 {
            return;
        }
        self.memory.push(Pair { s, y, rho: 1.0 / sy });
        if self.memory.len() > self.max_pairs {
            self.memory.remove(0);
        }
    }

    /// Relaxes `x0`, calling `fg` for value and gradient.
    ///
    /// `fg` returns `None` when the caller's budget is spent, which ends the
    /// relaxation where it stands. Returns the value, the point, and the number
    /// of evaluations used.
    pub fn minimize<F>(
        &mut self,
        x0: ArrayView1<f64>,
        max_iter: usize,
        mut fg: F,
    ) -> (f64, Array1<f64>, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        let mut x = x0.to_owned();
        let mut evals = 0usize;
        let (mut f, mut g) = match fg(x.view()) {
            Some(v) => v,
            None => return (f64::INFINITY, x, evals),
        };
        evals += 1;

        for _ in 0..max_iter {
            if g.iter().fold(0.0_f64, |a, v| a.max(v.abs())) < self.gtol {
                break;
            }
            let d = self.direction(g.view());
            let slope = d.dot(&g);
            if slope >= 0.0 {
                // A stored pair has gone stale: the direction points uphill.
                self.forget();
                continue;
            }
            let mut step = 1.0;
            let mut moved = false;
            for _ in 0..self.max_backtracks {
                let mut trial = x.clone();
                trial.scaled_add(step, &d);
                match fg(trial.view()) {
                    None => return (f, x, evals),
                    Some((ft, gt)) => {
                        evals += 1;
                        if ft <= f + self.armijo * step * slope {
                            let mut s = trial.clone();
                            s -= &x;
                            let mut y = gt.clone();
                            y -= &g;
                            self.push(s, y);
                            x = trial;
                            f = ft;
                            g = gt;
                            moved = true;
                            break;
                        }
                        step *= self.backtrack;
                    }
                }
            }
            if !moved {
                break;
            }
        }
        (f, x, evals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// Ill-conditioned quadratic: the case where curvature reuse should pay.
    fn quad(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let scales = [1.0, 10.0, 100.0, 1000.0];
        let mut f = 0.0;
        let mut g = Array1::zeros(x.len());
        for i in 0..x.len() {
            let c = scales[i % scales.len()];
            f += 0.5 * c * x[i] * x[i];
            g[i] = c * x[i];
        }
        (f, g)
    }

    #[test]
    fn converges_on_an_ill_conditioned_quadratic() {
        let mut opt = WarmLbfgs::default();
        let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let (f, x, evals) = opt.minimize(x0.view(), 200, |v| Some(quad(v)));
        assert!(f < 1e-10, "did not converge, f = {f}, evals = {evals}");
        assert!(x.iter().all(|v| v.abs() < 1e-4));
    }

    /// The property the type exists for: a second relaxation starting near the
    /// first one's answer costs fewer evaluations when curvature is retained.
    #[test]
    fn retained_curvature_costs_fewer_evaluations() {
        let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let mut warm = WarmLbfgs::default();
        let (_, xa, _) = warm.minimize(x0.view(), 200, |v| Some(quad(v)));
        let mut perturbed = xa.clone();
        for (i, v) in perturbed.iter_mut().enumerate() {
            *v += if i % 2 == 0 { 0.02 } else { -0.02 };
        }
        let (_, _, warm_evals) = warm.minimize(perturbed.view(), 200, |v| Some(quad(v)));

        let mut cold = WarmLbfgs::default();
        let (_, _, cold_evals) = cold.minimize(perturbed.view(), 200, |v| Some(quad(v)));

        assert!(
            warm_evals < cold_evals,
            "retained curvature should cost less: warm {warm_evals}, cold {cold_evals}"
        );
    }

    #[test]
    fn forget_clears_the_memory() {
        let mut opt = WarmLbfgs::default();
        let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0]);
        opt.minimize(x0.view(), 50, |v| Some(quad(v)));
        assert!(!opt.is_empty(), "a relaxation should store curvature");
        opt.forget();
        assert!(opt.is_empty());
    }

    #[test]
    fn stops_when_the_budget_ends() {
        let mut opt = WarmLbfgs::default();
        let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0]);
        let mut left = 3;
        let (_, _, evals) = opt.minimize(x0.view(), 200, |v| {
            if left == 0 {
                return None;
            }
            left -= 1;
            Some(quad(v))
        });
        assert!(evals <= 3, "spent {evals} with a budget of 3");
    }
}
