//! Choosing a move from where the chain is standing, not from a global rate.
//!
//! The allocator in [`crate::allocate`] keeps one success rate per move and
//! samples from it. That is the right model when a move has a rate; it is the
//! wrong model when a move has a *precondition*. Wales and Doye's angular move
//! is the clear case: it is not applied at some frequency, it is applied when
//! the worst-bound point crosses a pair-energy criterion. A context-free
//! allocator learns the average of the times it was and was not appropriate,
//! and that average describes no situation the chain is ever in.
//!
//! The model here is hierarchical. Each move's value is linear in a context
//! vector, with a component shared across moves and a per-move deviation:
//!
//! ```text
//! value(context c, move a) = c . (w_shared + w_a)
//! ```
//!
//! Sharing matters because the arms do not get equal data. A move the allocator
//! has learned to avoid is sampled rarely, so its own coefficients stay
//! uncertain forever; the shared component is fitted on every observation and
//! carries what is true of the landscape rather than of one move. In factor
//! terms this is a rank-one-plus-residual decomposition of the context-by-move
//! value matrix, which is what makes it estimable from far less data than the
//! full matrix.
//!
//! Selection is Thompson sampling: draw coefficients from each move's posterior,
//! score the current context, take the best. A floor keeps every move sampled at
//! some rate whatever the posterior says, for the same reason the screen in
//! [`crate::screen`] keeps an exploration floor: a rule that only ever picks
//! what it already believes never learns it was wrong.

use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;

/// A conjugate Bayesian linear model with a Gaussian likelihood.
///
/// Kept separate from the one in [`crate::screen`] because this one is sampled
/// from rather than asked for a tail probability, and the sampling is what
/// Thompson selection needs.
#[derive(Debug, Clone)]
struct BayesLinear {
    precision: Array2<f64>,
    rhs: Array1<f64>,
    n: usize,
}

impl BayesLinear {
    fn new(d: f64, dim: usize) -> Self {
        Self {
            precision: Array2::eye(dim) * d,
            rhs: Array1::zeros(dim),
            n: 0,
        }
    }

    fn observe(&mut self, x: ArrayView1<f64>, y: f64) {
        let d = self.rhs.len();
        for i in 0..d {
            for j in 0..d {
                self.precision[[i, j]] += x[i] * x[j];
            }
            self.rhs[i] += x[i] * y;
        }
        self.n += 1;
    }

    fn mean(&self) -> Array1<f64> {
        solve(self.precision.view(), self.rhs.view()).unwrap_or_else(|| Array1::zeros(self.rhs.len()))
    }

    /// Posterior variance of the value at `x`, `x' P^-1 x`.
    fn variance(&self, x: ArrayView1<f64>) -> f64 {
        match solve(self.precision.view(), x) {
            Some(z) => x.iter().zip(z.iter()).map(|(p, q)| p * q).sum::<f64>().max(0.0),
            None => 1.0,
        }
    }
}

/// A contextual allocator over a fixed set of moves.
pub struct ContextualAllocator {
    shared: BayesLinear,
    arms: Vec<BayesLinear>,
    /// Rate at which a move is chosen uniformly, whatever the posterior says.
    pub floor: f64,
    /// Scale on the posterior standard deviation when sampling.
    ///
    /// One is Thompson sampling proper. Larger explores more, smaller behaves
    /// more greedily; exposed because the reward here is bounded in `[0, 1]`
    /// and a posterior fitted on a bounded reward is over-confident at its
    /// edges.
    pub exploration: f64,
    /// Times each move was chosen.
    pub picks: Vec<usize>,
    /// Times each move was rewarded.
    pub wins: Vec<usize>,
    /// Choices made uniformly by the floor.
    pub forced: usize,
}

impl ContextualAllocator {
    /// Allocator over `n_moves` with a `dim`-dimensional context.
    ///
    /// The context must include an intercept if the caller wants one; nothing
    /// here adds it.
    pub fn new(n_moves: usize, dim: usize, floor: f64) -> Self {
        assert!(n_moves > 0, "an allocator needs at least one move");
        assert!(dim > 0, "a context needs at least one feature");
        assert!(
            (0.0..=1.0).contains(&floor),
            "the floor is a rate, got {floor}"
        );
        Self {
            shared: BayesLinear::new(1e-2, dim),
            arms: (0..n_moves).map(|_| BayesLinear::new(1.0, dim)).collect(),
            floor,
            exploration: 1.0,
            picks: vec![0; n_moves],
            wins: vec![0; n_moves],
            forced: 0,
        }
    }

    /// Moves this allocator chooses between.
    pub fn len(&self) -> usize {
        self.arms.len()
    }

    /// Whether there are no moves, which the constructor forbids.
    pub fn is_empty(&self) -> bool {
        self.arms.is_empty()
    }

    /// Posterior mean value of `move_index` in `context`.
    pub fn value(&self, move_index: usize, context: ArrayView1<f64>) -> f64 {
        let s = self.shared.mean();
        let a = self.arms[move_index].mean();
        context
            .iter()
            .zip(s.iter().zip(a.iter()))
            .map(|(c, (p, q))| c * (p + q))
            .sum()
    }

    /// Picks a move for `context` by Thompson sampling.
    pub fn select<R: Rng + ?Sized>(&mut self, context: ArrayView1<f64>, rng: &mut R) -> usize {
        let k = if rng.random::<f64>() < self.floor {
            self.forced += 1;
            rng.random_range(0..self.arms.len())
        } else {
            let mut best = 0usize;
            let mut best_score = f64::NEG_INFINITY;
            for a in 0..self.arms.len() {
                // Sampled value: posterior mean plus a normal draw scaled by
                // the posterior standard deviation at this context. A move
                // never tried in this region has a wide posterior and can win
                // on the draw, which is the exploration Thompson supplies.
                let sd = (self.shared.variance(context) + self.arms[a].variance(context))
                    .sqrt()
                    * self.exploration;
                let score = self.value(a, context) + sd * standard_normal(rng);
                if score > best_score {
                    best_score = score;
                    best = a;
                }
            }
            best
        };
        self.picks[k] += 1;
        k
    }

    /// Records what happened, with `reward` in `[0, 1]`.
    pub fn update(&mut self, move_index: usize, context: ArrayView1<f64>, reward: f64) {
        if move_index >= self.arms.len() || !reward.is_finite() {
            return;
        }
        if reward > 0.5 {
            self.wins[move_index] += 1;
        }
        // The shared component sees every observation; the arm sees what the
        // shared component does not already explain, so the two are not fitted
        // against each other.
        let s = self.shared.mean();
        let explained: f64 = context.iter().zip(s.iter()).map(|(c, p)| c * p).sum();
        self.shared.observe(context, reward);
        self.arms[move_index].observe(context, reward - explained);
    }
}

/// Solves `a z = b` for small symmetric positive-definite `a`, by Cholesky.
fn solve(a: ndarray::ArrayView2<f64>, b: ArrayView1<f64>) -> Option<Array1<f64>> {
    let n = b.len();
    if a.nrows() != n || a.ncols() != n {
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

/// Box-Muller, one draw.
fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-12);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Two contexts, two moves, and each move is right in one of them. A
    /// context-free allocator cannot represent this at all: both moves have the
    /// same marginal success rate.
    #[test]
    fn it_learns_which_move_belongs_in_which_context() {
        let mut a = ContextualAllocator::new(2, 2, 0.05);
        let mut rng = StdRng::seed_from_u64(1);
        let left = array![1.0, 0.0];
        let right = array![0.0, 1.0];
        for i in 0..4000 {
            let (ctx, good) = if i % 2 == 0 {
                (left.view(), 0usize)
            } else {
                (right.view(), 1usize)
            };
            let k = a.select(ctx, &mut rng);
            let reward = if k == good { 1.0 } else { 0.0 };
            a.update(k, ctx, reward);
        }
        assert!(
            a.value(0, left.view()) > a.value(1, left.view()),
            "move 0 should win on the left: {} against {}",
            a.value(0, left.view()),
            a.value(1, left.view())
        );
        assert!(
            a.value(1, right.view()) > a.value(0, right.view()),
            "move 1 should win on the right: {} against {}",
            a.value(1, right.view()),
            a.value(0, right.view())
        );
    }

    /// The selection has to follow what was learned, not merely the values.
    #[test]
    fn selection_follows_the_context() {
        let mut a = ContextualAllocator::new(2, 2, 0.0);
        let mut rng = StdRng::seed_from_u64(7);
        let left = array![1.0, 0.0];
        let right = array![0.0, 1.0];
        for i in 0..6000 {
            let (ctx, good) = if i % 2 == 0 {
                (left.view(), 0usize)
            } else {
                (right.view(), 1usize)
            };
            let k = a.select(ctx, &mut rng);
            a.update(k, ctx, if k == good { 1.0 } else { 0.0 });
        }
        let mut zero_on_left = 0;
        let mut one_on_right = 0;
        for _ in 0..400 {
            if a.select(left.view(), &mut rng) == 0 {
                zero_on_left += 1;
            }
            if a.select(right.view(), &mut rng) == 1 {
                one_on_right += 1;
            }
        }
        assert!(zero_on_left > 300, "{zero_on_left} of 400 on the left");
        assert!(one_on_right > 300, "{one_on_right} of 400 on the right");
    }

    /// A move the allocator has decided against still has to be tried, or a
    /// move that only pays in a region the chain has not reached yet is lost.
    #[test]
    fn the_floor_keeps_every_move_in_play() {
        let mut a = ContextualAllocator::new(4, 2, 0.2);
        let mut rng = StdRng::seed_from_u64(3);
        let ctx = array![1.0, 0.5];
        for _ in 0..4000 {
            let k = a.select(ctx.view(), &mut rng);
            // Only move 0 ever pays.
            a.update(k, ctx.view(), if k == 0 { 1.0 } else { 0.0 });
        }
        for k in 1..4 {
            assert!(
                a.picks[k] > 100,
                "move {k} was picked {} times in 4000",
                a.picks[k]
            );
        }
        assert!(a.picks[0] > a.picks[1], "the paying move should dominate");
    }

    /// With a constant context there is nothing to condition on, and the
    /// allocator has to degrade to ordinary bandit behaviour rather than break.
    #[test]
    fn a_constant_context_reduces_to_picking_the_better_move() {
        let mut a = ContextualAllocator::new(3, 1, 0.05);
        let mut rng = StdRng::seed_from_u64(11);
        let ctx = array![1.0];
        for i in 0..5000 {
            let k = a.select(ctx.view(), &mut rng);
            // Move 1 pays four times in five, the others one in five.
            let p = if k == 1 { 4 } else { 1 };
            a.update(k, ctx.view(), if i % 5 < p { 1.0 } else { 0.0 });
        }
        assert!(
            a.picks[1] > a.picks[0] && a.picks[1] > a.picks[2],
            "picks {:?} should favour move 1",
            a.picks
        );
    }

    #[test]
    fn a_bad_reward_or_index_is_ignored_rather_than_poisoning_the_posterior() {
        let mut a = ContextualAllocator::new(2, 2, 0.0);
        let ctx = array![1.0, 0.0];
        a.update(9, ctx.view(), 1.0);
        a.update(0, ctx.view(), f64::NAN);
        assert_eq!(a.wins.iter().sum::<usize>(), 0);
        assert!(a.value(0, ctx.view()).is_finite());
    }
}
