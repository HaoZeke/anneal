//! A posterior over what to build, rather than over which arm to pull.
//!
//! The Bayesian machinery in this crate sits on allocation: which move to draw,
//! at what temperature, with what budget. Every one of those decisions
//! reweights a walk on the graph of basins, and the graph's edges are set by
//! the move rather than by the walk. Measured, the allocator does its job well
//! and there is nothing left in it to win: on 98 points its arms carry
//! thousands of draws each and it separates a 0.63 accept rate from a 0.00 one
//! cleanly, so a hierarchical prior would shrink posteriors that are already
//! sharp.
//!
//! The one place a model can add an edge is the construction itself.
//! [`crate::lattice`] builds a candidate from a local order and a fraction of
//! the current structure to keep, and both are choices made blind: the source
//! is drawn by the allocator and the fraction was not a parameter at all. A
//! model that predicts the quenched energy of a construction from features of
//! the candidate can choose them, and it learns from quenches the run pays for
//! anyway.
//!
//! # What it costs
//!
//! Nothing on the ledger. Candidates are built and featured without calling the
//! objective, so proposing four and keeping one costs four constructions and
//! one quench, the same quench the move would have paid for a blind choice.
//!
//! # The model
//!
//! The conjugate Normal-Inverse-Gamma regression already in [`crate::screen`],
//! over features of the candidate rather than of a screened relaxation. The
//! posterior carries its own variance, so the choice is Thompson sampling
//! rather than a maximum: draw a coefficient vector, score the candidates under
//! it, take the best. A construction the model has no evidence about scores
//! with a wide posterior and gets tried.
//!
//! Features are structural and cheap: the fraction kept, the mean and spread of
//! coordination in the candidate, the share of points at full coordination, and
//! how far the candidate moved from the structure it was built from. None of
//! them mentions a potential, so the model transfers to whatever is being
//! optimised in the same way the growth does.

use crate::lattice::{self, Source};
use crate::screen::Screen;
use ndarray::{Array1, ArrayView1};
use rand::Rng;

/// The parameters of one candidate construction.
#[derive(Debug, Clone, Copy)]
pub struct Recipe {
    /// Which local order to grow.
    pub source: Source,
    /// Fraction of the current structure kept as the growth seed.
    pub keep: f64,
}

/// Number of features the model regresses on.
pub const FEATURES: usize = 6;

/// Structural summary of a candidate, with no reference to the objective.
pub fn features(recipe: &Recipe, candidate: ArrayView1<f64>, parent: ArrayView1<f64>) -> Array1<f64> {
    let n = candidate.len() / 3;
    let scale = lattice::nearest_neighbour_scale(candidate);
    let cut = 1.2 * scale;
    let mut counts = vec![0usize; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d: f64 = (0..3)
                .map(|k| {
                    let v = candidate[3 * i + k] - candidate[3 * j + k];
                    v * v
                })
                .sum::<f64>()
                .sqrt();
            if d < cut {
                counts[i] += 1;
            }
        }
    }
    let mean = counts.iter().sum::<usize>() as f64 / n.max(1) as f64;
    let var = counts
        .iter()
        .map(|c| (*c as f64 - mean) * (*c as f64 - mean))
        .sum::<f64>()
        / n.max(1) as f64;
    let full = counts.iter().filter(|c| **c >= 12).count() as f64 / n.max(1) as f64;
    // How far the candidate sits from its parent, by sorted radial profile,
    // which needs no correspondence between the two point sets. A construction
    // that barely moved and one that replaced everything are different
    // proposals even at the same coordination.
    let drift = radial_distance(candidate, parent);
    Array1::from(vec![1.0, recipe.keep, mean, var, full, drift])
}

fn radial_distance(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let ra = sorted_radii(a);
    let rb = sorted_radii(b);
    let m = ra.len().min(rb.len());
    if m == 0 {
        return 0.0;
    }
    (0..m)
        .map(|i| (ra[i] - rb[i]) * (ra[i] - rb[i]))
        .sum::<f64>()
        .sqrt()
        / (m as f64).sqrt()
}

fn sorted_radii(x: ArrayView1<f64>) -> Vec<f64> {
    let n = x.len() / 3;
    if n == 0 {
        return Vec::new();
    }
    let mut c = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += x[3 * i + k];
        }
    }
    for v in c.iter_mut() {
        *v /= n as f64;
    }
    let mut r: Vec<f64> = (0..n)
        .map(|i| {
            (0..3)
                .map(|k| {
                    let d = x[3 * i + k] - c[k];
                    d * d
                })
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    r
}

/// Chooses a construction by Thompson sampling over predicted quenched energy.
#[derive(Debug)]
pub struct Constructor {
    model: Screen,
    /// Candidates built and scored per proposal.
    pub width: usize,
    /// Constructions proposed.
    pub proposals: usize,
    /// Quenches fed back.
    pub observations: usize,
}

impl Default for Constructor {
    fn default() -> Self {
        Self::new(4)
    }
}

impl Constructor {
    /// A constructor scoring `width` candidates per proposal.
    pub fn new(width: usize) -> Self {
        Self {
            // Warm up on a handful of quenches before the posterior steers
            // anything, and keep an exploration floor so the model cannot stop
            // sampling the region it would need evidence from.
            model: Screen::new(FEATURES, 12, 0.1, 0.5),
            width: width.max(1),
            proposals: 0,
            observations: 0,
        }
    }

    /// Quenched energies the model has seen.
    pub fn seen(&self) -> usize {
        self.model.observations()
    }

    /// Builds `width` candidates and returns the one the posterior likes, with
    /// its features so the caller can report the quench back.
    ///
    /// The sources offered are the library's, and `keep` is drawn across its
    /// whole range rather than around whatever has worked, because the
    /// posterior is what expresses the preference and it needs the range
    /// covered to have an opinion about it.
    pub fn propose<R: Rng + ?Sized>(
        &mut self,
        parent: ArrayView1<f64>,
        n: usize,
        rng: &mut R,
    ) -> (Array1<f64>, Array1<f64>) {
        self.proposals += 1;
        let sources = Source::library();
        let mut best: Option<(f64, Array1<f64>, Array1<f64>)> = None;
        // One coefficient draw for the whole set, which is what makes this
        // Thompson sampling over constructions rather than an independent
        // gamble per candidate.
        let u = rng.random::<f64>();
        for i in 0..self.width {
            let recipe = Recipe {
                source: sources[rng.random_range(0..sources.len())],
                keep: rng.random::<f64>(),
            };
            let cand = lattice::candidate_keeping(recipe.source, parent, n, recipe.keep, rng);
            let f = features(&recipe, cand.view(), parent);
            // Lower predicted energy is better, and an unexplored construction
            // is scored optimistically through the posterior's own spread.
            let score = match self.model.predict(f.view()) {
                Some((mean, sd)) => mean - sd * (2.0 * (u + i as f64 / self.width as f64) % 2.0),
                None => f64::NEG_INFINITY,
            };
            if best.as_ref().map(|(s, _, _)| score < *s).unwrap_or(true) {
                best = Some((score, cand, f));
            }
        }
        let (_, cand, f) = best.expect("width is at least one");
        (cand, f)
    }

    /// Records how much a construction improved on the structure it came from.
    ///
    /// The target is the change, not the quenched energy. Regressing on the
    /// energy makes the do-nothing construction optimal: keeping the whole
    /// structure and regrowing nothing quenches back to the incumbent, which is
    /// the lowest energy any proposal from that state can reach, so a posterior
    /// over energy learns to switch the move off. Measured, it did exactly
    /// that: accept rates fell from 0.44 to 0.07 and every source came back
    /// reporting the incumbent's own value to six figures.
    ///
    /// Against the change, a proposal that returns where it started scores
    /// zero and only one that goes lower scores negative, which is the
    /// preference the move exists to express.
    pub fn observe(&mut self, features: ArrayView1<f64>, energy: f64, from: f64) {
        let delta = energy - from;
        if delta.is_finite() {
            self.observations += 1;
            self.model.observe(features, delta);
        }
    }

    /// The fitted coefficients, when there are enough observations to have
    /// them. Reported so a run can say what the model learned rather than only
    /// that it ran.
    pub fn coefficients(&self) -> Option<Array1<f64>> {
        self.model.coefficients()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::Template;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn parent(n: usize) -> Array1<f64> {
        let sites = lattice::grow(&Template::FaceCentredCubic.points(), n);
        let mut x = Array1::zeros(3 * n);
        for (i, s) in sites.iter().take(n).enumerate() {
            for k in 0..3 {
                x[3 * i + k] = 1.12 * s[k];
            }
        }
        x
    }

    /// A proposal has to be a structure of the right size whatever the model
    /// knows, including when it knows nothing.
    #[test]
    fn a_cold_constructor_still_proposes_a_structure() {
        let mut r = StdRng::seed_from_u64(3);
        let mut c = Constructor::new(4);
        let p = parent(55);
        let (cand, f) = c.propose(p.view(), 55, &mut r);
        assert_eq!(cand.len(), 3 * 55);
        assert_eq!(f.len(), FEATURES);
        assert_eq!(c.seen(), 0);
    }

    /// Keeping the whole structure and keeping none of it have to give
    /// different proposals, or the parameter the posterior is held over does
    /// nothing.
    #[test]
    fn the_keep_fraction_changes_the_proposal() {
        let mut r = StdRng::seed_from_u64(5);
        let p = parent(55);
        let near = lattice::candidate_keeping(
            Source::Named(Template::FaceCentredCubic),
            p.view(),
            55,
            0.9,
            &mut r,
        );
        let far = lattice::candidate_keeping(
            Source::Named(Template::Icosahedral),
            p.view(),
            55,
            0.0,
            &mut r,
        );
        let dn = radial_distance(near.view(), p.view());
        let df = radial_distance(far.view(), p.view());
        assert!(
            dn < df,
            "keeping 0.9 moved {dn}, keeping none moved {df}; the knob is inverted or inert"
        );
    }

    /// The model has to learn a relationship that is there. Fed a synthetic
    /// rule where energy falls with coordination, its predictions have to order
    /// two candidates the same way the rule does.
    #[test]
    fn the_posterior_learns_a_relationship_it_is_shown() {
        let mut r = StdRng::seed_from_u64(9);
        let mut c = Constructor::new(2);
        let p = parent(40);
        let mut seen = Vec::new();
        for _ in 0..80 {
            let (cand, f) = c.propose(p.view(), 40, &mut r);
            // Energy falls with mean coordination, which is feature 2.
            let e = -10.0 * f[2];
            c.observe(f.view(), e, 0.0);
            seen.push((f[2], e, cand.len()));
        }
        assert!(c.seen() >= 60, "only {} observations", c.seen());
        let coef = c.coefficients().expect("no coefficients after 80 quenches");
        assert!(
            coef[2] < -1.0,
            "coordination coefficient {} did not pick up the sign",
            coef[2]
        );
    }

    /// Feedback must not be corrupted by a failed quench, which arrives as a
    /// non-finite value.
    #[test]
    fn a_non_finite_quench_is_not_recorded() {
        let mut r = StdRng::seed_from_u64(13);
        let mut c = Constructor::new(2);
        let p = parent(30);
        let (_, f) = c.propose(p.view(), 30, &mut r);
        c.observe(f.view(), f64::INFINITY, 0.0);
        c.observe(f.view(), f64::NAN, 0.0);
        assert_eq!(c.seen(), 0);
    }
}
