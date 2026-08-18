//! Quality-diversity archive primitives.
//!
//! The occupancy catalog keeps one best structure per packing family
//! and draws new starts from it, which is MAP-Elites with a learned
//! descriptor in place of a designed one. Three pieces of that
//! literature are missing here and are supplied below, each as
//! arithmetic over descriptors so nothing in this module needs a
//! search, a socket, or a potential.
//!
//! ## Why a tessellation
//!
//! Families are discovered by leader clustering, so their number is
//! whatever the data produced: a campaign reported twenty-two families
//! across twenty-four replicas, which is nearly one cell per replica
//! and not an archive at all. Worse, a saturation statistic over a
//! support that grows as it samples is answering a moving question.
//! Tessellating the descriptor space instead makes the cell count a
//! parameter and the cells equal by construction, which is what
//! centroidal Voronoi tessellations are for in this setting.
//!
//! The tessellation needs a descriptor of fixed width. A DECAF class
//! histogram grows a column whenever the codebook grows one, so it is
//! the wrong input; the fixed-width cloud mean is the right one. That
//! constraint is real and is the reason this is a separate space from
//! the family key rather than a replacement for it.

use rand::Rng;

/// A centroidal Voronoi tessellation of a descriptor space.
///
/// Cells are named by their centroid and assignment is nearest
/// centroid, so the number of cells is chosen once and does not move.
#[derive(Clone, Debug, PartialEq)]
pub struct Tessellation {
    centroids: Vec<Vec<f64>>,
}

impl Tessellation {
    /// Relax `niches` centroids onto `samples` by Lloyd iteration.
    ///
    /// Samples are descriptors already seen, which is what makes the
    /// cells follow the region the search actually occupies rather
    /// than a box drawn around it. Returns `None` when there is no
    /// space to tessellate or fewer samples than niches asked for.
    pub fn build<R: Rng + ?Sized>(
        niches: usize,
        samples: &[Vec<f64>],
        iterations: usize,
        rng: &mut R,
    ) -> Option<Self> {
        if niches == 0 || samples.len() < niches {
            return None;
        }
        let width = samples[0].len();
        if width == 0 || samples.iter().any(|sample| sample.len() != width) {
            return None;
        }
        // Seed on distinct samples so no centroid starts empty.
        let mut chosen: Vec<usize> = Vec::with_capacity(niches);
        while chosen.len() < niches {
            let index = rng.random_range(0..samples.len());
            if !chosen.contains(&index) {
                chosen.push(index);
            }
        }
        let mut centroids: Vec<Vec<f64>> = chosen
            .into_iter()
            .map(|index| samples[index].clone())
            .collect();
        for _ in 0..iterations {
            let mut sums = vec![vec![0.0; width]; niches];
            let mut counts = vec![0usize; niches];
            for sample in samples {
                let cell = nearest(&centroids, sample)?;
                counts[cell] += 1;
                for (axis, value) in sample.iter().enumerate() {
                    sums[cell][axis] += value;
                }
            }
            for (cell, count) in counts.iter().enumerate() {
                if *count == 0 {
                    // An empty cell keeps its centroid rather than
                    // collapsing onto another, so the count stays put.
                    continue;
                }
                for axis in 0..width {
                    centroids[cell][axis] = sums[cell][axis] / *count as f64;
                }
            }
        }
        Some(Self { centroids })
    }

    /// Cells in the tessellation. Fixed by construction.
    pub fn niches(&self) -> usize {
        self.centroids.len()
    }

    /// Cell a descriptor belongs to, or `None` on a width mismatch.
    pub fn assign(&self, descriptor: &[f64]) -> Option<usize> {
        nearest(&self.centroids, descriptor)
    }

    /// Fraction of cells that any of `descriptors` reaches.
    ///
    /// This is the quantity a saturation test wants and the one a
    /// discovered-cell scheme cannot supply, because there the
    /// denominator moves.
    pub fn coverage<'a, I>(&self, descriptors: I) -> f64
    where
        I: IntoIterator<Item = &'a [f64]>,
    {
        if self.centroids.is_empty() {
            return 0.0;
        }
        let mut seen = vec![false; self.centroids.len()];
        for descriptor in descriptors {
            if let Some(cell) = self.assign(descriptor) {
                seen[cell] = true;
            }
        }
        seen.iter().filter(|reached| **reached).count() as f64 / self.centroids.len() as f64
    }
}

fn nearest(centroids: &[Vec<f64>], point: &[f64]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (index, centroid) in centroids.iter().enumerate() {
        if centroid.len() != point.len() {
            return None;
        }
        let distance: f64 = centroid
            .iter()
            .zip(point)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        if best.is_none_or(|(_, held)| distance < held) {
            best = Some((index, distance));
        }
    }
    best.map(|(index, _)| index)
}

/// Per-cell Beta-Bernoulli bandit over the archive.
///
/// Choosing which cell to draw a start from is a bandit problem and
/// deserves to be treated as one. The reward is binary, the catalog
/// kept what came back or it did not, so the conjugate model is Beta
/// over a Bernoulli rate and the selection rule is Thompson sampling:
/// draw a rate from each cell's posterior and take the highest.
///
/// That is better than the reward-and-decay heuristic it replaces on
/// three counts. It has no constants to pick, where the heuristic had
/// a decay and a floor chosen by hand. Its exploration is automatic:
/// a cell tried twice has a wide posterior and still wins draws, while
/// a cell tried two hundred times does not, so effort moves off a cell
/// only once there is evidence to move it. And it cannot write a cell
/// off, because a Beta posterior never reaches zero, which matters
/// because a descriptor can be wrong where a cell is right.
///
/// The allocator over move kernels in [`crate::allocate`] is the same
/// idea over a Gaussian reward; this is the Bernoulli case.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Curiosity {
    /// Successes plus one: starts from this cell the catalog kept.
    alpha: Vec<f64>,
    /// Failures plus one: starts it did not keep.
    beta: Vec<f64>,
}

impl Curiosity {
    /// A bandit over `cells`, each with a uniform prior.
    pub fn new(cells: usize) -> Self {
        Self {
            alpha: vec![1.0; cells],
            beta: vec![1.0; cells],
        }
    }

    /// Grow the table so `cells` are armed, new ones uniform.
    ///
    /// Cells are discovered as the search runs, so the table cannot be
    /// sized once. Growing never disturbs a posterior already earned.
    pub fn ensure(&mut self, cells: usize) {
        if cells > self.alpha.len() {
            self.alpha.resize(cells, 1.0);
            self.beta.resize(cells, 1.0);
        }
    }

    /// Posterior mean success rate of a cell, or the uniform prior for
    /// one the table has not armed.
    pub fn score(&self, cell: usize) -> f64 {
        let alpha = self.alpha.get(cell).copied().unwrap_or(1.0);
        let beta = self.beta.get(cell).copied().unwrap_or(1.0);
        alpha / (alpha + beta)
    }

    /// Times this cell has been drawn from.
    pub fn draws(&self, cell: usize) -> f64 {
        let alpha = self.alpha.get(cell).copied().unwrap_or(1.0);
        let beta = self.beta.get(cell).copied().unwrap_or(1.0);
        alpha + beta - 2.0
    }

    /// A start drawn from this cell produced something the catalog kept.
    pub fn reward(&mut self, cell: usize) {
        if let Some(alpha) = self.alpha.get_mut(cell) {
            *alpha += 1.0;
        }
    }

    /// A start drawn from this cell produced nothing.
    pub fn penalise(&mut self, cell: usize) {
        if let Some(beta) = self.beta.get_mut(cell) {
            *beta += 1.0;
        }
    }

    /// Thompson draw: sample a rate from each allowed cell's posterior
    /// and take the highest.
    pub fn select<R: Rng + ?Sized>(&self, allowed: &[usize], rng: &mut R) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for cell in allowed {
            let alpha = self.alpha.get(*cell).copied().unwrap_or(1.0);
            let beta = self.beta.get(*cell).copied().unwrap_or(1.0);
            let sampled = sample_beta(alpha, beta, rng);
            if best.is_none_or(|(_, held)| sampled > held) {
                best = Some((*cell, sampled));
            }
        }
        best.map(|(cell, _)| cell)
    }
}

/// One Beta draw, as the ratio of two Gammas.
fn sample_beta<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> f64 {
    let x = sample_gamma(alpha, rng);
    let y = sample_gamma(beta, rng);
    if x + y <= 0.0 { 0.5 } else { x / (x + y) }
}

/// One Gamma draw by Marsaglia and Tsang, with the shape boost that
/// carries shapes below one.
fn sample_gamma<R: Rng + ?Sized>(shape: f64, rng: &mut R) -> f64 {
    if !shape.is_finite() || shape <= 0.0 {
        return 0.0;
    }
    if shape < 1.0 {
        let boosted = sample_gamma(shape + 1.0, rng);
        let u: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        return boosted * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Box-Muller for the standard normal the method needs.
        let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2: f64 = rng.random::<f64>();
        let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let v = 1.0 + c * normal;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        if u.ln() < 0.5 * normal * normal + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// Novelty of a descriptor: mean distance to its `k` nearest
/// neighbours among those already seen.
///
/// On a deceptive landscape the objective's gradient points into the
/// trap, and this is the quantity a search follows instead. An empty
/// neighbourhood is maximally novel and answers infinity, which is the
/// honest value: nothing has been seen to compare against.
pub fn novelty(descriptor: &[f64], seen: &[Vec<f64>], k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let mut distances: Vec<f64> = seen
        .iter()
        .filter(|other| other.len() == descriptor.len())
        .map(|other| {
            other
                .iter()
                .zip(descriptor)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    if distances.is_empty() {
        return f64::INFINITY;
    }
    distances.sort_by(f64::total_cmp);
    let take = k.min(distances.len());
    distances.iter().take(take).sum::<f64>() / take as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn two_clouds() -> Vec<Vec<f64>> {
        let mut samples = Vec::new();
        for i in 0..20 {
            let jitter = i as f64 * 0.01;
            samples.push(vec![0.0 + jitter, 0.0]);
            samples.push(vec![5.0 + jitter, 5.0]);
        }
        samples
    }

    #[test]
    fn the_cell_count_is_chosen_not_discovered() {
        let mut rng = StdRng::seed_from_u64(3);
        let tessellation = Tessellation::build(8, &two_clouds(), 10, &mut rng).expect("builds");
        // The point of the exercise: twenty-two families across
        // twenty-four replicas is what discovery gives; this gives
        // whatever was asked for.
        assert_eq!(tessellation.niches(), 8);
    }

    #[test]
    fn a_tessellation_separates_two_clouds() {
        let mut rng = StdRng::seed_from_u64(5);
        let tessellation = Tessellation::build(2, &two_clouds(), 20, &mut rng).expect("builds");
        let low = tessellation.assign(&[0.0, 0.0]).expect("assigns");
        let high = tessellation.assign(&[5.0, 5.0]).expect("assigns");
        assert_ne!(low, high, "both clouds landed in one cell");
    }

    #[test]
    fn a_tessellation_refuses_what_it_cannot_tessellate() {
        let mut rng = StdRng::seed_from_u64(7);
        assert!(Tessellation::build(0, &two_clouds(), 5, &mut rng).is_none());
        // More niches than samples.
        assert!(Tessellation::build(500, &two_clouds(), 5, &mut rng).is_none());
        // Ragged descriptors, which is what a growing class histogram
        // looks like and the reason it is not the input here.
        let ragged = vec![vec![0.0, 1.0], vec![0.0]];
        assert!(Tessellation::build(1, &ragged, 5, &mut rng).is_none());
    }

    #[test]
    fn coverage_has_a_denominator_that_does_not_move() {
        let mut rng = StdRng::seed_from_u64(11);
        let tessellation = Tessellation::build(4, &two_clouds(), 20, &mut rng).expect("builds");
        let nothing: Vec<&[f64]> = Vec::new();
        assert_eq!(tessellation.coverage(nothing), 0.0);
        let one = [0.0, 0.0];
        let covered = tessellation.coverage(vec![&one[..]]);
        assert!(covered > 0.0 && covered <= 0.5, "coverage {covered}");
    }

    #[test]
    fn the_posterior_follows_what_worked() {
        let mut bandit = Curiosity::new(3);
        // A uniform prior is an even rate and no draws behind it.
        assert!((bandit.score(0) - 0.5).abs() < 1e-12);
        assert_eq!(bandit.draws(0), 0.0);
        bandit.reward(1);
        bandit.reward(1);
        bandit.penalise(2);
        assert!(bandit.score(1) > bandit.score(0));
        assert!(bandit.score(2) < bandit.score(0));
        assert_eq!(bandit.draws(1), 2.0);
    }

    #[test]
    fn a_failing_cell_is_never_written_off() {
        let mut bandit = Curiosity::new(1);
        for _ in 0..200 {
            bandit.penalise(0);
        }
        // A Beta posterior cannot reach zero, which is what stops a
        // descriptor that is wrong about a cell from excluding the
        // region behind it for good.
        assert!(bandit.score(0) > 0.0);
    }

    #[test]
    fn thompson_sampling_prefers_the_cell_that_pays() {
        let mut bandit = Curiosity::new(2);
        for _ in 0..30 {
            bandit.reward(0);
            bandit.penalise(1);
        }
        let mut rng = StdRng::seed_from_u64(13);
        let picked = (0..300)
            .filter(|_| bandit.select(&[0, 1], &mut rng) == Some(0))
            .count();
        assert!(picked > 270, "paying cell chosen only {picked} of 300");
    }

    #[test]
    fn an_untried_cell_still_wins_draws() {
        // The property a decay heuristic does not have: a cell tried
        // twice has a wide posterior and keeps being explored, so
        // effort leaves it only once there is evidence to leave on.
        let mut bandit = Curiosity::new(2);
        for _ in 0..12 {
            bandit.reward(0);
        }
        bandit.penalise(1);
        let mut rng = StdRng::seed_from_u64(19);
        let explored = (0..400)
            .filter(|_| bandit.select(&[0, 1], &mut rng) == Some(1))
            .count();
        assert!(
            explored > 10,
            "barely-tried cell explored only {explored} of 400"
        );
    }

    #[test]
    fn selection_answers_even_with_nothing_to_go_on() {
        let bandit = Curiosity::default();
        let mut rng = StdRng::seed_from_u64(17);
        assert_eq!(bandit.select(&[], &mut rng), None);
        // Cells outside the table carry the uniform prior; still an
        // answer, and still a fair one.
        assert_eq!(bandit.select(&[4], &mut rng), Some(4));
    }

    #[test]
    fn a_beta_draw_stays_in_the_unit_interval() {
        let mut rng = StdRng::seed_from_u64(23);
        for (alpha, beta) in [(0.5, 0.5), (1.0, 1.0), (30.0, 2.0), (2.0, 30.0)] {
            for _ in 0..200 {
                let drawn = sample_beta(alpha, beta, &mut rng);
                assert!(
                    (0.0..=1.0).contains(&drawn),
                    "Beta({alpha},{beta}) drew {drawn}"
                );
            }
        }
        // A shape the method cannot use answers zero rather than
        // looping or returning a NaN into a comparison.
        assert_eq!(sample_gamma(0.0, &mut rng), 0.0);
        assert_eq!(sample_gamma(f64::NAN, &mut rng), 0.0);
    }

    #[test]
    fn a_lopsided_posterior_concentrates() {
        let mut rng = StdRng::seed_from_u64(29);
        let draws: Vec<f64> = (0..500).map(|_| sample_beta(60.0, 2.0, &mut rng)).collect();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        // Beta(60,2) has mean 60/62; the sampler should find it.
        assert!((mean - 60.0 / 62.0).abs() < 0.02, "mean {mean}");
    }

    #[test]
    fn nothing_seen_is_maximally_novel() {
        assert_eq!(novelty(&[1.0, 1.0], &[], 3), f64::INFINITY);
        assert_eq!(novelty(&[1.0], &[vec![0.0]], 0), 0.0);
    }

    #[test]
    fn novelty_grows_with_distance_from_the_seen() {
        let seen = vec![vec![0.0, 0.0], vec![0.1, 0.0]];
        let near = novelty(&[0.05, 0.0], &seen, 2);
        let far = novelty(&[9.0, 0.0], &seen, 2);
        assert!(far > near, "far {far} near {near}");
    }
}
