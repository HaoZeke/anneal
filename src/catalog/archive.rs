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

/// Per-cell curiosity, in the MAP-Elites sense.
///
/// Selecting always from the emptiest cell is the obvious archive
/// policy and the wrong one once the descriptor is noisy: an empty
/// cell that nothing can reach is chosen forever. Curiosity scores
/// answer that by paying a cell when a start drawn from it produced
/// something the archive kept, and charging it when it did not, so
/// effort follows what has recently worked.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Curiosity {
    scores: Vec<f64>,
}

impl Curiosity {
    /// Scores for `cells`, each starting neutral.
    pub fn new(cells: usize) -> Self {
        Self {
            scores: vec![1.0; cells],
        }
    }

    /// Grow the table so `cells` are scored, new ones neutral.
    ///
    /// Families are discovered as the search runs, so the table cannot
    /// be sized once. Growing never disturbs a score already earned.
    pub fn ensure(&mut self, cells: usize) {
        if cells > self.scores.len() {
            self.scores.resize(cells, 1.0);
        }
    }

    /// Score of one cell, or zero when it is not a cell.
    pub fn score(&self, cell: usize) -> f64 {
        self.scores.get(cell).copied().unwrap_or(0.0)
    }

    /// A start drawn from this cell produced something kept.
    pub fn reward(&mut self, cell: usize) {
        if let Some(score) = self.scores.get_mut(cell) {
            *score += 1.0;
        }
    }

    /// A start drawn from this cell produced nothing.
    ///
    /// Floored above zero so a cell is never removed from
    /// consideration outright: the descriptor may be wrong, not the
    /// cell, and a scheme that can permanently exclude a region of the
    /// space cannot recover from a bad descriptor.
    pub fn penalise(&mut self, cell: usize) {
        if let Some(score) = self.scores.get_mut(cell) {
            *score = (*score * 0.5).max(0.05);
        }
    }

    /// Draw a cell in proportion to curiosity, restricted to `allowed`.
    pub fn select<R: Rng + ?Sized>(&self, allowed: &[usize], rng: &mut R) -> Option<usize> {
        let total: f64 = allowed.iter().map(|cell| self.score(*cell)).sum();
        if allowed.is_empty() || !(total > 0.0) {
            return allowed.first().copied();
        }
        let mut draw = rng.random::<f64>() * total;
        for cell in allowed {
            draw -= self.score(*cell);
            if draw <= 0.0 {
                return Some(*cell);
            }
        }
        allowed.last().copied()
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
    fn curiosity_follows_what_worked() {
        let mut curiosity = Curiosity::new(3);
        assert_eq!(curiosity.score(0), 1.0);
        curiosity.reward(1);
        curiosity.reward(1);
        curiosity.penalise(2);
        assert!(curiosity.score(1) > curiosity.score(0));
        assert!(curiosity.score(2) < curiosity.score(0));
    }

    #[test]
    fn a_penalised_cell_is_never_written_off() {
        let mut curiosity = Curiosity::new(1);
        for _ in 0..200 {
            curiosity.penalise(0);
        }
        // A descriptor can be wrong where a cell is not, so a scheme
        // that permanently excludes a region cannot recover from one.
        assert!(curiosity.score(0) > 0.0);
    }

    #[test]
    fn selection_prefers_the_curious_cell() {
        let mut curiosity = Curiosity::new(2);
        for _ in 0..10 {
            curiosity.reward(0);
        }
        curiosity.penalise(1);
        let mut rng = StdRng::seed_from_u64(13);
        let picked = (0..200)
            .filter(|_| curiosity.select(&[0, 1], &mut rng) == Some(0))
            .count();
        assert!(picked > 150, "curious cell chosen only {picked} of 200");
    }

    #[test]
    fn selection_answers_even_with_nothing_to_go_on() {
        let curiosity = Curiosity::default();
        let mut rng = StdRng::seed_from_u64(17);
        assert_eq!(curiosity.select(&[], &mut rng), None);
        // Cells outside the score table score zero; still an answer.
        assert_eq!(curiosity.select(&[4], &mut rng), Some(4));
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
